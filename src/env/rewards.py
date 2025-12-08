import numpy as np
from typing import Any, Dict, Optional, Tuple

from src.env import ram_map


class RewardSystem:
    """Reward shaping heuristics configured entirely via YAML."""

    def __init__(self, config: Dict[str, float]):
        """Store config and initialize mutable reward-tracking state."""
        self.config = config
        self.visited_maps: set[int] = set()
        self.seen_coords: set[Tuple[int, int, int]] = set()
        self.visited_warps: set[Tuple[int, int, int]] = set()
        self.warp_seen_positions: dict[int, set[Tuple[int, int]]] = {}
        self.last_coord: Tuple[int, int, int] = (0, 0, 0)
        self.last_map_id: Optional[int] = None
        self.steps_stagnant = 0
        self.max_party_size = 0
        self.last_xp = 0
        self.last_hp_fraction = 1.0
        self.last_enemy_hp_fraction: Optional[float] = None
        self.was_in_battle = False
        self.last_menu_cursor: Optional[Tuple[int, int]] = None
        self.last_menu_target: Optional[int] = None
        self.menu_steps_stagnant = 0
        self.last_menu_open = False
        self.map_graph: dict[int, set[int]] = {}
        self.discovered_edges: set[frozenset[int]] = set()

    def reset(self):
        """Clear all per-episode tracking variables."""
        self.visited_maps.clear()
        self.seen_coords.clear()
        self.visited_warps.clear()
        self.warp_seen_positions.clear()
        self.last_coord = (0, 0, 0)
        self.last_map_id = None
        self.steps_stagnant = 0
        self.max_party_size = 0
        self.last_xp = 0
        self.last_hp_fraction = 1.0
        self.last_enemy_hp_fraction = None
        self.was_in_battle = False
        self.last_menu_cursor = None
        self.last_menu_target = None
        self.menu_steps_stagnant = 0
        self.last_menu_open = False
        self._reset_graph()

    def compute_components(
        self,
        info: Dict[str, Any],
        memory_bus,
        obs,
        action: int,
        goal_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict of reward components keyed by specialist type plus a global term.
        The caller is responsible for combining/clipping as needed.
        """
        cfg = self.config
        rewards = {
            "global_reward": cfg.get("step_penalty", 0.0),
            "nav_reward": 0.0,
            "battle_reward": 0.0,
            "menu_reward": 0.0,
            "goal_bonus": 0.0,
        }

        map_id = info.get("map_id", 0)
        x, y = info.get("x", 0), info.get("y", 0)
        coord = (map_id, x, y)
        prev_coord = self.last_coord
        prev_map_id = self.last_map_id
        prev_hp_fraction = self.last_hp_fraction
        was_in_battle_prev = self.was_in_battle
        map_width = info.get("map_width")
        map_height = info.get("map_height")
        map_connections = info.get("map_connections") or {}
        connection_count = sum(1 for data in map_connections.values() if data.get("exists"))
        # Track warp positions seen per map for shaping toward transitions.
        map_warps = info.get("map_warps") or []
        if map_warps:
            warp_pos = self.warp_seen_positions.setdefault(map_id, set())
            for warp in map_warps:
                wx, wy = warp.get("x"), warp.get("y")
                if wx is not None and wy is not None:
                    warp_pos.add((int(wx), int(wy)))

        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            rewards["nav_reward"] += cfg.get("new_map", 0.0)
            if connection_count:
                rewards["nav_reward"] += cfg.get("nav_connection_discovery", 0.0) * float(connection_count)

        if coord not in self.seen_coords:
            self.seen_coords.add(coord)
            rewards["nav_reward"] += cfg.get("new_tile", 0.0)

        moved = coord != self.last_coord
        if not moved and action < 4:
            rewards["nav_reward"] += cfg.get("wall_bump", 0.0)
            self.steps_stagnant += 1
            if self.steps_stagnant > cfg.get("stale_threshold", 0):
                rewards["nav_reward"] += cfg.get("stale_penalty", 0.0)
        else:
            self.steps_stagnant = 0
            self.last_coord = coord

        if map_width and map_height and connection_count:
            # Reward reaching map edges that have connections defined to encourage transitions.
            edge_reward = cfg.get("nav_connection_edge", 0.0)
            if edge_reward != 0.0:
                if map_connections.get("north", {}).get("exists") and y == 0:
                    dest_map = map_connections.get("north", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
                if map_connections.get("south", {}).get("exists") and y == (map_height - 1):
                    dest_map = map_connections.get("south", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
                if map_connections.get("west", {}).get("exists") and x == 0:
                    dest_map = map_connections.get("west", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
                if map_connections.get("east", {}).get("exists") and x == (map_width - 1):
                    dest_map = map_connections.get("east", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
        # Reward reaching warp tiles to encourage entering/exiting buildings.
        warp_bonus = cfg.get("nav_warp", 0.0)
        warp_proximity = cfg.get("nav_warp_proximity", 0.0)
        if map_warps:
            for warp in map_warps:
                wx, wy = warp.get("x"), warp.get("y")
                dest_map = warp.get("dest_map")
                edge_known = self._edge_exists(map_id, dest_map)
                if wx is None or wy is None:
                    continue
                if x == wx and y == wy:
                    key = (map_id, wx, wy)
                    if not edge_known and dest_map is not None and key not in self.visited_warps:
                        rewards["nav_reward"] += warp_bonus
                        self.visited_warps.add(key)
                else:
                    # Small shaping toward warps that lead to unexplored maps.
                    dist = abs(int(wx) - int(x)) + abs(int(wy) - int(y))
                    if warp_proximity and dist > 0 and not edge_known and dest_map is not None:
                        rewards["nav_reward"] += warp_proximity / float(dist)

        # Track traversed map-to-map edges (warps and boundaries are treated the same).
        if prev_map_id is None:
            self.last_map_id = map_id
        elif map_id != prev_map_id:
            if self._register_edge(prev_map_id, map_id):
                rewards["nav_reward"] += cfg.get("nav_new_connection", 0.0)
            self.last_map_id = map_id

        # Directional shaping: reward alignment with an explicit goal vector if provided.
        if goal_ctx:
            goal_vec = goal_ctx.get("goal_vector")
            if goal_vec and moved:
                dx = x - prev_coord[1]
                dy = y - prev_coord[2]
                step_vec = np.array([dx, dy], dtype=float)
                step_norm = np.linalg.norm(step_vec)
                dir_vec = np.array(goal_vec[:2], dtype=float)
                dir_norm = np.linalg.norm(dir_vec)
                if step_norm > 0 and dir_norm > 0:
                    alignment = float(np.dot(step_vec / step_norm, dir_vec / dir_norm))
                    urgency = float(goal_vec[2]) if len(goal_vec) > 2 else 1.0
                    nav_weight = cfg.get("direction_follow_bonus", 0.0)
                    rewards["nav_reward"] += nav_weight * alignment * max(0.0, urgency)

        party_size = info.get("party_size", 0)
        if party_size > self.max_party_size:
            rewards["global_reward"] += cfg.get("catch_pokemon", 0.0) * (party_size - self.max_party_size)
            self.max_party_size = party_size

        current_xp = ram_map.read_first_mon_exp(memory_bus)
        if current_xp > self.last_xp > 0:
            rewards["global_reward"] += cfg.get("level_up", 0.0)
        self.last_xp = current_xp

        menu_active = self._menu_active(info)
        if menu_active and not self.last_menu_open:
            rewards["nav_reward"] += cfg.get("nav_menu_open_bonus", 0.0)

        battle_active = info.get("battle_active", False)
        if battle_active:
            rewards["battle_reward"] += cfg.get("battle_tick", 0.0)
        else:
            if self.was_in_battle:
                hp_percent = info.get("hp_percent")
                if hp_percent is None:
                    hp_cur = (memory_bus[ram_map.HP_CURRENT] << 8) | memory_bus[ram_map.HP_CURRENT + 1]
                    hp_max = (memory_bus[ram_map.HP_MAX] << 8) | memory_bus[ram_map.HP_MAX + 1]
                    # Treat empty party as full HP to avoid marking a loss when no Pokemon are present.
                    hp_percent = (hp_cur / hp_max) if hp_max > 0 else 1.0
                if hp_percent is not None and hp_percent < cfg.get("battle_loss_threshold", 0.0):
                    rewards["battle_reward"] += cfg.get("battle_loss", 0.0)
                else:
                    rewards["battle_reward"] += cfg.get("battle_win", 0.0)
            self.last_enemy_hp_fraction = None
        self.was_in_battle = battle_active

        hp_percent = info.get("hp_percent")
        if hp_percent is None:
            hp_cur = (memory_bus[ram_map.HP_CURRENT] << 8) | memory_bus[ram_map.HP_CURRENT + 1]
            hp_max = (memory_bus[ram_map.HP_MAX] << 8) | memory_bus[ram_map.HP_MAX + 1]
            # Treat empty party as full HP to avoid low-HP penalties when no Pokemon are present.
            hp_percent = (hp_cur / hp_max) if hp_max > 0 else 1.0
        enemy_hp_percent = info.get("enemy_hp_percent")
        if enemy_hp_percent is None:
            enemy_hp_cur, enemy_hp_max = ram_map.read_enemy_hp(memory_bus)
            enemy_hp_percent = (enemy_hp_cur / enemy_hp_max) if enemy_hp_max > 0 else None

        if battle_active:
            if enemy_hp_percent is not None:
                if self.last_enemy_hp_fraction is None:
                    self.last_enemy_hp_fraction = enemy_hp_percent
                enemy_delta = (self.last_enemy_hp_fraction or 0.0) - enemy_hp_percent
                if enemy_delta > 0:
                    rewards["battle_reward"] += cfg.get("battle_damage", 0.0) * enemy_delta
                self.last_enemy_hp_fraction = enemy_hp_percent
            if hp_percent is not None and self.last_hp_fraction is not None:
                damage_taken = self.last_hp_fraction - hp_percent
                if damage_taken > 0:
                    rewards["battle_reward"] += cfg.get("battle_damage_taken", 0.0) * damage_taken
            if hp_percent is not None and hp_percent >= cfg.get("battle_hp_high_threshold", 1.0):
                rewards["battle_reward"] += cfg.get("battle_hp_high_bonus", 0.0)

        if hp_percent is not None and hp_percent > self.last_hp_fraction and hp_percent >= cfg.get("heal_target", 0.0):
            rewards["global_reward"] += cfg.get("heal_success", 0.0)
        self.last_hp_fraction = hp_percent

        rewards["menu_reward"] = self.compute_menu_reward(info, goal_ctx)
        rewards["goal_bonus"] = self.compute_goal_bonus(
            info, goal_ctx, prev_hp_fraction=prev_hp_fraction, was_in_battle_prev=was_in_battle_prev
        )

        clip_value = cfg.get("reward_clip", np.inf)
        for key, value in rewards.items():
            rewards[key] = float(np.clip(value, -clip_value, clip_value))
        return rewards

    def compute_reward(self, info: Dict[str, Any], memory_bus, obs, action: int) -> float:
        """
        Backwards-compatible scalar reward. Prefer compute_components for new code.
        """
        rewards = self.compute_components(info, memory_bus, obs, action)
        return float(sum(rewards.values()))

    def compute_menu_reward(
        self, info: Dict[str, Any], goal_ctx: Optional[Dict[str, Any]]
    ) -> float:
        """
        Menu-specific shaping:
        - Reward cursor movement to escape stalling.
        - Reward highlighting the requested entry.
        - Penalize stale cursor positions.
        - Reward opening/being inside a menu (bag/PC/party/etc.).
        All weights come from the rewards.menu config block.
        """
        menu_cfg = self.config.get("menu", {})
        menu_active = self._menu_active(info)
        reward = 0.0
        if not menu_active:
            reward += menu_cfg.get("inactive_penalty", 0.0)
            if self.last_menu_open:
                reward += menu_cfg.get("close_penalty", 0.0)
            self.last_menu_cursor = None
            self.menu_steps_stagnant = 0
            self.last_menu_target = info.get("menu_target")
            self.last_menu_open = False
            clip_value = menu_cfg.get("reward_clip", self.config.get("reward_clip", np.inf))
            return float(np.clip(reward, -clip_value, clip_value))

        # Reward entering a menu once so the specialist gets a strong signal for
        # opening, but avoid a constant per-step bonus that encourages stalling.
        if not self.last_menu_open:
            reward += menu_cfg.get("opened_menu", 0.0)
        # No per-step open_bonus; use cursor/target rewards instead.

        cursor = info.get("menu_cursor")
        current_target = info.get("menu_target")
        desired_target = None
        desired_cursor = None
        if goal_ctx:
            target = goal_ctx.get("target", {}) or {}
            desired_target = target.get("menu_target", desired_target)
            desired_cursor = target.get("cursor", desired_cursor)

        if desired_target is not None and current_target is not None:
            if current_target == desired_target:
                reward += menu_cfg.get("correct_target", 0.0)

        if cursor is not None:
            if desired_cursor is not None and tuple(cursor) == tuple(desired_cursor):
                reward += menu_cfg.get("cursor_on_target", 0.0)
            if self.last_menu_cursor is not None and cursor != self.last_menu_cursor:
                reward += menu_cfg.get("cursor_move", 0.0)
                self.menu_steps_stagnant = 0
            else:
                self.menu_steps_stagnant += 1
                if self.menu_steps_stagnant >= menu_cfg.get("stale_threshold", 0):
                    reward += menu_cfg.get("stale_penalty", 0.0)
        else:
            self.menu_steps_stagnant = 0
        self.last_menu_cursor = cursor
        self.last_menu_target = current_target
        self.last_menu_open = True
        scale = menu_cfg.get("scale", 1.0)
        clip_value = menu_cfg.get("reward_clip", self.config.get("reward_clip", np.inf))
        reward *= scale
        return float(np.clip(reward, -clip_value, clip_value))

    def compute_goal_bonus(
        self,
        info: Dict[str, Any],
        goal_ctx: Optional[Dict[str, Any]],
        prev_hp_fraction: Optional[float] = None,
        was_in_battle_prev: bool = False,
    ) -> float:
        """
        Reward following explicit high-level goals. All weights are drawn from
        the rewards.goal_bonus config block and default to zero for safety.
        """
        if not goal_ctx:
            return 0.0
        goal_type = goal_ctx.get("goal_type")
        target = goal_ctx.get("target", {}) or {}
        cfg = self.config.get("goal_bonus", {})
        bonus = 0.0

        if goal_type == "explore":
            explore_cfg = cfg.get("explore", {})
            target_map = target.get("map_id") or target.get("map")
            if target_map is not None and info.get("map_id") == target_map:
                bonus += explore_cfg.get("map_match", 0.0)
            coord_target = target.get("coordinate") or target.get("coord")
            if coord_target and len(coord_target) >= 2:
                tx, ty = coord_target[0], coord_target[1]
                tmap = coord_target[2] if len(coord_target) > 2 else target_map
                same_map = tmap is None or info.get("map_id") == tmap
                if same_map and info.get("x") == tx and info.get("y") == ty:
                    bonus += explore_cfg.get("coordinate_match", 0.0)
        elif goal_type == "train":
            train_cfg = cfg.get("train", {})
            battle_active = info.get("battle_active", False)
            if was_in_battle_prev and not battle_active:
                # Treat exiting battle with nonzero HP as success
                hp_percent = info.get("hp_percent", self.last_hp_fraction or 0.0)
                if hp_percent is None or hp_percent > 0.0:
                    bonus += train_cfg.get("battle_complete", 0.0)
        elif goal_type == "survive":
            survive_cfg = cfg.get("survive", {})
            hp_target = target.get("hp_target") or target.get("hp_threshold")
            hp_percent = info.get("hp_percent")
            if hp_percent is not None and hp_target is not None:
                if prev_hp_fraction is not None and hp_percent > prev_hp_fraction and hp_percent >= hp_target:
                    bonus += survive_cfg.get("hp_recovered", 0.0)
                elif hp_percent >= hp_target:
                    bonus += survive_cfg.get("hp_recovered", 0.0)
        elif goal_type == "menu":
            menu_cfg = cfg.get("menu", {})
            target_menu = target.get("menu_target")
            cursor = info.get("menu_cursor")
            desired_cursor = target.get("cursor")
            current_menu = info.get("menu_target")
            if target_menu is not None and current_menu == target_menu:
                bonus += menu_cfg.get("menu_reached", 0.0)
            if desired_cursor is not None and cursor is not None and tuple(cursor) == tuple(desired_cursor):
                bonus += menu_cfg.get("cursor_match", 0.0)
        return float(bonus)

    def _reset_graph(self):
        """Clear the tracked overworld graph of maps and their traversed edges."""
        self.map_graph.clear()
        self.discovered_edges.clear()

    def _edge_exists(self, map_a: Optional[int], map_b: Optional[int]) -> bool:
        """Check if an undirected edge between two maps has already been discovered."""
        if map_a is None or map_b is None:
            return False
        edge = frozenset((int(map_a), int(map_b)))
        return edge in self.discovered_edges

    def _register_edge(self, map_a: Optional[int], map_b: Optional[int]) -> bool:
        """
        Record an undirected edge between map_a and map_b. Returns True if the edge
        was newly discovered (not previously present in the graph).
        """
        if map_a is None or map_b is None or map_a == map_b:
            return False
        edge = frozenset((int(map_a), int(map_b)))
        if edge in self.discovered_edges:
            return False
        self.discovered_edges.add(edge)
        self.map_graph.setdefault(int(map_a), set()).add(int(map_b))
        self.map_graph.setdefault(int(map_b), set()).add(int(map_a))
        return True

    @staticmethod
    def _menu_active(info: Dict[str, Any]) -> bool:
        """Consistent check for interactive menus across reward components."""
        menu_open = info.get("menu_open")
        has_options = info.get("menu_has_options")
        if menu_open is None and has_options is None:
            return False
        if menu_open is None:
            return bool(has_options)
        if has_options is None:
            return bool(menu_open)
        return bool(menu_open and has_options)
