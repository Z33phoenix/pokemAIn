import numpy as np
from typing import Any, Dict, Optional, Tuple

from src.core.game_interface import MemoryInterface


class RewardSystem:
    """
    Reward shaping heuristics configured entirely via YAML.

    Updated to use MemoryInterface abstraction for game-agnostic memory access.
    This allows the reward system to work with any game (Red, Emerald, etc.)
    without changes.
    """

    def __init__(self, config: Dict[str, float], memory_interface: Optional[MemoryInterface] = None):
        """
        Store config and initialize mutable reward-tracking state.

        Args:
            config: Reward configuration dictionary
            memory_interface: Optional memory interface (can be set later via set_memory_interface)
        """
        self.config = config
        self._memory_interface = memory_interface
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
        self.max_level_reward = 0.0
        self.consecutive_already_out = 0
        self.last_narrative = ""
        self.cursor_history: list[Optional[str]] = []  # Track recent cursor selections for cycling detection
        
        # Battle reward tracking to prevent farming
        self.current_battle_rewards = 0.0
        self.battle_turns = 0
        
        # Menu timeout tracking
        self.menu_open_steps = 0
        
        # Running from trainer battle tracking
        self.consecutive_run_attempts = 0

    def set_memory_interface(self, memory_interface: MemoryInterface):
        """Set the memory interface for this reward system."""
        self._memory_interface = memory_interface

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
        self.max_level_reward = 0.0
        self.consecutive_already_out = 0
        self.last_narrative = ""
        self.cursor_history = []
        self.current_battle_rewards = 0.0
        self.battle_turns = 0
        self.menu_open_steps = 0
        self.consecutive_run_attempts = 0

    def compute_components(
        self,
        info: Dict[str, Any],
        obs,
        action: int,
        goal_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Returns a dict of reward components keyed by specialist type plus a global term.
        The caller is responsible for combining/clipping as needed.

        Args:
            info: Game state information dictionary from environment
            obs: Observation (screen pixels)
            action: Action taken
            goal_ctx: Optional goal context for goal-aware rewards
        """
        cfg = self.config
        battle_active = info.get("battle_active", False)
        
        # Reduce step penalty during battles to not overwhelm damage rewards
        step_penalty = cfg.get("step_penalty", 0.0)
        if battle_active:
            step_penalty *= 0.2  # 80% reduction during battles
        
        rewards = {
            "global_reward": step_penalty,
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

        # Map dimensions (optional - used for edge rewards)
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
                if map_connections.get("south", {}).get("exists") and y == (map_height * 2 - 1): # Block units conversion check needed? usually height is blocks.
                    # Assuming height is in blocks (2x2 tiles), y is in tiles.
                    # Usually map_height * 2 - 1 is the bottom edge in tiles.
                    # However, read_map_height returns blocks. 
                    dest_map = map_connections.get("south", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
                if map_connections.get("west", {}).get("exists") and x == 0:
                    dest_map = map_connections.get("west", {}).get("dest_map")
                    if dest_map is not None and not self._edge_exists(map_id, dest_map):
                        rewards["nav_reward"] += edge_reward
                if map_connections.get("east", {}).get("exists") and x == (map_width * 2 - 1):
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

        # Experience tracking (use memory interface if available)
        current_xp = 0
        if self._memory_interface is not None:
            current_xp = self._memory_interface.get_first_pokemon_experience()
            if current_xp > self.last_xp > 0:
                rewards["global_reward"] += cfg.get("level_up", 0.0)
            self.last_xp = current_xp

        menu_active = self._menu_active(info)
        if menu_active and not self.last_menu_open:
            rewards["nav_reward"] += cfg.get("nav_menu_open_bonus", 0.0)

        battle_active = info.get("battle_active", False)
        max_battle_reward_per_battle = cfg.get("max_battle_reward_per_battle", 15.0)  # Cap total battle rewards
        min_battle_penalty_per_battle = cfg.get("min_battle_penalty_per_battle", -20.0)  # Cap total battle penalties
        
        if battle_active:
            # Track battle turns and apply mild time penalty
            self.battle_turns += 1
            battle_tick_reward = cfg.get("battle_tick", 0.0)
            
            # Only apply tick penalty if we haven't hit the penalty floor
            if self.current_battle_rewards > min_battle_penalty_per_battle:
                rewards["battle_reward"] += battle_tick_reward
                self.current_battle_rewards += battle_tick_reward
        else:
            if self.was_in_battle:
                # Battle just ended - check if we won or lost
                hp_percent = info.get("hp_percent", 1.0)
                battle_loss_threshold = cfg.get("battle_loss_threshold", 0.0)
                battle_win_reward = cfg.get("battle_win", 0.0)
                battle_loss_reward = cfg.get("battle_loss", 0.0)
                
                if hp_percent < battle_loss_threshold:
                    rewards["battle_reward"] += battle_loss_reward
                    print(f"💀 BATTLE LOST! HP: {hp_percent:.1%} → Reward: {battle_loss_reward}")
                else:
                    rewards["battle_reward"] += battle_win_reward
                    print(f"🏆 BATTLE WON! HP: {hp_percent:.1%} → Reward: {battle_win_reward}")
                
                # Reset battle tracking
                print(f"📊 BATTLE END: Total battle rewards this fight: {self.current_battle_rewards:.1f}")
                self.current_battle_rewards = 0.0
                self.battle_turns = 0
            elif not self.was_in_battle:
                # Not in battle, reset counters
                self.battle_turns = 0
                
            self.last_enemy_hp_fraction = None
        self.was_in_battle = battle_active

        # Get HP percentages from info (provided by gym)
        hp_percent = info.get("hp_percent", 1.0)
        enemy_hp_percent = info.get("enemy_hp_percent")

        if battle_active:
            if enemy_hp_percent is not None:
                if self.last_enemy_hp_fraction is None:
                    self.last_enemy_hp_fraction = enemy_hp_percent
                    
                enemy_delta = (self.last_enemy_hp_fraction or 0.0) - enemy_hp_percent
                
                if enemy_delta > 0:
                    damage_reward = cfg.get("battle_damage", 0.0) * enemy_delta
                    original_damage_reward = damage_reward
                    
                    # Cap battle damage rewards to prevent farming
                    if self.current_battle_rewards + damage_reward > max_battle_reward_per_battle:
                        damage_reward = max(0, max_battle_reward_per_battle - self.current_battle_rewards)
                        was_capped = True
                    else:
                        was_capped = False
                    
                    # Only log when actual damage occurs
                    print(f"💥 DAMAGE DEALT! Enemy HP: {enemy_hp_percent:.1%} → Δ{enemy_delta:.1%} → Reward: {damage_reward:.3f}" + 
                          (f" (capped from {original_damage_reward:.3f})" if was_capped else ""))
                    
                    rewards["battle_reward"] += damage_reward
                    self.current_battle_rewards += damage_reward
                    
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

        rewards["global_reward"] += self._compute_level_reward_bonus()

        # Track menu timeout and apply escalating penalties (EXCLUDE battle menus)
        menu_active = self._menu_active(info)
        battle_active = info.get("battle_active", False)
        
        # Only apply menu timeout penalties to NON-battle menus
        if menu_active and not battle_active:
            self.menu_open_steps += 1
            # Apply escalating penalty for staying in menus too long
            if self.menu_open_steps > 10:  # After 10 steps in menu
                timeout_penalty = cfg.get("menu", {}).get("menu_timeout_penalty", -5.0)
                # Exponentially worse the longer you stay
                multiplier = min((self.menu_open_steps - 10) * 0.5, 5.0)
                rewards["menu_reward"] += timeout_penalty * multiplier
        else:
            # Reset counter when not in non-battle menus
            self.menu_open_steps = 0

        rewards["menu_reward"] += self.compute_menu_reward(info, goal_ctx)
        rewards["goal_bonus"] = self.compute_goal_bonus(
            info, goal_ctx, prev_hp_fraction=prev_hp_fraction, was_in_battle_prev=was_in_battle_prev
        )

        # SEVERE penalty for repeatedly trying to switch to active Pokemon
        current_narrative = info.get("text_narrative", "")
        if current_narrative and "already out" in current_narrative.lower():
            # This message appears when trying to switch to the active Pokemon
            if current_narrative == self.last_narrative:
                # Same message as before - increment counter
                self.consecutive_already_out += 1
            else:
                # New instance of the message
                self.consecutive_already_out = 1

            # Apply SEVERE escalating penalty based on repetitions
            base_penalty = cfg.get("switch_active_pokemon_penalty", -10.0)
            # Scale penalty with consecutive attempts (gets exponentially worse)
            penalty_multiplier = min(self.consecutive_already_out ** 1.5, 10)  # Exponential scaling, cap at 10x
            severe_penalty = base_penalty * penalty_multiplier
            
            # Apply penalty, but reduce during battles to allow learning
            battle_active = info.get("battle_active", False)
            penalty_scale = 0.2 if battle_active else 1.0  # 80% reduction during battles
            
            scaled_penalty = severe_penalty * penalty_scale
            rewards["menu_reward"] += scaled_penalty
            rewards["battle_reward"] += scaled_penalty * 0.5
            rewards["global_reward"] += scaled_penalty * 0.3
        else:
            # Reset counter if we're not seeing the "already out" message
            if self.last_narrative and "already out" in self.last_narrative.lower():
                self.consecutive_already_out = 0

        # SEVERE penalty for repeatedly trying to run from trainer battles
        if current_narrative and any(phrase in current_narrative.lower() for phrase in 
                                   ["can't escape", "no! there's no running", "couldn't get away", "can't run away"]):
            # This message appears when trying to run from trainer battles
            if current_narrative == self.last_narrative:
                # Same message as before - increment counter
                self.consecutive_run_attempts += 1
            else:
                # New instance of the message
                self.consecutive_run_attempts = 1

            # Apply SEVERE escalating penalty based on repetitions
            base_penalty = cfg.get("run_from_trainer_penalty", -12.0)
            # Scale penalty with consecutive attempts (gets exponentially worse)
            penalty_multiplier = min(self.consecutive_run_attempts ** 1.5, 8)  # Exponential scaling, cap at 8x
            severe_penalty = base_penalty * penalty_multiplier
            
            # Apply penalty, but reduce during battles to allow learning  
            penalty_scale = 0.2 if battle_active else 1.0  # 80% reduction during battles
            
            scaled_penalty = severe_penalty * penalty_scale
            rewards["battle_reward"] += scaled_penalty
            rewards["menu_reward"] += scaled_penalty * 0.5
            rewards["global_reward"] += scaled_penalty * 0.3
            
            print(f"🚫 CAN'T RUN FROM TRAINER! Attempt #{self.consecutive_run_attempts} → Penalty: {severe_penalty:.1f}")
        else:
            # Reset counter if we're not seeing the "can't run" message
            if self.last_narrative and any(phrase in self.last_narrative.lower() for phrase in 
                                         ["can't escape", "no! there's no running", "couldn't get away", "can't run away"]):
                self.consecutive_run_attempts = 0

        self.last_narrative = current_narrative

        # Penalty for menu cycling (oscillating between same options)
        current_selection = info.get("text_selection", "")
        if current_selection:
            # Add to history (keep last 6 selections)
            self.cursor_history.append(current_selection)
            if len(self.cursor_history) > 6:
                self.cursor_history.pop(0)

            # Detect oscillation: if last 4 selections alternate between 2 values
            if len(self.cursor_history) >= 4:
                recent = self.cursor_history[-4:]
                unique_values = set(recent)
                if len(unique_values) == 2:
                    # Check if it's alternating (A-B-A-B pattern)
                    is_alternating = (recent[0] == recent[2] and
                                     recent[1] == recent[3] and
                                     recent[0] != recent[1])
                    if is_alternating:
                        menu_cycling_penalty = cfg.get("menu_cycling_penalty", -1.0)
                        # Reduce menu cycling penalty during battles
                        if battle_active:
                            menu_cycling_penalty *= 0.3  # 70% reduction during battles
                        rewards["menu_reward"] += menu_cycling_penalty
        else:
            # Clear history when not in menu
            self.cursor_history = []

        clip_value = cfg.get("reward_clip", np.inf)
        for key, value in rewards.items():
            rewards[key] = float(np.clip(value, -clip_value, clip_value))
        
        # DEBUG: Log when total reward is extremely negative
        total_reward = sum(rewards.values())
        if total_reward < -50:
            print(f"🚨 HUGE NEGATIVE REWARD: {total_reward:.1f}")
            for component, value in rewards.items():
                if abs(value) > 0.1:
                    print(f"   {component}: {value:.1f}")
        
        return rewards

    def compute_reward(self, info: Dict[str, Any], obs, action: int) -> float:
        """
        Backwards-compatible scalar reward. Prefer compute_components for new code.

        Args:
            info: Game state information dictionary
            obs: Observation (screen pixels)
            action: Action taken

        Returns:
            Total reward as a float
        """
        rewards = self.compute_components(info, obs, action)
        return float(sum(rewards.values()))

    def compute_menu_reward(
        self, info: Dict[str, Any], goal_ctx: Optional[Dict[str, Any]]
    ) -> float:
        """
        Menu-specific shaping. EXCLUDES battle menus from penalties.
        """
        menu_cfg = self.config.get("menu", {})
        menu_active = self._menu_active(info)
        battle_active = info.get("battle_active", False)
        reward = 0.0
        
        # Don't apply menu penalties to battle menus
        if not menu_active:
            # Only apply inactive penalty if not in battle
            if not battle_active:
                reward += menu_cfg.get("inactive_penalty", 0.0)
            if self.last_menu_open:
                reward += menu_cfg.get("close_penalty", 0.0)
            self.last_menu_cursor = None
            self.menu_steps_stagnant = 0
            self.last_menu_target = info.get("menu_target")
            self.last_menu_open = False
            clip_value = menu_cfg.get("reward_clip", self.config.get("reward_clip", np.inf))
            return float(np.clip(reward, -clip_value, clip_value))

        # Only penalize menu opening if NOT in battle
        if not self.last_menu_open and not battle_active:
            reward += menu_cfg.get("opened_menu", 0.0)

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
            # Only penalize cursor movement in non-battle menus
            if self.last_menu_cursor is not None and cursor != self.last_menu_cursor:
                if not battle_active:  # Don't penalize battle cursor movement
                    reward += menu_cfg.get("cursor_move", 0.0)
                self.menu_steps_stagnant = 0
            else:
                self.menu_steps_stagnant += 1
                # Only apply stale penalty to non-battle menus
                if not battle_active and self.menu_steps_stagnant >= menu_cfg.get("stale_threshold", 0):
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
            
            # --- FIX 2: Handle {x, y} dictionary format from LLM ---
            tx = target.get("x")
            ty = target.get("y")
            
            # Fallback to list format if present
            if tx is None and "coordinate" in target:
                coord = target["coordinate"]
                if len(coord) >= 2:
                    tx, ty = coord[0], coord[1]

            if tx is not None and ty is not None:
                # Optional: Handle map specific coords if provided
                tmap = target.get("map_id") 
                same_map = tmap is None or info.get("map_id") == tmap
                if same_map and info.get("x") == tx and info.get("y") == ty:
                    bonus += explore_cfg.get("coordinate_match", 0.0)
            # -------------------------------------------------------

        elif goal_type == "train":
            train_cfg = cfg.get("train", {})
            battle_active = info.get("battle_active", False)
            if was_in_battle_prev and not battle_active:
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

    def _compute_level_reward_bonus(self) -> float:
        """Apply the PokerL-style level reward based on total party levels."""
        scale = self.config.get("level_reward_scale", 0.0)
        if scale == 0.0 or self._memory_interface is None:
            return 0.0

        party_levels = self._memory_interface.get_party_levels()
        if not party_levels:
            return 0.0

        level_sum = max(sum(party_levels) - 4, 0)  # subtract starting Squirtle level like PokerL
        explore_thresh = 22
        if level_sum < explore_thresh:
            reward = level_sum
        else:
            reward = (level_sum - explore_thresh) / 4.0 + explore_thresh

        if reward <= self.max_level_reward:
            return 0.0

        delta = reward - self.max_level_reward
        self.max_level_reward = reward
        return scale * delta

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
