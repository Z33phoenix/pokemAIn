import numpy as np
from typing import Any, Dict, Optional, Tuple

from src.env import ram_map


class RewardSystem:
    """Reward shaping heuristics configured entirely via YAML."""

    def __init__(self, config: Dict[str, float]):
        self.config = config
        self.visited_maps: set[int] = set()
        self.seen_coords: set[Tuple[int, int, int]] = set()
        self.last_coord: Tuple[int, int, int] = (0, 0, 0)
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

    def reset(self):
        self.visited_maps.clear()
        self.seen_coords.clear()
        self.last_coord = (0, 0, 0)
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
        }

        map_id = info.get("map_id", 0)
        x, y = info.get("x", 0), info.get("y", 0)
        coord = (map_id, x, y)

        if map_id not in self.visited_maps:
            self.visited_maps.add(map_id)
            rewards["nav_reward"] += cfg.get("new_map", 0.0)

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
                    hp_percent = (hp_cur / hp_max) if hp_max > 0 else 0.0
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
            hp_percent = (hp_cur / hp_max) if hp_max > 0 else 0.0
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
