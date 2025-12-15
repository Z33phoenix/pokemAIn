"""
Heuristic Goal Engine - Logic-driven goal setting framework.

This module provides a rule-based goal setting system that can replace LLM-based 
goal generation with programmatic logic. It uses narrative text, story flags, and 
game state to determine appropriate goals using an adaptable rule system.

Key Features:
- Rule-based X->Y->Z logic (narrative + flags -> goal)
- Story progression awareness (badges, events, locations)
- Adaptable rule engine that can be easily extended
- JSON goal output compatible with existing Goal system

Usage:
    engine = HeuristicGoalEngine(config)
    goal = engine.generate_goal(state_summary, narrative_text, story_flags)
"""

from typing import Any, Dict, List, Optional, Tuple, Set
import re
from dataclasses import dataclass
from enum import Enum
from src.agent.goal_strategy import Goal


class StoryEvent(Enum):
    """Major story events that can trigger specific goals."""
    GAME_START = "game_start"
    PALLET_TOWN = "pallet_town"
    OAK_LAB = "oak_lab"
    ROUTE_1 = "route_1"
    VIRIDIAN_CITY = "viridian_city"
    VIRIDIAN_FOREST = "viridian_forest"
    PEWTER_CITY = "pewter_city"
    BROCK_FIGHT = "brock_fight"
    ROUTE_3 = "route_3"
    MT_MOON = "mt_moon"
    CERULEAN_CITY = "cerulean_city"
    MISTY_FIGHT = "misty_fight"
    UNKNOWN = "unknown"


@dataclass
class StoryFlags:
    """Consolidated story progression flags."""
    badge_count: int
    map_id: int
    player_x: int
    player_y: int
    party_size: int
    is_battle_active: bool
    is_menu_open: bool
    current_hp: int
    max_hp: int
    narrative_text: str
    selection_text: str


@dataclass 
class HeuristicRule:
    """A single heuristic rule for goal generation."""
    name: str
    priority: int
    conditions: Dict[str, Any]  # Conditions that must be met
    goal_template: Dict[str, Any]  # Goal template to instantiate
    description: str


class HeuristicGoalEngine:
    """
    Rule-based goal engine that generates goals based on narrative text 
    and story progression flags.
    """
    
    def __init__(self, config: Dict[str, Any], rules: Optional[Dict[str, List]] = None):
        self.config = config
        
        # Load rules from YAML or use provided rules
        if rules is None:
            from src.agent.heuristic_rule_loader import HeuristicRuleLoader
            loader = HeuristicRuleLoader()
            self.rule_sets = loader.load_all_rules()
            engine_config = loader.get_engine_config()
            self.debug = engine_config.get("debug", config.get("heuristic_debug", False))
        else:
            self.rule_sets = rules
            self.debug = config.get("heuristic_debug", False)
        
        # Extract rule sets for compatibility
        self.progression_rules = self.rule_sets.get("progression_rules", [])
        self.interaction_rules = self.rule_sets.get("interaction_rules", [])
        self.navigation_rules = self.rule_sets.get("navigation_rules", [])
        self.battle_rules = self.rule_sets.get("battle_rules", [])
        self.special_rules = self.rule_sets.get("special_rules", [])
        
        # Story state tracking
        self.last_story_event = StoryEvent.UNKNOWN
        self.visited_maps: Set[int] = set()
        self.completed_objectives: Set[str] = set()
        
    def generate_goal(
        self, 
        state_summary: Dict[str, Any], 
        director: Any = None
    ) -> Optional[Goal]:
        """
        Generate a goal based on current game state using heuristic rules.
        
        Args:
            state_summary: Current game state dictionary
            director: Director instance (for goal history access)
            
        Returns:
            Goal instance or None if no appropriate goal found
        """
        # Extract story flags from state summary
        story_flags = self._extract_story_flags(state_summary)
        
        if self.debug:
            print(f"[HEURISTIC] Analyzing state: badge_count={story_flags.badge_count}, "
                  f"map_id={story_flags.map_id}, narrative='{story_flags.narrative_text[:50]}...'")
        
        # Update internal state tracking
        self._update_story_tracking(story_flags)
        
        # Try rules in priority order: progression -> special -> interaction -> battle -> navigation
        for rule_set_name, rules in [
            ("progression", self.progression_rules),
            ("special", self.special_rules),
            ("interaction", self.interaction_rules), 
            ("battle", self.battle_rules),
            ("navigation", self.navigation_rules)
        ]:
            goal = self._evaluate_rule_set(rules, story_flags, rule_set_name)
            if goal:
                if self.debug:
                    print(f"[HEURISTIC] Generated {rule_set_name} goal: {goal.name}")
                return goal
                
        # Fallback: basic exploration goal
        return self._create_fallback_goal(story_flags)
    
    def _extract_story_flags(self, state_summary: Dict[str, Any]) -> StoryFlags:
        """Extract relevant flags from state summary."""
        # Handle both current_info (from Director) and direct game info
        current_info = state_summary.get("current_info", state_summary)
        text_info = current_info.get("text_info", {})
        
        return StoryFlags(
            badge_count=current_info.get("badges", current_info.get("badge_count", 0)),
            map_id=current_info.get("map_id", 0),
            player_x=current_info.get("x", 0),
            player_y=current_info.get("y", 0),
            party_size=current_info.get("party_size", 0),
            is_battle_active=current_info.get("battle_active", False),
            is_menu_open=current_info.get("menu_open", False),
            current_hp=current_info.get("hp_current", 0),
            max_hp=current_info.get("hp_max", 1),
            narrative_text=text_info.get("narrative", "").lower(),
            selection_text=text_info.get("selection", "").lower()
        )
    
    def _update_story_tracking(self, flags: StoryFlags):
        """Update internal tracking based on current flags."""
        self.visited_maps.add(flags.map_id)
        
        # Detect major story events
        if flags.map_id == 0 and "pallet" in flags.narrative_text:
            self.last_story_event = StoryEvent.PALLET_TOWN
        elif flags.map_id == 1 and "oak" in flags.narrative_text:
            self.last_story_event = StoryEvent.OAK_LAB
        elif flags.map_id == 2:  # Route 1
            self.last_story_event = StoryEvent.ROUTE_1
        elif flags.map_id == 1 and "viridian" in flags.narrative_text:
            self.last_story_event = StoryEvent.VIRIDIAN_CITY
        elif flags.badge_count == 1 and "brock" in flags.narrative_text:
            self.last_story_event = StoryEvent.BROCK_FIGHT
            
    def _evaluate_rule_set(
        self, 
        rules: List[HeuristicRule], 
        flags: StoryFlags,
        rule_set_name: str
    ) -> Optional[Goal]:
        """Evaluate a set of rules and return the first matching goal."""
        for rule in sorted(rules, key=lambda r: r.priority, reverse=True):
            if self._check_conditions(rule.conditions, flags):
                if self.debug:
                    print(f"[HEURISTIC] Rule '{rule.name}' matched in {rule_set_name}")
                return self._instantiate_goal(rule.goal_template, flags, rule.name)
        return None
    
    def _check_conditions(self, conditions: Dict[str, Any], flags: StoryFlags) -> bool:
        """Check if all conditions in a rule are met."""
        for key, expected in conditions.items():
            # Handle special condition keys
            if key == "hp_percentage_max":
                hp_percentage = (flags.current_hp / max(flags.max_hp, 1)) * 100
                if hp_percentage > expected:
                    return False
                continue
            elif key == "hp_percentage_min":
                hp_percentage = (flags.current_hp / max(flags.max_hp, 1)) * 100
                if hp_percentage < expected:
                    return False
                continue
                
            actual = getattr(flags, key, None)
            if actual is None:
                return False
            
            # Handle different condition types
            if isinstance(expected, dict):
                if "min" in expected and actual < expected["min"]:
                    return False
                if "max" in expected and actual > expected["max"]:
                    return False
                if "equals" in expected and actual != expected["equals"]:
                    return False
                if "contains" in expected and expected["contains"] not in str(actual).lower():
                    return False
                if "not_contains" in expected and expected["not_contains"] in str(actual).lower():
                    return False
                if "contains_any" in expected:
                    if not any(term in str(actual).lower() for term in expected["contains_any"]):
                        return False
                if "not_contains_any" in expected:
                    if any(term in str(actual).lower() for term in expected["not_contains_any"]):
                        return False
                if "not_empty" in expected and expected["not_empty"] and not str(actual).strip():
                    return False
            elif isinstance(expected, (list, tuple)):
                if actual not in expected:
                    return False
            else:
                if actual != expected:
                    return False
        return True
    
    def _instantiate_goal(
        self, 
        template: Dict[str, Any], 
        flags: StoryFlags, 
        rule_name: str
    ) -> Goal:
        """Create a Goal instance from a template with flag substitutions."""
        # Replace template variables with actual values
        goal_data = template.copy()
        
        # Substitute common variables
        substitutions = {
            "{current_map}": str(flags.map_id),
            "{player_x}": str(flags.player_x),
            "{player_y}": str(flags.player_y),
            "{badge_count}": str(flags.badge_count),
            "{rule_name}": rule_name
        }
        
        for key, value in goal_data.items():
            if isinstance(value, str):
                for pattern, replacement in substitutions.items():
                    value = value.replace(pattern, replacement)
                goal_data[key] = value
        
        # Create target with position if specified
        target = goal_data.get("target", {})
        if flags.player_x and flags.player_y and "explore" in goal_data.get("name", ""):
            target.update({"x": flags.player_x, "y": flags.player_y, "map_id": flags.map_id})
        
        # Compute goal vector for navigation goals
        goal_vector = None
        if "x" in target and "y" in target:
            dx = target["x"] - flags.player_x
            dy = target["y"] - flags.player_y
            mag = (dx**2 + dy**2)**0.5
            if mag > 0:
                goal_vector = [dx / mag, dy / mag]
            else:
                goal_vector = [0.0, 0.0]
        
        return Goal(
            name=goal_data.get("name", f"heuristic-{rule_name}"),
            goal_type=goal_data.get("goal_type", "NAVIGATE"),
            priority=goal_data.get("priority", 1),
            target=target,
            metadata={
                "strategy": "heuristic",
                "rule": rule_name,
                "story_event": self.last_story_event.value,
                **goal_data.get("metadata", {})
            },
            max_steps=goal_data.get("max_steps", 200),
            goal_vector=goal_vector
        )
    
    def _create_fallback_goal(self, flags: StoryFlags) -> Goal:
        """Create a basic exploration goal when no rules match."""
        return Goal(
            name=f"explore-map-{flags.map_id}",
            goal_type="NAVIGATE",
            priority=0,
            target={"map_id": flags.map_id, "description": "Explore current area"},
            metadata={
                "strategy": "heuristic",
                "rule": "fallback",
                "reason": "No specific rules matched, defaulting to exploration"
            },
            max_steps=300,
            goal_vector=None
        )
    
    # ========================================================================
    # RULE DEFINITIONS - These define the X->Y->Z heuristic logic
    # ========================================================================
    
    def _load_progression_rules(self) -> List[HeuristicRule]:
        """Load rules for story progression goals."""
        return [
            HeuristicRule(
                name="start_pokemon_journey",
                priority=10,
                conditions={
                    "badge_count": 0,
                    "party_size": {"min": 1},
                    "map_id": [0, 1],  # Pallet Town or Route 1
                    "narrative_text": {"contains": "pokemon"}
                },
                goal_template={
                    "name": "begin-pokemon-adventure",
                    "goal_type": "NAVIGATE", 
                    "priority": 3,
                    "target": {"description": "Start Pokemon journey", "map_id": 1},
                    "max_steps": 500,
                    "metadata": {"objective": "story_progression"}
                },
                description="Start the Pokemon journey when first Pokemon is received"
            ),
            
            HeuristicRule(
                name="get_first_badge",
                priority=9,
                conditions={
                    "badge_count": 0,
                    "map_id": [2, 3, 4],  # Routes leading to Pewter
                    "narrative_text": {"contains": "gym"}
                },
                goal_template={
                    "name": "challenge-pewter-gym",
                    "goal_type": "BATTLE",
                    "priority": 3,
                    "target": {"description": "Challenge Pewter City Gym", "gym": "pewter"},
                    "max_steps": 400,
                    "metadata": {"objective": "first_badge"}
                },
                description="Head to first gym when ready"
            ),
            
            HeuristicRule(
                name="progress_to_next_city", 
                priority=8,
                conditions={
                    "badge_count": {"min": 1, "max": 7},
                    "narrative_text": {"not_contains": "battle"},
                    "is_battle_active": False
                },
                goal_template={
                    "name": "progress-to-next-city-{badge_count}",
                    "goal_type": "NAVIGATE",
                    "priority": 2,
                    "target": {"description": "Progress to next city"},
                    "max_steps": 600,
                    "metadata": {"objective": "story_progression"}
                },
                description="Continue story progression between badges"
            )
        ]
    
    def _load_interaction_rules(self) -> List[HeuristicRule]:
        """Load rules for NPC and object interaction goals."""
        return [
            HeuristicRule(
                name="talk_to_professor_oak",
                priority=8,
                conditions={
                    "map_id": 1,  # Oak's Lab
                    "narrative_text": {"contains": "oak"},
                    "party_size": 0
                },
                goal_template={
                    "name": "talk-to-professor-oak",
                    "goal_type": "INTERACT",
                    "priority": 3,
                    "target": {"npc": "professor_oak", "action": "talk"},
                    "max_steps": 100,
                    "metadata": {"objective": "get_starter_pokemon"}
                },
                description="Interact with Professor Oak to get starter Pokemon"
            ),
            
            HeuristicRule(
                name="visit_pokemart",
                priority=6,
                conditions={
                    "narrative_text": {"contains": "pokemart"},
                    "selection_text": {"contains": "buy"}
                },
                goal_template={
                    "name": "visit-pokemart",
                    "goal_type": "INTERACT", 
                    "priority": 1,
                    "target": {"building": "pokemart", "action": "shop"},
                    "max_steps": 200,
                    "metadata": {"objective": "buy_items"}
                },
                description="Visit Pokemart when shopping options are available"
            ),
            
            HeuristicRule(
                name="heal_at_pokecenter",
                priority=7,
                conditions={
                    "narrative_text": {"contains": "heal"},
                    "current_hp": {"max": 50},  # Low HP percentage threshold
                    "is_battle_active": False
                },
                goal_template={
                    "name": "heal-at-pokecenter", 
                    "goal_type": "INTERACT",
                    "priority": 2,
                    "target": {"building": "pokecenter", "action": "heal"},
                    "max_steps": 150,
                    "metadata": {"objective": "restore_hp"}
                },
                description="Visit Pokemon Center when HP is low"
            )
        ]
    
    def _load_navigation_rules(self) -> List[HeuristicRule]:
        """Load rules for navigation and exploration goals."""
        return [
            HeuristicRule(
                name="explore_new_route",
                priority=5,
                conditions={
                    "map_id": {"min": 2, "max": 50},  # Route maps
                    "narrative_text": {"contains": "route"}
                },
                goal_template={
                    "name": "explore-route-{current_map}",
                    "goal_type": "NAVIGATE",
                    "priority": 1,
                    "target": {"map_id": "{current_map}", "description": "Explore new route"},
                    "max_steps": 400,
                    "metadata": {"objective": "exploration"}
                },
                description="Explore new route areas"
            ),
            
            HeuristicRule(
                name="enter_building",
                priority=4,
                conditions={
                    "narrative_text": {"contains": "door"},
                    "selection_text": {"contains": "yes"}
                },
                goal_template={
                    "name": "enter-building",
                    "goal_type": "INTERACT",
                    "priority": 2,
                    "target": {"action": "enter", "type": "building"},
                    "max_steps": 50,
                    "metadata": {"objective": "enter_building"}
                },
                description="Enter buildings when prompted"
            ),
            
            HeuristicRule(
                name="navigate_to_landmark",
                priority=3,
                conditions={
                    "narrative_text": {"contains": "city"},
                    "is_battle_active": False
                },
                goal_template={
                    "name": "navigate-to-landmark",
                    "goal_type": "NAVIGATE", 
                    "priority": 1,
                    "target": {"description": "Navigate to landmark"},
                    "max_steps": 500,
                    "metadata": {"objective": "navigation"}
                },
                description="Navigate to cities and landmarks"
            )
        ]
    
    def _load_battle_rules(self) -> List[HeuristicRule]:
        """Load rules for battle-related goals."""
        return [
            HeuristicRule(
                name="battle_wild_pokemon",
                priority=6,
                conditions={
                    "is_battle_active": True,
                    "narrative_text": {"contains": "wild"}
                },
                goal_template={
                    "name": "battle-wild-pokemon",
                    "goal_type": "BATTLE",
                    "priority": 2,
                    "target": {"battle_type": "wild", "action": "fight"},
                    "max_steps": 100,
                    "metadata": {"objective": "wild_battle"}
                },
                description="Fight wild Pokemon encounters"
            ),
            
            HeuristicRule(
                name="battle_gym_leader",
                priority=9,
                conditions={
                    "is_battle_active": True,
                    "narrative_text": {"contains": "leader"}
                },
                goal_template={
                    "name": "battle-gym-leader",
                    "goal_type": "BATTLE",
                    "priority": 3,
                    "target": {"battle_type": "trainer", "action": "fight"},
                    "max_steps": 200,
                    "metadata": {"objective": "gym_battle"}
                },
                description="Battle gym leaders"
            ),
            
            HeuristicRule(
                name="battle_trainer",
                priority=7,
                conditions={
                    "is_battle_active": True,
                    "narrative_text": {"contains": "trainer"}
                },
                goal_template={
                    "name": "battle-trainer",
                    "goal_type": "BATTLE",
                    "priority": 2,
                    "target": {"battle_type": "trainer", "action": "fight"},
                    "max_steps": 150,
                    "metadata": {"objective": "trainer_battle"}
                },
                description="Battle other trainers"
            )
        ]