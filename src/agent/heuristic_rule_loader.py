"""
Heuristic Rule Loader - Parses YAML configuration files for rule-based goal setting.

This module loads heuristic rules from external YAML configuration files,
enabling easy rule modification without code changes. It converts YAML
rule definitions into HeuristicRule objects for the goal engine.

Usage:
    loader = HeuristicRuleLoader("config/heuristic_rules.yaml")
    rules = loader.load_all_rules()
    engine = HeuristicGoalEngine(config, rules=rules)
"""

import yaml
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HeuristicRule:
    """A single heuristic rule for goal generation."""
    name: str
    priority: int
    conditions: Dict[str, Any]  # Conditions that must be met
    goal_template: Dict[str, Any]  # Goal template to instantiate
    description: str
    rule_set: str = "custom"  # Which rule set this belongs to


class HeuristicRuleLoader:
    """Loads heuristic rules from YAML configuration files."""
    
    def __init__(self, config_path: str = "config/heuristic_rules.yaml"):
        self.config_path = config_path
        self.debug = False
        
    def load_all_rules(self) -> Dict[str, List[HeuristicRule]]:
        """
        Load all rule sets from the configuration file.
        
        Returns:
            Dictionary mapping rule set names to lists of HeuristicRule objects
        """
        if not os.path.exists(self.config_path):
            print(f"[RULE_LOADER] Warning: Config file not found: {self.config_path}")
            return self._get_default_rules()
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"[RULE_LOADER] Error loading config: {e}")
            return self._get_default_rules()
        
        rule_sets = {}
        
        # Load each rule set
        rule_set_names = [
            "progression_rules", "interaction_rules", "navigation_rules", 
            "battle_rules", "special_rules"
        ]
        
        for rule_set_name in rule_set_names:
            rules = config.get(rule_set_name, [])
            rule_objects = []
            
            for rule_data in rules:
                try:
                    rule = self._parse_rule(rule_data, rule_set_name)
                    rule_objects.append(rule)
                except Exception as e:
                    print(f"[RULE_LOADER] Error parsing rule {rule_data.get('name', 'unknown')}: {e}")
                    continue
                    
            rule_sets[rule_set_name] = rule_objects
            
        if self.debug:
            total_rules = sum(len(rules) for rules in rule_sets.values())
            print(f"[RULE_LOADER] Loaded {total_rules} rules from {len(rule_sets)} rule sets")
            
        return rule_sets
    
    def _parse_rule(self, rule_data: Dict[str, Any], rule_set: str) -> HeuristicRule:
        """Parse a single rule from YAML data."""
        # Convert YAML conditions to engine format
        conditions = self._parse_conditions(rule_data.get("conditions", {}))
        
        # Convert YAML goal template to engine format
        goal_template = self._parse_goal_template(rule_data.get("goal", {}))
        
        return HeuristicRule(
            name=rule_data.get("name", "unnamed"),
            priority=rule_data.get("priority", 1),
            conditions=conditions,
            goal_template=goal_template,
            description=rule_data.get("description", ""),
            rule_set=rule_set
        )
    
    def _parse_conditions(self, yaml_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Convert YAML condition format to engine condition format."""
        engine_conditions = {}
        
        for key, value in yaml_conditions.items():
            if key == "badge_count":
                engine_conditions["badge_count"] = {"equals": value}
            elif key == "badge_count_range":
                engine_conditions["badge_count"] = {"min": value[0], "max": value[1]}
            elif key == "badge_count_min":
                engine_conditions["badge_count"] = {"min": value}
            elif key == "badge_count_max":
                engine_conditions["badge_count"] = {"max": value}
            elif key == "party_size":
                engine_conditions["party_size"] = {"equals": value}
            elif key == "party_size_min":
                engine_conditions["party_size"] = {"min": value}
            elif key == "party_size_max":
                engine_conditions["party_size"] = {"max": value}
            elif key == "map_id":
                engine_conditions["map_id"] = {"equals": value}
            elif key == "map_id_in":
                engine_conditions["map_id"] = value  # List of acceptable values
            elif key == "map_id_range":
                engine_conditions["map_id"] = {"min": value[0], "max": value[1]}
            elif key == "hp_percentage_max":
                # Convert to actual HP comparison (handled in engine)
                engine_conditions["hp_percentage_max"] = value
            elif key == "hp_percentage_min":
                engine_conditions["hp_percentage_min"] = value
            elif key == "battle_active":
                engine_conditions["is_battle_active"] = value
            elif key == "menu_open":
                engine_conditions["is_menu_open"] = value
            elif key == "narrative_contains":
                if isinstance(value, list):
                    # Any of the terms should be present
                    engine_conditions["narrative_text"] = {"contains_any": value}
                else:
                    engine_conditions["narrative_text"] = {"contains": value}
            elif key == "narrative_not_contains":
                if isinstance(value, list):
                    engine_conditions["narrative_text"] = {"not_contains_any": value}
                else:
                    engine_conditions["narrative_text"] = {"not_contains": value}
            elif key == "selection_contains":
                if isinstance(value, list):
                    engine_conditions["selection_text"] = {"contains_any": value}
                else:
                    engine_conditions["selection_text"] = {"contains": value}
            elif key == "selection_not_empty":
                if value:
                    engine_conditions["selection_text"] = {"not_empty": True}
            else:
                # Pass through unknown conditions
                engine_conditions[key] = value
                
        return engine_conditions
    
    def _parse_goal_template(self, yaml_goal: Dict[str, Any]) -> Dict[str, Any]:
        """Convert YAML goal format to engine goal template format."""
        return {
            "name": yaml_goal.get("name", "yaml-goal"),
            "goal_type": yaml_goal.get("goal_type", "NAVIGATE"),
            "priority": yaml_goal.get("priority", 1),
            "max_steps": yaml_goal.get("max_steps", 200),
            "target": yaml_goal.get("target", {}),
            "metadata": yaml_goal.get("metadata", {})
        }
    
    def _get_default_rules(self) -> Dict[str, List[HeuristicRule]]:
        """Return basic fallback rules if config file can't be loaded."""
        default_rule = HeuristicRule(
            name="default_exploration",
            priority=1,
            conditions={"map_id": {"min": 0}},  # Always matches
            goal_template={
                "name": "explore-area",
                "goal_type": "NAVIGATE",
                "priority": 1,
                "max_steps": 300,
                "target": {"description": "Explore current area"},
                "metadata": {"strategy": "heuristic", "rule": "default"}
            },
            description="Default exploration when no config available"
        )
        
        return {
            "navigation_rules": [default_rule],
            "progression_rules": [],
            "interaction_rules": [],
            "battle_rules": [],
            "special_rules": []
        }
    
    def get_engine_config(self) -> Dict[str, Any]:
        """Load engine configuration from YAML file."""
        if not os.path.exists(self.config_path):
            return {"debug": False}
            
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get("engine_config", {"debug": False})
        except Exception as e:
            print(f"[RULE_LOADER] Error loading engine config: {e}")
            return {"debug": False}