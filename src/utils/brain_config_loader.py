"""
Configuration loader for RL brains with memory presets.

This module helps manage VRAM-constrained training by providing
easy-to-use memory presets and configuration merging.
"""
import os
import yaml
from typing import Dict, Any, Optional


class BrainConfigLoader:
    """
    Load and manage brain configurations with memory presets.

    Example:
        loader = BrainConfigLoader()
        config = loader.get_brain_config("crossq", memory_preset="low")
        agent = create_agent("crossq", config)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the config loader.

        Args:
            config_path: Path to brain_configs.yaml (default: auto-detect)
        """
        if config_path is None:
            # Auto-detect config path
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            config_path = os.path.join(root_dir, "config", "brain_configs.yaml")

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML config file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_brain_config(
        self,
        brain_type: str,
        memory_preset: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get configuration for a specific brain type.

        Args:
            brain_type: Brain type ("crossq", "bbf", "rainbow")
            memory_preset: Memory preset name ("minimal", "low", "medium", "high")
            overrides: Additional config overrides

        Returns:
            Merged configuration dictionary

        Example:
            config = loader.get_brain_config(
                "crossq",
                memory_preset="low",
                overrides={"learning_rate": 0.0005}
            )
        """
        brain_type = brain_type.lower()

        # Get base brain config
        if brain_type not in self.config:
            raise ValueError(f"Unknown brain type: {brain_type}")

        brain_config = self.config[brain_type].copy()

        # Apply memory preset if specified
        if memory_preset:
            if memory_preset not in self.config.get("memory_presets", {}):
                raise ValueError(f"Unknown memory preset: {memory_preset}")

            preset = self.config["memory_presets"][memory_preset]

            # Merge preset into brain config
            brain_config["buffer_capacity"] = preset.get("buffer_capacity", brain_config.get("buffer_capacity"))
            brain_config["batch_size"] = preset.get("batch_size", brain_config.get("batch_size"))
            brain_config["feature_dim"] = preset.get("feature_dim", brain_config.get("feature_dim"))

            # Note: image_resolution would require environment changes
            if "image_resolution" in preset:
                brain_config["image_resolution"] = preset["image_resolution"]

        # Apply manual overrides
        if overrides:
            brain_config.update(overrides)

        return brain_config

    def get_memory_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get a specific memory preset configuration."""
        if preset_name not in self.config.get("memory_presets", {}):
            raise ValueError(f"Unknown memory preset: {preset_name}")
        return self.config["memory_presets"][preset_name].copy()

    def list_brain_types(self) -> list:
        """List available brain types."""
        return [k for k in self.config.keys() if k != "memory_presets"]

    def list_memory_presets(self) -> list:
        """List available memory presets."""
        return list(self.config.get("memory_presets", {}).keys())

    def estimate_vram_usage(self, brain_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Rough estimation of VRAM usage (in MB).

        This is approximate and depends on PyTorch overhead, etc.

        Returns:
            Dictionary with breakdown of VRAM usage
        """
        # Constants (bytes per parameter/float)
        BYTES_PER_FLOAT32 = 4
        MB = 1024 * 1024

        # Network parameters (rough estimate for CrossQ)
        feature_dim = brain_config.get("feature_dim", 512)
        action_dim = brain_config.get("action_dim", 8)

        # Encoder: ~500K parameters (Nature CNN)
        encoder_params = 500_000 if brain_config.get("use_encoder", True) else 0

        # Q-network: fc_input_dim -> 256 -> 256 -> 256 -> action_dim
        q_net_params = (
            feature_dim * 256 + 256 +  # Layer 1
            256 * 256 + 256 +           # Layer 2
            256 * 256 + 256 +           # Layer 3
            256 * action_dim + action_dim  # Output
        )

        # Total network parameters
        total_params = encoder_params + q_net_params
        network_vram_mb = (total_params * BYTES_PER_FLOAT32 * 2) / MB  # x2 for gradients

        # Batch memory (during training)
        batch_size = brain_config.get("batch_size", 32)
        batch_vram_mb = (batch_size * feature_dim * BYTES_PER_FLOAT32) / MB

        # PyTorch overhead (rough estimate)
        overhead_mb = 500

        # Total
        total_vram_mb = network_vram_mb + batch_vram_mb + overhead_mb

        return {
            "network_mb": round(network_vram_mb, 2),
            "batch_mb": round(batch_vram_mb, 2),
            "overhead_mb": overhead_mb,
            "total_mb": round(total_vram_mb, 2),
        }


# Convenience function for quick usage
def load_brain_config(
    brain_type: str,
    memory_preset: Optional[str] = "low",
    **overrides
) -> Dict[str, Any]:
    """
    Quick loader for brain configurations.

    Args:
        brain_type: "crossq", "bbf", or "rainbow"
        memory_preset: "minimal", "low", "medium", or "high"
        **overrides: Additional config overrides

    Returns:
        Brain configuration dictionary

    Example:
        config = load_brain_config("crossq", memory_preset="low", learning_rate=0.0005)
    """
    loader = BrainConfigLoader()
    return loader.get_brain_config(brain_type, memory_preset, overrides)
