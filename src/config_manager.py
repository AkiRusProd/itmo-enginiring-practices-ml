"""Configuration management system with composition, profiles, and environment support."""

import os
from pathlib import Path
from typing import Optional

import yaml

from schemas import AppConfig, ConfigProfile, ModelsConfig, TrainConfig


class ConfigManager:
    """
    Configuration manager with support for:
    - Multiple profiles (dev, test, prod)
    - Configuration merging and composition
    - Environment variable substitution
    - Configuration file hierarchy
    """

    # Default configuration file paths
    BASE_CONFIG = "params.yaml"
    PROFILE_CONFIGS = {
        ConfigProfile.DEV: "config/params.dev.yaml",
        ConfigProfile.TEST: "config/params.test.yaml",
        ConfigProfile.PROD: "config/params.prod.yaml",
    }

    def __init__(self, profile: Optional[ConfigProfile] = None):
        """
        Initialize ConfigManager.

        Args:
            profile: Configuration profile (dev/test/prod).
                    If None, read from CONFIG_PROFILE env variable or use DEV.
        """
        if profile is None:
            profile_str = os.getenv("CONFIG_PROFILE", "dev").lower()
            try:
                profile = ConfigProfile(profile_str)
            except ValueError:
                profile = ConfigProfile.DEV

        self.profile = profile
        self._config: Optional[AppConfig] = None

    @staticmethod
    def _load_yaml(path: str) -> dict:
        """Load YAML file with environment variable substitution."""
        if not Path(path).exists():
            return {}

        with open(path, "r") as f:
            content = f.read()

        # Simple environment variable substitution: ${VAR_NAME}
        for key, value in os.environ.items():
            content = content.replace(f"${{{key}}}", str(value))

        return yaml.safe_load(content) or {}

    @staticmethod
    def _merge_dicts(base: dict, override: dict, deep: bool = True) -> dict:
        """
        Merge override dict into base dict.

        Args:
            base: Base dictionary
            override: Override dictionary (takes precedence)
            deep: Whether to perform deep merge for nested dicts

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if (
                deep
                and key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = ConfigManager._merge_dicts(result[key], value, deep=True)
            else:
                result[key] = value

        return result

    def load_config(self) -> AppConfig:
        """
        Load configuration with the following hierarchy (highest to lowest priority):
        1. Environment variables (CONFIG_PROFILE)
        2. Profile-specific config (params.{profile}.yaml)
        3. Base config (params.yaml)

        Returns:
            Validated AppConfig object
        """
        # Load base configuration
        base_params = self._load_yaml(self.BASE_CONFIG)
        if not base_params:
            raise FileNotFoundError(f"Base configuration not found: {self.BASE_CONFIG}")

        # Load profile-specific configuration
        profile_config_path = self.PROFILE_CONFIGS.get(self.profile)
        profile_params = {}
        if profile_config_path:
            profile_params = self._load_yaml(profile_config_path)

        # Merge configurations: base + profile-specific
        merged_params = self._merge_dicts(base_params, profile_params)

        # Create and validate configuration
        config = AppConfig(
            profile=self.profile,
            train=TrainConfig(**merged_params.get("train", {})),
            models=ModelsConfig(**merged_params.get("models", {})),
            **{k: v for k, v in merged_params.items() if k not in ["train", "models"]},
        )

        self._config = config
        return config

    def get_config(self) -> AppConfig:
        """
        Get loaded configuration. Loads if not already loaded.

        Returns:
            AppConfig object
        """
        if self._config is None:
            self.load_config()
        return self._config

    def compose_configs(self, *configs: AppConfig) -> AppConfig:
        """
        Compose multiple configurations into one.
        Later configurations override earlier ones.

        Args:
            *configs: Variable number of AppConfig objects

        Returns:
            Merged AppConfig with current profile
        """
        if not configs:
            return self.get_config()

        result = configs[0]
        for config in configs[1:]:
            result = result.merge(config)

        # Ensure profile is set
        result.profile = self.profile
        return result

    def override_model_params(self, model_name: str, **params) -> AppConfig:
        """
        Override specific model parameters.

        Args:
            model_name: Name of the model
            **params: Model parameters to override

        Returns:
            Updated AppConfig
        """
        config = self.get_config()
        models_data = config.models.model_dump(exclude_none=True)

        if model_name not in models_data:
            models_data[model_name] = {}

        models_data[model_name] = {**models_data[model_name], **params}
        config.models = ModelsConfig(**models_data)

        return config

    def to_dict(self) -> dict:
        return self.get_config().model_dump(mode="json")

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False)


# Global instance
_config_manager: Optional[ConfigManager] = None


def initialize_config(profile: Optional[ConfigProfile] = None) -> AppConfig:
    """
    Initialize global configuration manager.

    Args:
        profile: Configuration profile

    Returns:
        Loaded AppConfig
    """
    global _config_manager
    _config_manager = ConfigManager(profile)
    return _config_manager.load_config()


def get_config_manager() -> ConfigManager:
    """Get global configuration manager. Initializes if needed."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_config()
    return _config_manager


def get_config() -> AppConfig:
    """Get global configuration."""
    return get_config_manager().get_config()
