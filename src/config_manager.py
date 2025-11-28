"""Система управления конфигурациями с поддержкой композиции, профилей и переменных окружения."""

import os
from pathlib import Path
from typing import Optional

import yaml

from schemas import AppConfig, ConfigProfile, ModelsConfig, TrainConfig


class ConfigManager:
    """
    Менеджер конфигурации с поддержкой:
    - нескольких профилей (dev, test, prod)
    - объединения и композиции конфигураций
    - подстановки переменных окружения
    - иерархии конфигурационных файлов
    """

    # Default configuration file paths
    BASE_CONFIG = "params.yaml"
    PROFILE_CONFIGS = {
        ConfigProfile.DEV: "config/params.dev.yaml",
        ConfigProfile.TEST: "config/params.test.yaml",
        ConfigProfile.PROD: "config/params.prod.yaml",
    }

    def __init__(self, profile: Optional[ConfigProfile] = None):
        """Инициализация ConfigManager.

        Args:
            profile: Профиль конфигурации (dev/test/prod). Если None,
                     профиль читается из переменной окружения `CONFIG_PROFILE`.
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
        """Загружает YAML-файл с подстановкой переменных окружения.

        Если файл не найден, возвращает пустой словарь. Выполняется простая
        подстановка `${VAR}` на значения из окружения перед парсингом YAML.
        """
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
        """Делает слияние двух словарей, где `override` имеет приоритет.

        Args:
            base: Базовый словарь.
            override: Словарь с переопределениями (приоритетнее).
            deep: Если True, выполняется глубокое слияние для вложенных словарей.

        Returns:
            Новый словарь, полученный после слияния.
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
        """Загружает и валидированно формирует объект конфигурации.

        Последовательность приоритетов (от высокого к низкому):
        1. Переменные окружения (текущий профиль через `CONFIG_PROFILE`)
        2. Профильный файл (например `params.dev.yaml`)
        3. Базовый файл `params.yaml`.

        Возвращает валидированный объект `AppConfig`.
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
        """Возвращает загруженную конфигурацию, загрузив при необходимости.

        Если конфигурация ещё не была загружена, выполняется `load_config()`.
        Возвращает объект `AppConfig`.
        """
        if self._config is None:
            self.load_config()
        return self._config

    def compose_configs(self, *configs: AppConfig) -> AppConfig:
        """Компонует несколько `AppConfig` в один.

        Более поздние конфигурации переопределяют поля более ранних. Результат
        будет иметь профиль, соответствующий текущему менеджеру.
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
        """Предопределяет параметры конкретной модели в конфигурации.

        Args:
            model_name: Имя модели для изменения параметров.
            **params: Параметры модели, которые будут добавлены/заменены.

        Возвращает обновлённый объект `AppConfig`.
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
    """Инициализирует глобальный менеджер конфигурации и загрузить конфиг.

    Args:
        profile: Необязательный профиль для инициализации менеджера.

    Возвращает загруженный `AppConfig`.
    """
    global _config_manager
    _config_manager = ConfigManager(profile)
    return _config_manager.load_config()


def get_config_manager() -> ConfigManager:
    """Возвращает глобальный экземпляр `ConfigManager`. Инициализирует при необходимости."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_config()
    return _config_manager


def get_config() -> AppConfig:
    """Получает глобальную конфигурацию приложения."""
    return get_config_manager().get_config()
