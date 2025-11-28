"""Система управления конфигурациями с поддержкой композиции, профилей и переменных окружения."""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from schemas import AppConfig, ConfigProfile, ModelsConfig, TrainConfig


class ConfigManager:
    """Менеджер конфигурации приложения.

    Обеспечивает загрузку, слияние и валидацию настроек из YAML-файлов.
    Поддерживает иерархию конфигураций:
    1. Переменные окружения (CONFIG_PROFILE)
    2. Профильные конфиги (dev, test, prod)
    3. Базовый конфиг (params.yaml)
    """

    # Default configuration file paths
    BASE_CONFIG = "params.yaml"
    PROFILE_CONFIGS = {
        ConfigProfile.DEV: "config/params.dev.yaml",
        ConfigProfile.TEST: "config/params.test.yaml",
        ConfigProfile.PROD: "config/params.prod.yaml",
    }

    def __init__(self, profile: Optional[ConfigProfile] = None):
        """Инициализирует менеджер конфигурации.

        Args:
            profile (Optional[ConfigProfile]): Явное указание профиля (dev/test/prod).
                Если не передано, пытается считать из переменной окружения `CONFIG_PROFILE`.
                По умолчанию используется `dev`.
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
        """Загружает YAML-файл с поддержкой подстановки переменных окружения.

        Считывает файл и заменяет конструкции вида `${VAR_NAME}` на значения
        соответствующих переменных окружения перед парсингом YAML.

        Args:
            path (str): Путь к YAML-файлу.

        Returns:
            dict: Словарь с загруженными данными или пустой словарь, если файл не найден.
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
        """Выполняет слияние двух словарей.

        Args:
            base (dict): Базовый словарь.
            override (dict): Словарь с переопределениями (имеет приоритет).
            deep (bool): Если True, выполняется рекурсивное слияние вложенных словарей.

        Returns:
            dict: Новый словарь, являющийся результатом слияния.
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
        """Загружает конфигурацию с учетом иерархии приоритетов.

        Последовательность загрузки:
        1. Базовый файл `params.yaml`.
        2. Файл профиля (например, `config/params.dev.yaml`), который переопределяет базовые значения.

        Returns:
            AppConfig: Валидированный Pydantic-объект конфигурации.

        Raises:
            FileNotFoundError: Если базовый файл конфигурации не найден.
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
        """Возвращает текущую загруженную конфигурацию.

        Если конфигурация еще не была загружена, вызывает `load_config()`.

        Returns:
            AppConfig: Объект конфигурации.
        """
        if self._config is None:
            self.load_config()
        return self._config

    def compose_configs(self, *configs: AppConfig) -> AppConfig:
        """Компонует несколько объектов конфигурации в один.

        Аргументы обрабатываются последовательно: каждая следующая конфигурация
        переопределяет значения предыдущей.

        Args:
            *configs: Переменное количество объектов AppConfig для слияния.

        Returns:
            AppConfig: Результирующий объединенный объект конфигурации.
        """
        if not configs:
            return self.get_config()

        result = configs[0]
        for config in configs[1:]:
            result = result.merge(config)

        # Ensure profile is set
        result.profile = self.profile
        return result

    def override_model_params(self, model_name: str, **params: Any) -> AppConfig:
        """Переопределяет параметры конкретной модели "на лету".

        Args:
            model_name (str): Имя модели (например, "RandomForest").
            **params: Именованные аргументы с новыми параметрами модели.

        Returns:
            AppConfig: Обновленный объект конфигурации.
        """
        config = self.get_config()
        models_data = config.models.model_dump(exclude_none=True)

        if model_name not in models_data:
            models_data[model_name] = {}

        models_data[model_name] = {**models_data[model_name], **params}
        config.models = ModelsConfig(**models_data)

        return config

    def to_dict(self) -> dict:
        """Преобразует текущую конфигурацию в словарь.

        Returns:
            dict: Словарь со всеми параметрами конфигурации.
        """
        return self.get_config().model_dump(mode="json")

    def to_yaml(self) -> str:
        """Преобразует текущую конфигурацию в строку формата YAML.

        Returns:
            str: Строковое представление конфигурации в YAML.
        """
        return yaml.dump(self.to_dict(), default_flow_style=False)


# Global instance
_config_manager: Optional[ConfigManager] = None


def initialize_config(profile: Optional[ConfigProfile] = None) -> AppConfig:
    """Инициализирует глобальный менеджер конфигурации.

    Args:
        profile (Optional[ConfigProfile]): Профиль конфигурации для инициализации.

    Returns:
        AppConfig: Загруженная конфигурация.
    """
    global _config_manager
    _config_manager = ConfigManager(profile)
    return _config_manager.load_config()


def get_config_manager() -> ConfigManager:
    """Возвращает глобальный экземпляр менеджера конфигурации.

    Если менеджер не существует, создает новый и загружает конфигурацию.

    Returns:
        ConfigManager: Экземпляр менеджера.
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.load_config()
    return _config_manager


def get_config() -> AppConfig:
    """Утилита для быстрого доступа к глобальной конфигурации.

    Returns:
        AppConfig: Текущая конфигурация приложения.
    """
    return get_config_manager().get_config()
