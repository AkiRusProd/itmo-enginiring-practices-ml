"""
Модуль инициализации и экспорта базовой конфигурации приложения.

Этот модуль отвечает за загрузку настроек через ConfigManager, определение
путей к файлам данных и логов, а также инициализацию объектов моделей
на основе загруженных параметров.
"""
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config_manager import get_config_manager, initialize_config
from schemas import AppConfig

MODEL_CLASSES = {
    "LogisticRegression": LogisticRegression,
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "KNeighbors": KNeighborsClassifier,
    "SVC": SVC,
    "LinearSVC": LinearSVC,
    "GaussianNB": GaussianNB,
    "MultinomialNB": MultinomialNB,
    "BernoulliNB": BernoulliNB,
    "AdaBoost": AdaBoostClassifier,
    "ExtraTrees": ExtraTreesClassifier,
    "Bagging": BaggingClassifier,
    "RidgeClassifier": RidgeClassifier,
}


# Initialize configuration using ConfigManager
_config = initialize_config()

# Export configuration values
SEED = _config.train.seed
TEST_SIZE = _config.train.test_size

RAW_DATA_PATH = _config.raw_data_path
DATA_PATH = _config.data_path
MODEL_DIR = _config.model_dir
LOG_DIR = _config.log_dir
TB_LOG_DIR = _config.tb_log_dir
METRICS_DIR = _config.metrics_dir
METRICS_FILE = f"{METRICS_DIR}/metrics.json"
PROFILE = _config.profile

FEATURES = _config.features
TARGET = _config.target

# Build only models listed in params.yaml
_models_cfg = _config.models.model_dump(exclude_none=True)
MODELS = {
    name: MODEL_CLASSES[name](**params)
    for name, params in _models_cfg.items()
    if name in MODEL_CLASSES
}


# For testing and direct access
def get_config() -> AppConfig:
    """Получает текущий валидированный объект конфигурации.

    Если конфигурация еще не была инициализирована, создает глобальный экземпляр
    менеджера конфигурации и загружает настройки.

    Returns:
        AppConfig: Объект конфигурации, содержащий параметры обучения, путей и моделей.
    """
    return get_config_manager().get_config()


def reload_config() -> AppConfig:
    """Принудительно перезагружает конфигурацию из YAML-файлов.

    Обновляет глобальную переменную `_config` актуальными данными с диска.
    Полезно при изменении файлов конфигурации во время выполнения (например, в ноутбуках).

    Returns:
        AppConfig: Обновленный объект конфигурации.
    """
    global _config
    _config = get_config_manager().load_config()
    return _config
