"""Application configuration with Pydantic validation."""

from pathlib import Path

import yaml
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

from schemas import AppConfig, ModelsConfig, TrainConfig

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


def _load_config() -> AppConfig:
    """Load and validate configuration from params.yaml using Pydantic."""
    params_path = Path("params.yaml")
    if not params_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {params_path}")

    with open(params_path, "r") as f:
        params = yaml.safe_load(f)

    # Validate using Pydantic
    config = AppConfig(
        train=TrainConfig(**params.get("train", {})),
        models=ModelsConfig(**params.get("models", {})),
    )
    return config


# Load and validate configuration
_config = _load_config()

# Export configuration values
SEED = _config.train.seed
TEST_SIZE = _config.train.test_size

RAW_DATA_PATH = _config.raw_data_path
DATA_PATH = _config.data_path
MODEL_DIR = _config.model_dir
LOG_DIR = _config.log_dir
METRICS_DIR = _config.metrics_dir
METRICS_FILE = f"{METRICS_DIR}/metrics.json"

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
    """Get the validated configuration object."""
    return _config
