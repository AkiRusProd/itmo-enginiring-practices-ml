"""Pydantic schemas for configuration and metrics validation."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConfigProfile(str, Enum):
    """Configuration profiles for different environments."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class TrainConfig(BaseModel):
    """Training configuration."""

    seed: int = Field(42, description="Random seed for reproducibility")
    test_size: float = Field(0.2, description="Test set size")

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v):
        if not 0 < v < 1:
            raise ValueError("test_size must be between 0 and 1")
        return v


class ModelConfig(BaseModel):
    """Base model configuration."""

    model_class: str = Field(..., description="Model class name")

    model_config = ConfigDict(extra="allow")


class ModelsConfig(BaseModel):
    """Container for all model configurations."""

    model_config = ConfigDict(extra="allow")

    def merge(self, other: "ModelsConfig") -> "ModelsConfig":
        """Merge with another ModelsConfig, with other taking precedence."""
        merged_data = self.model_dump(exclude_none=True)
        merged_data.update(other.model_dump(exclude_none=True))
        return ModelsConfig(**merged_data)


class AppConfig(BaseModel):
    """Application configuration."""

    profile: ConfigProfile = Field(
        ConfigProfile.DEV, description="Configuration profile (dev/test/prod)"
    )
    train: TrainConfig
    models: ModelsConfig
    seed: int = Field(42, description="Random seed")
    test_size: float = Field(0.2, description="Test set size")
    raw_data_path: str = Field("data/raw/dataset.csv", description="Path to raw data")
    data_path: str = Field(
        "data/processed/processed.csv", description="Path to processed data"
    )
    model_dir: str = Field("models", description="Directory for models")
    log_dir: str = Field("logs", description="Directory for logs")
    tb_log_dir: str = Field("tb_logs", description="Directory for TensorBoard logs")
    metrics_dir: str = Field("metrics", description="Directory for metrics")
    features: list[str] = Field(
        default_factory=lambda: [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
        ],
        description="Feature columns",
    )
    target: str = Field("Survived", description="Target column")

    model_config = ConfigDict(extra="allow")

    def merge(self, other: "AppConfig") -> "AppConfig":
        """Merge with another AppConfig, with other taking precedence."""
        merged_data = self.model_dump()
        other_data = other.model_dump()

        # Deep merge for nested structures
        for key, value in other_data.items():
            if key in ["train", "models"] and isinstance(value, dict):
                if key == "models":
                    # Merge models using ModelsConfig merge
                    base_models = ModelsConfig(**merged_data.get(key, {}))
                    other_models = ModelsConfig(**value)
                    merged_data[key] = base_models.merge(other_models).model_dump()
                else:
                    # Merge train config
                    merged_data[key] = {**merged_data.get(key, {}), **value}
            else:
                merged_data[key] = value

        return AppConfig(**merged_data)


class TrainMetrics(BaseModel):
    """Metrics from single model training."""

    model: str = Field(..., description="Model name")
    f1_score: float = Field(..., description="F1 score")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Training timestamp"
    )
    profile: Optional[ConfigProfile] = Field(
        default=None, description="Profile used for training"
    )


class EvalMetrics(BaseModel):
    """Advanced evaluation metrics."""

    accuracy: float = Field(..., description="Accuracy score")
    precision: float = Field(..., description="Precision score")
    recall: float = Field(..., description="Recall score")
    f1: float = Field(..., description="F1 score")
    confusion_matrix: list[list[int]] = Field(..., description="Confusion matrix")


class BestModelMetrics(BaseModel):
    """Best model selection metrics."""

    best_model: str = Field(..., description="Best model name")
    best_f1_score: float = Field(..., description="Best model f1 score")
    models: Dict[str, float] = Field(
        default_factory=dict, description="All models metrics"
    )
