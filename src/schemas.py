"""Pydantic schemas for configuration and metrics validation."""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


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

    class Config:
        extra = "allow"  # Allow additional fields for model-specific parameters


class ModelsConfig(BaseModel):
    """Container for all model configurations."""

    LogisticRegression: Optional[Dict[str, Any]] = None
    RandomForest: Optional[Dict[str, Any]] = None
    GradientBoosting: Optional[Dict[str, Any]] = None
    SVC: Optional[Dict[str, Any]] = None
    DecisionTree: Optional[Dict[str, Any]] = None
    AdaBoost: Optional[Dict[str, Any]] = None
    Bagging: Optional[Dict[str, Any]] = None
    KNeighbors: Optional[Dict[str, Any]] = None
    LinearSVC: Optional[Dict[str, Any]] = None
    GaussianNB: Optional[Dict[str, Any]] = None
    MultinomialNB: Optional[Dict[str, Any]] = None
    BernoulliNB: Optional[Dict[str, Any]] = None
    ExtraTrees: Optional[Dict[str, Any]] = None
    RidgeClassifier: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"


class AppConfig(BaseModel):
    """Application configuration."""

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

    class Config:
        extra = "allow"


class TrainMetrics(BaseModel):
    """Metrics from single model training."""

    model: str = Field(..., description="Model name")
    f1_score: float = Field(..., description="F1 score")
    timestamp: datetime = Field(
        default_factory=datetime.now, description="Training timestamp"
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
    best_accuracy: float = Field(..., description="Best model accuracy")
    models: Dict[str, float] = Field(
        default_factory=dict, description="All models metrics"
    )
