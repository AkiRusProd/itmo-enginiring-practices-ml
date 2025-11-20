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

params = yaml.safe_load(open("params.yaml"))
train_cfg = params["train"]

params = yaml.safe_load(open("params.yaml"))
models_cfg = params["models"]

SEED = train_cfg["seed"]  # 42
TEST_SIZE = train_cfg["test_size"]  # 0.2

RAW_DATA_PATH = "data/raw/dataset.csv"
DATA_PATH = "data/processed/processed.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
METRICS_DIR = "metrics"
METRICS_FILE = f"{METRICS_DIR}/metrics.json"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"

MODELS = {
    "LogisticRegression": LogisticRegression(
        **models_cfg.get("LogisticRegression", {})
    ),
    "RandomForest": RandomForestClassifier(**models_cfg.get("RandomForest", {})),
    "GradientBoosting": GradientBoostingClassifier(
        **models_cfg.get("GradientBoosting", {})
    ),
    "DecisionTree": DecisionTreeClassifier(**models_cfg.get("DecisionTree", {})),
    "KNeighbors": KNeighborsClassifier(**models_cfg.get("KNeighbors", {})),
    "SVC": SVC(**models_cfg.get("SVC", {})),
    "LinearSVC": LinearSVC(**models_cfg.get("LinearSVC", {})),
    "GaussianNB": GaussianNB(**models_cfg.get("GaussianNB", {})),
    "MultinomialNB": MultinomialNB(**models_cfg.get("MultinomialNB", {})),
    "BernoulliNB": BernoulliNB(**models_cfg.get("BernoulliNB", {})),
    "AdaBoost": AdaBoostClassifier(**models_cfg.get("AdaBoost", {})),
    "ExtraTrees": ExtraTreesClassifier(**models_cfg.get("ExtraTrees", {})),
    "Bagging": BaggingClassifier(**models_cfg.get("Bagging", {})),
    "RidgeClassifier": RidgeClassifier(**models_cfg.get("RidgeClassifier", {})),
}
