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

# Load params.yaml
params = yaml.safe_load(open("params.yaml"))
train_cfg = params["train"]
models_cfg = params["models"]

SEED = train_cfg["seed"]
TEST_SIZE = train_cfg["test_size"]

RAW_DATA_PATH = "data/raw/dataset.csv"
DATA_PATH = "data/processed/processed.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
METRICS_DIR = "metrics"
METRICS_FILE = f"{METRICS_DIR}/metrics.json"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"

# Build only models listed in params.yaml
MODELS = {name: MODEL_CLASSES[name](**params) for name, params in models_cfg.items()}
