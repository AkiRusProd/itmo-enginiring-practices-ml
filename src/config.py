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

SEED = 42

RAW_DATA_PATH = "data/raw/dataset.csv"
DATA_PATH = "data/processed/processed.csv"
MODEL_DIR = "models"
LOG_DIR = "logs"
METRICS_DIR = "metrics"
METRICS_FILE = f"{METRICS_DIR}/metrics.json"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
TARGET = "Survived"
TEST_SIZE = 0.2

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
    "DecisionTree": DecisionTreeClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "SVC_linear": SVC(kernel="linear", probability=True),
    "SVC_rbf": SVC(kernel="rbf", probability=True),
    "LinearSVC": LinearSVC(max_iter=2000),
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100),
    "Bagging": BaggingClassifier(n_estimators=100),
    "RidgeClassifier": RidgeClassifier(),
}
