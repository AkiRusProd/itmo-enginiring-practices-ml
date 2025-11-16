import json
import pickle  # nosec
import random
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorboardX import SummaryWriter

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


# ===== Утилиты для работы с моделями =====
def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec


# ===== Контекстный менеджер для эксперимента =====
@contextmanager
def experiment(exp_name):
    writer = SummaryWriter(log_dir=f"logs/{exp_name}")
    try:
        yield writer
    finally:
        writer.close()


# ===== Декоратор для автоматического логирования =====
def log_experiment():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp_name = kwargs.pop("exp_name")
            with experiment(exp_name) as writer:
                return func(*args, writer=writer, **kwargs)

        return wrapper

    return decorator


# ===== Загружаем предобработанные данные =====
train_df = pd.read_csv("data/processed/processed.csv")
X = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# ===== Список моделей =====
models = {
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


# ===== Функция для обучения модели с логированием =====
@log_experiment()
def train_model(name, model, X_train, y_train, X_val, y_val, writer=None):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{name} accuracy: {acc}")

    # Логируем в TensorBoard
    writer.add_scalar("Accuracy", acc, 0)
    writer.add_text("Params", str(model.get_params()), 0)

    # Сохраняем модель
    model_path = f"models/{name}.pkl"
    save_model(model, model_path)
    print(f"Saved model to {model_path}")

    return acc


# ===== Обучаем все модели и сохраняем метрики =====
metrics = {}
best_acc = -1
best_model_name = None

for i, (name, model) in enumerate(models.items(), start=1):
    exp_name = f"experiment_{i}_{name}"
    acc = train_model(name, model, X_train, y_train, X_val, y_val, exp_name=exp_name)
    metrics[name] = acc

    # Отслеживаем лучшую модель
    if acc > best_acc:
        best_acc = acc
        best_model_name = name

metrics["best_model"] = best_model_name
metrics["best_accuracy"] = best_acc

# Сохраняем метрики в JSON
Path("metrics").mkdir(parents=True, exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved metrics to metrics/metrics.json")
print(f"Best model: {best_model_name} with accuracy {best_acc}")
