import random

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Загружаем предобработанные данные
train = pd.read_csv("data/preprocessed/preprocessed.csv")

X = train[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(),
}

for i, (name, model) in enumerate(models.items(), start=1):
    writer = SummaryWriter(log_dir=f"logs/experiment_{i}_{name}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"{name} accuracy: {acc}")

    writer.add_scalar("Accuracy", acc, 0)
    writer.add_text("Params", str(model.get_params()), 0)
    writer.close()
