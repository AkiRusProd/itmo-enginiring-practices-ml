import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    FEATURES,
    LOG_DIR,
    METRICS_DIR,
    METRICS_FILE,
    MODEL_DIR,
    MODELS,
    SEED,
    TARGET,
    TEST_SIZE,
)
from utils import log_experiment, save_model

np.random.seed(SEED)
random.seed(SEED)

# Load data
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)


@log_experiment()
def train_model(name, model, X_train, y_train, X_val, y_val, writer=None):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    writer.add_scalar("Accuracy", acc, 0)
    writer.add_text("Params", str(model.get_params()), 0)

    save_model(model, f"{MODEL_DIR}/{name}.pkl")
    return acc


metrics = {}
best_model = None
best_acc = -1

# Создаем папку для моделей
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

for i, (name, model) in enumerate(MODELS.items(), start=1):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"exp_{i}_{name}_{timestamp}"

    acc = train_model(
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        exp_name=exp_name,
        log_dir=LOG_DIR,
    )
    metrics[name] = acc

    if acc > best_acc:
        best_acc, best_model = acc, name

metrics["best_model"] = best_model
metrics["best_accuracy"] = best_acc

best_model_path = Path(MODEL_DIR) / "best_model.pkl"
save_model(MODELS[best_model], best_model_path)
print(f"Best model ({best_model}) saved to {best_model_path}")

Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)
