import json
from pathlib import Path

import pandas as pd
from joblib import load  # nosec
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import DATA_PATH, FEATURES, METRICS_DIR, MODEL_DIR, TARGET

# Загружаем лучшую модель
best_model_path = f"{MODEL_DIR}/best_model.pkl"
model = load(best_model_path)  # nosec

# Загружаем данные
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

# Предсказания
preds = model.predict(X)

# Считаем несколько метрик
eval_metrics = {
    "accuracy": accuracy_score(y, preds),
    "precision": precision_score(y, preds, average="weighted"),
    "recall": recall_score(y, preds, average="weighted"),
    "f1": f1_score(y, preds, average="weighted"),
    "confusion_matrix": confusion_matrix(y, preds).tolist(),
}

# Создаем папку и сохраняем
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{METRICS_DIR}/best_model_advanced_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=4)

print(
    f"Evaluation complete. Metrics saved to {METRICS_DIR}/best_model_advanced_metrics.json"
)
