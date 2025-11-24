import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load  # nosec
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from base_config import (
    DATA_PATH,
    FEATURES,
    METRICS_DIR,
    MODEL_DIR,
    SEED,
    TARGET,
    TEST_SIZE,
)
from schemas import EvalMetrics

np.random.seed(SEED)

# Загружаем лучшую модель
best_model_path = f"{MODEL_DIR}/best_model.pkl"
model = load(best_model_path)  # nosec

# Загружаем данные
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

# Предсказания
preds = model.predict(X_val)

# Считаем несколько метрик и валидируем через Pydantic
eval_metrics_obj = EvalMetrics(
    accuracy=accuracy_score(y_val, preds),
    precision=precision_score(y_val, preds, average="weighted"),
    recall=recall_score(y_val, preds, average="weighted"),
    f1=f1_score(y_val, preds, average="weighted"),
    confusion_matrix=confusion_matrix(y_val, preds).tolist(),
)
eval_metrics = eval_metrics_obj.model_dump(mode="json")

# Создаем папку и сохраняем
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
with open(f"{METRICS_DIR}/best_model_advanced_metrics.json", "w") as f:
    json.dump(eval_metrics, f, indent=4)

print(
    f"Evaluation complete. Metrics saved to {METRICS_DIR}/best_model_advanced_metrics.json"
)
