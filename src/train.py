import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
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
from result_logger import PipelineLogger, log_results, print_results
from schemas import BestModelMetrics
from utils import log_experiment, save_model

# Инициализация логирования
pipeline_logger = PipelineLogger("train", log_dir=LOG_DIR)
pipeline_logger.info("Начало тренировки моделей")

np.random.seed(SEED)
random.seed(SEED)

# Load data
pipeline_logger.info(f"Загрузка данных из {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
X = df[FEATURES]
y = df[TARGET]

pipeline_logger.info(f"Размер данных: {df.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)

pipeline_logger.info(f"Разделение: train={X_train.shape[0]}, val={X_val.shape[0]}")


@log_experiment()
def train_model(name, model, X_train, y_train, X_val, y_val, writer=None):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")  # заменили на F1-score

    if writer:
        writer.add_scalar("F1_score", f1, 0)
        writer.add_text("Params", str(model.get_params()), 0)

    save_model(model, f"{MODEL_DIR}/{name}.pkl")
    return f1


metrics = {}
best_model = None
best_f1 = -1

# Создаем папку для моделей
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

pipeline_logger.info(f"Начало тренировки {len(MODELS)} моделей...")

for i, (name, model) in enumerate(MODELS.items(), start=1):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_name = f"exp_{i}_{name}_{timestamp}"

    pipeline_logger.info(f"[{i}/{len(MODELS)}] Тренировка модели: {name}")

    f1 = train_model(
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        exp_name=exp_name,
        log_dir=LOG_DIR,
    )
    metrics[name] = f1
    pipeline_logger.info(f"  {name}: F1-score = {f1:.4f}")

    if f1 > best_f1:
        best_f1, best_model = f1, name

# Validate using Pydantic schema
best_metrics = BestModelMetrics(
    best_model=best_model,
    best_f1_score=best_f1,
    models=metrics,
)
metrics_dict = best_metrics.model_dump(mode="json")

# Save best model
best_model_path = Path(MODEL_DIR) / "best_model.pkl"
save_model(MODELS[best_model], best_model_path)
pipeline_logger.info(f"\n✓ Лучшая модель: {best_model} (F1: {best_f1:.4f})")
pipeline_logger.info(f"✓ Сохранена в: {best_model_path}")

# Save metrics using validated schema
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
with open(METRICS_FILE, "w") as f:
    json.dump(metrics_dict, f, indent=4)

pipeline_logger.info(f"✓ Метрики сохранены в: {METRICS_FILE}")

# Логирование результатов
log_results(
    "success",
    f"Пайплайн завершен. Лучшая модель: {best_model} (F1: {best_f1:.4f})",
    metrics_dict,
)

# Вывод результатов
print_results(METRICS_FILE)
