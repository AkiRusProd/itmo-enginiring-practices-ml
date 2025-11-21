import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from base_config import (
    DATA_PATH,
    FEATURES,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    MODELS,
    PROFILE,
    SEED,
    TARGET,
    TB_LOG_DIR,
    TEST_SIZE,
)
from result_logger import PipelineLogger, log_results
from schemas import TrainMetrics
from utils import log_experiment, save_model

np.random.seed(SEED)

# Инициализация логирования
pipeline_logger = PipelineLogger("train_single", log_dir=LOG_DIR)


@log_experiment()
def train_single_model(model_name, writer=None):
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

    # Get specific model
    pipeline_logger.info(f"Тренировка модели: {model_name}")
    model = MODELS[model_name]

    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")  # заменили на F1-score

    pipeline_logger.info(f"✓ {model_name}: F1-score = {f1:.4f}")

    # Log metrics and params
    if writer:
        writer.add_scalar("F1_score", f1, 0)
        writer.add_text("Params", str(model.get_params()), 0)

    # Save model
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{MODEL_DIR}/{model_name}.pkl")
    pipeline_logger.info(f"✓ Модель сохранена в: {MODEL_DIR}/{model_name}.pkl")

    # Save metrics to JSON using Pydantic validation
    metrics_path = Path(f"{METRICS_DIR}/train_models") / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate metrics using Pydantic schema
    metrics_obj = TrainMetrics(model=model_name, f1_score=f1, profile=PROFILE)
    metrics_dict = metrics_obj.model_dump(mode="json")

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    pipeline_logger.info(f"✓ Метрики сохранены в: {metrics_path}")

    # Логирование результатов
    log_results(
        "success",
        f"Модель {model_name} обучена. F1-score: {f1:.4f}",
        metrics_dict,
    )

    print(f"{model_name}: F1-score = {f1:.4f}")
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to train")
    parser.add_argument(
        "--exp_name", required=False, default=None, help="Experiment name for logging"
    )
    args = parser.parse_args()

    exp_name = (
        args.exp_name
        or f"train_single_{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    pipeline_logger.info(f"Начало обучения отдельной модели: {args.model}")
    try:
        train_single_model(args.model, exp_name=exp_name, log_dir=TB_LOG_DIR)
        pipeline_logger.info(f"✓ Обучение модели {args.model} завершено успешно")
    except Exception as e:
        pipeline_logger.error(f"❌ Ошибка при обучении {args.model}: {e}")
        log_results(
            "failed",
            f"Обучение модели {args.model} завершилось с ошибкой: {e}",
        )
        raise
