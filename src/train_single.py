import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Task
from dotenv import load_dotenv
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
from config_manager import get_config
from result_logger import PipelineLogger, log_results
from schemas import TrainMetrics
from utils import log_experiment, save_model

np.random.seed(SEED)

# pipeline_logger удален отсюда, так как глобальные переменные теряются при запуске через add_function_step


@log_experiment()
def train_single_model(model_name, writer=None):
    """
    Функция обучения одной модели.
    Может вызываться как из __main__, так и из PipelineController.
    """

    # Инициализируем логгер ВНУТРИ функции, чтобы он был доступен в контексте пайплайна
    pipeline_logger = PipelineLogger("train_single", log_dir=LOG_DIR)

    # 1. Загрузка данных
    pipeline_logger.info(f"Загрузка данных из {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        pipeline_logger.error(f"Файл {DATA_PATH} не найден. Запустите preprocess.")
        raise

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # 2. Получение и обучение модели
    pipeline_logger.info(f"Тренировка модели: {model_name}")
    if model_name not in MODELS:
        raise ValueError(f"Модель {model_name} не найдена в конфигурации.")

    model = MODELS[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")

    pipeline_logger.info(f"✓ {model_name}: F1-score = {f1:.4f}")

    # 3. Логирование метрик в TensorBoard
    if writer:
        writer.add_scalar("F1_score", f1, 0)
        writer.add_text("Params", str(model.get_params()), 0)

    # 4. Сохранение модели локально
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    local_model_path = f"{MODEL_DIR}/{model_name}.pkl"
    save_model(model, local_model_path)
    pipeline_logger.info(f"✓ Модель сохранена в: {local_model_path}")

    # 5. Работа с ClearML Task
    task = Task.current_task()

    if task:
        task.add_tags([model_name, "candidate"])
        task.connect(model.get_params(), name="Model Params")

        task.update_output_model(
            model_path=local_model_path,
            model_name=model_name,
            comment="Scikit-Learn Candidate",
            auto_delete_file=False,
        )
        pipeline_logger.info("✓ Модель зарегистрирована в ClearML")

    # 6. Сохранение метрик в JSON
    metrics_path = Path(f"{METRICS_DIR}/train_models") / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    metrics_obj = TrainMetrics(model=model_name, f1_score=f1, profile=PROFILE)
    metrics_dict = metrics_obj.model_dump(mode="json")

    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    pipeline_logger.info(f"✓ Метрики сохранены в: {metrics_path}")

    log_results(
        "success",
        f"Модель {model_name} обучена. F1-score: {f1:.4f}",
        metrics_dict,
    )

    print(f"{model_name}: F1-score = {f1:.4f}")
    return f1


if __name__ == "__main__":
    load_dotenv()

    # Для скрипта создаем логгер здесь, но внутри функции он создаст свой локальный
    main_logger = PipelineLogger("train_single_main", log_dir=LOG_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to train")
    parser.add_argument(
        "--exp_name", required=False, default=None, help="Experiment name for logging"
    )
    temp_args, _ = parser.parse_known_args()

    task = Task.init(
        project_name="HW5_MLOps",
        task_name=f"Train candidate: {temp_args.model}",
        output_uri=True,
        auto_connect_frameworks=True,
    )

    try:
        full_config = get_config().model_dump(mode="json")
        task.connect_configuration(full_config, name="App Configuration")
    except Exception as e:
        main_logger.warning(f"Не удалось загрузить конфиг в ClearML: {e}")

    args = parser.parse_args()

    exp_name = (
        args.exp_name
        or f"train_single_{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    main_logger.info(f"Начало обучения отдельной модели: {args.model}")
    try:
        train_single_model(args.model, exp_name=exp_name, log_dir=TB_LOG_DIR)

        main_logger.info(f"✓ Обучение модели {args.model} завершено успешно")
        task.close()

    except Exception as e:
        main_logger.error(f"❌ Ошибка при обучении {args.model}: {e}")
        task.mark_failed(status_message=str(e))
        raise
