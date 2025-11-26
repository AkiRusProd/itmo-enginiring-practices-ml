import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clearml import Task
from dotenv import load_dotenv
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
from result_logger import PipelineLogger, log_results
from schemas import EvalMetrics


def evaluate_model():
    """
    Загружает лучшую модель (best_model.pkl), проводит оценку на валидационном наборе,
    строит графики (Confusion Matrix, Feature Importance) и отправляет их в ClearML.
    """
    # Инициализация локального логгера
    pipeline_logger = PipelineLogger("evaluation")

    np.random.seed(SEED)

    # Пытаемся получить текущую задачу (от PipelineController или из main)
    task = Task.current_task()
    logger = task.get_logger() if task else None

    if task:
        task.add_tags(["evaluation", "testing"])

    pipeline_logger.info("Starting model evaluation...")

    # 1. Загрузка модели
    best_model_path = f"{MODEL_DIR}/best_model.pkl"
    try:
        model = load(best_model_path)
        pipeline_logger.info(f"Loaded model from {best_model_path}")
    except FileNotFoundError:
        msg = (
            f"❌ Best model not found at {best_model_path}! Run select_best step first."
        )
        pipeline_logger.error(msg)
        # Если запущено в пайплайне, лучше поднять ошибку, чтобы шаг покраснел
        raise

    # 2. Загрузка данных
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        msg = f"❌ Data file not found at {DATA_PATH}!"
        pipeline_logger.error(msg)
        raise

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # 3. Предикт
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="weighted")
    rec = recall_score(y_val, preds, average="weighted")
    f1 = f1_score(y_val, preds, average="weighted")
    cm = confusion_matrix(y_val, preds)

    # 4. Логирование скаляров в ClearML
    if logger:
        logger.report_scalar("Evaluation", "Accuracy", acc, iteration=0)
        logger.report_scalar("Evaluation", "Precision", prec, iteration=0)
        logger.report_scalar("Evaluation", "Recall", rec, iteration=0)
        logger.report_scalar("Evaluation", "F1", f1, iteration=0)

    # 5. Логирование Confusion Matrix в ClearML
    if logger:
        labels = [str(c) for c in sorted(y.unique())]
        logger.report_confusion_matrix(
            title="Confusion Matrix",
            series="Test Set",
            matrix=cm,
            iteration=0,
            xlabels=labels,
            ylabels=labels,
        )

    # 6. Логирование Feature Importance (если поддерживается)
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
        plt.xlabel("Relative Importance")

        if logger:
            logger.report_matplotlib_figure(
                title="Feature Importance", series="Top Features", figure=plt
            )
        plt.close()

    # 7. Сохранение метрик локально и как артефакт
    eval_metrics_obj = EvalMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm.tolist(),
    )
    eval_metrics = eval_metrics_obj.model_dump(mode="json")

    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    json_path = f"{METRICS_DIR}/best_model_advanced_metrics.json"
    with open(json_path, "w") as f:
        json.dump(eval_metrics, f, indent=4)

    pipeline_logger.info(f"Evaluation complete. F1: {f1:.4f}")

    if task:
        task.upload_artifact("advanced_metrics", eval_metrics)
        pipeline_logger.info("Metrics uploaded to ClearML.")

    # Логирование результата через log_results (как во втором файле)
    # Исключаем матрицу ошибок из текстового лога, так как она большая
    tg_metrics = {k: v for k, v in eval_metrics.items() if k != "confusion_matrix"}
    log_results("success", "Evaluation completed successfully", metrics=tg_metrics)


if __name__ == "__main__":
    load_dotenv()

    # Логгер для блока main на случай падения до инициализации функции
    main_logger = PipelineLogger("eval_main")

    # Инициализация задачи только при ручном запуске
    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Advanced Evaluation",
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks=False,
    )

    try:
        evaluate_model()
    except Exception as e:
        main_logger.error(f"❌ Evaluation failed: {e}")
        log_results("failed", f"Evaluation crashed: {e}")
        task.mark_failed(status_message=str(e))
        raise

    main_logger.info("Closing task...")
    task.close()
    main_logger.info("Task closed.")
