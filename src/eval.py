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
from schemas import EvalMetrics


def evaluate_model():
    load_dotenv()
    np.random.seed(SEED)

    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Advanced Evaluation",
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks=False,
    )
    logger = task.get_logger()
    print("Task initialized. ID:", task.id)

    best_model_path = f"{MODEL_DIR}/best_model.pkl"
    try:
        model = load(best_model_path)
    except FileNotFoundError:
        print("❌ Best model not found!")
        task.close()  # Закрываем, если ошибка
        return

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="weighted")
    rec = recall_score(y_val, preds, average="weighted")
    f1 = f1_score(y_val, preds, average="weighted")
    cm = confusion_matrix(y_val, preds)

    # Логирование
    logger.report_scalar("Evaluation", "Accuracy", acc, iteration=0)
    logger.report_scalar("Evaluation", "Precision", prec, iteration=0)
    logger.report_scalar("Evaluation", "Recall", rec, iteration=0)
    logger.report_scalar("Evaluation", "F1", f1, iteration=0)

    labels = [str(c) for c in sorted(y.unique())]
    logger.report_confusion_matrix(
        title="Confusion Matrix",
        series="Test Set",
        matrix=cm,
        iteration=0,
        xlabels=labels,
        ylabels=labels,
    )

    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(10, 6))
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
        plt.xlabel("Relative Importance")

        logger.report_matplotlib_figure(
            title="Feature Importance", series="Top Features", figure=plt
        )
        plt.close()

    eval_metrics_obj = EvalMetrics(
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        confusion_matrix=cm.tolist(),
    )
    eval_metrics = eval_metrics_obj.model_dump(mode="json")

    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{METRICS_DIR}/best_model_advanced_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=4)

    print(f"Evaluation complete. F1: {f1:.4f}")

    # !!! ВАЖНО: Закрываем задачу
    print("Closing task...")
    task.close()
    print("Task closed.")


if __name__ == "__main__":
    evaluate_model()
