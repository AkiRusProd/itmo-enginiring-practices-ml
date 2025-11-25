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
from result_logger import PipelineLogger, log_results
from schemas import EvalMetrics

logger = PipelineLogger("evaluation")

np.random.seed(SEED)


def main():
    try:
        logger.info("Starting model evaluation...")

        model = load(f"{MODEL_DIR}/best_model.pkl")
        df = pd.read_csv(DATA_PATH)

        X_train, X_val, y_train, y_val = train_test_split(
            df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=SEED
        )

        preds = model.predict(X_val)

        metrics = EvalMetrics(
            accuracy=accuracy_score(y_val, preds),
            precision=precision_score(y_val, preds, average="weighted"),
            recall=recall_score(y_val, preds, average="weighted"),
            f1=f1_score(y_val, preds, average="weighted"),
            confusion_matrix=confusion_matrix(y_val, preds).tolist(),
        ).model_dump(mode="json")

        Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{METRICS_DIR}/best_model_advanced_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        tg_metrics = {k: v for k, v in metrics.items() if k != "confusion_matrix"}

        log_results("success", "Evaluation completed successfully", metrics=tg_metrics)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        log_results("failed", f"Evaluation crashed: {e}")
        raise e


if __name__ == "__main__":
    main()
