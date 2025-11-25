import glob
import json
from pathlib import Path
from shutil import copyfile

from base_config import METRICS_DIR, MODEL_DIR
from result_logger import PipelineLogger, log_results
from schemas import BestModelMetrics, TrainMetrics

logger = PipelineLogger("select_best")


def select_best_model():
    try:
        logger.info("Starting best model selection...")

        metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")
        if not metrics_files:
            raise FileNotFoundError(
                f"No metrics files found in {METRICS_DIR}/train_models"
            )

        all_metrics = {}
        best_acc = -1.0
        best_model = None

        # 1. Поиск лучшей модели
        for file in metrics_files:
            with open(file, "r") as f:
                data = TrainMetrics(**json.load(f))
                all_metrics[data.model] = data.f1_score

                if data.f1_score > best_acc:
                    best_acc = data.f1_score
                    best_model = data.model

        if not best_model:
            raise ValueError(
                "Could not determine best model (list is empty or errors occurred)"
            )

        # 2. Сохранение итоговых метрик
        final_metrics = BestModelMetrics(
            best_model=best_model,
            best_f1_score=best_acc,
            models=all_metrics,
        ).model_dump(mode="json")

        Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
            json.dump(final_metrics, f, indent=4)

        # 3. Копирование файла модели
        copyfile(f"{MODEL_DIR}/{best_model}.pkl", f"{MODEL_DIR}/best_model.pkl")

        logger.info(f"🏆 Selected: {best_model} (F1: {best_acc:.4f})")

        # 4. Уведомление в Telegram
        summary = {"selected_model": best_model, "best_f1_score": best_acc}
        log_results("success", f"Best model selected: {best_model}", metrics=summary)

    except Exception as e:
        logger.error(f"Selection failed: {e}")
        log_results("failed", f"Model selection crashed: {e}")
        raise e


if __name__ == "__main__":
    select_best_model()
