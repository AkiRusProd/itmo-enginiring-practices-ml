import glob
import json
from pathlib import Path
from shutil import copyfile

import pandas as pd
from clearml import Task
from dotenv import load_dotenv

from base_config import METRICS_DIR, MODEL_DIR
from result_logger import PipelineLogger, log_results
from schemas import BestModelMetrics, TrainMetrics


def select_best_model():
    """
    Анализирует метрики обученных моделей, выбирает лучшую,
    создает артефакт 'best_model.pkl' и регистрирует его в ClearML.
    """
    # Инициализация локального логгера
    pipeline_logger = PipelineLogger("select_best")

    # Пытаемся получить текущую задачу (от PipelineController или из main)
    task = Task.current_task()
    clearml_logger = task.get_logger() if task else None

    if task:
        task.add_tags(["optimizer", "selection"])

    pipeline_logger.info("Starting best model selection...")

    try:
        # 1. Сбор метрик из файлов
        metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")

        if not metrics_files:
            msg = f"❌ No metrics files found in {METRICS_DIR}/train_models. Did training run?"
            pipeline_logger.error(msg)
            raise FileNotFoundError(msg)

        all_metrics = {}
        best_acc = -1.0
        best_model = None
        comparison_data = []

        for file in metrics_files:
            with open(file, "r") as f:
                metrics_data = json.load(f)
                # Валидация через Pydantic
                metrics = TrainMetrics(**metrics_data)

                model_name = metrics.model
                acc = metrics.f1_score

                all_metrics[model_name] = acc
                comparison_data.append({"Model": model_name, "F1_Score": acc})

                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name

        if not best_model:
            raise ValueError(
                "Could not determine best model (list is empty or errors occurred)"
            )

        # 2. Логирование таблицы сравнения в ClearML
        if comparison_data and clearml_logger:
            df_comparison = pd.DataFrame(comparison_data).sort_values(
                by="F1_Score", ascending=False
            )
            clearml_logger.report_table(
                "Model Comparison", "Candidates", iteration=0, table_plot=df_comparison
            )
            pipeline_logger.info("Comparison table reported to ClearML.")

        # 3. Сохранение сводного JSON
        final_metrics = BestModelMetrics(
            best_model=best_model,
            best_f1_score=best_acc,
            models=all_metrics,
        )
        metrics_dict = final_metrics.model_dump(mode="json")

        Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
        with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

        # 4. Загрузка JSON как артефакта ClearML
        if task:
            task.upload_artifact("best_model_metrics", metrics_dict)

        # 5. Обработка лучшей модели (копирование и регистрация)
        pipeline_logger.info(f"🏆 Selected champion: {best_model} (F1: {best_acc:.4f})")

        if clearml_logger:
            clearml_logger.report_text(
                f"Selected champion: {best_model} (F1: {best_acc:.4f})"
            )
            clearml_logger.report_scalar(
                "Best Score", "F1", value=best_acc, iteration=1
            )

        src_path = f"{MODEL_DIR}/{best_model}.pkl"
        dst_path = f"{MODEL_DIR}/best_model.pkl"

        try:
            copyfile(src_path, dst_path)
            pipeline_logger.info(f"Copied {src_path} to {dst_path}")
        except FileNotFoundError:
            pipeline_logger.error(f"❌ Error: Model file {src_path} not found!")
            raise

        # Регистрируем модель как 'Production' версию в ClearML
        if task:
            task.update_output_model(
                model_path=dst_path,
                model_name="Best Model Production",
                tags=["production", "champion", best_model],
                auto_delete_file=False,
                comment=f"Winner from pipeline with F1={best_acc:.4f}",
            )
            pipeline_logger.info("Best model uploaded/registered in ClearML.")

        # 6. Уведомление в общий лог результатов
        summary = {"selected_model": best_model, "best_f1_score": best_acc}
        log_results("success", f"Best model selected: {best_model}", metrics=summary)

        return best_model

    except Exception as e:
        pipeline_logger.error(f"Selection failed: {e}")
        # Логируем ошибку и локально, и в ClearML
        log_results("failed", f"Model selection crashed: {e}")
        if task:
            task.mark_failed(status_message=str(e))
        raise e


if __name__ == "__main__":
    load_dotenv()

    # Логгер для main блока
    main_logger = PipelineLogger("select_best_main")

    # Инициализация задачи ТОЛЬКО при ручном запуске
    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Select Best Model",
        task_type=Task.TaskTypes.optimizer,
        output_uri=True,
        auto_connect_frameworks=False,
    )

    main_logger.info(f"Task initialized manually. ID: {task.id}")

    try:
        select_best_model()
    except Exception as e:
        main_logger.error(f"Ошибка при выполнении select_best_model: {e}")

    main_logger.info("Closing task...")
    task.close()
    main_logger.info("Task closed.")
