import glob
import json
from pathlib import Path
from shutil import copyfile

import pandas as pd
from clearml import Task
from dotenv import load_dotenv

from base_config import METRICS_DIR, MODEL_DIR
from schemas import BestModelMetrics, TrainMetrics


def select_best_model():
    """
    Анализирует метрики обученных моделей, выбирает лучшую,
    создает артефакт 'best_model.pkl' и регистрирует его в ClearML.
    """

    # Пытаемся получить текущую задачу (от PipelineController или из main)
    task = Task.current_task()
    logger = task.get_logger() if task else None

    if task:
        task.add_tags(["optimizer", "selection"])

    print("Starting selection of best model...")

    # 1. Сбор метрик из файлов
    # Пайплайн гарантирует, что шаги обучения завершены и файлы существуют
    metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")

    if not metrics_files:
        print("❌ No metrics files found. Did training run?")
        return None

    all_metrics = {}
    best_acc = -1
    best_model = None
    comparison_data = []

    for file in metrics_files:
        with open(file, "r") as f:
            metrics_data = json.load(f)
            metrics = TrainMetrics(**metrics_data)
            model_name = metrics.model
            acc = metrics.f1_score

            all_metrics[model_name] = acc
            comparison_data.append({"Model": model_name, "F1_Score": acc})

            if acc > best_acc:
                best_acc = acc
                best_model = model_name

    # 2. Логирование таблицы сравнения в ClearML
    if comparison_data and logger:
        df_comparison = pd.DataFrame(comparison_data).sort_values(
            by="F1_Score", ascending=False
        )
        logger.report_table(
            "Model Comparison", "Candidates", iteration=0, table_plot=df_comparison
        )
        print("Comparison table reported to ClearML.")

    # 3. Сохранение сводного JSON
    final_metrics = BestModelMetrics(
        best_model=best_model if best_model else "None",
        best_f1_score=best_acc,
        models=all_metrics,
    )
    metrics_dict = final_metrics.model_dump(mode="json")

    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # 4. Загрузка JSON как артефакта
    if task:
        task.upload_artifact("best_model_metrics", metrics_dict)

    # 5. Обработка лучшей модели (копирование и регистрация)
    if best_model:
        print(f"✅ Best model found: {best_model} with F1: {best_acc:.4f}")

        if logger:
            logger.report_text(f"Selected champion: {best_model} (F1: {best_acc:.4f})")
            logger.report_scalar("Best Score", "F1", value=best_acc, iteration=1)

        src_path = f"{MODEL_DIR}/{best_model}.pkl"
        dst_path = f"{MODEL_DIR}/best_model.pkl"

        # Копируем локально, чтобы файл был доступен следующему шагу (eval)
        try:
            copyfile(src_path, dst_path)
            print(f"Copied {src_path} to {dst_path}")
        except FileNotFoundError:
            print(f"❌ Error: Model file {src_path} not found!")
            return None

        # Регистрируем модель как 'Production' версию в ClearML
        if task:
            task.update_output_model(
                model_path=dst_path,
                model_name="Best Model Production",
                tags=["production", "champion", best_model],
                auto_delete_file=False,
                comment=f"Winner from pipeline with F1={best_acc:.4f}",
            )
            print("Best model uploaded/registered in ClearML.")

        return best_model
    else:
        print("⚠️ No models found to select from.")
        return None


if __name__ == "__main__":
    load_dotenv()

    # Инициализация задачи ТОЛЬКО при ручном запуске
    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Select Best Model",
        task_type=Task.TaskTypes.optimizer,
        output_uri=True,
        auto_connect_frameworks=False,
    )

    print(f"Task initialized manually. ID: {task.id}")

    select_best_model()

    print("Closing task...")
    task.close()
    print("Task closed.")
