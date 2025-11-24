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
    load_dotenv()

    # Инициализация
    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Select Best Model",
        task_type=Task.TaskTypes.optimizer,
        output_uri=True,
        auto_connect_frameworks=False,  # Отключаем авто-магию, так как тут нет обучения
    )
    logger = task.get_logger()

    print("Task initialized. ID:", task.id)  # Проверка, что таск создался

    metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")
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

    # Таблица
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data).sort_values(
            by="F1_Score", ascending=False
        )
        logger.report_table(
            "Model Comparison", "Candidates", iteration=0, table_plot=df_comparison
        )

    # Сохранение JSON
    final_metrics = BestModelMetrics(
        best_model=best_model if best_model else "None",
        best_f1_score=best_acc,
        models=all_metrics,
    )
    metrics_dict = final_metrics.model_dump(mode="json")

    Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    task.upload_artifact("best_model_metrics", metrics_dict)

    if best_model:
        print(f"Best model: {best_model} with f1 score: {best_acc:.4f}")
        logger.report_text(f"Selected champion: {best_model} (F1: {best_acc:.4f})")
        logger.report_scalar("Best Score", "F1", value=best_acc, iteration=1)

        src_path = f"{MODEL_DIR}/{best_model}.pkl"
        dst_path = f"{MODEL_DIR}/best_model.pkl"
        copyfile(src_path, dst_path)

        # Регистрируем модель
        task.update_output_model(
            model_path=dst_path,
            model_name="Best Model Production",
            tags=["production", "champion", best_model],
            auto_delete_file=False,
        )
        print("Model uploaded to ClearML.")
    else:
        print("No models found.")

    # !!! ВАЖНО: Ждем завершения отправки данных !!!
    print("Closing task...")
    task.close()
    print("Task closed.")


if __name__ == "__main__":
    select_best_model()
