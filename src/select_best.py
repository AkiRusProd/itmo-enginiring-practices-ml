import glob
import json
from pathlib import Path

from config import METRICS_DIR, MODEL_DIR
from schemas import BestModelMetrics, TrainMetrics


def select_best_model():
    metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")
    all_metrics = {}
    best_acc = -1
    best_model = None

    for file in metrics_files:
        with open(file, "r") as f:
            metrics_data = json.load(f)
            # Validate using Pydantic schema
            metrics = TrainMetrics(**metrics_data)
            model_name = metrics.model
            acc = metrics.f1_score

            all_metrics[model_name] = acc

            if acc > best_acc:
                best_acc = acc
                best_model = model_name

    # Create and validate final metrics using Pydantic schema
    final_metrics = BestModelMetrics(
        best_model=best_model,
        best_f1_score=best_acc,
        models=all_metrics,
    )
    metrics_dict = final_metrics.model_dump(mode="json")

    # Create final metrics file
    Path("metrics").mkdir(parents=True, exist_ok=True)

    with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    print(f"Best model: {best_model} with f1 score: {best_acc:.4f}")

    # Optionally, copy best model to a separate file
    if best_model:
        from shutil import copyfile

        copyfile(f"{MODEL_DIR}/{best_model}.pkl", f"{MODEL_DIR}/best_model.pkl")


if __name__ == "__main__":
    select_best_model()
