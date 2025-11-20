import glob
import json
from pathlib import Path

from config import METRICS_DIR, MODEL_DIR


def select_best_model():
    metrics_files = glob.glob(f"{METRICS_DIR}/train_models/*_metrics.json")
    all_metrics = {}
    best_acc = -1
    best_model = None

    for file in metrics_files:
        with open(file, "r") as f:
            metrics = json.load(f)
            model_name = metrics.get("model")
            acc = metrics.get("f1_score")
            if model_name is None or acc is None:
                continue

            all_metrics[model_name] = acc

            if acc > best_acc:
                best_acc = acc
                best_model = model_name

    # Create final metrics file
    Path("metrics").mkdir(parents=True, exist_ok=True)
    final_metrics = {**all_metrics, "best_model": best_model, "best_accuracy": best_acc}

    with open(f"{METRICS_DIR}/best_model_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"Best model: {best_model} with accuracy: {best_acc:.4f}")

    # Optionally, copy best model to a separate file
    if best_model:
        from shutil import copyfile

        copyfile(f"{MODEL_DIR}/{best_model}.pkl", f"{MODEL_DIR}/best_model.pkl")


if __name__ == "__main__":
    select_best_model()
