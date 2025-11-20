import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from config import (
    DATA_PATH,
    FEATURES,
    LOG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    MODELS,
    SEED,
    TARGET,
    TEST_SIZE,
)
from utils import log_experiment, save_model

np.random.seed(SEED)


@log_experiment()
def train_single_model(model_name, writer=None):
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Get specific model
    model = MODELS[model_name]

    # Train
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average="weighted")  # заменили на F1-score

    # Log metrics and params
    if writer:
        writer.add_scalar("F1_score", f1, 0)
        writer.add_text("Params", str(model.get_params()), 0)

    # Save model
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{MODEL_DIR}/{model_name}.pkl")

    # Save metrics to JSON
    metrics_path = Path(f"{METRICS_DIR}/train_models") / f"{model_name}_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": model_name,
        "f1_score": f1,
        "timestamp": datetime.now().isoformat(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"{model_name}: F1-score = {f1:.4f}")
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name to train")
    parser.add_argument(
        "--exp_name", required=False, default=None, help="Experiment name for logging"
    )
    args = parser.parse_args()

    exp_name = (
        args.exp_name
        or f"train_single_{args.model}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    train_single_model(args.model, exp_name=exp_name, log_dir=LOG_DIR)
