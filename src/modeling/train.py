import json
import os
import random

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

# фиксируем сиды для воспроизводимости
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

if __name__ == "__main__":
    # читаем параметры из params.yaml
    params = yaml.safe_load(open("params.yaml"))["train"]

    df = pd.read_csv("data/processed/processed.csv")

    # Простое предсказание: calories ~ protein
    X = df[["protein"]]
    y = df["calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=SEED
    )

    # Обучаем модель с параметрами из params.yaml
    model = SGDRegressor(
        max_iter=params["epochs"],
        learning_rate="constant",
        eta0=params["lr"],
        random_state=SEED,
    )
    model.fit(X_train, y_train)

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        import pickle  # nosec

        pickle.dump(model, f)

    # Сохраняем метрики
    score = model.score(X_test, y_test)
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump({"r2_score": score}, f)

    print(f"Training complete. R2 score: {score:.4f}")
