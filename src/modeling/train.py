import json
import os
import pickle

import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]
    df = pd.read_csv("data/processed/processed.csv")

    # Простое предсказание: calories ~ protein
    X = df[["protein"]]
    y = df["calories"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=42
    )

    # Обучаем модель
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Сохраняем модель
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Сохраняем метрики
    score = model.score(X_test, y_test)
    with open("metrics.json", "w") as f:
        json.dump({"r2_score": score}, f)
