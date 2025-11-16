import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import DATA_PATH, RAW_DATA_PATH, SEED

np.random.seed(SEED)
random.seed(SEED)


def preprocess(input_path=RAW_DATA_PATH, output_path=DATA_PATH):
    df = pd.read_csv(input_path)

    # Простая предобработка
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    # Создаем папку, если ее нет
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess()
