import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
random.seed(SEED)


def preprocess(
    input_path="data/raw/dataset.csv", output_path="data/preprocessed/preprocessed.csv"
):
    df = pd.read_csv(input_path)

    # Простая предобработка
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna("S", inplace=True)

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocess()
