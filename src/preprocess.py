import os

import pandas as pd

if __name__ == "__main__":
    """A mock preprocessing script that reads raw data and writes processed data."""
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv("data/raw/dataset.csv")
    df[["name", "calories", "protein"]].to_csv(
        "data/processed/processed.csv", index=False
    )
