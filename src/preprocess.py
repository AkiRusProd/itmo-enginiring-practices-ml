import random
from pathlib import Path

import numpy as np
import pandas as pd
from clearml import Task
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

from base_config import DATA_PATH, RAW_DATA_PATH, SEED

np.random.seed(SEED)
random.seed(SEED)


def preprocess(input_path=RAW_DATA_PATH, output_path=DATA_PATH):
    """Выполняет предобработку исходных данных и сохранить результат.

    Открывает CSV по `input_path`, заполняет пропуски, кодирует категориальные
    признаки и сохраняет обработанный датасет в `output_path`.

    Args:
        input_path: Путь к исходному CSV с сырыми данными.
        output_path: Путь для сохранения обработанных данных.
    """
    print("Starting preprocessing...")
    df = pd.read_csv(input_path)

    # Простая предобработка
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    df["Embarked"] = le.fit_transform(df["Embarked"])

    # Создаем папку, если ее нет
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    # Получаем текущую задачу (она может быть создана в main или пайплайном)
    task = Task.current_task()
    if task:
        task.add_tags(["preprocessing", "data"])
        # Загружаем обработанный файл как артефакт ClearML
        # Это позволит следующим шагам пайплайна скачать этот файл
        task.upload_artifact(name="processed_data", artifact_object=str(output_path))
        print("Artifact 'processed_data' uploaded to ClearML.")


if __name__ == "__main__":
    load_dotenv()
    # Инициализируем Task ТОЛЬКО если запускаем файл как скрипт
    # Если функцию запустит PipelineController, он сам создаст задачу
    task = Task.init(
        project_name="HW5_MLOps",
        task_name="Preprocess Data",
        task_type=Task.TaskTypes.data_processing,
        output_uri=True,  # Включает хранилище артефактов
    )

    preprocess()

    task.close()
