"""
Модуль генерации отчетов об экспериментах.

Отвечает за сбор метрик всех обученных моделей, создание визуализаций (графики
сравнения, матрица ошибок) и генерацию итогового Markdown-файла с отчетом.
"""
import glob
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Настройки путей
METRICS_DIR = Path("metrics")
DOCS_DIR = Path("docs")
IMAGES_DIR = DOCS_DIR / "assets" / "images"
REPORT_PATH = DOCS_DIR / "experiments.md"

# Создаем папку для картинок, если нет
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics():
    """Считывает метрики всех моделей из JSON-файлов.

    Сканирует директорию `metrics/train_models/`, загружает данные из каждого
    файла и собирает их в единый DataFrame.

    Returns:
        pd.DataFrame: Таблица с полями Model, F1 Score, Timestamp, Profile.
    """
    files = glob.glob(str(METRICS_DIR / "train_models" / "*_metrics.json"))
    data = []

    for file in files:
        with open(file, "r") as f:
            m = json.load(f)
            data.append(
                {
                    "Model": m.get("model"),
                    "F1 Score": m.get("f1_score"),
                    "Timestamp": m.get("timestamp"),
                    "Profile": m.get("profile"),
                }
            )

    return pd.DataFrame(data)


def load_best_metrics():
    """Загружает детальные метрики лучшей модели.

    Считывает файл `metrics/best_model_advanced_metrics.json`, созданный
    на этапе оценки.

    Returns:
        dict: Словарь с метриками (accuracy, precision, recall, f1, confusion_matrix)
        или None, если файл не найден.
    """
    path = METRICS_DIR / "best_model_advanced_metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def plot_model_comparison(df):
    """Строит и сохраняет график сравнения моделей по F1 Score.

    Args:
        df (pd.DataFrame): Датафрейм с метриками моделей.

    Returns:
        str: Относительный путь к сохраненному изображению или None, если данных нет.
    """
    if df.empty:
        return None

    # Увеличим размер фигуры, чтобы влезли подписи снизу
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    # Сортировка по убыванию
    df_sorted = df.sort_values(by="F1 Score", ascending=False)

    # ИЗМЕНЕНИЕ: x="Model", y="F1 Score" для вертикальных столбцов
    ax = sns.barplot(
        x="Model",
        y="F1 Score",
        data=df_sorted,
        palette="viridis",
        hue="Model",
        legend=False,
    )

    plt.title("Model Comparison (F1 Score)")
    plt.xlabel("Model")
    plt.ylabel("F1 Score")

    # Устанавливаем лимит по Y чуть больше 1.0, чтобы влезли цифры над столбцами
    plt.ylim(0, 1.1)

    # ИЗМЕНЕНИЕ: Поворачиваем подписи моделей на 45 градусов, чтобы не слипались
    plt.xticks(rotation=45, ha="right")

    # Добавляем значения над столбцами
    for i in ax.containers:
        ax.bar_label(i, fmt="%.3f", padding=3)

    plt.tight_layout()

    output_path = IMAGES_DIR / "model_comparison.png"
    plt.savefig(output_path)
    plt.close()

    return "assets/images/model_comparison.png"


def plot_confusion_matrix(matrix):
    """Визуализирует и сохраняет матрицу ошибок.

    Args:
        matrix (list): Список списков (2x2), представляющий матрицу ошибок.

    Returns:
        str: Относительный путь к сохраненному изображению.
    """
    if not matrix:
        return None

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"],
    )

    plt.title("Confusion Matrix (Best Model)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    output_path = IMAGES_DIR / "confusion_matrix.png"
    plt.savefig(output_path)
    plt.close()

    return "assets/images/confusion_matrix.png"


def generate_markdown(df, best_metrics, comparison_img, cm_img):
    """Формирует текст отчета в формате Markdown.

    Собирает воедино таблицы с метриками, ссылки на изображения и текстовые описания.

    Args:
        df (pd.DataFrame): Таблица сравнения моделей.
        best_metrics (dict): Метрики лучшей модели.
        comparison_img (str): Путь к графику сравнения.
        cm_img (str): Путь к изображению матрицы ошибок.

    Returns:
        str: Полный текст отчета в формате Markdown.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    md = "# Experiment Report\n\n"
    md += f"*Generated on: {timestamp}*\n\n"

    # 1. Секция сравнения моделей
    md += "## 1. Model Leaderboard\n\n"

    if comparison_img:
        md += f"![Model Comparison]({comparison_img})\n\n"

    if not df.empty:
        df_sorted = df.sort_values(by="F1 Score", ascending=False)
        md += df_sorted.to_markdown(index=False)
    else:
        md += "No training metrics found.\n"

    # 2. Секция лучшей модели
    md += "\n\n## 2. Best Model Detailed Evaluation\n\n"

    if best_metrics:
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        md += f"| **F1 Score** | {best_metrics.get('f1', 0):.4f} |\n"
        md += f"| Accuracy | {best_metrics.get('accuracy', 0):.4f} |\n"
        md += f"| Precision | {best_metrics.get('precision', 0):.4f} |\n"
        md += f"| Recall | {best_metrics.get('recall', 0):.4f} |\n\n"

        if cm_img:
            md += "### Confusion Matrix\n\n"
            md += f"![Confusion Matrix]({cm_img})\n"
    else:
        md += "No evaluation metrics found. Run `dvc repro evaluate` first.\n"

    return md


def main():
    """Основная функция запуска генерации отчета."""
    print("Generating report with visualizations...")
    df = load_metrics()
    best_metrics = load_best_metrics()

    comp_img = plot_model_comparison(df)
    cm_img = None
    if best_metrics:
        cm_img = plot_confusion_matrix(best_metrics.get("confusion_matrix"))

    report_content = generate_markdown(df, best_metrics, comp_img, cm_img)

    REPORT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"Report saved to {REPORT_PATH}")
    print(f"Images saved to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
