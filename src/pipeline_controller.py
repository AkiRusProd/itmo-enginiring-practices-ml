"""
Модуль управления ML-пайплайном через ClearML.

Определяет структуру DAG (Directed Acyclic Graph) пайплайна, включая этапы
предобработки, обучения множества моделей, выбора лучшей и финальной оценки.
Использует ClearML PipelineController для оркестрации выполнения функций.
"""
from datetime import datetime

from clearml import PipelineController
from dotenv import load_dotenv

from base_config import MODELS, TB_LOG_DIR
from eval import evaluate_model

# Импортируем ваши функции напрямую
from preprocess import preprocess
from select_best import select_best_model
from train_single import train_single_model


def run_pipeline() -> None:
    """
    Настраивает и запускает ML-пайплайн в ClearML.

    Пайплайн состоит из следующих этапов:
    1. **preprocess**: Подготовка данных (функция `src.preprocess.preprocess`).
    2. **train_...**: Параллельные шаги обучения для каждой модели из конфига.
       Вызывает функцию `train_single_model` с соответствующими параметрами.
    3. **select_best**: Сбор метрик и выбор чемпиона (функция `src.select_best.select_best_model`).
    4. **evaluate**: Оценка чемпиона на тестовой выборке (функция `src.eval.evaluate_model`).

    Функция запускает пайплайн локально (`start_locally`), что позволяет
    выполнять шаги как обычный Python-код, при этом регистрируя структуру
    эксперимента в ClearML.
    """
    PROJECT_NAME = "HW5_MLOps"

    load_dotenv()

    pipe = PipelineController(
        name="Function Based Pipeline",
        project=PROJECT_NAME,
        version="0.0.3",
        add_pipeline_tags=True,
    )

    pipe.set_default_execution_queue("default")

    # Шаг 1: Preprocess
    # Используем add_function_step для запуска python-функции
    pipe.add_function_step(
        name="preprocess",
        function=preprocess,
        function_kwargs={},  # Используем значения по умолчанию
    )

    # Шаг 2: Обучение моделей
    train_steps = []
    for model_name in MODELS.keys():
        step_name = f"train_{model_name}"
        train_steps.append(step_name)

        exp_name = (
            f"train_single_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )

        # Запускаем функцию train_single_model для каждой модели
        pipe.add_function_step(
            name=step_name,
            parents=["preprocess"],
            function=train_single_model,
            function_kwargs={
                "model_name": model_name,
                "exp_name": exp_name,
                "log_dir": TB_LOG_DIR,
            },
            # Мы можем захватить return value функции (f1), хотя в артефактах оно надежнее
            function_return=["f1_score"],
        )

    # Шаг 3: Выбор лучшей
    pipe.add_function_step(
        name="select_best",
        parents=train_steps,
        function=select_best_model,
    )

    # Шаг 4: Оценка
    pipe.add_function_step(
        name="evaluate",
        parents=["select_best"],
        function=evaluate_model,
    )

    print("🚀 Запуск пайплайна на локальной машине...")

    # run_pipeline_steps_locally=True выполнит код функций прямо здесь
    # Это создаст структуру пайплайна в ClearML и выполнит код
    pipe.start_locally(run_pipeline_steps_locally=True)


if __name__ == "__main__":
    run_pipeline()
