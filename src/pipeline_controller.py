from datetime import datetime

from clearml import PipelineController
from dotenv import load_dotenv

from base_config import MODELS, TB_LOG_DIR
from eval import evaluate_model

# Импортируем ваши функции напрямую
from preprocess import preprocess
from select_best import select_best_model
from train_single import train_single_model


def run_pipeline():
    """
    Запускает ML-пайплайн в ClearML, основанный на выполнении Python-функций напрямую.
    Пайплайн состоит из четырёх этапов:

    1. preprocess — предобработка данных.
       Выполняет подготовку датасета, очистку данных и извлечение необходимых признаков.

    2. train_* — обучение нескольких моделей.
       Для каждой модели из списка MODELS создаётся отдельный шаг.
       Каждый шаг вызывает train_single_model(), передавая:
           - model_name — имя модели
           - exp_name — уникальное имя эксперимента с timestamp
           - log_dir — путь для логов TensorBoard
       Возвращаемое значение: f1_score (метрика качества модели)

    3. select_best — выбор лучшей модели.
       Использует результаты обучения (f1_score) и определяет лучшую модель.

    4. evaluate — финальная оценка.
       Выполняет оценку выбранной модели на hold-out тестовых данных.

    По завершении пайплайн запускается локально с помощью:
        pipe.start_locally(run_pipeline_steps_locally=True)

    Это создаёт структуру пайплайна в ClearML и запускает шаги последовательно
    на локальной машине, сохраняя результаты в ClearML-систему отслеживания.
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
