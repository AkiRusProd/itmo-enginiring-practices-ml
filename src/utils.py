"""
Модуль вспомогательных утилит.

Содержит функции для сохранения/загрузки моделей через pickle и
инструменты для логирования экспериментов в TensorBoard.
"""
import pickle  # nosec
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from tensorboardX import SummaryWriter

from base_config import TB_LOG_DIR


def save_model(model, path):
    """Сохраняет объект модели в файл.

    Сериализует объект модели с помощью `pickle` и сохраняет по указанному пути.
    Автоматически создает родительские директории, если они не существуют.

    Args:
        model (object): Объект модели (scikit-learn или подобный) для сохранения.
        path (str): Путь к файлу (включая имя файла и расширение .pkl).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Загружает модель из pickle-файла.

    Args:
        path (str): Путь к файлу модели.

    Returns:
        object: Десериализованный объект модели.
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec


@contextmanager
def experiment(exp_name, log_dir=TB_LOG_DIR):
    """Контекстный менеджер для работы с TensorBoard.

    Создает `SummaryWriter` при входе в контекст и автоматически закрывает его
    при выходе.

    Args:
        exp_name (str): Уникальное имя эксперимента (создает подпапку).
        log_dir (str): Базовая директория для логов.

    Yields:
        SummaryWriter: Объект для записи логов.
    """
    writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")
    try:
        yield writer
    finally:
        writer.close()


def log_experiment():
    """Декоратор для автоматического логирования экспериментов.

    Оборачивает функцию, требующую логирования. Извлекает из именованных
    аргументов вызова параметры `exp_name` и `log_dir`, инициализирует
    SummaryWriter и передает его в декорируемую функцию аргументом `writer`.

    Returns:
        Callable: Декорированная функция.
    """

    def decorator(func):
        """Декоратор-обёртка, создающий `wrapper` для функции."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper: извлекает параметры логирования и вызывает функцию.

            Ожидает в `kwargs` наличие ключа `exp_name`.
            """

            exp_name = kwargs.pop("exp_name")
            log_dir = kwargs.pop("log_dir", "logs")
            with experiment(exp_name, log_dir=log_dir) as writer:
                return func(*args, writer=writer, **kwargs)

        return wrapper

    return decorator
