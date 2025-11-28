import pickle  # nosec
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from tensorboardX import SummaryWriter

from base_config import TB_LOG_DIR


def save_model(model, path):
    """Сохраняет объект модели в файл через pickle.

    Создаёт родительские директории при необходимости и сериализует
    объект модели в указанный путь.
    Args:
        model: Объект модели (scikit-learn или подобный) для сохранения.
        path: Путь к файлу, куда будет записана модель.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    """Загружает сериализованную модель из файла.

    Args:
        path: Путь к файлу с сериализованной моделью.

    Returns:
        Десериализованный объект модели.
    """
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec


@contextmanager
def experiment(exp_name, log_dir=TB_LOG_DIR):
    """Контекстный менеджер для записи логов эксперимента в TensorBoard.

    Args:
        exp_name: Имя эксперимента (используется в пути логов).
        log_dir: Базовая директория для логов TensorBoard.

    Возвращает объект `SummaryWriter`.
    """
    writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")
    try:
        yield writer
    finally:
        writer.close()


def log_experiment():
    """Декоратор для автоматического создания и передачи `SummaryWriter`.

    Возвращает декоратор, который ожидает в вызове аргумент `exp_name` и
    необязательный `log_dir`. Создаёт контекст эксперимента и передаёт
    `writer` в целевую функцию как именованный аргумент.
    """

    def decorator(func):
        """Декоратор-обёртка, создающий `wrapper` для функции."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper: извлекает параметры логирования и вызывает функцию.

            Ожидает в `kwargs` ключ `exp_name`. Передаёт `writer` в вызов.
            """

            exp_name = kwargs.pop("exp_name")
            log_dir = kwargs.pop("log_dir", "logs")
            with experiment(exp_name, log_dir=log_dir) as writer:
                return func(*args, writer=writer, **kwargs)

        return wrapper

    return decorator
