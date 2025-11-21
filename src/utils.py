import pickle  # nosec
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from tensorboardX import SummaryWriter

from config import TB_LOG_DIR


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)  # nosec


@contextmanager
def experiment(exp_name, log_dir=TB_LOG_DIR):
    writer = SummaryWriter(log_dir=f"{log_dir}/{exp_name}")
    try:
        yield writer
    finally:
        writer.close()


def log_experiment():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exp_name = kwargs.pop("exp_name")
            log_dir = kwargs.pop("log_dir", "logs")
            with experiment(exp_name, log_dir=log_dir) as writer:
                return func(*args, writer=writer, **kwargs)

        return wrapper

    return decorator
