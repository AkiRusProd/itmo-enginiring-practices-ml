"""Простая система логирования и уведомлений о результатах."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from config import LOG_DIR, METRICS_FILE


class PipelineLogger:
    """Логирование выполнения пайплайна."""

    def __init__(self, name: str = "pipeline", log_dir: Path = LOG_DIR):
        """Инициализация логгера.

        Args:
            name: Имя пайплайна
            log_dir: Директория для логов
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Создаем логгер
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Файловый логгер
        log_file = (
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Консольный логгер
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file

    def info(self, message: str) -> None:
        """Логирование информации."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Логирование ошибки."""
        self.logger.error(message)

    def warning(self, message: str) -> None:
        """Логирование предупреждения."""
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        """Логирование отладки."""
        self.logger.debug(message)


def log_results(
    status: str,
    message: str,
    metrics: Optional[Dict[str, Any]] = None,
    log_dir: Path = LOG_DIR,
) -> None:
    """Логирование результатов выполнения.

    Args:
        status: Статус ('success' или 'failed')
        message: Сообщение результата
        metrics: Дополнительные метрики
        log_dir: Директория для логов
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_log_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d')}.log"

    log_entry = f"[{timestamp}] Status: {status}\nMessage: {message}\n"
    if metrics:
        log_entry += f"Metrics: {json.dumps(metrics, indent=2)}\n"
    log_entry += "=" * 80 + "\n"

    # В консоль
    print(log_entry)

    # В файл
    with open(result_log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


def print_results(metrics_file: Path = METRICS_FILE) -> None:
    """Вывести итоговые результаты.

    Args:
        metrics_file: Файл с метриками
    """
    metrics_file_path = Path(metrics_file)

    if not metrics_file_path.exists():
        print("❌ Метрики не найдены. Запустите пайплайн.")
        return

    with open(metrics_file_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ ТРЕНИРОВКИ")
    print("=" * 80)

    best_model = metrics.get("best_model", "Unknown")
    best_f1_score = metrics.get("best_f1_score", metrics.get("best_accuracy", 0))

    print(f"\n✓ Лучшая модель: {best_model}")
    print(f"✓ F1-score: {best_f1_score:.4f}")

    print("\n📈 Результаты всех моделей:")
    for model_name, score in metrics.get("models", {}).items():
        print(f"  - {model_name}: {score:.4f}")

    print("\n" + "=" * 80 + "\n")
