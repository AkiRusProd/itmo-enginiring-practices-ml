import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from base_config import LOG_DIR, METRICS_FILE
from tg_bot_notifier import TelegramBotNotifier

tg_notifier = TelegramBotNotifier()


class PipelineLogger:
    """Логирование выполнения пайплайна."""

    def __init__(self, name: str = "pipeline", log_dir: Path = LOG_DIR):
        """Инициализация логгера."""
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        log_file = (
            self.log_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # Файловый хендлер
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.log_file = log_file

    def info(self, message: str) -> None:
        """Записывает информационное сообщение в лог."""
        self.logger.info(message)

    def error(self, message: str) -> None:
        """Логирует ошибку и шлет алерт в ТГ."""
        self.logger.error(message)
        tg_notifier.send_message(f"🚨 <b>CRITICAL ERROR ({self.name})</b>\n\n{message}")

    def warning(self, message: str) -> None:
        """Записывает предупреждение в лог."""
        self.logger.warning(message)

    def debug(self, message: str) -> None:
        """Записывает отладочное сообщение в лог."""
        self.logger.debug(message)


def log_results(
    status: str,
    message: str,
    metrics: Optional[Dict[str, Any]] = None,
    log_dir: Path = LOG_DIR,
) -> None:
    """Логирование результатов и отправка в Telegram."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_log_file = log_dir / f"results_{datetime.now().strftime('%Y%m%d')}.log"

    # 1. Лог в файл/консоль
    log_entry = f"[{timestamp}] Status: {status}\nMessage: {message}\n"
    if metrics:
        log_entry += f"Metrics: {json.dumps(metrics, indent=2)}\n"
    log_entry += "=" * 80 + "\n"

    print(log_entry)
    with open(result_log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # 2. Уведомление в Telegram
    icon = "✅" if status.lower() == "success" else "❌"

    tg_msg = (
        f"{icon} <b>Experiment Log</b>\n"
        f"📅 <i>{timestamp}</i>\n"
        f"Status: <b>{status.upper()}</b>\n\n"
        f"📝 {message}\n"
    )

    if metrics:
        tg_msg += "\n📊 <b>Metrics:</b>\n"
        for key, value in metrics.items():
            display_value = f"{value:.4f}" if isinstance(value, float) else value
            tg_msg += f"• {key}: <code>{display_value}</code>\n"

    tg_notifier.send_message(tg_msg)


def print_results(metrics_file: Path = METRICS_FILE) -> None:
    """Вывести результаты и отправить сводку в Telegram."""
    metrics_file_path = Path(metrics_file)

    if not metrics_file_path.exists():
        msg = "❌ Метрики не найдены. Запустите пайплайн."
        print(msg)
        tg_notifier.send_message(msg)
        return

    with open(metrics_file_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    # 1. Консольный вывод
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

    # 2. Telegram сводка
    tg_summary = (
        "🏆 <b>FINAL RESULTS SUMMARY</b>\n"
        "〰️〰️〰️〰️〰️〰️〰️〰️\n\n"
        f"🥇 <b>Best Model:</b> {best_model}\n"
        f"⭐️ <b>Best Score:</b> <code>{best_f1_score:.4f}</code>\n\n"
        "📈 <b>Leaderboard:</b>\n"
    )

    # Сортировка по убыванию метрики
    models_data = metrics.get("models", {})
    sorted_models = sorted(models_data.items(), key=lambda x: x[1], reverse=True)

    for i, (model_name, score) in enumerate(sorted_models, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "▫️"
        tg_summary += f"{medal} {model_name}: <code>{score:.4f}</code>\n"

    tg_notifier.send_message(tg_summary)
