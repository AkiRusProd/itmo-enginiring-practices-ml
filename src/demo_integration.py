#!/usr/bin/env python3
"""Интегрированный пайплайн с мониторингом и логированием результатов."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from result_logger import print_results

from .config import METRICS_FILE


def main():
    """Главная функция для демонстрации логирования и уведомлений."""
    print("\n" + "=" * 80)
    print("ML TRAINING PIPELINE - ДЕМОНСТРАЦИЯ ИНТЕГРАЦИИ И МОНИТОРИНГА")
    print("=" * 80)

    print("\n1️⃣  ЗАПУСК ПАЙПЛАЙНА")
    print("   - Используется ConfigManager для загрузки конфигурации")
    print("   - Поддержка профилей: dev, test, prod")
    print("   - Все параметры валидируются через Pydantic")

    print("\n2️⃣  МОНИТОРИНГ ВЫПОЛНЕНИЯ")
    print("   - Логирование каждого этапа пайплайна")
    print("   - Сохранение логов в файлы (logs/)")
    print("   - Отслеживание времени выполнения")

    print("\n3️⃣  РЕЗУЛЬТАТЫ")
    print_results(METRICS_FILE)

    print("4️⃣  УВЕДОМЛЕНИЯ")
    print("   - Результаты сохраняются в logs/results_*.log")
    print("   - Метрики сохраняются в metrics/metrics.json")
    print("   - Лучшая модель сохраняется в models/best_model.pkl")

    print("\n✓ Пайплайн завершен успешно!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
