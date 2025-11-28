# HW5 MLOps Pipeline: Titanic Classification

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Проектная работа в рамках курса MLOps (ITMO).**
Полноценный пайплайн машинного обучения для классификации датасета Titanic с использованием DVC, ClearML, Pydantic и автоматической генерацией отчетности.

---

## 📋 Отчет по ДЗ 6: Документация и отчеты

В этом релизе реализованы все требования домашнего задания №6.

### 1. ✅ Техническая документация (2 балла)
*   **Инструмент:** Использован **MkDocs** с темой Material.
*   **API Reference:** Настроена автоматическая генерация документации из docstrings (`src/*.py`) с помощью плагина `mkdocstrings`.
*   **Руководства:** Написаны `Quick Start`, `Deployment Guide` и описание архитектуры.
*   **Конфигурация:** См. файл [`mkdocs.yml`](./mkdocs.yml).

### 2. ✅ Публикация в Git Pages (3 балла)
*   **CI/CD:** Настроен GitHub Actions workflow [`gh-pages.yml`](./.github/workflows/gh-pages.yml).
*   **Автоматизация:** При пуше в ветку `main` документация собирается и деплоится в ветку `gh-pages`.
*   **Результат:** Сайт доступен по клику на бейдж выше.

### 3. ✅ Отчеты об экспериментах (2 балла)
*   **Генерация:** Реализован скрипт [`src/generate_report.py`](./src/generate_report.py).
*   **Визуализация:**
    *   📊 Сравнительная таблица всех обученных моделей (Leaderboard).
    *   📈 Вертикальный Bar-chart сравнения метрик (F1 Score).
    *   🟦 Матрица ошибок (Confusion Matrix) для лучшей модели.
*   **Автоматизация:** Отчет генерируется автоматически при запуске пайплайна и встраивается в документацию (`docs/experiments.md`).

### 4. ✅ Воспроизводимость (1 балл)
*   **Инструкции:** Полное описание установки и запуска приведено ниже.
*   **DVC:** Весь пайплайн (включая генерацию отчетов) запускается одной командой `dvc repro`.
*   **Environment:** Зависимости зафиксированы в `requirements.txt`.

---

## 🚀 Быстрый старт (Reproducibility)

Для воспроизведения результатов экспериментов выполните следующие шаги:

### 1. Клонирование и установка

```bash
git clone https://github.com/AkiRusProd/itmo-enginiring-practices-ml.git
cd itmo-enginiring-practices-ml

python3 -m venv .venv
source .venv/bin/activate  # Для Windows: .venv\Scripts\activate

poetry install
```

### 2. Настройка окружения

Создайте файл `.env` в корне проекта (опционально для ClearML и Telegram уведомлений):

```env
CLEARML_API_ACCESS_KEY=# Get this from http://localhost:8080/settings/workspace-configuration
CLEARML_API_SECRET_KEY=# Get this from http://localhost:8080/settings/workspace-configuration
CLEARML_HOST_IP=localhost
CLEARML_API_HOST=http://localhost:8008
CLEARML_WEB_HOST=http://localhost:8080
CLEARML_FILES_HOST=http://localhost:8081
CLEARML_AGENT_GIT_USER= # Optional: Git user for private repositories
CLEARML_AGENT_GIT_PASS= # Optional: Git password or token for private repositories
CLEARML_AGENT_ACCESS_KEY= # Optional: Agent access key
CLEARML_AGENT_SECRET_KEY= # Optional: Agent secret key
TG_BOT_TOKEN=#YOUR_BOT_TOKEN_HERE
TG_CHAT_ID=#YOUR_CHAT_ID_HERE
```

### 3. Запуск пайплайна

Используйте DVC для запуска всего цикла (препроцессинг -> обучение -> выбор лучшей -> оценка -> отчет):

```bash
dvc repro
```

После успешного выполнения:
1.  Лучшая модель будет сохранена в `models/best_model.pkl`.
2.  Метрики появятся в папке `metrics/`.
3.  **Отчет с графиками** будет сгенерирован в `docs/experiments.md`.

### 4. Просмотр документации локально

Чтобы увидеть сгенерированный отчет и документацию без деплоя:

```bash
mkdocs serve
```
Перейдите по адресу `http://127.0.0.1:8000`.

---

## 📂 Структура проекта

```text
.
├── .github/workflows/
│   └── gh-pages.yml       # CI/CD для документации
├── config/                # Конфигурационные файлы (yaml)
├── data/                  # Данные (под управлением DVC)
├── docs/                  # Исходники документации (Markdown)
│   ├── assets/images/     # Сгенерированные графики
│   └── ...
├── metrics/               # JSON метрики экспериментов
├── models/                # Сохраненные модели
├── src/                   # Исходный код
│   ├── generate_report.py # Скрипт генерации отчетов
│   ├── train_single.py    # Обучение моделей
│   ├── select_best.py     # Выбор лучшей модели
│   └── ...
├── dvc.yaml               # Описание пайплайна DVC
├── mkdocs.yml             # Конфигурация MkDocs
├── params.yaml            # Параметры экспериментов
└── requirements.txt       # Зависимости
```

---

## 📊 Пример генерируемых отчетов

Скрипт `src/generate_report.py` автоматически анализирует результаты обучения и строит графики, которые попадают в итоговую документацию:

| Сравнение моделей | Матрица ошибок |
|-------------------|----------------|
| ![Comparison](docs/assets/images/confusion_matrix.png) | ![Confusion Matrix](docs/assets/images/confusion_matrix.png) |

---

## 🛠 Разработка

*   **Linting:** `black src/`
*   **Adding Dependencies:** Добавьте пакет в `requirements.txt`
*   **Adding Docs:** Создайте новый `.md` файл в `docs/` и добавьте его в `nav` в `mkdocs.yml`.

## License

MIT

