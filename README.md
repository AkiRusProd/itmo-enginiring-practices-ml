# itmo-enginiring-practices-ml

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Template for course enginiring-practices-ml

## Project Organization

```
Directory structure:
└── akirusprod-itmo-enginiring-practices-ml/
    ├── README.md
    ├── Dockerfile
    ├── dvc.lock
    ├── dvc.yaml
    ├── LICENSE
    ├── Makefile
    ├── params.yaml
    ├── pyproject.toml
    ├── requirements.txt
    ├── .dvcignore
    ├── .pre-commit-config.yaml
    ├── config/
    │   ├── params.dev.yaml
    │   ├── params.prod.yaml
    │   └── params.test.yaml
    ├── data/
    │   └── raw/
    │       └── dataset.csv.dvc
    ├── docs/
    │   ├── README.md
    │   ├── mkdocs.yml
    │   ├── .gitkeep
    │   └── docs/
    │       ├── getting-started.md
    │       └── index.md
    ├── examples/
    │   └── config_composition_examples.py
    ├── notebooks/
    │   └── .gitkeep
    ├── references/
    │   └── .gitkeep
    ├── reports/
    │   ├── .gitkeep
    │   └── figures/
    │       └── .gitkeep
    ├── src/
    │   ├── __init__.py
    │   ├── base_config.py
    │   ├── config_manager.py
    │   ├── eval.py
    │   ├── preprocess.py
    │   ├── result_logger.py
    │   ├── schemas.py
    │   ├── select_best.py
    │   ├── train.py
    │   ├── train_single.py
    │   └── utils.py
    ├── tests/
    │   └── test_data.py
    └── .dvc/
        └── config
```


# Отчет по ClearML для MLOps (ДЗ5)

## 1.	Настройка ClearML (3 балла):
### 1.1 Установить и настроить ClearML Server
Установка и настройка:

1. Установку и запуск выполним через docker-compose:
    https://github.com/clearml/clearml-server/blob/master/docker/docker-compose.yml

2. Выдаем права на директорию для работы elastic-search.
    ```
    sudo chown -R 1000:1000 /opt/clearml/data/elastic_7
    sudo chmod -R 775 /opt/clearml/data/elastic_7
    ```

3. Запускаем контейнеры.
    ```
    docker compose -f docker-compose.clearml.yml up --build -d
    ```
4. Переходим по `http://localhost:8080`. После этого сервис должен запуститься:
    ![alt text](images/clearml_start.png)

5. Далее "Settings" -> "Workspace" -> "Create new credentials" (или просто http://localhost:8080/settings/workspace-configuration).

6. Не забываем поставить clearml в poetry: `poetry add clearml`

7. Далее создаем файл `.env` в корне проекта (Пример: [.env.example](.env.example)).


8. (Необязательный шаг, если не хотите использовать .env) Далее в cli с проектом пишем clearml-init. В директории юзера должен появиться `clearml.conf`
/home/rustam/clearml.conf

9. Перезапускаем контейнеры:
    ```
    docker compose -f docker-compose.clearml.yml down
    docker compose -f docker-compose.clearml.yml up -d
    ```

10. Дополнительно можно проверить, что все работает по адресу http://localhost:8008/debug.ping. Должны получить `result_msg	"OK"`.



### 1.2 Настроить базу данных и хранилище
Развернут локальный сервер. В качестве БД используется MongoDB (контейнер clearml-mongo), в качестве хранилища артефактов настроен локальный файловый сервер (clearml-fileserver), персистентность данных обеспечена через Docker Volumes.


### 1.3	Создать проект и эксперименты


![alt text](images/clearml_run.png)

### 1.4	Настроить аутентификацию
Настроили в 1.1.1

## 2.	Трекинг экспериментов (3 балла):
o	Настроить автоматическое логирование
o	Создать систему сравнения экспериментов
o	Настроить логирование метрик и параметров
o	Создать дашборды для анализа
## 3.	Управление моделями (3 балла):
o	Настроить регистрацию и версионирование моделей
o	Создать систему метаданных для моделей
o	Настроить автоматическое создание версий
o	Создать систему сравнения моделей
## 4.	Пайплайны (2 балла):
o	Создать ClearML пайплайны для ML workflow
o	Настроить автоматический запуск пайплайнов
o	Создать систему мониторинга выполнения
o	Настроить уведомления
## 5.	Отчет о проделанной работе (1 балл):
o	Создать отчет в формате Markdown
o	Описать настройку каждого инструмента
o	Добавить скриншоты результатов
o	Сохранить отчет в Git репозитории


https://github.com/clearml/clearml-server/blob/master/docker/docker-compose.yml

## 1.	Настройка выбранного инструмента оркестрации (4 балла):
### 1.1 Установить и настроить выбранный инструмент   
DVC уже установлен и настроен (см. ДЗ 2 и 3). 

### 1.2 Создать workflow для ML пайплайна
Был расширен workflow в dvc.py до 3-х этапов: preprocess, train_models, evaluate. Также была реализована более гибкая логика подбора параметров из params.py

### 1.3	Настроить зависимости между этапами
Зависимости данных также настроены в dvc.py. Рассмотреть граф зависимостей этапов пайплана можно через `dvc dag`:   
![alt text](images/dvc_dag.png)

### 1.4	Реализовать кэширование и параллельное выполнение
DVC автоматически кэширует результаты каждого этапа, если:
- этап имеет deps: (зависимости)
- этап имеет outs: (выходы)     

Таким образом, в dvc-пайплайне уже реализовано кэширование:
![alt text](images/dvc_cache.png)

DVC предоставляет параллельное выполнение стадий пайплаина через `dvc exp run --jobs`. DVC анализирует граф зависимостей. Стадии, которые не зависят друг от друга, выполняются параллельно. Максимальное количество параллельных задач задается через `--jobs`. Однако от себя добавлю, что кажется, что настоящей параллельности в dvc нет, потому что даже независимые пайплайны пишут в один dvc.lock, из-за чего делают они это последовательно.

## 2. Настройка выбранного инструмента конфигураций (3 балла):
### 1.1 Настроить выбранный инструмент для управления конфигурациями
Реализована система ConfigManager на основе Pydantic с поддержкой профилей dev/test/prod.

### 1.2 Создать конфигурации для разных алгоритмов
Созданы 3 файла конфигураций (params.dev.yaml, params.test.yaml, params.prod.yaml) с оптимизированными параметрами для разных мль алгоритмов.

### 1.3 Настроить валидацию конфигураций
Все параметры конфигурации проходят типизацию и кастомную валидацию через Pydantic схемы с проверкой диапазонов и форматов данных.
### 1.4 Создать систему композиции конфигураций
Реализована многоуровневая иерархия конфигураций (переменные окружения → профили → базовый конфиг) с методами слияния и переопределения параметров через merge() и override_model_params().


## 3. Интеграция и тестирование (2 балла)

### 3.1 Интегрировать выбранные инструменты
Интеграция завершена:
- ConfigManager + DVC pipeline полностью интегрированы
- train.py использует ConfigManager для загрузки конфигураций
- DVC пайплайн (dvc.yaml) использует python3 команды
- Все профили (dev/test/prod) работают через переменную `CONFIG_PROFILE` в dvc.yaml.

### 3.2 Создать систему мониторинга выполнения
Реализована в `src/result_logger.py`:
- **PipelineLogger** класс для логирования каждого этапа
- Логи сохраняются в `logs/train_YYYYMMDD_HHMMSS.log`
- Вывод одновременно в консоль и в файл
- Информация о начале/окончании каждого этапа

### 3.3 Настроить уведомления о результатах
Система уведомлений реализована:
- Результаты выводятся в консоль
- Результаты сохраняются в `logs/results_YYYYMMDD.log`
- JSON метрики в `metrics/metrics.json`
- Красивый вывод лучшей модели и всех результатов

### 3.4 Протестировать воспроизводимость
Воспроизводимость поддержана:
![alt text](images/test_repro.png)

## 4. Отчет о проделанной работе (1 балл)
### 1.1	Создать отчет в формате Markdown
Выполнено
### 1.2	Описать настройку выбранных инструментов
Выполнено
### 1.3	Добавить скриншоты результатов
Выполнено
### 1.4	Сохранить отчет в Git репозитории
Выполнено

