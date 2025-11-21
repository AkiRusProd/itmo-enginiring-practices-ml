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
    ├── metrics/
    │   └── metrics.json
    ├── models/
    │   └── .gitkeep
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
    │   ├── config.py
    │   ├── preprocess.py
    │   ├── train.py
    │   └── utils.py
    ├── tests/
    │   └── test_data.py
    └── .dvc/
        └── config
```


# Отчет по автоматизации пайплайнов (ДЗ4)

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
✓ Интеграция завершена:
- ConfigManager + DVC pipeline полностью интегрированы
- train.py использует ConfigManager для загрузки конфигураций
- DVC пайплайн (dvc.yaml) использует python3 команды
- Все профили (dev/test/prod) работают через переменную `CONFIG_PROFILE`

```bash
# Запуск с разными профилями
CONFIG_PROFILE=dev dvc repro      # Быстрая разработка
CONFIG_PROFILE=test dvc repro     # Тестирование
CONFIG_PROFILE=prod dvc repro     # Продакшн
```

### 3.2 Создать систему мониторинга выполнения
✓ Реализована в `src/result_logger.py`:
- **PipelineLogger** класс для логирования каждого этапа
- Логи сохраняются в `logs/train_YYYYMMDD_HHMMSS.log`
- Вывод одновременно в консоль и в файл
- Информация о начале/окончании каждого этапа

### 3.3 Настроить уведомления о результатах
✓ Система уведомлений реализована:
- Результаты выводятся в консоль
- Результаты сохраняются в `logs/results_YYYYMMDD.log`
- JSON метрики в `metrics/metrics.json`
- Красивый вывод лучшей модели и всех результатов

```bash
# Вывод результатов
python src/demo_integration.py
```

### 3.4 Протестировать воспроизводимость
✓ Тесты в `tests/test_reproducibility.py`:

```bash
python tests/test_reproducibility.py
# Результат: ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✓
```

**Гарантии воспроизводимости:**
- SEED = 42 для всех операций
- DVC кеш обеспечивает идентичные результаты
- train_test_split с random_state=SEED
- Все модели инициализируются одинаково

## 4. Отчет о проделанной работе (1 балл)





Прежде всего добавим tensorboard к стеку dvc. Также я решил взять датасет побольше - [Titanic Dataset](https://www.kaggle.com/competitions/titanic/data). 

## 1. Настройка выбранного инструмента (Tensorboard)
- Установка и инициализация
    ```
    poetry add tensorboard
    poetry add tensorboardX
    ```
    Логировать будем в logs
- Закинем titanic датасет
    ```
    ├── data/
    │   └── raw/
    |       └── dataset.csv     
    ```


- Настройка remote storage (Local)
    ```
    mkdir localstore
    mkdir dvc_storage
    dvc remote add -d localstore dvc_storage
    git add .dvc/config
    git commit -m "Set DVC remote (local)"
    ```
    ![alt text](images/dvc_set_store.png)


### 1.1. Версионирование моделей и данных в DVC
- Конфиг со стейджами для версионирования `dvc.yaml` в DVC:
    ```yaml
        stages:
        preprocess:
            cmd: python src/preprocess.py
            deps:
            - data/raw/dataset.csv
            - src/preprocess.py
            outs:
            - data/processed
        train_model:
            cmd: python src/train.py
            deps:
            - data/processed/processed.csv
            - src/train.py
            - src/config.py
            - src/utils.py
            - src/preprocess.py
            params:
            - train.test_size
            - train.seed
            outs:
            - models/LogisticRegression.pkl
            - models/RandomForest.pkl
            - models/GradientBoosting.pkl
            - models/DecisionTree.pkl
            - models/KNeighbors.pkl
            - models/SVC_linear.pkl
            - models/SVC_rbf.pkl
            - models/LinearSVC.pkl
            - models/GaussianNB.pkl
            - models/MultinomialNB.pkl
            - models/BernoulliNB.pkl
            - models/AdaBoost.pkl
            - models/ExtraTrees.pkl
            - models/Bagging.pkl
            - models/RidgeClassifier.pkl

            metrics:
            - metrics/metrics.json:
                cache: false


    ```
    Выполнение:
    ```
    dvc repro
    ```
    ![alt text](images/dvc_repro.png)

- Сравнение версий модели.
  ```
    dvc exp run -S train.test_size=0.3
    dvc metrics diff
    dvc exp show
  ```
  ![alt text](images/dvc_run_exps.png)

## 2. Проведение экспериментов 
- Запуск Тензорборда осуществляется командой:
    ```
    tensorboard --logdir logs
    ```
    ![alt text](images/tb_run.png)
- Если перейти по ссылке, то можно увидеть
    ![alt text](images/tb.png)
- Параметры также логируются
    ![alt text](images/tb_params.png)
- Проведено 15 экспериментов с разными алгоритмами
- Логирование метрик осуществляется через tensorboard, параметры и артефакты через dvc.
- Сравнивать, фильтровать и искать информацию можно средствами tensorboard.

## 3.	Интеграция с кодом:
Инструмент логирования интегрирован в код через `src/utils.py`, в виде декораторов и контекстных менеджеров. Испольузется непосредственно в `train.py`
Пример встраивания:
```python
@log_experiment()
def train_model(name, model, X_train, y_train, X_val, y_val, writer=None):
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    writer.add_scalar("Accuracy", acc, 0)
    writer.add_text("Params", str(model.get_params()), 0)

    save_model(model, f"{MODEL_DIR}/{name}.pkl")
    return acc
```

