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
    │   ├── preprocess.py
    │   └── modeling/
    │       ├── __init__.py
    │       └── train.py
    ├── tests/
    │   └── test_data.py
    └── .dvc/
        └── config
```


# Отчет по версионированию данных и моделей (ДЗ2)


## 1. Настройка версионирования данных (DVC)
- Установка и инициализация
    ```
    poetry add dvc
    dvc init
    ```
    ![alt text](images/dvc_init.png)
- Настройка remote storage (Local)
    ```
    mkdir localstore
    mkdir dvc_storage
    dvc remote add -d localstore dvc_storage
    git add .dvc/config
    git commit -m "Set DVC remote (local)"
    ```
    ![alt text](images/dvc_set_store.png)

- Версионирование данных
    
    В качестве датасета был взят https://www.kaggle.com/datasets/crawford/80-cereals и переименован в `dataset.csv`
    ```
    dvc add data/raw/dataset.csv
    git add data/raw/dataset.csv.dvc .gitignore
    git commit -m "Versioned dataset"
    ```
    ![alt text](images/dvc_dataset_add.png)
- Автоматическое создание версий
    ```
    dvc stage add -n preprocess \
    -d src/preprocess.py \
    -d data/raw/dataset.csv \
    -o data/processed \
    python src/preprocess.py
    ```
    ![alt text](images/dvc_add_ver.png)
    ```
    dvc dag
    dvc repro
    ```
    ![alt text](images/dvc_dag_repro.png)
## 2. Версионирование моделей в DVC
- Написание `src/modeling/train.py`, `params.yaml` и создание DVC stage:
    ```
    dvc stage add -n train_model \
        -d src/modeling/train.py \
        -d data/processed/processed.csv \
        -p train.lr,train.epochs,train.test_size \
        -o models/model.pkl \
        -M metrics.json \
        python src/modeling/train.py
    ```
    ![alt text](images/dvc_train_model.png)
    ```
    dvc dag
    dvc repro
    ```
    ![alt text](images/dvc_dag_repro_with_model.png)

- Сравнение версий модели.
  ```
    dvc exp run -S train.lr=0.05
    dvc exp run -S train.epochs=100
    dvc exp show
  ```
  ![alt text](images/dvc_run_exps.png)
- Другой вариант
    ```
    dvc metrics diff
    ```
  ![alt text](images/dvc_metrics_diff.png)

## 3. Воспроизводимость
- Версии уже зафиксированы
  ```
  poetry lock
  poetry export -f requirements.txt --output requirements.txt --without-hashes
  ```
- Инструкция по воспроизведению
  ```
    git clone --branch feature/hw2 https://github.com/AkiRusProd/itmo-enginiring-practices-ml.git
    pip install -r requirements.txt
    dvc pull
    dvc repro
  ```
- Или в контейнере
    ```
    docker build -t itmo-enginiring-practices-ml .
    docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/.dvc/cache:/app/.dvc/cache \
    -v $(pwd)/.git:/app/.git \
    -v $(pwd)/dvc.lock:/app/dvc.lock \
    -v $(pwd)/dvc.yaml:/app/dvc.yaml \
    itmo-enginiring-practices-ml
    ```
    ![alt text](images/dvc_docker2.png)
    Ничего не изменилось