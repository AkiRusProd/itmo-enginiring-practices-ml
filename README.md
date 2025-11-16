# itmo-enginiring-practices-ml

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Template for course enginiring-practices-ml

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── Dockerfile         <- Dockerfile boilerplate  
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```


# Отчет по трекину экспериментов (ДЗ3)

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


## 2. Версионирование моделей и данных в DVC
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
            - train.epochs
            - train.lr
            - train.test_size
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
    dvc exp run -S train.lr=0.05
    dvc exp run -S train.epochs=100
    dvc exp show
  ```
  ![alt text](images/dvc_run_exps.png)

## 3. Воспроизводимость
- Версии уже зафиксированы
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
    docker run --rm itmo-enginiring-practices-ml
    ```
    ![alt text](images/dvc_docker.png)
    Ничего не изменилось