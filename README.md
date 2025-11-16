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
    