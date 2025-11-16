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


# Отчет о настройке рабочего места Data Scientist

## 1. Структура проекта
- Использован [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/)
  ```bash
    pip install cookiecutter-data-science
    ccds https://github.com/drivendataorg/cookiecutter-data-science
  ```
- Созданы папки: src, notebooks, tests итд
- Добавлен README.md


## 2. Качество кода
- Настроены pre-commit hooks: Black, isort, Ruff, MyPy, Bandit
  ```
  pre-commit install
  ```
- Создан pyproject.toml
- Выполнены тестовые коммиты
   ![alt text](images/hooks_run.png)

## 3. Управление зависимостями
- Использован Poetry
  * Команды
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    poetry install
    poetry install --with dev
    ```

  * Установка зависимостей через poetry
  ![alt text](images/install_libs.png)
- Создан Dockerfile
  * Команды
    ```
    docker build -t itmo-enginiring-practices-ml .
    docker run --rm itmo-enginiring-practices-ml
    ```
  * Пример сборки
  ![alt text](images/docker_build.png)
  * Запуск и уничтожение контейнера
  ![alt text](images/docker_run.png)

- Сформирован requirements.txt
    ```
    poetry export -f requirements.txt --output requirements.txt --without-hashes
    ```

## 4. Git workflow
- Создан .gitignore
- Созданы ветки master, hw1


--------

