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
Развернут локальный сервер, используя [docker-compose.clearml.yml](docker-compose.clearml.yml). В качестве БД используется MongoDB (контейнер clearml-mongo), в качестве хранилища артефактов настроен локальный файловый сервер (clearml-fileserver), персистентность данных обеспечена через Docker Volumes.


### 1.3	Создать проект и эксперименты
Внедрили clearml в каждый этап пайплайна [dvc.py](dvc.py) в [src/](src).

![alt text](images/clearml_run.png)

### 1.4	Настроить аутентификацию
Настроили в 1.1.1

## 2.	Трекинг экспериментов (3 балла):
### 2.1	Настроить автоматическое логирование
### 2.2	Создать систему сравнения экспериментов
### 2.3	Настроить логирование метрик и параметров
### 2.4	Создать дашборды для анализа

## 3.	Управление моделями (3 балла):
### 3.1	Настроить регистрацию и версионирование моделей
###	3.2 Создать систему метаданных для моделей
###	3.3 Настроить автоматическое создание версий
###	3.4 Создать систему сравнения моделей

## 4.	Пайплайны (2 балла):
###	4.1 Создать ClearML пайплайны для ML workflow
###	4.2 Настроить автоматический запуск пайплайнов
###	4.3 Создать систему мониторинга выполнения
###	4.4 Настроить уведомления

## 5.	Отчет о проделанной работе (1 балл):
###	5.1 Создать отчет в формате Markdown
Выполнено
###	5.2 Описать настройку каждого инструмента
Выполнено
###	5.3 Добавить скриншоты результатов
Выполнено
###	5.4 Сохранить отчет в Git репозитории
Выполнено


