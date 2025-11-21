# Быстрая справка по системе композиции конфигураций

## Быстрый старт

### 1. Использование профилей
```bash
# DEV (по умолчанию - быстрая разработка)
export CONFIG_PROFILE=dev
python src/train.py

# TEST (сбалансированное тестирование)
export CONFIG_PROFILE=test
python src/train.py

# PROD (оптимальная производительность)
export CONFIG_PROFILE=prod
python src/train.py
```

### 2. В коде (обратная совместимость)
```python
# Старый способ (по-прежнему работает)
from base_config import SEED, TEST_SIZE, MODELS

# Новый способ
from base_config import get_config
config = get_config()
print(f"Profile: {config.profile}")
```

### 3. Расширенное использование
```python
from base_config_manager import ConfigManager
from schemas import ConfigProfile

# Загрузить PROD профиль
mgr = ConfigManager(ConfigProfile.PROD)
config = mgr.load_config()

# Переопределить параметры
updated = mgr.override_model_params("RandomForest", n_estimators=500)

# Слить конфигурации
merged = dev_config.merge(prod_config)

# Экспортировать
config_yaml = mgr.to_yaml()
config_dict = mgr.to_dict()
```

## Сравнение профилей

| Параметр | DEV | TEST | PROD |
|----------|-----|------|------|
| RandomForest n_estimators | 10 | 50 | 200 |
| LogisticRegression max_iter | 100 | 150 | 200 |
| SVC kernel | linear | rbf | rbf |
| DecisionTree max_depth | 5 | 8 | 15 |
| test_size | 0.3 | 0.25 | 0.2 |
| Использует все ядра | ❌ | ❌ | ✅ |

## Файлы конфигурации

```
params.yaml              # Базовая конфигурация (общие параметры)
config/
├── params.dev.yaml     # DEV переопределения (быстро)
├── params.test.yaml    # TEST переопределения (сбалансиро)
└── params.prod.yaml    # PROD переопределения (оптимально)
```

## Иерархия конфигураций

```
Приоритет (высокий → низкий):
1. CONFIG_PROFILE env var → выбирает профиль
2. config/params.{profile}.yaml → профиль-специфичные параметры
3. params.yaml → базовые параметры (по умолчанию)
```

## DVC Integration

```yaml
# dvc.yaml
stages:
  train_dev:
    cmd: CONFIG_PROFILE=dev python src/train.py
    
  train_prod:
    cmd: CONFIG_PROFILE=prod python src/train.py
```

```bash
# Запустить конкретный stage
dvc repro train_dev
dvc repro train_prod
```

## API Reference

### ConfigManager
```python
from base_config_manager import ConfigManager
from schemas import ConfigProfile

# Инициализация
mgr = ConfigManager(profile=ConfigProfile.DEV)

# Методы
config = mgr.load_config()                    # Загрузить конфиг
config = mgr.get_config()                     # Получить конфиг
config = mgr.override_model_params("RF", n_estimators=100)  # Переопределить
config = mgr.compose_configs(cfg1, cfg2)      # Слить конфиги
yaml_str = mgr.to_yaml()                      # В YAML
dict_obj = mgr.to_dict()                      # В словарь
```

### Global Functions
```python
from base_config_manager import (
    initialize_config,      # Инициализировать глобальный конфиг
    get_config_manager,     # Получить глобальный менеджер
    get_config              # Получить глобальный конфиг
)

config = initialize_config(ConfigProfile.PROD)
mgr = get_config_manager()
config = get_config()
```

### Schemas
```python
from schemas import (
    ConfigProfile,    # Enum: DEV, TEST, PROD
    AppConfig,        # Главный конфиг
    TrainConfig,      # Параметры тренировки
    ModelsConfig,     # Параметры моделей
)
```

## Примеры

### Пример 1: Быстрая разработка
```bash
export CONFIG_PROFILE=dev
python src/train_single.py --model RandomForest
```

### Пример 2: Тестирование с перепределением
```python
from base_config_manager import get_config_manager

mgr = get_config_manager()
mgr.override_model_params("LogisticRegression", max_iter=500)
mgr.override_model_params("RandomForest", n_estimators=100)
```

### Пример 3: Слияние конфигураций
```python
from base_config_manager import ConfigManager
from schemas import ConfigProfile

dev = ConfigManager(ConfigProfile.DEV).load_config()
prod = ConfigManager(ConfigProfile.PROD).load_config()
merged = dev.merge(prod)  # prod переопределяет dev
```

### Пример 4: Использование в DVC pipeline
```yaml
stages:
  train_models:
    foreach: ${models}
    do:
      cmd: CONFIG_PROFILE=${PROFILE} python src/train_single.py --model ${key}
      params:
        - train
        - models
```

## Часто задаваемые вопросы

**Q: Как выбрать профиль?**
- DEV: Для быстрой разработки (быстрое обучение)
- TEST: Для тестирования (сбалансированные параметры)
- PROD: Для production (оптимальные результаты)

**Q: Как переопределить параметры?**
```python
manager.override_model_params("RandomForest", n_estimators=500)
```

**Q: Как использовать в DVC?**
```bash
export CONFIG_PROFILE=prod
dvc repro
```

**Q: Совместимо ли со старым кодом?**
Да, 100% совместимо. Все старые импорты работают как раньше.

**Q: Как создать свой профиль?**
1. Создать файл `config/params.custom.yaml`
2. Добавить профиль в `schemas.py`
3. Использовать: `ConfigManager(ConfigProfile.CUSTOM)`

## Документация

Полная документация: `docs/CONFIGURATION_SYSTEM.md`
Примеры кода: `examples/config_composition_examples.py`
Отчет о реализации: `CONFIGURATION_COMPOSITION_REPORT.md`
