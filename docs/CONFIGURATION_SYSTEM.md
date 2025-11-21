# Configuration Composition System Documentation

## Overview

The configuration composition system provides advanced configuration management with:

- **Pydantic Validation**: Type-safe configuration with automatic validation
- **Multiple Profiles**: dev, test, and prod environments
- **Configuration Merging**: Compose and inherit configurations
- **Environment Variables**: Dynamic parameter substitution
- **Parameter Overrides**: Change specific model parameters at runtime

## Architecture

### Components

#### 1. **schemas.py** - Configuration Models
Defines Pydantic models for type-safe configuration:
- `ConfigProfile`: Enum for dev/test/prod profiles
- `TrainConfig`: Training parameters (seed, test_size)
- `ModelsConfig`: Model parameters for all supported algorithms
- `AppConfig`: Main application configuration
- Metrics schemas for validation

#### 2. **config_manager.py** - Configuration Manager
Implements the `ConfigManager` class:
- Loads and validates configurations from YAML files
- Supports configuration merging and composition
- Handles environment variable substitution
- Provides global instance access

#### 3. **config.py** - Backward Compatible Layer
Maintains compatibility with existing code:
- Uses `ConfigManager` internally
- Exports constants (SEED, TEST_SIZE, MODELS, etc.)
- Provides `get_config()` function

## Configuration Files Structure

```
project/
├── params.yaml                 # Base configuration
├── config/
│   ├── params.dev.yaml        # Development overrides
│   ├── params.test.yaml       # Test overrides
│   └── params.prod.yaml       # Production overrides
└── src/
    ├── config.py
    ├── config_manager.py
    └── schemas.py
```

## Configuration Loading Hierarchy

```
Priority (Highest to Lowest):
1. Environment variable: CONFIG_PROFILE
2. Profile-specific config: config/params.{profile}.yaml
3. Base config: params.yaml
```

### Example Loading Process

```python
# config/params.prod.yaml (highest priority)
models:
  RandomForest:
    n_estimators: 200

# params.yaml (base, lowest priority)
models:
  RandomForest:
    n_estimators: 100

# Result: n_estimators = 200 (prod profile wins)
```

## Usage Examples

### 1. Basic Usage (Backward Compatible)

```python
# Existing code continues to work
from config import SEED, TEST_SIZE, MODELS

# Configuration automatically loaded with default profile (dev)
print(f"Seed: {SEED}")
print(f"Test size: {TEST_SIZE}")
```

### 2. Load Specific Profile

```python
from config_manager import ConfigManager
from schemas import ConfigProfile

manager = ConfigManager(profile=ConfigProfile.PROD)
config = manager.load_config()

print(f"Profile: {config.profile}")
print(f"RandomForest n_estimators: {config.models.RandomForest['n_estimators']}")
```

### 3. Use Environment Variable

```bash
# Terminal
export CONFIG_PROFILE=prod
python train.py
```

```python
# train.py
from config import get_config

config = get_config()
print(f"Using profile: {config.profile}")
```

### 4. Override Model Parameters

```python
from config_manager import get_config_manager

manager = get_config_manager()

# Override specific model parameters
updated_config = manager.override_model_params(
    "RandomForest",
    n_estimators=150,
    max_depth=20
)

print(updated_config.models.RandomForest)
```

### 5. Compose Configurations

```python
from config_manager import ConfigManager
from schemas import ConfigProfile

# Create two configs
dev_config = ConfigManager(ConfigProfile.DEV).load_config()
prod_config = ConfigManager(ConfigProfile.PROD).load_config()

# Merge: prod overrides dev
merged = dev_config.merge(prod_config)

print(f"Test size: {merged.test_size}")  # From prod_config
```

### 6. Export Configuration

```python
from config_manager import get_config_manager

manager = get_config_manager()

# As dictionary
config_dict = manager.to_dict()

# As YAML
config_yaml = manager.to_yaml()

print(config_yaml)
```

## Profile Comparisons

### Development Profile (params.dev.yaml)
- **Purpose**: Fast iteration during development
- **Key Settings**:
  - Fewer training iterations (LogisticRegression: max_iter=100)
  - Fewer trees (RandomForest: n_estimators=10)
  - Faster kernels (SVC: kernel='linear')
  - Separate directories for isolation

### Test Profile (params.test.yaml)
- **Purpose**: Balanced testing
- **Key Settings**:
  - Moderate iterations (LogisticRegression: max_iter=150)
  - Moderate trees (RandomForest: n_estimators=50)
  - Standard settings
  - Separate directories for isolation

### Production Profile (params.prod.yaml)
- **Purpose**: Optimal performance
- **Key Settings**:
  - Full iterations (LogisticRegression: max_iter=200)
  - Maximum trees (RandomForest: n_estimators=200)
  - All CPU cores (`n_jobs=-1`)
  - Conservative learning rates
  - Separate directories for isolation

## Integration with DVC

### Using Profiles with DVC

```yaml
# dvc.yaml
stages:
  train_dev:
    cmd: CONFIG_PROFILE=dev python src/train.py
    deps:
      - data/processed/processed.csv
      - src/train.py
    outs:
      - models/dev/best_model.pkl

  train_prod:
    cmd: CONFIG_PROFILE=prod python src/train.py
    deps:
      - data/processed/processed.csv
      - src/train.py
    outs:
      - models/prod/best_model.pkl
```

### Running with DVC

```bash
# Run development stage
dvc repro train_dev

# Run production stage
dvc repro train_prod
```

## Advanced Features

### 1. Configuration Validation

All configurations are automatically validated with Pydantic:

```python
# Invalid test_size will raise error
config = AppConfig(train=TrainConfig(test_size=1.5))
# ValidationError: test_size must be between 0 and 1
```

### 2. Environment Variable Substitution

```yaml
# config/params.prod.yaml
data_path: ${DATA_DIR}/processed.csv
model_dir: ${MODEL_DIR}/prod
```

```python
import os
os.environ["DATA_DIR"] = "/mnt/data"
os.environ["MODEL_DIR"] = "/mnt/models"

config = ConfigManager().load_config()
print(config.data_path)  # /mnt/data/processed.csv
```

### 3. Dynamic Model Parameters

```python
manager = ConfigManager()
config = manager.override_model_params(
    "RandomForest",
    n_estimators=500,
    max_depth=20,
    random_state=42
)
```

### 4. Configuration Merging

```python
# Deep merge for nested configurations
base = AppConfig(...)
override = AppConfig(...)

merged = base.merge(override)  # Deep merge applied
```

## Migration Guide

### From Old System to New System

**Old code:**
```python
from config import SEED, TEST_SIZE, MODELS
```

**Still works!** No changes needed. The new system is backward compatible.

**To use new features:**
```python
from config_manager import get_config_manager
from schemas import ConfigProfile

manager = ConfigManager(profile=ConfigProfile.PROD)
config = manager.load_config()
```

## Best Practices

1. **Use Profiles for Different Environments**
   - dev: Fast iteration
   - test: Balanced testing
   - prod: Optimal performance

2. **Set CONFIG_PROFILE Environment Variable**
   - Makes code portable across environments
   - Integrates with CI/CD pipelines

3. **Override Parameters Programmatically**
   - Use `override_model_params()` for dynamic changes
   - Useful for hyperparameter tuning

4. **Validate Configurations Early**
   - Pydantic validates on load
   - Catches errors before training starts

5. **Keep Base Configuration General**
   - params.yaml: Common settings
   - params.*.yaml: Environment-specific overrides

## Troubleshooting

### Configuration Not Loading

```python
from pathlib import Path

# Check if files exist
print(Path("params.yaml").exists())
print(Path("config/params.dev.yaml").exists())
```

### Wrong Profile Loaded

```python
import os
from config import get_config

# Check current profile
print(os.getenv("CONFIG_PROFILE", "dev"))
print(get_config().profile)
```

### Validation Errors

```python
# Check what went wrong
from pydantic_core import ValidationError

try:
    config = AppConfig(...)
except ValidationError as e:
    print(e.errors())
```

## API Reference

### ConfigManager

```python
class ConfigManager:
    def __init__(self, profile: Optional[ConfigProfile] = None)
    def load_config() -> AppConfig
    def get_config() -> AppConfig
    def compose_configs(*configs: AppConfig) -> AppConfig
    def override_model_params(model_name: str, **params) -> AppConfig
    def to_dict() -> dict
    def to_yaml() -> str
```

### Global Functions

```python
def initialize_config(profile: Optional[ConfigProfile] = None) -> AppConfig
def get_config_manager() -> ConfigManager
def get_config() -> AppConfig
```

### Enums

```python
class ConfigProfile(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"
```

## Performance Considerations

- Configuration loading is lazy (on first access)
- Pydantic validation adds minimal overhead
- YAML parsing optimized with safe_load
- No repeated validation for same config

## Future Enhancements

- [ ] Configuration versioning
- [ ] Configuration hot-reloading
- [ ] Configuration schema export
- [ ] Configuration validation rules DSL
- [ ] Per-stage overrides in DVC
- [ ] Integration with Hydra framework
