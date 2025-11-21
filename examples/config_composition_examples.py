"""
Examples of using the Configuration Composition System

This file demonstrates how to use the new ConfigManager for:
- Loading profiles
- Merging configurations
- Overriding parameters
- Environment variable substitution
"""

# ============================================================================
# Example 1: Basic Configuration Loading (default: dev profile)
# ============================================================================

from src.config_manager import get_config_manager, initialize_config
from src.schemas import ConfigProfile

# Initialize with default (dev) profile
config = initialize_config()
print(f"Loaded profile: {config.profile}")
print(f"Test size: {config.test_size}")
print(f"Models: {list(config.models.model_dump(exclude_none=True).keys())}")


# ============================================================================
# Example 2: Load Specific Profile
# ============================================================================

from src.config_manager import ConfigManager

# Load test profile
manager = ConfigManager(profile=ConfigProfile.TEST)
test_config = manager.load_config()
print(f"\nTest profile - Test size: {test_config.test_size}")
print(
    f"Test profile - RandomForest n_estimators: {test_config.models.RandomForest.get('n_estimators')}"
)


# ============================================================================
# Example 3: Load Profile from Environment Variable
# ============================================================================

import os

# Set environment variable
os.environ["CONFIG_PROFILE"] = "prod"

# ConfigManager will automatically pick up the profile
manager = ConfigManager()
prod_config = manager.load_config()
print(f"\nProd profile - Test size: {prod_config.test_size}")
print(
    f"Prod profile - RandomForest n_estimators: {prod_config.models.RandomForest.get('n_estimators')}"
)


# ============================================================================
# Example 4: Override Specific Model Parameters
# ============================================================================

manager = ConfigManager(profile=ConfigProfile.DEV)
config = manager.load_config()

# Override LogisticRegression parameters
updated_config = manager.override_model_params("LogisticRegression", max_iter=500)
print(
    f"\nOverridden LogisticRegression max_iter: {updated_config.models.LogisticRegression.get('max_iter')}"
)


# ============================================================================
# Example 5: Compose Multiple Configurations
# ============================================================================

# Create base config
manager = ConfigManager(profile=ConfigProfile.TEST)
base_config = manager.load_config()

# Create another config with different values
manager2 = ConfigManager(profile=ConfigProfile.PROD)
prod_config = manager2.load_config()

# Merge: prod_config overrides base_config
merged_config = base_config.merge(prod_config)
print(f"\nMerged config - Test size: {merged_config.test_size}")
print(
    f"Merged config - RandomForest n_estimators: {merged_config.models.RandomForest.get('n_estimators')}"
)


# ============================================================================
# Example 6: Export Configuration as Dictionary or YAML
# ============================================================================

manager = ConfigManager(profile=ConfigProfile.DEV)
manager.load_config()

# Export as dictionary
config_dict = manager.to_dict()
print(f"\nConfig as dict: {config_dict['train']}")

# Export as YAML string
config_yaml = manager.to_yaml()
print(f"\nConfig as YAML:\n{config_yaml[:200]}...")


# ============================================================================
# Example 7: Use in Training Script
# ============================================================================

# In your training script, you can use:
"""
import os
from src.config import get_config, MODELS, SEED, TEST_SIZE

# Configuration automatically loaded with profile from CONFIG_PROFILE env var
config = get_config()

# Or use exported constants
print(f"Using seed: {SEED}")
print(f"Using test_size: {TEST_SIZE}")
print(f"Available models: {list(MODELS.keys())}")
"""


# ============================================================================
# Example 8: Configuration Hierarchy
# ============================================================================

"""
Configuration Loading Priority (highest to lowest):
1. Environment Variable: CONFIG_PROFILE
2. Profile-specific config: config/params.{profile}.yaml
3. Base config: params.yaml

For development:
- Start with params.yaml (base)
- Override with config/params.dev.yaml values
- Result is validated with Pydantic AppConfig schema

For production:
- Start with params.yaml (base)
- Override with config/params.prod.yaml values
- Result is validated with Pydantic AppConfig schema
"""


# ============================================================================
# Example 9: Using in Different Scripts with Different Profiles
# ============================================================================

"""
# script_dev.py
import os
os.environ["CONFIG_PROFILE"] = "dev"
from src.config import get_config
config = get_config()

# script_prod.py
import os
os.environ["CONFIG_PROFILE"] = "prod"
from src.config import get_config
config = get_config()

# DVC can set environment variable:
# dvc stage add -n train -c 'CONFIG_PROFILE=prod python src/train_single.py'
"""


# ============================================================================
# Example 10: Profile Differences Summary
# ============================================================================

"""
dev profile (for development):
- Faster training: fewer trees, lower iterations
- Larger test_size for quicker iteration
- Separate logs/models/metrics directories

test profile (for testing):
- Balanced parameters
- Moderate test_size
- Separate logs/models/metrics directories

prod profile (for production):
- Optimal parameters: more trees, full resources
- Standard test_size
- Separate logs/models/metrics directories
- Uses all CPU cores (-1)
- Conservative learning rates
"""
