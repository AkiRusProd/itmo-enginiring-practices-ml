# Quick Start Guide

## Prerequisites

- Python 3.9 or higher
- Git
- Docker (optional, for running services like ClearML Server)
- pip or poetry

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/AkiRusProd/itmo-enginiring-practices-ml.git
cd itmo-enginiring-practices-ml
```

### 2. Install dependencies

**Option A: Using pip**

```bash
pip install -r requirements.txt
```

**Option B: Using poetry**

```bash
poetry install
poetry shell
```

### 3. Download and prepare data

The project uses DVC for data management. Initialize DVC and pull the dataset:

```bash
dvc pull
```

If you want to download the raw Titanic dataset:

```bash
python -c "from sklearn.datasets import load_iris; import pandas as pd; pd.read_csv('data/raw/titanic.csv', nrows=1)"
```

## Running the Pipeline

### Full pipeline execution

Run the complete ML pipeline (preprocessing, training, evaluation, and report generation):

```bash
dvc repro
```

This will:
1. **Preprocess** the data
2. **Train** multiple models (Decision Tree, Random Forest, Logistic Regression, SVC, Gradient Boosting, AdaBoost, Bagging)
3. **Evaluate** models on test set
4. **Select** the best model
5. **Generate** an experiment report with visualizations

### Train a single model

To train only one model:

```bash
python src/train_single.py --model RandomForest --profile dev
```

Available models: `DecisionTree`, `RandomForest`, `LogisticRegression`, `SVC`, `GradientBoosting`, `AdaBoost`, `Bagging`

Available profiles: `dev`, `test`, `prod`

### Evaluate models

```bash
python src/eval.py
```

### Generate experiment report

```bash
python src/generate_report.py
```

This creates `docs/experiments.md` with visualizations and metrics.

## Configuration

The project uses a configuration system with environment-specific profiles:

- **`params.yaml`** - Base configuration
- **`config/params.dev.yaml`** - Development overrides
- **`config/params.test.yaml`** - Test overrides
- **`config/params.prod.yaml`** - Production overrides

To change the profile, set the environment variable:

```bash
export CONFIG_PROFILE=prod
```

See [Configuration System Documentation](../CONFIGURATION_SYSTEM.md) for more details.

## Project Structure

```
.
├── data/                 # Raw and processed datasets
├── src/                  # Source code
│   ├── preprocess.py     # Data preprocessing
│   ├── train.py          # Multi-model training
│   ├── train_single.py   # Single model training
│   ├── eval.py           # Model evaluation
│   ├── select_best.py    # Best model selection
│   ├── generate_report.py # Report generation
│   └── config_manager.py # Configuration management
├── docs/                 # Documentation
├── metrics/              # Training metrics
├── models/               # Trained model artifacts
├── reports/              # Generated reports
└── dvc.yaml              # DVC pipeline configuration
```

## Viewing Results

After running the pipeline, you can view:

1. **Experiment Report**: `docs/experiments.md`
2. **Metrics**: 
   - All models: `metrics/train_models/`
   - Best model: `metrics/best_model_advanced_metrics.json`
3. **Visualizations**: `docs/assets/images/`
4. **Logs**: `logs/` directory

## ClearML Integration (Optional)

To enable ClearML for experiment tracking:

1. Start ClearML Server:
   ```bash
   docker compose -f docker-compose.clearml.yml up -d
   ```

2. Configure credentials:
   ```bash
   clearml-init
   ```

3. Run training with ClearML logging:
   ```bash
   python src/train.py
   ```

See [Deployment Guide](deployment.md) for more details.

## Troubleshooting

### Missing data
```bash
dvc pull
```

### Permission errors
```bash
chmod -R 755 /path/to/project
```

### DVC errors
```bash
dvc cache dir  # Check cache location
dvc dag        # Visualize pipeline
```

## Next Steps

- Read the [Deployment Guide](deployment.md) for production setup
- Check [API Reference](../reference/config.md) for API documentation
- View [Experiment Results](../experiments.md) to see latest metrics

## Support

For issues or questions, please refer to:
- [GitHub Issues](https://github.com/AkiRusProd/itmo-enginiring-practices-ml/issues)
- [Configuration System Documentation](../CONFIGURATION_SYSTEM.md)
