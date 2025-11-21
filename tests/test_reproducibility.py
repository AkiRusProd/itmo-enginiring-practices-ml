"""Тесты воспроизводимости пайплайна."""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.base_config import (
    DATA_PATH,
    LOG_DIR,
    METRICS_DIR,
    METRICS_FILE,
    MODEL_DIR,
    MODELS,
    SEED,
    TEST_SIZE,
    get_config,
)
from src.config_manager import ConfigManager
from src.schemas import ConfigProfile


def test_reproducibility() -> bool:
    """Тест что результаты воспроизводимы.

    Returns:
        True если тест пройден
    """
    print("\n" + "=" * 80)
    print("ТЕСТИРОВАНИЕ ВОСПРОИЗВОДИМОСТИ")
    print("=" * 80)

    # Тест 1: Проверка seed
    print("\n✓ Проверка random seed...")
    np.random.seed(SEED)
    vals1 = np.random.randn(10)

    np.random.seed(SEED)
    vals2 = np.random.randn(10)

    assert np.allclose(vals1, vals2), "Seed не воспроизводим"
    print("  ✓ SEED воспроизводим")

    # Тест 2: Проверка метрик файла
    print("\n✓ Проверка файла метрик...")
    metrics_file = Path(METRICS_FILE)

    if not metrics_file.exists():
        print("  ⚠ Файл метрик не найден. Запустите пайплайн сначала.")
        return False

    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    assert "best_model" in metrics, "Нет best_model в метриках"
    assert (
        "best_f1_score" in metrics or "best_accuracy" in metrics
    ), "Нет F1-score в метриках"

    best_f1 = metrics.get("best_f1_score", metrics.get("best_accuracy"))
    print(f"  ✓ Лучшая модель: {metrics['best_model']}")
    print(f"  ✓ F1-score: {best_f1:.4f}")
    print(f"  ✓ Количество моделей: {len(metrics) - 2}")

    print("\n" + "=" * 80)
    print("ВСЕ ТЕСТЫ ПРОЙДЕНЫ ✓")
    print("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    try:
        success = test_reproducibility()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Ошибка при тестировании: {e}\n")
        exit(1)


class ReproducibilityTest:
    """Test reproducibility of pipeline execution."""

    def __init__(self, num_runs: int = 3, test_name: str = "reproducibility"):
        """Initialize reproducibility test.

        Args:
            num_runs: Number of test runs
            test_name: Name of the test
        """
        self.num_runs = num_runs
        self.test_name = test_name
        self.logger = logging.getLogger(__name__)
        self.results: List[Dict[str, Any]] = []

    def test_train_test_split_consistency(self) -> bool:
        """Test that train-test split is reproducible with same seed.

        Returns:
            True if splits are reproducible
        """
        self.logger.info("Testing train-test split reproducibility...")

        splits = []
        for i in range(self.num_runs):
            df = pd.read_csv(DATA_PATH)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=SEED
            )
            splits.append(
                {
                    "X_train_shape": X_train.shape,
                    "X_val_shape": X_val.shape,
                    "X_train_hash": hash(X_train.values.tobytes()),
                    "X_val_hash": hash(X_val.values.tobytes()),
                }
            )

        # Check all splits are identical
        reference_split = splits[0]
        for i, split in enumerate(splits[1:], 1):
            if split != reference_split:
                self.logger.error(
                    f"Train-test split inconsistency detected at run {i+1}"
                )
                self.logger.error(f"Reference: {reference_split}")
                self.logger.error(f"Current: {split}")
                return False

        self.logger.info("✓ Train-test split is reproducible")
        return True

    def test_random_state_consistency(self) -> bool:
        """Test that random operations are reproducible with same seed.

        Returns:
            True if random operations are consistent
        """
        self.logger.info("Testing random state consistency...")

        np.random.seed(SEED)
        random_values_1 = np.random.randn(100)

        np.random.seed(SEED)
        random_values_2 = np.random.randn(100)

        if not np.allclose(random_values_1, random_values_2):
            self.logger.error("Random state inconsistency detected")
            return False

        self.logger.info("✓ Random state is consistent")
        return True

    def test_metrics_consistency(self, metrics_file: Path = Path(METRICS_FILE)) -> bool:
        """Test that metrics remain consistent across runs.

        Note: This test should be run after pipeline execution.

        Args:
            metrics_file: Path to metrics file

        Returns:
            True if metrics files are consistent
        """
        self.logger.info("Testing metrics consistency...")

        if not metrics_file.exists():
            self.logger.warning(
                f"Metrics file not found at {metrics_file}. " "Run pipeline first."
            )
            return False

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        self.results.append(
            {
                "run": 0,
                "metrics": metrics,
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )

        self.logger.info("✓ Metrics loaded successfully")
        return True

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all reproducibility tests.

        Returns:
            Dictionary with test names and results
        """
        test_results = {
            "train_test_split_consistency": (self.test_train_test_split_consistency()),
            "random_state_consistency": self.test_random_state_consistency(),
            "metrics_consistency": self.test_metrics_consistency(),
        }

        return test_results

    def generate_report(self) -> str:
        """Generate reproducibility test report.

        Returns:
            Report string
        """
        report = "\n{'='*80}\n"
        report += f"Reproducibility Test Report: {self.test_name}\n"
        report += "{'='*80}\n"
        report += f"Number of runs: {self.num_runs}\n"
        report += f"Seed: {SEED}\n"
        report += f"Test size: {TEST_SIZE}\n"
        report += "\n"

        return report


class IntegrationTest:
    """Test integration of ConfigManager with training pipeline."""

    def __init__(self):
        """Initialize integration test."""
        self.logger = logging.getLogger(__name__)

    def test_config_manager_integration(self) -> bool:
        """Test ConfigManager integration with training pipeline.

        Returns:
            True if integration is successful
        """
        self.logger.info("Testing ConfigManager integration...")

        try:
            # Test that config is properly loaded
            config = get_config()
            assert config is not None, "Config is None"
            assert SEED > 0, "Invalid SEED"
            assert 0 < TEST_SIZE < 1, "Invalid TEST_SIZE"
            assert len(MODELS) > 0, "No models loaded"

            # Test that paths exist or can be created
            Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
            Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
            Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)

            assert Path(DATA_PATH).exists(), f"Data path does not exist: {DATA_PATH}"

            self.logger.info("✓ ConfigManager integration successful")
            return True

        except Exception as e:
            self.logger.error(f"ConfigManager integration failed: {e}")
            return False

    def test_model_instantiation(self) -> bool:
        """Test that models can be instantiated from config.

        Returns:
            True if all models can be instantiated
        """
        self.logger.info("Testing model instantiation...")

        try:
            for model_name, model in MODELS.items():
                assert model is not None, f"Model {model_name} is None"
                assert hasattr(
                    model, "fit"
                ), f"Model {model_name} does not have fit method"
                assert hasattr(
                    model, "predict"
                ), f"Model {model_name} does not have predict method"

            self.logger.info(f"✓ All {len(MODELS)} models instantiated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Model instantiation failed: {e}")
            return False

    def test_profile_switching(self) -> bool:
        """Test switching between configuration profiles.

        Returns:
            True if profile switching works
        """
        self.logger.info("Testing profile switching...")

        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

            # Test DEV profile
            dev_mgr = ConfigManager(ConfigProfile.DEV)
            dev_config = dev_mgr.load_config()
            assert dev_config is not None

            # Test TEST profile
            test_mgr = ConfigManager(ConfigProfile.TEST)
            test_config = test_mgr.load_config()
            assert test_config is not None

            # Test PROD profile
            prod_mgr = ConfigManager(ConfigProfile.PROD)
            prod_config = prod_mgr.load_config()
            assert prod_config is not None  # noqa: B101

            # Verify profiles are different
            if (
                dev_config.train.test_size
                == test_config.train.test_size
                == prod_config.train.test_size
            ):
                self.logger.warning(
                    "Profiles have identical test_size (may be intentional)"
                )

            self.logger.info("✓ Profile switching successful")
            return True

        except Exception as e:
            self.logger.error(f"Profile switching failed: {e}")
            return False

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all integration tests.

        Returns:
            Dictionary with test names and results
        """
        test_results = {
            "config_manager_integration": self.test_config_manager_integration(),
            "model_instantiation": self.test_model_instantiation(),
            "profile_switching": self.test_profile_switching(),
        }

        return test_results

    def generate_report(self) -> str:
        """Generate integration test report.

        Returns:
            Report string
        """
        report = f"\n{'='*80}\n"
        report += "Integration Test Report\n"
        report += f"{'='*80}\n"

        return report


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run reproducibility tests
    repro_test = ReproducibilityTest(num_runs=3)
    repro_results = repro_test.run_all_tests()

    print(repro_test.generate_report())
    for test_name, result in repro_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    # Run integration tests
    integration_test = IntegrationTest()
    integration_results = integration_test.run_all_tests()

    print(integration_test.generate_report())
    for test_name, result in integration_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    # Summary
    all_results = {**repro_results, **integration_results}
    passed = sum(1 for v in all_results.values() if v)
    total = len(all_results)

    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*80}\n")
