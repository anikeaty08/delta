"""Shared pytest fixtures for the test suite."""

import shutil
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

try:  # pragma: no cover
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data_dir(temp_dir: Path) -> Path:
    """Create a directory with sample data files."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_config() -> dict:
    """Provide a mock configuration dictionary."""
    return {
        "model": "resnet18",
        "dataset": "cifar10",
        "num_classes": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
    }


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Create a sample tensor for testing."""
    if torch is None:  # pragma: no cover
        pytest.skip("torch is not installed")
    return torch.randn(4, 3, 32, 32)


@pytest.fixture
def sample_numpy_array() -> np.ndarray:
    """Create a sample numpy array for testing."""
    return np.random.rand(100, 10)


@pytest.fixture
def mock_model_state():
    """Mock model state dictionary."""
    if torch is None:  # pragma: no cover
        pytest.skip("torch is not installed")
    return {
        "conv1.weight": torch.randn(64, 3, 7, 7),
        "conv1.bias": torch.randn(64),
        "fc.weight": torch.randn(1000, 512),
        "fc.bias": torch.randn(1000),
    }


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    if torch is not None:  # pragma: no cover
        torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    if torch is None:  # pragma: no cover
        pytest.skip("torch is not installed")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
