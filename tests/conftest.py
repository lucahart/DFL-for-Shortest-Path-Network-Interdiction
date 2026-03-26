from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: Fast unit tests.")
    config.addinivalue_line("markers", "integration: Integration tests.")
    config.addinivalue_line("markers", "regression: Regression tests.")
    config.addinivalue_line("markers", "slow: Slow-running tests.")
    config.addinivalue_line("markers", "torch: Requires torch.")
    config.addinivalue_line("markers", "pyepo: Requires pyepo.")
    config.addinivalue_line("markers", "gurobi: Requires gurobi/gurobipy.")


@pytest.fixture
def project_root() -> Path:
    return PROJECT_ROOT
