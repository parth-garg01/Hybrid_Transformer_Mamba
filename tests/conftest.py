import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL_SCALE = os.path.join(ROOT, "experiments", "full_scale")
if FULL_SCALE not in sys.path:
    sys.path.insert(0, FULL_SCALE)

import pytest


@pytest.fixture(autouse=True)
def _cpu_tests_use_stand_in(request, monkeypatch):
    """CPU tests are written against the GRU stand-in. On a machine with mamba_ssm
    installed (Kaggle), force the stand-in for every test except the test_gpu_* ones,
    which exercise the real backend."""
    if request.node.name.startswith("test_gpu"):
        return
    try:
        import models_scaled as ms
    except Exception:
        return
    monkeypatch.setattr(ms, "MAMBA_AVAILABLE", False)
    monkeypatch.setattr(ms, "MAMBA2_AVAILABLE", False)
    monkeypatch.setattr(ms, "MAMBA_BACKEND", "gru_fallback")
