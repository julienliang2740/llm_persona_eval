"""Shared fixtures. Unit tests never touch the network.

The toy target and the pilot config live in the data pipeline checkout; importing
persona_eval locates it and puts it on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from persona_eval import PIPELINE_ROOT  # noqa: E402

FIXTURE_TARGETS = PIPELINE_ROOT / "tests" / "fixtures" / "targets"
TOY_TARGET_ID = "toy"


@pytest.fixture(scope="session")
def toy_spec():
    from pipeline.target import load_target

    return load_target(FIXTURE_TARGETS, TOY_TARGET_ID)


@pytest.fixture(scope="session")
def pilot_config():
    from pipeline.config import load_config

    return load_config("configs/pilot.yaml")
