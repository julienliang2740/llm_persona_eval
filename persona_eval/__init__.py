"""Evaluation for the value-instantiation project.

This package reuses the data pipeline's model client, config, records and target loader
rather than copying them. The pipeline checkout is located once, here, and put on sys.path
before any submodule imports `pipeline` or `prompts`:

  1. $LLM_PERSONA_PIPELINE, if set
  2. ../llm_persona_data_pipeline next to this repository
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_PIPELINE = REPO_ROOT.parent / "llm_persona_data_pipeline"


def locate_pipeline() -> Path:
    candidate = Path(os.environ.get("LLM_PERSONA_PIPELINE") or _DEFAULT_PIPELINE).resolve()
    if not (candidate / "pipeline" / "model.py").exists():
        raise RuntimeError(
            f"llm_persona_data_pipeline not found at {candidate}. Check it out next to this "
            "repository or set LLM_PERSONA_PIPELINE to its path."
        )
    return candidate


PIPELINE_ROOT = locate_pipeline()
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
