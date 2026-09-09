"""Evaluation run directories: one folder per evaluation, holding everything it produced.

Repeatability is a requirement of the plan: "Retain case IDs, model/checkpoint, rubric
version, responses, settings, and grading records." That is only true if there is one
obvious place where all of it lands, so every command writes into `results/<run_id>/` and
records what it did in `run_record.json`.

Layout:

    results/<run_id>/
      run_record.json        what ran, when, with which models and settings
      suite.json             the frozen suite this run used, copied in
      answers_<arm>.jsonl    raw model answers, with sampling settings and token counts
      results_<arm>.jsonl    CaseResults: the grading records
      changes_<arm>.jsonl    ChangeVerdicts: variant comparisons
      capability_<arm>.jsonl capability check results
      analysis.json          the computed analysis
      report.md              the rendered report
      usage.jsonl            every model call and its cost
      log.txt

Arms are named, not numbered: `base` and `adapter` here, but a third checkpoint would just
be another arm and nothing in the layout changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
SUITES_DIR = REPO_ROOT / "suites"

RUN_RECORD = "run_record.json"
SUITE_COPY = "suite.json"
USAGE_FILE = "usage.jsonl"
LOG_FILE = "log.txt"
ANALYSIS_FILE = "analysis.json"
REPORT_FILE = "report.md"

logger = logging.getLogger("persona_eval.runs")


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime())


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_dir(run_id: str, results_dir: Path | None = None) -> Path:
    path = (results_dir or RESULTS_DIR) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_run(results_dir: Path | None = None) -> str | None:
    root = results_dir or RESULTS_DIR
    if not root.is_dir():
        return None
    runs = sorted(p.name for p in root.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def answers_path(directory: Path, arm: str) -> Path:
    return directory / f"answers_{arm}.jsonl"


def results_path(directory: Path, arm: str) -> Path:
    return directory / f"results_{arm}.jsonl"


def changes_path(directory: Path, arm: str) -> Path:
    return directory / f"changes_{arm}.jsonl"


def capability_path(directory: Path, arm: str) -> Path:
    return directory / f"capability_{arm}.jsonl"


def arms_present(directory: Path, prefix: str = "results") -> list[str]:
    """Which arms this run has files for, e.g. ['adapter', 'base']."""
    return sorted(p.stem[len(prefix) + 1 :] for p in directory.glob(f"{prefix}_*.jsonl"))


# --------------------------------------------------------------------------- persistence


def _plain(value: Any) -> Any:
    """Turn dataclasses and nested containers into JSON-safe values."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(_plain(row), ensure_ascii=False, default=str) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ------------------------------------------------------------------------------ records


def read_record(directory: Path) -> dict[str, Any]:
    path = directory / RUN_RECORD
    return json.loads(path.read_text()) if path.is_file() else {}


def write_record(directory: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into the run record. Every command calls this when it finishes."""
    record = read_record(directory)
    record.update(_plain(updates))
    record["updated_utc"] = now_utc()
    write_json(directory / RUN_RECORD, record)
    return record


def record_stage(directory: Path, stage: str, payload: dict[str, Any]) -> None:
    """Record one stage's outcome under `stages.<stage>` without clobbering the others."""
    record = read_record(directory)
    stages = record.get("stages") or {}
    stages[stage] = {**_plain(payload), "finished_utc": now_utc()}
    record["stages"] = stages
    record["updated_utc"] = now_utc()
    write_json(directory / RUN_RECORD, record)


def sampling_signature(config: Any, role_name: str) -> dict[str, Any]:
    """The generation settings for one arm, recorded so the report can prove they matched.

    The plan requires "compatible generation settings for the baseline and each checkpoint".
    Recording the signature turns that from a promise into something checkable after the run.
    """
    # Delegated to the answering module rather than re-derived here. Reading the config a
    # second time is how this went wrong before: this function recorded the role's
    # temperature and token budget while answering used the `evaluation:` block, so a run
    # could sample at one setting and certify another. There is one resolver, and both the
    # answers and this record now come from it.
    from persona_eval.run.answer import resolve_settings

    role = config.role(role_name)
    settings = resolve_settings(config, role)
    return {
        "role": role_name,
        "model": role.model,
        "base_url": role.base_url,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "top_p": settings.top_p,
        "seed": settings.seed,
        "system_prompt": settings.system_prompt or None,
        "max_tokens_source": getattr(settings, "max_tokens_source", None),
    }


def settings_match(signatures: Sequence[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Do the arms share the settings that affect comparability? Model id may differ."""
    if len(signatures) < 2:
        return True, []
    compared = ("temperature", "max_tokens", "top_p", "seed", "system_prompt")
    problems = []
    first = signatures[0]
    for other in signatures[1:]:
        for key in compared:
            if first.get(key) != other.get(key):
                problems.append(
                    f"{key}: {first.get('role')}={first.get(key)!r} vs {other.get('role')}={other.get(key)!r}"
                )
    return not problems, problems


def case_seed(case_id: str, base_seed: int = 13) -> int:
    """A per-case seed that is the same for every arm.

    At this sample size sampling noise is a real competitor to any effect, so both arms
    answer a given case from the same seed. It does not make the arms deterministic
    relative to each other (different weights walk different paths) but it removes one
    source of variance for free.
    """
    import hashlib

    digest = hashlib.sha256(f"{base_seed}:{case_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def configure_logging(directory: Path, verbose: bool = False) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(console)
    log_path = directory / LOG_FILE
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path for h in root.handlers):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
