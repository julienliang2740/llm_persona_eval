"""Persistence for an authored suite, and the summary that describes what is in it.

A suite is written once, costs money to author, and is then read by every run, every judge and
every report. It is stored as one JSON document rather than JSONL because the unit that must
stay consistent is the whole suite: a case whose family is missing, or a variant whose original
was lost, is not a partial suite but a broken one.

`suite_summary` exists so a person can see what they actually bought before spending anything
on running it: which family kinds, tasks, variants and dimensions are covered, how thin the
denominators are, and how much of the suite came from outside this project.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from persona_eval.suite.schema import (
    DIMENSIONS,
    FAMILY_KINDS,
    TASK_TYPES,
    VARIANTS,
    Suite,
    SuiteError,
)

logger = logging.getLogger("persona_eval.suite.io")


def save_suite(suite: Suite, path: str | Path) -> Path:
    """Write the suite as JSON. Refuses to write a suite that does not validate."""
    problems = suite.validate()
    if problems:
        raise SuiteError(
            f"refusing to save an invalid suite to {path}: "
            + "; ".join(problems[:5])
            + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = suite.to_dict()
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "wrote suite %s (%d families, %d cases, version %s) to %s",
        suite.suite_id,
        len(suite.families),
        len(suite.cases),
        suite.version,
        target,
    )
    return target


def load_suite(path: str | Path) -> Suite:
    """Read a suite back and check it, so a hand edit cannot reach the runner unnoticed."""
    source = Path(path)
    if not source.exists():
        raise SuiteError(f"no suite at {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    suite = Suite.from_dict(payload)
    problems = suite.validate()
    if problems:
        raise SuiteError(
            f"{source} is not a valid suite: "
            + "; ".join(problems[:5])
            + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        )
    stored_version = payload.get("suite_version")
    if stored_version and stored_version != suite.version:
        # The version is the hash of the cases, so a mismatch means the file was edited after
        # it was written. That is allowed, but a run must not silently report the old version.
        logger.warning(
            "%s records suite_version %s but its cases hash to %s; the file has been edited "
            "since it was written",
            source,
            stored_version,
            suite.version,
        )
    return suite


def _counter(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def suite_summary(suite: Suite) -> dict[str, Any]:
    """Counts by family kind, task, variant and dimension, plus the thin denominators.

    Ordered by the schema's own vocabulary rather than by frequency, so two summaries from
    different runs line up column for column.
    """
    kind_by_family = {family.family_id: family.kind for family in suite.families}
    domain_by_family = {family.family_id: family.domain for family in suite.families}

    case_kinds = _counter(kind_by_family.get(case.family_id, "?") for case in suite.cases)
    case_tasks = _counter(case.task for case in suite.cases)
    case_variants = _counter(case.variant for case in suite.cases)
    case_domains = _counter(domain_by_family.get(case.family_id, "?") for case in suite.cases)
    measures = _counter(case.measures for case in suite.cases if case.measures)

    dimension_coverage = {dimension: 0 for dimension in DIMENSIONS}
    for case in suite.cases:
        for dimension in case.rubric.dimensions:
            dimension_coverage[dimension] = dimension_coverage.get(dimension, 0) + 1

    per_family = [len(suite.cases_of(family.family_id)) for family in suite.families]
    provenance: dict[str, int] = {}
    for family in suite.families:
        source = str(family.provenance).split(":", 1)[0] or "authored"
        provenance[source] = provenance.get(source, 0) + 1

    negative_control_ids = {
        family.family_id for family in suite.families if family.kind == "negative_control"
    }

    return {
        "suite_id": suite.suite_id,
        "suite_version": suite.version,
        "target_id": suite.target_id,
        "spec_version": suite.spec_version,
        "spec_hash": suite.spec_hash,
        "created_utc": suite.created_utc,
        "families": len(suite.families),
        "cases": len(suite.cases),
        "families_by_kind": {
            kind: sum(1 for f in suite.families if f.kind == kind) for kind in FAMILY_KINDS
        },
        "cases_by_family_kind": {kind: case_kinds.get(kind, 0) for kind in FAMILY_KINDS},
        "cases_by_task": {task: case_tasks.get(task, 0) for task in TASK_TYPES},
        "cases_by_variant": {variant: case_variants.get(variant, 0) for variant in VARIANTS},
        "cases_by_domain": dict(sorted(case_domains.items())),
        "cases_by_measure": dict(sorted(measures.items())),
        "dimension_coverage": dimension_coverage,
        "dimensions_uncovered": [d for d, n in dimension_coverage.items() if n == 0],
        # The plan asks for eligible-case denominators next to every rate, and these are the
        # two that are always thin enough to mislead.
        "negative_control_cases": sum(
            1 for case in suite.cases if case.family_id in negative_control_ids
        ),
        "continuation_cases": sum(1 for case in suite.cases if case.is_continuation),
        "families_without_cases": [
            family.family_id for family in suite.families if not suite.cases_of(family.family_id)
        ],
        "cases_per_family": {
            "min": min(per_family) if per_family else 0,
            "max": max(per_family) if per_family else 0,
            "mean": round(sum(per_family) / len(per_family), 2) if per_family else 0.0,
        },
        "provenance": dict(sorted(provenance.items())),
        "deterministic_checks": sum(len(case.deterministic_checks) for case in suite.cases),
    }


__all__ = ["load_suite", "save_suite", "suite_summary"]
