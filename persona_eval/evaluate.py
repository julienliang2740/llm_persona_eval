"""Run an endpoint over a run's eval.jsonl and judge each answer.

`--endpoint-role` names a model role from the pipeline config, so the same code evaluates
the local base model before fine-tuning and the adapted model after it. With two result
files it prints and writes a before/after table.

Moved from llm_persona_data_pipeline/pipeline/evaluate.py; the pipeline package it imports
is located by persona_eval/__init__.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pipeline import records
from pipeline.config import RunConfig
from pipeline.model import LocalEndpointUnavailable, ModelClient, gather_bounded
from pipeline.target import TargetSpec, render_for_reviewer
from prompts import render

from persona_eval.prompts import EVAL_ANSWER_SYSTEM_PROMPT, EVAL_JUDGE_PROMPT

logger = logging.getLogger("persona_eval.evaluate")


def results_path(run_dir: Path, label: str) -> Path:
    return run_dir / f"eval_results_{label}.jsonl"


async def judge_answers(
    config: RunConfig,
    spec: TargetSpec,
    run_dir: Path,
    answers: list[dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    """Judge a list of {prompt_id, model, text} answers against the target.

    This is the single judging path. The answers either come from calling a model role
    live over eval.jsonl, or from an answers file written elsewhere (for instance by
    llm_persona_training/generate_with_adapter.py), so a before/after pair can mix the two.
    """
    from pipeline.export import EVAL_FILE

    items = {row["meta"]["prompt_id"]: row for row in records.iter_jsonl(run_dir / EVAL_FILE)}
    spec_text = render_for_reviewer(spec)
    judge_role = config.role("judge") if "judge" in config.roles else config.role("reviewer")

    async with ModelClient.from_config(config, run_dir / records.USAGE_FILE, "evaluate") as client:

        async def judge_one(answer: dict[str, Any]) -> dict[str, Any] | None:
            item = items.get(answer["prompt_id"])
            if item is None:
                logger.warning(
                    "answer for %s has no matching row in %s; skipped",
                    answer["prompt_id"],
                    EVAL_FILE,
                )
                return None
            payload, _ = await client.complete_json(
                judge_role,
                [
                    {
                        "role": "user",
                        "content": render(
                            EVAL_JUDGE_PROMPT,
                            target_spec=spec_text,
                            user_prompt=item["prompt"],
                            expected_behavior=item["expected_behavior"],
                            pass_fail_notes="\n".join(f"- {n}" for n in item["pass_fail_notes"]),
                            candidate_answer=answer["text"],
                        ),
                    }
                ],
                stage=f"evaluate.judge.{label}",
                record_id=answer["prompt_id"],
            )
            return {
                "prompt_id": answer["prompt_id"],
                "model": answer.get("model", ""),
                "text": answer["text"],
                "endpoint": label,
                "family_id": item["family_id"],
                "case_type": item["case_type"],
                "variant": item["variant"],
                # Carried so a before/after table can show whether the model held up on
                # reframed prompts and on both members of a contrastive pair.
                "counterfactual_group_id": item["meta"].get("counterfactual_group_id"),
                "judge": {
                    "pass": bool(payload.get("pass")),
                    "action_summary": str(payload.get("action_summary", "")).strip(),
                    "principle_notes": [str(n) for n in (payload.get("principle_notes") or [])],
                    "failure_modes_hit": [str(f) for f in (payload.get("failure_modes_hit") or [])],
                    "rationale": str(payload.get("rationale", "")).strip(),
                    "judge_model": judge_role.model,
                },
            }

        judged = await gather_bounded([judge_one(answer) for answer in answers])

    rows: list[dict[str, Any]] = []
    for result in judged:
        if isinstance(result, Exception):
            logger.error("eval judging failed: %s", result)
            continue
        if result is not None:
            rows.append(result)
    return rows


async def answer_eval_set(
    config: RunConfig, run_dir: Path, endpoint_role: str, label: str
) -> list[dict[str, Any]]:
    """Run one configured model role over every eval.jsonl prompt."""
    from pipeline.export import EVAL_FILE

    eval_items = list(records.iter_jsonl(run_dir / EVAL_FILE))
    role = config.role(endpoint_role)
    configured_max = config.evaluation.get("answer_max_tokens")
    max_answer_tokens = int(configured_max) if configured_max else role.max_tokens
    temperature = float(config.evaluation.get("temperature", 0.7))

    async with ModelClient.from_config(config, run_dir / records.USAGE_FILE, "evaluate") as client:

        async def answer(item: dict[str, Any]) -> dict[str, Any]:
            response = await client.complete(
                role,
                [
                    {"role": "system", "content": EVAL_ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": item["prompt"]},
                ],
                temperature=temperature,
                max_tokens=max_answer_tokens,
                stage=f"evaluate.answer.{label}",
                record_id=item["meta"]["prompt_id"],
            )
            return {
                "prompt_id": item["meta"]["prompt_id"],
                "model": role.model,
                "text": response.text,
            }

        results = await gather_bounded([answer(item) for item in eval_items])

    answers: list[dict[str, Any]] = []
    for result in results:
        if isinstance(result, LocalEndpointUnavailable):
            raise RuntimeError(str(result)) from None
        if isinstance(result, Exception):
            logger.error("eval answer failed: %s", result)
            continue
        answers.append(result)
    return answers


def load_answers_file(path: Path) -> list[dict[str, Any]]:
    """Read answers written outside the pipeline: {prompt_id, prompt, model, text} rows."""
    answers = []
    for row in records.iter_jsonl(path):
        if "prompt_id" not in row or "text" not in row:
            raise RuntimeError(
                f"{path} rows must have 'prompt_id' and 'text'. Got keys: {sorted(row)}"
            )
        answers.append(
            {"prompt_id": row["prompt_id"], "model": row.get("model", path.stem), "text": row["text"]}
        )
    if not answers:
        raise RuntimeError(f"{path} contains no answers.")
    return answers


async def run_stage(
    config: RunConfig,
    spec: TargetSpec,
    run_dir: Path,
    endpoint_role: str | None = "base",
    label: str | None = None,
    answers_file: Path | None = None,
) -> dict[str, Any]:
    """Entry point for `main.py evaluate` in this repository.

    Two ways in: call a configured model role live over eval.jsonl, or judge an
    answers file produced elsewhere. Both write the same results shape.
    """
    from pipeline.export import EVAL_FILE

    if not (run_dir / EVAL_FILE).exists() or not list(records.iter_jsonl(run_dir / EVAL_FILE)):
        raise RuntimeError(
            f"No evaluation items in {run_dir / EVAL_FILE}. Run the export stage first."
        )

    if answers_file is not None:
        label = label or Path(answers_file).stem
        answers = load_answers_file(Path(answers_file))
    else:
        if not endpoint_role:
            raise RuntimeError("Pass either --endpoint-role or --answers-file.")
        label = label or endpoint_role
        answers = await answer_eval_set(config, run_dir, endpoint_role, label)

    out_path = results_path(run_dir, label)
    already = {row["prompt_id"]: row for row in records.iter_jsonl(out_path)}
    pending = [answer for answer in answers if answer["prompt_id"] not in already]
    rows = list(already.values())
    if pending:
        rows += await judge_answers(config, spec, run_dir, pending, label)
    if not rows:
        raise RuntimeError(
            f"No evaluation results for '{label}'. Check the errors above; nothing was "
            f"written to {out_path.name}."
        )
    records.write_jsonl(out_path, rows)
    logger.info("evaluate: wrote %d results to %s", len(rows), out_path)
    return _summarise(rows, label)


def _tally(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, int]]:
    buckets: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = buckets.setdefault(str(row.get(key) or "unknown"), {"n": 0, "pass": 0})
        bucket["n"] += 1
        bucket["pass"] += 1 if _passed(row) else 0
    return buckets


def _summarise(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    passed = sum(1 for row in rows if _passed(row))
    return {
        "endpoint": label,
        "n": len(rows),
        "pass": passed,
        "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
        "by_case_type": _tally(rows, "case_type"),
        # Reframed prompts are the transfer test: a model that only passes `base` has
        # learned the training situations, not the target.
        "by_variant": _tally(rows, "variant"),
    }


def _passed(row: dict[str, Any]) -> bool:
    return bool((row.get("judge") or {}).get("pass"))


def before_after_table(before_path: Path, after_path: Path) -> str:
    """Markdown comparison of two eval result files, plus the per-item flips."""
    before = {row["prompt_id"]: row for row in records.iter_jsonl(before_path)}
    after = {row["prompt_id"]: row for row in records.iter_jsonl(after_path)}
    shared = [pid for pid in before if pid in after]
    if not shared:
        return f"No prompts in common between {before_path.name} and {after_path.name}."

    def group(key: str) -> dict[str, list[str]]:
        buckets: dict[str, list[str]] = {}
        for prompt_id in shared:
            buckets.setdefault(str(before[prompt_id].get(key) or "unknown"), []).append(prompt_id)
        return buckets

    lines = [
        f"# Before / after on {len(shared)} evaluation prompts",
        "",
        f"before: `{before_path.name}` ({before[shared[0]].get('model','?')})",
        f"after:  `{after_path.name}` ({after[shared[0]].get('model','?')})",
        "",
        "| case type | n | before pass | after pass | change |",
        "|---|---|---|---|---|",
    ]
    for case_type, prompt_ids in sorted(group("case_type").items()):
        before_pass = sum(1 for pid in prompt_ids if _passed(before[pid]))
        after_pass = sum(1 for pid in prompt_ids if _passed(after[pid]))
        lines.append(
            f"| {case_type} | {len(prompt_ids)} | {before_pass} | {after_pass} "
            f"| {after_pass - before_pass:+d} |"
        )
    total_before = sum(1 for pid in shared if _passed(before[pid]))
    total_after = sum(1 for pid in shared if _passed(after[pid]))
    lines.append(
        f"| **all** | {len(shared)} | {total_before} | {total_after} "
        f"| {total_after - total_before:+d} |"
    )

    lines += [
        "",
        "By prompt variant. A gain confined to `base` is a gain on the training "
        "situations, not on the target.",
        "",
        "| variant | n | before pass | after pass | change |",
        "|---|---|---|---|---|",
    ]
    for variant, prompt_ids in sorted(group("variant").items()):
        before_pass = sum(1 for pid in prompt_ids if _passed(before[pid]))
        after_pass = sum(1 for pid in prompt_ids if _passed(after[pid]))
        lines.append(
            f"| {variant} | {len(prompt_ids)} | {before_pass} | {after_pass} "
            f"| {after_pass - before_pass:+d} |"
        )

    gained = [pid for pid in shared if _passed(after[pid]) and not before[pid].get("pass")]
    lost = [pid for pid in shared if _passed(before[pid]) and not after[pid].get("pass")]
    lines += ["", f"Newly passing: {len(gained)}", f"Newly failing: {len(lost)}", ""]
    for prompt_id in lost[:5]:
        lines.append(f"- regression `{prompt_id}`: {(after[prompt_id].get('judge') or {}).get('rationale','')[:200]}")
    return "\n".join(lines) + "\n"


def write_before_after(before_path: Path, after_path: Path, out_path: Path) -> str:
    table = before_after_table(before_path, after_path)
    out_path.write_text(table, encoding="utf-8")
    return table
