#!/usr/bin/env python
"""Evaluation CLI for llm_persona.

Build a held-out evaluation suite from a value specification, answer it with two or more
model arms under identical settings, grade every answer against frozen per-case rubrics, and
report the diagnostics separately instead of collapsing them into one score.

    python main.py author  --target confucian --out suites/confucian-v1.json
    python main.py inspect --suite suites/confucian-v1.json
    python main.py run     --suite suites/confucian-v1.json --arms base=base,adapter=adapter
    python main.py report  --run <run_id>

Stage by stage, if you want to drive it yourself:

    python main.py answer     --suite <path> --run <id> --arm base --role base
    python main.py judge      --run <id> --arm base
    python main.py capability --run <id> --arm base --role base

Arms are `label=config_role` pairs. Both arms in this project are the same Q4 base model
served by llama.cpp, one with the LoRA adapter applied, so `--arms base=base,adapter=adapter`
compares exactly one difference.

`evaluate` and `compare` are the older single-score commands, kept for the pipeline's own
eval.jsonl exports. New work should use the commands above; docs/DESIGN.md says why.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from persona_eval import PIPELINE_ROOT
from persona_eval import evaluate as legacy_evaluate
from persona_eval import runs
from persona_eval.run.answer import answer_cases, settings_disagreement
from persona_eval.run.judge import judge_cases, judge_changes
from persona_eval.suite import io as suite_io
from persona_eval.suite.author import SuitePlan, author_suite
from persona_eval.suite.contamination import (
    contamination_summary,
    lexical_contamination,
    load_training_prompts,
)
from persona_eval.suite.external import fetch_daily_dilemmas
from persona_eval.suite.review import apply_reviews, review_suite
from persona_eval.suite.schema import CaseResult, ChangeVerdict, Suite

from pipeline import records
from pipeline.config import ConfigError, load_config, resolve_run_dir
from pipeline.target import SpecError, load_target

logger = logging.getLogger("persona_eval.main")

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_TRAINING = PIPELINE_ROOT / "runs" / "confucian" / "pooled-confucian-v2" / "sft_train.jsonl"


# --------------------------------------------------------------------------- shared setup


def load_eval_config(path: str | Path):
    """Load an eval config. Paths resolve against this repo; the pipeline supplies keys."""
    config_path = Path(path)
    if not config_path.is_absolute():
        local = REPO_ROOT / config_path
        config_path = local if local.exists() else config_path
    return load_config(config_path, repo_root=PIPELINE_ROOT)


def load_spec(config, target: str, targets_dir: str | None = None):
    directory = Path(targets_dir) if targets_dir else config.targets_dir
    return load_target(directory, target, strict=bool(config.raw.get("strict_specs", False)))


def parse_arms(text: str) -> list[tuple[str, str]]:
    """`base=base,adapter=adapter` -> [(label, role), ...]. A bare name means label == role."""
    arms: list[tuple[str, str]] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        label, _, role = chunk.partition("=")
        arms.append((label.strip(), (role or label).strip()))
    if not arms:
        raise SystemExit("--arms needs at least one label=role pair")
    return arms


def training_prompts(path: str | Path | None) -> list[str]:
    source = Path(path) if path else DEFAULT_TRAINING
    if not source.is_file():
        logger.warning("training prompts not found at %s; contamination check will be empty", source)
        return []
    return load_training_prompts(source)


# -------------------------------------------------------------------------------- author


async def cmd_author(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    spec = load_spec(config, args.target, args.targets_dir)
    out = Path(args.out) if args.out else runs.SUITES_DIR / f"{args.target}-{runs.new_run_id()}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    usage_path = out.parent / f"{out.stem}.usage.jsonl"

    overrides: dict[str, Any] = {}
    if args.external:
        overrides["external_situations"] = args.external
    plan = SuitePlan.default(spec, scale=args.scale, **overrides)
    for kind_count in args.families or []:
        kind, _, count = kind_count.partition("=")
        plan.families_per_kind[kind.strip()] = int(count)
    problems = plan.validate()
    if problems:
        raise SystemExit("plan problems:\n  - " + "\n  - ".join(problems))
    logger.info(
        "authoring %s: %s families, %d cases planned",
        args.target,
        sum(plan.families_per_kind.values()),
        sum(
            count * sum(len(cell.variants) for cell in plan.coverage_for(kind))
            for kind, count in plan.families_per_kind.items()
        ),
    )

    external = ()
    if plan.external_situations:
        external = await fetch_daily_dilemmas(limit=plan.external_situations * 3)
        logger.info("fetched %d external situations", len(external))

    known = training_prompts(args.training_prompts)
    suite, report = await author_suite(
        config, spec, plan, usage_path=usage_path, external=external, training_prompts=known,
        situations_role=args.situations_role, rubric_role=args.rubric_role,
    )

    if not args.no_review:
        reviews = await review_suite(config, spec, suite, reviewer_role=args.review_role, usage_path=usage_path)
        suite, review_report = apply_reviews(suite, reviews)
        report["rubric_review"] = review_report
        runs.write_jsonl(out.parent / f"{out.stem}.reviews.jsonl", reviews)
        logger.info(
            "rubric review: kept %d, dropped %d (%s)",
            review_report["kept"],
            review_report["dropped"],
            review_report["blocking_defect_counts"] or "no blocking defects",
        )

    problems = suite.validate()
    if problems:
        logger.error("authored suite does not validate:\n  - %s", "\n  - ".join(problems[:20]))
        return 2

    contamination = lexical_contamination(suite, known) if known else []
    report["contamination"] = contamination_summary(contamination) if contamination else {"checked": 0}
    suite_io.save_suite(suite, out)
    runs.write_json(out.parent / f"{out.stem}.authoring.json", report)
    if contamination:
        runs.write_jsonl(out.parent / f"{out.stem}.contamination.jsonl", contamination)

    summary = suite_io.suite_summary(suite)
    print(json.dumps({"suite": str(out), **summary}, indent=2)[:2000])
    print(f"\nauthoring report: {out.parent / (out.stem + '.authoring.json')}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Validate a suite and check it against training data. No model calls, no cost."""
    suite = suite_io.load_suite(args.suite)
    problems = suite.validate()
    summary = suite_io.suite_summary(suite)
    known = training_prompts(args.training_prompts)
    contamination = lexical_contamination(suite, known) if known else []
    print(json.dumps(summary, indent=2))
    if contamination:
        print("\ncontamination vs training data:")
        print(json.dumps(contamination_summary(contamination), indent=2))
        worst = sorted(contamination, key=lambda r: -r.get("max_similarity", 0))[:5]
        for row in worst:
            print(f"  {row.get('case_id')}: max_similarity={row.get('max_similarity'):.3f}")
    if problems:
        print(f"\n{len(problems)} validation problems:")
        for problem in problems[:20]:
            print(f"  - {problem}")
        return 2
    print("\nsuite validates.")
    return 0


# ------------------------------------------------------------------------------- running


async def cmd_answer(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    suite = suite_io.load_suite(args.suite)
    directory = runs.run_dir(args.run or runs.new_run_id())
    runs.configure_logging(directory, args.verbose)
    suite_io.save_suite(suite, directory / runs.SUITE_COPY)

    answers = await answer_cases(
        config,
        suite,
        suite.cases,
        endpoint_role=args.role,
        arm=args.arm,
        usage_path=directory / runs.USAGE_FILE,
        limit=args.limit,
        **_self_consistency(args),
    )
    path = runs.answers_path(directory, args.arm)
    runs.write_jsonl(path, answers)
    signature = runs.sampling_signature(config, args.role)
    runs.record_stage(directory, f"answer_{args.arm}", {"n": len(answers), "settings": signature, "path": str(path)})
    print(f"{len(answers)} answers -> {path}")
    return 0


async def cmd_judge(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    directory = runs.run_dir(args.run)
    runs.configure_logging(directory, args.verbose)
    suite = suite_io.load_suite(directory / runs.SUITE_COPY)
    spec = load_spec(config, args.target, args.targets_dir)
    answers = runs.read_jsonl(runs.answers_path(directory, args.arm))
    if not answers:
        raise SystemExit(f"no answers for arm {args.arm!r} in {directory}")

    results = await judge_cases(
        config,
        spec,
        suite,
        answers,
        judge_role_name=args.judge_role,
        usage_path=directory / runs.USAGE_FILE,
        repeat_fraction=args.repeat_fraction,
        second_judge_role_name=args.second_judge_role,
        second_judge_fraction=args.second_judge_fraction,
    )
    runs.write_jsonl(runs.results_path(directory, args.arm), results)

    verdicts = await judge_changes(
        config, spec, suite, answers,
        judge_role_name=args.judge_role,
        usage_path=directory / runs.USAGE_FILE,
        **_change_kwargs(args),
    )
    runs.write_jsonl(runs.changes_path(directory, args.arm), verdicts)
    runs.record_stage(
        directory,
        f"judge_{args.arm}",
        {"results": len(results), "verdicts": len(verdicts), "judge_role": args.judge_role},
    )
    print(f"{len(results)} graded, {len(verdicts)} change verdicts -> {directory}")
    return 0


async def cmd_capability(args: argparse.Namespace) -> int:
    from persona_eval.capability.checks import run_capability, summarise

    config = load_eval_config(args.config)
    directory = runs.run_dir(args.run)
    runs.configure_logging(directory, args.verbose)
    rows = await run_capability(
        config, endpoint_role=args.role, arm=args.arm, usage_path=directory / runs.USAGE_FILE, limit=args.limit
    )
    runs.write_jsonl(runs.capability_path(directory, args.arm), rows)
    summary = summarise(rows)
    runs.record_stage(directory, f"capability_{args.arm}", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _self_consistency(args: argparse.Namespace) -> dict[str, Any]:
    """The noise-floor probe, passed only when the installed answerer accepts it.

    A value of 1 or more is a count of cases; a value below 1 is a fraction of the eligible
    originals. Without this, an invariance rate has nothing to be read against: two answers
    to a paraphrase are two draws from a model at temperature 0.7, and the probe measures how
    often a position moves when the question did not change at all.
    """
    import inspect

    value = getattr(args, "self_consistency", 0.0) or 0.0
    if not value:
        return {}
    if "self_consistency" not in set(inspect.signature(answer_cases).parameters):
        logger.warning("answer_cases does not accept self_consistency in this build; no noise floor will be measured")
        return {}
    return {"self_consistency": int(value) if value >= 1 else float(value)}


def _change_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Optional judge_changes controls, passed only when the installed signature accepts them.

    The repeat pass is opt-in and separate from `--repeat-fraction`, which governs dimension
    scores. They are deliberately not the same knob: repeat change verdicts are only safe to
    request once the analysis splits verdicts on judge_pass, and until then a repeat would
    inflate every behaviour rate rather than measure its stability.
    """
    import inspect

    wanted = {"repeat_fraction": getattr(args, "change_repeat_fraction", 0.0) or 0.0}
    accepted = set(inspect.signature(judge_changes).parameters)
    passed = {k: v for k, v in wanted.items() if k in accepted and v}
    for key, value in wanted.items():
        if value and key not in accepted:
            logger.warning("judge_changes does not accept %s=%s in this build; ignored", key, value)
    return passed


def _load_results(directory: Path, arm: str) -> list[CaseResult]:
    return [CaseResult.from_dict(row) for row in runs.read_jsonl(runs.results_path(directory, arm))]


def _load_verdicts(directory: Path, arm: str) -> list[ChangeVerdict]:
    return [ChangeVerdict.from_dict(row) for row in runs.read_jsonl(runs.changes_path(directory, arm))]


def cmd_report(args: argparse.Namespace) -> int:
    # Imported here, not at module scope. Answering is the expensive, unrepeatable part of a
    # run; rendering is cheap and can be redone at any time. A half-written reporting module
    # must not be able to stop a run before it generates a single answer.
    from persona_eval.report.aggregate import analyse
    from persona_eval.report.render import render_report

    config = load_eval_config(args.config)
    directory = runs.run_dir(args.run)
    suite = suite_io.load_suite(directory / runs.SUITE_COPY)
    arms = runs.arms_present(directory)
    if not arms:
        raise SystemExit(f"no graded arms in {directory}")

    results: list[CaseResult] = []
    verdicts: list[ChangeVerdict] = []
    capability: list[dict[str, Any]] = []
    for arm in arms:
        results += _load_results(directory, arm)
        verdicts += _load_verdicts(directory, arm)
        capability += runs.read_jsonl(runs.capability_path(directory, arm))

    analysis = analyse(
        suite,
        results,
        verdicts,
        capability=capability,
        baseline_arm=args.baseline if args.baseline in arms else arms[0],
        curator_judge=args.curator_judge,
    )
    record = runs.read_record(directory)
    signatures = [
        stage.get("settings")
        for name, stage in (record.get("stages") or {}).items()
        if name.startswith("answer_") and stage.get("settings")
    ]
    matched, mismatches = runs.settings_match(signatures)
    # The record says what the run intended; the answers say what it did. `settings_disagreement`
    # reads the fingerprint stamped on each answer, so it catches a run whose record and
    # behaviour parted company. Both are reported; either one failing invalidates a comparison.
    observed: list[str] = []
    for arm in arms:
        observed += [f"{arm}: {problem}" for problem in settings_disagreement(runs.read_jsonl(runs.answers_path(directory, arm)))]
    if observed:
        matched = False
        mismatches = mismatches + observed
    meta = {
        "run_id": args.run,
        "run_dir": str(directory),
        "arms": arms,
        "baseline": args.baseline if args.baseline in arms else arms[0],
        "settings_matched": matched,
        "settings_mismatches": mismatches,
        "sampling": signatures,
        "suite_version": suite.version,
        "spec_version": suite.spec_version,
        "config": str(config.path),
        "authoring": suite.authoring,
    }
    runs.write_json(directory / runs.ANALYSIS_FILE, analysis)
    text = render_report(analysis, suite, meta)
    (directory / runs.REPORT_FILE).write_text(text)
    runs.record_stage(directory, "report", {"arms": arms, "settings_matched": matched})
    print(f"report -> {directory / runs.REPORT_FILE}")
    if not matched:
        print("WARNING: arms did not share generation settings: " + "; ".join(mismatches))
    return 0


async def cmd_form_control(args: argparse.Namespace) -> int:
    """Measure what the tuned model's answer shape is worth, with substance held constant.

    The judging prompt tells the judge to credit use rather than mention. Nothing else checks
    that it obeyed, and the check that does exist, quote verification, is EASIER for an arm
    whose template hands the judge a quotable sentence per rubric item. So this recasts
    base-arm answers into that template without changing what they say, grades them blind
    through the same path, and reports the gap. That gap is how much of any measured
    advantage the shape alone explains.
    """
    from persona_eval.run.form_control import measure_form_premium

    config = load_eval_config(args.config)
    directory = runs.run_dir(args.run)
    runs.configure_logging(directory, args.verbose)
    suite = suite_io.load_suite(directory / runs.SUITE_COPY)
    spec = load_spec(config, args.target, args.targets_dir)
    results = _load_results(directory, args.arm)
    if not results:
        raise SystemExit(f"no graded results for arm {args.arm!r} in {directory}")

    premium = await measure_form_premium(
        config, spec, suite, results,
        n=args.n, arm=args.arm,
        rewriter_role_name=args.rewriter_role,
        judge_role_name=args.judge_role,
        usage_path=directory / runs.USAGE_FILE,
    )
    payload = premium.to_dict() if hasattr(premium, "to_dict") else premium
    runs.write_json(directory / "form_premium.json", payload)
    runs.record_stage(directory, "form_control", {"arm": args.arm, "n": args.n})
    print(json.dumps(payload, indent=2, default=str)[:2500])
    return 0


async def cmd_rival(args: argparse.Namespace) -> int:
    """Bound the author effect: re-grade the same answers against an independently written standard.

    The two-judge audit measures judge contamination and is blind to author contamination,
    which is the larger channel: one model wrote every situation, every rubric and every
    anchor, and also decided which training rows survived review. Because both judges read
    the same rubric, that taste cancels out of the comparison by construction. Here a model
    that never saw the original standard writes a rival one from the same specification, and
    the answers already collected are graded again against it. If the arms' gap is the same
    size under both standards, the standard was not doing the work.
    """
    from persona_eval.suite.rival import (
        author_rival_rubrics,
        divergence_summary,
        rival_suite,
        select_rival_families,
    )

    config = load_eval_config(args.config)
    directory = runs.run_dir(args.run)
    runs.configure_logging(directory, args.verbose)
    suite = suite_io.load_suite(directory / runs.SUITE_COPY)
    spec = load_spec(config, args.target, args.targets_dir)
    usage_path = directory / runs.USAGE_FILE

    family_ids = select_rival_families(suite, fraction=args.fraction)
    logger.info("rivalling %d of %d families: %s", len(family_ids), len(suite.families), ", ".join(family_ids))
    rivals, rival_report = await author_rival_rubrics(
        config, spec, suite, family_ids, role_name=args.rival_role, usage_path=usage_path
    )
    if not rivals:
        raise SystemExit("no rival rubrics survived authoring; nothing to compare")
    rivalled = rival_suite(suite, rivals)
    rivalled_ids = set(rival_report.get("rivalled_case_ids") or [c.case_id for c in rivals])
    suite_io.save_suite(rivalled, runs.rival_dir(directory) / runs.SUITE_COPY)

    # Grade the answers we already have against the rival standard. Nothing else moves: same
    # answers, same judge, same dimensions, only the standard differs.
    per_arm: dict[str, Any] = {}
    for arm in runs.arms_present(directory):
        answers = [a for a in runs.read_jsonl(runs.answers_path(directory, arm)) if a.get("case_id") in rivalled_ids]
        if not answers:
            continue
        results = await judge_cases(
            config, spec, rivalled, answers, judge_role_name=args.judge_role, usage_path=usage_path
        )
        # Stamp the standard these scores were produced against. Without it the report cannot
        # tell a rival-graded score from an original-graded one, and the two would pool into
        # a single mean, averaging away the very comparison this command exists to make.
        for result in results:
            result.standard = "rival"
        runs.write_jsonl(runs.results_path(runs.rival_dir(directory), arm), results)
        per_arm[arm] = len(results)

    rival_report["divergence_summary"] = divergence_summary(rival_report.get("divergence") or [])
    rival_report["rejudged"] = per_arm
    runs.write_json(runs.rival_dir(directory) / "rival_report.json", rival_report)
    runs.record_stage(directory, "rival", {"families": len(family_ids), "cases": len(rivalled_ids), "rejudged": per_arm})
    print(json.dumps({"families": family_ids, "cases": len(rivalled_ids), "rejudged": per_arm,
                      "divergence": rival_report["divergence_summary"]}, indent=2, default=str)[:1500])
    return 0


async def cmd_run(args: argparse.Namespace) -> int:
    """Everything, in order, one arm at a time.

    Arms run sequentially on purpose: both local endpoints are llama.cpp servers on the same
    eight cores, so answering two arms at once halves each one's throughput and buys nothing.
    """
    config = load_eval_config(args.config)
    suite = suite_io.load_suite(args.suite)
    spec = load_spec(config, args.target, args.targets_dir)
    arms = parse_arms(args.arms)
    run_id = args.run or runs.new_run_id()
    directory = runs.run_dir(run_id)
    runs.configure_logging(directory, args.verbose)
    suite_io.save_suite(suite, directory / runs.SUITE_COPY)
    usage_path = directory / runs.USAGE_FILE
    runs.write_record(
        directory,
        {
            "run_id": run_id,
            "suite": str(args.suite),
            "suite_version": suite.version,
            "target": args.target,
            "arms": [label for label, _ in arms],
            "config": str(config.path),
            "started_utc": runs.now_utc(),
        },
    )
    logger.info("run %s | suite %s | %d cases | arms %s", run_id, suite.suite_id, len(suite.cases), arms)

    for label, role in arms:
        logger.info("---- arm %s (role %s): answering %d cases", label, role, len(suite.cases))
        answers = await answer_cases(
            config, suite, suite.cases, endpoint_role=role, arm=label, usage_path=usage_path,
            limit=args.limit, **_self_consistency(args),
        )
        runs.write_jsonl(runs.answers_path(directory, label), answers)
        runs.record_stage(
            directory,
            f"answer_{label}",
            {"n": len(answers), "settings": runs.sampling_signature(config, role),
             "disagreements": settings_disagreement(answers)},
        )
        if not args.skip_capability:
            from persona_eval.capability.checks import run_capability, summarise

            logger.info("---- arm %s: capability checks", label)
            rows = await run_capability(config, endpoint_role=role, arm=label, usage_path=usage_path)
            runs.write_jsonl(runs.capability_path(directory, label), rows)
            runs.record_stage(directory, f"capability_{label}", summarise(rows))

    for label, _ in arms:
        answers = runs.read_jsonl(runs.answers_path(directory, label))
        logger.info("---- arm %s: grading %d answers", label, len(answers))
        results = await judge_cases(
            config, spec, suite, answers,
            judge_role_name=args.judge_role,
            usage_path=usage_path,
            repeat_fraction=args.repeat_fraction,
            second_judge_role_name=args.second_judge_role,
            second_judge_fraction=args.second_judge_fraction,
        )
        runs.write_jsonl(runs.results_path(directory, label), results)
        verdicts = await judge_changes(
            config, spec, suite, answers,
            judge_role_name=args.judge_role,
            usage_path=usage_path,
            **_change_kwargs(args),
        )
        runs.write_jsonl(runs.changes_path(directory, label), verdicts)
        runs.record_stage(directory, f"judge_{label}", {"results": len(results), "verdicts": len(verdicts)})

    report_args = argparse.Namespace(
        config=args.config, run=run_id, baseline=arms[0][0], curator_judge=args.curator_judge, verbose=args.verbose
    )
    cmd_report(report_args)
    return 0


# -------------------------------------------------------------------------------- legacy


async def run_legacy(args: argparse.Namespace) -> int:
    config = load_config(args.config, repo_root=PIPELINE_ROOT)
    targets_dir = Path(args.targets_dir) if args.targets_dir else config.targets_dir
    spec = load_target(targets_dir, args.target, strict=bool(config.raw.get("strict_specs")))
    run_dir = resolve_run_dir(config, args.target, args.run)
    runs.configure_logging(run_dir, args.verbose)
    if args.command == "evaluate":
        summary = await legacy_evaluate.run_stage(
            config, spec, run_dir,
            endpoint_role=None if args.answers_file else args.endpoint_role,
            label=args.label,
            answers_file=Path(args.answers_file) if args.answers_file else None,
        )
        print(json.dumps(summary, indent=2))
        return 0
    legacy_evaluate.write_before_after(run_dir, Path(args.before), Path(args.after))
    return 0


# ----------------------------------------------------------------------------------- cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, target=True):
        p.add_argument("--config", default=str(DEFAULT_CONFIG))
        if target:
            p.add_argument("--target", default="confucian")
            p.add_argument("--targets-dir", default=None)
        p.add_argument("-v", "--verbose", action="store_true")
        return p

    p = common(sub.add_parser("author", help="write a held-out evaluation suite from a value specification"))
    p.add_argument("--out", default=None)
    p.add_argument("--scale", type=int, default=1, help="whole multiples of the default family counts")
    p.add_argument("--families", action="append", default=None, metavar="KIND=N", help="override one family count")
    p.add_argument("--external", type=int, default=0, help="families seeded from a public dataset")
    p.add_argument("--training-prompts", default=None)
    p.add_argument("--no-review", action="store_true", help="skip the rubric audit (not recommended)")
    p.add_argument("--review-role", default="judge")
    p.add_argument("--situations-role", default="author_situations", help="model role that invents situations; runs warm")
    p.add_argument("--rubric-role", default="author_rubric", help="model role that writes rubrics; runs cold")

    p = common(sub.add_parser("inspect", help="validate a suite and check it against training data"), target=False)
    p.add_argument("--suite", required=True)
    p.add_argument("--training-prompts", default=None)

    p = common(sub.add_parser("answer", help="answer a suite with one arm"))
    p.add_argument("--suite", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--arm", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--self-consistency", type=float, default=0.0, help="answer this many originals twice (>=1 a count, <1 a fraction) for a noise floor")

    p = common(sub.add_parser("judge", help="grade one arm's answers and judge its variant changes"))
    p.add_argument("--run", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--judge-role", default="judge")
    p.add_argument("--second-judge-role", default=None)
    p.add_argument("--second-judge-fraction", type=float, default=1.0)
    p.add_argument("--repeat-fraction", type=float, default=0.0)
    p.add_argument("--change-repeat-fraction", type=float, default=0.0, help="re-judge a sample of change verdicts to measure their stability")
    p.add_argument("--self-consistency", type=float, default=0.0, help="answer this fraction of originals twice to establish a noise floor")

    p = common(sub.add_parser("capability", help="run the general-capability checks for one arm"))
    p.add_argument("--run", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--role", required=True)
    p.add_argument("--limit", type=int, default=None)

    p = common(sub.add_parser("report", help="analyse a run and render its report"), target=False)
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", default="base")
    p.add_argument("--curator-judge", default=None, help="judge model that also curated the training data")

    p = common(sub.add_parser("run", help="answer, grade and report every arm, in order"))
    p.add_argument("--suite", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--arms", default="base=base,adapter=adapter")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--judge-role", default="judge")
    p.add_argument("--second-judge-role", default=None)
    p.add_argument("--second-judge-fraction", type=float, default=1.0)
    p.add_argument("--repeat-fraction", type=float, default=0.0)
    p.add_argument("--change-repeat-fraction", type=float, default=0.0, help="re-judge a sample of change verdicts to measure their stability")
    p.add_argument("--self-consistency", type=float, default=0.0, help="answer this fraction of originals twice to establish a noise floor")
    p.add_argument("--curator-judge", default=None)
    p.add_argument("--skip-capability", action="store_true")

    p = common(sub.add_parser("form-control", help="measure what the tuned model's answer shape is worth on its own"))
    p.add_argument("--run", required=True)
    p.add_argument("--arm", default="base", help="the arm whose answers are recast; use the untuned one")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--rewriter-role", default="judge_second")
    p.add_argument("--judge-role", default="judge")

    p = common(sub.add_parser("rival", help="bound the author effect with an independently written rubric"))
    p.add_argument("--run", required=True)
    p.add_argument("--fraction", type=float, default=1 / 3)
    p.add_argument("--rival-role", default="judge", help="model family that writes the rival standard; must differ from the author")
    p.add_argument("--judge-role", default="judge")

    for name in ("evaluate", "compare"):
        p = common(sub.add_parser(name, help="legacy: single-score judging of a pipeline run's eval.jsonl"))
        p.set_defaults(config="configs/pilot.yaml")
        p.add_argument("--run", default=None)
        p.add_argument("--endpoint-role", "--endpoint", dest="endpoint_role", default="base")
        p.add_argument("--answers-file", default=None)
        p.add_argument("--label", default=None)
        p.add_argument("--before", default=None)
        p.add_argument("--after", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "author":
            return asyncio.run(cmd_author(args))
        if args.command == "inspect":
            return cmd_inspect(args)
        if args.command == "answer":
            return asyncio.run(cmd_answer(args))
        if args.command == "judge":
            return asyncio.run(cmd_judge(args))
        if args.command == "capability":
            return asyncio.run(cmd_capability(args))
        if args.command == "report":
            return cmd_report(args)
        if args.command == "form-control":
            return asyncio.run(cmd_form_control(args))
        if args.command == "rival":
            return asyncio.run(cmd_rival(args))
        if args.command == "run":
            return asyncio.run(cmd_run(args))
        return asyncio.run(run_legacy(args))
    except (ConfigError, SpecError) as error:
        logger.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
