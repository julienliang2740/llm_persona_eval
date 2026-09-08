#!/usr/bin/env python
"""Evaluation CLI.

    python main.py evaluate --target confucian --endpoint-role base --label before
    python main.py evaluate --target confucian --answers-file ../llm_persona_training/out/answers.jsonl --label after
    python main.py compare  --target confucian --before <eval_results_before.jsonl> --after <eval_results_after.jsonl>

Reads the pipeline run directory (runs/<target>/<run_id>/ in llm_persona_data_pipeline, the
latest run unless --run is given), judges answers against its eval.jsonl, and writes
eval_results_<label>.jsonl and before_after.md next to it, so results stay with the export
they were scored against. Model roles and the judge come from the pipeline config.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from persona_eval import PIPELINE_ROOT  # noqa: F401  (puts the pipeline on sys.path)
from persona_eval import evaluate

from pipeline import records
from pipeline.config import ConfigError, load_config, resolve_run_dir
from pipeline.target import SpecError, load_target


def configure_logging(run_dir: Path, verbose: bool) -> None:
    """Log to stdout and append to the run's log.txt, as the pipeline stages do."""
    level = logging.DEBUG if verbose else logging.INFO
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(run_dir / records.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    for noisy in ("httpx", "httpcore", "httpcore.http11", "httpcore.connection"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description="Evaluate a run's held-out set.")
    parser.add_argument("command", choices=("evaluate", "compare"))
    parser.add_argument("--target", required=True, help="target id, a directory under the pipeline's targets/")
    parser.add_argument("--config", default="configs/pilot.yaml", help="pipeline config, relative to the pipeline repo")
    parser.add_argument("--run", default=None, help="run id; defaults to the latest run for this target")
    parser.add_argument("--targets-dir", default=None, help="override where targets/ is read from")
    parser.add_argument(
        "--endpoint-role",
        "--endpoint",
        dest="endpoint_role",
        default="base",
        help="evaluate: which configured model role to run over eval.jsonl",
    )
    parser.add_argument(
        "--answers-file",
        default=None,
        help="evaluate: judge answers generated elsewhere instead of calling a model. "
        "Rows are {prompt_id, prompt, model, text}.",
    )
    parser.add_argument("--label", default=None, help="evaluate: name for this result file")
    parser.add_argument("--before", default=None, help="compare: earlier eval_results_*.jsonl")
    parser.add_argument("--after", default=None, help="compare: later eval_results_*.jsonl")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    targets_dir = Path(args.targets_dir) if args.targets_dir else config.targets_dir
    run_dir = resolve_run_dir(config, args.target, args.run)
    configure_logging(run_dir, args.verbose)

    if args.command == "compare":
        if not (args.before and args.after):
            print("error: compare needs --before and --after", file=sys.stderr)
            return 2
        table = evaluate.write_before_after(
            Path(args.before), Path(args.after), run_dir / "before_after.md"
        )
        print(table)
        return 0

    spec = load_target(targets_dir, args.target, strict=bool(config.raw.get("strict_specs")))
    summary = await evaluate.run_stage(
        config,
        spec,
        run_dir,
        args.endpoint_role,
        args.label,
        Path(args.answers_file) if args.answers_file else None,
    )
    print(json.dumps({"run_dir": str(run_dir), "evaluate": summary}, indent=2, default=str))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(run(args))
    except (ConfigError, SpecError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
