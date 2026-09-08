"""Evaluate's two input modes and the shared results shape. No network."""

from __future__ import annotations

import asyncio

import pytest

from pipeline import records
from persona_eval.evaluate import _passed, _summarise, before_after_table, load_answers_file, run_stage
from pipeline.records import write_jsonl


def write_eval_set(run_dir):
    write_jsonl(
        run_dir / "eval.jsonl",
        [
            {
                "prompt": "What should I do about the late supplier?",
                "case_type": "divergence",
                "family_id": "fam_1",
                "variant": "base",
                "expected_behavior": "Name the unknown and act.",
                "pass_fail_notes": ["PASS if it names the unknown."],
                "reference_answer": "Call them today.",
                "meta": {"prompt_id": "pr_1"},
            }
        ],
    )


def test_answers_file_is_read_into_the_common_shape(tmp_path):
    path = tmp_path / "answers.jsonl"
    write_jsonl(
        path,
        [{"prompt_id": "pr_1", "prompt": "q", "model": "tuned-adapter", "text": "an answer"}],
    )
    answers = load_answers_file(path)
    assert answers == [{"prompt_id": "pr_1", "model": "tuned-adapter", "text": "an answer"}]


def test_answers_file_without_a_model_falls_back_to_the_file_name(tmp_path):
    path = tmp_path / "after_lora.jsonl"
    write_jsonl(path, [{"prompt_id": "pr_1", "text": "an answer"}])
    assert load_answers_file(path)[0]["model"] == "after_lora"


def test_a_malformed_answers_file_names_the_missing_keys(tmp_path):
    path = tmp_path / "answers.jsonl"
    write_jsonl(path, [{"id": "pr_1", "answer": "wrong keys"}])
    with pytest.raises(RuntimeError) as error:
        load_answers_file(path)
    assert "prompt_id" in str(error.value)


def test_an_empty_answers_file_is_an_error(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError):
        load_answers_file(path)


def test_evaluate_refuses_without_an_exported_eval_set(tmp_path, pilot_config, toy_spec):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(RuntimeError) as error:
        asyncio.run(run_stage(pilot_config, toy_spec, run_dir, "base", None, None))
    assert "export" in str(error.value)


def test_pass_is_read_from_the_nested_judge_verdict():
    assert _passed({"judge": {"pass": True}}) is True
    assert _passed({"judge": {"pass": False}}) is False
    assert _passed({}) is False


def test_summary_groups_by_case_type():
    rows = [
        {"case_type": "divergence", "judge": {"pass": True}},
        {"case_type": "divergence", "judge": {"pass": False}},
        {"case_type": "ordinary", "judge": {"pass": True}},
    ]
    summary = _summarise(rows, "after")
    assert summary["n"] == 3 and summary["pass"] == 2
    assert summary["by_case_type"]["divergence"] == {"n": 2, "pass": 1}


def test_before_after_works_across_a_live_run_and_an_answers_file(tmp_path):
    """A live 'before' and an offline 'after' must compare cleanly."""
    before = tmp_path / "eval_results_before.jsonl"
    after = tmp_path / "eval_results_after.jsonl"
    common = {"family_id": "f1", "case_type": "divergence", "variant": "base", "text": "x"}
    write_jsonl(
        before,
        [dict(common, prompt_id="pr_1", model="qwen2.5-7b-instruct", judge={"pass": False})],
    )
    write_jsonl(
        after,
        [dict(common, prompt_id="pr_1", model="lora-adapter", judge={"pass": True})],
    )
    table = before_after_table(before, after)
    assert "| divergence | 1 | 0 | 1 | +1 |" in table
    assert "qwen2.5-7b-instruct" in table and "lora-adapter" in table


def test_before_after_table(tmp_path):
    before = tmp_path / "eval_results_before.jsonl"
    after = tmp_path / "eval_results_after.jsonl"
    def row(prompt_id, case_type, passed, model="base", rationale=""):
        return {
            "prompt_id": prompt_id,
            "family_id": "f" + prompt_id,
            "case_type": case_type,
            "variant": "base",
            "model": model,
            "text": "an answer",
            "judge": {"pass": passed, "rationale": rationale},
        }

    write_jsonl(before, [row("p1", "divergence", False), row("p2", "ordinary", True)])
    write_jsonl(
        after,
        [
            row("p1", "divergence", True, "tuned"),
            row("p2", "ordinary", False, "tuned", "lost the point"),
        ],
    )
    table = before_after_table(before, after)
    assert "| divergence | 1 | 0 | 1 | +1 |" in table
    assert "| ordinary | 1 | 1 | 0 | -1 |" in table
    assert "Newly passing: 1" in table
    assert "regression `p2`" in table


def test_before_after_with_no_shared_prompts(tmp_path):
    before = tmp_path / "a.jsonl"
    after = tmp_path / "b.jsonl"
    write_jsonl(before, [{"prompt_id": "p1", "case_type": "ordinary", "judge": {"pass": True}}])
    write_jsonl(after, [{"prompt_id": "p9", "case_type": "ordinary", "judge": {"pass": True}}])
    assert "No prompts in common" in before_after_table(before, after)


def test_before_after_breaks_the_change_down_by_prompt_variant(tmp_path):
    """A gain confined to `base` is a gain on the training situations, not the target."""
    before = tmp_path / "eval_results_before.jsonl"
    after = tmp_path / "eval_results_after.jsonl"

    def row(prompt_id, variant, passed):
        return {
            "prompt_id": prompt_id,
            "family_id": "f" + prompt_id,
            "case_type": "ordinary",
            "variant": variant,
            "model": "m",
            "text": "an answer",
            "judge": {"pass": passed},
        }

    write_jsonl(before, [row("p1", "base", False), row("p2", "fiction", False)])
    write_jsonl(after, [row("p1", "base", True), row("p2", "fiction", False)])
    table = before_after_table(before, after)
    assert "| base | 1 | 0 | 1 | +1 |" in table
    assert "| fiction | 1 | 0 | 0 | +0 |" in table
