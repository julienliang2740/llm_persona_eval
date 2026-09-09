"""Tests for run directories and the comparability record. No network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from persona_eval.runs import (
    arms_present,
    case_seed,
    read_jsonl,
    read_record,
    record_stage,
    results_path,
    run_dir,
    sampling_signature,
    settings_match,
    write_jsonl,
    write_record,
)


@dataclass
class _Role:
    model: str
    base_url: str
    temperature: float
    max_tokens: int


class _Config:
    """Enough of RunConfig for the signature resolver, which reads `config.evaluation`."""

    def __init__(self, roles, evaluation):
        self._roles = roles
        self.raw = {"evaluation": evaluation}

    @property
    def evaluation(self):
        return self.raw.get("evaluation", {})

    def role(self, name):
        return self._roles[name]


def test_run_record_merges_across_stages(tmp_path: Path):
    directory = run_dir("r1", tmp_path)
    write_record(directory, {"suite_id": "s1"})
    record_stage(directory, "answer", {"arm": "base", "n": 10})
    record_stage(directory, "judge", {"arm": "base", "n": 10})
    record = read_record(directory)
    assert record["suite_id"] == "s1"
    assert set(record["stages"]) == {"answer", "judge"}
    assert record["stages"]["answer"]["n"] == 10
    assert "finished_utc" in record["stages"]["judge"]


def test_jsonl_round_trip_handles_dataclasses(tmp_path: Path):
    @dataclass
    class Row:
        a: int
        b: str

    path = tmp_path / "rows.jsonl"
    assert write_jsonl(path, [Row(1, "x"), {"a": 2, "b": "y"}]) == 2
    assert read_jsonl(path) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert read_jsonl(tmp_path / "missing.jsonl") == []


def test_arms_present_reads_the_directory(tmp_path: Path):
    directory = run_dir("r1", tmp_path)
    for arm in ("base", "adapter"):
        write_jsonl(results_path(directory, arm), [{"case_id": "c1"}])
    assert arms_present(directory) == ["adapter", "base"]


def test_settings_match_catches_an_unfair_comparison():
    # No `temperature` in the evaluation block, so each role's own temperature is used and a
    # divergent role is visible. This mirrors the real failure: the record must reflect what
    # answering resolves, not what the config nominally says.
    evaluation = {"top_p": 0.95, "seed": 13}
    config = _Config(
        {
            "base": _Role("qwen", "http://127.0.0.1:8080/v1", 0.7, 1000),
            "adapter": _Role("qwen", "http://127.0.0.1:8081/v1", 0.7, 1000),
            "hot": _Role("qwen", "http://127.0.0.1:8082/v1", 1.0, 1000),
        },
        evaluation,
    )
    fair = [sampling_signature(config, "base"), sampling_signature(config, "adapter")]
    ok, problems = settings_match(fair)
    assert ok and not problems
    # Different model ids are allowed; different sampling is not.
    unfair = [sampling_signature(config, "base"), sampling_signature(config, "hot")]
    ok, problems = settings_match(unfair)
    assert not ok and any("temperature" in p for p in problems)


def test_case_seed_is_stable_and_arm_independent():
    # The same case must draw the same seed on every arm and every run, or the comparison
    # inherits sampling noise it does not need.
    assert case_seed("case-1") == case_seed("case-1")
    assert case_seed("case-1") != case_seed("case-2")
    assert 0 <= case_seed("case-1") < 2**32
