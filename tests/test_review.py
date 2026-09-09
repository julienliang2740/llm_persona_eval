"""Tests for the rubric audit. No network: the reviewer client is stubbed."""

from __future__ import annotations

import asyncio

import pytest

from persona_eval.suite.review import (
    CaseReview,
    Defect,
    _parse_review,
    apply_reviews,
    build_review_messages,
)
from dataclasses import replace

from persona_eval.suite.schema import (
    Anchor,
    Case,
    ChangeExpectation,
    Family,
    Rubric,
    Suite,
)


def _rubric(dimensions=("salience", "roles_relationships", "conflict_recognition")) -> Rubric:
    return Rubric(
        dimensions=tuple(dimensions),
        must_notice=("the tenant has no heating and a child is ill",),
        must_not_infer=("that the agent may ignore the letting agreement entirely",),
        acceptable_outputs=("authorise the repair and answer for it afterwards",),
        unacceptable_reasoning=("that rules never bind when someone is uncomfortable",),
        anchors=tuple(
            Anchor(dimension=d, score_0="misses it", score_1="names it vaguely", score_2="names it and uses it")
            for d in dimensions
        ),
    )


def _suite(cases) -> Suite:
    family = Family(
        family_id="fam1",
        kind="standard",
        title="Heating",
        situation="A letting agent finds a tenant without heating for three weeks.",
        domain="work",
        provenance="authored",
        held_out_rationale="written for this suite after training data was frozen",
        principles_in_play=("CM01",),
    )
    return Suite(
        suite_id="s1",
        target_id="confucian",
        spec_version="0.1",
        spec_hash="abc",
        created_utc="2026-09-09T00:00:00Z",
        families=(family,),
        cases=tuple(cases),
    )


def _case(case_id: str, variant: str = "original", task: str = "notice") -> Case:
    change = (
        None
        if variant == "original"
        else ChangeExpectation(
            what_changed="the tenant's flat is repainted",
            should_change=variant in {"relevant_change", "correction"},
            expected_direction="no movement",
            justification="paint carries no weight on a heating failure",
        )
    )
    return Case(
        case_id=case_id,
        family_id="fam1",
        task=task,
        variant=variant,
        turns=("What matters most here?",),
        rubric=_rubric(),
        change_expectation=change,
    )


def test_review_prompt_contains_the_rubric_and_not_a_model_name():
    suite = _suite([_case("c1")])
    messages = build_review_messages("SPEC TEXT", suite, suite.case("c1"))
    content = messages[0]["content"]
    assert "SPEC TEXT" in content
    assert "the tenant has no heating and a child is ill" in content
    assert "0 = misses it" in content and "2 = names it and uses it" in content
    # The reviewer audits a standard, so it must never learn which model will be graded.
    for leak in ("adapter", "base model", "qwen", "fine-tune"):
        assert leak not in content.lower()


def test_review_prompt_states_the_variant_claim_for_auditing():
    suite = _suite([_case("c1"), _case("c2", variant="irrelevant_change")])
    content = build_review_messages("SPEC", suite, suite.case("c2"))[0]["content"]
    assert "SHOULD NOT move" in content
    assert "paint carries no weight" in content


def test_parse_review_normalises_and_keeps_unknown_defect_types():
    payload = {
        "verdict": "REVISE",
        "defects": [
            {"type": "form_over_substance", "severity": "blocking", "detail": "d", "suggested_fix": "f"},
            {"type": "something_new", "severity": "nonsense", "detail": "d2"},
        ],
        "strongest_point": "tests role conflict",
    }
    review = _parse_review("c1", payload, "kimi")
    assert review.verdict == "revise"
    assert review.should_drop  # a blocking defect drops the case even without verdict=drop
    assert review.defects[1].type == "something_new"
    assert review.defects[1].severity == "concern"  # unknown severity falls back, not crashes


def test_parse_review_survives_garbage():
    review = _parse_review("c1", "not a dict", "kimi")
    assert review.verdict == "revise" and review.error


def test_apply_reviews_drops_blocking_and_keeps_concerns():
    suite = _suite([_case("c1"), _case("c2", task="decide")])
    reviews = [
        CaseReview(case_id="c1", verdict="accept", defects=[Defect("vague_must_notice", "concern", "generic")]),
        CaseReview(case_id="c2", verdict="accept", defects=[Defect("invented_principle", "blocking", "not in spec")]),
    ]
    kept, report = apply_reviews(suite, reviews)
    assert [c.case_id for c in kept.cases] == ["c1"]
    assert report["dropped"] == 1 and report["kept"] == 1
    assert report["blocking_defect_counts"] == {"invented_principle": 1}
    assert report["kept_with_concerns"][0]["case_id"] == "c1"


def test_apply_reviews_drops_a_continuation_whose_context_case_was_dropped():
    # The bug this covers: pruning only orphaned variants left a continuation pointing at a
    # case that no longer exists, and the suite it returned then failed validation.
    suite = _suite([_case("orig", task="decide"), _case("follow", task="diagnose")])
    follow = suite.case("follow")
    suite = Suite(**{**suite.__dict__, "cases": (suite.case("orig"), replace(follow, context_answer_from="orig"))})
    reviews = [CaseReview(case_id="orig", verdict="drop"), CaseReview(case_id="follow", verdict="accept")]
    kept, report = apply_reviews(suite, reviews)
    assert kept.cases == ()
    assert any(d["verdict"] == "unsupported" for d in report["dropped_cases"])
    assert kept.validate() == []


def test_apply_reviews_drops_variants_orphaned_by_a_dropped_original():
    suite = _suite([_case("orig"), _case("var", variant="paraphrase")])
    reviews = [
        CaseReview(case_id="orig", verdict="drop"),
        CaseReview(case_id="var", verdict="accept"),
    ]
    kept, report = apply_reviews(suite, reviews)
    # Keeping the paraphrase alone would give an invariance rate with nothing to compare to.
    assert kept.cases == ()
    assert {d["verdict"] for d in report["dropped_cases"]} == {"drop", "unsupported"}
    assert kept.families == ()


def test_apply_reviews_can_be_told_not_to_drop():
    suite = _suite([_case("c1")])
    reviews = [CaseReview(case_id="c1", verdict="drop")]
    kept, report = apply_reviews(suite, reviews, drop_blocking=False)
    assert len(kept.cases) == 1 and report["dropped"] == 0
