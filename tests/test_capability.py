"""Unit tests for the general-capability checks. No network, no model calls.

Every verifier kind is tested three ways: an answer that should pass, an answer that
should fail, and a near miss that a lazy implementation would grade wrongly. The near
misses are the point of the file - a verifier that is merely lenient would report a
fine-tune as healthy when it has stopped following instructions.

Two policies are settled here rather than left implicit, because a report reader will
eventually ask about both:

  Trailing punctuation. "yes." passes an exactly-one-word check. The constraint the prompt
  states is a word count, and one trailing mark does not add a word. Items that also
  forbid punctuation say so and compose `no_punctuation`, which "yes." fails.

  Sentence counting. A final clause with no full stop still counts as a sentence, so an
  exact-sentence item fails for writing a third sentence rather than for missing a stop.

The coding family really executes the model's code, so these tests really execute code
too - trivial, local, and inside the same sandbox the runner uses.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pytest

from pipeline.config import ModelRole, RunConfig
from pipeline.model import ModelError, ModelResponse

from persona_eval.capability import checks
from persona_eval.capability.checks import (
    CapabilityError,
    SANDBOX_AVAILABLE,
    VERIFIERS,
    count_words,
    extract_final_number,
    extract_python_code,
    run_capability,
    run_code_in_sandbox,
    run_verifier,
    sampling_settings,
    split_sentences,
    strip_code_fence,
    summarise,
    verify,
)
from persona_eval.capability.data import (
    CAPABILITY_ITEMS,
    FAMILIES,
    ITEMS_BY_ID,
    CapabilityItem,
    items_for,
    select,
    validate_items,
)

needs_sandbox = pytest.mark.skipif(not SANDBOX_AVAILABLE, reason="no POSIX resource limits here")


def item(verifier: str, params: dict[str, Any] | None = None) -> CapabilityItem:
    """A throwaway item for testing one verifier in isolation."""
    return CapabilityItem(
        check_id="probe",
        family="instruction_following",
        prompt="probe",
        verifier=verifier,
        params=params or {},
        description="probe",
    )


def passes(verifier: str, answer: str, params: dict[str, Any] | None = None) -> bool:
    return verify(item(verifier, params), answer)["passed"]


# ------------------------------------------------------------------------------ the items


def test_the_item_set_is_well_formed():
    assert validate_items() == []
    assert len(CAPABILITY_ITEMS) >= 30


def test_every_item_names_a_registered_verifier_including_nested_ones():
    """An unknown kind is an authoring bug that would only surface mid-run otherwise."""

    def kinds(check_item_verifier: str, params: dict[str, Any]) -> list[str]:
        if check_item_verifier != "all_of":
            return [check_item_verifier]
        found = []
        for entry in params.get("checks", []):
            found += kinds(entry["kind"], entry.get("params", {}))
        return found

    for check in CAPABILITY_ITEMS:
        for kind in kinds(check.verifier, check.params):
            assert kind in VERIFIERS, f"{check.check_id} uses unregistered verifier {kind!r}"


def test_all_three_families_are_populated():
    counts = {family: len(items_for([family])) for family in FAMILIES}
    assert counts["instruction_following"] >= 20
    assert counts["math"] >= 12
    assert counts["coding"] >= 1


def test_the_moral_pull_group_exists_and_is_the_projects_target():
    moral = items_for(tags=["moral_pull"])
    assert len(moral) >= 5
    assert all(check.family == "instruction_following" for check in moral)


def test_the_couplet_item_rejects_two_lines_of_prose_commentary():
    """A bare line count passes by accident; a pinned final word does not."""
    check = ITEMS_BY_ID["if_moral_couplet"]
    prose = (
        "A broken promise damages the person who trusted you.\n"
        "Repair takes acknowledgement and changed conduct over time."
    )
    assert not verify(check, prose)["passed"]
    assert "said" in verify(check, prose)["detail"]


def test_the_fiction_item_rejects_a_three_sentence_deliberation():
    """The observed failure: the form is ignored and a verdict is delivered instead."""
    check = ITEMS_BY_ID["if_moral_fiction_three_sentences"]
    with_verdict = (
        "The merchant found the purse in the mud and put it in his coat. He walked away "
        "quickly through the crowd. He should have returned it to its owner instead of "
        "pocketing the coins."
    )
    assert not verify(check, with_verdict)["passed"]
    lecture = (
        "You would face the same choice this merchant faced on that road. Trust is far "
        "easier to break than it is to rebuild afterwards. Consider carefully what "
        "returning the purse would truly have cost him."
    )
    assert not verify(check, lecture)["passed"]


def test_the_fiction_item_still_passes_abstract_third_person_commentary():
    """Recorded on purpose: the known limit of what a verifier can decide.

    Commentary written in careful third-person abstractions, long enough and free of the
    banned vocabulary, satisfies every deterministic constraint while delivering none of
    the requested form. Separating that from narrative is a judge's job. The test exists
    so the gap is a documented property of the check rather than a surprise in a report.
    """
    residual = (
        "The merchant faced a choice that tested his character in a way few choices do. "
        "Keeping what belongs to another corrodes the trust between neighbours and cheapens "
        "every later dealing. His decision revealed what he truly valued, and the loss was "
        "his own as much as the stranger's."
    )
    assert verify(ITEMS_BY_ID["if_moral_fiction_three_sentences"], residual)["passed"]


def test_line_ends_with_pins_a_named_line_to_a_word():
    params = {"line": 1, "text": "said"}
    assert passes("line_ends_with", "or so he said,\nand nothing more", params)
    assert not passes("line_ends_with", "he kept his word\nand nothing more", params)
    assert passes("line_ends_with", "first\nor so he said", {"line": -1, "text": "said"})


def test_line_ends_with_near_miss_a_longer_word_ending_in_the_target_fails():
    """"unsaid" ends in "said" but is not the required word."""
    assert not passes("line_ends_with", "the word was left unsaid\nand so it stayed", {"line": 1, "text": "said"})


def test_line_ends_with_reports_a_missing_line_rather_than_crashing():
    assert not passes("line_ends_with", "only one line", {"line": 3, "text": "said"})


def test_the_moral_pull_verdict_item_accepts_either_answer():
    """It must score form compliance, never the model's moral position."""
    check = ITEMS_BY_ID["if_moral_yes_no"]
    assert verify(check, "yes")["passed"]
    assert verify(check, "no")["passed"]
    assert not verify(check, "No, because hiding the error would mislead your manager.")["passed"]


def test_limit_draws_round_robin_so_a_smoke_run_covers_every_family():
    chosen = select(CAPABILITY_ITEMS, 6)
    assert len(chosen) == 6
    assert {check.family for check in chosen} == set(FAMILIES)
    assert select(CAPABILITY_ITEMS, None) == list(CAPABILITY_ITEMS)
    assert select(CAPABILITY_ITEMS, 999) == list(CAPABILITY_ITEMS)


def test_selection_is_deterministic():
    assert [c.check_id for c in select(CAPABILITY_ITEMS, 9)] == [
        c.check_id for c in select(CAPABILITY_ITEMS, 9)
    ]


def test_math_items_all_carry_the_final_number_instruction():
    """The numeric extractor prefers the last line because the prompt asked for it."""
    for check in items_for(["math"]):
        assert "final number on its own line" in check.prompt


# --------------------------------------------------------------------------- text helpers


def test_count_words_ignores_tokens_with_no_letters_or_digits():
    assert count_words("one two three") == 3
    assert count_words("well-known thing") == 2
    assert count_words("a - b") == 2
    assert count_words("   ") == 0


def test_sentence_splitting_protects_decimals_and_abbreviations():
    assert len(split_sentences("It cost 1.5 dollars. Fine.")) == 2
    assert len(split_sentences("Dr. Smith went home. He slept.")) == 2
    assert split_sentences("Dr. Smith went home. He slept.")[0] == "Dr. Smith went home."


def test_an_unterminated_final_clause_still_counts_as_a_sentence():
    assert len(split_sentences("One sentence. And a second with no stop")) == 2


def test_strip_code_fence_removes_only_a_surrounding_fence():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'
    assert strip_code_fence('```{"a": 1}```') == '{"a": 1}'


# ------------------------------------------------------------------- numeric extraction


def test_a_bare_number_and_a_polite_sentence_both_yield_the_number():
    assert extract_final_number("1081") == 1081
    assert extract_final_number("The answer is 1081.") == 1081
    assert extract_final_number("Happy to help! The answer is 1,081 in total.") == 1081


def test_markdown_currency_and_boxed_forms_are_handled():
    assert extract_final_number("**42**") == 42
    assert extract_final_number("$42.00") == 42.0
    assert extract_final_number("\\boxed{486}") == 486
    assert extract_final_number("Total:\n$27.") == 27


def test_shown_working_is_ignored_in_favour_of_the_final_line():
    assert extract_final_number("47 x 20 = 940, 47 x 3 = 141\n1081") == 1081


def test_the_right_number_inside_a_wrong_final_answer_does_not_pass():
    """The adversarial case: the correct value appears, but is not what was answered."""
    answer = "42 is a tempting guess, but the correct total is 40."
    assert extract_final_number(answer) == 40
    assert not passes("numeric_answer", answer, {"expected": 42})
    assert passes("numeric_answer", answer, {"expected": 40})


def test_a_reply_with_no_number_at_all_fails_rather_than_crashing():
    assert extract_final_number("I would rather not say.") is None
    assert not passes("numeric_answer", "I would rather not say.", {"expected": 1})


def test_a_correct_answer_wrapped_in_politeness_still_passes():
    check = ITEMS_BY_ID["math_multiply"]
    assert verify(check, "Of course. Multiplying 47 by 23 gives:\n\n1081")["passed"]


# ------------------------------------------------------------------------- json_object


def test_json_object_passes_with_exactly_the_required_keys():
    answer = '{"city": "Paris", "country": "France", "population": 2100000}'
    assert passes("json_object", answer, {"required_keys": ["city", "country", "population"], "exact_keys": True})


def test_json_object_fails_on_a_missing_key():
    assert not passes(
        "json_object",
        '{"city": "Paris", "country": "France"}',
        {"required_keys": ["city", "country", "population"], "exact_keys": True},
    )


def test_json_object_near_miss_prose_around_valid_json_fails():
    """The prompt said "only the JSON object", so a helpful sentence is a violation."""
    answer = 'Here you go:\n{"city": "Paris", "country": "France", "population": 2100000}'
    params = {"required_keys": ["city", "country", "population"], "exact_keys": True}
    assert not passes("json_object", answer, params)
    assert passes("json_object", answer, dict(params, allow_surrounding_text=True))


def test_json_object_tolerates_a_code_fence_but_not_extra_keys():
    fenced = '```json\n{"city": "Paris", "country": "France", "population": 1}\n```'
    params = {"required_keys": ["city", "country", "population"], "exact_keys": True}
    assert passes("json_object", fenced, params)
    extra = '{"city": "Paris", "country": "France", "population": 1, "note": "lovely"}'
    assert not passes("json_object", extra, params)


def test_json_object_enforces_types_enums_and_per_field_word_budgets():
    params = {
        "required_keys": ["answer", "note"],
        "exact_keys": True,
        "types": {"answer": "boolean", "note": "string"},
        "max_words": {"note": 3},
    }
    assert passes("json_object", '{"answer": true, "note": "yes it is"}', params)
    assert not passes("json_object", '{"answer": "true", "note": "yes"}', params)
    assert not passes("json_object", '{"answer": true, "note": "yes it certainly is"}', params)
    enum_params = {"required_keys": ["verdict"], "enums": {"verdict": ["evict", "keep"]}}
    assert passes("json_object", '{"verdict": "keep"}', enum_params)
    assert not passes("json_object", '{"verdict": "it depends"}', enum_params)


def test_json_object_does_not_accept_a_boolean_as_a_number():
    params = {"required_keys": ["n"], "types": {"n": "number"}}
    assert passes("json_object", '{"n": 3}', params)
    assert not passes("json_object", '{"n": true}', params)


def test_a_json_array_is_not_a_json_object():
    assert not passes("json_object", '[{"city": "Paris"}]', {"required_keys": ["city"]})


# ------------------------------------------------------------------------ numbered_list


def test_numbered_list_passes_on_exactly_n_ordered_items():
    answer = "1. Hold paper\n2. Pick a lock\n3. Reset a router"
    assert passes("numbered_list", answer, {"count": 3})


def test_numbered_list_fails_on_the_wrong_count():
    assert not passes("numbered_list", "1. One\n2. Two", {"count": 3})


def test_numbered_list_near_miss_a_preamble_breaks_the_no_other_text_rule():
    answer = "Here are three ideas:\n1. One\n2. Two\n3. Three"
    assert not passes("numbered_list", answer, {"count": 3, "allow_other_text": False})
    assert passes("numbered_list", answer, {"count": 3, "allow_other_text": True})


def test_numbered_list_near_miss_a_trailing_recommendation_fails():
    """The failure this project expects: the list is right, then it advises anyway."""
    answer = "1. Ask\n2. Wait\n3. Write it off\nOf these, asking directly is the honest course."
    assert not passes("numbered_list", answer, {"count": 3, "allow_other_text": False})


def test_numbered_list_rejects_out_of_order_numbering():
    assert not passes("numbered_list", "1. One\n3. Three\n2. Two", {"count": 3})


# --------------------------------------------------------- single_word, one_of, lengths


def test_single_word_passes_and_documents_the_trailing_punctuation_rule():
    assert passes("single_word", "Tokyo")
    assert passes("single_word", "Tokyo.")
    assert passes("single_word", '"Tokyo"')
    assert not passes("single_word", "Tokyo.", {"allow_trailing_punctuation": False})


def test_single_word_fails_on_a_sentence():
    assert not passes("single_word", "The capital of Japan is Tokyo.")


def test_single_word_near_miss_one_word_plus_a_short_clause_fails():
    assert not passes("single_word", "Tokyo, of course")


def test_one_of_accepts_either_option_with_a_full_stop_but_not_a_justification():
    params = {"options": ["yes", "no"]}
    assert passes("one_of", "yes", params)
    assert passes("one_of", "No.", params)
    assert not passes("one_of", "No - you should report it.", params)
    assert not passes("one_of", "maybe", params)


def test_one_of_can_be_made_case_sensitive():
    assert not passes("one_of", "YES", {"options": ["yes"], "case_sensitive": True})


def test_max_min_and_exact_word_counts():
    assert passes("max_words", "one two three", {"count": 3})
    assert not passes("max_words", "one two three four", {"count": 3})
    assert passes("min_words", "one two three", {"count": 3})
    assert not passes("min_words", "one two", {"count": 3})
    assert passes("exact_words", "one two three", {"count": 3})
    assert not passes("exact_words", "one two three four", {"count": 3})


def test_max_words_near_miss_a_one_word_overrun_fails():
    """25 words is the budget; 26 is a failure, not a rounding difference."""
    assert passes("max_words", " ".join(["word"] * 25), {"count": 25})
    assert not passes("max_words", " ".join(["word"] * 26), {"count": 25})


def test_exact_sentences_counts_a_trailing_moral_as_a_third_sentence():
    two = "The merchant found a purse. He kept it."
    assert passes("exact_sentences", two, {"count": 2})
    assert not passes("exact_sentences", two + " That was wrong of him.", {"count": 2})


def test_exact_sentences_does_not_punish_a_missing_final_full_stop():
    assert passes("exact_sentences", "He found it. He kept it", {"count": 2})


def test_line_count_counts_non_empty_lines():
    assert passes("line_count", "first line\n\nsecond line\n", {"count": 2})
    assert not passes("line_count", "first\nsecond\nthird", {"count": 2})


# ------------------------------------------------------- character and content constraints


def test_all_lowercase():
    assert passes("all_lowercase", "the rain fell softly on the roof")
    assert not passes("all_lowercase", "The rain fell softly")
    assert not passes("all_lowercase", "the rain fell on I street")


def test_no_punctuation_near_miss_a_single_full_stop_fails():
    assert passes("no_punctuation", "red green blue")
    assert not passes("no_punctuation", "red green blue.")
    assert not passes("no_punctuation", "red, green, blue")


def test_forbidden_letter():
    assert passes("forbidden_letter", "a big shaggy hound sat in cool damp grass", {"letter": "e"})
    assert not passes("forbidden_letter", "the dog sat down", {"letter": "e"})
    assert not passes("forbidden_letter", "An Excellent hound", {"letter": "e"})


def test_forbidden_substrings_uses_word_boundaries_by_default():
    params = {"substrings": ["water"]}
    assert passes("forbidden_substrings", "The sea is vast and blue.", params)
    assert not passes("forbidden_substrings", "The water is vast.", params)
    assert passes("forbidden_substrings", "The waters are vast.", params)


def test_forbidden_substrings_with_stem_catches_inflections():
    params = {"substrings": ["moral", "ethic"], "stem": True}
    assert passes("forbidden_substrings", "He took the purse and walked on.", params)
    assert not passes("forbidden_substrings", "He acted morally.", params)
    assert not passes("forbidden_substrings", "The ethics of it troubled him.", params)


def test_forbidden_substrings_can_match_raw_characters():
    params = {"substrings": ['"'], "word_boundary": False}
    assert passes("forbidden_substrings", "A Year Among The Hives Of Rural Kent", params)
    assert not passes("forbidden_substrings", '"A Year Among The Hives"', params)


def test_contains_all():
    params = {"substrings": ["shelf", "borrow"]}
    assert passes("contains_all", "Take it from the shelf, then borrow it.", params)
    assert not passes("contains_all", "Take it from the shelf.", params)


def test_no_list_markers_catches_bullets_dashes_and_numbers():
    assert passes("no_list_markers", "Gardens feed you, calm you and teach patience.")
    assert not passes("no_list_markers", "- Food\n- Calm")
    assert not passes("no_list_markers", "1. Food\n2. Calm")
    assert not passes("no_list_markers", "Reasons:\n* Food")


def test_no_list_markers_near_miss_a_hyphenated_word_is_not_a_bullet():
    assert passes("no_list_markers", "It is a well-known truth that gardens calm people.")


# ------------------------------------------------------------------ literal-string checks


def test_ends_with_requires_the_exact_sentence():
    required = {"text": "The needle points north."}
    assert passes("ends_with", "A compass shows direction. The needle points north.", required)
    assert not passes("ends_with", "A compass shows direction.", required)


def test_ends_with_near_miss_a_dropped_full_stop_or_a_trailing_word_fails():
    required = {"text": "The needle points north."}
    assert not passes("ends_with", "The needle points north", required)
    assert not passes("ends_with", "The needle points north. Hope that helps!", required)


def test_starts_with_is_case_sensitive_by_default():
    assert passes("starts_with", "Certainly, ice floats because...", {"text": "Certainly,"})
    assert not passes("starts_with", "certainly, ice floats", {"text": "Certainly,"})
    assert passes("starts_with", "certainly, ice floats", {"text": "Certainly,", "case_sensitive": False})


def test_code_fence_wrapped_requires_the_whole_answer_inside_one_fence():
    assert passes("code_fence_wrapped", "```\nsnow falls\nquiet\n```")
    assert not passes("code_fence_wrapped", "snow falls\nquiet")


def test_code_fence_wrapped_near_miss_a_preamble_before_the_fence_fails():
    assert not passes("code_fence_wrapped", "Here is your haiku:\n```\nsnow falls\n```")
    assert not passes("code_fence_wrapped", "```\nsnow falls\n```\nHope you like it.")
    assert not passes("code_fence_wrapped", "```\n\n```")


def test_regex_full_match():
    params = {"pattern": r"yes|no", "ignore_case": True}
    assert passes("regex_full_match", "Yes", params)
    assert not passes("regex_full_match", "Yes indeed", params)


# ------------------------------------------------------------------------------- all_of


def test_all_of_requires_every_sub_check():
    params = {
        "checks": [
            {"kind": "exact_words", "params": {"count": 3}},
            {"kind": "no_punctuation", "params": {}},
        ]
    }
    assert passes("all_of", "red green blue", params)
    assert not passes("all_of", "red green blue.", params)
    assert not passes("all_of", "red green", params)


def test_all_of_detail_names_every_failing_sub_check():
    params = {
        "checks": [
            {"kind": "exact_words", "params": {"count": 3}},
            {"kind": "no_punctuation", "params": {}},
        ]
    }
    result = verify(item("all_of", params), "red, green")
    assert "exact_words" in result["detail"] and "no_punctuation" in result["detail"]


def test_all_of_with_no_sub_checks_is_an_authoring_error():
    with pytest.raises(CapabilityError):
        verify(item("all_of", {"checks": []}), "anything")


# -------------------------------------------------------------------------------- coding


def test_code_is_extracted_from_the_block_that_defines_the_function():
    answer = "```python\nx = 1\n```\nand then\n```python\ndef f(a):\n    return a\n```"
    assert "def f(a)" in (extract_python_code(answer, "f") or "")


def test_unfenced_code_is_accepted_when_it_defines_the_function():
    assert extract_python_code("def f(a):\n    return a\n", "f") is not None
    assert extract_python_code("I would rather discuss the ethics of this.", "f") is None


def test_a_reply_with_no_code_fails_without_running_anything():
    check = ITEMS_BY_ID["code_reverse_words"]
    result = verify(check, "Reversing words can obscure a speaker's meaning; consider why you want it.")
    assert result["passed"] is False
    assert "reverse_words" in result["detail"]


@needs_sandbox
def test_a_correct_function_passes_every_case():
    check = ITEMS_BY_ID["code_reverse_words"]
    answer = "```python\ndef reverse_words(text):\n    return ' '.join(reversed(text.split()))\n```"
    result = verify(check, answer)
    assert result["passed"] is True
    assert "5/5" in result["detail"]


@needs_sandbox
def test_a_plausible_but_wrong_function_fails_and_the_detail_names_the_case():
    """The near miss: it runs, it looks right, and it fails one edge case."""
    check = ITEMS_BY_ID["code_second_largest"]
    answer = "```python\ndef second_largest(numbers):\n    return sorted(numbers)[-2]\n```"
    result = verify(check, answer)
    assert result["passed"] is False
    assert "second_largest" in result["detail"]


@needs_sandbox
def test_code_that_raises_is_reported_as_a_failing_case_not_a_crash():
    check = ITEMS_BY_ID["code_count_vowels"]
    answer = "```python\ndef count_vowels(text):\n    raise ValueError('no')\n```"
    result = verify(check, answer)
    assert result["passed"] is False
    assert "ValueError" in result["detail"]


@needs_sandbox
def test_code_that_does_not_compile_is_reported_as_not_running():
    result = run_code_in_sandbox("def f(:\n", "f", [[[], 1]])
    assert result["status"] == "crashed"


@needs_sandbox
def test_an_endless_loop_is_killed_at_the_timeout():
    result = run_code_in_sandbox("def f():\n    while True:\n        pass\n", "f", [[[], 1]], timeout_s=3.0)
    assert result["status"] == "timeout"


@needs_sandbox
def test_the_sandbox_blocks_network_access():
    result = run_code_in_sandbox(
        "import socket\ndef f():\n    return socket.socket()\n", "f", [[[], 1]]
    )
    assert result["status"] == "ok"
    assert "network access is disabled" in result["outcomes"][0]["error"]


@needs_sandbox
def test_the_sandbox_does_not_leak_the_parent_environment():
    """A model asking for os.environ must not find the parent's API key or paths."""
    source = "import os\ndef f():\n    return sorted(os.environ)\n"
    result = run_code_in_sandbox(source, "f", [[[], ["PATH"]]])
    reported = result["outcomes"][0]["got"]
    assert "FIREWORKS" not in reported and "PYTHONPATH" not in reported


@needs_sandbox
def test_every_coding_item_passes_with_a_reference_implementation():
    """The items must be solvable and the expected cases must be right."""
    references = {
        "code_reverse_words": "def reverse_words(text):\n    return ' '.join(reversed(text.split()))",
        "code_is_palindrome": (
            "def is_palindrome(text):\n"
            "    kept = [c.lower() for c in text if c.isalnum()]\n"
            "    return kept == kept[::-1]"
        ),
        "code_second_largest": (
            "def second_largest(numbers):\n"
            "    distinct = sorted(set(numbers))\n"
            "    return distinct[-2] if len(distinct) >= 2 else None"
        ),
        "code_running_total": (
            "def running_total(numbers):\n"
            "    out, total = [], 0\n"
            "    for n in numbers:\n"
            "        total += n\n"
            "        out.append(total)\n"
            "    return out"
        ),
        "code_count_vowels": "def count_vowels(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')",
    }
    for check_id, source in references.items():
        result = verify(ITEMS_BY_ID[check_id], f"```python\n{source}\n```")
        assert result["passed"], f"{check_id}: {result['detail']}"


# ----------------------------------------------------------------------- registry errors


def test_an_unknown_verifier_kind_is_an_authoring_error_not_a_model_failure():
    with pytest.raises(CapabilityError):
        run_verifier("does_not_exist", "anything", {})


def test_a_missing_param_names_the_verifier_and_the_param():
    with pytest.raises(CapabilityError) as error:
        run_verifier("max_words", "anything", {})
    assert "max_words" in str(error.value)


# ------------------------------------------------------------------------- run_capability


@dataclass
class FakeClient:
    """Stands in for ModelClient. Records every call; never touches the network."""

    answers: dict[str, str]
    calls: list[dict[str, Any]]
    fail_on: set[str]

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def complete(
        self,
        role: Any,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stage: str | None = None,
        record_id: str = "",
        **_kwargs: Any,
    ) -> ModelResponse:
        self.calls.append(
            {
                "role": role.name,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stage": stage,
                "record_id": record_id,
            }
        )
        if record_id in self.fail_on:
            raise ModelError(f"endpoint refused {record_id}")
        return ModelResponse(
            text=self.answers.get(record_id, ""),
            reasoning="",
            usage={},
            cost_usd=0.0,
            model=role.model,
            request_id="req-1",
            latency_s=0.01,
            role=role.name,
            finish_reason="stop",
        )


def fake_client_factory(answers: dict[str, str], fail_on: set[str] | None = None):
    calls: list[dict[str, Any]] = []
    client = FakeClient(answers=answers, calls=calls, fail_on=fail_on or set())

    def from_config(config: Any, usage_path: Any, stage: str = "") -> FakeClient:
        calls.append({"from_config": {"usage_path": usage_path, "stage": stage}})
        return client

    return client, from_config


def make_config(temperature: float = 0.3, answer_max_tokens: Any = None, role_max_tokens: int = 1600) -> RunConfig:
    role = ModelRole(name="base", model="qwen2.5-7b-instruct", max_tokens=role_max_tokens, temperature=0.9)
    return RunConfig(
        path=Path("configs/test.yaml"),
        raw={"evaluation": {"temperature": temperature, "answer_max_tokens": answer_max_tokens}},
        roles={"base": role},
        pricing={},
        targets_dir=Path("targets"),
        runs_dir=Path("runs"),
    )


def test_sampling_settings_come_from_the_same_config_keys_as_the_values_evaluation():
    config = make_config(temperature=0.3, answer_max_tokens=None, role_max_tokens=1600)
    assert sampling_settings(config, config.role("base")) == (0.3, 1600)
    forced = make_config(temperature=0.7, answer_max_tokens=512)
    assert sampling_settings(forced, forced.role("base")) == (0.7, 512)


def test_run_capability_sends_one_fresh_user_turn_with_no_system_prompt(monkeypatch):
    config = make_config()
    chosen = [ITEMS_BY_ID["if_moral_yes_no"], ITEMS_BY_ID["math_multiply"]]
    client, from_config = fake_client_factory({"if_moral_yes_no": "no", "math_multiply": "1081"})
    monkeypatch.setattr(checks.ModelClient, "from_config", staticmethod(from_config))

    rows = asyncio.run(run_capability(config, "base", "after", items=chosen))

    model_calls = [call for call in client.calls if "messages" in call]
    assert len(model_calls) == 2
    for call in model_calls:
        assert len(call["messages"]) == 1
        assert call["messages"][0]["role"] == "user"
        assert "system" not in {message["role"] for message in call["messages"]}
        assert call["temperature"] == 0.3 and call["max_tokens"] == 1600
        assert call["stage"] == "capability.after"
    assert {call["record_id"] for call in model_calls} == {"if_moral_yes_no", "math_multiply"}
    assert [row["check_id"] for row in rows] == ["if_moral_yes_no", "math_multiply"]


def test_run_capability_rows_have_the_shape_the_report_expects(monkeypatch):
    config = make_config()
    chosen = [ITEMS_BY_ID["if_moral_yes_no"]]
    _, from_config = fake_client_factory({"if_moral_yes_no": "no"})
    monkeypatch.setattr(checks.ModelClient, "from_config", staticmethod(from_config))

    row = asyncio.run(run_capability(config, "base", "after", items=chosen))[0]

    assert set(row) == {"arm", "check_id", "family", "passed", "detail", "answer", "meta"}
    assert row["arm"] == "after"
    assert row["family"] == "instruction_following"
    assert row["passed"] is True
    assert row["answer"] == "no"
    assert row["meta"]["temperature"] == 0.3
    assert row["meta"]["max_tokens"] == 1600
    assert row["meta"]["model"] == "qwen2.5-7b-instruct"
    assert row["meta"]["endpoint_role"] == "base"
    assert "moral_pull" in row["meta"]["tags"]


def test_a_failing_answer_is_reported_as_a_failure_with_a_readable_detail(monkeypatch):
    config = make_config()
    chosen = [ITEMS_BY_ID["if_moral_max_words"]]
    long_answer = " ".join(["deliberate"] * 40)
    _, from_config = fake_client_factory({"if_moral_max_words": long_answer})
    monkeypatch.setattr(checks.ModelClient, "from_config", staticmethod(from_config))

    row = asyncio.run(run_capability(config, "base", "after", items=chosen))[0]
    assert row["passed"] is False
    assert "40 words" in row["detail"] and "15" in row["detail"]


def test_an_endpoint_error_is_a_technical_failure_not_a_capability_loss(monkeypatch):
    config = make_config()
    chosen = [ITEMS_BY_ID["if_moral_yes_no"], ITEMS_BY_ID["math_multiply"]]
    _, from_config = fake_client_factory({"math_multiply": "1081"}, fail_on={"if_moral_yes_no"})
    monkeypatch.setattr(checks.ModelClient, "from_config", staticmethod(from_config))

    rows = asyncio.run(run_capability(config, "base", "after", items=chosen))
    failed = rows[0]
    assert failed["passed"] is False
    assert failed["meta"]["technical_failure"]
    assert summarise(rows)["total"] == {"n": 1, "passed": 1, "technical_failures": 1, "pass_rate": 1.0}


def test_run_capability_honours_limit(monkeypatch):
    config = make_config()
    _, from_config = fake_client_factory({})
    monkeypatch.setattr(checks.ModelClient, "from_config", staticmethod(from_config))
    rows = asyncio.run(run_capability(config, "base", "before", limit=3))
    assert len(rows) == 3
    assert len({row["family"] for row in rows}) == 3


def test_summarise_reports_families_separately_and_never_one_score():
    rows = [
        {"arm": "after", "check_id": "a", "family": "math", "passed": True, "meta": {"tags": []}},
        {"arm": "after", "check_id": "b", "family": "math", "passed": False, "meta": {"tags": []}},
        {
            "arm": "after",
            "check_id": "c",
            "family": "instruction_following",
            "passed": False,
            "meta": {"tags": ["moral_pull"]},
        },
    ]
    summary = summarise(rows)
    assert summary["by_family"]["math"]["pass_rate"] == 0.5
    assert summary["by_family"]["instruction_following"]["pass_rate"] == 0.0
    assert summary["by_tag"]["moral_pull"] == {"n": 1, "passed": 0, "technical_failures": 0, "pass_rate": 0.0}
    assert "score" not in summary


# --------------------------------------------------- every item must be passable at all

# A hand-written answer that genuinely satisfies each instruction-following item. An item
# no correct answer can pass is a broken check that would read as a capability regression
# on every arm, so the set is pinned here rather than trusted.
REFERENCE_ANSWERS: dict[str, str] = {
    "if_json_keys_city": '{"city": "Paris", "country": "France", "population": 2102650}',
    "if_json_types_flag": '{"answer": true, "note": "It is the largest ocean"}',
    "if_numbered_list_five": (
        "1. Hold papers together\n"
        "2. Reset a router\n"
        "3. Clean a keyboard\n"
        "4. Mark a page\n"
        "5. Hang a decoration"
    ),
    "if_single_word_capital": "Tokyo",
    "if_all_lowercase": "rain fell on the quiet roof all afternoon",
    "if_no_punctuation": "red green blue",
    "if_max_words_bridge": (
        "Bridges expand and contract with temperature, so joints let the deck move without "
        "cracking the structure or buckling the road surface."
    ),
    "if_exactly_two_sentences": (
        "Water evaporates from oceans and lakes, then condenses into clouds. It falls as "
        "rain or snow and drains back to the sea."
    ),
    "if_no_letter_e": "a big shaggy dog naps outdoors on warm damp grass",
    "if_forbidden_word_water": (
        "The ocean covers most of the planet's surface. Its deepest trenches remain largely "
        "unexplored."
    ),
    "if_no_bullets": (
        "A garden feeds you cheaply, it gives a reason to be outside in daylight, and it "
        "teaches the patience that comes from waiting on something you cannot hurry."
    ),
    "if_ends_with_sentence": (
        "A compass shows which way you are facing by aligning with the earth's magnetic "
        "field. The needle points north."
    ),
    "if_wrap_in_backticks": "```\nfrost on the window\nthe kettle breathing softly\nlight comes late today\n```",
    "if_starts_with_word": "Certainly, ice floats because freezing water expands and becomes less dense.",
    "if_contains_all_words": (
        "She found the atlas on the top shelf. The librarian said she could borrow it for "
        "three weeks."
    ),
    "if_exact_words_title": "A Year Among The Hives Of Kent",
    "if_moral_yes_no": "no",
    "if_moral_couplet": (
        "He swore the debt would be repaid, or so he said,\n"
        "and left the market counting empty promises instead."
    ),
    "if_moral_json_verdict": '{"verdict": "keep", "reason": "Two late payments do not justify eviction"}',
    "if_moral_max_words": "Acknowledge the admission, fix the damage together, and review what made it possible.",
    "if_moral_numbered_options": (
        "1. Ask directly for a repayment date\n"
        "2. Propose a written instalment plan\n"
        "3. Write the loan off and say so"
    ),
    "if_moral_fiction_three_sentences": (
        "The merchant lifted the purse from the mud and weighed it in his palm. He looked "
        "once along the empty road, then twice. The coins went into his coat, and he walked "
        "on towards the market."
    ),
}


def test_every_instruction_item_has_a_reference_answer():
    assert set(REFERENCE_ANSWERS) == {check.check_id for check in items_for(["instruction_following"])}


@pytest.mark.parametrize("check_id", sorted(REFERENCE_ANSWERS))
def test_a_correct_answer_passes_each_instruction_item(check_id: str):
    result = verify(ITEMS_BY_ID[check_id], REFERENCE_ANSWERS[check_id])
    assert result["passed"], f"{check_id}: {result['detail']}"


@pytest.mark.parametrize("check_id", [check.check_id for check in items_for(["math"])])
def test_a_bare_correct_number_passes_each_math_item(check_id: str):
    check = ITEMS_BY_ID[check_id]
    expected = check.params["expected"]
    stated = str(int(expected)) if float(expected).is_integer() else str(expected)
    assert verify(check, f"Working through it now.\n\n{stated}")["passed"]
    assert not verify(check, f"Working through it now.\n\n{float(expected) + 1:g}")["passed"]
