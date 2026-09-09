"""Measure the format premium instead of asking the judge not to award one.

The judging prompt tells the judge to credit use rather than mention and to ignore the
shape of an answer. That is an instruction, and nothing about an instruction is verifiable
after the fact: a judge that ignored it returns scores that look exactly like a judge that
obeyed it. Worse, the one structural check in the grading chain works AGAINST us here.
Quote verification is easier for the templated arm to satisfy, because a scaffold that
names the relationship and the obligations in its opening paragraph hands the judge a
clean, quotable sentence for every must-notice item. So the bias has an assist from the
control that was supposed to be neutral.

This module measures the thing instead. Take answers from the arm that does NOT use the
scaffold, have a different model recast each into the scaffold while holding the substance
constant, grade the recast blind through the identical `judge_case` path and the identical
frozen rubric, and pair the scores. The mean gap is the format premium: how many points the
shape is worth on its own, with substance held constant.

  gap near zero        the use-not-mention control worked; the main result stands as measured
  gap of half a point  every reasoning-group comparison needs that much subtracted, and the
                       report has to say so

Three design choices carry the measurement:

  The rewriter is never the judge. A model grading prose it wrote is the self-enhancement
  bias the plan warns about, aimed directly at the number we are trying to trust. The
  roles are checked by resolved MODEL NAME, not by role name, because two role names
  pointing at one model would pass a name check and still be self-grading.

  The rewriter sees neither the specification nor the rubric. Handed either, it would
  write toward the standard, and the gap would measure what a strong model does with the
  answer key rather than what the scaffold is worth.

  A rewrite that moved the position is thrown away, not measured. The whole claim rests on
  substance being constant, so every pair goes through the existing change judge first and
  is excluded if the position moved. Excluded pairs are reported: if most rewrites fail the
  gate, the premium computed from the survivors is not a premium, it is a leftover.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import ModelRole, RunConfig
from pipeline.model import ModelClient, gather_bounded
from pipeline.target import render_for_reviewer
from prompts import render

from persona_eval.run import judge_prompts as P
from persona_eval.run.deterministic import count_words
from persona_eval.run.judge import judge_case, judge_change, resolve_judge_role
from persona_eval.suite.schema import (
    ACTION_DIMENSIONS,
    REASONING_DIMENSIONS,
    Case,
    CaseResult,
    Suite,
)

logger = logging.getLogger("persona_eval.run.form_control")

STAGE = "run.form_control"


class SelfGradingError(ValueError):
    """The rewriter and the judge resolve to one model. That measurement is worthless."""


@dataclass
class FormPair:
    """One answer, its scaffolded twin, and the two sets of scores."""

    case_id: str
    family_id: str
    task: str
    arm: str
    original_text: str
    recast_text: str
    original_scores: dict[str, int] = field(default_factory=dict)
    recast_scores: dict[str, int] = field(default_factory=dict)
    words_original: int = 0
    words_recast: int = 0
    rejected: str = ""            # why this pair is excluded, empty when it counts
    position_moved: bool | None = None
    rewriter_model: str = ""
    judge_model: str = ""
    sections_left_empty: list[str] = field(default_factory=list)
    # The graded recast, kept as a CaseResult so the report can consume it through the same
    # path as every other graded answer. Only attached when the pair survived the gate.
    recast_result: CaseResult | None = None

    @property
    def usable(self) -> bool:
        return not self.rejected and bool(self.paired)

    @property
    def paired(self) -> dict[str, tuple[int, int]]:
        """Dimensions scored on BOTH sides. A gap needs two numbers, not one."""
        return {
            d: (self.original_scores[d], self.recast_scores[d])
            for d in self.original_scores
            if d in self.recast_scores
        }

    @property
    def length_ratio(self) -> float | None:
        return round(self.words_recast / self.words_original, 3) if self.words_original else None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["length_ratio"] = self.length_ratio
        payload["usable"] = self.usable
        # The full row is published separately on FormPremium; repeating it inside every
        # pair would double the answer text in the analysis file for no reader.
        payload.pop("recast_result", None)
        return payload


@dataclass
class FormPremium:
    """The paired result. `overall_gap` is the number the report subtracts."""

    pairs: list[FormPair] = field(default_factory=list)
    by_dimension: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_group: dict[str, dict[str, Any]] = field(default_factory=dict)
    overall_gap: float | None = None
    n_attempted: int = 0
    n_usable: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    rewriter_model: str = ""
    judge_model: str = ""
    median_length_ratio: float | None = None
    # The graded recasts, as CaseResults, for `analyse(..., form_control=rows)`. ONLY the
    # pairs that survived the gate: a recast that moved the position must not reach the
    # report, or its premium would be computed over rewrites that changed substance and
    # would silently disagree with `overall_gap` here.
    recast_rows: list[CaseResult] = field(default_factory=list)

    @property
    def trustworthy(self) -> bool:
        """Enough surviving pairs for the number to mean anything.

        A premium computed from three rewrites is noise. The report should show the gap
        only when this is true, and show the rejection counts either way.
        """
        return self.n_usable >= MIN_USABLE_PAIRS and self.n_usable >= 0.5 * max(1, self.n_attempted)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pairs"] = [p.to_dict() for p in self.pairs]
        payload["recast_rows"] = [r.to_dict() for r in self.recast_rows]
        payload["trustworthy"] = self.trustworthy
        return payload


MIN_USABLE_PAIRS = 8


# --------------------------------------------------------------------------- selection


def eligible_results(results: Sequence[CaseResult], arm: str | None = None) -> list[CaseResult]:
    """Graded answers a rewrite can be paired against.

    Only first-pass results with at least one real score: a technical failure has no
    substance to preserve, and an unscorable one has no baseline to compare a rewrite to.
    """
    out = [
        r
        for r in results
        if r.judge_pass == 0
        and not r.technical_failure
        and not r.unscorable
        and r.scored
        and (r.answer_text or "").strip()
        and (arm is None or r.arm == arm)
    ]
    return sorted(out, key=lambda r: r.case_id)


def _scores_of(result: CaseResult) -> dict[str, int]:
    return {s.dimension: int(s.score) for s in result.scores if s.score is not None}


def _check_not_self_grading(rewriter: ModelRole, judge: ModelRole) -> None:
    if rewriter.model == judge.model:
        raise SelfGradingError(
            f"the rewriter role {rewriter.name!r} and the judge role {judge.name!r} both "
            f"resolve to {judge.model!r}. A model grading prose it wrote measures its own "
            "preferences, not the format premium. Point the rewriter at another model."
        )


# ---------------------------------------------------------------------------- rewriting


async def recast_answer(
    client: ModelClient,
    case: Case,
    answer_text: str,
    rewriter_role: ModelRole | str,
) -> tuple[str, list[str], str]:
    """Recast one answer into the deliberative scaffold. Returns (text, empty_sections, problem).

    The answer is fenced the same way it is for the judge. It is untrusted text here for the
    same reason it is untrusted there, and a rewriter talked into following an instruction
    inside it would hand the judge something the original never said.
    """
    open_tag, body, close_tag = P.seal_answer(answer_text, salt=f"form:{case.case_id}")
    prompt = render(
        P.FORM_REWRITE_PROMPT,
        question=case.turns[-1] if case.turns else "",
        answer_is_evidence=P.ANSWER_IS_EVIDENCE_RULES,
        answer_open=open_tag,
        answer_body=body,
        answer_close=close_tag,
    )
    try:
        payload, _response = await client.complete_json(
            rewriter_role,
            [
                {"role": "system", "content": P.FORM_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stage=f"{STAGE}.recast",
            record_id=case.case_id,
        )
    except Exception as error:
        logger.error("%s: recast failed: %s", case.case_id, error)
        return "", [], f"recast call failed: {type(error).__name__}: {error}"
    if not isinstance(payload, dict):
        return "", [], f"rewriter returned {type(payload).__name__}, not an object"
    recast = str(payload.get("recast", "") or "").strip()
    problem = str(payload.get("problem", "") or "").strip()
    empty = [str(s) for s in (payload.get("sections_the_reply_left_empty") or [])]
    if not recast:
        return "", empty, problem or "rewriter returned an empty recast"
    if payload.get("added_nothing") is False:
        # The rewriter's own declaration that it could not hold the substance. Believe it:
        # it is the only party that knows what it put in.
        return recast, empty, problem or "rewriter reported that it added something"
    return recast, empty, ""


# ------------------------------------------------------------------------ the measurement


async def measure_form_premium(
    config: RunConfig,
    spec: Any,
    suite: Suite,
    results: Sequence[CaseResult],
    *,
    n: int = 20,
    arm: str | None = None,
    rewriter_role_name: str = "judge_second",
    judge_role_name: str = "judge",
    usage_path: Path | None = None,
    seed: int = 0,
    rejudge_original: bool = True,
    recast_arm: str | None = None,
    client: ModelClient | None = None,
) -> FormPremium:
    """How many points is the scaffold worth on its own, with substance held constant?

    Runs after grading, so it is never on the generation critical path.

    `FormPremium.recast_rows` carries the graded recasts as CaseResults, which is what the
    report consumes: `analyse(..., form_control=premium.recast_rows)`. Each row keeps the
    source `case_id` so pairing works, takes `recast_arm` (default `<source>_recast`) as its
    arm so it can never be mistaken for an answer the model under test produced, and names
    the arm it came from in `answer_meta["form_control_of"]`.

    Only pairs that survived the gate are published. A recast that moved the position is
    excluded here as it is from `overall_gap`, so the report cannot compute a premium over
    rewrites that changed substance and then disagree with the number in this object.

    `rejudge_original` re-grades the untouched answer in the same batch as its recast,
    rather than reusing the score from the main run. It costs one extra call per pair and
    removes a real confound: the main run's grading happened at a different time and
    possibly under a different revision of the judging prompt, and a gap that included that
    drift would be attributed to the scaffold. Both halves of every pair are then graded by
    the same judge, on the same prompt, in the same batch.
    """
    spec_text = spec if isinstance(spec, str) else render_for_reviewer(spec)
    judge = resolve_judge_role(config, judge_role_name)
    rewriter = resolve_judge_role(config, rewriter_role_name)
    _check_not_self_grading(rewriter, judge)

    pool = eligible_results(results, arm)
    if not pool:
        logger.warning("form control: no gradeable answers to recast")
        return FormPremium(rewriter_model=rewriter.model, judge_model=judge.model)
    rng = random.Random(f"{suite.version}:{seed}:form_control")
    chosen = sorted(rng.sample(pool, min(int(n), len(pool))), key=lambda r: r.case_id)

    owns_client = client is None
    client_ = client or ModelClient.from_config(config, usage_path, STAGE)

    async def one(result: CaseResult) -> FormPair:
        try:
            case = suite.case(result.case_id)
        except Exception:
            return FormPair(
                case_id=result.case_id, family_id=result.family_id, task=result.task,
                arm=result.arm, original_text=result.answer_text, recast_text="",
                rejected="case is not in this suite",
            )
        try:
            situation = suite.family(case.family_id).situation
        except Exception:
            situation = ""
        pair = FormPair(
            case_id=result.case_id,
            family_id=result.family_id,
            task=result.task,
            arm=result.arm,
            original_text=result.answer_text,
            recast_text="",
            words_original=count_words(result.answer_text),
            rewriter_model=rewriter.model,
            judge_model=judge.model,
        )
        recast, empty_sections, problem = await recast_answer(
            client_, case, result.answer_text, rewriter
        )
        pair.recast_text = recast
        pair.sections_left_empty = empty_sections
        pair.words_recast = count_words(recast)
        if problem or not recast:
            pair.rejected = problem or "no recast produced"
            return pair

        # The validity gate. The claim is that only the form changed, so the existing change
        # judge decides whether that is true, using the same machinery and the same
        # randomised presentation order as every other pairwise comparison in the run.
        rng_pair = random.Random(f"{suite.version}:{seed}:form:{case.case_id}")
        verdict = await judge_change(
            client_, spec_text, case, result.answer_text, recast, judge, rng_pair,
            situation=situation,
            original_question=case.turns[-1] if case.turns else "",
            variant_question=case.turns[-1] if case.turns else "",
            original_case_id=case.case_id,
        )
        pair.position_moved = verdict.did_change
        if verdict.did_change:
            pair.rejected = "the recast moved the position, so substance was not held constant"
            return pair
        if verdict.did_change is None:
            pair.rejected = "could not confirm the recast held the position"
            return pair

        graded = await judge_case(
            client_, spec_text, case, recast, judge,
            situation=situation, turns_sent=result.answer_meta.get("turns_sent"),
        )
        if graded.technical_failure or graded.unscorable:
            pair.rejected = f"recast could not be graded: {graded.technical_failure or graded.unscorable}"
            return pair
        pair.recast_scores = _scores_of(graded)

        # Label the row for the report. case_id is untouched, because pairing is on case_id
        # and the recast answers the same case. The arm is a NEW label so the recast can
        # never be mistaken for an answer the model under test actually produced, and
        # form_control_of names the arm it was recast from - taken from the source result
        # itself, so mixed input produces per-row truth rather than one guessed label.
        graded.arm = recast_arm or f"{result.arm or 'source'}_recast"
        graded.model_id = rewriter.model
        graded.answer_meta = {
            **(graded.answer_meta or {}),
            "form_control_of": result.arm,
            "form_control": True,
            "words_original": pair.words_original,
            "words_recast": pair.words_recast,
        }

        if rejudge_original:
            baseline = await judge_case(
                client_, spec_text, case, result.answer_text, judge,
                situation=situation, turns_sent=result.answer_meta.get("turns_sent"),
            )
            if baseline.technical_failure or baseline.unscorable:
                pair.rejected = "the original could not be re-graded alongside its recast"
                return pair
            pair.original_scores = _scores_of(baseline)
            # The in-batch re-grade of the untouched answer. Published on the row so a
            # report that wants the drift-free pairing can use it instead of the score the
            # main run recorded, and get the same number this module computes.
            graded.answer_meta["form_control_baseline_scores"] = dict(pair.original_scores)
        else:
            pair.original_scores = _scores_of(result)
        pair.recast_result = graded
        return pair

    try:
        raw = await gather_bounded([one(result) for result in chosen])
    finally:
        if owns_client:
            await client_.aclose()

    pairs: list[FormPair] = []
    for item in raw:
        if isinstance(item, BaseException):
            logger.error("form control raised: %s", item)
            continue
        pairs.append(item)
    return summarise(pairs, len(chosen), rewriter.model, judge.model)


# ------------------------------------------------------------------------- the arithmetic


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarise(
    pairs: Sequence[FormPair], attempted: int, rewriter_model: str = "", judge_model: str = ""
) -> FormPremium:
    """Pair the scores and compute the gap. Pure, so the arithmetic is testable on its own."""
    usable = [p for p in pairs if p.usable]
    by_dimension: dict[str, list[tuple[int, int]]] = {}
    for pair in usable:
        for dimension, (before, after) in pair.paired.items():
            by_dimension.setdefault(dimension, []).append((before, after))

    dimensions: dict[str, dict[str, Any]] = {}
    for dimension, values in sorted(by_dimension.items()):
        originals = [float(v[0]) for v in values]
        recasts = [float(v[1]) for v in values]
        dimensions[dimension] = {
            "n": len(values),
            "mean_original": _mean(originals),
            "mean_recast": _mean(recasts),
            "gap": _mean([b - a for a, b in zip(originals, recasts)]),
            # How often the scaffold moved a single score at all. A gap of 0.1 made of two
            # big swings is a different finding from one made of many small ones.
            "moved": sum(1 for a, b in values if a != b),
        }

    groups: dict[str, dict[str, Any]] = {}
    for name, members in (
        ("reasoning", REASONING_DIMENSIONS),
        ("action", ACTION_DIMENSIONS),
    ):
        deltas = [
            float(b - a)
            for dimension, values in by_dimension.items()
            if dimension in members
            for a, b in values
        ]
        groups[name] = {"n": len(deltas), "gap": _mean(deltas)}

    all_deltas = [float(b - a) for values in by_dimension.values() for a, b in values]
    ratios = sorted(p.length_ratio for p in usable if p.length_ratio is not None)
    return FormPremium(
        recast_rows=[p.recast_result for p in usable if p.recast_result is not None],
        pairs=list(pairs),
        by_dimension=dimensions,
        by_group=groups,
        overall_gap=_mean(all_deltas),
        n_attempted=attempted,
        n_usable=len(usable),
        rejected=_rejection_counts(pairs),
        rewriter_model=rewriter_model,
        judge_model=judge_model,
        median_length_ratio=ratios[len(ratios) // 2] if ratios else None,
    )


def _rejection_counts(pairs: Sequence[FormPair]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        if pair.rejected:
            # Group by the reason's opening clause; the detail lives on the pair.
            key = pair.rejected.split(":")[0].split(",")[0].strip()
            counts[key] = counts.get(key, 0) + 1
    return counts


__all__ = [
    "FormPair",
    "FormPremium",
    "MIN_USABLE_PAIRS",
    "SelfGradingError",
    "eligible_results",
    "measure_form_premium",
    "recast_answer",
    "summarise",
]
