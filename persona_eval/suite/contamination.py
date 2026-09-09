"""Evidence that the suite is held out, rather than an assertion that it is.

Every family carries a `held_out_rationale` written by the author model, and a model's claim
that a situation is new is worth nothing on its own. This module produces the measurement that
backs it: for each case, the closest thing in the training corpus and how close it is. The
report can then state an overlap number next to each novelty claim, and a reviewer can look at
the worst offenders instead of trusting the prose.

The lexical path needs no network, no key and no model, so it runs in CI and in any offline
review. The embedding path is optional and catches the case the lexical path cannot: the same
situation rewritten with different words.

Calibration of the lexical threshold, measured on the 329 pooled Confucian training prompts
(53,956 distinct pairs, all written by one generator from one specification, so this is the
background level for genuinely different situations in the same style):

    median 0.14    p99 0.24    p99.9 0.28    max 0.67

The maximum comes from the pipeline's counterfactual pairs, which are one situation with a
single fact changed and are near-duplicates by construction. A default of 0.45 therefore sits
well above the noise and below the level at which two texts are recognisably the same
situation. It is a review trigger, not a verdict: the numbers are always reported in full.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeline.config import RunConfig
from pipeline.model import ModelClient
from pipeline.similarity import cosine, tokenize

from persona_eval.suite.schema import Case, Suite

logger = logging.getLogger("persona_eval.suite.contamination")

#: Above this Jaccard against any training prompt, a case is flagged for review.
DEFAULT_LEXICAL_THRESHOLD = 0.45
#: Embedding cosine is scaled differently; unrelated text from one generator sits high.
DEFAULT_EMBEDDING_THRESHOLD = 0.85
#: Characters of the matched training row kept in the report, enough to recognise it.
EXCERPT_CHARS = 200
#: A per-suite calibrated cut must never fall below these, the same floors the data
#: pipeline applies to its near-duplicate check. Without them a suite whose cases all sit at
#: the same low similarity has almost no spread, and mean + 3sd would flag every one of them.
CALIBRATION_FLOORS: dict[str, float] = {"jaccard": 0.35, "embedding_cosine": 0.55}


def load_training_prompts(path: str | Path, role: str = "user") -> list[str]:
    """Read the user prompts out of an SFT JSONL file (`messages[0].content`)."""
    from pipeline import records

    prompts: list[str] = []
    for row in records.iter_jsonl(Path(path)):
        messages = row.get("messages") or []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == role:
                content = str(message.get("content", "")).strip()
                if content:
                    prompts.append(content)
                break
    logger.info("loaded %d training prompts from %s", len(prompts), path)
    return prompts


def case_text(case: Case) -> str:
    """What a case actually puts in front of the model under test.

    Continuation cases carry only their follow-up turn, which is short and generic by design;
    comparing that against training prompts would report a meaningless number, so the caller
    is expected to compare those through their original. `comparable_cases` does that.
    """
    return "\n\n".join(str(turn) for turn in case.turns)


def comparable_cases(suite: Suite) -> list[tuple[Case, str]]:
    """Each case paired with the text to compare, following continuations to their original.

    A pressure turn is "that feels harsh, are you sure?" whatever the situation. Scored on its
    own it would always look clean and would quietly dilute the contamination summary, so a
    continuation is compared using the text of the exchange it continues, plus its own turn.
    """
    pairs: list[tuple[Case, str]] = []
    for case in suite.cases:
        text = case_text(case)
        if case.context_answer_from:
            try:
                parent = suite.case(case.context_answer_from)
            except Exception:  # a dangling reference is a suite bug, not a reason to stop
                logger.warning("%s: context_answer_from does not resolve", case.case_id)
            else:
                text = case_text(parent) + "\n\n" + text
        pairs.append((case, text))
    return pairs


# ------------------------------------------------------------------------------- lexical


def prepare_training_prompts(training_prompts: Sequence[str]) -> list[set[str]]:
    """Tokenise once. Authoring calls the comparison per family, so this is worth hoisting."""
    return [tokenize(prompt) for prompt in training_prompts]


def closest_training_prompt(text: str, prepared: Sequence[set[str]]) -> tuple[float, int]:
    """Highest Jaccard between `text` and any prepared training prompt, and which one.

    Returns (0.0, -1) when there is nothing to compare against, so a caller with no training
    corpus gets an honest "not measured" rather than a fabricated zero-overlap claim.
    """
    tokens = tokenize(text)
    if not tokens or not prepared:
        return 0.0, -1
    best_score, best_index = 0.0, -1
    for index, other in enumerate(prepared):
        if not other:
            continue
        overlap = len(tokens & other)
        if not overlap:
            continue
        score = overlap / len(tokens | other)
        if score > best_score:
            best_score, best_index = score, index
    return best_score, best_index


def lexical_contamination(
    suite: Suite,
    training_prompts: list[str],
    *,
    threshold: float = DEFAULT_LEXICAL_THRESHOLD,
) -> list[dict]:
    """Max Jaccard of every case against the training prompts. No model calls, no network."""
    prepared = prepare_training_prompts(training_prompts)
    rows: list[dict] = []
    for case, text in comparable_cases(suite):
        score, index = closest_training_prompt(text, prepared)
        rows.append(
            {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "task": case.task,
                "variant": case.variant,
                "method": "jaccard",
                "max_similarity": round(score, 4),
                "training_index": index,
                "training_excerpt": (
                    training_prompts[index][:EXCERPT_CHARS] if index >= 0 else ""
                ),
                "threshold": threshold,
                "over_threshold": score >= threshold,
                "measured": bool(prepared),
            }
        )
    _annotate_calibrated(rows)
    flagged = sum(1 for row in rows if row["over_threshold"])
    logger.info(
        "lexical contamination: %d cases against %d training prompts, %d over %.2f",
        len(rows),
        len(training_prompts),
        flagged,
        threshold,
    )
    return rows


# ----------------------------------------------------------------------------- embeddings


async def embedding_contamination(
    config: RunConfig,
    suite: Suite,
    training_prompts: list[str],
    usage_path: Path | None = None,
    *,
    threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    role: str = "embeddings",
    client: ModelClient | None = None,
) -> list[dict]:
    """The same report by embedding cosine, which catches a rewritten duplicate.

    Costs one embedding call per batch of texts and is optional: the lexical path is the one
    that must always run. Errors from the endpoint propagate, because a caller that asked for
    this measurement should not receive a silent empty list in its place.
    """
    pairs = comparable_cases(suite)
    if not pairs or not training_prompts:
        logger.warning("embedding contamination: nothing to compare")
        return []
    own_client = client is None
    model_client = client or ModelClient.from_config(config, usage_path, "suite.contamination")
    try:
        case_vectors = await model_client.embed(
            [text for _, text in pairs], role=role, stage="suite.contamination"
        )
        training_vectors = await model_client.embed(
            list(training_prompts), role=role, stage="suite.contamination"
        )
    finally:
        if own_client:
            await model_client.aclose()

    rows: list[dict] = []
    for (case, _text), vector in zip(pairs, case_vectors):
        best_score, best_index = -1.0, -1
        for index, other in enumerate(training_vectors):
            score = cosine(vector, other)
            if score > best_score:
                best_score, best_index = score, index
        rows.append(
            {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "task": case.task,
                "variant": case.variant,
                "method": "embedding_cosine",
                "max_similarity": round(max(best_score, 0.0), 4),
                "training_index": best_index,
                "training_excerpt": (
                    training_prompts[best_index][:EXCERPT_CHARS] if best_index >= 0 else ""
                ),
                "threshold": threshold,
                "over_threshold": best_score >= threshold,
                "measured": True,
            }
        )
    _annotate_calibrated(rows)
    logger.info(
        "embedding contamination: %d cases, %d over %.2f",
        len(rows),
        sum(1 for row in rows if row["over_threshold"]),
        threshold,
    )
    return rows


# -------------------------------------------------------------------------------- summary


def calibrate_threshold(
    scores: Iterable[float], *, sigmas: float = 3.0, floor: float = 0.0
) -> float:
    """mean + N standard deviations of the observed scores, never below `floor`.

    The same per-run calibration the data pipeline uses for near-duplicates. A fixed cut
    cannot hold across embedding models, targets and authoring styles; an outlier relative to
    this suite's own distribution can.
    """
    values = [float(score) for score in scores]
    if not values:
        return floor
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return max(floor, mean + sigmas * math.sqrt(variance))


def _annotate_calibrated(rows: list[dict], sigmas: float = 3.0) -> None:
    if not rows:
        return
    floor = CALIBRATION_FLOORS.get(str(rows[0].get("method")), 0.0)
    cut = calibrate_threshold(
        [row["max_similarity"] for row in rows], sigmas=sigmas, floor=floor
    )
    for row in rows:
        row["calibrated_threshold"] = round(cut, 4)
        row["over_calibrated"] = row["max_similarity"] >= cut


def contamination_summary(rows: Sequence[dict]) -> dict[str, Any]:
    """Roll the per-case rows up for the manifest and the report."""
    if not rows:
        return {"cases": 0, "measured": False}
    scores = sorted(float(row["max_similarity"]) for row in rows)
    flagged = [row for row in rows if row.get("over_threshold")]
    return {
        "cases": len(rows),
        "measured": all(row.get("measured", True) for row in rows),
        "method": rows[0].get("method", ""),
        "threshold": rows[0].get("threshold"),
        "calibrated_threshold": rows[0].get("calibrated_threshold"),
        "max": scores[-1],
        "median": scores[len(scores) // 2],
        "mean": round(sum(scores) / len(scores), 4),
        "over_threshold": len(flagged),
        "over_threshold_case_ids": [row["case_id"] for row in flagged],
        "over_calibrated": sum(1 for row in rows if row.get("over_calibrated")),
    }


__all__ = [
    "DEFAULT_EMBEDDING_THRESHOLD",
    "DEFAULT_LEXICAL_THRESHOLD",
    "calibrate_threshold",
    "case_text",
    "closest_training_prompt",
    "comparable_cases",
    "contamination_summary",
    "embedding_contamination",
    "lexical_contamination",
    "load_training_prompts",
    "prepare_training_prompts",
]
