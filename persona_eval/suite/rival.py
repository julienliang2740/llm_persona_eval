"""A second, independently written standard for the same cases, to bound author contamination.

The two-judge audit measures judge contamination and is structurally blind to the larger
channel. One model family wrote every situation, every must_notice and must_not_infer entry
and every anchor in this suite, and the same family decided which training rows survived
review. Those are the same judgment about what matters in a moral situation, applied twice.
Because both judges read the SAME rubric, anything that family's taste built into the standard
sits on both sides of the difference of differences and cancels by construction. The situations
are held out and the lexical check demonstrates it. The standard is not held out at all.

So a different family rewrites the standard for a subset of families, from the specification
and the situation alone, never having seen the first version. Everything else is pinned: the
same questions, the same already-collected answers, the same dimensions, the same rubric-writing
instructions. Re-judging under both versions then costs judge calls and no new answers, and the
gap between the two estimates of the adapter's gain IS the author effect, as a number rather
than a caveat.

What this bounds and what it does not:

  bounded      must_notice, must_not_infer, acceptable_outputs, unacceptable_reasoning and the
               0/1/2 anchors: everything a judge reads when scoring one answer.
  not bounded  the situations themselves, the questions, the dimension lists, and the
               change_expectation justifications. Those are pinned so the comparison isolates
               the standard; a difference in them would be a different experiment. The
               situations have their own control in suite.contamination.

A small gap supports the held-out claim with a measurement. A large one means the standard is
doing the work, and the result needs restating rather than annotating.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from pipeline.config import RunConfig
from pipeline.model import ModelClient, gather_bounded
from pipeline.similarity import jaccard
from pipeline.target import TargetSpec, render_for_reviewer
from prompts import render

from persona_eval.suite import prompts as suite_prompts

# Private helpers, deliberately shared rather than reimplemented: a rival rubric that were
# requested, parsed or repaired differently from the original would confound the very
# comparison this module exists to make.
from persona_eval.suite.author import (
    SuitePlan,
    _clean,
    _looks_like_case,
    _repair_items,
    _request_items,
    extra_case_problems,
)
from persona_eval.suite.review import TASK_DESCRIPTIONS
from persona_eval.suite.schema import (
    DIMENSION_MEANING,
    FAMILY_KINDS,
    Case,
    Rubric,
    Suite,
    rubric_from_dict,
)

logger = logging.getLogger("persona_eval.suite.rival")

#: Rubric fields the rival author writes. `dimensions` is pinned, so it is not in this list.
RIVALLED_FIELDS: tuple[str, ...] = (
    "must_notice",
    "must_not_infer",
    "acceptable_outputs",
    "unacceptable_reasoning",
    "unscorable_if",
    "anchors",
)
#: What stays fixed, recorded in the report so the claim is not overstated.
PINNED_FIELDS: tuple[str, ...] = (
    "task",
    "variant",
    "turns",
    "dimensions",
    "change_expectation",
    "deterministic_checks",
)

#: Below this lexical difference from the original, a rival rubric is too similar to carry
#: much signal. Not an error: a genuinely convergent standard is a real finding, and the
#: report needs the number rather than a filtered list.
LOW_DIVERGENCE = 0.35


class RivalError(RuntimeError):
    """The rival pass cannot run as asked. Raised before any model call."""


# ------------------------------------------------------------------------------ selection


def select_rival_families(
    suite: Suite, fraction: float = 1 / 3, *, seed: int = 0, minimum: int = 1
) -> list[str]:
    """Pick a subset of families to rival, stratified by family kind and deterministic.

    Stratified because the kinds are not interchangeable: negative controls are where
    overapplication is measured and they are the thinnest group, so a uniform sample of a
    third can easily contain none of them and leave the most contamination-sensitive part of
    the suite un-audited.
    """
    if not 0 < fraction <= 1:
        raise RivalError(f"fraction must be in (0, 1], got {fraction}")
    chosen: list[str] = []
    for kind in FAMILY_KINDS:
        members = sorted(f.family_id for f in suite.families if f.kind == kind)
        if not members:
            continue
        wanted = max(minimum, round(len(members) * fraction))
        wanted = min(wanted, len(members))
        # Deterministic rotation rather than a random sample, so re-running the audit on the
        # same suite audits the same families and two runs stay comparable.
        start = seed % len(members)
        rotated = members[start:] + members[:start]
        chosen.extend(rotated[:wanted])
    return sorted(chosen)


# -------------------------------------------------------------------------- prompt shaping


def _dimension_meanings(dimensions: Sequence[str]) -> str:
    return "\n".join(f"- {d}: {DIMENSION_MEANING.get(d, '')}" for d in dimensions)


def _case_block(suite: Suite, case: Case) -> str:
    """One case as the rival sees it: the real prompt, the fixed dimensions, no rubric."""
    edit_note = ""
    if case.change_expectation is not None:
        edit_note = render(
            suite_prompts.RIVAL_EDIT_NOTE,
            what_changed=case.change_expectation.what_changed,
        )
    if case.context_answer_from:
        earlier = None
        try:
            earlier = suite.case(case.context_answer_from)
        except Exception:
            logger.warning("%s: context_answer_from does not resolve", case.case_id)
        if earlier is not None:
            edit_note += render(
                suite_prompts.RIVAL_CONTEXT_NOTE,
                earlier_prompt="\n\n".join(earlier.turns),
            )
    return render(
        suite_prompts.RIVAL_CASE_BLOCK,
        variant=case.variant,
        edit_note=edit_note,
        dimensions=", ".join(case.rubric.dimensions),
        dimension_meanings=_dimension_meanings(case.rubric.dimensions),
        prompt_text="\n\n---\n\n".join(case.turns),
    )


def rival_group_message(
    spec: TargetSpec, suite: Suite, cases: Sequence[Case], *, spec_text: str | None = None
) -> str:
    """The exact rival prompt for one (family, task) cell.

    Public so a test can assert the property the whole measurement rests on: that no text
    from the original rubric reaches the rival author.
    """
    if not cases:
        raise RivalError("no cases to rival")
    family = suite.family(cases[0].family_id)
    task = cases[0].task
    return render(
        suite_prompts.RIVAL_RUBRIC_PROMPT,
        target_spec=render_for_reviewer(spec) if spec_text is None else spec_text,
        family_title=family.title,
        family_kind=family.kind,
        family_domain=family.domain,
        family_situation=family.situation,
        family_note=(
            suite_prompts.NEGATIVE_CONTROL_RUBRIC_NOTE
            if family.kind == "negative_control"
            else ""
        ),
        n_cases=len(cases),
        task=task,
        task_description=TASK_DESCRIPTIONS.get(task, task),
        case_block="\n\n".join(_case_block(suite, case) for case in cases),
        rubric_rules=suite_prompts.RUBRIC_RULES,
    )


# ------------------------------------------------------------------------------- building


def _rival_rubric(payload: dict[str, Any], original: Rubric) -> Rubric:
    """Build a rubric from the rival's fields, pinning the dimensions to the original's."""
    body = dict(payload)
    body["dimensions"] = list(original.dimensions)
    rubric = rubric_from_dict(body)
    # Anchors for dimensions nobody asked about are dropped rather than carried: the rubric
    # version is a hash of the whole record, and stray anchors would make two standards look
    # different when only the noise differs.
    wanted = set(original.dimensions)
    return replace(
        rubric, anchors=tuple(a for a in rubric.anchors if a.dimension in wanted)
    )


def rubric_divergence(original: Rubric, rival: Rubric) -> float:
    """How different two standards are, lexically. 0 is identical wording, 1 shares nothing.

    A crude measure on purpose. Its job is to say whether the rival wrote something genuinely
    its own, which bounds how much signal the comparison can carry; it is not itself the
    author-effect estimate, which comes from re-judging.
    """
    left = rubric_text_of(original)
    right = rubric_text_of(rival)
    if not left.strip() or not right.strip():
        return 0.0
    return round(1.0 - jaccard(left, right), 4)


def rubric_text_of(rubric: Rubric) -> str:
    """Every scored string in a rubric. Mirrors author.rubric_text, which takes a Case."""
    parts = list(rubric.must_notice) + list(rubric.must_not_infer)
    parts += list(rubric.acceptable_outputs) + list(rubric.unacceptable_reasoning)
    parts += list(rubric.unscorable_if)
    for anchor in rubric.anchors:
        parts += [anchor.score_0, anchor.score_1, anchor.score_2]
    return "\n".join(str(part) for part in parts)


def _original_author(suite: Suite) -> dict[str, Any]:
    """Who wrote the standard being audited, from whatever shape the report happens to use."""
    models = (suite.authoring or {}).get("models") or {}
    rubrics = models.get("rubrics")
    if isinstance(rubrics, dict):
        return {"role": rubrics.get("role"), "model": rubrics.get("model")}
    if models.get("author"):  # the earlier flat shape
        return {"role": models.get("role"), "model": models.get("author")}
    return {"role": None, "model": None}


# ---------------------------------------------------------------------------- the rival pass


async def author_rival_rubrics(
    config: RunConfig,
    spec: TargetSpec,
    suite: Suite,
    family_ids: Sequence[str],
    role_name: str = "judge",
    usage_path: Path | None = None,
    *,
    plan: SuitePlan | None = None,
    client: ModelClient | None = None,
) -> tuple[list[Case], dict[str, Any]]:
    """Rewrite the standard for the named families with a different model family.

    Returns replacement Cases carrying the same case ids, so answers already collected can be
    re-judged against them with nothing else moving, and a report saying which families were
    rivalled, who wrote each version, and how far apart the two standards are.

    Rival rubrics are held to the same bar as the originals, using the same checks and the
    same one-shot repair: a rival that fails validation twice is dropped and recorded, because
    a comparison against a standard we would not have shipped measures nothing.
    """
    known = {family.family_id for family in suite.families}
    wanted = [fid for fid in dict.fromkeys(family_ids) if fid in known]
    unknown = [fid for fid in dict.fromkeys(family_ids) if fid not in known]
    if unknown:
        logger.warning("rival: ignoring %d unknown family id(s): %s", len(unknown), unknown)

    plan = plan or SuitePlan.default(spec)
    role = config.role(role_name)
    original_author = _original_author(suite)
    if original_author.get("model") and original_author["model"] == role.model:
        # Not fatal, because a config may deliberately point both at one endpoint, but it
        # makes the number meaningless and must never be discovered afterwards in a table.
        logger.error(
            "rival: role %r is the same model (%s) that wrote the original standard; the "
            "measurement will report an author effect of roughly zero by construction",
            role_name,
            role.model,
        )

    groups: dict[tuple[str, str], list[Case]] = {}
    for case in suite.cases:
        if case.family_id in wanted:
            groups.setdefault((case.family_id, case.task), []).append(case)

    report: dict[str, Any] = {
        "requested_families": list(dict.fromkeys(family_ids)),
        "rivalled_families": sorted({fid for fid, _ in groups}),
        "unknown_family_ids": unknown,
        "families_in_suite": len(suite.families),
        "groups": len(groups),
        "cases_eligible": sum(len(cases) for cases in groups.values()),
        "cases_rivalled": 0,
        "returned": 0,
        "repaired_groups": 0,
        "rivalled_case_ids": [],
        "dropped": [],
        "divergence": [],
        "models": {"original": original_author, "rival": {"role": role_name, "model": role.model}},
        "rivalled_fields": list(RIVALLED_FIELDS),
        "pinned_fields": list(PINNED_FIELDS),
        "warnings": [],
    }
    if not groups:
        report["warnings"].append("no cases matched the requested families; nothing was rivalled")
        return [], report

    spec_text = render_for_reviewer(spec)
    debug_dir = (usage_path.parent / "debug") if usage_path is not None else None
    own_client = client is None
    model_client = client or ModelClient.from_config(config, usage_path, "suite.rival")

    async def run_group(key: tuple[str, str], cases: list[Case]) -> dict[str, Any]:
        family_id, task = key
        family = suite.family(family_id)
        label = f"rival_{family_id}_{task}"
        message = rival_group_message(spec, suite, cases, spec_text=spec_text)
        outcome: dict[str, Any] = {
            "cases": [], "dropped": [], "returned": 0, "repaired": False
        }
        items = await _request_items(
            model_client,
            role,
            message,
            wanted=len(cases),
            keys=("rubrics", "rubric", "cases", "items"),
            looks_like_item=_looks_like_rubric,
            json_shape=suite_prompts.RIVAL_RUBRIC_JSON_SHAPE,
            label=label,
            record_id=f"{family_id}:{task}",
            debug_dir=debug_dir,
        )
        outcome["returned"] = len(items)
        by_variant = _index_by_variant(items, [case.variant for case in cases])

        failed: dict[str, tuple[dict[str, Any], list[str]]] = {}
        for case in cases:
            item = by_variant.get(case.variant)
            if item is None:
                outcome["dropped"].append(
                    {
                        "case_id": case.case_id,
                        "family_id": family_id,
                        "task": task,
                        "variant": case.variant,
                        "problems": ["the rival author returned no rubric for this variant"],
                    }
                )
                continue
            built, problems = _try_build(case, item, family, spec, plan)
            if problems:
                failed[case.variant] = (item, problems)
            elif built is not None:
                outcome["cases"].append(built)

        if failed:
            outcome["repaired"] = True
            repaired_items = await _repair_items(
                model_client,
                role,
                message,
                [item for item, _ in failed.values()],
                errors=[problem for _, problems in failed.values() for problem in problems],
                subjects=[f"variant {variant}" for variant in failed],
                keys=("rubrics", "rubric", "cases", "items"),
                looks_like_item=_looks_like_rubric,
                label=label,
                record_id=f"{family_id}:{task}:repair",
                debug_dir=debug_dir,
                stage="suite.rival.repair",
            )
            repaired = _index_by_variant(repaired_items, list(failed))
            by_id = {case.variant: case for case in cases}
            for variant, (_item, problems) in failed.items():
                replacement = repaired.get(variant)
                case = by_id[variant]
                if replacement is None:
                    outcome["dropped"].append(
                        {
                            "case_id": case.case_id,
                            "family_id": family_id,
                            "task": task,
                            "variant": variant,
                            "problems": problems + ["repair returned no replacement"],
                        }
                    )
                    continue
                built, still = _try_build(case, replacement, family, spec, plan)
                if still or built is None:
                    outcome["dropped"].append(
                        {
                            "case_id": case.case_id,
                            "family_id": family_id,
                            "task": task,
                            "variant": variant,
                            "problems": still,
                            "first_attempt_problems": problems,
                        }
                    )
                else:
                    outcome["cases"].append(built)
        return outcome

    try:
        results = await gather_bounded(
            [run_group(key, cases) for key, cases in groups.items()]
        )
    finally:
        if own_client:
            await model_client.aclose()

    rivals: list[Case] = []
    for (key, cases), result in zip(groups.items(), results):
        if isinstance(result, Exception):
            logger.error("rival group %s/%s failed: %s", key[0], key[1], result)
            report["warnings"].append(f"rival group {key[0]}/{key[1]} failed: {result}")
            report["dropped"].append(
                {
                    "family_id": key[0],
                    "task": key[1],
                    "variant": "(all)",
                    "problems": [f"the rival call raised {type(result).__name__}: {result}"],
                }
            )
            continue
        rivals.extend(result["cases"])
        report["returned"] += result["returned"]
        report["dropped"] += result["dropped"]
        report["repaired_groups"] += int(result["repaired"])

    for case in rivals:
        original = suite.case(case.case_id)
        report["divergence"].append(
            {
                "case_id": case.case_id,
                "family_id": case.family_id,
                "task": case.task,
                "variant": case.variant,
                "divergence": rubric_divergence(original.rubric, case.rubric),
                "original_rubric_version": original.rubric.version,
                "rival_rubric_version": case.rubric.version,
            }
        )
    report["cases_rivalled"] = len(rivals)
    report["rivalled_case_ids"] = [case.case_id for case in rivals]
    report["divergence_summary"] = divergence_summary(report["divergence"])
    identical = [row["case_id"] for row in report["divergence"] if row["divergence"] == 0.0]
    if identical:
        report["warnings"].append(
            f"{len(identical)} rival rubric(s) are word-for-word the original; check that the "
            "rival author was not shown the first version"
        )
    low = report["divergence_summary"].get("below_low_divergence", 0)
    if low:
        report["warnings"].append(
            f"{low} rival rubric(s) diverge less than {LOW_DIVERGENCE} from the original, so "
            "the comparison carries little signal for those cases"
        )
    logger.info(
        "rival: %d cases across %d families rewritten by %s (%d dropped, median divergence %s)",
        len(rivals),
        len(report["rivalled_families"]),
        role.model,
        len(report["dropped"]),
        report["divergence_summary"].get("median"),
    )
    return rivals, report


def _looks_like_rubric(item: Any) -> bool:
    return isinstance(item, dict) and bool(
        item.get("must_notice") or item.get("anchors") or item.get("acceptable_outputs")
    )


def _index_by_variant(
    items: Sequence[dict[str, Any]], order: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Label rubrics by variant, falling back to the order they were asked for."""
    indexed: dict[str, dict[str, Any]] = {}
    unlabelled: list[dict[str, Any]] = []
    for item in items:
        variant = _clean(item.get("variant"))
        if variant in order and variant not in indexed:
            indexed[variant] = item
        elif variant not in order:
            unlabelled.append(item)
    for variant, item in zip([v for v in order if v not in indexed], unlabelled):
        indexed[variant] = item
    return indexed


def _try_build(
    case: Case, item: dict[str, Any], family: Any, spec: TargetSpec, plan: SuitePlan
) -> tuple[Case | None, list[str]]:
    """Swap in the rival rubric and hold it to the bar the original was held to."""
    try:
        rebuilt = replace(case, rubric=_rival_rubric(item, case.rubric))
        problems = rebuilt.validate() + extra_case_problems(rebuilt, family, spec, plan)
    except Exception as error:
        # One malformed rubric must not discard the rest of an answered group.
        logger.exception("%s: building the rival rubric raised", case.case_id)
        return None, [f"building the rival rubric raised {type(error).__name__}: {error}"]
    return (None, problems) if problems else (rebuilt, [])


def divergence_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Whether the rival wrote something genuinely its own, which bounds the signal."""
    values = sorted(float(row["divergence"]) for row in rows)
    if not values:
        return {"cases": 0}
    return {
        "cases": len(values),
        "min": values[0],
        "median": values[len(values) // 2],
        "max": values[-1],
        "mean": round(sum(values) / len(values), 4),
        "identical": sum(1 for value in values if value == 0.0),
        "below_low_divergence": sum(1 for value in values if value < LOW_DIVERGENCE),
        "low_divergence_threshold": LOW_DIVERGENCE,
    }


#: What a score was graded against. `unknown` means the result predates the rival pass or
#: names a rubric version neither standard produced, which is a defect, not a third standard.
ORIGINAL, RIVAL, UNKNOWN = "original", "rival", "unknown"


def standard_labels(report: dict[str, Any]) -> dict[str, dict[str, str]]:
    """{case_id: {rubric_version: "original" | "rival"}}, from a rival report.

    A CaseResult records `rubric_version` but nothing naming which standard produced it, so
    scores graded against two different standards would otherwise pool into one mean in
    silence. That is the worst available failure: it would not error, it would not look odd,
    and it would quietly average a measurement with its own control.

    Keyed by case first because a rival rubric that came back word for word identical shares
    its original's version hash. Same version, same case, same score, so the label is
    ambiguous only where it cannot matter, and `divergence_summary` already warns when that
    happens at all.
    """
    labels: dict[str, dict[str, str]] = {}
    for row in report.get("divergence") or []:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            continue
        versions = labels.setdefault(case_id, {})
        original = str(row.get("original_rubric_version", ""))
        rival = str(row.get("rival_rubric_version", ""))
        if original:
            versions[original] = ORIGINAL
        if rival:
            # An identical rival must not overwrite the original's label: where the two hashes
            # collide the standards are the same text, so `original` is the truthful answer.
            versions.setdefault(rival, RIVAL)
    return labels


def label_result(
    labels: dict[str, dict[str, str]], case_id: str, rubric_version: str
) -> str:
    """Which standard one CaseResult was graded against. Never guesses."""
    return labels.get(case_id, {}).get(rubric_version, UNKNOWN)


def rival_suite(suite: Suite, rivals: Sequence[Case]) -> Suite:
    """The same suite with the rival standard swapped in, for the second judging pass.

    Only the rivalled cases change; every other case keeps its original rubric, so a
    comparison must be restricted to `report["rivalled_case_ids"]`. Delegates to the authoring
    module so a rival suite is assembled and validated exactly as any other suite is.
    """
    from persona_eval.suite.author import replace_cases

    return replace_cases(suite, rivals)


__all__ = [
    "LOW_DIVERGENCE",
    "ORIGINAL",
    "RIVAL",
    "UNKNOWN",
    "label_result",
    "standard_labels",
    "PINNED_FIELDS",
    "RIVALLED_FIELDS",
    "RivalError",
    "author_rival_rubrics",
    "divergence_summary",
    "rival_group_message",
    "rival_suite",
    "rubric_divergence",
    "rubric_text_of",
    "select_rival_families",
]
