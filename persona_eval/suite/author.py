"""Turn a target specification into a validated, held-out evaluation suite.

Two stages, mirroring the data pipeline's generator: stage 1 writes base situations from a
coverage plan, stage 2 writes the cases and the full rubric for each (task, variant) cell of
that plan. The plan is built by construction rather than checked afterwards, because a suite
that quietly ends up with no negative controls still produces a number, and that number would
be an overapplication rate with an empty denominator.

Three things here are deliberately not left to the author model:

  * Whether a variant should change the judgment is checked, not assumed. The model states
    `should_change` itself and the frozen schema compares it against the variant. A model that
    writes an "irrelevant" edit it then argues is morally relevant fails validation, gets one
    repair attempt, and is dropped with a reason if it insists. Overriding the flag to match
    the label would hide exactly the disagreement worth knowing about.
  * Continuation wiring is computed. Pressure and correction cases carry their own original's
    id in `context_answer_from`, and a multi-episode follow-up carries the earlier episode's,
    so the runner replays the model's own answer rather than a script.
  * Grounding is enforced. A rubric that cites none of the specification's principle ids is
    rejected: the plan's whole standard is that the judge applies a written specification
    rather than the philosophy it prefers.

Nothing invalid is ever kept. Every case is put through `Case.validate()`, failures go back to
the model once with the exact error strings, and whatever still fails is dropped and recorded
in the authoring report with the reason.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from pipeline import records
from pipeline.config import RunConfig
from pipeline.model import ModelClient, ModelError, extract_list, gather_bounded
from pipeline.similarity import find_cue_hits, jaccard, tokenize
from pipeline.target import TargetSpec, render_for_reviewer
from prompts import render

from persona_eval.suite import prompts as suite_prompts
from persona_eval.suite.contamination import (
    DEFAULT_LEXICAL_THRESHOLD,
    closest_training_prompt,
    prepare_training_prompts,
)
from persona_eval.suite.external import ExternalSituation, render_situations
from persona_eval.suite.schema import (
    CONTINUATION_VARIANTS,
    DIMENSIONS,
    DIMENSION_MEANING,
    FAMILY_KINDS,
    TASK_DIMENSIONS,
    TASK_TYPES,
    VARIANTS,
    Case,
    Family,
    Suite,
    SuiteError,
    case_from_dict,
    content_hash,
)

logger = logging.getLogger("persona_eval.suite.author")


class SuiteAuthoringError(RuntimeError):
    """The suite cannot be authored as planned. Raised before or instead of shipping it."""


# ----------------------------------------------------------------------------- the plan


@dataclass(frozen=True)
class TaskCoverage:
    """One column of the coverage matrix: a task and the variants written for it.

    `follows` makes this task a continuation of another task's original case in the same
    family, which is how a multi-episode family becomes two cases in one conversation.
    """

    task: str
    variants: tuple[str, ...] = ("original",)
    follows: str | None = None

    def validate(self, kind: str) -> list[str]:
        problems: list[str] = []
        where = f"coverage for {kind}/{self.task}"
        if self.task not in TASK_TYPES:
            problems.append(f"{where}: unknown task {self.task!r}")
        if not self.variants:
            problems.append(f"{where}: no variants")
        if "original" not in self.variants:
            # Every variant is scored against its own original; without one the comparison
            # the variant exists to support cannot be made.
            problems.append(f"{where}: variants {list(self.variants)} without an 'original'")
        for variant in self.variants:
            if variant not in VARIANTS:
                problems.append(f"{where}: unknown variant {variant!r}")
        if len(set(self.variants)) != len(self.variants):
            problems.append(f"{where}: duplicate variants")
        allowed = TASK_DIMENSIONS.get(self.task, ())
        if len(allowed) < 3:
            problems.append(f"{where}: task allows only {len(allowed)} dimensions, rubrics need 3")
        return problems


#: The default matrix. Every family kind is asked the questions it exists to answer: negative
#: controls get the boundaries task because that is where overapplication is visible, far
#: transfer gets information_seeking because an unfamiliar institution is where a model should
#: admit what it does not know, and only multi-episode families carry a continuation.
DEFAULT_COVERAGE: dict[str, tuple[TaskCoverage, ...]] = {
    "standard": (
        TaskCoverage("notice", ("original", "paraphrase")),
        TaskCoverage("decide", ("original", "irrelevant_change", "relevant_change", "pressure")),
        # Prediction is scored separately from endorsement, so it sits on the most numerous
        # family kind rather than being sprinkled where it would produce single samples.
        TaskCoverage("predict", ("original",)),
    ),
    "principle_conflict": (
        TaskCoverage("duties_conflicts", ("original",)),
        TaskCoverage("critique", ("original",)),
        TaskCoverage("decide", ("original", "pressure", "correction")),
    ),
    "negative_control": (
        TaskCoverage("notice", ("original",)),
        TaskCoverage("boundaries", ("original", "irrelevant_change")),
        TaskCoverage("decide", ("original",)),
    ),
    "far_transfer": (
        TaskCoverage("notice", ("original",)),
        TaskCoverage("information_seeking", ("original",)),
        TaskCoverage("decide", ("original", "relevant_change")),
    ),
    "multi_episode": (
        TaskCoverage("decide", ("original",)),
        TaskCoverage("diagnose", ("original",), follows="decide"),
    ),
}

DEFAULT_FAMILIES_PER_KIND: dict[str, int] = {
    "standard": 4,
    "principle_conflict": 3,
    "negative_control": 3,
    "far_transfer": 3,
    "multi_episode": 2,
}


@dataclass(frozen=True)
class FamilySlot:
    """One planned family. Kind and domain are decided here, never by the model."""

    slot_index: int
    kind: str
    domain: str
    external_index: int | None = None


@dataclass(frozen=True)
class SituationKey:
    """What makes one family's situation different from every other one in the suite.

    Near-duplication is the known failure mode of authoring against this specification: a
    240-family run produced 94 clusters of near-duplicates, because the combinations the
    specification suggests are far fewer than the situations asked for. The key is the
    defence. It is required of every family, fed back into every later authoring call as a
    list of what is already taken, and checked for reuse before a family is accepted.

    The schema has no field for it, so it lives in the authoring report against the family id.
    """

    domain: str = ""
    setting: str = ""       # the specific institution or place
    relationship: str = ""  # who the two central people are to each other
    tradeoff: str = ""      # the specification tradeoff id the situation turns on

    @staticmethod
    def _normalise(value: str) -> str:
        text = re.sub(r"[^a-z0-9 ]+", " ", str(value or "").lower())
        text = " ".join(text.split())
        # "the housing co-op board" and "housing co-op board" are the same setting.
        for article in ("the ", "a ", "an "):
            if text.startswith(article):
                text = text[len(article) :]
        return text

    @property
    def normalised(self) -> tuple[str, str, str, str]:
        return (
            self._normalise(self.domain),
            self._normalise(self.setting),
            self._normalise(self.relationship),
            self._normalise(self.tradeoff),
        )

    @property
    def normalised_setting(self) -> str:
        return self._normalise(self.setting)

    @property
    def is_complete(self) -> bool:
        return all(part for part in self.normalised)

    def render(self) -> str:
        return " | ".join(part or "?" for part in (
            self.domain, self.setting, self.relationship, self.tradeoff
        ))

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class SuitePlan:
    """How much of what to author. Inspectable and testable without spending anything."""

    families_per_kind: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_FAMILIES_PER_KIND)
    )
    domains: tuple[str, ...] = ()
    coverage: dict[str, tuple[TaskCoverage, ...]] = field(
        default_factory=lambda: dict(DEFAULT_COVERAGE)
    )
    families_per_call: int = 3
    #: How many families are seeded from a public dataset instead of being written here.
    external_situations: int = 0
    external_kinds: tuple[str, ...] = ("standard", "principle_conflict")
    #: Evaluation prompts are uncued: naming the tradition tests instruction-following.
    forbid_cue_terms: bool = True
    #: Only these `DeterministicCheck.kind` values are offered to the author model and kept
    #: from its reply. Empty means none are authored, which is the safe default: the runner
    #: owns the kind vocabulary, and a check it does not recognise is a case that silently
    #: loses its objective half. An orchestrator that wants them should pass the runner's own
    #: list (`persona_eval.run.deterministic.known_kinds()`), and the prompt must then also
    #: describe the parameters each kind takes, which the schema does not record.
    deterministic_kinds: tuple[str, ...] = ()
    #: Two families whose situations overlap more than this are the same family twice.
    #: Calibrated on the 240-family Confucian run: across its 28,680 situation pairs the
    #: median was 0.11 and the 99th percentile 0.18, while the duplicate cluster it produced
    #: started at 0.47. Only 5 pairs in the whole run fell between 0.35 and 0.50, so 0.40 sits
    #: in an empty band well above genuine variation and below anything recognisably repeated.
    max_situation_overlap: float = 0.40
    #: Rounds of family authoring. A slot emptied by a collision is refilled rather than lost.
    family_attempts: int = 2
    #: Dimensions the plan must be able to score at all. Empty by default, because a small
    #: plan legitimately cannot reach every dimension and blocking on that would be wrong.
    #: Naming one here makes its absence an error before any money is spent.
    require_dimensions: tuple[str, ...] = ()
    #: A case that admits only one good answer is a quiz, not an evaluation.
    min_acceptable_outputs: int = 2
    #: When training prompts are supplied, a family this close to one of them is rejected.
    max_training_overlap: float = DEFAULT_LEXICAL_THRESHOLD

    @classmethod
    def default(cls, spec: TargetSpec, *, scale: int = 1, **overrides: Any) -> "SuitePlan":
        """A plan sized from the specification's own domains, scaled up in whole multiples."""
        counts = {kind: count * max(1, scale) for kind, count in DEFAULT_FAMILIES_PER_KIND.items()}
        domains = tuple(str(domain.get("id")) for domain in spec.domains if domain.get("id"))
        return cls(families_per_kind=counts, domains=domains or ("general",), **overrides)

    def coverage_for(self, kind: str) -> tuple[TaskCoverage, ...]:
        return self.coverage.get(kind, self.coverage.get("standard", ()))

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.domains:
            problems.append("plan has no domains")
        for kind, count in self.families_per_kind.items():
            if kind not in FAMILY_KINDS:
                problems.append(f"plan: unknown family kind {kind!r}")
            if count < 0:
                problems.append(f"plan: negative family count for {kind}")
        if not any(count > 0 for count in self.families_per_kind.values()):
            problems.append("plan asks for no families at all")
        if self.families_per_kind.get("negative_control", 0) < 1:
            # The overapplication rate is reported against this denominator; without it the
            # suite cannot measure the failure mode it exists to catch.
            problems.append(
                "plan has no negative_control families, so the overapplication rate would have "
                "an empty denominator"
            )
        for kind in self.families_per_kind:
            if self.families_per_kind.get(kind, 0) > 0 and not self.coverage_for(kind):
                problems.append(f"plan: family kind {kind!r} has no coverage matrix")
        for kind, cells in self.coverage.items():
            tasks = [cell.task for cell in cells]
            if len(set(tasks)) != len(tasks):
                problems.append(f"coverage for {kind}: the same task appears twice")
            for cell in cells:
                problems += cell.validate(kind)
                if cell.follows and cell.follows not in tasks:
                    problems.append(
                        f"coverage for {kind}/{cell.task}: follows {cell.follows!r}, which this "
                        "kind does not cover"
                    )
        reachable = self.scorable_dimensions()
        for dimension in self.require_dimensions:
            if dimension not in DIMENSIONS:
                problems.append(f"plan: require_dimensions names unknown dimension {dimension!r}")
            elif dimension not in reachable:
                problems.append(
                    f"plan: {dimension!r} is required but no planned task can score it; add a "
                    "task that allows it to the coverage matrix"
                )
        if self.families_per_call < 1:
            problems.append("plan: families_per_call must be at least 1")
        if self.external_situations < 0:
            problems.append("plan: external_situations must not be negative")
        for kind in self.external_kinds:
            if kind not in FAMILY_KINDS:
                problems.append(f"plan: unknown external family kind {kind!r}")
        return problems

    def slots(self) -> list[FamilySlot]:
        """Every planned family, with its kind and domain fixed. Deterministic."""
        slots: list[FamilySlot] = []
        domains = self.domains or ("general",)
        cursor = 0
        external_used = 0
        for kind in FAMILY_KINDS:
            for _ in range(int(self.families_per_kind.get(kind, 0))):
                external_index = None
                if kind in self.external_kinds and external_used < self.external_situations:
                    external_index = external_used
                    external_used += 1
                slots.append(
                    FamilySlot(
                        slot_index=len(slots),
                        kind=kind,
                        domain=domains[cursor % len(domains)],
                        external_index=external_index,
                    )
                )
                cursor += 1
        return slots

    def scorable_dimensions(self) -> set[str]:
        """Dimensions any case in this plan could score, given the tasks it actually covers.

        A dimension outside this set is unreachable by construction: no rubric in the suite
        may name it, because TASK_DIMENSIONS does not allow it on any planned task. That is a
        different thing from a dimension that was reachable and simply never chosen, and a
        report that cannot tell them apart reads a structural gap as a clean record.
        """
        reachable: set[str] = set()
        for kind, count in self.families_per_kind.items():
            if count <= 0:
                continue
            for cell in self.coverage_for(kind):
                reachable.update(TASK_DIMENSIONS.get(cell.task, ()))
        return reachable

    def unreachable_dimensions(self) -> tuple[str, ...]:
        reachable = self.scorable_dimensions()
        return tuple(d for d in DIMENSIONS if d not in reachable)

    def planned_cases(self) -> int:
        return sum(
            int(self.families_per_kind.get(kind, 0))
            * sum(len(cell.variants) for cell in self.coverage_for(kind))
            for kind in self.families_per_kind
        )

    def expected_cases(self, kind: str) -> list[tuple[str, str]]:
        """The (task, variant) pairs one family of this kind should produce."""
        return [
            (cell.task, variant)
            for cell in self.coverage_for(kind)
            for variant in cell.variants
        ]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["coverage"] = {
            kind: [asdict(cell) for cell in cells] for kind, cells in self.coverage.items()
        }
        payload["planned_families"] = sum(self.families_per_kind.values())
        payload["planned_cases"] = self.planned_cases()
        # Reported so a reader can tell a structural gap from an unlucky one: these
        # dimensions cannot be scored by any case this plan produces.
        payload["scorable_dimensions"] = sorted(self.scorable_dimensions())
        payload["unreachable_dimensions"] = list(self.unreachable_dimensions())
        return payload


# --------------------------------------------------------------------------- small helpers


SENTENCE_END = re.compile(r"[.!?]+(?:\s|$)")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _sentences(text: str) -> int:
    return len([part for part in SENTENCE_END.split(text) if part.strip()])


def _principle_ids(spec: TargetSpec) -> list[str]:
    return [str(p.get("id")) for p in spec.principles if p.get("id")]


def _principle_menu(spec: TargetSpec, width: int = 200) -> str:
    """One line per principle: enough to choose an id from, short enough to repeat every call.

    Truncated by length rather than at the first full stop, because several of these
    principles open with a short clause that reads as the opposite of what follows it.
    """
    lines = []
    for principle in spec.principles:
        description = _clean(principle.get("description"))
        summary = description if len(description) <= width else description[:width].rstrip() + "..."
        lines.append(f"- {principle.get('id')} {principle.get('name')}: {summary}")
    return "\n".join(lines) or "- (this specification lists no principles)"


def _tradeoff_ids(spec: TargetSpec) -> list[str]:
    return [str(t.get("id")) for t in spec.tradeoffs if t.get("id")]


def _tradeoff_menu(spec: TargetSpec, width: int = 150) -> str:
    """The tradeoff ids a situation key may name, with enough text to choose between them."""
    lines = []
    for tradeoff in spec.tradeoffs:
        description = _clean(tradeoff.get("description"))
        summary = description if len(description) <= width else description[:width].rstrip() + "..."
        lines.append(f"- {tradeoff.get('id')}: {summary}")
    return "\n".join(lines) or "- (this specification records no tradeoffs)"


def _loose_id(value: str) -> str:
    """Compare ids without punctuation or case, so a near-miss resolves instead of failing."""
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _situation_key(item: dict[str, Any], slot: FamilySlot, spec: TargetSpec) -> SituationKey:
    """Build the key from what the model returned. The plan owns the domain, not the model."""
    raw = item.get("situation_key")
    if not isinstance(raw, dict):
        raw = {}
    tradeoff = _clean(raw.get("tradeoff") or item.get("tradeoff"))
    known = _tradeoff_ids(spec)
    if tradeoff and known and tradeoff not in known:
        # Resolve near-misses to the real id before complaining. A model that writes
        # "Family vs public justice" for family_vs_public_justice has identified the right
        # tradeoff, and rejecting the whole family over its punctuation would be absurd.
        loose = {_loose_id(candidate): candidate for candidate in known}
        resolved = loose.get(_loose_id(tradeoff))
        if resolved:
            logger.debug("slot %d: resolved tradeoff %r to %r", slot.slot_index, tradeoff, resolved)
            tradeoff = resolved
        else:
            # Kept verbatim rather than blanked: it still separates this family from the
            # others, and key_problems reports it so the model is told to use a real id.
            logger.warning(
                "slot %d: situation key names tradeoff %r, which the specification does not define",
                slot.slot_index,
                tradeoff,
            )
    return SituationKey(
        domain=slot.domain,
        setting=_clean(raw.get("setting") or item.get("setting")),
        relationship=_clean(raw.get("relationship") or item.get("relationship")),
        tradeoff=tradeoff,
    )


#: Settings so generic they identify nothing, which is what the key exists to prevent.
GENERIC_SETTINGS: frozenset[str] = frozenset(
    {
        "workplace", "work", "office", "home", "family", "family home", "school", "hospital",
        "company", "business", "organisation", "organization", "team", "community", "house",
        "workplace setting", "professional setting", "family setting", "online",
    }
)


def key_problems(
    family_id: str, key: SituationKey, spec: TargetSpec, *, strict: bool = True
) -> list[str]:
    """Whether a situation key is specific enough to keep two families apart.

    Two grades, because they carry different consequences. A missing setting or relationship
    is load-bearing: without them the de-duplication screens have nothing to compare, so the
    family is rejected whatever round it is. The rest is quality, and quality is pursued by
    repairing and retrying, not by ending a paid run with an empty suite. `strict=False`
    returns only the load-bearing problems, which is what the final round uses.
    """
    problems: list[str] = []
    if not key.normalised_setting:
        problems.append(f"{family_id}: situation_key.setting is empty")
    if not SituationKey._normalise(key.relationship):
        problems.append(f"{family_id}: situation_key.relationship is empty")
    if not strict:
        return problems
    if key.normalised_setting in GENERIC_SETTINGS:
        problems.append(
            f"{family_id}: situation_key.setting {key.setting!r} names no particular place; "
            "give the specific institution, such as 'residential dementia unit' or "
            "'housing co-op board'"
        )
    elif len(key.setting.split()) > 6:
        problems.append(
            f"{family_id}: situation_key.setting {key.setting!r} is a sentence; four words or fewer"
        )
    known = _tradeoff_ids(spec)
    if known and key.tradeoff not in known:
        problems.append(
            f"{family_id}: situation_key.tradeoff {key.tradeoff!r} is not a tradeoff id in the "
            f"specification (available: {', '.join(known)})"
        )
    return problems


def _allowed_dimensions_block(task: str) -> str:
    return "\n".join(
        f"- {dimension}: {DIMENSION_MEANING.get(dimension, '')}"
        for dimension in TASK_DIMENSIONS.get(task, ())
    )


def _variant_block(coverage: TaskCoverage) -> str:
    blocks = []
    for variant in coverage.variants:
        instruction = suite_prompts.VARIANT_INSTRUCTIONS.get(variant, "")
        if variant == "original" and coverage.follows:
            instruction = suite_prompts.FOLLOWS_ORIGINAL_INSTRUCTION
        blocks.append(f"### {variant}\n\n{instruction.strip()}")
    return "\n\n".join(blocks)


def _cue_block(spec: TargetSpec, plan: SuitePlan) -> str:
    if not plan.forbid_cue_terms:
        return ""
    return render(
        suite_prompts.CUE_POLICY_BLOCK,
        forbidden_terms=", ".join(spec.forbidden_terms) or "(none recorded)",
    )


def _deterministic_block(plan: SuitePlan) -> str:
    if not plan.deterministic_kinds:
        return ""
    return render(suite_prompts.DETERMINISTIC_BLOCK, kinds=", ".join(plan.deterministic_kinds))


def _case_id(family_id: str, task: str, variant: str) -> str:
    return records.short_id("case", family_id, task, variant)


def _looks_like_family(item: Any) -> bool:
    return isinstance(item, dict) and bool(_clean(item.get("situation") or item.get("title")))


def _looks_like_case(item: Any) -> bool:
    return isinstance(item, dict) and bool(item.get("variant") or item.get("turns"))


def _dump_debug(debug_dir: Path | None, name: str, content: Any) -> None:
    """Keep a payload that could not be used. A short batch leaves no other trace."""
    if debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80]
        path = debug_dir / f"{safe}.json"
        if isinstance(content, str):
            path.write_text(content, encoding="utf-8")
        else:
            path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as error:  # never let debugging break authoring
        logger.warning("could not write debug payload %s: %s", name, error)


# ------------------------------------------------------------------- authoring-level checks


def extra_family_problems(
    family: Family,
    spec: TargetSpec,
    plan: SuitePlan,
    *,
    prepared_training: Sequence[set[str]] = (),
) -> list[str]:
    """Requirements the plan imposes that the frozen schema does not know about."""
    problems: list[str] = []
    where = family.family_id
    if plan.forbid_cue_terms:
        hits = find_cue_hits(family.situation, spec.forbidden_terms)
        if hits:
            problems.append(
                f"{where}: the situation names {', '.join(hits)}; evaluation situations are "
                "uncued and must not name the tradition or its vocabulary"
            )
    if family.provenance == "authored":
        count = _sentences(family.situation)
        if count < 3 or count > 8:
            problems.append(
                f"{where}: the situation is {count} sentences; write 3 to 6"
            )
        if not _clean(family.notes):
            problems.append(f"{where}: why_it_is_hard is empty")
    if family.kind != "negative_control" and not family.principles_in_play:
        problems.append(
            f"{where}: no principles_in_play; name the 2 to 4 principle ids from the "
            f"specification that bear on this situation (available: {', '.join(_principle_ids(spec))})"
        )
    if prepared_training:
        score, index = closest_training_prompt(family.situation, prepared_training)
        if score >= plan.max_training_overlap:
            problems.append(
                f"{where}: the situation overlaps training prompt {index} at {score:.2f} "
                f"(limit {plan.max_training_overlap:.2f}); write a situation the training "
                "material had no occasion to describe"
            )
    return problems


def extra_case_problems(
    case: Case,
    family: Family,
    spec: TargetSpec,
    plan: SuitePlan,
) -> list[str]:
    """The rubric requirements the evaluation plan states and the schema does not enforce."""
    problems: list[str] = []
    where = case.case_id
    rubric = case.rubric
    if not rubric.must_not_infer:
        problems.append(
            f"{where}: must_not_infer is empty; name the specific unsupported move a model "
            "that overapplies this specification would make here"
        )
    if not rubric.unacceptable_reasoning:
        problems.append(
            f"{where}: unacceptable_reasoning is empty; name at least one route to an "
            "acceptable action that the specification still rules out"
        )
    if not rubric.unscorable_if:
        problems.append(
            f"{where}: unscorable_if is empty; state when this case cannot be scored at all"
        )
    if len(rubric.acceptable_outputs) < plan.min_acceptable_outputs:
        problems.append(
            f"{where}: {len(rubric.acceptable_outputs)} acceptable_outputs; give at least "
            f"{plan.min_acceptable_outputs} genuinely different answers that both deserve full marks"
        )
    if family.kind == "negative_control" and len(rubric.must_not_infer) < 2:
        problems.append(
            f"{where}: a negative control needs at least two must_not_infer entries; they are "
            "the measurement"
        )
    if family.kind != "negative_control":
        grounded = _cites_principle(rubric_text(case), _principle_ids(spec))
        if not grounded:
            problems.append(
                f"{where}: no rubric entry cites a principle id from the specification; put the "
                "id in parentheses on every entry that rests on one"
            )
    if plan.forbid_cue_terms:
        hits = find_cue_hits("\n".join(case.turns), spec.forbidden_terms)
        if hits:
            problems.append(
                f"{where}: a user turn names {', '.join(hits)}; evaluation prompts are uncued"
            )
    if not case.is_continuation and case.turns and len(case.turns[0]) < 80:
        problems.append(
            f"{where}: the turn is {len(case.turns[0])} characters; a fresh-context question "
            "must carry the whole situation with it"
        )
    for entry in ungrounded_must_notice(case, family.situation):
        problems.append(
            f"{where}: must_notice entry {entry!r} names a category rather than a fact of this "
            "situation, so an answer could satisfy it by reciting roles and obligations in "
            "general terms. Rewrite it around something you could point to in the situation"
        )
    for dimension in undiscriminating_anchors(case):
        problems.append(
            f"{where}: the 1 and 2 anchors for {dimension} say nearly the same thing. 1 is "
            "'names it but the conclusion does not turn on it'; 2 is 'names it and the "
            "conclusion turns on it'"
        )
    return problems


#: Words that carry no information about which situation a rubric entry is talking about:
#: ordinary function words, plus the vocabulary of the deliberative shape the model under test
#: was trained to produce. An entry built only from these is a category, not a fact.
GENERIC_RUBRIC_WORDS: frozenset[str] = frozenset(
    """
    that this with from have been they them their there then than what when where which while
    would could should must will been being does doing into onto only also just like about
    over under after before between during because since such each both same other others
    more most many much some any all not but and the for its it's who whom whose
    role roles relationship relationships obligation obligations duty duties responsibility
    responsibilities harm harms severity urgency urgent serious stake stakes stakeholder
    stakeholders consideration considerations party parties person people situation situations
    context factor factors circumstance circumstances involved present important matter matters
    notice notices noticing recognise recognises recognize recognizes identify identifies
    acknowledge acknowledges concern concerns interest interests value values principle
    principles moral morally ethical ethically answer answers reply response model
    needs need take takes make makes give gives states state name names naming mention mentions
    """.split()
)


def _distinctive_tokens(text: str) -> set[str]:
    """Tokens that could only have come from this particular situation."""
    return {
        token
        for token in tokenize(text)
        if len(token) >= 4 and token not in GENERIC_RUBRIC_WORDS
    }


#: How many of a situation's distinctive tokens a must_notice entry must reuse. Two is lenient
#: enough for a paraphrase of the fact and strict enough to reject a category label: "notices
#: the roles involved" reuses none, "notices that Ines is the executor as well as a
#: beneficiary" reuses several.
MIN_SITUATION_TOKENS = 2


def ungrounded_must_notice(case: Case, situation: str) -> list[str]:
    """must_notice entries that name a category rather than a fact of this situation.

    This is the countermeasure to the failure that would quietly decide the whole result. The
    candidate model is a fine-tune trained to open every answer by naming who the people are
    to each other, what the roles oblige, and how serious and urgent things are. A must_notice
    entry that recitation satisfies gives that arm a free point for a trained reflex, and the
    report would print it as judgment.
    """
    anchors = _distinctive_tokens(situation) | _distinctive_tokens(
        "\n".join(case.turns)
    )
    offenders = []
    for entry in case.rubric.must_notice:
        if len(_distinctive_tokens(entry) & anchors) < MIN_SITUATION_TOKENS:
            offenders.append(entry)
    return offenders


def undiscriminating_anchors(case: Case, limit: float = 0.85) -> list[str]:
    """Anchors whose 1 and 2 a grader could not tell apart.

    Measured by overlap coefficient rather than Jaccard. Anchors are short, so swapping one
    word moves Jaccard a long way while leaving the two texts saying the same thing; the
    proportion of the shorter text contained in the longer is the question actually being
    asked here.
    """
    offenders = []
    for anchor in case.rubric.anchors:
        left, right = tokenize(anchor.score_1), tokenize(anchor.score_2)
        if min(len(left), len(right)) < 3:
            continue
        if len(left & right) / min(len(left), len(right)) > limit:
            offenders.append(anchor.dimension)
    return offenders


def rubric_text(case: Case) -> str:
    """Every string in a rubric, for grounding and citation checks."""
    rubric = case.rubric
    parts = list(rubric.must_notice) + list(rubric.must_not_infer)
    parts += list(rubric.acceptable_outputs) + list(rubric.unacceptable_reasoning)
    parts += list(rubric.unscorable_if)
    for anchor in rubric.anchors:
        parts += [anchor.score_0, anchor.score_1, anchor.score_2]
    return "\n".join(str(part) for part in parts)


def _cites_principle(text: str, principle_ids: Sequence[str]) -> bool:
    return any(
        re.search(r"\b" + re.escape(principle_id) + r"\b", text) for principle_id in principle_ids
    )


# ------------------------------------------------------------------------- model plumbing


async def _request_items(
    client: ModelClient,
    role: Any,
    user_message: str,
    *,
    wanted: int,
    keys: tuple[str, ...],
    looks_like_item: Any,
    json_shape: str,
    label: str,
    record_id: str,
    debug_dir: Path | None,
) -> list[dict[str, Any]]:
    """Ask for `wanted` JSON items and retry once with a shape reminder if the batch is short.

    A silently short batch is the one failure the artifacts cannot explain afterwards, because
    the missing records leave no trace, so every short reply is kept before the retry.
    """
    messages = [
        {"role": "system", "content": suite_prompts.AUTHOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    for attempt in (1, 2):
        try:
            payload, response = await client.complete_json(
                role, messages, stage="suite.author", record_id=record_id
            )
        except ModelError as error:
            _dump_debug(
                debug_dir, f"{label}_attempt{attempt}_unparsed", getattr(error, "raw_text", "")
            )
            raise
        items = [
            item
            for item in extract_list(payload, *keys, looks_like_item=looks_like_item)
            if looks_like_item(item)
        ]
        if len(items) >= wanted or attempt == 2:
            if len(items) < wanted:
                _dump_debug(
                    debug_dir,
                    f"{label}_attempt{attempt}_short",
                    {"wanted": wanted, "got": len(items), "payload": payload, "raw": response.text},
                )
                logger.warning("%s returned %d of %d items", label, len(items), wanted)
            return items
        _dump_debug(
            debug_dir,
            f"{label}_attempt1_short",
            {"wanted": wanted, "got": len(items), "payload": payload, "raw": response.text},
        )
        messages = [
            {"role": "system", "content": suite_prompts.AUTHOR_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_message
                + render(suite_prompts.SHAPE_REMINDER, expected=wanted, shape=json_shape),
            },
        ]
    return []


async def _repair_items(
    client: ModelClient,
    role: Any,
    user_message: str,
    previous: list[dict[str, Any]],
    *,
    errors: Sequence[str] = (),
    subjects: Sequence[str] = (),
    instruction: str | None = None,
    keys: tuple[str, ...],
    looks_like_item: Any,
    label: str,
    record_id: str,
    debug_dir: Path | None,
    stage: str = "suite.author.repair",
) -> list[dict[str, Any]]:
    """One repair attempt, showing the model its own output and what was wrong with it.

    `instruction` overrides the default validator wording, which is how a rubric reviewer's
    findings are fed back through the same loop: same prompt, same shape, different complaint.
    """
    if instruction is None:
        instruction = render(
            suite_prompts.REPAIR_PROMPT,
            errors="\n".join(f"- {error}" for error in errors),
            subjects=", ".join(subjects) or "(all of them)",
        )
    messages = [
        {"role": "system", "content": suite_prompts.AUTHOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
        {
            "role": "assistant",
            "content": json.dumps({keys[0]: previous}, ensure_ascii=False)[:20000],
        },
        {"role": "user", "content": instruction},
    ]
    try:
        payload, _ = await client.complete_json(
            role, messages, stage=stage, record_id=record_id
        )
    except ModelError as error:
        _dump_debug(debug_dir, f"{label}_repair_unparsed", getattr(error, "raw_text", ""))
        logger.warning("%s: repair attempt failed to parse (%s)", label, error)
        return []
    return [
        item
        for item in extract_list(payload, *keys, looks_like_item=looks_like_item)
        if looks_like_item(item)
    ]


# ---------------------------------------------------------------------- stage 1: families


def _build_family(
    item: dict[str, Any],
    slot: FamilySlot,
    spec: TargetSpec,
    *,
    situation: str | None = None,
    provenance: str = "authored",
    held_out_rationale: str | None = None,
) -> Family:
    known = set(_principle_ids(spec))
    claimed = [str(value).strip() for value in (item.get("principles_in_play") or [])]
    principles = tuple(value for value in claimed if value in known)
    unknown = [value for value in claimed if value not in known]
    if unknown:
        logger.warning(
            "slot %d (%s): dropped principle ids not in the specification: %s",
            slot.slot_index,
            slot.kind,
            unknown,
        )
    text = _clean(situation if situation is not None else item.get("situation"))
    return Family(
        family_id=records.short_id("fam", spec.target_id, slot.kind, text),
        kind=slot.kind,
        title=_clean(item.get("title"))[:80] or text[:60],
        situation=text,
        domain=_clean(item.get("domain")) if situation is not None else slot.domain,
        provenance=provenance,
        held_out_rationale=_clean(
            held_out_rationale
            if held_out_rationale is not None
            else item.get("held_out_rationale")
        ),
        # A negative control that lists principles is not a negative control; the schema
        # rejects it, so the plan's decision wins over whatever the model returned.
        principles_in_play=() if slot.kind == "negative_control" else principles,
        notes=_clean(item.get("why_it_is_hard")),
    )


def _match_by_index(
    items: Sequence[dict[str, Any]], batch: Sequence[FamilySlot]
) -> list[tuple[FamilySlot, dict[str, Any]]]:
    """Pair returned families with their assignments by `index`, then by position.

    The plan owns the kind and the domain, so a family matched to the wrong assignment gets
    the wrong domain recorded and the coverage table stops meaning anything. The echoed index
    is the primary key; position only fills the gaps a model left by omitting it.
    """
    by_index: dict[int, dict[str, Any]] = {}
    leftovers: list[dict[str, Any]] = []
    for item in items:
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            index = -1
        if 1 <= index <= len(batch) and index not in by_index:
            by_index[index] = item
        else:
            leftovers.append(item)
    free = [index for index in range(1, len(batch) + 1) if index not in by_index]
    for index, item in zip(free, leftovers):
        by_index[index] = item
    return [
        (slot, by_index[position])
        for position, slot in enumerate(batch, 1)
        if position in by_index
    ]


def _external_held_out_rationale(
    situation: ExternalSituation, score: float, measured: bool
) -> str:
    """Stated as evidence, not as a claim: the row is public and the overlap is measured."""
    base = (
        f"Situation taken verbatim from the public {situation.source} test split "
        f"(row {situation.row_index}). It was written for that dataset, not for this project, "
        "and the training corpus was authored from the specification rather than from dataset rows."
    )
    if measured:
        return base + f" Highest lexical overlap with any training prompt: {score:.2f}."
    return base + " Overlap against the training corpus was not measured for this run."


class _FamilyRegistry:
    """What has already been accepted, and what a new family must not collide with.

    Separated from the authoring loop so the collision rules can be tested directly. Three
    screens run in order, cheapest first: an exact repeat of the situation key, a reused
    setting, and a lexical near-duplicate of the situation text.
    """

    def __init__(self, plan: SuitePlan) -> None:
        self.plan = plan
        self.families: list[Family] = []
        self.keys: dict[tuple[str, ...], str] = {}
        self.settings: dict[str, str] = {}
        self.key_of: dict[str, SituationKey] = {}
        self.slot_of: dict[str, int] = {}

    @property
    def filled_slots(self) -> set[int]:
        return set(self.slot_of.values())

    def collisions(self, family: Family, key: SituationKey) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        if key.is_complete and key.normalised in self.keys:
            found.append(
                {
                    "family_id": family.family_id,
                    "kind": family.kind,
                    "reason": "situation_key already used",
                    "detail": key.render(),
                    "against": self.keys[key.normalised],
                    "score": None,
                }
            )
        setting = key.normalised_setting
        if setting and setting in self.settings:
            found.append(
                {
                    "family_id": family.family_id,
                    "kind": family.kind,
                    "reason": "setting already used",
                    "detail": key.setting,
                    "against": self.settings[setting],
                    "score": None,
                }
            )
        for existing in self.families:
            overlap = jaccard(existing.situation, family.situation)
            if overlap > self.plan.max_situation_overlap:
                found.append(
                    {
                        "family_id": family.family_id,
                        "kind": family.kind,
                        "reason": "near-duplicate situation",
                        "detail": f"{overlap:.2f} lexical overlap, limit "
                        f"{self.plan.max_situation_overlap:.2f}",
                        "against": existing.family_id,
                        "score": round(overlap, 4),
                    }
                )
                break
        return found

    def add(self, slot: FamilySlot, family: Family, key: SituationKey) -> None:
        self.families.append(family)
        self.key_of[family.family_id] = key
        self.slot_of[family.family_id] = slot.slot_index
        if key.is_complete:
            self.keys[key.normalised] = family.family_id
        if key.normalised_setting:
            self.settings[key.normalised_setting] = family.family_id

    def used_keys_block(self) -> str:
        """The compact feedback each later batch is shown: titles and keys already taken."""
        if not self.families:
            return "- (none yet: this is the first batch)"
        return "\n".join(
            f'- "{family.title}" -> {self.key_of[family.family_id].render()}'
            for family in self.families
        )

    def used_situations_block(self, width: int = 160) -> str:
        if not self.families:
            return "- (none yet)"
        return "\n".join(f"- {family.situation[:width]}" for family in self.families)


async def author_families(
    client: ModelClient,
    config: RunConfig,
    spec: TargetSpec,
    plan: SuitePlan,
    *,
    external: Sequence[ExternalSituation] = (),
    training_prompts: Sequence[str] = (),
    debug_dir: Path | None = None,
    role_name: str = "generator",
) -> tuple[list[Family], dict[str, Any]]:
    """Stage 1. Batches of same-kind families, written one batch at a time.

    Sequential on purpose, and this is the expensive decision in the module. Running the
    batches together would be four times faster and would reproduce the failure this stage
    exists to prevent: the last data run generated 240 families from this specification and
    the validator threw away 188 of 479 responses as near-duplicates in 94 clusters, because
    the specification's own combinations are far fewer than the situations asked for. A batch
    that cannot see what its siblings wrote will write them again. So each call is shown the
    titles and situation keys already accepted, and is forbidden to reuse them.

    Slots emptied by a collision or a validation failure are refilled by a further round,
    which sees the whole accumulated list, rather than being written off.
    """
    role = config.role(role_name)
    spec_text = render_for_reviewer(spec)
    menu = _principle_menu(spec)
    tradeoff_menu = _tradeoff_menu(spec)
    cue_block = _cue_block(spec, plan)
    prepared_training = prepare_training_prompts(list(training_prompts))
    slots = plan.slots()

    external_by_slot: dict[int, ExternalSituation] = {
        slot.slot_index: external[slot.external_index]
        for slot in slots
        if slot.external_index is not None and slot.external_index < len(external)
    }
    external_slots = [slot for slot in slots if slot.external_index is not None]
    authored_slots = [slot for slot in slots if slot.slot_index not in external_by_slot]

    registry = _FamilyRegistry(plan)
    report: dict[str, Any] = {
        "planned": len(slots),
        "external_planned": len(external_slots),
        "external_used": len(external_by_slot),
        "returned": 0,
        "kept": 0,
        "repaired": 0,
        "rounds": 0,
        "dropped": [],
        "collisions": [],
        "situation_keys": {},
        "warnings": [],
    }
    if len(external_by_slot) < len(external_slots):
        report["warnings"].append(
            f"{len(external_slots) - len(external_by_slot)} slot(s) planned for external "
            "situations were authored instead: the dataset supplied fewer situations than asked"
        )

    def accept(
        slot: FamilySlot, family: Family, key: SituationKey, *, strict: bool, last_round: bool = False
    ) -> bool:
        """Validate, screen for collisions, and record. Every rejection is reported."""
        problems = family.validate() + extra_family_problems(
            family, spec, plan, prepared_training=prepared_training
        )
        if strict:
            problems += key_problems(family.family_id, key, spec, strict=not last_round)
            if last_round:
                # The cosmetic key checks have had their repair and their retry by now.
                # Failing the whole run over a setting of five words, or a tradeoff id the
                # specification does not define, would buy nothing the collision screens do
                # not already buy, so they are recorded and the family is kept.
                cosmetic = [
                    problem
                    for problem in key_problems(family.family_id, key, spec)
                    if problem not in problems
                ]
                report["warnings"] += cosmetic
        if problems:
            report["dropped"].append(
                {"family_id": family.family_id, "kind": family.kind, "problems": problems}
            )
            return False
        collisions = registry.collisions(family, key)
        if collisions:
            report["collisions"] += collisions
            report["dropped"].append(
                {
                    "family_id": family.family_id,
                    "kind": family.kind,
                    "problems": [
                        f"{family.family_id}: {c['reason']} ({c['detail']}), against {c['against']}"
                        for c in collisions
                    ],
                }
            )
            return False
        if any(existing.family_id == family.family_id for existing in registry.families):
            return False
        registry.add(slot, family, key)
        report["situation_keys"][family.family_id] = key.render()
        return True

    # -- external families first, so their keys constrain every authored batch -----------
    if external_by_slot:
        ordered = [(slot, external_by_slot[slot.slot_index]) for slot in external_slots
                   if slot.slot_index in external_by_slot]
        message = render(
            suite_prompts.EXTERNAL_FAMILY_PROMPT,
            target_spec=spec_text,
            n_situations=len(ordered),
            principle_menu=menu,
            tradeoff_menu=tradeoff_menu,
            domains=", ".join(plan.domains),
            situations=render_situations([situation for _, situation in ordered]),
        )
        items = await _request_items(
            client,
            role,
            message,
            wanted=len(ordered),
            keys=("families", "family", "items"),
            looks_like_item=lambda item: isinstance(item, dict) and bool(item.get("title")),
            json_shape=suite_prompts.EXTERNAL_FAMILY_JSON_SHAPE,
            label="families_external",
            record_id="external",
            debug_dir=debug_dir,
        )
        report["returned"] += len(items)
        by_index: dict[int, dict[str, Any]] = {}
        for position, item in enumerate(items, 1):
            try:
                index = int(item.get("index", position))
            except (TypeError, ValueError):
                index = position
            by_index.setdefault(index, item)
        for position, (slot, situation) in enumerate(ordered, 1):
            item = dict(by_index.get(position, {}))
            score, _ = closest_training_prompt(situation.situation, prepared_training)
            if item.get("domain") not in plan.domains:
                item["domain"] = slot.domain
            family = _build_family(
                item,
                slot,
                spec,
                situation=situation.situation,
                provenance=situation.provenance,
                held_out_rationale=_external_held_out_rationale(
                    situation, score, bool(prepared_training)
                ),
            )
            # Not strict: the situation is a dataset row, so an incomplete key is a gap in
            # the metadata call rather than a reason to throw the row away.
            accept(slot, family, _situation_key(item, slot, spec), strict=False)

    # -- authored families, one batch at a time ------------------------------------------
    async def run_batch(kind: str, batch: list[FamilySlot], *, last_round: bool) -> None:
        message = render(
            suite_prompts.FAMILY_AUTHORING_PROMPT,
            target_spec=spec_text,
            n_families=len(batch),
            kind=kind,
            assignments="\n".join(
                f"- family {position}: domain={slot.domain!r}"
                for position, slot in enumerate(batch, 1)
            ),
            kind_instructions=suite_prompts.FAMILY_KIND_INSTRUCTIONS.get(kind, ""),
            principles_note=(
                "Leave this list EMPTY for this kind."
                if kind == "negative_control"
                else "Use the ids exactly as printed."
            ),
            situation_key_block=render(
                suite_prompts.SITUATION_KEY_BLOCK,
                used_keys=registry.used_keys_block(),
                tradeoff_menu=tradeoff_menu,
            ),
            held_out_block=suite_prompts.HELD_OUT_BLOCK,
            cue_block=cue_block,
            principle_menu=menu,
            used_situations=registry.used_situations_block(),
        )
        items = await _request_items(
            client,
            role,
            message,
            wanted=len(batch),
            keys=("families", "family", "situations", "items"),
            looks_like_item=_looks_like_family,
            json_shape=suite_prompts.FAMILY_JSON_SHAPE,
            label=f"families_{kind}_{batch[0].slot_index}",
            record_id=f"{kind}:{batch[0].slot_index}",
            debug_dir=debug_dir,
        )
        report["returned"] += len(items)
        matched = _match_by_index(items, batch)
        for slot in [s for s in batch if s not in {m for m, _ in matched}]:
            report["dropped"].append(
                {
                    "family_id": "(not built)",
                    "kind": slot.kind,
                    "problems": [
                        f"slot {slot.slot_index}: the author model returned no family for this "
                        "assignment"
                    ],
                }
            )
        pending: list[tuple[FamilySlot, dict[str, Any], list[str]]] = []
        for slot, item in matched:
            family = _build_family(item, slot, spec)
            key = _situation_key(item, slot, spec)
            problems = (
                family.validate()
                + extra_family_problems(family, spec, plan, prepared_training=prepared_training)
                + key_problems(family.family_id, key, spec, strict=not last_round)
            )
            if problems:
                pending.append((slot, item, problems))
            else:
                accept(slot, family, key, strict=True, last_round=last_round)
        if not pending:
            return
        report["repaired"] += 1
        repaired = await _repair_items(
            client,
            role,
            message,
            [item for _, item, _ in pending],
            errors=[problem for _, _, problems in pending for problem in problems],
            subjects=[
                _clean(item.get("title")) or f"family {position + 1}"
                for position, (_, item, _) in enumerate(pending)
            ],
            keys=("families", "family", "situations", "items"),
            looks_like_item=_looks_like_family,
            label=f"families_repair_{batch[0].slot_index}",
            record_id=f"repair:{batch[0].slot_index}",
            debug_dir=debug_dir,
        )
        for (slot, _item, _problems), replacement in zip(pending, repaired):
            accept(
                slot,
                _build_family(replacement, slot, spec),
                _situation_key(replacement, slot, spec),
                strict=True,
                last_round=last_round,
            )
        for slot, _item, problems in pending[len(repaired) :]:
            report["dropped"].append(
                {
                    "family_id": "(not built)",
                    "kind": slot.kind,
                    "problems": problems + ["repair attempt returned no replacement"],
                }
            )

    for round_number in range(1, max(1, plan.family_attempts) + 1):
        remaining = [
            slot for slot in authored_slots if slot.slot_index not in registry.filled_slots
        ]
        if not remaining:
            break
        report["rounds"] = round_number
        if round_number > 1:
            logger.info(
                "families: round %d refilling %d slot(s) emptied by validation or a collision",
                round_number,
                len(remaining),
            )
        grouped: dict[str, list[FamilySlot]] = {}
        for slot in remaining:
            grouped.setdefault(slot.kind, []).append(slot)
        for kind, group in grouped.items():
            for start in range(0, len(group), plan.families_per_call):
                batch = group[start : start + plan.families_per_call]
                try:
                    # Sequential: each call must see what the previous ones produced.
                    await run_batch(
                        kind, batch, last_round=round_number >= max(1, plan.family_attempts)
                    )
                except Exception as error:  # one failed batch must not lose the others
                    logger.error("family batch %s/%d failed: %s", kind, batch[0].slot_index, error)
                    report["warnings"].append(
                        f"family batch {kind}/{batch[0].slot_index} failed: {error}"
                    )

    families = list(registry.families)
    unfilled = [
        slot.slot_index for slot in authored_slots if slot.slot_index not in registry.filled_slots
    ]
    if unfilled:
        report["warnings"].append(
            f"slot indices {unfilled} are still empty after {report['rounds']} round(s); the "
            "suite is smaller than planned"
        )

    report["kept"] = len(families)
    logger.info(
        "families: %d kept of %d planned (%d dropped, %d repair calls)",
        len(families),
        len(slots),
        len(report["dropped"]),
        report["repaired"],
    )
    return families, report


# ------------------------------------------------------------------------- stage 2: cases


def _case_payload(case: Case) -> dict[str, Any]:
    """A case in the shape the author model returned it, so a repair can be asked in context."""
    payload: dict[str, Any] = {
        "variant": case.variant,
        "turns": list(case.turns),
        "rubric": {
            "dimensions": list(case.rubric.dimensions),
            "must_notice": list(case.rubric.must_notice),
            "must_not_infer": list(case.rubric.must_not_infer),
            "acceptable_outputs": list(case.rubric.acceptable_outputs),
            "unacceptable_reasoning": list(case.rubric.unacceptable_reasoning),
            "unscorable_if": list(case.rubric.unscorable_if),
            "anchors": [asdict(anchor) for anchor in case.rubric.anchors],
        },
    }
    if case.change_expectation is not None:
        payload["change_expectation"] = asdict(case.change_expectation)
    if case.notes:
        payload["notes"] = case.notes
    return payload


def _trim_restated_turns(turns: list[str], original_text: str) -> tuple[list[str], bool]:
    """Drop a leading turn that just repeats the original question.

    Author models habitually restate the situation before the follow-up. In a continuation
    case the runner already replays the exchange, so the restatement would send it twice.
    """
    if len(turns) < 2 or not original_text:
        return turns, False
    trimmed = list(turns)
    dropped = False
    while len(trimmed) > 1 and jaccard(trimmed[0], original_text) > 0.5:
        trimmed.pop(0)
        dropped = True
    return trimmed, dropped


def _build_case(
    item: dict[str, Any],
    *,
    family: Family,
    coverage: TaskCoverage,
    variant: str,
    plan: SuitePlan,
    original_text: str,
) -> tuple[Case, list[str]]:
    """Assemble one Case. The suite owns ids, wiring and provenance; the model owns content."""
    warnings: list[str] = []
    raw_turns = item.get("turns") or []
    if isinstance(raw_turns, str):
        raw_turns = [raw_turns]
    turns = [_clean(turn) for turn in raw_turns if _clean(turn)]

    context_answer_from: str | None = None
    if variant in CONTINUATION_VARIANTS:
        context_answer_from = _case_id(family.family_id, coverage.task, "original")
    elif coverage.follows:
        context_answer_from = _case_id(family.family_id, coverage.follows, "original")

    if context_answer_from is not None:
        turns, dropped = _trim_restated_turns(turns, original_text)
        if dropped:
            warnings.append(
                f"{family.family_id}/{coverage.task}/{variant}: dropped a leading turn that "
                "restated the original question"
            )

    checks = [
        check
        for check in (item.get("deterministic_checks") or [])
        if isinstance(check, dict) and check.get("kind") in plan.deterministic_kinds
    ]
    payload = {
        "case_id": _case_id(family.family_id, coverage.task, variant),
        "family_id": family.family_id,
        "task": coverage.task,
        "variant": variant,
        "turns": turns,
        "rubric": item.get("rubric") or {},
        "change_expectation": item.get("change_expectation") if variant != "original" else None,
        "deterministic_checks": checks,
        "context_answer_from": context_answer_from,
        "source": family.provenance,
        "notes": _clean(item.get("notes")),
    }
    return case_from_dict(payload), warnings


def case_group_message(
    spec: TargetSpec,
    plan: SuitePlan,
    family: Family,
    coverage: TaskCoverage,
    *,
    spec_text: str | None = None,
    cue_block: str | None = None,
    deterministic_block: str | None = None,
) -> str:
    """The authoring prompt for one (family, task) cell.

    Public because a repair has to be asked in the same terms as the original request. A
    reviewer's findings are fed back through this same message, so a fix cannot be produced
    against a different standard from the one the case was written to.
    """
    spec_text = render_for_reviewer(spec) if spec_text is None else spec_text
    cue_block = _cue_block(spec, plan) if cue_block is None else cue_block
    deterministic_block = (
        _deterministic_block(plan) if deterministic_block is None else deterministic_block
    )
    return render(
        suite_prompts.CASE_AUTHORING_PROMPT,
        target_spec=spec_text,
        family_title=family.title,
        family_kind=family.kind,
        family_domain=family.domain,
        family_situation=family.situation,
        why_it_is_hard=family.notes or "(not recorded)",
        principles_in_play=", ".join(family.principles_in_play) or "(none: this is a control)",
        n_cases=len(coverage.variants),
        task=coverage.task,
        task_instructions=suite_prompts.TASK_INSTRUCTIONS.get(coverage.task, ""),
        allowed_dimensions=_allowed_dimensions_block(coverage.task),
        family_note=(
            suite_prompts.NEGATIVE_CONTROL_RUBRIC_NOTE
            if family.kind == "negative_control"
            else ""
        ),
        continuation_note=(
            suite_prompts.MULTI_EPISODE_CASE_NOTE
            if coverage.follows
            else (
                suite_prompts.CONTINUATION_NOTE
                if any(variant in CONTINUATION_VARIANTS for variant in coverage.variants)
                else ""
            )
        ),
        variant_block=_variant_block(coverage),
        rubric_rules=suite_prompts.RUBRIC_RULES,
        cue_block=cue_block,
        deterministic_block=deterministic_block,
    )


async def _author_case_group(
    client: ModelClient,
    role: Any,
    spec: TargetSpec,
    plan: SuitePlan,
    family: Family,
    coverage: TaskCoverage,
    *,
    spec_text: str,
    cue_block: str,
    deterministic_block: str,
    debug_dir: Path | None,
) -> dict[str, Any]:
    """One model call for one (family, task) cell, then validation and one repair attempt."""
    label = f"cases_{family.family_id}_{coverage.task}"
    message = case_group_message(
        spec,
        plan,
        family,
        coverage,
        spec_text=spec_text,
        cue_block=cue_block,
        deterministic_block=deterministic_block,
    )
    outcome: dict[str, Any] = {
        "family_id": family.family_id,
        "task": coverage.task,
        "cases": [],
        "dropped": [],
        "warnings": [],
        "repaired": False,
        "returned": 0,
    }
    items = await _request_items(
        client,
        role,
        message,
        wanted=len(coverage.variants),
        keys=("cases", "case", "items"),
        looks_like_item=_looks_like_case,
        json_shape=suite_prompts.CASE_JSON_SHAPE,
        label=label,
        record_id=f"{family.family_id}:{coverage.task}",
        debug_dir=debug_dir,
    )
    outcome["returned"] = len(items)

    def index_by_variant(
        entries: Sequence[dict[str, Any]], order: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        """Label entries by variant, falling back to the order they were asked for.

        A model that returns the right cases with the labels missing has still done the work,
        and the order is the only other signal available.
        """
        indexed: dict[str, dict[str, Any]] = {}
        unlabelled: list[dict[str, Any]] = []
        for entry in entries:
            variant = _clean(entry.get("variant"))
            if variant in order and variant not in indexed:
                indexed[variant] = entry
            elif variant not in order:
                unlabelled.append(entry)
        for variant, entry in zip([v for v in order if v not in indexed], unlabelled):
            indexed[variant] = entry
        return indexed

    returned = index_by_variant(items, coverage.variants)
    original_text = _clean(" ".join((returned.get("original") or {}).get("turns") or []))

    built: dict[str, Case] = {}
    failed: dict[str, tuple[dict[str, Any], list[str]]] = {}
    for variant in coverage.variants:
        item = returned.get(variant)
        if item is None:
            outcome["dropped"].append(
                {
                    "case_id": _case_id(family.family_id, coverage.task, variant),
                    "family_id": family.family_id,
                    "task": coverage.task,
                    "variant": variant,
                    "problems": ["the author model returned no case for this variant"],
                }
            )
            continue
        try:
            case, warnings = _build_case(
                item,
                family=family,
                coverage=coverage,
                variant=variant,
                plan=plan,
                original_text=original_text,
            )
            outcome["warnings"] += warnings
            problems = case.validate() + extra_case_problems(case, family, spec, plan)
        except Exception as error:
            # A bug in building or checking ONE case must not discard the others in a group
            # that has already been paid for. A real run lost six paid case groups, and every
            # answer in them, to a NameError raised at this point.
            logger.exception("%s/%s: building the case raised", label, variant)
            outcome["dropped"].append(
                {
                    "case_id": _case_id(family.family_id, coverage.task, variant),
                    "family_id": family.family_id,
                    "task": coverage.task,
                    "variant": variant,
                    "problems": [f"building this case raised {type(error).__name__}: {error}"],
                }
            )
            _dump_debug(debug_dir, f"{label}_{variant}_raised", item)
            continue
        if problems:
            failed[variant] = (item, problems)
        else:
            built[variant] = case

    if failed:
        outcome["repaired"] = True
        repaired_items = await _repair_items(
            client,
            role,
            message,
            [item for item, _ in failed.values()],
            errors=[problem for _, problems in failed.values() for problem in problems],
            subjects=[f"variant {variant}" for variant in failed],
            keys=("cases", "case", "items"),
            looks_like_item=_looks_like_case,
            label=label,
            record_id=f"{family.family_id}:{coverage.task}:repair",
            debug_dir=debug_dir,
        )
        repaired = index_by_variant(repaired_items, list(failed))
        for variant, (item, problems) in failed.items():
            replacement = repaired.get(variant)
            if replacement is None:
                outcome["dropped"].append(
                    {
                        "case_id": _case_id(family.family_id, coverage.task, variant),
                        "family_id": family.family_id,
                        "task": coverage.task,
                        "variant": variant,
                        "problems": problems + ["repair attempt returned no replacement"],
                    }
                )
                continue
            case, warnings = _build_case(
                replacement,
                family=family,
                coverage=coverage,
                variant=variant,
                plan=plan,
                original_text=original_text,
            )
            outcome["warnings"] += warnings
            still = case.validate() + extra_case_problems(case, family, spec, plan)
            if still:
                outcome["dropped"].append(
                    {
                        "case_id": case.case_id,
                        "family_id": family.family_id,
                        "task": coverage.task,
                        "variant": variant,
                        "problems": still,
                        "first_attempt_problems": problems,
                    }
                )
            else:
                built[variant] = case

    outcome["cases"] = [built[variant] for variant in coverage.variants if variant in built]
    return outcome


async def author_cases(
    client: ModelClient,
    config: RunConfig,
    spec: TargetSpec,
    plan: SuitePlan,
    families: Sequence[Family],
    *,
    debug_dir: Path | None = None,
    role_name: str = "generator",
) -> tuple[list[Case], dict[str, Any]]:
    """Stage 2. One call per (family, task) cell of the coverage matrix, run together."""
    role = config.role(role_name)
    spec_text = render_for_reviewer(spec)
    cue_block = _cue_block(spec, plan)
    deterministic_block = _deterministic_block(plan)

    jobs = [
        (family, coverage)
        for family in families
        for coverage in plan.coverage_for(family.kind)
    ]
    results = await gather_bounded(
        [
            _author_case_group(
                client,
                role,
                spec,
                plan,
                family,
                coverage,
                spec_text=spec_text,
                cue_block=cue_block,
                deterministic_block=deterministic_block,
                debug_dir=debug_dir,
            )
            for family, coverage in jobs
        ]
    )

    cases: list[Case] = []
    report: dict[str, Any] = {
        "planned": sum(len(coverage.variants) for _, coverage in jobs),
        "groups": len(jobs),
        "returned": 0,
        "kept": 0,
        "repaired_groups": 0,
        "dropped": [],
        "warnings": [],
    }
    for (family, coverage), result in zip(jobs, results):
        if isinstance(result, Exception):
            logger.error(
                "case group %s/%s failed: %s: %s",
                family.family_id,
                coverage.task,
                type(result).__name__,
                result,
            )
            report["warnings"].append(
                f"case group {family.family_id}/{coverage.task} failed: {result}"
            )
            report["dropped"].append(
                {
                    "family_id": family.family_id,
                    "task": coverage.task,
                    "variant": "(all)",
                    "problems": [f"the authoring call raised {type(result).__name__}: {result}"],
                }
            )
            continue
        cases.extend(result["cases"])
        report["returned"] += result["returned"]
        report["dropped"] += result["dropped"]
        report["warnings"] += result["warnings"]
        report["repaired_groups"] += int(result["repaired"])

    report["kept"] = len(cases)
    logger.info(
        "cases: %d kept of %d planned across %d groups (%d dropped, %d groups repaired)",
        len(cases),
        report["planned"],
        len(jobs),
        len(report["dropped"]),
        report["repaired_groups"],
    )
    return cases, report


# ----------------------------------------------------------------------------- assembling


def prune_unsupported(cases: Sequence[Case]) -> tuple[list[Case], list[dict[str, Any]]]:
    """Drop cases whose comparison partner did not survive, until the set is stable.

    A variant with no original cannot be scored for invariance or sensitivity, and a
    continuation whose context case is gone would be answered without its context. Both are
    suite-level errors, so they are removed here rather than left for `Suite.validate` to
    reject the whole run.
    """
    kept = list(cases)
    removed: list[dict[str, Any]] = []
    while True:
        by_id = {case.case_id: case for case in kept}
        originals = {
            (case.family_id, case.task) for case in kept if case.variant == "original"
        }
        survivors: list[Case] = []
        for case in kept:
            if case.variant != "original" and (case.family_id, case.task) not in originals:
                removed.append(
                    {
                        "case_id": case.case_id,
                        "family_id": case.family_id,
                        "task": case.task,
                        "variant": case.variant,
                        "problems": ["its original case did not survive validation"],
                    }
                )
                continue
            if case.context_answer_from and case.context_answer_from not in by_id:
                removed.append(
                    {
                        "case_id": case.case_id,
                        "family_id": case.family_id,
                        "task": case.task,
                        "variant": case.variant,
                        "problems": ["the case it continues did not survive validation"],
                    }
                )
                continue
            survivors.append(case)
        if len(survivors) == len(kept):
            return survivors, removed
        kept = survivors


@dataclass(frozen=True)
class RepairRequest:
    """A case that needs rewriting, and why. Built from a reviewer's findings.

    `persona_eval.suite.review` produces `Defect(type, severity, detail, suggested_fix)` per
    case. `from_review` turns those into this, so a reviewer's audit can be fed back through
    the authoring loop instead of only being able to drop the case.
    """

    case_id: str
    problems: tuple[str, ...]
    suggested_fixes: tuple[str, ...] = ()

    @classmethod
    def from_review(cls, review: Any) -> "RepairRequest":
        """Build from anything with `case_id` and `defects` of (type, detail, suggested_fix)."""
        problems, fixes = [], []
        for defect in getattr(review, "defects", None) or []:
            kind = str(getattr(defect, "type", "") or "").strip()
            detail = str(getattr(defect, "detail", "") or "").strip()
            fix = str(getattr(defect, "suggested_fix", "") or "").strip()
            if detail or kind:
                problems.append(f"[{kind or 'unspecified'}] {detail}".strip())
            if fix:
                fixes.append(fix)
        return cls(
            case_id=str(getattr(review, "case_id", "")),
            problems=tuple(problems),
            suggested_fixes=tuple(fixes),
        )


async def repair_cases(
    config: RunConfig,
    spec: TargetSpec,
    plan: SuitePlan,
    suite: Suite,
    requests: Sequence[RepairRequest],
    *,
    usage_path: Path | None = None,
    client: ModelClient | None = None,
    role_name: str = "generator",
) -> tuple[list[Case], dict[str, Any]]:
    """Rewrite cases against findings, and return only the ones that come back valid.

    Grouped by (family, task) so one call fixes every defective variant of a cell and sees
    the others as context, exactly as the original authoring call did. A replacement that
    still fails validation is discarded: a case is never returned in a worse state than the
    one the reviewer complained about.

    Returns the repaired cases and a report. The caller decides whether to swap them in,
    because dropping a case and rewriting it are different editorial decisions.
    """
    wanted = {request.case_id: request for request in requests if request.case_id}
    groups: dict[tuple[str, str], list[Case]] = {}
    unknown: list[str] = []
    for case_id in wanted:
        try:
            case = suite.case(case_id)
        except SuiteError:
            unknown.append(case_id)
            continue
        groups.setdefault((case.family_id, case.task), []).append(case)

    report: dict[str, Any] = {
        "requested": len(wanted),
        "groups": len(groups),
        "repaired": [],
        "unchanged": [],
        "unknown_case_ids": unknown,
    }
    if not groups:
        return [], report

    role = config.role(role_name)
    spec_text = render_for_reviewer(spec)
    cue_block = _cue_block(spec, plan)
    deterministic_block = _deterministic_block(plan)
    own_client = client is None
    model_client = client or ModelClient.from_config(config, usage_path, "suite.author.repair")

    repaired: list[Case] = []
    try:
        for (family_id, task), targets in groups.items():
            family = suite.family(family_id)
            coverage = next(
                (cell for cell in plan.coverage_for(family.kind) if cell.task == task), None
            )
            if coverage is None:
                # Reconstruct the cell from the suite itself when the plan has moved on.
                variants = tuple(
                    case.variant
                    for case in suite.cases_of(family_id)
                    if case.task == task
                )
                coverage = TaskCoverage(task=task, variants=variants or ("original",))
            message = case_group_message(
                spec,
                plan,
                family,
                coverage,
                spec_text=spec_text,
                cue_block=cue_block,
                deterministic_block=deterministic_block,
            )
            group_cases = [c for c in suite.cases_of(family_id) if c.task == task]
            target_variants = {case.variant for case in targets}
            defects, fixes = [], []
            for case in targets:
                request = wanted[case.case_id]
                defects += [f"variant {case.variant}: {problem}" for problem in request.problems]
                fixes += [f"variant {case.variant}: {fix}" for fix in request.suggested_fixes]
            instruction = render(
                suite_prompts.REVIEW_REPAIR_PROMPT,
                defects="\n".join(f"- {defect}" for defect in defects) or "- (none given)",
                suggestions=(
                    "Fixes the reviewer suggested. Follow them where they are right, and say "
                    "nothing if you depart from one:\n\n"
                    + "\n".join(f"- {fix}" for fix in fixes)
                    if fixes
                    else ""
                ),
                subjects=", ".join(f"variant {variant}" for variant in sorted(target_variants)),
            )
            items = await _repair_items(
                model_client,
                role,
                message,
                [_case_payload(case) for case in group_cases],
                instruction=instruction,
                keys=("cases", "case", "items"),
                looks_like_item=_looks_like_case,
                label=f"review_repair_{family_id}_{task}",
                record_id=f"{family_id}:{task}:review_repair",
                debug_dir=(usage_path.parent / "debug") if usage_path is not None else None,
                stage="suite.author.review_repair",
            )
            by_variant: dict[str, dict[str, Any]] = {}
            for item in items:
                variant = _clean(item.get("variant"))
                if variant in target_variants and variant not in by_variant:
                    by_variant[variant] = item
            original_text = _clean(
                " ".join(
                    next(
                        (c.turns for c in group_cases if c.variant == "original"),
                        (),
                    )
                )
            )
            for case in targets:
                item = by_variant.get(case.variant)
                if item is None:
                    report["unchanged"].append(
                        {"case_id": case.case_id, "reason": "the model returned no replacement"}
                    )
                    continue
                rebuilt, _warnings = _build_case(
                    item,
                    family=family,
                    coverage=coverage,
                    variant=case.variant,
                    plan=plan,
                    original_text=original_text,
                )
                problems = rebuilt.validate() + extra_case_problems(rebuilt, family, spec, plan)
                if problems:
                    report["unchanged"].append(
                        {"case_id": case.case_id, "reason": "the rewrite did not validate",
                         "problems": problems}
                    )
                    continue
                if rebuilt.rubric.version == case.rubric.version:
                    report["unchanged"].append(
                        {"case_id": case.case_id, "reason": "the rewrite was identical"}
                    )
                    continue
                repaired.append(rebuilt)
                report["repaired"].append(
                    {
                        "case_id": case.case_id,
                        "was_rubric_version": case.rubric.version,
                        "now_rubric_version": rebuilt.rubric.version,
                    }
                )
    finally:
        if own_client:
            await model_client.aclose()

    logger.info(
        "review repair: %d of %d requested cases rewritten", len(repaired), len(wanted)
    )
    return repaired, report


def replace_cases(suite: Suite, replacements: Sequence[Case]) -> Suite:
    """Swap rewritten cases into a suite, keeping ids and order, then revalidate.

    Case ids are derived from (family, task, variant), so a rewrite keeps the id it had and
    everything already recorded against it still lines up.
    """
    by_id = {case.case_id: case for case in replacements}
    swapped = tuple(by_id.get(case.case_id, case) for case in suite.cases)
    updated = replace(suite, cases=swapped)
    problems = updated.validate()
    if problems:
        raise SuiteAuthoringError(
            "the repaired suite does not validate: " + "; ".join(problems[:10])
        )
    return updated


def build_suite(
    spec: TargetSpec,
    families: Sequence[Family],
    cases: Sequence[Case],
    authoring: dict[str, Any],
    *,
    created_utc: str | None = None,
) -> Suite:
    """Assemble and validate. Families with no surviving cases are dropped, not carried."""
    with_cases = {case.family_id for case in cases}
    kept_families = [family for family in families if family.family_id in with_cases]
    dropped = [family.family_id for family in families if family.family_id not in with_cases]
    if dropped:
        # An empty family would inflate the family counts the report divides rates by.
        authoring.setdefault("families", {}).setdefault("without_cases", []).extend(dropped)
        logger.warning("dropping %d families with no surviving cases: %s", len(dropped), dropped)
    stamp = created_utc or datetime.now(timezone.utc).isoformat(timespec="seconds")
    suite = Suite(
        suite_id=records.short_id("suite", spec.target_id, spec.version, stamp),
        target_id=spec.target_id,
        spec_version=str(spec.version),
        spec_hash=content_hash(spec.raw),
        created_utc=stamp,
        families=tuple(kept_families),
        cases=tuple(cases),
        authoring=authoring,
    )
    problems = suite.validate()
    if problems:
        raise SuiteAuthoringError(
            "the assembled suite does not validate, which is a bug in authoring rather than a "
            "model failure: " + "; ".join(problems[:10])
        )
    return suite


async def author_suite(
    config: RunConfig,
    spec: TargetSpec,
    plan: SuitePlan,
    usage_path: Path | None = None,
    *,
    external: Sequence[ExternalSituation] = (),
    training_prompts: Sequence[str] = (),
    client: ModelClient | None = None,
    situations_role: str = "generator",
    rubric_role: str = "generator",
) -> tuple[Suite, dict[str, Any]]:
    """Author a whole suite and report what it cost, what was dropped, and why.

    `external` seeds part of the suite from a public dataset (see suite.external);
    `training_prompts` turns the held-out claim into a measurement by rejecting any family
    that overlaps the training corpus. `client` is for tests and for reusing an open client.

    The two stages take separate model roles because they want opposite sampling. Inventing
    situations wants a warm temperature, since near-duplication and not imprecision is this
    specification's known failure mode; writing anchors and JSON wants a cold one. Both
    default to "generator" so a config with a single role keeps working unchanged.
    """
    problems = plan.validate()
    if problems:
        raise SuiteAuthoringError("the coverage plan is not usable: " + "; ".join(problems))
    unreachable = plan.unreachable_dimensions()
    if unreachable:
        logger.warning(
            "the coverage matrix cannot score %s on any planned task; these will be absent "
            "from the report by construction, not by chance",
            ", ".join(unreachable),
        )

    situations = config.role(situations_role)
    rubrics = config.role(rubric_role)
    debug_dir = (usage_path.parent / "debug") if usage_path is not None else None
    started = datetime.now(timezone.utc)

    own_client = client is None
    model_client = client or ModelClient.from_config(config, usage_path, "suite.author")
    try:
        families, family_report = await author_families(
            model_client,
            config,
            spec,
            plan,
            external=external,
            training_prompts=training_prompts,
            debug_dir=debug_dir,
            role_name=situations_role,
        )
        if not families:
            raise SuiteAuthoringError(
                "stage 1 produced no usable families; see the dropped list in the authoring "
                "report and the debug payloads"
            )
        cases, case_report = await author_cases(
            model_client,
            config,
            spec,
            plan,
            families,
            debug_dir=debug_dir,
            role_name=rubric_role,
        )
    finally:
        if own_client:
            await model_client.aclose()

    cases, pruned = prune_unsupported(cases)
    case_report["dropped"] += pruned
    case_report["kept"] = len(cases)

    report: dict[str, Any] = {
        "target_id": spec.target_id,
        "spec_version": str(spec.version),
        "spec_hash": content_hash(spec.raw),
        "started_utc": started.isoformat(timespec="seconds"),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {
            "situations": {
                "role": situations_role,
                "model": situations.model,
                "temperature": situations.temperature,
            },
            "rubrics": {
                "role": rubric_role,
                "model": rubrics.model,
                "temperature": rubrics.temperature,
            },
        },
        "plan": plan.to_dict(),
        "families": family_report,
        "cases": case_report,
        "external": {
            "supplied": len(external),
            "sources": sorted({situation.source for situation in external}),
        },
        "training_prompts_checked": len(training_prompts),
        "usage": dict(getattr(model_client.totals, "by_model", {}) or {}),
    }
    suite = build_suite(spec, families, cases, report)
    report["suite_id"] = suite.suite_id
    report["suite_version"] = suite.version
    report["kept"] = {"families": len(suite.families), "cases": len(suite.cases)}
    logger.info(
        "suite %s: %d families, %d cases (%d cases dropped, %d families dropped)",
        suite.suite_id,
        len(suite.families),
        len(suite.cases),
        len(case_report["dropped"]),
        len(family_report["dropped"]),
    )
    return suite, report


__all__ = [
    "DEFAULT_COVERAGE",
    "DEFAULT_FAMILIES_PER_KIND",
    "GENERIC_RUBRIC_WORDS",
    "GENERIC_SETTINGS",
    "MIN_SITUATION_TOKENS",
    "FamilySlot",
    "RepairRequest",
    "SituationKey",
    "SuiteAuthoringError",
    "SuitePlan",
    "TaskCoverage",
    "author_cases",
    "author_families",
    "author_suite",
    "build_suite",
    "case_group_message",
    "extra_case_problems",
    "extra_family_problems",
    "key_problems",
    "prune_unsupported",
    "repair_cases",
    "replace_cases",
    "rubric_text",
    "undiscriminating_anchors",
    "ungrounded_must_notice",
]
