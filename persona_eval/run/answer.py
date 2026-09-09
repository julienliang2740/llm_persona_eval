"""Get answers out of a model under test, under conditions the report can verify.

The comparison this suite exists to make is between arms - a baseline checkpoint and a
tuned one - so the interesting engineering here is not "call a model" but "call two models
in a way that cannot quietly differ". Three rules from the plan drive the design:

  Task isolation. "Fresh contexts for independent noticing and decision probes; explicit
  continuation only for pressure, correction, or episode tests." So a case gets a brand
  new conversation unless the suite says otherwise, and this module records which of the
  two it was for every answer rather than leaving it to be inferred from the variant name.

  No cue. The evaluation is uncued: no system prompt is sent unless one is explicitly
  configured, because "you are a Confucian assistant" would measure instruction-following
  rather than instantiation. `answer_meta.system_prompt` is always recorded, empty or not,
  so a report can prove the arms were asked the same way.

  Matched settings. Temperature, token budget, top_p and seed are resolved ONCE per call
  and stamped on every answer along with a fingerprint hash. Two arms whose fingerprints
  differ are not comparable, and the report can say so instead of a reader having to trust
  that nobody edited the config between runs.

Continuation is the one genuinely awkward part. `Case.context_answer_from` names another
case whose MODEL ANSWER has to be replayed as the assistant turn before this case's final
question - so the pressure case cannot be answered until the original has been. This
module therefore resolves the dependency graph itself: it pulls in referenced cases that
were not requested, sorts cases into dependency levels, and answers level by level, with
everything inside a level running concurrently. Answers produced by an earlier call can be
handed back in via `prior_answers` so a resumed run does not pay for them twice.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeline.config import ModelRole, RunConfig
from pipeline.model import LocalEndpointUnavailable, ModelClient, gather_bounded

from persona_eval.suite.schema import Case, Suite, content_hash

logger = logging.getLogger("persona_eval.run.answer")

# Answering a case is one exchange, so a run is identified by stage + case id in the usage
# ledger and every cost line can be traced back to an arm.
STAGE = "run.answer"

MAX_CONTEXT_DEPTH = 8  # a continuation chain deeper than this is a suite bug, not a design


@dataclass(frozen=True)
class GenerationSettings:
    """The sampling settings every arm must share.

    `max_tokens_source` is recorded because the pipeline config allows
    `evaluation.answer_max_tokens: null`, which means "let each endpoint use its own
    budget". That is convenient and it is also the single easiest way to make two arms
    incomparable: a longer budget on one side buys it answers that reach a recommendation
    while the other is cut off. When the budget comes from the role rather than from the
    shared config this says so, and `matched_across_arms` is False.
    """

    temperature: float
    max_tokens: int
    top_p: float | None = None
    seed: int | None = None
    system_prompt: str = ""
    max_tokens_source: str = "config"  # config | role

    @property
    def matched_across_arms(self) -> bool:
        return self.max_tokens_source == "config"

    @property
    def fingerprint(self) -> str:
        """Hash of the settings that affect the answer. Equal fingerprints, comparable arms."""
        return content_hash(
            {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": self.top_p,
                "seed": self.seed,
                "system_prompt": self.system_prompt,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint
        payload["matched_across_arms"] = self.matched_across_arms
        return payload


def resolve_settings(config: RunConfig, role: ModelRole) -> GenerationSettings:
    """Read the shared generation settings out of the pipeline config.

    Everything except the token budget comes from `evaluation:` in the config, which is one
    block shared by every arm; that is what makes the settings matched by construction
    rather than by discipline.
    """
    evaluation = config.evaluation or {}
    configured_max = evaluation.get("answer_max_tokens")
    if configured_max:
        max_tokens, source = int(configured_max), "config"
    else:
        max_tokens, source = int(role.max_tokens), "role"
    top_p = evaluation.get("top_p")
    seed = evaluation.get("seed")
    return GenerationSettings(
        temperature=float(evaluation.get("temperature", role.temperature)),
        max_tokens=max_tokens,
        top_p=float(top_p) if top_p is not None else None,
        seed=int(seed) if seed is not None else None,
        # Deliberately "" by default. The plan's evaluation is uncued; a system prompt is
        # opt-in and is recorded verbatim when one is used.
        system_prompt=str(evaluation.get("answer_system_prompt") or ""),
        max_tokens_source=source,
    )


def _role_with_seed(role: ModelRole, seed: int | None) -> ModelRole:
    """Attach a sampling seed through the role's extra_body.

    The shared client builds its request body from the role, so this is the only way to
    pass a provider-specific parameter without editing the pipeline. Endpoints that ignore
    `seed` simply ignore it; the value is recorded either way, and a report that finds
    identical settings but differing answers has the seed line to look at.
    """
    if seed is None:
        return role
    return replace(role, extra_body={**dict(role.extra_body or {}), "seed": seed})


# ------------------------------------------------------------------- message assembly


def build_messages(
    suite: Suite,
    case: Case,
    answer_text_by_case: dict[str, str],
    settings: GenerationSettings,
    _depth: int = 0,
) -> list[dict[str, str]]:
    """The exact turns sent for one case.

    Fresh context is the default: one user message holding the case's only turn. Two
    shapes of continuation are supported, because the schema allows both:

      * the case carries its own earlier turns (`turns = (earlier..., final)`) and
        `context_answer_from` supplies only the assistant reply that goes before the final
        turn. This is the normal pressure/correction shape;
      * the case carries only its final turn, in which case the referenced case's whole
        message list is rebuilt in front of it. This is the shape a long multi-episode
        chain takes, where repeating every earlier turn on every episode would be
        error-prone.

    The case's own turns win where both are available: the suite is the record of what was
    asked, and reconstructing it from a neighbour would let a suite edit silently change a
    conversation it did not touch.
    """
    if _depth > MAX_CONTEXT_DEPTH:
        raise ValueError(f"{case.case_id}: context chain deeper than {MAX_CONTEXT_DEPTH}")

    messages: list[dict[str, str]] = []
    if _depth == 0 and settings.system_prompt:
        messages.append({"role": "system", "content": settings.system_prompt})

    turns = list(case.turns)
    if case.context_answer_from:
        prior_text = answer_text_by_case.get(case.context_answer_from)
        if not prior_text:
            raise ValueError(
                f"{case.case_id}: needs the answer to {case.context_answer_from!r} and it is "
                "missing or empty"
            )
        if len(turns) >= 2:
            head = [{"role": "user", "content": t} for t in turns[:-1]]
        else:
            prior_case = suite.case(case.context_answer_from)
            head = build_messages(
                suite, prior_case, answer_text_by_case, settings, _depth=_depth + 1
            )
        messages += head
        messages.append({"role": "assistant", "content": prior_text})
        messages.append({"role": "user", "content": turns[-1]})
        return messages

    messages += [{"role": "user", "content": t} for t in turns]
    return messages


def isolation_of(suite: Suite, case: Case) -> str:
    """"fresh" or "continuation", checked against the plan's rule rather than assumed.

    A case that carries a conversation without being a pressure/correction variant or a
    multi-episode family breaks task isolation: its noticing probe is contaminated by an
    earlier exchange. That is a suite defect, so it is logged loudly and recorded on the
    answer, but the answer is still produced - the run reports the problem instead of
    dying halfway through a paid sweep.
    """
    if not case.is_continuation and len(case.turns) <= 1:
        return "fresh"
    try:
        kind = suite.family(case.family_id).kind
    except Exception:
        kind = "unknown"
    if case.is_continuation or kind == "multi_episode":
        return "continuation"
    logger.error(
        "%s: task isolation violated - variant %r in a %r family carries %d turns",
        case.case_id,
        case.variant,
        kind,
        len(case.turns),
    )
    return "continuation_unexpected"


# ----------------------------------------------------------------- dependency ordering


def _expand_with_dependencies(suite: Suite, cases: Sequence[Case]) -> tuple[list[Case], set[str]]:
    """Add the cases a requested case needs answered first. Returns (all, requested ids)."""
    requested = {c.case_id for c in cases}
    by_id = {c.case_id: c for c in cases}
    queue = list(cases)
    while queue:
        case = queue.pop()
        parent_id = case.context_answer_from
        if not parent_id or parent_id in by_id:
            continue
        try:
            parent = suite.case(parent_id)
        except Exception:
            logger.error("%s: context_answer_from names unknown case %r", case.case_id, parent_id)
            continue
        by_id[parent_id] = parent
        queue.append(parent)
    return list(by_id.values()), requested


def dependency_levels(cases: Sequence[Case], already_answered: Iterable[str] = ()) -> list[list[Case]]:
    """Group cases so everything in level N only depends on levels below it.

    Cases in one level are independent and run concurrently; a level waits for the one
    before it. Anything left over is in a cycle and is returned as a final level so the
    caller can mark it failed rather than deadlock.
    """
    done = set(already_answered)
    pending = {c.case_id: c for c in cases}
    levels: list[list[Case]] = []
    while pending:
        ready = [
            c
            for c in pending.values()
            if not c.context_answer_from
            or c.context_answer_from in done
            or c.context_answer_from not in pending
        ]
        if not ready:
            logger.error(
                "context_answer_from cycle among: %s", ", ".join(sorted(pending))
            )
            levels.append(sorted(pending.values(), key=lambda c: c.case_id))
            break
        ready.sort(key=lambda c: c.case_id)
        levels.append(ready)
        for case in ready:
            done.add(case.case_id)
            pending.pop(case.case_id)
    return levels


# ------------------------------------------------------------------------ the main call


def _answer_record(
    case: Case,
    arm: str,
    model_id: str,
    text: str,
    settings: GenerationSettings,
    endpoint_role: str,
    isolation: str,
    requested: bool,
    messages: list[dict[str, str]] | None = None,
    response: Any = None,
    error: str = "",
) -> dict[str, Any]:
    usage = getattr(response, "usage", None) or {}
    finish_reason = str(getattr(response, "finish_reason", "") or "")
    meta: dict[str, Any] = {
        "endpoint_role": endpoint_role,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens") or 0),
        "finish_reason": finish_reason,
        # The plan keeps technical failures apart from philosophical ones, and a cut-off
        # answer is the commonest technical failure there is. The judge is told about it
        # rather than being left to grade half a recommendation.
        "hit_token_limit": finish_reason == "length",
        "latency_s": round(float(getattr(response, "latency_s", 0.0) or 0.0), 3),
        "request_id": str(getattr(response, "request_id", "") or ""),
        "settings": settings.to_dict(),
        "system_prompt": settings.system_prompt,
        "isolation": isolation,
        "context_answer_from": case.context_answer_from or "",
        "turns_sent": messages or [],
        "n_turns_sent": len(messages or []),
        "requested": requested,
        "error": error,
    }
    return {
        "case_id": case.case_id,
        "family_id": case.family_id,
        "task": case.task,
        "variant": case.variant,
        "arm": arm,
        "model_id": model_id,
        "text": text,
        "meta": meta,
    }


async def answer_cases(
    config: RunConfig,
    suite: Suite,
    cases: Sequence[Case],
    endpoint_role: str,
    arm: str,
    usage_path: Path | None = None,
    limit: int | None = None,
    *,
    settings: GenerationSettings | None = None,
    prior_answers: Sequence[dict[str, Any]] | None = None,
    client: ModelClient | None = None,
) -> list[dict[str, Any]]:
    """Answer `cases` with one model role, resolving continuation dependencies first.

    `prior_answers` are answers already produced for THIS arm (a resumed run, or a
    previous call whose originals the pressure cases now need). They are used to satisfy
    dependencies and are not re-answered or returned.

    `limit` caps the requested cases; dependencies of the surviving cases are still pulled
    in, because a pressure case answered without its original is not a pressure test.

    Returns one record per case answered, including dependency-only cases, which carry
    `meta.requested = False` so a caller can drop them from the headline counts.
    """
    role = config.role(endpoint_role)
    settings = settings or resolve_settings(config, role)
    if not settings.matched_across_arms:
        logger.warning(
            "arm %r: answer_max_tokens is unset, so the %d-token budget comes from role %r. "
            "Arms with different role budgets are not comparable; set evaluation."
            "answer_max_tokens to match them.",
            arm,
            settings.max_tokens,
            endpoint_role,
        )
    call_role = _role_with_seed(role, settings.seed)

    selected = list(cases)[: limit if limit is not None else None]
    all_cases, requested_ids = _expand_with_dependencies(suite, selected)

    answer_text_by_case: dict[str, str] = {
        row["case_id"]: row.get("text", "")
        for row in (prior_answers or [])
        if row.get("text")
    }
    known = {row["case_id"] for row in (prior_answers or [])}
    todo = [c for c in all_cases if c.case_id not in known]
    levels = dependency_levels(todo, already_answered=known)

    records: list[dict[str, Any]] = []
    owns_client = client is None
    client = client or ModelClient.from_config(config, usage_path, STAGE)
    try:
        for depth, level in enumerate(levels):

            async def answer_one(case: Case) -> dict[str, Any]:
                isolation = isolation_of(suite, case)
                requested = case.case_id in requested_ids
                try:
                    messages = build_messages(suite, case, answer_text_by_case, settings)
                except ValueError as error:
                    # An upstream answer is missing, usually because the original case
                    # itself failed. Never fabricate the missing turn: record the reason
                    # and let the report show the continuation as unrun.
                    logger.error("%s: %s", case.case_id, error)
                    return _answer_record(
                        case, arm, call_role.model, "", settings, endpoint_role,
                        isolation, requested, error=str(error),
                    )
                started = time.monotonic()
                try:
                    response = await client.complete(
                        call_role,
                        messages,
                        temperature=settings.temperature,
                        max_tokens=settings.max_tokens,
                        stage=f"{STAGE}.{arm}",
                        record_id=case.case_id,
                    )
                except LocalEndpointUnavailable:
                    raise
                except Exception as error:  # ModelError and anything else the client raises
                    logger.error("%s: answering failed: %s", case.case_id, error)
                    return _answer_record(
                        case, arm, call_role.model, "", settings, endpoint_role,
                        isolation, requested, messages=messages,
                        error=f"{type(error).__name__}: {error}",
                    )
                record = _answer_record(
                    case, arm, response.model or call_role.model, response.text, settings,
                    endpoint_role, isolation, requested, messages=messages, response=response,
                )
                # latency_s is the model's own request time; wall_s adds the wait for a
                # concurrency slot. A run whose wall time dwarfs its latency is queueing,
                # not a slow model, and only the pair distinguishes the two.
                record["meta"]["wall_s"] = round(time.monotonic() - started, 3)
                return record

            results = await gather_bounded([answer_one(case) for case in level])
            for result in results:
                if isinstance(result, LocalEndpointUnavailable):
                    # The local base endpoint being down is an operator problem, not a
                    # per-case failure: stopping is more useful than 200 empty answers.
                    raise RuntimeError(str(result)) from None
                if isinstance(result, BaseException):
                    logger.error("answering raised at depth %d: %s", depth, result)
                    continue
                records.append(result)
                if result.get("text"):
                    answer_text_by_case[result["case_id"]] = result["text"]
    finally:
        if owns_client:
            await client.aclose()

    logger.info(
        "arm %s: %d answers (%d requested, %d pulled in as context) over %d dependency levels",
        arm,
        len(records),
        sum(1 for r in records if r["meta"]["requested"]),
        sum(1 for r in records if not r["meta"]["requested"]),
        len(levels),
    )
    return records


def settings_disagreement(answers: Sequence[dict[str, Any]]) -> list[str]:
    """Every distinct settings fingerprint present. More than one means incomparable arms.

    Exposed so the report can make the check rather than trusting that this module made
    it, which is the point of stamping the fingerprint on every answer.
    """
    seen: dict[str, str] = {}
    for row in answers:
        settings = (row.get("meta") or {}).get("settings") or {}
        fingerprint = str(settings.get("fingerprint", ""))
        if fingerprint:
            seen.setdefault(fingerprint, row.get("arm", ""))
    return sorted(seen)


__all__ = [
    "GenerationSettings",
    "answer_cases",
    "build_messages",
    "dependency_levels",
    "isolation_of",
    "resolve_settings",
    "settings_disagreement",
]
