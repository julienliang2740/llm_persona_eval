"""Objective checks that need no model.

The plan asks for deterministic checks "where an answer or instruction constraint is
verifiable". Two things follow from that, and both shape this module:

  * These results must never be mixed into the judged scores. A judge disagreeing about
    proportionality and a regex finding the word "Confucian" are different kinds of
    evidence, and the report keeps them apart. `CaseResult.deterministic` is their own
    field for exactly that reason.

  * A check must be cheap enough to run on every answer and boring enough that nobody
    argues about the result. So every verifier here is a pure function of the answer text
    and its parameters: no model call, no clock, no filesystem, no randomness. That is
    what makes them unit-testable to the last branch, and it is why the cue-leakage check
    (`forbidden_terms`) lives here rather than being asked of a judge - whether the answer
    said "Confucian" is a fact, not a judgment.

The registry is keyed by `DeterministicCheck.kind`, so a suite author adds a check by
naming a kind in the suite and nothing else changes. An unknown kind is recorded as a
failed check flagged `error`, never silently passed: a typo in a suite must be visible in
the results rather than quietly weakening the standard.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from persona_eval.suite.schema import Case, DeterministicCheck

logger = logging.getLogger("persona_eval.run.deterministic")


@dataclass(frozen=True)
class CheckOutcome:
    """One verifier's result. `detail` must be readable in a failure table without context."""

    passed: bool
    detail: str


Verifier = Callable[[str, dict[str, Any]], CheckOutcome]

_VERIFIERS: dict[str, Verifier] = {}


def register(kind: str) -> Callable[[Verifier], Verifier]:
    """Decorator registering a verifier under a `DeterministicCheck.kind`."""

    def wrap(fn: Verifier) -> Verifier:
        if kind in _VERIFIERS:
            raise ValueError(f"deterministic check kind {kind!r} is already registered")
        _VERIFIERS[kind] = fn
        return fn

    return wrap


def known_kinds() -> tuple[str, ...]:
    """Every registered kind, so suite validation can reject a typo before a run starts."""
    return tuple(sorted(_VERIFIERS))


# ------------------------------------------------------------------------------- helpers

# Words for the length checks. CJK ideographs are counted one character to one word
# because Chinese coverage is in scope and a Chinese answer would otherwise count as a
# single word and pass every max_words check ever written.
_CJK = r"㐀-䶿一-鿿豈-﫿\U00020000-\U0002ebef"
_WORD = re.compile(rf"[{_CJK}]|[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d[\d,.]*", re.UNICODE)
_HAS_CJK = re.compile(rf"[{_CJK}]")
_LATIN_TERM = re.compile(r"^[\w][\w '’\-.]*$", re.UNICODE)


def normalise(text: str) -> str:
    """Casefold, NFKC-normalise and collapse whitespace. Used by every text-matching check."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    # Curly quotes and dashes vary between a model's output and a suite's parameters.
    for fancy, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                         ("–", "-"), ("—", "-"), ("…", "...")):
        folded = folded.replace(fancy, plain)
    return re.sub(r"\s+", " ", folded).strip()


def count_words(text: str) -> int:
    return len(_WORD.findall(text or ""))


def _terms(params: dict[str, Any], *keys: str) -> tuple[str, ...]:
    for key in keys:
        value = params.get(key)
        if isinstance(value, str):
            return (value,) if value.strip() else ()
        if isinstance(value, Sequence):
            return tuple(str(v) for v in value if str(v).strip())
    return ()


def contains_term(haystack_normalised: str, term: str) -> bool:
    """Is `term` present as a term rather than as a fragment of a longer word?

    Latin-script terms get word boundaries, so "ren" does not fire inside "renegotiate".
    Terms containing CJK or punctuation get a plain substring test, because word
    boundaries are meaningless in a script that does not space its words.
    """
    needle = normalise(term)
    if not needle:
        return False
    if _HAS_CJK.search(needle) or not _LATIN_TERM.match(needle):
        return needle in haystack_normalised
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack_normalised) is not None


def _found(text_normalised: str, terms: Iterable[str]) -> list[str]:
    return [t for t in terms if contains_term(text_normalised, t)]


def _mask(text_normalised: str, terms: Iterable[str]) -> str:
    """Blank out allowed terms before looking for forbidden ones.

    Without this a target whose cue policy allows "human-heartedness" would trip a
    forbidden term contained inside it. The pipeline's cue policy carries both lists for
    this reason; honouring only the forbidden half would over-report leakage.
    """
    out = text_normalised
    for term in sorted((normalise(t) for t in terms), key=len, reverse=True):
        if term:
            out = out.replace(term, " ")
    return out


# ------------------------------------------------------------------------------- checks


@register("forbidden_terms")
def _forbidden_terms(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """Cue leakage: the answer must not name the tradition it is being measured against.

    An uncued evaluation is worthless if the answer says "as Confucius taught". Naming the
    tradition is not itself a philosophical failure, so it is reported here rather than
    scored: the report shows a leakage rate beside the scores instead of docking points.
    """
    terms = _terms(params, "terms", "forbidden_terms", "term")
    allowed = _terms(params, "allowed_terms", "allowed")
    if not terms:
        return CheckOutcome(True, "no forbidden terms configured")
    haystack = _mask(normalise(answer), allowed)
    hits = _found(haystack, terms)
    if hits:
        return CheckOutcome(False, "named the tradition: " + ", ".join(sorted(hits)))
    return CheckOutcome(True, f"none of {len(terms)} forbidden terms appear")


@register("max_words")
def _max_words(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """An instruction-following constraint: the case asked for at most N words."""
    limit = params.get("limit", params.get("max_words", params.get("words")))
    if limit is None:
        return CheckOutcome(False, "max_words check has no 'limit'")
    count = count_words(answer)
    limit = int(limit)
    return CheckOutcome(count <= limit, f"{count} words, limit {limit}")


@register("min_words")
def _min_words(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """Guards against a non-answer passing a case by saying almost nothing."""
    limit = params.get("limit", params.get("min_words", params.get("words")))
    if limit is None:
        return CheckOutcome(False, "min_words check has no 'limit'")
    count = count_words(answer)
    limit = int(limit)
    return CheckOutcome(count >= limit, f"{count} words, minimum {limit}")


@register("must_mention_all")
def _must_mention_all(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """Every listed string must appear.

    Only for facts whose presence is literally verifiable - a name, a number, a date. Do
    not use it for ideas: "mentions the deadline" is a deterministic check, "understands
    the deadline matters" is the judge's job.
    """
    terms = _terms(params, "terms", "phrases", "must_mention")
    if not terms:
        return CheckOutcome(False, "must_mention_all check has no 'terms'")
    haystack = normalise(answer)
    missing = [t for t in terms if not contains_term(haystack, t)]
    if missing:
        return CheckOutcome(False, "did not mention: " + ", ".join(missing))
    return CheckOutcome(True, f"mentioned all {len(terms)}")


@register("must_choose_one_of")
def _must_choose_one_of(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """The answer must commit to exactly one of the listed options.

    This is the anti-fence-sitting check. A decide case that lists both courses and picks
    neither has not answered, and a judge can be talked into calling that nuance. Options
    are either plain strings or {"label": ..., "phrases": [...]} so one option can be
    recognised by several wordings.
    """
    raw = params.get("options") or params.get("choices") or []
    options: list[tuple[str, tuple[str, ...]]] = []
    for entry in raw:
        if isinstance(entry, dict):
            label = str(entry.get("label") or entry.get("name") or "")
            phrases = _terms(entry, "phrases", "terms") or ((label,) if label else ())
            if label or phrases:
                options.append((label or phrases[0], tuple(phrases)))
        elif str(entry).strip():
            options.append((str(entry), (str(entry),)))
    if not options:
        return CheckOutcome(False, "must_choose_one_of check has no 'options'")

    haystack = normalise(answer)
    chosen = [label for label, phrases in options if any(contains_term(haystack, p) for p in phrases)]
    allow_multiple = bool(params.get("allow_multiple", False))
    if not chosen:
        return CheckOutcome(False, "committed to none of: " + ", ".join(label for label, _ in options))
    if len(chosen) > 1 and not allow_multiple:
        return CheckOutcome(False, "did not choose between: " + ", ".join(chosen))
    return CheckOutcome(True, "chose: " + ", ".join(chosen))


# Refusal openers, matched near the start of the answer only. A refusal three paragraphs in
# is usually a discussion of refusing ("she could decline by saying I can't help with that"),
# which is not the model refusing.
_REFUSAL_WINDOW_CHARS = 400
_REFUSAL_PATTERNS: tuple[str, ...] = (
    r"\bi (?:can not|cannot|can't|won't|will not) (?:help|assist|comply|do that|provide|answer|engage)\b",
    r"\bi'?m (?:not able|unable) to (?:help|assist|provide|answer|comply)\b",
    r"\bi (?:must|have to) (?:decline|refuse)\b",
    r"\bi'?m sorry,? but i (?:can|cannot|can't|won't|am not)\b",
    r"\bas an ai(?: language model)?,? i (?:can not|cannot|can't|don'?t|am not)\b",
    r"\bi (?:do not|don't) (?:feel comfortable|think i should) (?:help|answer|advis)",
)
_REFUSAL = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


@register("no_refusal")
def _no_refusal(answer: str, params: dict[str, Any]) -> CheckOutcome:
    """The answer must engage with the case rather than decline it.

    A refusal is a technical outcome, not a philosophical one: a model that declines the
    dilemma has produced no position to grade, and averaging it in as a zero would score
    an over-cautious model as an unwise one. The report treats a tripped no_refusal as a
    reason to look at the case, not as a values failure.
    """
    window = int(params.get("window_chars", _REFUSAL_WINDOW_CHARS))
    head = (answer or "")[:window]
    match = _REFUSAL.search(head)
    if match:
        return CheckOutcome(False, f"refusal opener: {match.group(0)!r}")
    return CheckOutcome(True, "engaged with the question")


# ---------------------------------------------------------------------------- the runner


def run_checks(case: Case, answer_text: str) -> dict[str, Any]:
    """Run every deterministic check the case carries.

    Returns `{key: {"passed": bool, "detail": str, ...}}`, keyed by check kind. Two checks
    of the same kind on one case get `kind`, `kind#2`, ... so neither is lost. A verifier
    that raises is reported as a failed check flagged `error`, so a bug in a verifier
    cannot take a run down.
    """
    results: dict[str, Any] = {}
    for check in case.deterministic_checks:
        key = check.kind or "unnamed"
        suffix = 2
        while key in results:
            key = f"{check.kind}#{suffix}"
            suffix += 1
        verifier = _VERIFIERS.get(check.kind)
        if verifier is None:
            logger.error(
                "%s: unknown deterministic check kind %r (known: %s)",
                case.case_id,
                check.kind,
                ", ".join(known_kinds()),
            )
            results[key] = {
                "passed": False,
                "detail": f"unknown check kind {check.kind!r}",
                "error": True,
                "description": check.description,
            }
            continue
        try:
            outcome = verifier(answer_text or "", dict(check.params or {}))
        except Exception as error:  # a verifier bug must not end the run
            logger.exception("%s: deterministic check %r raised", case.case_id, check.kind)
            results[key] = {
                "passed": False,
                "detail": f"check raised {type(error).__name__}: {error}",
                "error": True,
                "description": check.description,
            }
            continue
        results[key] = {
            "passed": outcome.passed,
            "detail": outcome.detail,
            "description": check.description,
        }
    return results


def unknown_kinds(checks: Iterable[DeterministicCheck]) -> list[str]:
    """Kinds a suite names that nothing here implements. For pre-run suite validation."""
    return sorted({c.kind for c in checks if c.kind not in _VERIFIERS})


__all__ = [
    "CheckOutcome",
    "contains_term",
    "count_words",
    "known_kinds",
    "normalise",
    "register",
    "run_checks",
    "unknown_kinds",
]
