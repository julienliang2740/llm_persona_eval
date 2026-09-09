"""Running the general-capability checks and verifying the answers.

Why this is a separate component. The plan's fourth deliverable is "check general
ability", reported apart from anything philosophical: "It should reveal degradation
without being folded into a single values score." Nothing in this module produces a
values number, and nothing here consults a judge. Every item is graded by a pure
function over the answer text, so a capability regression is a fact about the output
rather than an opinion about it.

Two design commitments make the before/after comparison honest:

  Same sampling as the values evaluation. Temperature and the answer token budget are
  read from the same config keys `persona_eval.evaluate` uses, and both are recorded on
  every row. A capability drop measured at a different temperature would be an artefact.

  Fresh context, no system prompt. Each item is one user message in its own conversation.
  The values evaluation sends a neutral system prompt; here even that is omitted, so the
  only thing standing between the model and the instruction is the model.

Verification of the coding family executes model-written code. It runs in a subprocess
with a hard wall-clock timeout, CPU/memory/file-size limits, a fresh working directory, a
stripped environment and sockets disabled. That is a guard against a runaway or careless
program, not a security boundary against a hostile one; see SANDBOX_NOTE.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from pipeline.config import RunConfig
from pipeline.model import LocalEndpointUnavailable, ModelClient, ModelError, gather_bounded

from persona_eval.capability.data import CAPABILITY_ITEMS, CapabilityItem, select

logger = logging.getLogger("persona_eval.capability.checks")

SANDBOX_NOTE = (
    "Model-written code runs under: a fresh temporary working directory, an environment "
    "reduced to PATH/HOME/LC_ALL, python -I -S -B (no user site, no inherited PYTHON* "
    "variables, no script directory on sys.path), RLIMIT_CPU, RLIMIT_AS and RLIMIT_FSIZE, "
    "its own session so the whole process group can be killed, and a socket module whose "
    "constructors raise. This stops accidents. It does not contain a determined attacker, "
    "and the checks are authored here rather than downloaded precisely so the prompts that "
    "produce the code are known."
)

CODE_WALL_TIMEOUT_S = 10.0
CODE_CPU_SECONDS = 5
CODE_MEMORY_BYTES = 512 * 1024 * 1024
CODE_MAX_FILE_BYTES = 1024 * 1024
CODE_RESULT_MARKER = "<<<capability-result>>>"

try:  # POSIX only; the coding family falls back to static checks without it.
    import resource as _resource
except ImportError:  # pragma: no cover - not reachable on the target platform
    _resource = None

SANDBOX_AVAILABLE = _resource is not None and os.name == "posix"


class CapabilityError(ValueError):
    """An unknown verifier kind or malformed params. An authoring bug, not a model failure."""


# ------------------------------------------------------------------------- text utilities

_PUNCTUATION = set(string.punctuation) | set("\u2014\u2013\u2018\u2019\u201c\u201d\u2026")
_WORD_STRIP = "\"'`\u2018\u2019\u201c\u201d"
_TRAILING_PUNCTUATION = ".,!?;:\u2026"
_PUNCTUATION_TRAIL = "".join(sorted(_PUNCTUATION)) + " \t"

# Abbreviations whose full stop does not end a sentence. Small on purpose: a longer list
# buys nothing on prompts this short and hides which rule fired.
_ABBREVIATIONS = ("mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "e.g", "i.e", "etc")


def count_words(text: str) -> int:
    """Whitespace-separated tokens containing at least one alphanumeric character.

    "well-known" is one word; a bare "-" or "**" is none. This is the convention IFEval
    uses and the one a person counting by hand would use.
    """
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


def split_sentences(text: str) -> list[str]:
    """Sentences, counting a final unterminated fragment as one.

    Decimals and a short list of abbreviations are protected so "1.5" and "e.g." do not
    inflate the count. An answer that ends without a full stop still counts its last
    clause, so an exact-sentence item fails for writing too much rather than for
    punctuation.
    """
    protected = re.sub(r"(\d)\.(\d)", "\\1\u0000\\2", text)
    for abbreviation in _ABBREVIATIONS:
        protected = re.sub(
            rf"(?<![A-Za-z]){re.escape(abbreviation)}\.",
            lambda match: match.group()[:-1] + "\u0000",
            protected,
            flags=re.IGNORECASE,
        )
    protected = protected.replace("...", "\u0000\u0000\u0000")
    parts = re.findall(r"[^.!?]+(?:[.!?]+[\"')\]]*|$)", protected)
    return [part.replace("\u0000", ".").strip() for part in parts if any(c.isalnum() for c in part)]


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def strip_code_fence(text: str) -> str:
    """Remove one surrounding triple-backtick fence, if the whole answer is inside it.

    Deliberate leniency: an item that tests JSON keys should fail for the wrong keys, not
    for a fence the model habitually adds. `code_fence_wrapped` tests fencing separately.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    closing = body.rfind("```")
    inner = body[:closing] if closing != -1 else body
    newline = inner.find("\n")
    if newline == -1:
        return inner.strip()
    first_line = inner[:newline].strip()
    # A language tag, or nothing. Anything else is content that happened to follow the
    # opening fence on the same line, and must be kept.
    if not first_line or re.fullmatch(r"[A-Za-z0-9_+#.-]{1,12}", first_line):
        return inner[newline + 1 :].strip()
    return inner.strip()


def normalise_token(text: str, *, allow_trailing_punctuation: bool = True) -> str:
    """Trim whitespace, surrounding quotes and (optionally) one trailing punctuation run."""
    token = text.strip().strip(_WORD_STRIP).strip()
    if allow_trailing_punctuation:
        token = token.rstrip(_TRAILING_PUNCTUATION).strip()
    return token


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# --------------------------------------------------------------------- numeric extraction

_NUMBER_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# Only phrases that announce a final answer. "equals" and "the total is" were deliberately
# left out: they occur mid-working, and taking the last of them picks up a verification
# step rather than the answer.
_ANSWER_MARKER = re.compile(r"(?:final answer|the answer is|answer\s*[:=]|answer is)", re.IGNORECASE)


def _to_number(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:  # pragma: no cover - the pattern cannot produce this
        return None


def _numbers_in(text: str) -> list[float]:
    values = [_to_number(match.group()) for match in _NUMBER_PATTERN.finditer(text)]
    return [value for value in values if value is not None]


def extract_final_number(text: str) -> float | None:
    """The number the answer is offering, or None if there is no number at all.

    Resolution order, most explicit first:

      1. the last non-empty line, when that line contains exactly one number. Every math
         prompt asks for the final number on its own line, so this is the intended shape
         and it survives "Answer: 1081", "$1,081." and "**1081**";
      2. the last \\boxed{...};
      3. the first number after the last final-answer phrase;
      4. the last number on the last non-empty line;
      5. the last number anywhere.

    A model that states the right number and then keeps arithmetic going after it is
    scored on what it ended with. That is the same rule a person marking the paper applies.
    """
    cleaned = text.replace("**", "").replace("__", "")
    cleaned = re.sub(r"[$\u00a3\u20ac\u00a5]", "", cleaned)

    lines = [line for line in cleaned.splitlines() if line.strip()]
    if lines:
        on_last_line = _numbers_in(lines[-1])
        if len(on_last_line) == 1:
            return on_last_line[0]

    boxed = re.findall(r"\\boxed\s*\{([^}]*)\}", cleaned)
    for candidate in reversed(boxed):
        numbers = _numbers_in(candidate)
        if numbers:
            return numbers[0]

    markers = list(_ANSWER_MARKER.finditer(cleaned))
    if markers:
        after = _numbers_in(cleaned[markers[-1].end() :])
        if after:
            return after[0]

    if lines:
        on_last_line = _numbers_in(lines[-1])
        if on_last_line:
            return on_last_line[-1]

    everywhere = _numbers_in(cleaned)
    return everywhere[-1] if everywhere else None


# -------------------------------------------------------------------------- the verifiers
#
# Every verifier is (answer_text, params) -> (passed, detail). Pure: no model calls, no
# clock, no randomness. The coding verifier is the one exception to "no side effects": it
# spawns a sandboxed subprocess, which is still deterministic for a given answer.


def _check_json_object(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    body = strip_code_fence(answer)
    payload: Any = None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        if params.get("allow_surrounding_text"):
            block = _first_balanced_object(body)
            if block is not None:
                try:
                    payload = json.loads(block)
                except json.JSONDecodeError:
                    payload = None
        if payload is None:
            return False, "the reply is not a single JSON value"
    if not isinstance(payload, dict):
        return False, f"the reply is JSON but a {type(payload).__name__}, not an object"

    required = list(params.get("required_keys") or [])
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"missing key(s): {', '.join(missing)}"
    if params.get("exact_keys"):
        extra = [key for key in payload if key not in required]
        if extra:
            return False, f"unexpected key(s): {', '.join(sorted(extra))}"

    json_types: dict[str, tuple[type, ...]] = {
        "string": (str,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }
    for key, wanted in (params.get("types") or {}).items():
        if wanted not in json_types:
            raise CapabilityError(f"unknown JSON type {wanted!r} in params")
        value = payload.get(key)
        # bool is a subclass of int; "number" must not silently accept true.
        if wanted == "number" and isinstance(value, bool):
            return False, f"key {key!r} is a boolean, not a number"
        if not isinstance(value, json_types[wanted]):
            return False, f"key {key!r} is {type(value).__name__}, not {wanted}"

    for key, options in (params.get("enums") or {}).items():
        value = payload.get(key)
        if value not in options:
            return False, f"key {key!r} is {value!r}, not one of {options}"

    for key, budget in (params.get("max_words") or {}).items():
        words = count_words(str(payload.get(key, "")))
        if words > int(budget):
            return False, f"key {key!r} has {words} words, over the limit of {budget}"

    return True, f"valid JSON object with keys {sorted(payload)}"


def _check_numbered_list(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    count = int(params["count"])
    numbered: list[int] = []
    others: list[str] = []
    for line in non_empty_lines(answer):
        match = re.match(r"^(\d+)[.)]\s+\S", line)
        if match:
            numbered.append(int(match.group(1)))
        else:
            others.append(line)
    if len(numbered) != count:
        return False, f"{len(numbered)} numbered lines, expected {count}"
    if numbered != list(range(1, count + 1)):
        return False, f"numbering is {numbered}, expected 1..{count}"
    if not params.get("allow_other_text", False) and others:
        return False, f"{len(others)} line(s) outside the list, first: {others[0][:70]!r}"
    return True, f"{count} numbered lines, nothing else" if not others else f"{count} numbered lines"


def _check_line_count(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    count = int(params["count"])
    lines = non_empty_lines(answer)
    if len(lines) != count:
        return False, f"{len(lines)} non-empty lines, expected {count}"
    return True, f"{count} non-empty lines"


def _check_line_ends_with(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    """A named line must end on a required word, ignoring trailing punctuation.

    This is how a verse form becomes machine-checkable without a rhyme dictionary. Pinning
    the last word of the first line forces the model to actually build a line to a given
    ending, which two lines of restated prose cannot do by accident.
    """
    index = int(params.get("line", 1))
    required = str(params["text"])
    lines = non_empty_lines(answer)
    if not lines:
        return False, "empty answer"
    position = index - 1 if index > 0 else len(lines) + index
    if not 0 <= position < len(lines):
        return False, f"{len(lines)} non-empty lines; there is no line {index}"
    line = lines[position].rstrip(_PUNCTUATION_TRAIL)
    flags = 0 if params.get("case_sensitive", False) else re.IGNORECASE
    if re.search(rf"\b{re.escape(required)}$", line, flags):
        return True, f"line {index} ends on {required!r}"
    return False, f"line {index} ends {line[-40:]!r}, not on {required!r}"


def _check_single_word(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    """Exactly one word.

    Trailing punctuation passes by default: the constraint the prompt states is a word
    count, and "Tokyo." is one word by any ordinary reading. Items that also forbid
    punctuation compose `single_word` with `no_punctuation` instead of relying on this.
    """
    allow = bool(params.get("allow_trailing_punctuation", True))
    raw = answer.strip().strip(_WORD_STRIP).strip()
    if not raw:
        return False, "empty answer"
    if not allow and raw[-1] in _TRAILING_PUNCTUATION:
        return False, f"trailing punctuation {raw[-1]!r} is not allowed on this item"
    token = normalise_token(answer, allow_trailing_punctuation=allow)
    words = token.split()
    if len(words) != 1:
        return False, f"{len(words)} words, expected 1: {token[:70]!r}"
    return True, f"one word: {token!r}"


def _check_one_of(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    options = [str(option) for option in params["options"]]
    allow = bool(params.get("allow_trailing_punctuation", True))
    token = normalise_token(answer, allow_trailing_punctuation=allow)
    if not params.get("case_sensitive", False):
        token_key = token.lower()
        lookup = {option.lower(): option for option in options}
    else:
        token_key = token
        lookup = {option: option for option in options}
    if token_key in lookup:
        return True, f"answered {lookup[token_key]!r}"
    return False, f"the whole reply is not one of {options}: {token[:80]!r}"


def _check_max_words(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    limit = int(params["count"])
    words = count_words(answer)
    if words > limit:
        return False, f"{words} words, over the limit of {limit}"
    return True, f"{words} words, within {limit}"


def _check_min_words(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    floor = int(params["count"])
    words = count_words(answer)
    if words < floor:
        return False, f"{words} words, under the floor of {floor}"
    return True, f"{words} words, at least {floor}"


def _check_exact_words(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    wanted = int(params["count"])
    words = count_words(answer)
    if words != wanted:
        return False, f"{words} words, expected exactly {wanted}"
    return True, f"exactly {wanted} words"


def _check_exact_sentences(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    wanted = int(params["count"])
    sentences = split_sentences(answer)
    if len(sentences) != wanted:
        return False, f"{len(sentences)} sentences, expected exactly {wanted}"
    return True, f"exactly {wanted} sentences"


def _check_all_lowercase(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    offenders = sorted({char for char in answer if char.isupper()})
    if offenders:
        return False, f"contains capital letter(s): {''.join(offenders[:8])}"
    return True, "no capital letters"


def _check_no_punctuation(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    offenders = sorted({char for char in answer if char in _PUNCTUATION})
    if offenders:
        return False, f"contains punctuation: {''.join(offenders[:8])}"
    return True, "no punctuation"


def _check_forbidden_letter(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    letter = str(params["letter"]).lower()
    if letter in answer.lower():
        occurrences = answer.lower().count(letter)
        return False, f"contains the letter {letter!r} {occurrences} time(s)"
    return True, f"no letter {letter!r}"


def _forbidden_pattern(substring: str, params: dict[str, Any]) -> re.Pattern[str]:
    if params.get("stem"):
        return re.compile(rf"\b{re.escape(substring)}\w*", re.IGNORECASE)
    if params.get("word_boundary", True):
        return re.compile(rf"\b{re.escape(substring)}\b", re.IGNORECASE)
    return re.compile(re.escape(substring), re.IGNORECASE)


def _check_forbidden_substrings(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    hits: list[str] = []
    for substring in params["substrings"]:
        match = _forbidden_pattern(str(substring), params).search(answer)
        if match:
            hits.append(match.group())
    if hits:
        return False, f"uses forbidden text: {', '.join(sorted(set(hits)))}"
    return True, "none of the forbidden text appears"


def _check_contains_all(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    missing = [
        str(substring)
        for substring in params["substrings"]
        if not _forbidden_pattern(str(substring), params).search(answer)
    ]
    if missing:
        return False, f"missing required text: {', '.join(missing)}"
    return True, "all required text present"


_LIST_MARKER = re.compile(r"^\s*(?:[-*\u2022+\u2013\u2014]\s+|\d+[.)]\s+)")


def _check_no_list_markers(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    for line in answer.splitlines():
        if _LIST_MARKER.match(line):
            return False, f"line starts a list: {line.strip()[:70]!r}"
    return True, "no bullet or numbered lines"


def _check_ends_with(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    required = str(params["text"])
    body = answer.strip()
    if not params.get("case_sensitive", True):
        body, required_key = body.lower(), required.lower()
    else:
        required_key = required
    if body.endswith(required_key):
        return True, f"ends with {required!r}"
    return False, f"ends with {answer.strip()[-len(required) - 20:]!r}, not {required!r}"


def _check_starts_with(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    required = str(params["text"])
    body = answer.strip()
    if not params.get("case_sensitive", True):
        body, required_key = body.lower(), required.lower()
    else:
        required_key = required
    if body.startswith(required_key):
        return True, f"starts with {required!r}"
    return False, f"starts with {answer.strip()[: len(required) + 20]!r}, not {required!r}"


def _check_code_fence_wrapped(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    body = answer.strip()
    if not body.startswith("```"):
        return False, f"does not open with a fence: {body[:60]!r}"
    if not body.endswith("```") or len(body) < 7:
        return False, "does not close with a fence at the very end"
    inner = body[3:-3]
    newline = inner.find("\n")
    content = inner[newline + 1 :] if newline != -1 else inner
    if not content.strip():
        return False, "the fence is empty"
    return True, "the whole answer is inside one fence"


def _check_regex_full_match(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    pattern = re.compile(str(params["pattern"]), re.IGNORECASE if params.get("ignore_case") else 0)
    body = answer.strip()
    if pattern.fullmatch(body):
        return True, "matches the required pattern"
    return False, f"does not match {params['pattern']!r}: {body[:80]!r}"


def _check_numeric_answer(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    expected = float(params["expected"])
    tolerance = float(params.get("tolerance", 1e-6))
    found = extract_final_number(answer)
    if found is None:
        return False, "no number in the reply"
    if abs(found - expected) <= tolerance:
        return True, f"final number {_pretty(found)} matches"
    return False, f"final number {_pretty(found)}, expected {_pretty(expected)}"


def _pretty(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _check_all_of(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    """Every sub-check must pass. The detail names the first failure and counts the rest."""
    checks = params.get("checks") or []
    if not checks:
        raise CapabilityError("all_of with no checks")
    failures: list[str] = []
    details: list[str] = []
    for entry in checks:
        kind = entry["kind"]
        passed, detail = run_verifier(kind, answer, entry.get("params") or {})
        details.append(f"{kind}: {detail}")
        if not passed:
            failures.append(f"{kind}: {detail}")
    if failures:
        return False, "; ".join(failures)
    return True, "; ".join(details)


# ------------------------------------------------------------------ the coding sandbox

_SANDBOX_PRELUDE = '''\
_CAPABILITY_MARKER = "<<<capability-result>>>"

import socket as _capability_socket


def _capability_blocked(*_args, **_kwargs):
    raise RuntimeError("network access is disabled in the capability sandbox")


_capability_socket.socket = _capability_blocked
_capability_socket.create_connection = _capability_blocked
_capability_socket.socketpair = _capability_blocked

# ---- model-written code below this line ----
'''

_SANDBOX_FOOTER = '''

# ---- harness below this line ----
def _capability_report():
    import json as _capability_json

    with open("cases.json", encoding="utf-8") as _handle:
        _cases = _capability_json.load(_handle)
    _fn = globals().get(__ENTRY_POINT__)
    if not callable(_fn):
        print(_CAPABILITY_MARKER + _capability_json.dumps(
            {"error": "no callable named " + __ENTRY_POINT__}
        ))
        return
    _outcomes = []
    for _args, _expected in _cases:
        try:
            _got = _fn(*_args)
        except BaseException as _exc:
            _outcomes.append({
                "ok": False,
                "args": _args,
                "expected": _expected,
                "error": type(_exc).__name__ + ": " + str(_exc)[:120],
            })
            continue
        _outcomes.append({
            "ok": _got == _expected,
            "args": _args,
            "expected": _expected,
            "got": repr(_got)[:120],
        })
    print(_CAPABILITY_MARKER + _capability_json.dumps({"outcomes": _outcomes}))


_capability_report()
'''


def extract_python_code(answer: str, entry_point: str) -> str | None:
    """The Python source the answer is offering, or None if there is none.

    Prefers a fenced block that actually defines the wanted function, then any fenced
    block, then the raw answer when it defines the function without a fence.
    """
    blocks = re.findall(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", answer, re.DOTALL)
    definition = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE)
    for block in blocks:
        if definition.search(block):
            return block
    if blocks:
        return blocks[0]
    if definition.search(answer):
        return answer
    return None


def _sandbox_limits() -> None:  # pragma: no cover - runs in the forked child
    assert _resource is not None
    _resource.setrlimit(_resource.RLIMIT_CPU, (CODE_CPU_SECONDS, CODE_CPU_SECONDS))
    _resource.setrlimit(_resource.RLIMIT_AS, (CODE_MEMORY_BYTES, CODE_MEMORY_BYTES))
    _resource.setrlimit(_resource.RLIMIT_FSIZE, (CODE_MAX_FILE_BYTES, CODE_MAX_FILE_BYTES))
    os.setsid()


def run_code_in_sandbox(
    source: str, entry_point: str, cases: Sequence[Any], timeout_s: float = CODE_WALL_TIMEOUT_S
) -> dict[str, Any]:
    """Execute `source` against `cases` in a restricted subprocess. See SANDBOX_NOTE.

    Returns {"status": ok|timeout|crashed|no_result, ...}. Never raises for anything the
    model wrote; only a broken harness raises.
    """
    workdir = Path(tempfile.mkdtemp(prefix="persona_eval_code_"))
    try:
        script = (
            _SANDBOX_PRELUDE
            + source
            + _SANDBOX_FOOTER.replace("__ENTRY_POINT__", json.dumps(entry_point))
        )
        (workdir / "check.py").write_text(script, encoding="utf-8")
        (workdir / "cases.json").write_text(json.dumps(list(cases)), encoding="utf-8")
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
        }
        process = subprocess.Popen(  # noqa: S603 - the command is fixed, the payload is data
            [sys.executable, "-I", "-S", "-B", "check.py"],
            cwd=str(workdir),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_sandbox_limits,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):  # pragma: no cover
                process.kill()
            process.communicate()
            return {"status": "timeout", "timeout_s": timeout_s}

        for line in reversed(stdout.splitlines()):
            if line.startswith(CODE_RESULT_MARKER):
                payload = json.loads(line[len(CODE_RESULT_MARKER) :])
                payload["status"] = "ok"
                return payload
        if process.returncode != 0:
            return {"status": "crashed", "returncode": process.returncode, "stderr": stderr[-400:]}
        return {"status": "no_result", "stderr": stderr[-400:], "stdout": stdout[-200:]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


_PLACEHOLDER = re.compile(r"\b(?:pass|NotImplementedError|TODO|FIXME|\.\.\.)\b")


def _static_code_check(source: str, entry_point: str) -> tuple[bool, str]:
    """Fallback when no sandbox is available: structure only, never correctness."""
    if not re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", source, re.MULTILINE):
        return False, f"static check only: no def {entry_point}(...)"
    if "return" not in source:
        return False, "static check only: the function never returns"
    if _PLACEHOLDER.search(source) and source.count("\n") < 4:
        return False, "static check only: the body is a placeholder"
    return True, f"static check only (no sandbox on this platform): defines {entry_point} and returns"


def _check_python_function(answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    entry_point = str(params["entry_point"])
    cases = params["cases"]
    source = extract_python_code(answer, entry_point)
    if source is None:
        return False, f"no Python source defining {entry_point} in the reply"
    if not SANDBOX_AVAILABLE:
        return _static_code_check(source, entry_point)

    outcome = run_code_in_sandbox(
        source, entry_point, cases, float(params.get("timeout_s", CODE_WALL_TIMEOUT_S))
    )
    status = outcome.get("status")
    if status == "timeout":
        return False, f"did not finish within {outcome['timeout_s']}s"
    if status == "crashed":
        return False, f"the code did not run: {outcome.get('stderr', '').strip()[-200:]}"
    if status == "no_result" or "outcomes" not in outcome:
        if outcome.get("error"):
            return False, str(outcome["error"])
        return False, f"no result from the sandbox: {str(outcome.get('stderr', ''))[-200:]}"

    outcomes = outcome["outcomes"]
    failed = [row for row in outcomes if not row.get("ok")]
    if failed:
        first = failed[0]
        got = first.get("error") or first.get("got")
        return (
            False,
            f"{len(outcomes) - len(failed)}/{len(outcomes)} cases pass; "
            f"{entry_point}(*{first['args']}) gave {got}, expected {first['expected']!r}",
        )
    return True, f"{len(outcomes)}/{len(outcomes)} cases pass"


# ---------------------------------------------------------------------------- the registry

VERIFIERS: dict[str, Callable[[str, dict[str, Any]], tuple[bool, str]]] = {
    "all_lowercase": _check_all_lowercase,
    "all_of": _check_all_of,
    "code_fence_wrapped": _check_code_fence_wrapped,
    "contains_all": _check_contains_all,
    "ends_with": _check_ends_with,
    "exact_sentences": _check_exact_sentences,
    "exact_words": _check_exact_words,
    "forbidden_letter": _check_forbidden_letter,
    "forbidden_substrings": _check_forbidden_substrings,
    "json_object": _check_json_object,
    "line_count": _check_line_count,
    "line_ends_with": _check_line_ends_with,
    "max_words": _check_max_words,
    "min_words": _check_min_words,
    "no_list_markers": _check_no_list_markers,
    "no_punctuation": _check_no_punctuation,
    "numbered_list": _check_numbered_list,
    "numeric_answer": _check_numeric_answer,
    "one_of": _check_one_of,
    "python_function": _check_python_function,
    "regex_full_match": _check_regex_full_match,
    "single_word": _check_single_word,
    "starts_with": _check_starts_with,
}


def run_verifier(kind: str, answer: str, params: dict[str, Any]) -> tuple[bool, str]:
    verifier = VERIFIERS.get(kind)
    if verifier is None:
        raise CapabilityError(f"unknown verifier kind {kind!r}. Have: {sorted(VERIFIERS)}")
    try:
        return verifier(answer, params)
    except CapabilityError:
        raise
    except KeyError as error:
        raise CapabilityError(f"verifier {kind!r} needs param {error}") from None


def verify(item: CapabilityItem, answer_text: str) -> dict[str, Any]:
    """Grade one answer. Pure with respect to the model: no calls, no judge, no network.

    Returns {"passed": bool, "detail": str}. `detail` is written to be read in a report
    without the answer next to it: it says what the answer did, not only that it failed.
    """
    passed, detail = run_verifier(item.verifier, answer_text or "", dict(item.params))
    return {"passed": bool(passed), "detail": detail}


# ------------------------------------------------------------------------------- running


def sampling_settings(config: RunConfig, role: Any) -> tuple[float, int]:
    """Temperature and answer budget, read exactly as persona_eval.evaluate reads them.

    Duplicating the two lines rather than importing them keeps this module free of the
    judge path, but the keys must not drift: a capability comparison made at different
    settings from the values comparison would not be evidence about the same model.
    """
    temperature = float(config.evaluation.get("temperature", 0.7))
    configured_max = config.evaluation.get("answer_max_tokens")
    max_tokens = int(configured_max) if configured_max else int(role.max_tokens)
    return temperature, max_tokens


def build_messages(item: CapabilityItem) -> list[dict[str, str]]:
    """One user turn, fresh context, no system prompt. See the module docstring."""
    return [{"role": "user", "content": item.prompt}]


def _row(
    arm: str,
    item: CapabilityItem,
    passed: bool,
    detail: str,
    answer: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "arm": arm,
        "check_id": item.check_id,
        "family": item.family,
        "passed": bool(passed),
        "detail": detail,
        "answer": answer,
        "meta": meta,
    }


async def run_capability(
    config: RunConfig,
    endpoint_role: str,
    arm: str,
    items: Iterable[CapabilityItem] | None = None,
    usage_path: Path | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Answer every check item with one model role and verify the answers.

    `arm` is the label this model is reported under; `endpoint_role` is the config role
    that is actually called, so the same code runs the local base model and an adapted
    checkpoint. Rows come back in item order, one per item, including for items whose
    call failed: those carry `meta["technical_failure"]` so the report can hold them out
    of the denominator rather than counting a network error as a capability loss.
    """
    role = config.role(endpoint_role)
    temperature, max_tokens = sampling_settings(config, role)
    chosen = select(list(items) if items is not None else list(CAPABILITY_ITEMS), limit)
    if not chosen:
        raise RuntimeError("No capability items selected.")
    logger.info(
        "capability: %d items on arm %r via role %r (%s) at temperature %.2f, max_tokens %d",
        len(chosen),
        arm,
        endpoint_role,
        role.model,
        temperature,
        max_tokens,
    )

    base_meta = {
        "endpoint_role": endpoint_role,
        "model": role.model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "sandbox_available": SANDBOX_AVAILABLE,
    }

    async with ModelClient.from_config(config, usage_path, "capability") as client:

        async def answer_and_verify(item: CapabilityItem) -> dict[str, Any]:
            meta = dict(base_meta, verifier=item.verifier, tags=list(item.tags), description=item.description)
            try:
                response = await client.complete(
                    role,
                    build_messages(item),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stage=f"capability.{arm}",
                    record_id=item.check_id,
                )
            except LocalEndpointUnavailable:
                raise
            except ModelError as error:
                logger.error("capability %s: model call failed: %s", item.check_id, error)
                meta["technical_failure"] = str(error)
                return _row(arm, item, False, f"technical failure: {error}", "", meta)
            meta.update(
                finish_reason=response.finish_reason,
                truncated=response.truncated,
                latency_s=round(response.latency_s, 3),
                request_id=response.request_id,
            )
            try:
                # Verification of the coding family spawns a subprocess; keep it off the
                # event loop so the other items keep flowing.
                verdict = await asyncio.to_thread(verify, item, response.text)
            except CapabilityError as error:
                logger.error("capability %s: %s", item.check_id, error)
                meta["technical_failure"] = str(error)
                return _row(arm, item, False, f"verifier error: {error}", response.text, meta)
            return _row(arm, item, verdict["passed"], verdict["detail"], response.text, meta)

        results = await gather_bounded([answer_and_verify(item) for item in chosen])

    rows: list[dict[str, Any]] = []
    for item, result in zip(chosen, results):
        if isinstance(result, LocalEndpointUnavailable):
            raise RuntimeError(str(result)) from None
        if isinstance(result, BaseException):
            logger.error("capability %s: unexpected failure: %s", item.check_id, result)
            meta = dict(base_meta, verifier=item.verifier, technical_failure=repr(result))
            rows.append(_row(arm, item, False, f"technical failure: {result}", "", meta))
            continue
        rows.append(result)
    return rows


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts per family and per tag, for the CLI. Never a single capability score.

    Technical failures are counted and excluded from the denominators: an endpoint error
    is not evidence about the model's ability, and the plan keeps the two apart.
    """
    def bucket() -> dict[str, int]:
        return {"n": 0, "passed": 0, "technical_failures": 0}

    totals = bucket()
    by_family: dict[str, dict[str, int]] = {}
    by_tag: dict[str, dict[str, int]] = {}
    for row in rows:
        meta = row.get("meta") or {}
        failed_technically = bool(meta.get("technical_failure"))
        targets = [totals, by_family.setdefault(row["family"], bucket())]
        targets += [by_tag.setdefault(tag, bucket()) for tag in (meta.get("tags") or [])]
        for target in targets:
            if failed_technically:
                target["technical_failures"] += 1
                continue
            target["n"] += 1
            target["passed"] += 1 if row.get("passed") else 0
    for target in [totals, *by_family.values(), *by_tag.values()]:
        target["pass_rate"] = round(target["passed"] / target["n"], 3) if target["n"] else None
    return {
        "arm": rows[0]["arm"] if rows else "",
        "total": totals,
        "by_family": by_family,
        "by_tag": by_tag,
    }


__all__ = [
    "CODE_WALL_TIMEOUT_S",
    "CapabilityError",
    "SANDBOX_AVAILABLE",
    "SANDBOX_NOTE",
    "VERIFIERS",
    "build_messages",
    "count_words",
    "extract_final_number",
    "extract_python_code",
    "run_capability",
    "run_code_in_sandbox",
    "run_verifier",
    "sampling_settings",
    "split_sentences",
    "strip_code_fence",
    "summarise",
    "verify",
]
