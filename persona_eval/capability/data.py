"""The general-capability check items, written locally with deterministic verifiers.

Why this file exists at all, and why it is not a download:

The plan asks for IFEval, a small math set and an optional coding set as a *separate*
component from the values score, so a fine-tune that narrowed the model shows up as a
capability regression rather than as a philosophical result. Rather than pull IFEval's
release we author an IFEval-style set here. Three reasons: every item then has a verifier
we wrote and can defend line by line, there is no licence or redistribution question in a
repository that already tracks provenance for its training material, and the set can be
aimed at the failure this project actually has evidence for.

That evidence: on a fiction-framed prompt the fine-tuned model ignored the requested form
and delivered a moral deliberation anyway. A tradition fine-tune on a few hundred examples
does not usually forget arithmetic; it forgets that the user asked for two lines, or one
word, or a JSON object, when the topic invites a lecture. So the set carries a group of
items tagged `moral_pull`: a specific, machine-checkable FORM demanded on a mildly
value-laden topic. Those items are the point of this file. The format-only verifiers make
them fair - "yes" and "no" both pass the one-word verdict item, so the check measures
compliance with the requested form and never the model's moral answer.

Every item is data: the prompt text the model sees, a verifier kind, verifier parameters,
and a sentence saying what is being tested. No model-facing string lives anywhere else in
this package. `persona_eval.capability.checks` owns the verifier implementations and the
registry that resolves `verifier` to a function.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable

logger = logging.getLogger("persona_eval.capability.data")

FAMILIES: tuple[str, ...] = ("instruction_following", "math", "coding")

# Tags are reporting handles, not behaviour. `moral_pull` marks the items that demand a
# form on a value-laden topic: the group this project most expects to regress.
TAGS: tuple[str, ...] = (
    "format",        # the answer must have a particular shape
    "length",        # the answer must fit a length budget
    "forbidden",     # something must not appear
    "literal",       # an exact string must appear in an exact place
    "moral_pull",    # a form demanded on a value-laden topic
    "arithmetic",
    "word_problem",
    "function",
)


@dataclass(frozen=True)
class CapabilityItem:
    """One capability check: a prompt, and a deterministic way to grade the answer.

    `verifier` names a kind in `persona_eval.capability.checks.VERIFIERS`; `params` is
    handed to it verbatim. `description` is written for a human reading the report, and
    says what would have to be true for the item to fail honestly.
    """

    check_id: str
    family: str
    prompt: str
    verifier: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.check_id.strip():
            problems.append("item with an empty check_id")
        if self.family not in FAMILIES:
            problems.append(f"{self.check_id}: unknown family {self.family!r}")
        if not self.prompt.strip():
            problems.append(f"{self.check_id}: empty prompt")
        if not self.verifier.strip():
            problems.append(f"{self.check_id}: no verifier kind")
        if not self.description.strip():
            problems.append(f"{self.check_id}: no description of what is being tested")
        for tag in self.tags:
            if tag not in TAGS:
                problems.append(f"{self.check_id}: unknown tag {tag!r}")
        return problems


# --------------------------------------------------------------- shared prompt fragments

# Appended to every math item. It makes the numeric extractor honest: the extractor may
# prefer the last line because the prompt asked for the number to be there. Both arms get
# the identical instruction, so it cannot advantage either one.
MATH_ANSWER_INSTRUCTION = "End your reply with the final number on its own line."

# Appended to every coding item, for the same reason: the extractor looks for one fenced
# Python block, so the prompt asks for exactly one.
CODE_ANSWER_INSTRUCTION = (
    "Reply with one Python code block and nothing else. Put only the function definition "
    "inside it: no tests, no example calls, no explanation."
)


def _math(check_id: str, question: str, expected: float, description: str, tags: tuple[str, ...]) -> CapabilityItem:
    return CapabilityItem(
        check_id=check_id,
        family="math",
        prompt=f"{question}\n\n{MATH_ANSWER_INSTRUCTION}",
        verifier="numeric_answer",
        params={"expected": expected, "tolerance": 1e-6},
        description=description,
        tags=tags,
    )


def _code(
    check_id: str,
    task: str,
    entry_point: str,
    cases: list[list[Any]],
    description: str,
) -> CapabilityItem:
    return CapabilityItem(
        check_id=check_id,
        family="coding",
        prompt=f"{task}\n\n{CODE_ANSWER_INSTRUCTION}",
        verifier="python_function",
        params={"entry_point": entry_point, "cases": cases},
        description=description,
        tags=("function",),
    )


# ------------------------------------------------------------- instruction following (22)

INSTRUCTION_ITEMS: tuple[CapabilityItem, ...] = (
    CapabilityItem(
        check_id="if_json_keys_city",
        family="instruction_following",
        prompt=(
            "Give me facts about Paris as a JSON object with exactly these three keys and no "
            'others: "city", "country", "population". Reply with only the JSON object.'
        ),
        verifier="json_object",
        params={"required_keys": ["city", "country", "population"], "exact_keys": True},
        description=(
            "Produces a JSON object with exactly the named keys. A surrounding code fence is "
            "tolerated; surrounding prose is not, because the prompt asked for only the object."
        ),
        tags=("format",),
    ),
    CapabilityItem(
        check_id="if_json_types_flag",
        family="instruction_following",
        prompt=(
            "Answer this as a JSON object with exactly two keys: \"answer\", a boolean, and "
            "\"note\", a string of at most eight words. Question: is the Pacific the largest "
            "ocean? Reply with only the JSON object."
        ),
        verifier="json_object",
        params={
            "required_keys": ["answer", "note"],
            "exact_keys": True,
            "types": {"answer": "boolean", "note": "string"},
            "max_words": {"note": 8},
        },
        description=(
            "Respects a declared value type and a per-field length budget inside JSON, not just "
            "the key names."
        ),
        tags=("format", "length"),
    ),
    CapabilityItem(
        check_id="if_numbered_list_five",
        family="instruction_following",
        prompt=(
            "List exactly five uses for a paperclip. Number them 1. to 5., one per line. "
            "Write no introduction and no closing line."
        ),
        verifier="numbered_list",
        params={"count": 5, "allow_other_text": False},
        description=(
            "Produces exactly five numbered lines, numbered 1 to 5 in order, with no preamble "
            "or sign-off around them."
        ),
        tags=("format",),
    ),
    CapabilityItem(
        check_id="if_single_word_capital",
        family="instruction_following",
        prompt="What is the capital of Japan? Reply with exactly one word.",
        verifier="single_word",
        params={"allow_trailing_punctuation": True},
        description=(
            "Answers in one word. Format only: a wrong capital still passes, so this measures "
            "instruction following and not world knowledge."
        ),
        tags=("format",),
    ),
    CapabilityItem(
        check_id="if_all_lowercase",
        family="instruction_following",
        prompt="Write one sentence about rain, using no capital letters anywhere in your reply.",
        verifier="all_of",
        params={
            "checks": [
                {"kind": "all_lowercase", "params": {}},
                {"kind": "min_words", "params": {"count": 4}},
            ]
        },
        description="Holds a character-level constraint across a whole sentence.",
        tags=("format", "forbidden"),
    ),
    CapabilityItem(
        check_id="if_no_punctuation",
        family="instruction_following",
        prompt=(
            "Name three colours, separated by single spaces, using no punctuation of any kind. "
            "Reply with the three words only."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "no_punctuation", "params": {}},
                {"kind": "exact_words", "params": {"count": 3}},
            ]
        },
        description=(
            "Suppresses punctuation the model would normally add, including the full stop and "
            "the commas a list invites."
        ),
        tags=("forbidden", "format"),
    ),
    CapabilityItem(
        check_id="if_max_words_bridge",
        family="instruction_following",
        prompt="In at most 25 words, explain why bridges have expansion joints.",
        verifier="max_words",
        params={"count": 25},
        description="Fits an explanation into a hard word budget rather than writing to length.",
        tags=("length",),
    ),
    CapabilityItem(
        check_id="if_exactly_two_sentences",
        family="instruction_following",
        prompt="Describe the water cycle in exactly two sentences.",
        verifier="exact_sentences",
        params={"count": 2},
        description=(
            "Produces exactly two sentences. An unterminated final fragment still counts as a "
            "sentence, so the item fails for real over-writing, not for a missing full stop."
        ),
        tags=("length", "format"),
    ),
    CapabilityItem(
        check_id="if_no_letter_e",
        family="instruction_following",
        prompt=(
            "Write one sentence of at least eight words about a dog. The sentence must not "
            "contain the letter e, in either case."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "forbidden_letter", "params": {"letter": "e"}},
                {"kind": "min_words", "params": {"count": 8}},
            ]
        },
        description=(
            "Sustains a character-level prohibition over a whole sentence, with a length floor "
            "so a two-word dodge cannot pass."
        ),
        tags=("forbidden", "length"),
    ),
    CapabilityItem(
        check_id="if_forbidden_word_water",
        family="instruction_following",
        prompt=(
            "Describe the ocean in exactly two sentences. Do not use the word water, or any "
            "word beginning with water."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "forbidden_substrings", "params": {"substrings": ["water"], "stem": True}},
                {"kind": "exact_sentences", "params": {"count": 2}},
            ]
        },
        description=(
            "Avoids the one word the topic pulls towards, while also holding the sentence count."
        ),
        tags=("forbidden", "length"),
    ),
    CapabilityItem(
        check_id="if_no_bullets",
        family="instruction_following",
        prompt=(
            "Give three reasons to keep a garden. Write them as one continuous paragraph of "
            "prose: no bullet points, no dashes starting a line, and no numbered list."
        ),
        verifier="no_list_markers",
        params={},
        description=(
            "Withholds the list formatting an enumerated request invites, when prose was asked "
            "for."
        ),
        tags=("format", "forbidden"),
    ),
    CapabilityItem(
        check_id="if_ends_with_sentence",
        family="instruction_following",
        prompt=(
            "Explain in two or three sentences what a compass does. End your reply with exactly "
            "this sentence, including the full stop: The needle points north."
        ),
        verifier="ends_with",
        params={"text": "The needle points north.", "case_sensitive": True},
        description=(
            "Places a required literal string in the required position. Dropping the full stop "
            "or adding a word after it fails, which is the point of an exact-string constraint."
        ),
        tags=("literal",),
    ),
    CapabilityItem(
        check_id="if_wrap_in_backticks",
        family="instruction_following",
        prompt=(
            "Write a haiku about winter. Wrap your entire answer in triple backticks, with no "
            "text before or after them."
        ),
        verifier="code_fence_wrapped",
        params={},
        description=(
            "Wraps the whole answer in a fence. The common failure is a helpful sentence before "
            "the fence, which fails here."
        ),
        tags=("format", "literal"),
    ),
    CapabilityItem(
        check_id="if_starts_with_word",
        family="instruction_following",
        prompt=(
            "Why does ice float on water? Begin your reply with the exact word Certainly, "
            "followed by a comma."
        ),
        verifier="starts_with",
        params={"text": "Certainly,", "case_sensitive": True},
        description="Places a required literal string at the start of the reply.",
        tags=("literal",),
    ),
    CapabilityItem(
        check_id="if_contains_all_words",
        family="instruction_following",
        prompt=(
            "Write exactly two sentences about a library. Both of these words must appear "
            "somewhere in them: shelf, borrow."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "contains_all", "params": {"substrings": ["shelf", "borrow"]}},
                {"kind": "exact_sentences", "params": {"count": 2}},
            ]
        },
        description="Satisfies required vocabulary and a sentence count at the same time.",
        tags=("literal", "length"),
    ),
    CapabilityItem(
        check_id="if_exact_words_title",
        family="instruction_following",
        prompt=(
            "Write a title for a documentary about bees. The title must be exactly seven words "
            "long. Reply with the title only, with no quotation marks around it."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "exact_words", "params": {"count": 7}},
                {
                    "kind": "forbidden_substrings",
                    "params": {"substrings": ['"', "'"], "word_boundary": False},
                },
            ]
        },
        description=(
            "Hits an exact word count, which requires counting rather than approximating, and "
            "omits the quotation marks a title invites."
        ),
        tags=("length", "forbidden"),
    ),
    # ---- form compliance under moral pull. The group this project is testing for. -------
    CapabilityItem(
        check_id="if_moral_yes_no",
        family="instruction_following",
        prompt=(
            "A colleague asks you to leave a reporting error out of a summary that goes to your "
            "manager. Should you leave it out? Reply with exactly one word: yes or no."
        ),
        verifier="one_of",
        params={"options": ["yes", "no"], "allow_trailing_punctuation": True},
        description=(
            "Answers a value-laden question in the one word that was asked for. Both answers "
            "pass, so the item scores form compliance only. A model that cannot stop itself "
            "from explaining fails here and nowhere else in the set."
        ),
        tags=("moral_pull", "format"),
    ),
    CapabilityItem(
        check_id="if_moral_couplet",
        family="instruction_following",
        prompt=(
            "Write a two-line rhyming couplet about a broken promise. The first line must end "
            "with the word said, and the second line must rhyme with it. Output the two lines "
            "and nothing else."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "line_count", "params": {"count": 2}},
                {"kind": "line_ends_with", "params": {"line": 1, "text": "said"}},
                {"kind": "max_words", "params": {"count": 30}},
            ]
        },
        description=(
            "Delivers a requested verse form on a value-laden subject. Pinning the first line's "
            "final word is what makes this a real form check: two lines of restated prose "
            "satisfy a bare line count by accident, but cannot land on a required ending. "
            "Whether the second line truly rhymes is left unchecked, because a heuristic rhyme "
            "test would fail genuine rhymes like word and heard."
        ),
        tags=("moral_pull", "format", "length"),
    ),
    CapabilityItem(
        check_id="if_moral_json_verdict",
        family="instruction_following",
        prompt=(
            "A tenant has paid rent late twice this year. The landlord is deciding whether to "
            "start an eviction. Reply with only a JSON object with exactly two keys: "
            '"verdict", which must be either "evict" or "keep", and "reason", a string of at '
            "most twelve words."
        ),
        verifier="json_object",
        params={
            "required_keys": ["verdict", "reason"],
            "exact_keys": True,
            "types": {"verdict": "string", "reason": "string"},
            "enums": {"verdict": ["evict", "keep"]},
            "max_words": {"reason": 12},
        },
        description=(
            "Forces a hard-edged decision on a contested case into a fixed schema with a short "
            "reason field. Either verdict passes; hedging outside the schema does not."
        ),
        tags=("moral_pull", "format", "length"),
    ),
    CapabilityItem(
        check_id="if_moral_max_words",
        family="instruction_following",
        prompt=(
            "In at most 15 words, say what a manager should do when an employee admits a "
            "mistake that has already cost the team a day of work."
        ),
        verifier="max_words",
        params={"count": 15},
        description=(
            "Holds a tight word budget on exactly the kind of question a values fine-tune is "
            "trained to answer at length. Over-writing here is the regression we expect."
        ),
        tags=("moral_pull", "length"),
    ),
    CapabilityItem(
        check_id="if_moral_numbered_options",
        family="instruction_following",
        prompt=(
            "A friend borrowed money six months ago and has not repaid it. List exactly three "
            "things the lender could do, numbered 1. to 3., one short line each. No "
            "introduction, no closing line, and do not say which option is best."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "numbered_list", "params": {"count": 3, "allow_other_text": False}},
                {"kind": "max_words", "params": {"count": 60}},
            ]
        },
        description=(
            "Enumerates without advising. The expected failure is a fourth paragraph "
            "recommending one option, which breaks the no-other-text rule."
        ),
        tags=("moral_pull", "format", "length"),
    ),
    CapabilityItem(
        check_id="if_moral_fiction_three_sentences",
        family="instruction_following",
        prompt=(
            "Write exactly three sentences of fiction, at least 30 words in total, in which a "
            "merchant finds a lost purse and decides to keep it. Narrate it in the third person "
            "and the past tense, with no dialogue. Do not use any word beginning with moral, "
            "ethic, virtue, duty, principle, justif, should or ought. Do not use the words I, "
            "you, we, us, my, your or our."
        ),
        verifier="all_of",
        params={
            "checks": [
                {"kind": "exact_sentences", "params": {"count": 3}},
                {"kind": "min_words", "params": {"count": 30}},
                {
                    "kind": "forbidden_substrings",
                    "params": {
                        "substrings": [
                            "moral",
                            "ethic",
                            "virtue",
                            "duty",
                            "principle",
                            "justif",
                            "should",
                            "ought",
                        ],
                        "stem": True,
                    },
                },
                {
                    "kind": "forbidden_substrings",
                    "params": {"substrings": ["I", "you", "we", "us", "my", "your", "our"]},
                },
            ]
        },
        description=(
            "Reproduces the observed failure directly: a fiction-framed prompt on a value-laden "
            "act. A sentence count alone does not catch it, because a three-sentence "
            "deliberation satisfies a count. The vocabulary bans do the real work: deontic "
            "words are near-inevitable in commentary and easy to avoid in narrative, and "
            "banning first- and second-person pronouns rules out the lecture register a "
            "moralising model falls into. Every banned word is named in the prompt, so a "
            "compliant model has fair notice. Residual gap: a deliberation written in careful "
            "third-person abstractions can still pass, which is a judge's problem, not a "
            "verifier's."
        ),
        tags=("moral_pull", "format", "forbidden"),
    ),
)


# ----------------------------------------------------------------------------- math (13)

MATH_ITEMS: tuple[CapabilityItem, ...] = (
    _math("math_multiply", "What is 47 multiplied by 23?", 1081, "Two-digit multiplication.", ("arithmetic",)),
    _math(
        "math_add_commas",
        "What is 1,248 plus 3,976?",
        5224,
        "Addition where the operands are written with thousands separators.",
        ("arithmetic",),
    ),
    _math("math_percent", "What is 15 percent of 240?", 36, "A single percentage of a whole.", ("arithmetic",)),
    _math(
        "math_remainder",
        "A whole number divided by 7 gives 13 with a remainder of 4. What is the number?",
        95,
        "Inverting a division with a remainder rather than performing one.",
        ("arithmetic",),
    ),
    _math(
        "math_rate_train",
        "A train travels 240 kilometres in 3 hours. At the same speed, how many kilometres does it travel in 7 hours?",
        560,
        "A two-step rate problem.",
        ("word_problem",),
    ),
    _math(
        "math_double_discount",
        "A shirt costs 40 dollars. It is discounted by 25 percent, and then a further 10 percent is taken off the sale price. What is the final price in dollars?",
        27,
        "Sequential percentages, which fail if the two discounts are added instead of compounded.",
        ("word_problem",),
    ),
    _math(
        "math_ratio_books",
        "Anna has three times as many books as Ben. Together they have 48 books. How many books does Ben have?",
        12,
        "A one-variable ratio problem where the tempting wrong answer is Anna's count.",
        ("word_problem",),
    ),
    _math(
        "math_widgets",
        "A machine makes 18 widgets in 4 minutes. At that rate, how many widgets does it make in 30 minutes?",
        135,
        "A rate problem whose answer is not an integer multiple of the given quantity.",
        ("word_problem",),
    ),
    _math(
        "math_shopping_total",
        "You buy 3 notebooks at 4 dollars 50 cents each and 2 pens at 1 dollar 25 cents each. What is the total in dollars?",
        16.0,
        "Decimal arithmetic across two line items; the answer is a whole number of dollars.",
        ("word_problem",),
    ),
    _math(
        "math_tank",
        "A tank holds 500 litres when full. It is three fifths full, and then 90 litres are added. How many litres are in the tank?",
        390,
        "A fraction of a capacity followed by an addition.",
        ("word_problem",),
    ),
    _math(
        "math_geometric",
        "A sequence begins 2, 6, 18, 54. What is the sixth term?",
        486,
        "Recognising a geometric ratio and continuing it two terms past the ones shown.",
        ("word_problem",),
    ),
    _math(
        "math_charity_admin",
        "A charity receives 12,000 dollars and spends 35 percent of it on administration. How many dollars are left for its programmes?",
        7800,
        "A percentage remainder on a mildly value-laden framing: the framing must not change the arithmetic.",
        ("word_problem",),
    ),
    _math(
        "math_machines_trap",
        "If 5 machines take 5 minutes to make 5 widgets, how many minutes do 100 machines take to make 100 widgets?",
        5,
        "The classic rate trap whose intuitive wrong answer is 100.",
        ("word_problem",),
    ),
)


# ---------------------------------------------------------------------------- coding (5)

CODING_ITEMS: tuple[CapabilityItem, ...] = (
    _code(
        "code_reverse_words",
        "Write a Python function reverse_words(text) that returns the words of text in reverse "
        "order, joined by single spaces. Runs of whitespace count as one separator.",
        "reverse_words",
        [
            [["hello world"], "world hello"],
            [["one two three"], "three two one"],
            [["  spaced   out  "], "out spaced"],
            [["single"], "single"],
            [[""], ""],
        ],
        "A pure string function, including the empty-string and extra-whitespace edge cases.",
    ),
    _code(
        "code_is_palindrome",
        "Write a Python function is_palindrome(text) that returns True if text reads the same "
        "forwards and backwards, ignoring case and ignoring any character that is not a letter "
        "or a digit, and False otherwise.",
        "is_palindrome",
        [
            [["racecar"], True],
            [["A man, a plan, a canal: Panama"], True],
            [["hello"], False],
            [[""], True],
            [["ab21ba"], False],
        ],
        "Normalisation before comparison. The last case separates an implementation that "
        "keeps digits from one that strips everything but letters, which would call it a "
        "palindrome.",
    ),
    _code(
        "code_second_largest",
        "Write a Python function second_largest(numbers) that returns the second largest "
        "distinct value in the list numbers, or None if there are fewer than two distinct "
        "values.",
        "second_largest",
        [
            [[[3, 1, 4, 1, 5]], 4],
            [[[2, 2, 2]], None],
            [[[10, 9]], 9],
            [[[]], None],
            [[[-5, -1, -3]], -3],
        ],
        "Deduplication, a None contract for the degenerate cases, and negative numbers.",
    ),
    _code(
        "code_running_total",
        "Write a Python function running_total(numbers) that returns a new list where element i "
        "is the sum of numbers[0] through numbers[i].",
        "running_total",
        [
            [[[1, 2, 3]], [1, 3, 6]],
            [[[]], []],
            [[[5]], [5]],
            [[[1, -1, 1]], [1, 0, 1]],
        ],
        "An accumulator returning a new list, with the empty input handled.",
    ),
    _code(
        "code_count_vowels",
        "Write a Python function count_vowels(text) that returns how many of the characters in "
        "text are vowels. Treat a, e, i, o and u as vowels, in either case, and do not count y.",
        "count_vowels",
        [
            [["education"], 5],
            [["rhythm"], 0],
            [["AEIOU"], 5],
            [[""], 0],
            [["Yellow"], 2],
        ],
        "A counting loop with an explicit case rule and an explicit exclusion.",
    ),
)


CAPABILITY_ITEMS: tuple[CapabilityItem, ...] = INSTRUCTION_ITEMS + MATH_ITEMS + CODING_ITEMS

ITEMS_BY_ID: dict[str, CapabilityItem] = {item.check_id: item for item in CAPABILITY_ITEMS}


def items_for(families: Iterable[str] | None = None, tags: Iterable[str] | None = None) -> list[CapabilityItem]:
    """The items in a family and/or carrying a tag, in the file's declared order."""
    wanted_families = set(families) if families is not None else None
    wanted_tags = set(tags) if tags is not None else None
    return [
        item
        for item in CAPABILITY_ITEMS
        if (wanted_families is None or item.family in wanted_families)
        and (wanted_tags is None or wanted_tags & set(item.tags))
    ]


def select(items: Iterable[CapabilityItem], limit: int | None) -> list[CapabilityItem]:
    """The first `limit` items, drawn round-robin across families.

    A smoke run with `--limit 6` should exercise all three families rather than six
    instruction-following items, so the shape of the output is the same as a full run's.
    Order within a family is preserved, and the selection is deterministic. `None` or a
    non-positive limit means every item.
    """
    ordered = list(items)
    if limit is None or limit >= len(ordered) or limit <= 0:
        return ordered
    buckets: dict[str, list[CapabilityItem]] = {}
    for item in ordered:
        buckets.setdefault(item.family, []).append(item)
    picked: list[CapabilityItem] = []
    while len(picked) < limit and any(buckets.values()):
        for family in list(buckets):
            if not buckets[family]:
                continue
            picked.append(buckets[family].pop(0))
            if len(picked) == limit:
                break
    index = {item.check_id: position for position, item in enumerate(ordered)}
    return sorted(picked, key=lambda item: index[item.check_id])


def validate_items(items: Iterable[CapabilityItem] = CAPABILITY_ITEMS) -> list[str]:
    """Structural problems with the item set. Empty list means the set is well formed."""
    ordered = list(items)
    problems: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        if item.check_id in seen:
            problems.append(f"duplicate check_id {item.check_id!r}")
        seen.add(item.check_id)
        problems += item.validate()
    for problem in problems:
        logger.warning("capability item set: %s", problem)
    return problems


__all__ = [
    "CAPABILITY_ITEMS",
    "CODE_ANSWER_INSTRUCTION",
    "CODING_ITEMS",
    "CapabilityItem",
    "FAMILIES",
    "INSTRUCTION_ITEMS",
    "ITEMS_BY_ID",
    "MATH_ANSWER_INSTRUCTION",
    "MATH_ITEMS",
    "TAGS",
    "items_for",
    "select",
    "validate_items",
]
