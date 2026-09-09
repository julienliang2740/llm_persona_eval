"""Render one Analysis as the markdown report the evaluation plan asks a run to produce.

Why this module exists separately from `aggregate.py`: the plan's real risk is not that a
number is computed wrongly but that a reader takes a number to mean more than it does. Every
rate in this suite comes from a few dozen answers, graded by a model, on prompts a sibling
model wrote. So the ordering, the wording and the caveats are the product here, and they are
worth keeping in one file where they can be read end to end and argued with.

Three rules the layout enforces:

  The report opens with what a reader must not conclude, before any table. A caveat at the
  bottom of a long document is a caveat nobody read.

  Action scores and reasoning scores get separate tables, never a shared row and never a
  combined mean. The plan's sentence "improvements in action choice must not hide worse
  reasoning" is a layout requirement, not only an arithmetic one.

  Regressions sort above improvements and are named in words. A minus sign in a table of
  twenty numbers is not prominence.

House style follows `pipeline/report.py`: short prose between tables, tables for numbers,
bold on the figure that matters, no emoji, and a section that prints a plain sentence when it
has nothing to show rather than disappearing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from persona_eval.report.aggregate import (
    GROUP_MEANING,
    MATERIAL_DID,
    MATERIAL_PREMIUM,
    SELF_CONSISTENCY,
    MEASURE_MEANING,
    MEASURE_ORDER,
    MIN_CELL_N,
    SMALL_N,
    Analysis,
    ChangeStat,
    WorstExample,
)
from persona_eval.suite.schema import (
    ACTION_DIMENSIONS,
    DIMENSION_MEANING,
    DIMENSIONS,
    SCORE_LABELS,
    TASK_DIMENSIONS,
    Suite,
)

logger = logging.getLogger("persona_eval.report.render")

REPORT_FILE = "report.md"

#: Slice kinds that count as a slice in the "N of M regressed" headline. The dimension-by-kind
#: cells are excluded: they are a matrix view of the same data at much smaller n.
HEADLINE_SLICES: tuple[str, ...] = ("group", "dimension", "family_kind", "measures")

ANSWER_EXCERPT_CHARS = 700
PROMPT_EXCERPT_CHARS = 320
QUOTE_CHARS = 240


# ---------------------------------------------------------------------------------- helpers


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _cell(text: Any, limit: int = 120) -> str:
    """Table-safe: pipes escaped, newlines flattened, length capped."""
    return _truncate(str(text), limit).replace("|", "\\|")


def _num(value: float | None, places: int = 2) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def _signed(value: float | None, places: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{places}f}"


def _rate(value: float | None, n: int) -> str:
    """A percentage that carries its own warning when the denominator is too small.

    Applied everywhere a percentage is printed. A bold 100% over two cases and a bold 100%
    over sixty look identical on a page, and the first is the one a reader will quote.
    """
    if value is None:
        return "-"
    return f"{value:.0%}" + ("" if n >= MIN_CELL_N else "†")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """`3 families` / `1 family`. Counts appear in prose here, so they have to read as English."""
    return f"{count} {singular}" if count == 1 else f"{count} {plural or singular + 's'}"


def _direction(delta: float | None) -> str:
    if delta is None:
        return "not comparable"
    if delta < -0.001:
        return "**regression**"
    if delta > 0.001:
        return "improvement"
    return "level"


def _p(value: float | None) -> str:
    if value is None:
        return "-"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


# ------------------------------------------------------------------------------ front matter


def _header(analysis: Analysis, suite: Suite, meta: Mapping[str, Any]) -> list[str]:
    title = str(meta.get("title") or f"Evaluation report: {suite.target_id or 'target'}")
    arms = ", ".join(f"`{arm}`" for arm in analysis.arms) or "no arm"
    graded = sum(stat.graded for stat in analysis.integrity)
    lines = [
        f"# {title}",
        "",
        f"Suite `{suite.suite_id or '?'}` version `{suite.version[:12]}`, "
        f"specification `{suite.spec_version or '?'}` (`{(suite.spec_hash or '')[:12]}`), "
        f"{analysis.suite_families} families and {analysis.suite_cases} cases, "
        f"answered by {arms}.",
        "",
    ]
    baseline = analysis.baseline_arm
    lines.append(
        f"{graded} graded answers in total; `{baseline}` is the baseline every comparison is "
        f"measured against."
        if baseline
        else f"{graded} graded answers in total. No baseline arm was identified, so no "
        f"comparison is reported."
    )
    lines += [
        "",
        "**What was measured.** Each answer was scored 0, 1 or 2 on a small set of dimensions "
        "chosen for its task, against anchors written into the suite before any answer existed. "
        "Separately, each variant of a situation was compared with its original to ask whether "
        "the judgment moved when the suite said it should. Nothing here is averaged into a "
        "single values score, and this report contains no such number.",
        "",
        "**What this cannot show.**",
        "",
        "- The samples are small. Rates below rest on tens of answers, and the cases inside one "
        "family are the same situation reworded, so they are not independent evidence. Family "
        "counts sit next to case counts throughout for that reason.",
        "- The grader is a language model applying a written specification. Agreement with that "
        "specification is what is measured. Whether the specification is faithful to the value "
        "system it names is a separate question this suite does not answer.",
        "- The prompts are held out of training, but they were authored by a model from the same "
        "family as the models under test, so a blind spot they share would be invisible here.",
        "- A difference between two arms is a difference between two arms. Isolating the "
        "contribution of any one part of the training curriculum needs the control arms named in "
        "the plan, which a two-arm run does not have.",
        "- Scores describe behaviour on these prompts. They are not evidence about what a model "
        "believes or about the causal path behind an answer.",
    ]
    if analysis.notes:
        lines += ["", "**Recorded during aggregation.**", ""]
        lines += [f"- {note}" for note in analysis.notes]
    return lines


def _how_to_read(analysis: Analysis) -> list[str]:
    lines = ["", "## How to read this", ""]
    lines += [
        "Every criterion score is one of three values.",
        "",
        "| score | meaning |",
        "|---|---|",
    ]
    for value in (0, 1, 2):
        lines.append(f"| {value} | {SCORE_LABELS[value]} |")
    lines += [
        "",
        "A dimension that does not apply to a task is not scored, and a dimension the judge "
        "could not score is recorded as unscorable. Neither is counted as a 0. Means are taken "
        "over scored observations only, and the unscorable count is printed beside them.",
        "",
        "Dimensions are reported in two groups that are never averaged together.",
        "",
        "| group | what it covers |",
        "|---|---|",
    ]
    for group, meaning in GROUP_MEANING.items():
        lines.append(f"| {group} | {meaning} |")
    scored = {stat.dimension for stat in analysis.dimension_stats if stat.n or stat.unscorable}
    shown = [d for d in DIMENSIONS if d in scored] or list(DIMENSIONS)
    lines += [
        "",
        "The dimensions this run scored:",
        "",
        "| dimension | group | what it distinguishes |",
        "|---|---|---|",
    ]
    for dimension in shown:
        group = next(
            (s.group for s in analysis.dimension_stats if s.dimension == dimension), "reasoning"
        )
        lines.append(f"| {dimension} | {group} | {DIMENSION_MEANING.get(dimension, '')} |")
    lines += [
        "",
        "A **family** is one base situation. Its **cases** are that situation across tasks and "
        "controlled variants: a paraphrase, an irrelevant detail changed, a relevant fact "
        "changed, the asker pushing back, the asker correcting a mistake. Ten cases from two "
        "families are two situations, not ten.",
        "",
        f"Differences are reported as `arm - baseline`, so a negative number is a regression. "
        f"Where a comparison has fewer than {SMALL_N} paired cases it is labelled small, and the "
        f"better/worse/level counts should be read instead of the mean difference.",
        "",
        f"**A percentage marked † rests on fewer than {MIN_CELL_N} observations.** It is printed "
        f"because hiding it would be worse, but it cannot support a conclusion: at that size one "
        f"case moves the figure by twenty points or more. The marker is applied to every rate in "
        f"this report, including the headline behaviour rates.",
    ]
    return lines


# -------------------------------------------------------------------------- diagnostic scores


def _dimension_coverage(analysis: Analysis, suite: Suite) -> list[str]:
    """Why a dimension is missing or thin: the suite could not reach it, did not pick it, or
    picked it and got nothing back.

    Those three have nothing in common except an absent row, and conflating them is how a
    report turns ordinary sampling into a finding. A dimension no task allows is a structural
    gap. A dimension every rubric was free to choose and none did is sampling, because each
    rubric picks three to five of the five to seven its task permits. A dimension a rubric does
    choose and no judgment comes back for is a defect in the run.

    The ceiling is computed from the suite rather than from an authoring plan, so it describes
    what was actually built and works on any suite the report is handed.
    """
    reachable = {
        dimension: sum(
            1 for case in suite.cases if dimension in TASK_DIMENSIONS.get(case.task, ())
        )
        for dimension in DIMENSIONS
    }
    selected = {
        dimension: sum(1 for case in suite.cases if dimension in case.rubric.dimensions)
        for dimension in DIMENSIONS
    }
    scored = {stat.dimension for stat in analysis.dimension_stats if stat.n}

    def status(dimension: str) -> str:
        if dimension in scored:
            return "scored"
        if not reachable[dimension]:
            return "**unreachable**"
        if not selected[dimension]:
            return "reachable, never chosen"
        return "**chosen, not returned**"

    lines = [
        "",
        "### Dimension coverage",
        "",
        "A dimension missing from the tables above has three possible causes, and they mean "
        "different things. This is which one applies.",
        "",
        "| dimension | group | cases whose task allows it | cases whose rubric scores it | status |",
        "|---|---|---|---|---|",
    ]
    for dimension in DIMENSIONS:
        group = "action" if dimension in ACTION_DIMENSIONS else "reasoning"
        lines.append(
            f"| {dimension} | {group} | {reachable[dimension]} | {selected[dimension]} "
            f"| {status(dimension)} |"
        )

    def agree(names: list[str]) -> tuple[str, str]:
        """Verb and pronoun for a list that is often one item and often several."""
        return ("is", "it") if len(names) == 1 else ("are", "them")

    unreachable = [d for d in DIMENSIONS if not reachable[d]]
    never_chosen = [d for d in DIMENSIONS if reachable[d] and not selected[d]]
    not_returned = [d for d in DIMENSIONS if selected[d] and d not in scored]
    lines.append("")
    if unreachable:
        lines.append(
            "**A structural gap.** No task in this suite permits "
            + ", ".join(unreachable)
            + ", so no answer could have been scored on it however the model replied. That is a "
            "gap in what the suite can see, not a clean record for any arm."
        )
    if never_chosen:
        _verb, pronoun = agree(never_chosen)
        lines.append(
            "**A coverage gap.** "
            + ", ".join(never_chosen)
            + f" could have been scored and no rubric selected {pronoun}. Each rubric picks "
            "three to five of the dimensions its task allows, and the coverage matrix does not "
            "currently ensure every dimension is reached, so this is unintended rather than a "
            "design choice. It is a gap to close in the suite, and no evidence either way about "
            "any arm."
        )
    if not_returned:
        verb, pronoun = agree(not_returned)
        lines.append(
            "**A defect in the run.** "
            + ", ".join(not_returned)
            + f" {verb} scored by at least one rubric and no judgment came back for {pronoun}. "
            "That is worth chasing before the numbers above are used."
        )
    # Below the interpretability floor before a single rubric has chosen anything. This is a
    # property of which task types the suite uses, not of how the rubrics were written, so
    # adding families of the same tasks will not fix it.
    thin = [d for d in DIMENSIONS if 0 < reachable[d] < MIN_CELL_N]
    if thin:
        verb, _pronoun = agree(thin)
        lines.append(
            "**Cannot reach an interpretable sample.** "
            + ", ".join(f"{d} ({_plural(reachable[d], 'case')})" for d in thin)
            + f" {verb} permitted by too few cases to clear the {MIN_CELL_N}-case floor even if "
            f"every one of those rubrics chose it. Only some task types allow these, so more "
            f"families of the same tasks will not help; more of those task types would."
        )
    return lines


def _diagnostic_scores(analysis: Analysis, suite: Suite) -> list[str]:
    lines = ["", "## Diagnostic scores", ""]
    if not analysis.dimension_stats:
        lines.append("No dimension was scored in this run.")
        # Still emitted: both read the suite rather than the results, and a reader whose
        # tables are empty is exactly the one who needs to know whether the suite could have
        # scored anything in the first place.
        lines += _dimension_coverage(analysis, suite)
        lines += _dimension_kind_matrix(analysis)
        return lines
    lines.append(
        "Action and reasoning are separate tables. An arm can recommend a defensible action "
        "for an explanation the specification does not support, and the two tables are what "
        "make that visible."
    )
    for group in ("action", "reasoning"):
        dimensions = analysis.dimensions_in(group)
        lines += ["", f"### {group.title()} dimensions", ""]
        if not dimensions:
            lines.append(f"No {group} dimension was scored in this run.")
            continue
        lines.append(f"{GROUP_MEANING[group]}")
        lines += [
            "",
            "| dimension | arm | mean | n cases | families | 0 | 1 | 2 | unscorable |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for dimension in dimensions:
            for arm in analysis.arms:
                stat = analysis.stat(arm, dimension)
                if stat is None:
                    continue
                lines.append(
                    f"| {dimension} | `{arm}` | **{_num(stat.mean)}** | {stat.n} "
                    f"| {stat.families} | {stat.counts.get(0, 0)} | {stat.counts.get(1, 0)} "
                    f"| {stat.counts.get(2, 0)} | {stat.unscorable} |"
                )
        lines += [
            "",
            f"Pooled over the {group} dimensions above, with the range the pooling hides:",
            "",
            "| arm | mean | lowest dimension | highest dimension | n observations | n cases "
            "| families | 0 | 1 | 2 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm in analysis.arms:
            stat = analysis.group_stat(arm, group)
            if stat is None:
                continue
            means = [
                (analysis.stat(arm, d).mean, d)
                for d in dimensions
                if analysis.stat(arm, d) is not None and analysis.stat(arm, d).mean is not None
            ]
            low = min(means, default=(None, "-"))
            high = max(means, default=(None, "-"))
            lines.append(
                f"| `{arm}` | {_num(stat.mean)} | {_num(low[0])} ({low[1]}) "
                f"| {_num(high[0])} ({high[1]}) | {stat.n} | {stat.cases} | {stat.families} "
                f"| {stat.counts.get(0, 0)} | {stat.counts.get(1, 0)} | {stat.counts.get(2, 0)} |"
            )
        lines.append("")
        lines.append(
            "The pooled mean is deliberately not emphasised. It weights each dimension by how "
            "often it was scored, so it moves when the task mix changes, and averaging eight "
            "dimensions into one number is the collapse this report exists to avoid. The "
            "lowest and highest columns show how far apart the dimensions it covers actually "
            "are; the per-dimension table above is the reading that means something."
        )
    lines += _dimension_coverage(analysis, suite)
    lines += _dimension_kind_matrix(analysis)
    return lines


def _cell_value(stat, delta: bool = False) -> str:
    """One matrix cell: the figure, its n, and a dagger when the cell is too thin to read."""
    if stat is None:
        return "-"
    if delta:
        if stat.paired == 0 or stat.delta is None:
            return "-"
        mark = "" if stat.paired >= MIN_CELL_N else "†"
        return f"{stat.delta:+.2f} ({stat.paired}){mark}"
    if stat.n == 0:
        return "-"
    mark = "" if stat.interpretable else "†"
    return f"{_num(stat.mean)} ({stat.n}){mark}"


def _dimension_kind_matrix(analysis: Analysis) -> list[str]:
    """Every dimension score again, split by kind of family.

    This is the breakdown that says whether a movement in the tables above is the finding or
    an average over two opposite findings.
    """
    lines = ["", "### Scores by family kind", ""]
    if not analysis.dimension_kind_stats or not analysis.kinds_present:
        lines.append("No family kind could be resolved, so no breakdown is available.")
        return lines
    kinds = list(analysis.kinds_present)
    lines += [
        "A mean over every family kind at once can hide the two results that matter most. A "
        "model that has learned to reach for one philosophy everywhere scores well on ordinary "
        "families and badly on negative controls, where the specification is not supposed to "
        "apply. A model that memorised situations rather than judgment holds up on the familiar "
        "families and falls away on far transfer. Averaged together those two look like no "
        "change at all.",
        "",
        f"Cells are the mean with the number of scored cases in brackets. A cell marked † holds "
        f"fewer than {MIN_CELL_N} scored cases and is printed for completeness, not to be read.",
    ]
    dimensions: list[str] = []
    for stat in analysis.dimension_kind_stats:
        if stat.dimension not in dimensions:
            dimensions.append(stat.dimension)

    for arm in analysis.arms:
        lines += [
            "",
            f"**`{arm}`**",
            "",
            "| dimension | group | " + " | ".join(kinds) + " |",
            "|---|---|" + "---|" * len(kinds),
        ]
        for dimension in dimensions:
            cells = [analysis.kind_stat(arm, dimension, kind) for kind in kinds]
            if all(cell is None or cell.n == 0 for cell in cells):
                continue
            group = next((c.group for c in cells if c is not None), "reasoning")
            rendered = " | ".join(_cell_value(cell) for cell in cells)
            lines.append(f"| {dimension} | {group} | {rendered} |")

    single = sorted(
        {
            stat.kind
            for stat in analysis.dimension_kind_stats
            if stat.n and stat.single_family
        }
    )
    if single:
        lines += [
            "",
            "Drawn from a single family, so every case in the column is one situation reworded: "
            + ", ".join(single)
            + ". A column like that describes one scenario, not a kind of scenario.",
        ]

    for arm in analysis.arms:
        if arm == analysis.baseline_arm:
            continue
        rows = []
        for dimension in dimensions:
            cells = [analysis.kind_change(arm, dimension, kind) for kind in kinds]
            if all(cell is None for cell in cells):
                continue
            group = next((c.group for c in cells if c is not None), "reasoning")
            rows.append(
                f"| {dimension} | {group} | "
                + " | ".join(_cell_value(cell, delta=True) for cell in cells)
                + " |"
            )
        if not rows:
            continue
        lines += [
            "",
            f"**`{arm}` minus `{analysis.baseline_arm}`, paired within each cell**",
            "",
            "| dimension | group | " + " | ".join(kinds) + " |",
            "|---|---|" + "---|" * len(kinds),
        ]
        lines += rows
        lines += [
            "",
            "Negative is a regression. Each cell pairs only the cases both arms answered inside "
            "that kind of family, so the counts in brackets are smaller than the counts above.",
        ]
    return lines


# ---------------------------------------------------------------------- family-level behaviour


def _behaviour(analysis: Analysis) -> list[str]:
    lines = ["", "## Family-level behaviour", ""]
    if not analysis.behaviour_stats:
        lines.append(
            "No change verdicts were recorded, so nothing is known about whether these arms "
            "hold a position under rewording or move under a genuine correction."
        )
    else:
        floors = [s for s in analysis.behaviour_stats if s.measures == SELF_CONSISTENCY]
        if floors:
            lines += [
                "The first row for each arm is its noise floor: the same question asked twice "
                "with nothing changed, and how often the arm gave the same answer. Every rate "
                "below it is built on top of that. An arm that reproduces its own position only "
                "seven times in ten cannot demonstrate invariance at seven in ten, because it "
                "would score that by doing nothing.",
                "",
            ]
        else:
            lines += [
                "No self-consistency probe was run, so these rates have no noise floor to be "
                "read against. An arm that answers the same question differently on two "
                "occasions would score against itself here, and nothing in this run distinguishes "
                "that from a judgment that genuinely moved.",
                "",
            ]
        lines += [
            "Each verdict compares one variant against the original of the same situation and "
            "asks whether the judgment moved. The suite committed to the expected answer before "
            "any model saw the case.",
            "",
            "| behaviour | arm | correct | n verdicts | rate | noise floor | above floor "
            "| families | variants |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        order = {name: index for index, name in enumerate(MEASURE_ORDER)}
        arm_order = {arm: index for index, arm in enumerate(analysis.arms)}
        for stat in sorted(
            analysis.behaviour_stats,
            key=lambda s: (
                arm_order.get(s.arm, len(arm_order)),
                s.arm,
                order.get(s.measures, len(order)),
                s.measures,
            ),
        ):
            variants = ", ".join(f"{k} x{v}" for k, v in sorted(stat.variants.items())) or "-"
            undecided = f" (+{stat.undecided} undecided)" if stat.undecided else ""
            floor = "-" if stat.noise_floor is None else f"{stat.noise_floor:.0%} ({stat.floor_kind})"
            above = (
                "-"
                if stat.above_floor is None
                else ("**" + f"{stat.above_floor:+.0%}" + "**" if stat.at_or_below_floor else f"{stat.above_floor:+.0%}")
            )
            lines.append(
                f"| {stat.measures} | `{stat.arm}` | {stat.correct} | {stat.n}{undecided} "
                f"| **{_rate(stat.rate, stat.n)}** | {floor} | {above} "
                f"| {stat.families} | {_cell(variants, 60)} |"
            )
        drowned = [s for s in analysis.behaviour_stats if s.at_or_below_floor]
        if drowned:
            lines += ["", ""]
            for stat in drowned:
                if stat.floor_kind == "consistency":
                    lines.append(
                        f"**`{stat.arm}` measured nothing on {stat.measures}.** Its rate of "
                        f"{_rate(stat.rate, stat.n)} is at or below its own noise floor of "
                        f"{_pct(stat.noise_floor)}, which is how often it reproduces its answer "
                        f"to an unchanged question. A result at the floor is what sampling noise "
                        f"alone produces, so this number is not evidence that the arm holds or "
                        f"fails to hold a position."
                    )
                else:
                    lines.append(
                        f"**`{stat.arm}` measured nothing on {stat.measures}.** Its rate of "
                        f"{_rate(stat.rate, stat.n)} is at or below the "
                        f"{_pct(stat.noise_floor)} it would score by changing its answer at "
                        f"random, given how inconsistent it is on unchanged questions. Moving "
                        f"when the facts move is only evidence if the arm stays put when they "
                        f"do not."
                    )
        lines += ["", "What each behaviour means:", ""]
        seen = {stat.measures for stat in analysis.behaviour_stats}
        for measure in list(MEASURE_ORDER) + sorted(seen - set(MEASURE_ORDER)):
            if measure in seen:
                lines.append(f"- **{measure}**: {MEASURE_MEANING.get(measure, 'not defined.')}")
        lines.append("")
        lines.append(
            "A verdict recorded as undecided is one the judge could not call. It is outside the "
            "rate rather than counted against the arm. The floor column names which of the two "
            "baselines applies: `consistency` where the expected answer is that the position "
            "held, and `chance` where it is that the position moved, since an arm that moves at "
            "random scores for free on those."
        )
        wrong = [
            (stat, example)
            for stat in analysis.behaviour_stats
            for example in stat.wrong_examples
        ]
        if wrong:
            lines += ["", "Verdicts that went the wrong way, with the judge's evidence:", ""]
            for stat, (case_id, evidence) in wrong[:8]:
                lines.append(
                    f"- `{stat.arm}` / {stat.measures} / `{case_id}`: "
                    f"{_truncate(evidence, 220) or 'no evidence recorded'}"
                )

    if analysis.family_kind_stats:
        lines += [
            "",
            "### Variant behaviour by family kind",
            "",
            "Far transfer, negative controls and principle conflicts are kinds of family rather "
            "than kinds of variant, so they are reported here. The dimension scores behind these "
            "two group means are broken out cell by cell under Diagnostic scores above.",
            "",
            "| kind | arm | families | cases | action mean (n) | reasoning mean (n) | "
            "variant verdicts correct |",
            "|---|---|---|---|---|---|---|",
        ]
        for stat in analysis.family_kind_stats:
            verdicts = (
                f"{stat.verdicts_correct}/{stat.verdicts} "
                f"({_rate(stat.verdict_rate, stat.verdicts)})"
                if stat.verdicts
                else "-"
            )
            lines.append(
                f"| {stat.kind} | `{stat.arm}` | {stat.families} | {stat.cases} "
                f"| {_num(stat.action_mean)} ({stat.action_n}) "
                f"| {_num(stat.reasoning_mean)} ({stat.reasoning_n}) | {verdicts} |"
            )
    return lines


# --------------------------------------------------------------------------- overapplication


def _overapplication(analysis: Analysis) -> list[str]:
    lines = ["", "## Overapplication rate", ""]
    if not analysis.flag_stats:
        lines.append("No graded case, so no false-positive rate can be computed.")
        return lines
    lines += [
        "Overapplication is the specification being brought to bear where the case does not "
        "support it. It is the failure a values score cannot see, because an answer full of "
        "unsupported principles can still recommend a reasonable action.",
        "",
        "The denominator is graded cases: an answer exists, it was not a technical failure, and "
        "the judge did not declare it unscorable. A flag cannot be observed on a case nobody "
        "graded, so those are excluded here and counted in the reliability section instead.",
        "",
        "| scope | arm | eligible cases | families | with an explicit must-not-infer rule "
        "| overapplied | rate | unacceptable reasoning | missed a must-notice fact |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    scopes: list[str] = []
    for stat in analysis.flag_stats:
        if stat.scope not in scopes:
            scopes.append(stat.scope)
    for scope in scopes:
        for stat in analysis.flag_stats:
            if stat.scope != scope:
                continue
            lines.append(
                f"| {scope} | `{stat.arm}` | {stat.eligible_cases} | {stat.eligible_families} "
                f"| {stat.with_explicit_must_not_infer} "
                f"| {stat.overapplied_cases} ({stat.overapplied_families} fam) "
                f"| **{_rate(stat.overapplied_rate, stat.eligible_cases)}** "
                f"| {stat.unacceptable_reasoning_cases} "
                f"({_rate(stat.unacceptable_reasoning_rate, stat.eligible_cases)}) "
                f"| {stat.missed_must_notice_cases} "
                f"({_rate(stat.missed_must_notice_rate, stat.eligible_cases)}) |"
            )
    negative = [s for s in analysis.flag_stats if s.scope == "negative_control"]
    lines.append("")
    if negative and any(stat.eligible_cases for stat in negative):
        lines.append(
            "Negative controls are the row to read first. Those families were written so that "
            "the target principles add little or do not apply, so anything flagged there is a "
            "false positive with no ambiguity about whether the principle belonged."
        )
    else:
        lines.append(
            "This run graded no negative-control case. Without them an overapplication rate has "
            "no clean false-positive baseline, because on an ordinary family a judge and an "
            "author can disagree about whether a principle belonged."
        )
    examples = [
        (stat, item)
        for stat in analysis.flag_stats
        if stat.scope == "all"
        for item in stat.top_overapplied
    ]
    if examples:
        lines += ["", "The principles most often applied where the case did not support them:", ""]
        for stat, (text, count) in examples[:10]:
            lines.append(f"- `{stat.arm}`: {_truncate(text, 160)} (x{count})")
    missed = [
        (stat, item)
        for stat in analysis.flag_stats
        if stat.scope == "all"
        for item in stat.top_missed
    ]
    if missed:
        lines += ["", "The facts most often missed that the rubric said must be noticed:", ""]
        for stat, (text, count) in missed[:10]:
            lines.append(f"- `{stat.arm}`: {_truncate(text, 160)} (x{count})")
    return lines


# -------------------------------------------------------------------------- change from baseline


def _change_rows(lines: list[str], changes: Sequence[ChangeStat], label: str) -> None:
    lines += [
        "",
        f"### {label}",
        "",
        "| slice | group | baseline | arm | difference | direction | better | worse | level "
        "| paired cases | families | sign test |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    # Regressions first, largest first, so the reader meets the bad news at the top.
    ordered = sorted(changes, key=lambda c: (c.delta if c.delta is not None else 0.0, c.slice_name))
    for change in ordered:
        note = "small" if change.small_sample else ""
        lines.append(
            f"| {change.slice_name} | {change.group} | {_num(change.baseline_mean)} "
            f"| {_num(change.arm_mean)} | **{_signed(change.delta)}** | {_direction(change.delta)} "
            f"| {change.better} | {change.worse} | {change.level} | {change.paired}{' (' + note + ')' if note else ''} "
            f"| {change.families} | {_p(change.p_value)} |"
        )


def _premium_note(analysis: Analysis, rows: Sequence[ChangeStat]) -> list[str]:
    """Print the format premium next to the group comparison it could explain."""
    material = [premium for premium in analysis.format_premium if premium.material]
    if not material:
        return []
    lines = [""]
    for premium in material:
        gain = next((row for row in rows if row.slice_name == premium.group), None)
        if gain is None or gain.delta is None:
            continue
        if gain.delta <= 0:
            continue
        share = premium.premium / gain.delta if gain.delta else None
        verdict = (
            "the format explains the whole of it"
            if share is not None and share >= 1.0
            else f"the format explains about {share:.0%} of it"
            if share is not None
            else ""
        )
        lines.append(
            f"**The {premium.group} gain of {_signed(gain.delta)} sits against a format premium "
            f"of {_signed(premium.premium)}**, measured by rewriting `{premium.source_arm}` "
            f"answers into the adapted shape with the substance unchanged. On these numbers "
            f"{verdict}. See the format premium section."
        )
    return lines if len(lines) > 1 else []


def _change_from_baseline(analysis: Analysis) -> list[str]:
    lines = ["", "## Change from baseline", ""]
    if not analysis.changes or not analysis.baseline_arm:
        lines.append(
            "Only one arm produced graded answers, so there is no comparison to make. Absolute "
            "scores above are not a result on their own."
        )
        return lines

    for arm in analysis.arms:
        if arm == analysis.baseline_arm:
            continue
        # The dimension-by-kind cells are rendered as a matrix under Diagnostic scores. Counting
        # them here would put "20 of 51 slices regressed" on a page whose slices are supposed to
        # be whole dimensions and whole family kinds, and would let a one-case cell win.
        subset = [c for c in analysis.changes if c.arm == arm and c.slice_kind in HEADLINE_SLICES]
        if not subset:
            continue
        lines += [
            "",
            f"### `{arm}` against `{analysis.baseline_arm}`",
            "",
            "Every comparison is paired: only the cases both arms answered and both arms were "
            "scored on are counted. The difference is `arm - baseline`, so a negative number is "
            "a regression.",
        ]
        imbalance = analysis.graded_imbalance
        if imbalance is not None:
            fewest, most, gap = imbalance
            lines += [
                "",
                f"**These pairs are not a random sample of the suite.** `{fewest}` was graded on "
                f"{_plural(gap, 'case')} fewer than `{most}`, and every figure below drops the "
                f"cases only one arm managed. Where those losses were truncations they fall on "
                f"the arm's longest answers, so the surviving pairs favour it. The integrity "
                f"section lists which cases went and why.",
            ]
        regressions = [c for c in subset if c.delta is not None and c.delta < -0.001]
        improvements = [c for c in subset if c.delta is not None and c.delta > 0.001]
        if regressions:
            # Rank among slices that clear the interpretability floor when there are any. A
            # two-point swing over one case is the largest number on the page and the least
            # informative thing on it, and naming it as the headline finding would be wrong.
            solid = [c for c in regressions if c.paired >= MIN_CELL_N]
            worst = min(solid or regressions, key=lambda c: c.delta or 0.0)
            lines += [
                "",
                f"**{len(regressions)} of {len(subset)} slices regressed.** The largest"
                + ("" if solid else " (and no regressed slice clears the interpretability floor)")
                + f" is {worst.slice_name} at {_signed(worst.delta)} over "
                f"{_plural(worst.paired, 'paired case')} from "
                f"{_plural(worst.families, 'family', 'families')}, with {worst.worse} worse "
                f"against {worst.better} better.",
            ]
            if solid and len(solid) < len(regressions):
                biggest = min(regressions, key=lambda c: c.delta or 0.0)
                if biggest is not worst:
                    lines.append("")
                    lines.append(
                        f"A larger number appears on {biggest.slice_name} at "
                        f"{_signed(biggest.delta)}, but it rests on "
                        f"{_plural(biggest.paired, 'paired case')} and is not quotable."
                    )
        else:
            lines += ["", f"No slice regressed. {len(improvements)} of {len(subset)} improved."]

        for kind, label in (
            ("group", "Action and reasoning"),
            ("dimension", "By dimension"),
            ("family_kind", "By family kind"),
            ("measures", "By variant behaviour"),
        ):
            rows = [c for c in subset if c.slice_kind == kind]
            if rows:
                _change_rows(lines, rows, label)
            if kind == "group":
                lines += _premium_note(analysis, rows)
        lines += [
            "",
            "The sign test is exact and two-sided over the discordant pairs only. It has very "
            "little power at this size: five cases better and none worse cannot go below "
            "p = 0.06 however clean the direction looks. It also assumes the pairs are "
            "independent, and cases from one family are not, so read it as an optimistic bound "
            "and read the better/worse/level counts as the result.",
            "",
            "A rise in the action table with a fall in the reasoning table is not an "
            "improvement. It is the model reaching an acceptable answer for reasons the "
            "specification does not support, which is the failure this report is arranged to "
            "surface.",
        ]
    return lines


# ---------------------------------------------------------------------------- worst examples


def _worst(analysis: Analysis) -> list[str]:
    lines = ["", "## Worst examples", ""]
    if not analysis.zero_patterns and not analysis.worst_examples:
        lines.append("Nothing was scored, so there is no failure to show.")
        return lines

    lines += [
        "Where each arm scored 0, counted first so the pattern is visible before the anecdote.",
        "",
        "| arm | dimension | group | zeros | scored | rate | families |",
        "|---|---|---|---|---|---|---|",
    ]
    shown = [pattern for pattern in analysis.zero_patterns if pattern.zeros]
    for pattern in shown[:24]:
        lines.append(
            f"| `{pattern.arm}` | {pattern.dimension} | {pattern.group} | **{pattern.zeros}** "
            f"| {pattern.scored} | {_rate(pattern.rate, pattern.scored)} | {pattern.families} |"
        )
    if not shown:
        lines.append("| - | no dimension scored 0 | - | 0 | - | - | - |")

    if not analysis.worst_examples:
        lines += ["", "No case scored 0 on any dimension."]
        return lines

    lines += [
        "",
        f"The cases behind those counts, ranked by the dimension that failed most often. "
        f"{len(analysis.worst_examples)} shown"
        + (f", {analysis.worst_omitted} omitted." if analysis.worst_omitted else "."),
    ]
    # Grouped by the dimension that failed, in the order the ranking first reaches each one.
    # Grouping here rather than trusting the sort keeps a dimension from getting two headings
    # when two arms fail it at different rates.
    grouped: dict[str, list[WorstExample]] = {}
    for example in analysis.worst_examples:
        grouped.setdefault(example.lead_dimension, []).append(example)
    for dimension, group in grouped.items():
        lines += ["", f"### Failures of {dimension}", ""]
        for example in group:
            lines += _worst_example(example)
    return lines


def _worst_example(example: WorstExample) -> list[str]:
    """One failing case: what was asked, what was answered, and what the judge pointed at."""
    lines = [
        f"**`{example.arm}` on `{example.case_id}`** — "
        f"{example.family_title or example.family_id} "
        f"({example.family_kind} family, {example.task} task, {example.variant} variant); "
        f"scored 0 on {', '.join(example.zero_dimensions) or 'a dimension'}.",
        "",
    ]
    if example.also_failed_by:
        lines += [
            "Also scored 0 on this case by: "
            + ", ".join(f"`{arm}`" for arm in example.also_failed_by)
            + ". A case both arms fail is a property of the case or the rubric as much as "
            "of the model.",
            "",
        ]
    if example.other_zeros_in_family:
        lines += [
            f"{_plural(example.other_zeros_in_family, 'other case')} in the same family also "
            f"scored 0 for this arm, so this is not an isolated slip.",
            "",
        ]
    if example.prompt_excerpt:
        lines += [f"*Asked:* {_truncate(example.prompt_excerpt, PROMPT_EXCERPT_CHARS)}", ""]
    lines += [f"*Answered:* {_truncate(example.answer_excerpt, ANSWER_EXCERPT_CHARS)}", ""]
    for name, quote in example.quotes:
        lines.append(f"*Judge quoted for {name}:* {_truncate(quote, QUOTE_CHARS)}")
    for name, note in example.notes:
        lines.append(f"*Judge note on {name}:* {_truncate(note, QUOTE_CHARS)}")
    if example.judge_rationale:
        lines.append(f"*Judge rationale:* {_truncate(example.judge_rationale, 400)}")
    if example.missed_must_notice:
        lines.append("*Missed:* " + _truncate("; ".join(example.missed_must_notice), 300))
    if example.overapplied:
        lines.append("*Overapplied:* " + _truncate("; ".join(example.overapplied), 300))
    lines.append("")
    return lines


# ------------------------------------------------------------ reliability and unresolved counts


def _verdict_stability(analysis: Analysis) -> list[str]:
    """Whether a re-judged change verdict came back the same way.

    Each behaviour rate is a pile of single binary judgments. Without re-judging some of them,
    a 60% invariance rate and a coin flip look the same on the page.
    """
    if not analysis.verdict_stability:
        if analysis.behaviour_stats:
            return [
                "",
                "No change verdict was re-judged, so the behaviour rates above carry no evidence "
                "of their own stability. Each rests on one binary judgment per probe.",
            ]
        return []
    lines = [
        "",
        "Change verdicts re-judged. Each behaviour rate is built from single binary judgments, "
        "so this is what says whether those judgments are repeatable at all.",
        "",
        "| arm | scope | probes re-judged | same verdict | agreement | families |",
        "|---|---|---|---|---|---|",
    ]
    for stat in analysis.verdict_stability:
        lines.append(
            f"| `{stat.arm}` | {stat.scope} | {stat.pairs} | {stat.agree} "
            f"| **{_rate(stat.rate, stat.pairs)}** | {stat.families} |"
        )
    flipped = [(stat, item) for stat in analysis.verdict_stability for item in stat.flipped]
    if flipped:
        lines += ["", "Probes whose verdict flipped between passes:", ""]
        for stat, (case_id, movement) in flipped[:8]:
            lines.append(f"- `{stat.arm}` ({stat.scope}) `{case_id}`: did_change {movement}")
        lines += [
            "",
            "A probe that flips is one whose contribution to a behaviour rate is arbitrary. If "
            "the flip rate is comparable to the gap between two arms' rates, the gap is noise.",
        ]
    return lines


def _truncation_warning(analysis: Analysis) -> list[str]:
    """Truncation named as a bias, not just a count.

    A truncated answer becomes a technical failure and leaves every denominator. That is the
    right treatment for a network error, which strikes at random. It is the wrong treatment for
    a token ceiling, which strikes the arm that writes longest, on the cases where it rambles,
    which are disproportionately the cases it was answering worst.
    """
    truncating = [stat for stat in analysis.integrity if stat.truncated]
    if not truncating:
        return []
    worst = max(truncating, key=lambda stat: stat.truncated)
    lines = [
        "",
        "**Some answers hit the token ceiling.** "
        + ", ".join(
            f"`{stat.arm}` truncated {_plural(stat.truncated, 'answer')}"
            + (f", {stat.truncated_and_lost} of which left the denominators" if stat.truncated_and_lost else "")
            for stat in truncating
        )
        + ".",
    ]
    if any(stat.truncated_and_lost for stat in truncating):
        lines.append("")
        lines.append(
            f"Truncation does not remove cases at random. The arm that writes longest hits the "
            f"ceiling most, and it hits it on the cases where it rambles, which tend to be the "
            f"cases it was handling worst. Dropping those raises that arm's mean and shortens "
            f"its denominator at the same time. `{worst.arm}` lost the most this way, so read "
            f"its scores as an upper bound."
        )
    if truncating and worst.truncated_cases:
        lines.append("")
        lines.append(
            "Truncated cases include: "
            + ", ".join(f"`{case_id}`" for case_id in worst.truncated_cases[:6])
            + "."
        )
    return lines


def _imbalance_warning(analysis: Analysis) -> list[str]:
    """State plainly when the arms did not get graded on the same number of cases."""
    imbalance = analysis.graded_imbalance
    if imbalance is None:
        if len(analysis.integrity) > 1:
            return [
                "",
                "Both arms were graded on the same number of cases, so no paired comparison in "
                "this report is drawing on a subset chosen by which arm failed.",
            ]
        return []
    fewest, most, gap = imbalance
    lines = [
        "",
        f"**The arms were not graded on the same cases.** `{fewest}` was graded on "
        f"{_plural(gap, 'case')} fewer than `{most}`. Every paired comparison in this report "
        f"drops the cases only one arm managed, so those comparisons are between the two arms "
        f"on the subset where `{fewest}` succeeded. If it failed on its hardest cases, that "
        f"subset flatters it.",
    ]
    for loss in analysis.pairing_losses:
        if loss.balanced:
            continue
        lines.append("")
        lines.append(
            f"`{loss.arm}` against `{loss.baseline_arm}`: {loss.both} cases paired, "
            f"{loss.baseline_only} scored only for `{loss.baseline_arm}`, {loss.arm_only} only "
            f"for `{loss.arm}`."
        )
        if loss.baseline_only_cases:
            lines.append(
                f"Dropped because `{loss.arm}` had no score: "
                + ", ".join(f"`{case_id}`" for case_id in loss.baseline_only_cases[:6])
                + "."
            )
        if loss.arm_only_cases:
            lines.append(
                f"Dropped because `{loss.baseline_arm}` had no score: "
                + ", ".join(f"`{case_id}`" for case_id in loss.arm_only_cases[:6])
                + "."
            )
    return lines


def _reliability(analysis: Analysis) -> list[str]:
    lines = ["", "## Grading reliability and unresolved counts", ""]
    if not analysis.reliability:
        lines.append(
            "No case was judged a second time, so this run measures nothing about the judge's "
            "consistency. Every score above is one model's single reading of one answer, and a "
            "difference between arms could be judge noise."
        )
    else:
        lines += [
            "A sample of answers was judged twice. Agreement is measured over the dimension "
            "scores both passes produced.",
            "",
            "| arm | scope | cases re-judged | families | dimension pairs | exact | within 1 "
            "| mean absolute difference | scorability disagreements |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for stat in analysis.reliability:
            exact = (
                f"{stat.exact}/{stat.pairs} ({_rate(stat.exact / stat.pairs, stat.pairs)})"
                if stat.pairs
                else "-"
            )
            within = (
                f"{stat.within_one}/{stat.pairs} ({_rate(stat.within_one / stat.pairs, stat.pairs)})"
                if stat.pairs
                else "-"
            )
            lines.append(
                f"| `{stat.arm}` | {stat.scope} | {stat.cases} | {stat.families} | {stat.pairs} "
                f"| **{exact}** | {within} | {_num(stat.mean_abs_diff)} "
                f"| {stat.scorability_disagreements} |"
            )
        lines += [
            "",
            "`same judge` is the primary judge looking at the same answer twice, which measures "
            "its consistency with itself. `second judge` is a different model on the same "
            "answer, which is the only one of the two that can show whether the reading of the "
            "specification is shared rather than idiosyncratic.",
        ]
        lines += [
            "",
            "A scorability disagreement is one pass scoring a dimension the other declared "
            "unscorable. It matters more than a one-point gap, because it means the two passes "
            "disagreed about whether the case could be graded at all.",
        ]
        agreement = [s for s in analysis.reliability if s.overapplication_pairs]
        if agreement:
            lines += ["", "Agreement on the overapplication flag, which the section above rests on:", ""]
            for stat in agreement:
                rate = stat.overapplication_agree / stat.overapplication_pairs
                lines.append(
                    f"- `{stat.arm}` ({stat.scope}): "
                    f"**{stat.overapplication_agree}/{stat.overapplication_pairs}** "
                    f"({_rate(rate, stat.overapplication_pairs)}) of re-judged cases got the "
                    f"same flag."
                )
        worst = [(stat, item) for stat in analysis.reliability for item in stat.disagreements]
        if worst:
            lines += ["", "The largest disagreements between passes:", ""]
            for stat, (case_id, dimension, first, repeat) in worst[:8]:
                lines.append(
                    f"- `{stat.arm}` / `{case_id}` / {dimension}: first pass {first}, "
                    f"second pass {repeat}."
                )
        models = [
            f"`{stat.arm}` ({stat.scope}) judged by "
            + (", ".join(stat.first_pass_models) or "an unrecorded model")
            + " then "
            + (", ".join(stat.repeat_pass_models) or "an unrecorded model")
            for stat in analysis.reliability
        ]
        lines += ["", "Judges: " + "; ".join(models) + "."]
        if any(
            set(stat.first_pass_models) == set(stat.repeat_pass_models)
            for stat in analysis.reliability
        ):
            lines.append("")
            lines.append(
                "Where both passes used the same model, this measures that model's consistency "
                "with itself and not whether its reading of the specification is right. A second "
                "model, or a human, is the check that would show that."
            )

    # Independent of the dimension-score reliability above: a run can re-judge change verdicts
    # without re-judging any scores, and the behaviour rates need this either way.
    lines += _verdict_stability(analysis)

    lines += ["", "### What could not be graded", ""]
    if not analysis.integrity:
        lines.append("No results were recorded.")
        return lines
    lines += [
        "Difficult cases are counted here rather than dropped. A technical failure is a run "
        "problem and is never a philosophical failure; an unscorable case is one the rubric "
        "itself said could not be judged.",
        "",
        "| arm | results | graded | unscorable | technical failures | truncated "
        "| truncated and lost | empty answers | graded but no score recorded |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for stat in analysis.integrity:
        lines.append(
            f"| `{stat.arm}` | {stat.results} | {stat.graded} | **{stat.unscorable}** "
            f"| **{stat.technical_failures}** | {stat.truncated} | **{stat.truncated_and_lost}** "
            f"| {stat.empty_answers} | {stat.no_scores_recorded} |"
        )
    lines += _truncation_warning(analysis)
    lines += _imbalance_warning(analysis)
    losses = [stat for stat in analysis.integrity if stat.judging_failures or stat.inapplicable_dimensions]
    if losses:
        lines += [
            "",
            "Individual dimension scores that never arrived, split by cause. A dimension the "
            "task cannot express is correctly not scored; a dimension the grading lost is a "
            "hole in the measurement, and only the second should worry a reader.",
            "",
            "| arm | lost to a grading problem | correctly inapplicable | checks that could not run |",
            "|---|---|---|---|",
        ]
        for stat in losses:
            lines.append(
                f"| `{stat.arm}` | **{stat.judging_failures}** | {stat.inapplicable_dimensions} "
                f"| {stat.deterministic_errors} |"
            )
        lost = sum(stat.judging_failures for stat in analysis.integrity)
        if lost:
            lines += [
                "",
                f"{_plural(lost, 'dimension score')} were lost to a grading problem: an "
                f"unverifiable quote, a missing score, or a score outside 0/1/2. Those are "
                f"absent from every mean above without shrinking any case count, so a run can "
                f"lose a large share of its individual scores while every case still looks "
                f"graded.",
            ]
    reasons = [
        (stat.arm, "unscorable", text, count)
        for stat in analysis.integrity
        for text, count in stat.unscorable_reasons
    ] + [
        (stat.arm, "technical", text, count)
        for stat in analysis.integrity
        for text, count in stat.technical_reasons
    ]
    if reasons:
        lines += ["", "Reasons recorded:", ""]
        for arm, kind, text, count in reasons[:12]:
            lines.append(f"- `{arm}` ({kind}): {_truncate(text, 160)} (x{count})")
    unscorable_ids = [
        (stat.arm, case_id) for stat in analysis.integrity for case_id in stat.unscorable_cases
    ]
    if unscorable_ids:
        lines += [
            "",
            "Unscorable case ids: "
            + ", ".join(f"`{arm}`/`{case_id}`" for arm, case_id in unscorable_ids[:12])
            + ".",
        ]
    if analysis.deterministic:
        lines += [
            "",
            "### Objective checks on the answers",
            "",
            "Verifiable constraints checked programmatically, with no judge involved.",
            "",
            "| arm | check | passed | n | rate | could not run | families |",
            "|---|---|---|---|---|---|---|",
        ]
        for stat in analysis.deterministic:
            lines.append(
                f"| `{stat.arm}` | {stat.check} | {stat.passed} | {stat.n} "
                f"| **{_rate(stat.rate, stat.n)}** | {stat.errors} | {stat.families} |"
            )
        errored = [stat for stat in analysis.deterministic if stat.errors]
        if errored:
            lines += [
                "",
                "A check in the could-not-run column is an unknown check kind or a verifier that "
                "raised. That is a defect in the suite or the checker, so it is outside the pass "
                "rate rather than counted as a failure by the arm: "
                + ", ".join(f"`{stat.check}` x{stat.errors}" for stat in errored)
                + ".",
            ]
    return lines


def _format_premium_section(analysis: Analysis) -> list[str]:
    """What the answer shape alone is worth, with the substance held constant."""
    lines = ["", "## Format premium", ""]
    if not analysis.format_premium:
        lines.append(
            "No form control was run. Nothing here separates a gain in judgment from a gain in "
            "presentation, so any improvement reported above may be partly or wholly the shape "
            "of the answers rather than their content."
        )
        return lines
    sample = analysis.format_premium[0]
    lines += [
        f"`{sample.source_arm}` answers were rewritten into the adapted model's deliberative "
        f"scaffold with the substance held constant, then judged blind against the same rubric "
        f"as everything else. The gap between the original and the recast version is what the "
        f"shape alone is worth.",
        "",
        "| group | source mean | recast mean | premium | better | worse | level | cases "
        "| families | sign test |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for premium in analysis.format_premium:
        lines.append(
            f"| {premium.group} | {_num(premium.source_mean)} | {_num(premium.recast_mean)} "
            f"| **{_signed(premium.premium)}** | {premium.better} | {premium.worse} "
            f"| {premium.level} | {premium.cases}{' (small)' if premium.small_sample else ''} "
            f"| {premium.families} | {_p(premium.p_value)} |"
        )
    material = [premium for premium in analysis.format_premium if premium.material]
    lines.append("")
    if material:
        detail = "; ".join(
            f"{_signed(premium.premium)} on {premium.group}" for premium in material
        )
        lines.append(
            f"**The shape alone is worth {detail}.** The same substance, rewritten into the "
            f"adapted model's format, scores that much higher from a judge that could not see "
            f"which was which. Any gain the adapted arm shows on those groups that is no larger "
            f"than this premium is a gain the format explains, not a change in judgment."
        )
    else:
        lines.append(
            f"**No material format premium.** Recasting the substance into the adapted shape "
            f"moved the score by less than {MATERIAL_PREMIUM:.2f} on the 0-2 scale, so the "
            f"differences reported above are not obviously explained by presentation. The "
            f"sample is {_plural(sample.cases, 'case')}, small enough that a premium of this "
            f"size could still be missed."
        )
    lines += [
        "",
        "This is the control for the failure the judging setup is most exposed to. The adapted "
        "model was trained to produce a particular deliberative shape, and a judge reading for "
        "substance can still be moved by a well-organised answer. Holding the substance fixed "
        "and varying only the shape is what separates the two.",
    ]
    return lines


# ------------------------------------------------------------------- judges against each other


def _judge_disagreement(analysis: Analysis) -> list[str]:
    """Whether the two judges disagree in one arm's favour, stated in words before numbers."""
    lines = ["", "## Judge disagreement", ""]
    if analysis.judge_coverage:
        lines += [
            "Which judge scored what:",
            "",
            "| judge | answers scored | families | first pass | repeat pass |",
            "|---|---|---|---|---|",
        ]
        for coverage in analysis.judge_coverage:
            lines.append(
                f"| `{coverage.judge}` | {coverage.cases} | {coverage.families} "
                f"| {coverage.primary_pass} | {coverage.repeat_pass} |"
            )
        lines.append("")

    if not analysis.judge_divergence:
        lines.append(
            "Only one judge scored this run, or no answer was scored by two judges on both arms. "
            "Nothing here can say whether the result depends on who did the grading."
        )
        return lines

    declared = analysis.judge_divergence[0].curator_declared
    curator = analysis.judge_divergence[0].curator_judge
    others = sorted({d.independent_judge for d in analysis.judge_divergence})
    other_names = ", ".join(f"`{name}`" for name in others)

    if declared:
        lines += [
            f"Two judges scored the same answers. `{curator}` reviewed or filtered the training "
            f"data, so the model under test was optimised toward its taste. {other_names} never "
            f"saw that data. If the first scores the fine-tune higher than the second does, part "
            f"of the improvement is a preference the fine-tune was trained to satisfy rather "
            f"than a change a disinterested reader would recognise.",
            "",
            "The figure that isolates that is the difference of differences:",
            "",
            f"    (`{curator}`: arm minus baseline) minus (independent judge: arm minus baseline)",
            "",
            "Both judges saw the same four answers per case, so anything the answers themselves "
            "explain cancels out. What is left is the two judges disagreeing about how much "
            "better the arm got. **A positive number means the interested judge sees a gain the "
            "independent judge does not.**",
        ]
    else:
        lines += [
            "More than one judge scored the same answers, but no curator judge was declared for "
            "this run. The report can say how far apart the judges are and not which of them "
            "has an interest in the outcome. Each difference of differences below is oriented "
            "as (first judge named: arm minus baseline) minus (second: arm minus baseline). "
            "Read its size, not its sign.",
        ]

    grouped: dict[tuple[str, str, str], list] = {}
    for divergence in analysis.judge_divergence:
        key = (divergence.curator_judge, divergence.independent_judge, divergence.arm)
        grouped.setdefault(key, []).append(divergence)

    for (curator_model, independent_model, arm), items in grouped.items():
        lines += [
            "",
            f"### `{curator_model}` against `{independent_model}`, on `{arm}`",
            "",
        ]
        groups = [d for d in items if d.slice_kind == "group"]
        if declared and groups:
            lines.append(_judge_verdict(groups, curator_model, independent_model, arm))
            lines.append("")
        lines += [
            "Columns are the mean under each judge on each arm, then each judge's own "
            "arm-minus-baseline delta, then the difference between those two deltas.",
            "",
            f"| slice | group | curator base | curator arm | indep base | indep arm "
            f"| curator delta | indep delta | difference | favours `{arm}` "
            f"| favours `{analysis.baseline_arm}` | level | cases | families | sign test |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for divergence in sorted(
            items,
            key=lambda d: (
                0 if d.slice_kind == "group" else 1,
                -(d.difference_of_differences or 0.0),
                d.slice_name,
            ),
        ):
            lines.append(
                f"| {divergence.slice_name} | {divergence.group} "
                f"| {_num(divergence.curator_baseline)} | {_num(divergence.curator_arm)} "
                f"| {_num(divergence.independent_baseline)} | {_num(divergence.independent_arm)} "
                f"| {_signed(divergence.curator_delta)} | {_signed(divergence.independent_delta)} "
                f"| **{_signed(divergence.difference_of_differences)}** "
                f"| {divergence.favours_arm} | {divergence.favours_baseline} "
                f"| {divergence.level} | {divergence.cases} | {divergence.families} "
                f"| {_p(divergence.p_value)} |"
            )

    lines += [
        "",
        "Only cases where both judges scored both arms are counted, so this sample is the "
        "intersection of two samples and is smaller than any other table in the report. The "
        "sign test counts the cases where the two judges disagreed about the direction, and "
        "carries the same small-sample warning as everywhere else.",
        "",
        "A judge can be involved in the training data and still be right about an answer. This "
        "section does not show that the curator judge is wrong. It puts a size on how much of "
        "the reported gain only one of the two judges can see.",
    ]
    return lines


def _judge_verdict(groups, curator: str, independent: str, arm: str) -> str:
    """The plain-language sentence a reader should take away, before they read the table."""
    material = [d for d in groups if d.material]
    worst = max(groups, key=lambda d: d.difference_of_differences or 0.0)
    cases = _plural(worst.cases, "case")
    if material:
        detail = "; ".join(
            f"{d.slice_name} by {_signed(d.difference_of_differences)}" for d in material
        )
        groups = ", ".join(f"{d.slice_name}" for d in material)
        return (
            f"**`{curator}` is flattering `{arm}`.** It rates that arm above `{independent}` "
            f"does on {detail}, on the 0-2 scale, over {cases} both judges scored on both arms. "
            f"Subtract that much from this arm's {groups} comparison in the change section: "
            f"that part of the {groups} gain is visible only to the judge the training was "
            f"tuned toward. Other groups are unaffected and should not be discounted."
        )
    lowest = min(groups, key=lambda d: d.difference_of_differences or 0.0)
    if (lowest.difference_of_differences or 0.0) <= -MATERIAL_DID:
        return (
            f"**`{curator}` is harder on `{arm}` than `{independent}` is**, by "
            f"{_signed(lowest.difference_of_differences)} on {lowest.slice_name}. Whatever else "
            f"is happening, the gain reported above is not an artefact of the interested judge."
        )
    return (
        f"**No material flattery detected.** The two judges differ by less than "
        f"{MATERIAL_DID:.2f} on the 0-2 scale about how much `{arm}` improved, so the headline "
        f"numbers do not need discounting on this account. The sample is {cases}, small enough "
        f"that a real effect of this size could still be missed."
    )


# ------------------------------------------------------------------------------- bias audits


def _bias(analysis: Analysis) -> list[str]:
    lines = ["", "## Judge bias audits", ""]
    lines.append(
        "The plan requires that grading not systematically favour a longer answer or an answer "
        "shown first. Both are known failure modes of model judges, so both are measured here "
        "rather than assumed away."
    )

    lines += ["", "### Verbosity", ""]
    usable = [audit for audit in analysis.verbosity if audit.n]
    if not usable:
        lines.append("No arm had enough scored answers to correlate length against score.")
    else:
        source = usable[0].length_source
        lines += [
            f"Answer length is measured in {source}. The correlation is Spearman over "
            f"tie-averaged ranks, chosen because lengths are skewed and case means take few "
            f"distinct values; Pearson is shown next to it for comparison.",
            "",
            "| arm | group | n cases | mean length | median | Spearman | Pearson "
            "| shortest third | longest third |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for audit in analysis.verbosity:
            lines.append(
                f"| `{audit.arm}` | {audit.group} | {audit.n} | {_num(audit.mean_length, 0)} "
                f"| {_num(audit.median_length, 0)} | **{_num(audit.spearman)}** "
                f"| {_num(audit.pearson)} | {_num(audit.short_tercile_mean)} "
                f"| {_num(audit.long_tercile_mean)} |"
            )
        lines += [
            "",
            "The two tercile columns are the robust reading: the mean score of the shortest "
            "third of answers against the longest third, each over "
            f"{max((audit.tercile_n for audit in analysis.verbosity), default=0)} cases or fewer.",
            "",
            "A positive coefficient does not prove the judge rewards length. Longer answers may "
            "genuinely cover more of what the rubric asked for. It marks the arm whose long "
            "answers should be read by hand.",
        ]

    lines += ["", "### Position", ""]
    if not analysis.position:
        lines.append(
            "No change verdict recorded which answer was shown first, so position bias could "
            "not be checked. That is a gap in the run, not evidence of its absence."
        )
        return lines
    lines += [
        "Each change verdict shows the judge two answers. If the verdict depends on which came "
        "first, the family-behaviour rates above are partly an artefact of presentation order.",
        "",
        "| scope | order | verdicts | correct | correct rate | judged as changed | change rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for audit in analysis.position:
        for order in audit.orders:
            decided, correct = audit.correct_by_order.get(order, (0, 0))
            moved, changed = audit.change_by_order.get(order, (0, 0))
            lines.append(
                f"| `{audit.arm}` | {_cell(order, 40)} | {decided} | {correct} "
                f"| **{_rate(correct / decided, decided) if decided else '-'}** | {changed} "
                f"| {_rate(changed / moved, moved) if moved else '-'} |"
            )
    pooled = next((audit for audit in analysis.position if audit.arm == "all"), None)
    if pooled is not None:
        lines.append("")
        if pooled.correct_gap is None:
            decided = sum(n for n, _ in pooled.correct_by_order.values())
            if len(pooled.orders) != 2:
                reason = (
                    f"verdicts were recorded under "
                    f"{_plural(len(pooled.orders), 'distinct order label')}"
                )
            elif not decided:
                reason = "no verdict under either order was decided"
            else:
                reason = "one of the two orders has no decided verdict"
            lines.append(
                f"Position bias is untested in this run: {reason}, so the two orders cannot be "
                f"compared. Recording a balanced order for every comparison would fix this."
            )
        else:
            lines.append(
                f"Pooled across arms, the correct rate differs by **{pooled.correct_gap:+.0%}** "
                f"between the two orders (Fisher exact p = {_p(pooled.p_value)}"
                + (
                    f", change-rate gap {pooled.change_gap:+.0%}"
                    if pooled.change_gap is not None
                    else ""
                )
                + ")."
            )
            if not pooled.balanced:
                lines.append("")
                lines.append(
                    "The two orders were not assigned in comparable numbers, so this gap is "
                    "confounded with whatever decided the ordering. Balance the assignment "
                    "before reading anything into it."
                )
        if pooled.order_by_measure:
            lines += [
                "",
                "Order assignment per behaviour, so an unbalanced design is visible:",
                "",
                "| behaviour | " + " | ".join(_cell(o, 30) for o in pooled.orders) + " |",
                "|---|" + "---|" * len(pooled.orders),
            ]
            for measure, counts in sorted(pooled.order_by_measure.items()):
                cells = " | ".join(str(counts.get(order, 0)) for order in pooled.orders)
                lines.append(f"| {measure} | {cells} |")
        lines.append("")
        lines.append(
            "Fisher's exact test conditions on the observed margins and assumes independent "
            "verdicts. Verdicts about variants of one family are not independent, so a small "
            "p-value means look at the ordering, not that the judge is biased."
        )
    return lines


# --------------------------------------------------------------------- sample counts by family


def _sample_counts(analysis: Analysis, suite: Suite) -> list[str]:
    lines = ["", "## Sample counts by scenario family", ""]
    if not analysis.family_counts:
        lines.append("The suite defines no family.")
        return lines
    graded_total = sum(stat.graded for stat in analysis.integrity)
    lines += [
        f"{analysis.suite_families} families, {analysis.suite_cases} cases, {graded_total} graded "
        f"answers across {len(analysis.arms)} arm(s). The cases inside one family are one "
        f"situation across tasks and variants. Treating them as independent samples would "
        f"overstate the evidence by roughly the number of cases per family.",
        "",
        "| family | kind | title | suite cases | tasks | variants | "
        + " | ".join(f"graded `{arm}`" for arm in analysis.arms)
        + " |",
        "|---|---|---|---|---|---|" + "---|" * len(analysis.arms),
    ]
    for family in analysis.family_counts:
        graded = " | ".join(str(family.graded_by_arm.get(arm, 0)) for arm in analysis.arms)
        lines.append(
            f"| `{family.family_id}` | {family.kind} | {_cell(family.title, 60)} "
            f"| {family.suite_cases} | {len(family.tasks)} "
            f"| {_cell(', '.join(family.variants), 60)} | {graded} |"
        )
    kinds: dict[str, list[int]] = {}
    for family in analysis.family_counts:
        row = kinds.setdefault(family.kind, [0, 0, 0])
        row[0] += 1
        row[1] += 1 if family.suite_cases else 0
        row[2] += family.suite_cases
    lines += [
        "",
        "| family kind | families | families with cases | cases |",
        "|---|---|---|---|",
    ]
    for kind, (families, contributing, cases) in sorted(kinds.items()):
        lines.append(f"| {kind} | {families} | {contributing} | {cases} |")
    empty = [f.family_id for f in analysis.family_counts if not f.suite_cases]
    if empty:
        lines += [
            "",
            f"{_plural(len(empty), 'family', 'families')} in the suite carry no case: "
            + ", ".join(f"`{family_id}`" for family_id in empty)
            + ". They are outside every denominator above, which is why the two family columns "
            "differ.",
        ]
    per_family = analysis.suite_cases / analysis.suite_families if analysis.suite_families else 0
    lines += [
        "",
        f"Mean of {per_family:.1f} cases per family. Every rate in this report prints its family "
        f"count alongside its case count for this reason.",
    ]
    return lines


# --------------------------------------------------------------------------- capability results


def _capability(analysis: Analysis) -> list[str]:
    lines = ["", "## Capability results", ""]
    lines.append(
        "General ability is a separate question from adherence to the specification, and is "
        "never folded into the scores above. It exists to reveal degradation: a model that "
        "improved on the value dimensions while losing instruction following has not improved."
    )
    if not analysis.capability:
        lines += ["", "No capability checks were run."]
        return lines
    lines += [
        "",
        "| check family | arm | passed | n | rate |",
        "|---|---|---|---|---|",
    ]
    arm_order = {arm: index for index, arm in enumerate(analysis.arms)}
    for stat in sorted(
        analysis.capability,
        key=lambda s: (s.family, arm_order.get(s.arm, len(arm_order)), s.arm),
    ):
        unknown = f" (+{stat.unknown} unrecorded)" if stat.unknown else ""
        lines.append(
            f"| {stat.family} | `{stat.arm}` | {stat.passed} | {stat.n}{unknown} "
            f"| **{_rate(stat.rate, stat.n)}** |"
        )
    failures = [(stat, item) for stat in analysis.capability for item in stat.failures]
    if failures:
        lines += ["", "Checks that failed:", ""]
        for stat, (check_id, detail) in failures[:12]:
            lines.append(
                f"- `{stat.arm}` / {stat.family} / `{check_id}`: "
                f"{_truncate(detail, 200) or 'no detail recorded'}"
            )
    if len(analysis.arms) > 1:
        lines += [
            "",
            "Compare the arms row by row. A capability drop alongside a values gain is the "
            "trade this section exists to make visible.",
        ]
    return lines


# ------------------------------------------------------------------------------- run record


def _run_record(analysis: Analysis, suite: Suite, meta: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Run record", ""]
    lines.append(
        "Kept so a result can be reproduced or disputed: which suite, which specification, "
        "which model answered and which graded."
    )
    lines += [
        "",
        "| field | value |",
        "|---|---|",
        f"| suite | `{suite.suite_id or '?'}` version `{suite.version}` |",
        f"| target | `{suite.target_id or '?'}` |",
        f"| specification | `{suite.spec_version or '?'}` hash `{suite.spec_hash or '?'}` |",
        f"| suite created | {suite.created_utc or 'not recorded'} |",
        f"| baseline arm | `{analysis.baseline_arm or 'none'}` |",
        f"| judge model(s) | {', '.join(f'`{m}`' for m in analysis.judge_models) or 'not recorded'} |",
        f"| distinct rubric versions | {len(analysis.rubric_versions)} |",
    ]
    for arm in analysis.arms:
        models = ", ".join(f"`{m}`" for m in analysis.model_by_arm.get(arm, ())) or "not recorded"
        lines.append(f"| arm `{arm}` | {models} |")
    for key in sorted(meta):
        if key == "title":
            continue
        value = meta[key]
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"| {_cell(key, 60)} | {_cell(rendered, 200)} |")
    lines += [
        "",
        "The judge never saw which arm produced an answer. Blindness is a property of the call "
        "the judge was made with, not an instruction it was asked to honour.",
    ]
    return lines


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ------------------------------------------------------------------------------- entry point


def render_report(analysis: Analysis, suite: Suite, meta: Mapping[str, Any] | None = None) -> str:
    """Render the full markdown report. Pure: no file writes, no model calls, no clock.

    Sections follow the order of the plan's "What the report should show" table, with the bias
    audits inserted after grading reliability because they are a statement about the
    measurement rather than about a model.
    """
    meta = dict(meta or {})
    lines: list[str] = []
    lines += _header(analysis, suite, meta)
    lines += _how_to_read(analysis)
    lines += _diagnostic_scores(analysis, suite)
    lines += _behaviour(analysis)
    lines += _overapplication(analysis)
    lines += _change_from_baseline(analysis)
    lines += _worst(analysis)
    lines += _reliability(analysis)
    lines += _format_premium_section(analysis)
    lines += _judge_disagreement(analysis)
    lines += _bias(analysis)
    lines += _sample_counts(analysis, suite)
    lines += _capability(analysis)
    lines += _run_record(analysis, suite, meta)
    text = "\n".join(lines).rstrip() + "\n"
    logger.info(
        "report: rendered %d lines for %d arm(s) over %d families",
        len(lines),
        len(analysis.arms),
        analysis.suite_families,
    )
    return text


#: The section headings this report always emits, in order. Tests assert on this list, and a
#: caller wanting to check a rendered report is complete can too.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "## How to read this",
    "## Diagnostic scores",
    "## Family-level behaviour",
    "## Overapplication rate",
    "## Change from baseline",
    "## Worst examples",
    "## Grading reliability and unresolved counts",
    "## Format premium",
    "## Judge disagreement",
    "## Judge bias audits",
    "## Sample counts by scenario family",
    "## Capability results",
    "## Run record",
)


__all__ = ["REPORT_FILE", "REQUIRED_SECTIONS", "render_report"]
