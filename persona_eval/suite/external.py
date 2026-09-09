"""Situations from a public dataset, so the suite is not entirely self-authored.

A suite written from the same specification the model was trained on shares that
specification's blind spots: it can only ask about situations the specification's authors
thought of, and it inherits their sense of what is hard. Seeding part of the suite with
situations nobody on this project wrote breaks that circularity, and it makes the provenance
column mean something.

Two rules follow from the evaluation plan and are enforced here rather than left to a prompt.
The dataset's own labels ("values_aggregated", the negative consequence it supplies) are
DROPPED: they encode that dataset's standard, not our target specification, and the plan hides
supplied labels unless they are explicitly being tested. And the fetch degrades to an empty
list on any network or parsing failure, because an authoring run that costs money must not die
because a public HTTP service was briefly unavailable.

The rows are public and were not written for this project, but "public" is not "held out": a
dataset row may still resemble something in the training corpus. Provenance is recorded here;
persona_eval.suite.contamination is what actually checks the overlap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import httpx

logger = logging.getLogger("persona_eval.suite.external")

#: The Hugging Face datasets server. Reachable without a token for public datasets.
DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"

DAILY_DILEMMAS = {
    "dataset": "kellycyy/daily_dilemmas",
    "config": "Dilemmas_with_values_aggregated",
    "split": "test",
}
#: Rows in the aggregated config repeat one dilemma once per candidate action, so the row
#: count is several times the number of distinct situations.
DAILY_DILEMMAS_SOURCE = "daily_dilemmas"

DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True)
class ExternalSituation:
    """One situation lifted verbatim from a public dataset, with its provenance.

    `situation` is the dataset's text, whitespace-normalised and otherwise untouched: the
    value of an external seed is that nobody here wrote it, so nobody here rewrites it.
    """

    situation: str
    provenance: str          # "<dataset>:<row idx>", stored on the Family
    source: str              # dataset short name
    row_index: int
    topic: str = ""          # the dataset's own grouping, used only as a domain hint

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalise(text: str) -> str:
    return " ".join(str(text or "").split())


def _rows_url_params(spec: dict[str, str], offset: int, length: int) -> dict[str, Any]:
    return {
        "dataset": spec["dataset"],
        "config": spec["config"],
        "split": spec["split"],
        "offset": offset,
        "length": length,
    }


def parse_daily_dilemmas(payload: Any, *, min_chars: int = 80) -> list[ExternalSituation]:
    """Turn a datasets-server page into situations, dropping the dataset's own labels.

    Kept separate from the fetch so the parsing can be tested without a network call.
    """
    situations: list[ExternalSituation] = []
    seen_dilemmas: set[int] = set()
    seen_text: set[str] = set()
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    for entry in rows:
        if not isinstance(entry, dict):
            continue
        row = entry.get("row") or {}
        text = _normalise(row.get("dilemma_situation"))
        if len(text) < min_chars:
            continue
        # One dilemma appears once per candidate action; keep the first only, or the suite
        # would build several families on one situation and treat them as independent.
        dilemma_idx = row.get("dilemma_idx")
        if isinstance(dilemma_idx, int):
            if dilemma_idx in seen_dilemmas:
                continue
            seen_dilemmas.add(dilemma_idx)
        if text in seen_text:
            continue
        seen_text.add(text)
        row_index = int(entry.get("row_idx", row.get("idx", len(situations))))
        situations.append(
            ExternalSituation(
                situation=text,
                provenance=f"{DAILY_DILEMMAS_SOURCE}:{row_index}",
                source=DAILY_DILEMMAS_SOURCE,
                row_index=row_index,
                topic=_normalise(row.get("topic_group")),
            )
        )
    return situations


async def fetch_daily_dilemmas(
    *,
    limit: int = 20,
    offset: int = 0,
    page_size: int = 100,
    max_pages: int = 10,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    url: str = DATASETS_SERVER_ROWS,
    client: httpx.AsyncClient | None = None,
) -> list[ExternalSituation]:
    """Fetch up to `limit` distinct situations. Returns [] on any failure, never raises.

    `client` is here for tests, which pass an httpx.MockTransport client so no test ever
    touches the network. When it is None a client is created and closed here.
    """
    if limit <= 0:
        return []
    own_client = client is None
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
    collected: list[ExternalSituation] = []
    try:
        cursor = offset
        # Deduplication means a page yields fewer situations than it holds rows, so keep
        # paging until the request is satisfied or the split runs out. `max_pages` bounds it
        # regardless: a server that keeps returning the same page must not hang a run.
        for _ in range(max_pages):
            if len(collected) >= limit:
                break
            params = _rows_url_params(DAILY_DILEMMAS, cursor, page_size)
            response = await http.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
            page = parse_daily_dilemmas(payload)
            known = {situation.situation for situation in collected}
            collected.extend(s for s in page if s.situation not in known)
            rows_returned = len(payload.get("rows", []) or []) if isinstance(payload, dict) else 0
            total = payload.get("num_rows_total") if isinstance(payload, dict) else None
            cursor += rows_returned
            if rows_returned == 0 or (isinstance(total, int) and cursor >= total):
                break
    except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError, TypeError) as error:
        # Deliberately broad on the parsing side: a dataset whose schema has moved must
        # degrade to "no external seeds", not stop an authoring run that costs money.
        logger.warning(
            "external situations unavailable (%s: %s); the suite will be authored without "
            "them and every family will be recorded as self-authored",
            type(error).__name__,
            error,
        )
        return []
    finally:
        if own_client:
            await http.aclose()
    if len(collected) < limit:
        logger.warning(
            "asked for %d external situations and found %d; authoring continues with what "
            "arrived",
            limit,
            len(collected),
        )
    logger.info("external: %d situations from %s", len(collected[:limit]), DAILY_DILEMMAS_SOURCE)
    return collected[:limit]


def render_situations(situations: Sequence[ExternalSituation]) -> str:
    """Number the situations for the metadata prompt; the numbers come back as `index`."""
    return "\n\n".join(
        f"{position}. {situation.situation}" for position, situation in enumerate(situations, 1)
    )


def provenance_counts(provenances: Iterable[str]) -> dict[str, int]:
    """Count families by source, splitting "<dataset>:<row>" back to the dataset name."""
    counts: dict[str, int] = {}
    for provenance in provenances:
        source = str(provenance).split(":", 1)[0] or "authored"
        counts[source] = counts.get(source, 0) + 1
    return counts


__all__ = [
    "DAILY_DILEMMAS",
    "DAILY_DILEMMAS_SOURCE",
    "DATASETS_SERVER_ROWS",
    "ExternalSituation",
    "fetch_daily_dilemmas",
    "parse_daily_dilemmas",
    "provenance_counts",
    "render_situations",
]
