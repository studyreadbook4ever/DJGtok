"""Exact contiguous directed-stop matching, independent of learned embeddings.

Stop IDs must be canonical within a shared provider namespace. Prefix IDs when
combining providers (for example ``agency:stop:123``); normalize cross-provider
IDs before comparing. Equal names or coordinates are not evidence of equal IDs.
Matching uses (from_stop_id, to_stop_id, mode), ignores line IDs, and proves only
the supplied planned stop sequence, not road geometry or timetable feasibility.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from djgtok.schema import Journey, _snapshot_journey, subjourney

Edge = tuple[str, str, str]
StopIndex = Annotated[int, Field(ge=0, strict=True)]
StopRange = tuple[StopIndex, StopIndex]


class Match(BaseModel):
    """Longest common consecutive edge sequence; stop ranges are half-open.

    Coverage is shared_edge_count / total edges of that input, never probability.
    Zero overlap has both ranges None and both coverages zero. Equal longest
    sequences select the earliest query start, then earliest document start.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    shared_edge_count: int = Field(ge=0, strict=True)
    query_stop_range: StopRange | None
    document_stop_range: StopRange | None
    query_coverage: float = Field(ge=0, le=1, strict=True)
    document_coverage: float = Field(ge=0, le=1, strict=True)
    match_kind: Literal["exact_directed_stop_id_and_mode"] = "exact_directed_stop_id_and_mode"

    @model_validator(mode="after")
    def consistent_ranges(self) -> Match:
        if self.shared_edge_count == 0:
            if (self.query_stop_range is not None or self.document_stop_range is not None
                    or self.query_coverage != 0 or self.document_coverage != 0):
                raise ValueError("zero overlap requires null ranges and zero coverage")
        else:
            for interval in (self.query_stop_range, self.document_stop_range):
                if (interval is None or interval[0] < 0
                        or interval[1] - interval[0] != self.shared_edge_count + 1):
                    raise ValueError("matched stop ranges must contain shared_edge_count + 1 stops")
            if self.query_coverage == 0 or self.document_coverage == 0:
                raise ValueError("nonzero overlap requires positive coverage")
        return self


class OverlapHit(Match):
    """A retrieval hit whose integer score is the exact shared edge count.

    Evidence is copied from the supplied inputs; it is not independent source
    verification. Source rows, when supplied, are cropped to these same ranges.
    """

    journey_id: str = Field(min_length=1, strict=True)
    score: int = Field(ge=1, strict=True)
    score_kind: Literal["shared_directed_edge_count"] = "shared_directed_edge_count"
    evidence: dict[str, Any]

    @model_validator(mode="after")
    def score_matches_count(self) -> OverlapHit:
        if self.score != self.shared_edge_count:
            raise ValueError("score must equal shared_edge_count")
        return self


def _edges(journey: Journey) -> tuple[Edge, ...]:
    return tuple((left.id, right.id, mode) for left, right, mode in zip(
        journey.stops[:-1], journey.stops[1:], journey.modes, strict=True))


def _compare_edges(query_edges: tuple[Edge, ...], document_edges: tuple[Edge, ...]) -> Match:
    positions: dict[Edge, list[int]] = defaultdict(list)
    for position, edge in enumerate(document_edges, 1):
        positions[edge].append(position)
    previous: dict[int, int] = {}
    best_count, query_start, document_start = 0, 0, 0
    for query_end, edge in enumerate(query_edges, 1):
        current = {}
        for document_end in positions.get(edge, ()):
            length = previous.get(document_end - 1, 0) + 1
            current[document_end] = length
            q_start, d_start = query_end - length, document_end - length
            if (-length, q_start, d_start) < (-best_count, query_start, document_start):
                best_count, query_start, document_start = length, q_start, d_start
        previous = current
    return Match(
        shared_edge_count=best_count,
        query_stop_range=(query_start, query_start + best_count + 1) if best_count else None,
        document_stop_range=(document_start, document_start + best_count + 1) if best_count else None,
        query_coverage=best_count / len(query_edges),
        document_coverage=best_count / len(document_edges),
    )


def compare_routes(query: Journey, document: Journey) -> Match:
    """Compare validated snapshots by longest consecutive directed ID+mode match.

    Repeated stops/loops are preserved; disconnected shared edges do not add up.
    A reversed sequence only matches where directed edges really recur in both.
    """
    return _compare_edges(_edges(_snapshot_journey(query)), _edges(_snapshot_journey(document)))


def _evidence(journey: Journey, interval: StopRange) -> dict[str, Any]:
    cropped = subjourney(journey, *interval)
    result: dict[str, Any] = {
        "journey_id": journey.id,
        "stop_range": list(interval),
        "source": journey.source,
        "stops": [stop.model_dump() for stop in cropped.stops],
        "modes": list(cropped.modes),
        "line_ids": list(cropped.line_ids),
    }
    for key in ("source_rows", "source_reference", "source_url", "source_download_url",
                "source_sha256", "source_sheet", "source_workbook", "route_id", "route_name"):
        if key in cropped.metadata:
            result[key] = deepcopy(cropped.metadata[key])
    return result


class OverlapIndex:
    """In-memory exact matcher with an inverted directed-edge candidate filter.

    Construction snapshots the corpus. Search never depends on later caller
    mutations and returns only positive overlaps. Ranking: shared edge count,
    query coverage, document coverage (all descending), then journey ID ascending.
    This exact implementation targets bounded route-pattern corpora, not a
    distributed or approximate nearest-neighbor index.
    """

    def __init__(self, journeys: Iterable[Journey]):
        self._journeys: dict[str, Journey] = {}
        self._edges_by_id: dict[str, tuple[Edge, ...]] = {}
        self._inverted: dict[Edge, set[str]] = defaultdict(set)
        for journey in journeys:
            snapshot = _snapshot_journey(journey)
            if snapshot.id in self._journeys:
                raise ValueError(f"duplicate journey ID: {snapshot.id}")
            self._journeys[snapshot.id] = snapshot
            edges = _edges(snapshot)
            self._edges_by_id[snapshot.id] = edges
            for edge in set(edges):
                self._inverted[edge].add(snapshot.id)

    @classmethod
    def build(cls, journeys: Iterable[Journey]) -> OverlapIndex:
        return cls(journeys)

    def __len__(self) -> int:
        return len(self._journeys)

    @property
    def journey_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._journeys))

    def get_journey(self, journey_id: str) -> Journey:
        """Return an independent snapshot; unknown IDs raise KeyError."""
        return self._journeys[journey_id].model_copy(deep=True)

    def search(
        self, query: Journey, top_k: int = 5, min_shared_edges: int = 1,
        exclude_ids: Iterable[str] = (),
    ) -> list[OverlapHit]:
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if type(min_shared_edges) is not int or min_shared_edges < 1:
            raise ValueError("min_shared_edges must be a positive integer")
        if isinstance(exclude_ids, (str, bytes)):
            raise ValueError("exclude_ids must be an iterable of journey ID strings")
        excluded = set(exclude_ids)
        if any(not isinstance(identifier, str) for identifier in excluded):
            raise ValueError("exclude_ids must contain only strings")
        query = _snapshot_journey(query)
        query_edges = _edges(query)
        candidates: set[str] = set()
        for edge in set(query_edges):
            candidates.update(self._inverted.get(edge, ()))
        matches = []
        for identifier in candidates - excluded:
            match = _compare_edges(query_edges, self._edges_by_id[identifier])
            if match.shared_edge_count >= min_shared_edges:
                matches.append((identifier, match))
        matches.sort(key=lambda item: (
            -item[1].shared_edge_count, -item[1].query_coverage,
            -item[1].document_coverage, item[0],
        ))
        hits = []
        for identifier, match in matches[:top_k]:
            assert match.query_stop_range is not None and match.document_stop_range is not None
            hits.append(OverlapHit(
                **match.model_dump(), journey_id=identifier, score=match.shared_edge_count,
                evidence={"query": _evidence(query, match.query_stop_range),
                          "document": _evidence(self._journeys[identifier], match.document_stop_range)},
            ))
        return hits
