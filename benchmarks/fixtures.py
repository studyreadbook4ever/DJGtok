"""Entirely synthetic fixtures: these IDs and coordinates are not real stops."""
from __future__ import annotations

from dataclasses import dataclass

from djgtok import Journey, Stop


def synthetic_journey(identifier: str, names: str, modes: list[str] | None = None) -> Journey:
    return Journey(
        id=identifier,
        stops=[Stop(id=f"synthetic:stop:{name}", lat=10.0, lon=20.0 + (ord(name) - 65) * 0.001)
               for name in names],
        modes=modes if modes is not None else ["bus"] * (len(names) - 1),
        line_ids=[f"synthetic:line:{identifier}"] * (len(names) - 1),
        source="synthetic-benchmark-fixture",
        metadata={"synthetic": True},
    )


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    query: Journey
    documents: tuple[Journey, ...]
    expected_scores: dict[str, int]
    expected_first: str | None


def synthetic_cases() -> list[SyntheticCase]:
    linear = synthetic_journey("query-linear", "ABCDEFGHI")
    renamed = linear.model_copy(deep=True, update={
        "id": "query-unmatched-ids",
        "stops": [stop.model_copy(update={"id": "unmatched:" + stop.id}) for stop in linear.stops],
    })
    return [
        SyntheticCase(
            "longer_corridor_before_short_identical_fragment", linear,
            (linear, synthetic_journey("long", "XBCDEFGHY"),
             synthetic_journey("short", "XBCDY"), synthetic_journey("reverse", "IHGFEDCBA"),
             synthetic_journey("wrong-mode", "ABCDEFGHI", ["rail"] * 8),
             synthetic_journey("disconnected", "AXBZCYDWEVF")),
            {"long": 6, "short": 2}, "long",
        ),
        SyntheticCase(
            "repeated_loop_order", synthetic_journey("query-loop", "ABCBCDE"),
            (synthetic_journey("loop-match", "XBCBCDY"),
             synthetic_journey("reordered", "ABDCBCE")),
            {"loop-match": 4, "reordered": 2}, "loop-match",
        ),
        SyntheticCase(
            "mode_is_part_of_edge_identity",
            synthetic_journey("query-mode", "ABCDEF", ["bus", "bus", "rail", "rail", "bus"]),
            (synthetic_journey("mixed-correct", "ABCDEF", ["bus", "bus", "rail", "rail", "bus"]),
             synthetic_journey("all-bus", "ABCDEF")),
            {"mixed-correct": 5, "all-bus": 2}, "mixed-correct",
        ),
        SyntheticCase("coordinate_equality_does_not_replace_canonical_ids", renamed, (linear,), {}, None),
    ]
