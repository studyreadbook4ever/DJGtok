"""Exact matching contracts and adversarial regressions, using synthetic IDs only."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from djgtok import Journey, Match, OverlapHit, OverlapIndex, Stop, compare_routes, reverse_journey


def route(identifier, stop_ids, mode="bus", line="line", metadata=None):
    return Journey(id=identifier, stops=[Stop(id=str(value), lat=37.5, lon=127) for value in stop_ids],
                   modes=[mode] * (len(stop_ids) - 1), line_ids=[line] * (len(stop_ids) - 1),
                   metadata=metadata or {})


def test_long_72_edge_overlap_ranks_above_short_four_edge_overlap():
    query = route("q", [f"s{i}" for i in range(100)])
    long = route("z-long", [f"s{i}" for i in range(10, 83)])
    short = route("a-short", [f"s{i}" for i in range(20, 25)])
    hits = OverlapIndex([short, long, query]).search(query, exclude_ids=["q"])
    assert [hit.journey_id for hit in hits] == ["z-long", "a-short"]
    assert [hit.score for hit in hits] == [72, 4]
    assert hits[0].query_stop_range == (10, 83)
    assert hits[0].document_stop_range == (0, 73)
    assert hits[0].query_coverage == pytest.approx(72 / 99)
    assert hits[0].document_coverage == 1
    assert hits[0].score_kind == "shared_directed_edge_count"


def test_noncontiguous_edges_cannot_add_up_to_a_long_match():
    query = route("q", ["a", "b", "c", "d", "e"])
    disconnected = route("d", ["a", "b", "x", "b", "c", "y", "c", "d", "z", "d", "e"])
    match = compare_routes(query, disconnected)
    assert match.shared_edge_count == 1
    assert match.query_stop_range == (0, 2)
    assert match.document_stop_range == (0, 2)


def test_forward_reverse_modes_and_namespaces_have_explicit_semantics():
    query = route("q", ["agency:a", "agency:b", "agency:c"])
    assert compare_routes(query, reverse_journey(query)).shared_edge_count == 0
    assert compare_routes(query, route("tram", ["agency:a", "agency:b", "agency:c"], mode="tram")).shared_edge_count == 0
    assert compare_routes(query, route("other", ["other:a", "other:b", "other:c"])).shared_edge_count == 0
    changed_line = route("new-line", ["agency:a", "agency:b", "agency:c"], line="different")
    assert compare_routes(query, changed_line).shared_edge_count == 2
    zero = compare_routes(query, route("none", ["x", "y"]))
    assert zero.query_stop_range is None and zero.document_stop_range is None
    assert zero.query_coverage == zero.document_coverage == 0


def test_mode_change_splits_an_otherwise_continuous_match():
    query = route("q", list("abcde"))
    document = route("d", list("abcde"))
    document.modes[1] = "rail"
    result = compare_routes(query, document)
    assert result.shared_edge_count == 2
    assert result.query_stop_range == result.document_stop_range == (2, 5)


def test_loops_preserve_repeated_edges_and_choose_earliest_equal_witness():
    query = route("q", list("ababac"))
    document = route("d", list("xabababy"))
    result = compare_routes(query, document)
    assert result.shared_edge_count == 4
    assert result.query_stop_range == (0, 5)
    assert result.document_stop_range == (1, 6)
    # Reversing a loop can retain genuinely present directed subsegments.
    assert compare_routes(route("loop", list("aba")), reverse_journey(route("r", list("aba")))).shared_edge_count == 2


def test_tie_order_is_stable_and_coverage_prefers_tighter_documents():
    query = route("q", list("abcd"))
    routes = [route("z", list("ab")), route("a", list("ab")), route("0-long", list("xabx"))]
    assert [hit.journey_id for hit in OverlapIndex(routes).search(query)] == ["a", "z", "0-long"]
    assert [hit.model_dump() for hit in OverlapIndex(routes).search(query)] == [
        hit.model_dump() for hit in OverlapIndex(reversed(routes)).search(query)]


def test_index_snapshots_inputs_outputs_and_source_evidence():
    query = route("q", list("abcd"), metadata={"source_rows": [10, 11, 12, 13]})
    document = route("d", list("xbcdy"), metadata={
        "source_rows": [30, 31, 32, 33, 34],
        "source_reference": {"original_rows": [30, 31, 32, 33, 34], "sheet": "Data"},
        "source_sha256": "supplied-provenance", "nested": {"tags": ["original"]},
    })
    index = OverlapIndex.build([document])
    document.stops[1].id = "changed"
    document.metadata["nested"]["tags"].append("changed")
    stored = index.get_journey("d")
    assert stored.stops[1].id == "b"
    assert stored.metadata["nested"] == {"tags": ["original"]}
    stored.stops[1].id = "changed-again"
    hit = index.search(query)[0]
    assert hit.shared_edge_count == 2
    assert hit.evidence["query"]["source_rows"] == [11, 12, 13]
    assert hit.evidence["document"]["source_rows"] == [31, 32, 33]
    assert hit.evidence["document"]["source_reference"]["original_rows"] == [31, 32, 33]
    hit.evidence["document"]["stops"][0]["id"] = "mutated-result"
    assert index.search(query)[0].evidence["document"]["stops"][0]["id"] == "b"
    assert query.metadata["source_rows"] == [10, 11, 12, 13]


def test_empty_index_filters_exclusions_and_duplicate_ids():
    query = route("q", list("abcd"))
    assert OverlapIndex([]).search(query) == []
    index = OverlapIndex([query, route("short", list("ab")), route("none", list("xy"))])
    assert len(index) == 3
    assert index.journey_ids == ("none", "q", "short")
    assert index.search(query, min_shared_edges=2, exclude_ids=["q"]) == []
    assert len(index.search(query, top_k=1)) == 1
    with pytest.raises(ValueError, match="duplicate"):
        OverlapIndex([query, query])
    with pytest.raises(KeyError):
        index.get_journey("missing")


@pytest.mark.parametrize("kwargs", [
    {"top_k": 0}, {"top_k": True}, {"top_k": 1.5}, {"min_shared_edges": 0},
    {"min_shared_edges": False}, {"exclude_ids": "q"}, {"exclude_ids": [1]},
])
def test_invalid_search_arguments_fail_explicitly(kwargs):
    query = route("q", list("ab"))
    with pytest.raises(ValueError):
        OverlapIndex([query]).search(query, **kwargs)


def test_match_and_hit_json_roundtrips_preserve_null_and_half_open_ranges():
    query, document = route("q", list("abc")), route("d", list("xbcx"))
    match = compare_routes(query, document)
    assert Match.model_validate_json(match.model_dump_json()) == match
    hit = OverlapIndex([document]).search(query)[0]
    assert OverlapHit.model_validate_json(hit.model_dump_json()) == hit
    zero = compare_routes(query, route("none", list("xy")))
    assert json.loads(zero.model_dump_json())["query_stop_range"] is None
    with pytest.raises(ValidationError):
        Match.model_validate(zero.model_dump() | {"query_stop_range": [0, 1]})
    with pytest.raises(ValidationError):
        OverlapHit.model_validate(hit.model_dump() | {"score": 2})


def test_longest_match_agrees_with_exhaustive_subsequence_oracle():
    rng = random.Random(19)
    for _ in range(120):
        query = route("q", [rng.choice("abcd") for _ in range(rng.randint(2, 10))])
        document = route("d", [rng.choice("abcd") for _ in range(rng.randint(2, 10))])
        query.modes = [rng.choice(["bus", "rail"]) for _ in query.modes]
        document.modes = [rng.choice(["bus", "rail"]) for _ in document.modes]
        q_edges = [(a.id, b.id, m) for a, b, m in zip(query.stops, query.stops[1:], query.modes)]
        d_edges = [(a.id, b.id, m) for a, b, m in zip(document.stops, document.stops[1:], document.modes)]
        candidates = [(0, 0, 0)]
        for q_start in range(len(q_edges)):
            for d_start in range(len(d_edges)):
                for length in range(1, min(len(q_edges) - q_start, len(d_edges) - d_start) + 1):
                    if q_edges[q_start:q_start + length] == d_edges[d_start:d_start + length]:
                        candidates.append((length, q_start, d_start))
        count, q_start, d_start = min(candidates, key=lambda x: (-x[0], x[1], x[2]))
        result = compare_routes(query, document)
        assert result.shared_edge_count == count
        assert result.query_stop_range == ((q_start, q_start + count + 1) if count else None)
        assert result.document_stop_range == ((d_start, d_start + count + 1) if count else None)


def test_public_core_import_and_use_does_not_import_optional_dependencies():
    source_root = Path(__file__).resolve().parents[1] / "src"
    program = '''
import importlib.abc
import sys
class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "fastapi", "uvicorn", "openpyxl"}:
            raise RuntimeError("Core imported optional dependency " + fullname)
sys.meta_path.insert(0, RejectOptional())
from djgtok import Journey, Stop, TransitTokenizer, OverlapIndex
j = Journey(id="q", stops=[Stop(id="a",lat=37,lon=127),Stop(id="b",lat=37.1,lon=127)], modes=["bus"],line_ids=["l"])
assert TransitTokenizer.fit([j]).encode(j)[0].shape == (1,3)
assert OverlapIndex([j]).search(j)[0].score == 1
'''
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True,
                               env={**os.environ, "PYTHONPATH": str(source_root)}, check=False)
    assert completed.returncode == 0, completed.stderr
