"""Regression checks for leakage, false positives and evaluation denominators."""
from pathlib import Path

import numpy as np
import pytest

from djgtok.experimental.evaluate import (
    GeographicBaseline,
    QueryCase,
    audit_splits,
    directed_overlap,
    make_query_cases,
    perturb_cases,
    sha256_path,
    summarize_rankings,
    topology,
)
from djgtok.schema import Journey, Stop, reverse_journey, split_group, subjourney


def journey(identifier="parent", names="ABCDEFGH", line="route-A", **metadata):
    return Journey(
        id=identifier,
        stops=[Stop(id=name, lat=42.0, lon=-71.0 + (ord(name) - ord("A")) * 0.005) for name in names],
        modes=["bus"] * (len(names) - 1),
        line_ids=[line] * (len(names) - 1),
        source="test-fixture",
        metadata=metadata,
    )


def test_exact_baseline_requires_contiguous_directed_edges():
    query = journey(names="ABC")
    assert directed_overlap(query, journey(names="XABCY")) == 1.0
    assert directed_overlap(query, journey(names="CBA")) == 0.0
    assert directed_overlap(query, journey(names="AXBYC")) == 0.0
    assert directed_overlap(query, journey(names="ABXC")) == 0.5


def test_split_audit_rejects_group_leakage_even_for_different_ids():
    records = [journey("train-variant", split="train"), journey("test-variant", split="test")]
    with pytest.raises(ValueError, match="split leakage"):
        audit_splits(records, split_fn=lambda record: record.metadata["split"])


def test_crop_queries_preserve_stop_order_group_and_are_reproducible():
    parent = journey()
    cases = make_query_cases([parent], max_queries=10, seed=19)
    again = make_query_cases([parent], max_queries=10, seed=19)
    assert [case.journey.model_dump() for case in cases] == [case.journey.model_dump() for case in again]
    assert len(cases) == 1
    case = cases[0]
    assert 2 <= len(case.journey.stops) < len(parent.stops)
    assert case.parent_id == parent.id
    assert split_group(case.journey) == split_group(parent)
    assert directed_overlap(case.journey, parent) == 1.0
    assert case.reverse_eligible


def test_reversal_stress_eligibility_excludes_out_and_back_ambiguity():
    parent = journey(names="ABCCBA")
    cases = make_query_cases([parent], max_queries=1, seed=0)
    assert not cases[0].reverse_eligible


def test_observation_stress_is_labeled_without_changing_topology_or_lines():
    parent = journey()
    cases = make_query_cases([parent], max_queries=1, seed=0)
    renamed = perturb_cases(cases, seed=9, noisy=False)[0]
    assert topology(renamed.journey) == topology(cases[0].journey)
    assert [(stop.lat, stop.lon) for stop in renamed.journey.stops] == [
        (stop.lat, stop.lon) for stop in cases[0].journey.stops
    ]
    assert renamed.journey.line_ids == cases[0].journey.line_ids
    assert renamed.parent_id == parent.id
    assert renamed.journey.metadata["synthetic_observation"] is True
    assert directed_overlap(renamed.journey, parent) == 0.0
    noisy = perturb_cases(cases, seed=9, noisy=True)[0]
    assert topology(noisy.journey) != topology(cases[0].journey)
    assert noisy.journey.model_dump() == perturb_cases(cases, seed=9, noisy=True)[0].journey.model_dump()


def test_geographic_baseline_prefers_forward_to_same_geometry_reversed():
    parent = journey()
    reverse = reverse_journey(parent)
    baseline = GeographicBaseline([reverse, parent])
    scores = dict(baseline.search(subjourney(parent, 2, 6)))
    assert scores[parent.id] > scores[reverse.id]
    assert all(np.isfinite(value) for value in scores.values())


def test_metrics_do_not_hide_ties_or_direction_failures():
    query = journey()
    case = QueryCase(query, "parent", "reverse:parent", True)
    result = summarize_rankings(
        [case], [[("another-valid-route", 1.0), ("parent", 1.0), ("reverse:parent", 0.0)]],
        [3.0], method="test", scenario="test",
    )
    assert result["known_parent_recall"]["at_1"] == 0.0
    assert result["known_parent_recall_uniform_tie_break"]["at_1"] == 0.5
    assert result["known_parent_recall"]["at_5"] == 1.0
    assert result["top_score_tie_query_rate"] == 1.0
    assert result["direction_stress"]["paired_reverse_outranks_parent_rate"] == 0.0
    failure = summarize_rankings(
        [case], [[("reverse:parent", 0.8), ("parent", 0.2)]], [2.0], method="test", scenario="test",
    )
    assert failure["direction_stress"]["paired_reverse_outranks_parent_rate"] == 1.0
    assert failure["direction_stress"]["paired_reverse_top1_rate"] == 1.0


def test_metrics_reject_missing_parent_instead_of_silently_counting_failure():
    case = QueryCase(journey(), "parent", "reverse:parent", True)
    with pytest.raises(ValueError, match="contain parent"):
        summarize_rankings([case], [[("reverse:parent", 0.1)]], [1.0], method="test", scenario="test")


def test_rule_relevance_uses_clean_reference_for_renamed_noisy_query():
    parent = journey()
    clean = subjourney(parent, 0, 4)  # A -> B -> C -> D: three directed edges.
    original = QueryCase(clean, parent.id, f"reverse:{parent.id}", True)
    observed = perturb_cases([original], seed=12, noisy=True)[0]
    alternative = journey("alternative", names="ABC")
    reverse = reverse_journey(parent)
    assert directed_overlap(observed.journey, alternative) == 0.0
    ranking = [(alternative.id, 0.9), (parent.id, 0.8), (reverse.id, 0.1)]
    result = summarize_rankings(
        [observed], [ranking], [1.0], method="test", scenario="renamed_ids_and_coordinate_noise",
        reference_queries={parent.id: clean},
        documents_by_id={document.id: document for document in [alternative, parent, reverse]},
    )
    relevance = result["rule_defined_shared_segment_relevance"]
    assert relevance["mean_top1_directed_query_coverage"] == pytest.approx(2 / 3)
    assert relevance["hit_at_5_coverage_gte_0_5"] == 1.0
    assert result["known_parent_recall"]["at_1"] == 0.0
    assert ranking[0][0] == alternative.id  # Relevance never changes the ranking.


def test_content_hash_changes_when_artifact_changes(tmp_path: Path):
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"first checkpoint")
    original = sha256_path(artifact)
    artifact.write_bytes(b"second checkpoint")
    assert sha256_path(artifact) != original


def test_legacy_mode_agnostic_protocol_and_long_valid_ids_are_preserved():
    parent = journey(identifier="x" * 500)
    rail = parent.model_copy(update={"modes": ["rail"] * (len(parent.stops) - 1)})
    assert directed_overlap(parent, rail) == 1.0  # Historical protocol only.
    from djgtok import compare_routes
    assert compare_routes(parent, rail).shared_edge_count == 0
    cases = make_query_cases([parent], max_queries=1, seed=42)
    noisy = perturb_cases(cases, seed=42, noisy=True)
    for case in cases + noisy:
        Journey.model_validate(case.journey.model_dump())
        assert len(case.journey.id) <= 500
