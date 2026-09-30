"""Benchmark checks must expose direction, mode and ordering failures."""
import json
from pathlib import Path

from benchmarks.fixtures import synthetic_cases, synthetic_journey
from benchmarks.run_benchmark import oracle_longest_edges, summarize_timings, synthetic_benchmark, write_report


def test_synthetic_suite_checks_longer_match_loops_mode_and_unknown_ids():
    report = synthetic_benchmark(warmup=0, repeats=1)
    assert report["all_cases_passed"]
    assert len(report["cases"]) == 4
    assert report["cases"][0]["top_result"] == "long"
    assert report["cases"][-1]["actual_shared_edge_counts"] == {}


def test_independent_oracle_rejects_wrong_direction_and_mode():
    query = synthetic_journey("q", "ABCDE")
    assert oracle_longest_edges(query, synthetic_journey("reversed", "EDCBA")) == 0
    assert oracle_longest_edges(query, synthetic_journey("wrong-mode", "ABCDE", ["rail"] * 4)) == 0
    assert oracle_longest_edges(query, synthetic_journey("disconnected", "AXBZC")) == 0
    assert oracle_longest_edges(query, synthetic_journey("partial", "XBCDY")) == 2


def test_loop_oracle_counts_consecutive_occurrences_not_edge_set_membership():
    case = synthetic_cases()[1]
    assert oracle_longest_edges(case.query, case.documents[0]) == 4
    assert oracle_longest_edges(case.query, case.documents[1]) == 2


def test_timing_summary_records_nearest_rank_p95_and_no_hidden_warmup_samples():
    result = summarize_timings([1.0, 3.0, 2.0, 9.0], warmup=2, repeats=2, query_count=2)
    assert result["median_ms"] == 2.5
    assert result["p95_ms"] == 9.0
    assert result["sample_count"] == 4


def test_public_report_excludes_fixture_stop_lists_and_paths(tmp_path: Path):
    report = synthetic_benchmark(warmup=0, repeats=1)
    output = tmp_path / "public.json"
    write_report(report, output)
    text = output.read_text()
    assert "synthetic:stop:" not in text
    assert str(tmp_path) not in text
    assert "\"stops\"" not in text
    assert json.loads(text)["all_cases_passed"] is True
