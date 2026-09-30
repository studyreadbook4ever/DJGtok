"""Run synthetic checks or aggregate-only benchmarks on user-supplied JSONL.

No input paths, stop names, stop IDs, coordinates, model weights, or complete
journeys are written to reports. This module never downloads datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from djgtok import Journey, OverlapIndex, compare_routes, reverse_journey, subjourney

from .fixtures import synthetic_cases

SEOUL_SOURCE = {
    "provider": "Seoul Metropolitan Government",
    "dataset_url": "https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do",
    "license": "Korea Open Government License Type 1 (attribution)",
    "snapshot": "2026-09-02",
    "workbook_sha256": "b54816e9b7e77bcf49dd7562b993db06a093daa666351f115ea4d1dd0dd8b1df",
    "redistribution": "No raw or processed source dataset is included in this repository.",
}
SEOUL_CASES = (("750A", "750B", 72), ("702A", "702B", 57), ("6411", "8641", 46))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    versions = {}
    for name in ("numpy", "pydantic", "djgtok"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "source-checkout-not-installed"
    cpu = platform.processor() or platform.machine()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "os": platform.system(), "machine": platform.machine(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count(), "package_versions": versions}


def summarize_timings(values_ms: list[float], warmup: int, repeats: int, query_count: int) -> dict:
    if not values_ms or any(not math.isfinite(value) or value < 0 for value in values_ms):
        raise ValueError("Timing samples must be finite nonnegative values")
    ordered = sorted(values_ms)
    return {"median_ms": statistics.median(ordered),
            "p95_ms": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
            "sample_count": len(ordered), "warmup_passes": warmup, "measured_passes": repeats,
            "queries_per_pass": query_count, "p95_convention": "nearest rank",
            "scope": "in-process query search and hit construction; excludes JSONL loading and index construction"}


def measure_queries(index: OverlapIndex, queries: list[Journey], *, warmup: int, repeats: int,
                    top_k: int = 5, exclude_self: bool = False) -> dict:
    if warmup < 0 or repeats < 1 or not queries:
        raise ValueError("Require nonempty queries, warmup >= 0 and repeats >= 1")
    for _ in range(warmup):
        for query in queries:
            index.search(query, top_k=top_k, exclude_ids=(query.id,) if exclude_self else ())
    times = []
    for _ in range(repeats):
        for query in queries:
            start = time.perf_counter_ns()
            index.search(query, top_k=top_k, exclude_ids=(query.id,) if exclude_self else ())
            times.append((time.perf_counter_ns() - start) / 1_000_000)
    return summarize_timings(times, warmup, repeats, len(queries))


def edge_sequence(journey: Journey) -> list[tuple[str, str, str]]:
    return [(left.id, right.id, mode) for left, right, mode in
            zip(journey.stops, journey.stops[1:], journey.modes)]


def oracle_longest_edges(query: Journey, document: Journey) -> int:
    """Independent quadratic dynamic-programming oracle for benchmark checks."""
    left, right = edge_sequence(query), edge_sequence(document)
    previous, best = [0] * (len(right) + 1), 0
    for edge in left:
        current = [0] * (len(right) + 1)
        for position, candidate in enumerate(right, 1):
            if edge == candidate:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def synthetic_benchmark(*, warmup: int = 3, repeats: int = 20) -> dict:
    results = []
    for case in synthetic_cases():
        index = OverlapIndex(case.documents)
        hits = index.search(case.query, top_k=len(case.documents), exclude_ids=(case.query.id,))
        actual = {hit.journey_id: hit.shared_edge_count for hit in hits}
        first = hits[0].journey_id if hits else None
        oracle = {document.id: oracle_longest_edges(case.query, document) for document in case.documents
                  if document.id != case.query.id}
        oracle = {key: value for key, value in oracle.items() if value > 0}
        passed = actual == case.expected_scores == oracle and first == case.expected_first
        if not passed:
            raise AssertionError(f"Synthetic case {case.name} failed: {actual}, expected {case.expected_scores}")
        results.append({"case": case.name, "passed": passed, "corpus_size": len(case.documents),
                        "query_edges": len(case.query.modes), "expected_shared_edge_counts": case.expected_scores,
                        "actual_shared_edge_counts": actual, "top_result": first,
                        "latency": measure_queries(index, [case.query], warmup=warmup, repeats=repeats, exclude_self=True)})
    return {"schema_version": 1, "task": "synthetic exact directed stop-ID+mode retrieval",
            "environment": environment(), "all_cases_passed": True, "cases": results,
            "data": "All fixture IDs and coordinates are invented, explicitly synthetic, and unrelated to actual stops."}


def load_dataset(path: Path) -> list[Journey]:
    result, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            journey = Journey.model_validate_json(line)
            if journey.id in seen:
                raise ValueError(f"Duplicate journey ID at JSONL line {line_number}")
            seen.add(journey.id)
            result.append(journey)
    if not result:
        raise ValueError("Input JSONL is empty")
    return result


def route_split(journey: Journey) -> str:
    group = str(journey.metadata.get("split_group") or "|".join(sorted(set(journey.line_ids))))
    bucket = int(hashlib.sha256(("transitvec-v1:" + group).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def coordinate_topology(journey: Journey) -> tuple:
    return tuple((round(stop.lat, 5), round(stop.lon, 5)) for stop in journey.stops)


def crop_cohort(journeys: list[Journey], *, seed: int = 42, max_queries: int = 200) -> tuple[list, list, dict]:
    """Replay the archived generated-query cohort; no algorithm training occurs."""
    if max_queries < 1:
        raise ValueError("max_queries must be positive")
    partitions = {name: [journey for journey in journeys if route_split(journey) == name]
                  for name in ("train", "val", "test")}
    seen = {coordinate_topology(journey) for journey in partitions["train"] + partitions["val"]}
    eligible = sorted((journey for journey in partitions["test"] if coordinate_topology(journey) not in seen),
                      key=lambda journey: journey.id)
    rng, queries = np.random.default_rng(seed), []
    for index in rng.permutation(len(eligible))[:max_queries]:
        parent = eligible[int(index)]
        count = len(parent.stops)
        upper = count - 1 if count > 2 else 2
        lower = min(upper, max(2, math.ceil(count * 0.4)))
        length = int(rng.integers(lower, upper + 1))
        start = int(rng.integers(0, count - length + 1))
        queries.append((subjourney(parent, start, start + length), parent))
    if not queries:
        raise ValueError("No queries remain in the deterministic test cohort")
    return queries, partitions["test"], {
        "seed": seed, "max_queries": max_queries,
        "partition_counts": {key: len(value) for key, value in partitions.items()},
        "excluded_full_coordinate_topology_duplicates": len(partitions["test"]) - len(eligible),
        "query_count": len(queries),
        "split_rule": "sha256('transitvec-v1:' + route_group), first 8 hex digits modulo 100; <70 train, <85 val, otherwise test",
        "interpretation": "Historical route-group test cohort replay only; exact matcher has no training or model selection stage.",
    }


def benchmark_crop_cohort(journeys: list[Journey], *, warmup: int, repeats: int, seed: int, max_queries: int) -> dict:
    queries, corpus, protocol = crop_cohort(journeys, seed=seed, max_queries=max_queries)
    reversals = [reverse_journey(journey) for journey in corpus]
    index = OverlapIndex(corpus + reversals)
    at1 = at5 = eligible_reverse = reverse_above_parent = reverse_returned = 0
    for query, parent in queries:
        hits = index.search(query, top_k=5)
        ids = [hit.journey_id for hit in hits]
        at1 += bool(ids) and ids[0] == parent.id
        at5 += parent.id in ids
        reversal = reverse_journey(parent)
        if oracle_longest_edges(query, reversal) == 0:
            eligible_reverse += 1
            parent_score = len(query.modes)  # Original contiguous crop is fully present in its parent.
            reverse_score = compare_routes(query, reversal).shared_edge_count
            reverse_above_parent += reverse_score > parent_score
            reverse_returned += any(hit.journey_id == reversal.id for hit in hits)
    count = len(queries)
    return {
        "protocol": protocol, "corpus_real_routes": len(corpus), "synthetic_reverse_decoys": len(reversals),
        "known_parent_recall_at_1": at1 / count, "known_parent_recall_at_5": at5 / count,
        "known_parent_hits_at_1": at1, "known_parent_hits_at_5": at5,
        "strict_reverse_eligible_queries": eligible_reverse,
        "paired_reverse_outranks_parent_count": reverse_above_parent,
        "paired_zero_overlap_reverse_returned_in_top5_count": reverse_returned,
        "latency": measure_queries(index, [query for query, _ in queries], warmup=warmup, repeats=repeats),
        "limitations": [
            "Known-parent labels are generated from exact source crops; alternative documents can be equally correct.",
            "Direction checks include only pairs with zero matching oriented edge in the clean query's paired reversal.",
            "These are rule-defined structural checks, not independent human relevance or statistical generalization claims.",
        ],
    }


def seoul_benchmark(path: Path, *, warmup: int = 3, repeats: int = 10, seed: int = 42,
                    max_queries: int = 200) -> dict:
    dataset_hash = sha256_file(path)
    journeys = load_dataset(path)
    source_hashes = {journey.metadata.get("source_sha256") for journey in journeys}
    if source_hashes != {SEOUL_SOURCE["workbook_sha256"]}:
        raise ValueError("This named Seoul snapshot benchmark requires the documented 2026-09-02 source SHA-256")
    if any(journey.metadata.get("kind") != "planned_seoul_bus_pattern" for journey in journeys):
        raise ValueError("Provide original Seoul source patterns, not generated slices or synthetic reversals")
    by_name = {}
    for journey in journeys:
        name = journey.metadata.get("route_name")
        if name in by_name:
            raise ValueError("Source route names must be unique for the named-pair benchmark")
        by_name[name] = journey
    started = time.perf_counter_ns()
    index = OverlapIndex(journeys)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000
    pairs, timing_queries = [], []
    for query_name, target_name, expected_edges in SEOUL_CASES:
        query, target = by_name[query_name], by_name[target_name]
        timing_queries.append(query)
        hits = index.search(query, top_k=len(journeys), exclude_ids=(query.id,))
        rank = next((position for position, hit in enumerate(hits, 1) if hit.journey_id == target.id), None)
        if rank is None:
            raise AssertionError(f"Known shared-segment target {target_name} is missing")
        target_hit = hits[rank - 1]
        actual_edges = oracle_longest_edges(query, target)
        if actual_edges != expected_edges or target_hit.shared_edge_count != expected_edges:
            raise AssertionError(f"Raw-audited {query_name}/{target_name} shared-edge count does not reproduce")
        oracle_max = max(oracle_longest_edges(query, document) for document in journeys if document.id != query.id)
        maximum_first = hits[0].shared_edge_count == oracle_max
        ordered_lengths = all(left.shared_edge_count >= right.shared_edge_count for left, right in zip(hits, hits[1:]))
        target_before_shorter = all(position >= rank for position, hit in enumerate(hits, 1)
                                    if hit.shared_edge_count < expected_edges)
        if not (maximum_first and ordered_lengths and target_before_shorter):
            raise AssertionError("Longer shared corridors were ranked below shorter matches")
        pairs.append({
            "query_route": query_name, "target_route": target_name,
            "query_is_full_source_route": True, "self_excluded": True,
            "target_rank": rank, "target_shared_edges": target_hit.shared_edge_count,
            "query_edge_count": len(query.modes), "target_edge_count": len(target.modes),
            "target_query_coverage": target_hit.query_coverage,
            "target_document_coverage": target_hit.document_coverage,
            "independent_oracle_max_shared_edges": oracle_max,
            "longest_shared_corridor_ranked_first": maximum_first,
            "target_precedes_every_shorter_match": target_before_shorter,
            "full_stop_sequences_identical": [stop.id for stop in query.stops] == [stop.id for stop in target.stops],
        })
    crops = benchmark_crop_cohort(journeys, warmup=warmup, repeats=repeats, seed=seed, max_queries=max_queries)
    if sha256_file(path) != dataset_hash:
        raise ValueError("Input dataset changed during benchmark execution")
    return {
        "schema_version": 1, "task": "exact consecutive directed stop-ID+mode matching",
        "environment": environment(), "source": SEOUL_SOURCE, "processed_dataset_sha256": dataset_hash,
        "corpus": {"journeys": len(journeys), "stop_rows": sum(len(journey.stops) for journey in journeys),
                   "edges": sum(len(journey.modes) for journey in journeys)},
        "index_build_ms": build_ms, "full_route_cases": pairs,
        "full_route_search_latency": measure_queries(index, timing_queries, warmup=warmup, repeats=repeats, exclude_self=True),
        "generated_crop_cohort": crops,
        "limitations": [
            "This is a deterministic known-ID overlap task, not a neural quality improvement or semantic similarity benchmark.",
            "Canonical stop namespaces and mode labels must agree; renamed IDs with equal coordinates are not equivalent inputs.",
            "The source contains route stop sequences, not verified timetables, road shapes, transfer feasibility, or passenger preference labels.",
            "Recorded timings are warm local measurements on this corpus, not all-city scale guarantees or route-planning latency.",
            "No raw source data, complete journeys, actual stop lists, coordinates, or model weights are redistributed.",
        ],
    }


def historical_aggregates(evaluation_path: Path, audit_path: Path, training_path: Path) -> dict:
    """Whitelist aggregates; never copy private paths or per-query examples."""
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    metrics = []
    for item in evaluation["metrics"]:
        metrics.append({
            "method": item["method"], "scenario": item["scenario"], "query_count": item["query_count"],
            "known_parent_recall": item["known_parent_recall"],
            "top_score_tie_query_rate": item["top_score_tie_query_rate"],
            "direction_stress": {key: item["direction_stress"][key] for key in
                                 ("eligible_query_count", "paired_reverse_outranks_parent_rate", "paired_reverse_top1_rate")},
            "rule_defined_shared_segment_relevance": {key: item["rule_defined_shared_segment_relevance"][key]
                                                       for key in ("mean_top1_directed_query_coverage", "hit_at_5_coverage_gte_0_5")},
        })
    dense = audit["dense_verification"]
    cross_route = [{
        "query_route": item["query_route_name"], "target_route": item["raw_selected_target_name"],
        "target_rank_excluding_self": item["raw_selected_target_rank_excluding_self"],
        "query_training_split": item["query_training_split"],
    } for item in dense["examples"]]
    best_record = min((record for record in training["history"] if record["step"] > 0), key=lambda record: record["val_loss"])
    return {
        "schema_version": 1, "status": "archived experimental neural results; not the current exact matcher",
        "source": SEOUL_SOURCE,
        "provenance_sha256": {"evaluation_report": sha256_file(evaluation_path), "raw_audit_report": sha256_file(audit_path),
                              "training_report": sha256_file(training_path), "evaluated_model": evaluation["model_sha256"]},
        "corpus": {key: evaluation["corpus"][key] for key in
                   ("planned_original_records", "planned_test_parents", "synthetic_reverse_decoys", "total_documents",
                    "query_parent_candidates", "excluded_duplicate_train_val_topology_parents", "queries_with_exact_subpath_seen_in_train_or_val")},
        "training": {"optimization_steps": max(record["step"] for record in training["history"]),
                     "selected_step": best_record["step"], "selected_validation_loss": best_record["val_loss"],
                     "train_patterns": training["metadata"]["train_pattern_count"],
                     "validation_patterns": training["metadata"]["val_pattern_count"],
                     "selection": "minimum fixed validation contrastive loss; no test-based checkpoint selection"},
        "metrics": metrics, "full_route_cross_route_examples": cross_route,
        "limitations": [
            "The neural shared scorer maximized one local chunk cosine; short identical chunks could saturate near 1 and hide longer overlap.",
            "750A to 750B ranked 27th despite 72 consecutive shared directed edges. This negative result is retained.",
            "Crops and noise are generated, not independent human relevance labels. Three cropped query sequences also occur in train/validation routes.",
            "Full-route illustrative queries were training routes. They are not held-out generalization results.",
            "Exact known-ID overlap addresses a different task from learned coordinate-robust retrieval; comparing the two is not a claim of improved neural quality.",
        ],
    }


def write_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("synthetic", "seoul"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--warmup", type=int, default=3)
        sub.add_argument("--repeats", type=int, default=10)
        if command == "seoul":
            sub.add_argument("--data", type=Path, required=True, help="User-provided processed Seoul JSONL; never downloaded or copied")
            sub.add_argument("--seed", type=int, default=42)
            sub.add_argument("--max-queries", type=int, default=200)
    archive = subparsers.add_parser("historical")
    archive.add_argument("--evaluation", type=Path, required=True)
    archive.add_argument("--raw-audit", type=Path, required=True)
    archive.add_argument("--training", type=Path, required=True)
    archive.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "synthetic":
        report = synthetic_benchmark(warmup=args.warmup, repeats=args.repeats)
    elif args.command == "seoul":
        report = seoul_benchmark(args.data, warmup=args.warmup, repeats=args.repeats,
                                 seed=args.seed, max_queries=args.max_queries)
    else:
        report = historical_aggregates(args.evaluation, args.raw_audit, args.training)
    write_report(report, args.output)
    print(json.dumps({"benchmark": args.command, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
