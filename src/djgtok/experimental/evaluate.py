"""Held-out, generated-query retrieval evaluation; not a human relevance benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from djgtok.schema import (
    Journey,
    assign_split,
    derived_id,
    is_parent_pattern,
    load_journeys,
    reverse_journey,
    split_group,
    subjourney,
)


@dataclass(frozen=True)
class QueryCase:
    journey: Journey
    parent_id: str
    reverse_id: str
    reverse_eligible: bool


def sha256_path(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"No artifact files to hash: {path}")
    for file in files:
        if path.is_dir():
            digest.update(str(file.relative_to(path)).encode() + b"\0")
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def directed_overlap(query: Journey, document: Journey) -> float:
    """Legacy evaluation: longest stop-ID edge run / query edges, ignoring mode.

    Contiguity matters: two nonadjacent shared stops do not constitute a trip.
    This baseline deliberately requires IDs, unlike the coordinate model.
    The historical protocol is preserved here. Public matching additionally
    requires equal modes; use djgtok.compare_routes for that contract.
    """
    query_edges = [(a.id, b.id) for a, b in zip(query.stops, query.stops[1:])]
    document_edges = [(a.id, b.id) for a, b in zip(document.stops, document.stops[1:])]
    positions: dict[tuple[str, str], list[int]] = {}
    for column, edge in enumerate(document_edges):
        positions.setdefault(edge, []).append(column)
    previous: dict[int, int] = {}
    longest = 0
    for edge in query_edges:
        current = {column: previous.get(column - 1, 0) + 1 for column in positions.get(edge, [])}
        longest = max(longest, max(current.values(), default=0))
        previous = current
    return longest / len(query_edges)


def topology(journey: Journey) -> tuple[tuple[float, float], ...]:
    # Coordinate topology also catches aliases with different stop IDs.
    return tuple((round(stop.lat, 5), round(stop.lon, 5)) for stop in journey.stops)


def audit_splits(
    journeys: Sequence[Journey],
    split_fn: Callable[[Journey], str] = assign_split,
    group_fn: Callable[[Journey], str] = split_group,
) -> dict:
    groups: dict[str, set[str]] = {key: set() for key in ("train", "val", "test")}
    counts = Counter()
    for journey in journeys:
        split = split_fn(journey)
        if split not in groups:
            raise ValueError(f"Unknown split: {split}")
        groups[split].add(group_fn(journey))
        counts[split] += 1
    intersections = {
        f"{left}_{right}": sorted(groups[left] & groups[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(intersections.values()):
        raise ValueError(f"Route-group split leakage: {intersections}")
    return {
        "journey_counts": {key: counts[key] for key in groups},
        "group_counts": {key: len(value) for key, value in groups.items()},
        "groups": {key: sorted(value) for key, value in groups.items()},
        "group_intersections": intersections,
        "route_groups_disjoint": True,
    }


def make_query_cases(parents: Sequence[Journey], *, max_queries: int, seed: int) -> list[QueryCase]:
    if max_queries <= 0:
        raise ValueError("max_queries must be positive")
    rng = np.random.default_rng(seed)
    ordered = sorted(parents, key=lambda journey: journey.id)
    cases = []
    for index in rng.permutation(len(ordered))[:max_queries]:
        parent = ordered[int(index)]
        count = len(parent.stops)
        # Queries are strict crops except when a parent consists of one edge.
        upper = count - 1 if count > 2 else 2
        lower = min(upper, max(2, int(np.ceil(count * 0.4))))
        length = int(rng.integers(lower, upper + 1))
        start = int(rng.integers(0, count - length + 1))
        query = subjourney(parent, start, start + length)
        reverse = reverse_journey(parent)
        query = query.model_copy(update={
            "id": derived_id(parent.id, "", prefix="eval:clean:"),
            "metadata": {**query.metadata, "evaluation_generated_crop": True},
        })
        cases.append(QueryCase(query, parent.id, reverse.id, directed_overlap(query, reverse) == 0.0))
    return cases


def perturb_cases(cases: Sequence[QueryCase], *, seed: int, noisy: bool) -> list[QueryCase]:
    """Stress observations only; these are not claimed to be GTFS itineraries."""
    rng = np.random.default_rng(seed)
    scenario = "renamed_ids_and_coordinate_noise" if noisy else "renamed_ids"
    result = []
    for case in cases:
        stops = []
        for stop in case.journey.stops:
            lat, lon = stop.lat, stop.lon
            if noisy:
                # Independent 15 metre standard-deviation Gaussian GPS-like noise.
                lat += float(rng.normal(0, 15.0)) / 111_320.0
                lon += float(rng.normal(0, 15.0)) / (111_320.0 * max(0.01, np.cos(np.radians(stop.lat))))
                lat = min(90.0, max(-90.0, lat))
                lon = (lon + 180.0) % 360.0 - 180.0
            renamed = "observation:" + hashlib.sha256(stop.id.encode()).hexdigest()[:24]
            stops.append(stop.model_copy(update={"id": renamed, "lat": lat, "lon": lon}))
        query = case.journey.model_copy(update={
            "id": derived_id(case.parent_id, "", prefix=f"eval:{scenario}:"),
            "stops": stops,
            "source": "synthetic-observation-stress",
            "metadata": {**case.journey.metadata, "synthetic_observation": True,
                         "renamed_stop_ids": True, "coordinate_noise_std_m": 15.0 if noisy else 0.0},
        })
        result.append(QueryCase(query, case.parent_id, case.reverse_id, case.reverse_eligible))
    return result


def _windows(journey: Journey) -> list[Journey]:
    ranges = {(0, len(journey.stops))}
    edges = len(journey.stops) - 1
    for width in (4, 8, 16):
        if width < edges:
            starts = list(range(0, edges - width + 1, max(1, width // 2))) + [edges - width]
            ranges.update((start, start + width + 1) for start in starts)
    return [subjourney(journey, start, end) for start, end in sorted(ranges)]


def _geo_feature(journey: Journey) -> np.ndarray:
    coords = np.array([(stop.lat, stop.lon) for stop in journey.stops], dtype=np.float64)
    centroid = coords.mean(axis=0)
    delta = coords[-1] - coords[0]
    # North/east direction in a local equirectangular plane.
    direction = np.array([delta[0], delta[1] * np.cos(np.radians(centroid[0]))])
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 0 else np.zeros(2)
    return np.concatenate((centroid, direction))


class GeographicBaseline:
    """Fixed, untrained centroid + endpoint direction score over the same windows."""
    def __init__(self, documents: Sequence[Journey]):
        self.ids = [document.id for document in documents]
        self.features = [_geo_feature(window) for document in documents for window in _windows(document)]
        self.owners = np.array([index for index, document in enumerate(documents) for _ in _windows(document)])
        self.features = np.stack(self.features)

    def search(self, query: Journey) -> list[tuple[str, float]]:
        query_features = np.stack([_geo_feature(window) for window in _windows(query)])
        lat_delta = (query_features[:, None, 0] - self.features[None, :, 0]) * 111.32
        mean_lat = (query_features[:, None, 0] + self.features[None, :, 0]) / 2
        lon_delta = ((query_features[:, None, 1] - self.features[None, :, 1])
                     * 111.32 * np.cos(np.radians(mean_lat)))
        distance_km = np.hypot(lat_delta, lon_delta)
        alignment = np.clip(query_features[:, 2:] @ self.features[:, 2:].T, 0.0, 1.0)
        # Fixed 5 km scale; selected a priori, never optimized on held-out data.
        window_scores = (np.exp(-distance_km / 5.0) * (0.25 + 0.75 * alignment)).max(axis=0)
        scores = np.full(len(self.ids), -np.inf)
        np.maximum.at(scores, self.owners, window_scores)
        return sorted(zip(self.ids, scores.tolist()), key=lambda item: (-item[1], item[0]))


def summarize_rankings(cases: Sequence[QueryCase], rankings: Sequence[Sequence[tuple[str, float]]],
                       latency_ms: Sequence[float], *, method: str, scenario: str,
                       reference_queries: Mapping[str, Journey] | None = None,
                       documents_by_id: Mapping[str, Journey] | None = None) -> dict:
    if not cases or len(cases) != len(rankings) or len(cases) != len(latency_ms):
        raise ValueError("Evaluation needs aligned nonempty cases, rankings and timings")
    if (reference_queries is None) != (documents_by_id is None):
        raise ValueError("Rule-defined relevance requires both clean references and corpus documents")
    recalls = {k: 0 for k in (1, 5, 10)}
    tie_adjusted = {k: 0.0 for k in (1, 5, 10)}
    reverse_eligible = reverse_wins = reverse_top1 = reverse_ties = top_ties = 0
    top1_coverages, coverage_hits_at_5 = [], []
    failures = []
    for case, ranking in zip(cases, rankings):
        ids = [item[0] for item in ranking]
        if not ranking or len(set(ids)) != len(ids) or not all(np.isfinite(score) for _, score in ranking):
            raise ValueError(f"Invalid ranking for {case.journey.id}")
        if case.parent_id not in ids or case.reverse_id not in ids:
            raise ValueError("Full evaluation rankings must contain parent and paired reverse decoy")
        top5_coverage = None
        if reference_queries is not None and documents_by_id is not None:
            if case.parent_id not in reference_queries or any(identifier not in documents_by_id for identifier in ids[:5]):
                raise ValueError("Missing clean query or document for rule-defined relevance")
            reference = reference_queries[case.parent_id]
            top5_coverage = [directed_overlap(reference, documents_by_id[identifier]) for identifier in ids[:5]]
            top1_coverages.append(top5_coverage[0])
            coverage_hits_at_5.append(any(coverage >= 0.5 for coverage in top5_coverage))
        scores = dict(ranking)
        rank = ids.index(case.parent_id) + 1
        parent_score = scores[case.parent_id]
        higher = sum(score > parent_score + 1e-8 for _, score in ranking)
        ties = sum(abs(score - parent_score) <= 1e-8 for _, score in ranking)
        for k in recalls:
            recalls[k] += rank <= k
            tie_adjusted[k] += min(1.0, max(0.0, (k - higher) / ties))
        top_ties += sum(abs(score - ranking[0][1]) <= 1e-8 for _, score in ranking) > 1
        if case.reverse_eligible:
            reverse_eligible += 1
            reverse_score = scores[case.reverse_id]
            reverse_wins += reverse_score > parent_score + 1e-8
            reverse_ties += abs(reverse_score - parent_score) <= 1e-8
            reverse_top1 += ids[0] == case.reverse_id
        if rank > 1 and len(failures) < 10:
            failures.append({"query_id": case.journey.id, "parent_id": case.parent_id,
                             "parent_rank": rank, "parent_score": float(parent_score),
                             "paired_reverse_score": float(scores[case.reverse_id]),
                             "top5_directed_query_coverage": top5_coverage,
                             "top_5": [{"journey_id": identifier, "score": float(score)}
                                       for identifier, score in ranking[:5]]})
    count = len(cases)
    return {
        "method": method, "scenario": scenario, "query_count": count,
        "known_parent_recall": {f"at_{k}": recalls[k] / count for k in recalls},
        "known_parent_recall_uniform_tie_break": {f"at_{k}": tie_adjusted[k] / count for k in recalls},
        "top_score_tie_query_rate": top_ties / count,
        "rule_defined_shared_segment_relevance": {
            "mean_top1_directed_query_coverage": float(np.mean(top1_coverages)),
            "hit_at_5_coverage_gte_0_5": float(np.mean(coverage_hits_at_5)),
            "coverage_definition": "longest consecutive oriented stop-ID edge overlap / original clean query edge count",
            "reference": "original clean generated crop, including for renamed-ID and coordinate-noise scenarios",
            "label_status": "rule-defined partial-overlap relevance; not independent human relevance judgments",
        } if top1_coverages else None,
        "direction_stress": {
            "eligible_query_count": reverse_eligible,
            "eligibility": "clean query has no directed edge in its own parent's synthetic reversal",
            "paired_reverse_outranks_parent_rate": reverse_wins / reverse_eligible if reverse_eligible else None,
            "paired_reverse_top1_rate": reverse_top1 / reverse_eligible if reverse_eligible else None,
            "paired_reverse_score_tie_rate": reverse_ties / reverse_eligible if reverse_eligible else None,
        },
        "latency_ms": {"p50": float(np.percentile(latency_ms, 50)),
                       "p95": float(np.percentile(latency_ms, 95)), "mean": float(np.mean(latency_ms))},
        "failure_examples": failures,
    }


def _run_method(cases: Sequence[QueryCase], search: Callable, *, method: str, scenario: str,
                reference_queries: Mapping[str, Journey], documents_by_id: Mapping[str, Journey]) -> dict:
    search(cases[0].journey)  # Warm-up is deliberately excluded from latency statistics.
    timings, rankings = [], []
    for case in cases:
        started = time.perf_counter()
        ranking = search(case.journey)
        timings.append((time.perf_counter() - started) * 1000)
        # Standardized ID tie break avoids accidental corpus-order favoritism.
        rankings.append(sorted([(str(identifier), float(score)) for identifier, score in ranking],
                               key=lambda item: (-item[1], item[0])))
    return summarize_rankings(cases, rankings, timings, method=method, scenario=scenario,
                              reference_queries=reference_queries, documents_by_id=documents_by_id)


def run_evaluation(data_path: str | Path, model_path: str | Path, output_path: str | Path, *,
                   device: str = "cpu", max_queries: int = 200, seed: int = 42,
                   batch_size: int = 64, untrained_model_path: str | Path | None = None,
                   threads: int = 2) -> dict:
    import torch

    from .dense import DenseIndex
    from .model import EncoderBundle

    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    data_hash = sha256_path(data_path)
    model_hash = sha256_path(model_path)
    all_journeys = load_journeys(data_path)
    # Imported slices must not become independently split parent examples.
    originals = [journey for journey in all_journeys if is_parent_pattern(journey, require_source_kind=True)]
    if not originals:
        raise ValueError("Evaluation requires imported GTFS or Seoul planned parent patterns")
    split_audit = audit_splits(originals)
    train = [journey for journey in originals if assign_split(journey) == "train"]
    validation = [journey for journey in originals if assign_split(journey) == "val"]
    heldout = [journey for journey in originals if assign_split(journey) == "test"]
    seen_topologies = {topology(journey) for journey in train + validation}
    parents = [journey for journey in heldout if topology(journey) not in seen_topologies]
    if not parents:
        raise ValueError("No test parents remain after excluding train/validation duplicate topologies")
    cases = make_query_cases(parents, max_queries=max_queries, seed=seed)
    documents = sorted(heldout + [reverse_journey(journey) for journey in heldout], key=lambda journey: journey.id)
    if len({journey.id for journey in documents}) != len(documents):
        raise ValueError("Synthetic reverse identifiers collide with corpus journey IDs")
    bundle = EncoderBundle.load(model_path, device=device)
    if bundle.fingerprint != model_hash:
        raise ValueError("Checkpoint changed while evaluation was starting")
    training_metadata = getattr(bundle, "training_metadata", {})
    used_groups = set(training_metadata.get("train_groups", [])) | set(training_metadata.get("val_groups", []))
    group_leakage = sorted(used_groups & set(split_audit["groups"]["test"]))
    if group_leakage:
        raise ValueError(f"Checkpoint has seen test route groups: {group_leakage}")
    checkpoint_data_hash = training_metadata.get("data_sha256")
    if checkpoint_data_hash is not None and checkpoint_data_hash != data_hash:
        raise ValueError("Checkpoint training data hash differs from the evaluation dataset")
    index = DenseIndex.build(documents, bundle, batch_size=batch_size)
    geographic = GeographicBaseline(documents)
    searches = {
        "learned_whole": lambda query: [(hit.journey_id, hit.score) for hit in index.search(query, bundle, top_k=len(documents), mode="whole")],
        "learned_shared": lambda query: [(hit.journey_id, hit.score) for hit in index.search(query, bundle, top_k=len(documents), mode="shared")],
        "directed_exact_contiguous": lambda query: [(document.id, directed_overlap(query, document)) for document in documents],
        "geographic_shared": geographic.search,
    }
    untrained_hash = None
    if untrained_model_path is not None:
        untrained = EncoderBundle.load(untrained_model_path, device=device)
        untrained_index = DenseIndex.build(documents, untrained, batch_size=batch_size)
        searches["untrained_whole"] = lambda query: [(hit.journey_id, hit.score) for hit in untrained_index.search(query, untrained, top_k=len(documents), mode="whole")]
        searches["untrained_shared"] = lambda query: [(hit.journey_id, hit.score) for hit in untrained_index.search(query, untrained, top_k=len(documents), mode="shared")]
        untrained_hash = untrained.fingerprint
    scenarios = {
        "clean_source_crop": cases,
        "renamed_ids": perturb_cases(cases, seed=seed + 1, noisy=False),
        "renamed_ids_and_coordinate_noise": perturb_cases(cases, seed=seed + 1, noisy=True),
    }
    metrics = []
    reference_queries = {case.parent_id: case.journey for case in cases}
    documents_by_id = {document.id: document for document in documents}
    for scenario, scenario_cases in scenarios.items():
        for method, search in searches.items():
            metrics.append(_run_method(scenario_cases, search, method=method, scenario=scenario,
                                       reference_queries=reference_queries, documents_by_id=documents_by_id))
    training_sequences = [topology(journey) for journey in train + validation]
    query_overlap = 0
    for case in cases:
        query_topology = topology(case.journey)
        if any(any(sequence[start:start + len(query_topology)] == query_topology
                   for start in range(len(sequence) - len(query_topology) + 1))
               for sequence in training_sequences):
            query_overlap += 1
    if sha256_path(data_path) != data_hash or sha256_path(model_path) != model_hash:
        raise ValueError("Dataset or checkpoint changed during evaluation")
    if untrained_model_path and sha256_path(untrained_model_path) != untrained_hash:
        raise ValueError("Untrained checkpoint changed during evaluation")
    report = {
        "schema_version": 1,
        "evaluation_type": "generated partial-journey known-parent retrieval on route-group heldout public transit patterns",
        "dataset_sources": sorted({journey.source for journey in originals}),
        "pattern_kinds": sorted({journey.metadata["kind"] for journey in originals}),
        "data_path": str(Path(data_path)), "model_path": str(Path(model_path)),
        "data_sha256": data_hash, "model_sha256": model_hash,
        "untrained_model_sha256": untrained_hash,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "torch": str(torch.__version__), "platform": platform.platform()},
        "parameters": {"seed": seed, "max_queries": max_queries, "device": device, "torch_threads": threads,
                       "batch_size": batch_size, "chunk_edge_windows": [4, 8, 16],
                       "latency_retrieval_top_k": len(documents),
                       "geographic_distance_scale_km": 5.0},
        "split_audit": split_audit,
        "checkpoint_audit": {"recorded_group_check_available": bool(used_groups),
                             "test_group_overlap": group_leakage,
                             "training_data_hash_matches": checkpoint_data_hash == data_hash if checkpoint_data_hash else None},
        "corpus": {"loaded_dataset_records": len(all_journeys), "planned_original_records": len(originals),
                   "excluded_non_parent_records": len(all_journeys) - len(originals),
                   "planned_test_parents": len(heldout), "synthetic_reverse_decoys": len(heldout),
                   "total_documents": len(documents), "query_parent_candidates": len(parents),
                   "excluded_duplicate_train_val_topology_parents": len(heldout) - len(parents),
                   "queries_with_exact_subpath_seen_in_train_or_val": query_overlap},
        "ranking_policy": "cosine or baseline score descending; journey ID ascending for exact score ties; expected recall under uniform score ties also reported",
        "metrics": metrics,
        "limitations": [
            "Queries are generated contiguous source-pattern crops, not human searches; parent identity is known but complete relevance judgments are unavailable.",
            "Other documents may be equally valid shared-segment answers; parent Recall can penalize these. Exact-match score ties are reported explicitly.",
            "Shared-segment coverage and hit@5 use a rule-defined contiguous directed-edge match to the original clean generated query, including under observation noise; these are not independent human relevance labels and do not alter ranking.",
            "Training positive supervision uses directed-edge sets; it does not guarantee discrimination between all repeated or differently ordered loops.",
            "Route-line groups are disjoint; different lines can share stops and subpaths. Full duplicate coordinate topologies are excluded from query parents; remaining subpath overlap is counted.",
            "Reverse decoys and ID/coordinate perturbations are labelled synthetic stress cases, not operating services or observed passenger journeys.",
            "Renamed IDs intentionally disable the exact-ID baseline; gains there measure observation robustness, not superiority on clean source data.",
            "This evaluates geometry and direction retrieval, not timetable feasibility, transfer validity, fastest paths, passenger intent, or live traffic.",
            "Reported warm search latency requests the entire ranking to audit parent/reverse scores, includes query encoding/ranking, excludes index construction and HTTP overhead, and is not a production top-10 throughput benchmark.",
            "An untrained encoder comparison is omitted when no --untrained-model is supplied; geographic and exact baselines are always included.",
        ],
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--untrained-model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    report = run_evaluation(args.data, args.model, args.output, device=args.device,
                            max_queries=args.max_queries, seed=args.seed, batch_size=args.batch_size,
                            untrained_model_path=args.untrained_model, threads=args.threads)
    print(json.dumps({"report": args.output, "queries": report["metrics"][0]["query_count"],
                      "model_sha256": report["model_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
