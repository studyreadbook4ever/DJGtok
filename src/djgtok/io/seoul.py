"""Import the Seoul Metropolitan Government's ordered bus-stop XLSX snapshot.

No routing, stop interpolation, direction inference, or GTFS conversion occurs.
Invalid routes are explicitly excluded as a whole rather than joined across gaps.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import openpyxl

HEADERS = ("ROUTE_ID", "노선명", "순번", "NODE_ID", "ARS_ID", "정류소명", "X좌표", "Y좌표")
DATASET_URL = "https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do"
FILENAME = "서울시버스노선별정류소정보(20260902).xlsx"


class SeoulDataError(ValueError):
    """Workbook structure cannot be interpreted without guessing."""


def _identifier(value: Any, column: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"missing_or_invalid_{column}")
    if isinstance(value, (int, float)):
        if not math.isfinite(value) or int(value) != value or value < 0:
            raise ValueError(f"noninteger_{column}")
        return str(int(value))
    text = str(value).strip()
    if not text:
        raise ValueError(f"missing_{column}")
    return text


def _sequence(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError("missing_or_invalid_sequence")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("noninteger_sequence") from exc
    if not math.isfinite(number) or number != int(number) or number < 1:
        raise ValueError("nonpositive_or_noninteger_sequence")
    return int(number)


def _coordinate(value: Any, axis: str) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_{axis}") from exc
    # Includes Seoul's suburban/airport routes; prevents zero/projected coordinates
    # being silently interpreted as Korean longitude/latitude.
    lower, upper = (124, 132) if axis == "longitude" else (33, 39)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"out_of_korea_range_{axis}")
    return number


def extract_seoul(workbook_path: str | Path, *, download_manifest: dict | None = None,
                  sheet_name: str = "Data") -> tuple[list[dict], dict]:
    """Return complete source-ordered route patterns and an explicit audit report.

    Every included stop retains its Excel row, original NODE_ID/ARS_ID/sequence,
    name and coordinates. Repeated stops remain repeated. A route with an invalid
    row, duplicate sequence or missing sequence is excluded in its entirety.
    """
    path = Path(workbook_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = download_manifest or {}
    if provenance.get("sha256") not in (None, digest):
        raise SeoulDataError("Workbook SHA256 differs from its download manifest")
    try:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise SeoulDataError(f"Cannot open Seoul workbook {path}: {exc}") from exc
    grouped: dict[str, list] = defaultdict(list)
    invalid_by_route: dict[str, list] = defaultdict(list)
    route_row_numbers: dict[str, list] = defaultdict(list)
    names_by_route: dict[str, set] = defaultdict(set)
    excluded_rows = []
    input_rows = empty_rows = 0
    try:
        if sheet_name not in book.sheetnames:
            raise SeoulDataError(f"Missing sheet {sheet_name!r}; found {book.sheetnames}")
        sheet = book[sheet_name]
        rows = sheet.iter_rows(values_only=True)
        header = tuple(next(rows, ()))
        if header != HEADERS:
            raise SeoulDataError(f"Unexpected source header: {header!r}; expected {HEADERS!r}")
        for row_number, row in enumerate(rows, 2):
            if all(value is None for value in row):
                empty_rows += 1
                continue
            input_rows += 1
            try:
                route_id = _identifier(row[0], "route_id")
            except ValueError as exc:
                excluded_rows.append({"source_row": row_number, "reason": str(exc)})
                continue
            route_row_numbers[route_id].append(row_number)
            try:
                route_name = _identifier(row[1], "route_name")
                names_by_route[route_id].add(route_name)
                seq = _sequence(row[2])
                node_id = _identifier(row[3], "node_id")
                ars_id = _identifier(row[4], "ars_id")
                name = _identifier(row[5], "stop_name")
                lon, lat = _coordinate(row[6], "longitude"), _coordinate(row[7], "latitude")
            except ValueError as exc:
                invalid_by_route[route_id].append({"source_row": row_number, "reason": str(exc)})
                continue
            grouped[route_id].append({"id": f"seoul:stop:{node_id}", "node_id": node_id,
                                      "ars_id": ars_id, "name": name, "lat": lat, "lon": lon,
                                      "route_seq": seq, "source_row": row_number})
    finally:
        book.close()
    if not input_rows:
        raise SeoulDataError("Workbook has no data rows")
    journeys, excluded_routes = [], []
    for route_id in sorted(route_row_numbers):
        stops = sorted(grouped[route_id], key=lambda stop: stop["route_seq"])
        seqs = [stop["route_seq"] for stop in stops]
        counts = Counter(seqs)
        duplicates = sorted(seq for seq, count in counts.items() if count > 1)
        # A corrupt huge sequence must not allocate an enormous missing-ID list.
        missing = sorted(set(range(1, max(seqs, default=0) + 1)) - set(seqs)) if max(seqs, default=0) <= 10_000 else []
        reasons = sorted({issue["reason"] for issue in invalid_by_route[route_id]})
        names = sorted(names_by_route[route_id])
        if len(names) != 1:
            reasons.append("inconsistent_or_missing_route_name")
        if not 2 <= len(stops) <= 512:
            reasons.append("stop_count_outside_2_to_512")
        if duplicates:
            reasons.append("duplicate_sequence")
        if missing or seqs != list(range(1, len(stops) + 1)):
            reasons.append("noncontinuous_sequence")
        if reasons:
            excluded_routes.append({"route_id": route_id, "route_name": "/".join(names),
                                    "reasons": sorted(set(reasons)),
                                    "source_rows": route_row_numbers[route_id],
                                    "missing_sequences": missing, "duplicate_sequences": duplicates,
                                    "invalid_rows": invalid_by_route[route_id]})
            continue
        source_rows = [stop["source_row"] for stop in stops]
        pattern_id = f"seoul:bus:{route_id}:pattern"
        stop_ids = [stop["node_id"] for stop in stops]
        topology = hashlib.sha256("|".join(stop_ids).encode()).hexdigest()[:20]
        metadata = {
            "kind": "planned_seoul_bus_pattern", "parent_pattern_id": pattern_id,
            "topology_family_id": f"seoul:topology:{topology}",
            "route_id": route_id, "route_name": names[0],
            "source_url": provenance.get("dataset_url"), "source_download_url": provenance.get("url"),
            "source_workbook": path.name, "source_sheet": sheet_name,
            "source_rows": source_rows, "source_sha256": digest,
            "source_reference": {"workbook": path.name, "sheet": sheet_name, "original_rows": source_rows},
            "source_snapshot_date": provenance.get("snapshot_date"),
            "provider": provenance.get("provider", "not verified for this local workbook"),
            "source_verified": bool(provenance.get("dataset_url") == DATASET_URL and provenance.get("sha256") == digest),
            "direction_semantics": "official sequence retained; outbound/return boundary not supplied",
            "roundtrip_status": "unsplit source sequence; may include outbound and return",
            "contains_repeated_stops": len(set(stop_ids)) != len(stop_ids),
            "is_closed_stop_loop": stop_ids[0] == stop_ids[-1],
            "coordinate_interpretation": "source X=longitude,Y=latitude; datum not declared in workbook",
            "timetable_validated": False, "boarding_permissions_available": False,
            "geometry_kind": "stop_to_stop_straight_segments_not_road_shape",
        }
        journeys.append({"id": pattern_id, "stops": stops, "modes": ["bus"] * (len(stops) - 1),
                         "line_ids": [f"seoul:bus:{route_id}"] * (len(stops) - 1),
                         "source": "seoul-open-data" if metadata["source_verified"] else "local-seoul-format-workbook",
                         "metadata": metadata})
    report = {
        "dataset": "서울시 버스 노선 정보 조회: 서울시버스노선별정류소정보",
        "provider": provenance.get("provider"), "source_url": provenance.get("dataset_url"),
        "source_workbook": path.name, "source_sheet": sheet_name, "source_sha256": digest,
        "source_header": list(HEADERS), "download": provenance,
        "input_rows": input_rows, "empty_rows": empty_rows, "input_routes": len(route_row_numbers),
        "patterns": len(journeys), "journeys": len(journeys), "derived_slices": 0,
        "included_rows": sum(len(journey["stops"]) for journey in journeys),
        "excluded_route_rows": sum(len(route["source_rows"]) for route in excluded_routes),
        "excluded_routes": excluded_routes, "excluded_rows": excluded_rows,
        "exclusion_policy": "Exclude the entire route for invalid source rows, sequence gaps/duplicates, inconsistent name or >512 stops",
        "direction_policy": "Preserve original sequence; no inferred inbound/outbound split",
        "geometry_kind": "straight connections of official stop coordinates, not roadway geometry",
        "service_limitations": "No timetables, boarding restrictions, real-time status, or transfer feasibility in this workbook",
    }
    if not journeys:
        raise SeoulDataError(f"No complete routes remain; exclusions: {excluded_routes[:3]}")
    return journeys, report
