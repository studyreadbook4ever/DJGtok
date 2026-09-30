"""Auditable Seoul source rows; tests use clearly synthetic local XLSX fixtures."""
from __future__ import annotations

import hashlib

import pytest

openpyxl = pytest.importorskip("openpyxl")

from djgtok.schema import Journey
from djgtok.io.seoul import HEADERS, SeoulDataError, extract_seoul


def fixture_rows():
    # Interleaved routes and unsorted original rows deliberately exercise sequence
    # ordering. A repeats NODE_ID 10, which must not be deduplicated as a loop.
    return [
        [100, "0017", 3, 10, "01001", "출발역", 127.01, 37.51],
        [200, "fixture-B", 2, 22, "02002", "도착역", 127.02, 37.52],
        [100, "0017", 1, 10, "01001", "출발역", 127.01, 37.51],
        [200, "fixture-B", 1, 21, "02001", "기점", 127.015, 37.515],
        [100, "0017", 2, 11, "01002", "중간역", 127.011, 37.511],
        [100, "0017", 4, 12, "01003", "종점", 127.012, 37.512],
    ]


def workbook(tmp_path, rows=None, headers=HEADERS):
    path = tmp_path / "synthetic-seoul-fixture.xlsx"
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Data"
    sheet.append(headers)
    for row in fixture_rows() if rows is None else rows:
        sheet.append(row)
    book.save(path)
    book.close()
    return path


def test_exact_source_values_and_row_provenance_survive_ordering(tmp_path):
    path = workbook(tmp_path)
    journeys, report = extract_seoul(path)
    assert report["patterns"] == 2
    assert report["input_rows"] == report["included_rows"] == 6
    route = next(item for item in journeys if item["metadata"]["route_id"] == "100")
    Journey.model_validate(route)
    assert route["metadata"]["route_name"] == "0017"
    assert [stop["route_seq"] for stop in route["stops"]] == [1, 2, 3, 4]
    assert [stop["node_id"] for stop in route["stops"]] == ["10", "11", "10", "12"]
    assert [stop["source_row"] for stop in route["stops"]] == [4, 6, 2, 7]
    assert route["metadata"]["source_rows"] == [4, 6, 2, 7]
    assert route["metadata"]["source_reference"]["original_rows"] == [4, 6, 2, 7]
    assert route["stops"][0]["ars_id"] == "01001"
    assert route["stops"][0]["lat"] == 37.51
    assert route["stops"][0]["lon"] == 127.01
    assert route["metadata"]["contains_repeated_stops"] is True
    assert route["metadata"]["is_closed_stop_loop"] is False
    assert route["metadata"]["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert route["metadata"]["timetable_validated"] is False
    assert route["metadata"]["source_verified"] is False
    assert route["source"] == "local-seoul-format-workbook"


@pytest.mark.parametrize("column,value,reason", [
    (2, 5, "noncontinuous_sequence"),
    (2, 1, "duplicate_sequence"),
    (2, None, "missing_or_invalid_sequence"),
    (2, 10**12, "noncontinuous_sequence"),
    (3, None, "missing_or_invalid_node_id"),
    (4, None, "missing_or_invalid_ars_id"),
    (6, 0, "out_of_korea_range_longitude"),
    (7, 0, "out_of_korea_range_latitude"),
    (1, "different name", "inconsistent_or_missing_route_name"),
])
def test_bad_source_row_excludes_whole_route_without_connecting_across_it(tmp_path, column, value, reason):
    rows = fixture_rows()
    rows[1][column] = value  # B's second source stop becomes invalid.
    journeys, report = extract_seoul(workbook(tmp_path, rows))
    assert [item["metadata"]["route_id"] for item in journeys] == ["100"]
    assert report["patterns"] == 1
    assert report["included_rows"] == 4
    assert report["excluded_route_rows"] == 2
    assert report["excluded_routes"][0]["route_id"] == "200"
    assert report["excluded_routes"][0]["source_rows"] == [3, 5]
    assert reason in report["excluded_routes"][0]["reasons"]


def test_unassignable_route_id_is_explicitly_reported(tmp_path):
    rows = fixture_rows() + [[None, "unknown", 1, 33, "03003", "정류장", 127.03, 37.53]]
    _, report = extract_seoul(workbook(tmp_path, rows))
    assert report["excluded_rows"] == [{"source_row": 8, "reason": "missing_or_invalid_route_id"}]
    assert report["included_rows"] + report["excluded_route_rows"] + len(report["excluded_rows"]) == report["input_rows"]


def test_source_manifest_hash_mismatch_rejected(tmp_path):
    with pytest.raises(SeoulDataError, match="SHA256"):
        extract_seoul(workbook(tmp_path), download_manifest={"sha256": "wrong"})


def test_changed_header_fails_instead_of_guessing_columns(tmp_path):
    headers = list(HEADERS)
    headers[2] = "unknown_order_field"
    with pytest.raises(SeoulDataError, match="Unexpected source header"):
        extract_seoul(workbook(tmp_path, headers=headers))


def test_no_complete_routes_fails_instead_of_emitting_empty_success(tmp_path):
    rows = fixture_rows()[:1]
    with pytest.raises(SeoulDataError, match="No complete routes"):
        extract_seoul(workbook(tmp_path, rows))


def test_empty_source_fails(tmp_path):
    with pytest.raises(SeoulDataError, match="no data rows"):
        extract_seoul(workbook(tmp_path, []))
