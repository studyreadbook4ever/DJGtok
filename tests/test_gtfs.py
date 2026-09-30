"""Synthetic feed tests exercise GTFS semantics, without network or large data."""
from __future__ import annotations

import csv
import io
import zipfile

import pytest

from djgtok.io.gtfs import GTFSDataError, extract_gtfs, make_subjourneys


def feed_tables():
    return {
        "stops.txt": [["stop_id", "stop_name", "stop_lat", "stop_lon"]]
                     + [[str(i), f"Stop {i}", 42 + i / 1000, -71] for i in range(7)],
        "routes.txt": [["route_id", "route_type"], ["r", "3"]],
        "trips.txt": [["route_id", "service_id", "trip_id", "direction_id"],
                      ["r", "s", "t1", "0"], ["r", "s", "t2", "0"],
                      ["r", "reference", "ref", "0"]],
        "calendar_dates.txt": [["service_id", "date", "exception_type"],
                               ["s", "20260929", "1"], ["reference", "20260929", "1"]],
        "calendar_attributes.txt": [["service_id", "service_schedule_typicality"],
                                    ["s", "1"], ["reference", "6"]],
        "stop_times.txt": [["trip_id", "stop_sequence", "stop_id", "arrival_time", "departure_time", "pickup_type", "drop_off_type"]]
                          + [[t, i, str(i), f"25:0{i}:00", f"25:0{i}:10", 1 if i == 1 else 0, 1 if i == 5 else 0]
                             for i in [6, 2, 0, 5, 3, 1, 4] for t in ["t2", "ref", "t1"]],
    }


def write_feed(tmp_path, tables):
    path = tmp_path / "feed.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for filename, rows in tables.items():
            stream = io.StringIO()
            csv.writer(stream).writerows(rows)
            archive.writestr(filename, stream.getvalue())
    return path


def test_unsorted_times_collapse_duplicate_trips_preserve_25h_and_skip_reference(tmp_path):
    patterns, stats = extract_gtfs(write_feed(tmp_path, feed_tables()), source="fixture")
    assert len(patterns) == 1
    pattern = patterns[0]
    assert [stop["id"] for stop in pattern["stops"]] == [f"fixture:{i}" for i in range(7)]
    assert pattern["modes"] == ["bus"] * 6
    assert pattern["line_ids"] == ["fixture:r"] * 6
    assert pattern["metadata"]["trip_count"] == 2
    assert set(pattern["metadata"]["trip_ids"]) == {"t1", "t2"}
    assert pattern["metadata"]["representative_arrival_times"][0] == "25:00:00"
    assert pattern["metadata"]["representative_departure_times"][-1] == "25:06:10"
    assert pattern["metadata"]["timetable_validated"] is False
    assert stats["skipped_reference_trips"] == 1
    assert stats["input_stop_times"] == 21


def test_slices_are_contiguous_and_obey_boarding_alighting_restrictions(tmp_path):
    patterns, _ = extract_gtfs(write_feed(tmp_path, feed_tables()), source="fixture")
    slices = make_subjourneys(patterns, min_stops=2, max_per_pattern=20)
    assert slices
    parent = patterns[0]
    for item in slices:
        meta = item["metadata"]
        start, end = meta["slice_start"], meta["slice_end_exclusive"]
        assert item["stops"] == parent["stops"][start:end]
        assert item["modes"] == parent["modes"][start:end - 1]
        assert meta["parent_pattern_id"] == parent["id"]
        assert meta["topology_family_id"] == parent["metadata"]["topology_family_id"]
        assert start != 1 and end - 1 != 5
        assert meta["pickup_types"] == parent["metadata"]["pickup_types"][start:end]
        assert meta["drop_off_types"] == parent["metadata"]["drop_off_types"][start:end]
        assert meta["representative_arrival_times"] == parent["metadata"]["representative_arrival_times"][start:end]


def test_restriction_changes_produce_separate_patterns_with_same_topology(tmp_path):
    tables = feed_tables()
    for row in tables["stop_times.txt"][1:]:
        if row[0] == "t2" and row[1] == 0:
            row[5] = 2
    patterns, _ = extract_gtfs(write_feed(tmp_path, tables), source="fixture")
    assert len(patterns) == 2
    assert patterns[0]["id"] != patterns[1]["id"]
    assert patterns[0]["metadata"]["topology_family_id"] == patterns[1]["metadata"]["topology_family_id"]


@pytest.mark.parametrize("mutation,expected", [
    (lambda t: t["stop_times.txt"].append(t["stop_times.txt"][1].copy()), "Duplicate stop_sequence"),
    (lambda t: t["stop_times.txt"][1].__setitem__(0, "unknown"), "unknown trip"),
    (lambda t: t["stop_times.txt"][1].__setitem__(2, "unknown"), "unknown/unlocated stop"),
    (lambda t: t["stop_times.txt"][1].__setitem__(5, 4), "Invalid sequence/restriction"),
    (lambda t: t["stop_times.txt"][1].__setitem__(3, "25:99:00"), "Invalid GTFS time"),
    (lambda t: t["trips.txt"][1].__setitem__(0, "missing-route"), "unknown route"),
    (lambda t: t["trips.txt"][1].__setitem__(1, "missing-service"), "unknown service"),
    (lambda t: t["stops.txt"][1].__setitem__(2, "nan"), "Coordinates out of bounds"),
    (lambda t: t.pop("routes.txt"), "Expected one routes.txt"),
    (lambda t: t["stops.txt"][0].__setitem__(0, "not_stop_id"), "missing columns"),
])
def test_malformed_feed_fails_with_informative_error(tmp_path, mutation, expected):
    tables = feed_tables()
    mutation(tables)
    with pytest.raises(GTFSDataError, match=expected):
        extract_gtfs(write_feed(tmp_path, tables))


def test_missing_trip_times_fails_instead_of_silently_dropping_trip(tmp_path):
    tables = feed_tables()
    tables["stop_times.txt"] = [row for row in tables["stop_times.txt"] if row[0] != "t2"]
    with pytest.raises(GTFSDataError, match="trips have no stop_times"):
        extract_gtfs(write_feed(tmp_path, tables))


def test_max_patterns_is_stable_and_does_not_change_pattern_identity(tmp_path):
    tables = feed_tables()
    tables["trips.txt"][2][3] = "1"
    path = write_feed(tmp_path, tables)
    all_patterns, stats = extract_gtfs(path)
    limited, limited_stats = extract_gtfs(path, max_patterns=1)
    assert len(all_patterns) == 2 and len(limited) == 1
    assert limited == all_patterns[:1]
    assert limited_stats["unique_patterns_before_limit"] == stats["patterns"] == 2


def test_invalid_zip_fails_with_context(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_text("not a ZIP")
    with pytest.raises(GTFSDataError, match="Cannot open GTFS ZIP"):
        extract_gtfs(path)
