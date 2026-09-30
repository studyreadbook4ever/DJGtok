"""Standard-library GTFS ingestion into directed, planned transit journeys.

This is a pattern importer, not a timetable router. It never creates transfers.
"""
from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import math
import sqlite3
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

Journey = dict[str, Any]
MODES = {0: "tram", 1: "subway", 2: "rail", 3: "bus", 4: "ferry",
         5: "other", 6: "other", 7: "other", 11: "bus", 12: "rail"}


class GTFSDataError(ValueError):
    """The input cannot safely be represented as directed stop patterns."""


def route_mode(route_type: str | int) -> str:
    """Map core and common extended GTFS route types to a small mode set."""
    try:
        value = int(route_type)
    except (TypeError, ValueError) as exc:
        raise GTFSDataError(f"Invalid route_type: {route_type!r}") from exc
    if value in MODES:
        return MODES[value]
    for low, high, mode in ((100, 199, "rail"), (200, 299, "bus"),
                            (400, 499, "subway"), (700, 899, "bus"),
                            (900, 999, "tram"), (1000, 1099, "ferry")):
        if low <= value <= high:
            return mode
    return "other"


def _rows(archive: zipfile.ZipFile, name: str, required: Iterable[str] = (),
          optional: bool = False) -> Iterator[dict[str, str]]:
    matches = [entry for entry in archive.infolist()
               if Path(entry.filename).name == name and not entry.is_dir()]
    if not matches and optional:
        return
    if len(matches) != 1:
        raise GTFSDataError(f"Expected one {name}; found {len(matches)}")
    entry = matches[0]
    if entry.file_size > 2_000_000_000:
        raise GTFSDataError(f"{name} exceeds the 2 GB per-table safety limit")
    with archive.open(entry) as binary:
        reader = csv.DictReader(io.TextIOWrapper(binary, encoding="utf-8-sig", newline=""))
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise GTFSDataError(f"{name} missing columns: {', '.join(sorted(missing))}")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise GTFSDataError(f"{name}:{line}: malformed CSV row")
            yield row


def _unique(rows: Iterable[dict[str, str]], key: str, filename: str) -> dict[str, dict[str, str]]:
    result = {}
    for row in rows:
        value = row[key]
        if not value or value in result:
            raise GTFSDataError(f"{filename}: missing or duplicate {key} {value!r}")
        result[value] = row
    return result


def _time(value: str, label: str) -> None:
    if not value:
        return  # GTFS permits missing intermediate, non-timepoint times.
    try:
        h, m, s = (int(part) for part in value.split(":"))
        if not (h >= 0 and 0 <= m < 60 and 0 <= s < 60):
            raise ValueError("time component out of range")
    except ValueError as exc:
        raise GTFSDataError(f"Invalid GTFS time at {label}: {value!r}") from exc


def extract_gtfs(feed: str | Path, *, source: str = "gtfs", max_patterns: int | None = None,
                 min_stops: int = 3) -> tuple[list[Journey], dict[str, Any]]:
    """Read GTFS, collapse repeated trips and return patterns plus diagnostics.

    Pattern identity includes route, direction, ordered stops and pickup/dropoff
    rules. A separate topology_family_id ignores route and restrictions to allow
    stricter leakage-resistant splits. max_patterns samples by stable hash only
    after deduplication, avoiding a first-route selection bias. All tables are
    streamed; the largest table is sorted in a temporary on-disk SQLite file.
    """
    if max_patterns is not None and max_patterns < 1:
        raise ValueError("max_patterns must be positive or None")
    if min_stops < 2:
        raise ValueError("min_stops must be >= 2")
    feed = Path(feed)
    try:
        archive = zipfile.ZipFile(feed)
    except (OSError, zipfile.BadZipFile) as exc:
        raise GTFSDataError(f"Cannot open GTFS ZIP {feed}: {exc}") from exc
    with archive, tempfile.TemporaryDirectory(prefix="djgtok-gtfs-") as tmp:
        stops = _unique(_rows(archive, "stops.txt", ("stop_id", "stop_lat", "stop_lon")),
                        "stop_id", "stops.txt")
        routes = _unique(_rows(archive, "routes.txt", ("route_id", "route_type")),
                         "route_id", "routes.txt")
        trips = _unique(_rows(archive, "trips.txt", ("trip_id", "route_id", "service_id")),
                        "trip_id", "trips.txt")
        reference_services = {row["service_id"] for row in _rows(
            archive, "calendar_attributes.txt", optional=True)
            if row.get("service_schedule_typicality") == "6"}
        feed_info = list(_rows(archive, "feed_info.txt", optional=True))
        calendar = list(_rows(archive, "calendar.txt", optional=True))
        exceptions = list(_rows(archive, "calendar_dates.txt", optional=True))
        service_ids = {row["service_id"] for row in calendar + exceptions}
        if not service_ids:
            raise GTFSDataError("Need calendar.txt or calendar_dates.txt service definitions")
        coordinates = {}
        for stop_id, row in stops.items():
            if not row["stop_lat"] or not row["stop_lon"]:
                continue  # Non-platform nodes may legally omit coordinates.
            try:
                lat, lon = float(row["stop_lat"]), float(row["stop_lon"])
            except ValueError as exc:
                raise GTFSDataError(f"Invalid coordinates for stop {stop_id}") from exc
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise GTFSDataError(f"Coordinates out of bounds for stop {stop_id}")
            coordinates[stop_id] = {"id": f"{source}:{stop_id}", "lat": lat, "lon": lon,
                                    "name": row.get("stop_name", ""),
                                    "parent_station": row.get("parent_station", "")}
        for trip_id, trip in trips.items():
            if trip["route_id"] not in routes:
                raise GTFSDataError(f"Trip {trip_id} references unknown route {trip['route_id']}")
            if trip["service_id"] not in service_ids:
                raise GTFSDataError(f"Trip {trip_id} references unknown service {trip['service_id']}")
        database = sqlite3.connect(str(Path(tmp) / "times.sqlite"))
        try:
            database.execute("PRAGMA journal_mode=OFF")
            database.execute("PRAGMA synchronous=OFF")
            database.execute("CREATE TABLE times (trip TEXT, seq INTEGER, stop TEXT, arrival TEXT, departure TEXT, pickup INTEGER, dropoff INTEGER)")
            row_count, batch = 0, []
            for row in _rows(archive, "stop_times.txt", ("trip_id", "stop_id", "stop_sequence", "arrival_time", "departure_time")):
                trip_id, stop_id = row["trip_id"], row["stop_id"]
                if trip_id not in trips:
                    raise GTFSDataError(f"stop_times references unknown trip {trip_id}")
                if stop_id not in coordinates:
                    raise GTFSDataError(f"Trip {trip_id} references unknown/unlocated stop {stop_id}")
                try:
                    seq = int(row["stop_sequence"])
                    pickup, dropoff = int(row.get("pickup_type") or 0), int(row.get("drop_off_type") or 0)
                except ValueError as exc:
                    raise GTFSDataError(f"Invalid sequence/restriction on trip {trip_id}") from exc
                if seq < 0 or pickup not in range(4) or dropoff not in range(4):
                    raise GTFSDataError(f"Invalid sequence/restriction on trip {trip_id}")
                _time(row["arrival_time"], trip_id)
                _time(row["departure_time"], trip_id)
                batch.append((trip_id, seq, stop_id, row["arrival_time"], row["departure_time"], pickup, dropoff))
                row_count += 1
                if len(batch) >= 20_000:
                    database.executemany("INSERT INTO times VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
            if batch:
                database.executemany("INSERT INTO times VALUES (?,?,?,?,?,?,?)", batch)
            database.commit()
            database.execute("CREATE INDEX by_trip ON times (trip, seq)")
            patterns: dict[str, Journey] = {}
            skipped_short = skipped_reference = parsed_trips = 0
            cursor = database.execute("SELECT trip, seq, stop, arrival, departure, pickup, dropoff FROM times ORDER BY trip, seq")
            for trip_id, group in itertools.groupby(cursor, key=lambda row: row[0]):
                rows = list(group)
                trip = trips[trip_id]
                parsed_trips += 1
                if len({row[1] for row in rows}) != len(rows):
                    raise GTFSDataError(f"Duplicate stop_sequence within trip {trip_id}")
                if trip["service_id"] in reference_services:
                    skipped_reference += 1
                    continue
                if len(rows) < min_stops:
                    skipped_short += 1
                    continue
                route_id = trip["route_id"]
                route = routes[route_id]
                stop_ids = [row[2] for row in rows]
                pickup, dropoff = [row[5] for row in rows], [row[6] for row in rows]
                identity = [route_id, trip.get("direction_id", ""), stop_ids, pickup, dropoff]
                digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()[:20]
                pattern_id = f"{source}:pattern:{digest}"
                if pattern_id in patterns:
                    metadata = patterns[pattern_id]["metadata"]
                    metadata["trip_count"] += 1
                    metadata["trip_ids"].append(trip_id)
                    if trip["service_id"] not in metadata["service_ids"]:
                        metadata["service_ids"].append(trip["service_id"])
                    continue
                topology_hash = hashlib.sha256(json.dumps(stop_ids, separators=(",", ":")).encode()).hexdigest()[:20]
                mode = route_mode(route["route_type"])
                metadata = {"kind": "planned_gtfs_pattern", "parent_pattern_id": pattern_id,
                            "topology_family_id": f"{source}:topology:{topology_hash}",
                            "route_id": route_id, "route_name": route.get("route_short_name") or route.get("route_long_name", route_id),
                            "direction_id": trip.get("direction_id", ""), "headsign": trip.get("trip_headsign", ""),
                            "representative_trip_id": trip_id, "trip_ids": [trip_id], "trip_count": 1,
                            "service_ids": [trip["service_id"]], "pickup_types": pickup,
                            "drop_off_types": dropoff, "representative_arrival_times": [row[3] for row in rows],
                            "representative_departure_times": [row[4] for row in rows],
                            "time_basis": "GTFS service-day local time; may exceed 24:00:00",
                            "timetable_validated": False}
                patterns[pattern_id] = {"id": pattern_id, "stops": [coordinates[s].copy() for s in stop_ids],
                                        "modes": [mode] * (len(rows) - 1),
                                        "line_ids": [f"{source}:{route_id}"] * (len(rows) - 1),
                                        "source": source, "metadata": metadata}
        finally:
            database.close()
        if parsed_trips != len(trips):
            raise GTFSDataError(f"{len(trips) - parsed_trips} trips have no stop_times")
        selected = sorted(patterns.values(), key=lambda journey: journey["id"])
        if max_patterns:
            selected = selected[:max_patterns]
        if not selected:
            raise GTFSDataError("No patterns remain after minimum-length and reference-service filters")
        report = {"source": source, "feed_sha256": hashlib.sha256(feed.read_bytes()).hexdigest(),
                  "feed_info": feed_info, "input_stops": len(stops), "input_routes": len(routes),
                  "input_trips": len(trips), "input_stop_times": row_count,
                  "skipped_short_trips": skipped_short, "skipped_reference_trips": skipped_reference,
                  "unique_patterns_before_limit": len(patterns), "patterns": len(selected),
                  "modes": dict(Counter(journey["modes"][0] for journey in selected)),
                  "calendar_service_count": len(service_ids),
                  "semantics": "Planned single-vehicle patterns; service date/transfer feasibility not evaluated"}
        return selected, report


def make_subjourneys(patterns: Iterable[Journey], *, min_stops: int = 4,
                     max_per_pattern: int = 6) -> list[Journey]:
    """Deterministic contiguous slices with permitted GTFS boarding/alighting.

    Parent IDs remain attached. Split parents BEFORE using these augmentations.
    Pickup/dropoff 2 or 3 are conditional service, preserved in metadata.
    """
    if min_stops < 2 or max_per_pattern < 0:
        raise ValueError("min_stops must be >=2 and max_per_pattern nonnegative")
    result = []
    for pattern in patterns:
        n = len(pattern["stops"])
        meta = pattern["metadata"]
        pickup, dropoff = meta["pickup_types"], meta["drop_off_types"]
        candidates = set()
        for fraction in (0.35, 0.5, 0.7):
            length = max(min_stops, round(n * fraction))
            if length >= n:
                continue
            for start in {0, (n - length) // 2, n - length}:
                end = start + length
                if pickup[start] != 1 and dropoff[end - 1] != 1:
                    candidates.add((start, end))
        candidates = sorted(candidates, key=lambda pair: hashlib.sha256(f"{pattern['id']}:{pair}".encode()).digest())
        for start, end in candidates[:max_per_pattern]:
            metadata = {key: value for key, value in meta.items()
                        if key not in {"trip_ids", "representative_arrival_times", "representative_departure_times", "pickup_types", "drop_off_types"}}
            metadata.update({"kind": "contiguous_gtfs_subjourney", "slice_start": start,
                             "slice_end_exclusive": end, "pickup_types": pickup[start:end],
                             "drop_off_types": dropoff[start:end],
                             "conditional_boarding": pickup[start] in (2, 3),
                             "conditional_alighting": dropoff[end - 1] in (2, 3),
                             "representative_arrival_times": meta["representative_arrival_times"][start:end],
                             "representative_departure_times": meta["representative_departure_times"][start:end]})
            result.append({"id": f"{pattern['id']}:slice:{start}:{end}",
                           "stops": pattern["stops"][start:end],
                           "modes": pattern["modes"][start:end - 1],
                           "line_ids": pattern["line_ids"][start:end - 1],
                           "source": pattern["source"], "metadata": metadata})
            result[-1] = deepcopy(result[-1])
    return result


def write_journeys(path: str | Path, journeys: Iterable[Journey]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as output:
        for journey in journeys:
            output.write(json.dumps(journey, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_journeys(path: str | Path) -> list[Journey]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
