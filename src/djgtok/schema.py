"""Validated journey interchange format; split BEFORE deriving training views."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Mode = Literal["walk", "bus", "rail", "subway", "tram", "ferry", "other"]
MODES = ("walk", "bus", "rail", "subway", "tram", "ferry", "other")
PARENT_PATTERN_KINDS = frozenset({"planned_gtfs_pattern", "planned_seoul_bus_pattern"})


def is_parent_pattern(journey: Journey, *, require_source_kind: bool = False) -> bool:
    kind = journey.metadata.get("kind")
    return kind in PARENT_PATTERN_KINDS or (kind is None and not require_source_kind)


class Stop(BaseModel):
    """A stop in a provider's canonical ID namespace, at WGS84 coordinates."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False, validate_assignment=True)
    id: str = Field(min_length=1, max_length=200, strict=True)
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)

    @field_validator("id")
    @classmethod
    def canonical_id(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("stop ID must be nonblank without surrounding whitespace")
        return value

    @field_validator("lat", "lon", mode="before")
    @classmethod
    def numeric_coordinate(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("coordinates must be numbers, not strings or booleans")
        return value

    @model_validator(mode="after")
    def json_extras(self) -> Stop:
        _require_json(self.model_extra or {}, "stop extras")
        if "source_row" in (self.model_extra or {}):
            row = self.model_extra["source_row"]
            if type(row) is not int or row < 1:
                raise ValueError("source_row must be a positive integer")
        return self


class Journey(BaseModel):
    """Ordered stops and one mode/line ID for each consecutive directed edge."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    id: str = Field(min_length=1, max_length=500, strict=True)
    stops: list[Stop] = Field(min_length=2, max_length=512)
    modes: list[Mode]
    line_ids: list[str]
    source: str = "user"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id", "source")
    @classmethod
    def nonblank_identifier(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("identifiers must be nonblank without surrounding whitespace")
        return value

    @field_validator("line_ids")
    @classmethod
    def nonblank_lines(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or value != value.strip() for value in values):
            raise ValueError("line IDs must be nonblank without surrounding whitespace")
        return values

    @model_validator(mode="after")
    def aligned_edges(self) -> Journey:
        if len(self.modes) != len(self.stops) - 1 or len(self.line_ids) != len(self.modes):
            raise ValueError("modes and line_ids must have exactly len(stops)-1 entries")
        _require_json(self.metadata, "metadata")
        for key in _STOP_METADATA:
            if key in self.metadata:
                _validate_stop_values(self.metadata[key], key, len(self.stops))
        if "source_rows" in self.metadata:
            for stop, row in zip(self.stops, self.metadata["source_rows"], strict=True):
                if "source_row" in (stop.model_extra or {}) and stop.model_extra["source_row"] != row:
                    raise ValueError("stop source_row and metadata.source_rows must agree")
        reference = self.metadata.get("source_reference")
        if reference is not None and not isinstance(reference, dict):
            raise ValueError("source_reference must be an object")
        if isinstance(reference, dict) and "original_rows" in reference:
            rows = reference["original_rows"]
            _validate_stop_values(rows, "original_rows", len(self.stops))
            if "source_rows" in self.metadata and rows != self.metadata["source_rows"]:
                raise ValueError("source_rows and source_reference.original_rows must agree")
        if "slice_start" in self.metadata:
            value = self.metadata["slice_start"]
            if type(value) is not int or value < 0:
                raise ValueError("slice_start must be a nonnegative integer")
        return self


_STOP_METADATA = (
    "pickup_types", "drop_off_types", "representative_arrival_times",
    "representative_departure_times", "source_rows",
)


def _require_json(value: Any, label: str) -> None:
    """Reject Python-only objects and nonfinite values before JSON persistence."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} must contain string object keys")
        for child in value.values():
            _require_json(child, label)
    elif isinstance(value, list):
        for child in value:
            _require_json(child, label)
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError(f"{label} must contain JSON values only")
    elif isinstance(value, float) and not (-float("inf") < value < float("inf")):
        raise ValueError(f"{label} must contain finite numbers")


def _validate_stop_values(values: Any, label: str, count: int) -> None:
    if not isinstance(values, list) or len(values) != count:
        raise ValueError(f"{label} must have exactly len(stops) entries")
    if label in ("source_rows", "original_rows"):
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError(f"{label} must contain positive integer source row numbers")


def _snapshot_journey(journey: Journey) -> Journey:
    """Revalidate mutable containers, then isolate all stops/extras/metadata."""
    if not isinstance(journey, Journey):
        raise TypeError("expected a Journey instance")
    return Journey.model_validate(deepcopy(journey.model_dump()))


def subjourney(journey: Journey, start: int, end: int) -> Journey:
    """Return an independent slice using half-open STOP indices, preserving evidence."""
    journey = _snapshot_journey(journey)
    if type(start) is not int or type(end) is not int:
        raise ValueError("subjourney indices must be integers, not booleans")
    if not 0 <= start < end <= len(journey.stops) or end - start < 2:
        raise ValueError("subjourney requires at least two consecutive stops")
    meta = {**journey.metadata, "parent_journey_id": journey.id, "stop_range": [start, end],
            "split_group": split_group(journey)}
    for key in _STOP_METADATA:
        if key in meta:
            meta[key] = meta[key][start:end]
    if isinstance(meta.get("source_reference"), dict) and "original_rows" in meta["source_reference"]:
        meta["source_reference"] = {**meta["source_reference"],
                                    "original_rows": meta["source_reference"]["original_rows"][start:end]}
    if "slice_start" in meta:
        base = meta["slice_start"]
        meta["slice_start"], meta["slice_end_exclusive"] = base + start, base + end
    if meta.get("pickup_types"):
        meta["conditional_boarding"] = meta["pickup_types"][0] in (2, 3)
    if meta.get("drop_off_types"):
        meta["conditional_alighting"] = meta["drop_off_types"][-1] in (2, 3)
    return Journey(id=derived_id(journey.id, f"[{start}:{end}]"), stops=journey.stops[start:end],
                   modes=journey.modes[start:end-1], line_ids=journey.line_ids[start:end-1],
                   source=journey.source, metadata=meta)


def reverse_journey(journey: Journey) -> Journey:
    """Synthetic direction stress case, NOT a claim of an operating reverse service."""
    journey = _snapshot_journey(journey)
    meta = {**journey.metadata, "synthetic": True, "reversed_from": journey.id,
            "split_group": split_group(journey), "timetable_validated": False,
            "kind": "synthetic_direction_stress"}
    # These times/boarding permissions describe the real forward service. They
    # must never make this synthetic reverse look like a scheduled service.
    for key in ("pickup_types", "drop_off_types", "representative_arrival_times", "representative_departure_times"):
        meta.pop(key, None)
    for key in ("conditional_boarding", "conditional_alighting", "slice_start", "slice_end_exclusive", "stop_range"):
        meta.pop(key, None)
    if "source_rows" in meta:
        meta["source_rows"] = list(reversed(meta["source_rows"]))
    if isinstance(meta.get("source_reference"), dict) and "original_rows" in meta["source_reference"]:
        meta["source_reference"] = {**meta["source_reference"],
                                    "original_rows": list(reversed(meta["source_reference"]["original_rows"]))}
    return Journey(id=derived_id(journey.id, "", prefix="reverse:"), stops=list(reversed(journey.stops)),
                   modes=list(reversed(journey.modes)), line_ids=list(reversed(journey.line_ids)),
                   source="synthetic-direction-stress",
                   metadata=meta)


def derived_id(original: str, suffix: str, prefix: str = "") -> str:
    value = prefix + original + suffix
    return value if len(value) <= 500 else prefix + original[:380] + ":" + hashlib.sha256(value.encode()).hexdigest() + suffix


def split_group(journey: Journey) -> str:
    # Route line groups hold ALL variants, service days, directions and slices together.
    return str(journey.metadata.get("split_group") or "|".join(sorted(set(journey.line_ids))))


def assign_split(journey: Journey) -> str:
    bucket = int(hashlib.sha256(("transitvec-v1:" + split_group(journey)).encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def load_journeys(path: str | Path) -> list[Journey]:
    journeys, ids = [], set()
    with Path(path).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                journey = Journey.model_validate_json(line)
                if journey.id in ids:
                    raise ValueError(f"duplicate journey ID {journey.id}")
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
            journeys.append(journey)
            ids.add(journey.id)
    if not journeys:
        raise ValueError("Journey dataset is empty")
    return journeys


def save_journeys(journeys: list[Journey], path: str | Path) -> None:
    """Validate and atomically save nonempty, unique-ID JSONL snapshots."""
    snapshots = [_snapshot_journey(journey) for journey in journeys]
    if not snapshots:
        raise ValueError("Journey dataset is empty")
    if len({journey.id for journey in snapshots}) != len(snapshots):
        raise ValueError("duplicate journey ID in dataset")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent,
                                         prefix=f".{dest.name}.", delete=False) as handle:
            temporary = handle.name
            for journey in snapshots:
                handle.write(json.dumps(journey.model_dump(), ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temporary, dest)
    finally:
        if temporary is not None and Path(temporary).exists():
            Path(temporary).unlink()
