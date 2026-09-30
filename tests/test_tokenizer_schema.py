"""Contracts for directed transit features, valid crops, and split lineage."""
from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from djgtok import OverlapIndex
from djgtok.schema import (
    Journey,
    Stop,
    assign_split,
    load_journeys,
    reverse_journey,
    save_journeys,
    split_group,
    subjourney,
)
from djgtok.tokenizer import TransitTokenizer


def make_journey(identifier="journey", count=7, line="mbta:test", lat=42.0, lon=-71.0):
    return Journey(id=identifier,
                   stops=[Stop(id=f"s{i}", lat=lat + i * .001, lon=lon + i * .0003)
                          for i in range(count)],
                   modes=["bus"] * (count - 1), line_ids=[line] * (count - 1),
                   source="test", metadata={"parent_pattern_id": "original-pattern"})


def test_directed_tokens_distinguish_reversed_travel():
    forward = make_journey()
    reverse = reverse_journey(forward)
    tokenizer = TransitTokenizer.fit([forward, reverse])
    forward_tokens = tokenizer.inspect(forward)["tokens"]
    reverse_tokens = tokenizer.inspect(reverse)["tokens"]
    assert all(a != b for a, b in zip(forward_tokens, reversed(reverse_tokens), strict=True))
    assert not np.array_equal(tokenizer.encode(forward)[1], tokenizer.encode(reverse)[1][::-1])
    assert reverse.source == "synthetic-direction-stress"
    assert reverse.metadata["synthetic"] is True
    assert reverse.metadata["reversed_from"] == forward.id


def test_shared_physical_corridor_does_not_require_known_line_or_stop_ids():
    original = make_journey()
    other_line = make_journey("other", line="new-provider:new-line")
    for position, stop in enumerate(other_line.stops):
        stop.id = f"another-system-{position}"
    tokenizer = TransitTokenizer.fit([original])
    original_ids, original_features = tokenizer.encode(original)
    other_ids, other_features = tokenizer.encode(other_line)
    np.testing.assert_array_equal(other_ids, original_ids)
    np.testing.assert_array_equal(other_features, original_features)


def test_unknown_geographic_cells_still_have_finite_informative_features():
    known = make_journey()
    unknown = make_journey("unseen-city", lat=37.5, lon=127.0)
    tokenizer = TransitTokenizer.fit([known])
    ids, features = tokenizer.encode(unknown)
    assert np.all(ids == 0)
    assert tokenizer.inspect(unknown)["unknown_fraction"] == 1.0
    assert np.isfinite(features).all()
    assert features.shape == (len(unknown.stops) - 1, tokenizer.feature_dim)
    assert np.any(features != 0)
    assert not np.array_equal(features, tokenizer.encode(known)[1])


def test_zero_length_edges_are_finite_and_keep_all_stops():
    journey = make_journey(count=3)
    journey.stops[1] = journey.stops[0].model_copy(deep=True)
    tokenizer = TransitTokenizer.fit([journey])
    ids, features = tokenizer.encode(journey)
    assert len(ids) == len(features) == 2
    assert np.isfinite(features).all()


def test_tokenizer_roundtrip_preserves_exact_encoding():
    journey = make_journey()
    fitted = TransitTokenizer.fit([journey])
    restored = TransitTokenizer.from_dict(json.loads(json.dumps(fitted.to_dict())))
    for before, after in zip(fitted.encode(journey), restored.encode(journey), strict=True):
        np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="version"):
        TransitTokenizer.from_dict({**fitted.to_dict(), "version": 999})


def test_maximum_stop_count_is_encoded_without_silent_truncation():
    journey = make_journey(count=512)
    tokenizer = TransitTokenizer.fit([journey], max_vocab=64)
    ids, features = tokenizer.encode(journey)
    assert ids.shape == (511, 3)
    assert features.shape == (511, tokenizer.feature_dim)
    with pytest.raises(ValidationError):
        make_journey(count=513)


@pytest.mark.parametrize("changes", [
    {"stops": [{"id": "x", "lat": 42, "lon": -71}]},
    {"modes": []}, {"line_ids": []}, {"modes": ["spaceship"] * 6},
    {"id": ""}, {"id": "x" * 501},
])
def test_invalid_journey_shapes_and_ids_are_rejected(changes):
    with pytest.raises(ValidationError):
        Journey.model_validate(make_journey().model_dump() | changes)


@pytest.mark.parametrize("lat,lon", [(float("nan"), 0), (0, float("inf")), (91, 0), (0, -181)])
def test_nonfinite_and_out_of_range_coordinates_are_rejected(lat, lon):
    with pytest.raises(ValidationError):
        Stop(id="bad", lat=lat, lon=lon)


def test_crop_preserves_contiguous_order_lineage_and_aligned_gtfs_metadata():
    journey = make_journey()
    journey.metadata.update({
        "pickup_types": [0, 1, 2, 3, 0, 1, 0],
        "drop_off_types": [1, 0, 0, 0, 2, 3, 0],
        "representative_arrival_times": [f"25:0{i}:00" for i in range(7)],
        "representative_departure_times": [f"25:0{i}:20" for i in range(7)],
    })
    crop = subjourney(journey, 2, 6)
    assert [stop.id for stop in crop.stops] == ["s2", "s3", "s4", "s5"]
    assert crop.modes == journey.modes[2:5]
    assert crop.line_ids == journey.line_ids[2:5]
    assert crop.metadata["parent_journey_id"] == journey.id
    assert crop.metadata["parent_pattern_id"] == "original-pattern"
    assert crop.metadata["stop_range"] == [2, 6]
    for field in ("pickup_types", "drop_off_types", "representative_arrival_times", "representative_departure_times"):
        assert crop.metadata[field] == journey.metadata[field][2:6]


def test_every_schema_accepted_id_remains_usable_for_index_and_transforms():
    journey = make_journey(identifier="x" * 500)
    crop = subjourney(journey, 1, 5)
    reverse = reverse_journey(journey)
    assert len(crop.id) <= 500 and len(reverse.id) <= 500
    assert crop.id != reverse.id != journey.id
    assert crop.id == subjourney(journey, 1, 5).id
    index = OverlapIndex.build([journey])
    assert index.search(journey)[0].journey_id == journey.id


def test_source_row_evidence_stays_aligned_after_crop_and_synthetic_reverse():
    journey = make_journey()
    rows = [19, 35, 48, 103, 160, 207, 320]
    journey.metadata.update(source_rows=rows, source_reference={"sheet": "Data", "original_rows": rows})
    crop = subjourney(subjourney(journey, 1, 6), 1, 4)
    assert crop.metadata["source_rows"] == crop.metadata["source_reference"]["original_rows"] == rows[2:5]
    reverse = reverse_journey(crop)
    assert reverse.metadata["source_rows"] == reverse.metadata["source_reference"]["original_rows"] == rows[2:5][::-1]
    assert reverse.metadata["synthetic"] is True
    assert journey.metadata["source_rows"] == journey.metadata["source_reference"]["original_rows"] == rows


def test_line_split_keeps_service_variants_directions_and_crops_together():
    parent = make_journey()
    variants = [parent, reverse_journey(parent), subjourney(parent, 2, 6),
                make_journey(identifier="different-service-day", count=5)]
    assert len({split_group(item) for item in variants}) == 1
    assert len({assign_split(item) for item in variants}) == 1


def test_multiline_crop_retains_parent_split_group():
    parent = make_journey()
    parent.line_ids = ["line-A"] * 3 + ["line-B"] * 3
    child = subjourney(parent, 0, 3)
    assert split_group(child) == split_group(parent)
    assert assign_split(child) == assign_split(parent)


def test_loader_reports_duplicate_ids_and_bad_line_number(tmp_path):
    path = tmp_path / "journeys.jsonl"
    journey = make_journey()
    path.write_text(journey.model_dump_json() + "\n" + journey.model_dump_json() + "\n")
    with pytest.raises(ValueError, match=r":2:.*duplicate journey ID"):
        load_journeys(path)


@pytest.mark.parametrize("transform", [lambda item: subjourney(item, 1, 5), reverse_journey])
def test_transform_does_not_alias_stop_extras_or_nested_metadata(transform):
    parent = make_journey()
    parent.stops[2].details = {"labels": ["original"]}
    parent.metadata["nested"] = {"labels": ["original"]}
    derived = transform(parent)
    derived.metadata["nested"]["labels"].append("changed")
    copied_stop = next(stop for stop in derived.stops if stop.id == "s2")
    copied_stop.details["labels"].append("changed")
    copied_stop.id = "new-id"
    assert parent.metadata["nested"] == {"labels": ["original"]}
    assert parent.stops[2].details == {"labels": ["original"]}
    assert parent.stops[2].id == "s2"


@pytest.mark.parametrize("start,end", [(True, 4), (1.0, 4), (0, False), (-1, 3), (0, 1), (2, 20)])
def test_invalid_slice_indices_are_rejected(start, end):
    with pytest.raises(ValueError):
        subjourney(make_journey(), start, end)


@pytest.mark.parametrize("metadata", [
    {"source_rows": None}, {"source_rows": [1]},
    {"source_rows": [True] * 7}, {"source_rows": [0] * 7},
    {"source_rows": list(range(1, 8)), "source_reference": {"original_rows": list(range(2, 9))}},
    {"source_reference": "not-an-object"}, {"slice_start": "0"},
    {"nested": {"bad": float("nan")}}, {"nested": {1: "not-a-string-key"}},
    {"python_object": object()},
])
def test_metadata_rejects_unaligned_evidence_and_non_json_values(metadata):
    with pytest.raises(ValidationError):
        Journey.model_validate(make_journey().model_dump() | {"metadata": metadata})


@pytest.mark.parametrize("value", [True, "37.5", None])
def test_coordinates_do_not_coerce_boolean_string_or_null(value):
    with pytest.raises(ValidationError):
        Stop(id="stop", lat=value, lon=127)


def test_mutated_journey_is_revalidated_at_core_boundaries():
    journey = make_journey()
    tokenizer = TransitTokenizer.fit([journey])
    journey.modes.pop()
    for operation in (lambda: tokenizer.encode(journey), lambda: subjourney(journey, 1, 3),
                      lambda: reverse_journey(journey), lambda: OverlapIndex([journey])):
        with pytest.raises(ValidationError):
            operation()


@pytest.mark.parametrize("changes", [
    {"origin_lat": float("nan")}, {"origin_lon": float("inf")},
    {"origin_lat": 91}, {"origin_lon": -181}, {"origin_lat": True},
    {"resolutions": []}, {"resolutions": [0]}, {"resolutions": [-1]},
    {"resolutions": [float("nan")]}, {"resolutions": [True]},
    {"resolutions": [200, 200]}, {"periods": None}, {"periods": [float("inf")]},
    {"periods": [0]}, {"version": True}, {"vocab": {"invalid": 2}},
    {"vocab": {"200:0:0:8:bus": 3}}, {"vocab": {"200:0:0:8:bus": True}},
    {"unknown_option": 1},
])
def test_serialized_tokenizer_rejects_invalid_configuration(changes):
    tokenizer = TransitTokenizer(37.5, 127.0)
    with pytest.raises(ValueError):
        TransitTokenizer.from_dict(tokenizer.to_dict() | changes)


@pytest.mark.parametrize("max_vocab", [-1, True, 1.5, None])
def test_fit_rejects_invalid_vocabulary_limits(max_vocab):
    with pytest.raises(ValueError, match="max_vocab"):
        TransitTokenizer.fit([make_journey()], max_vocab=max_vocab)


def test_tokenizer_json_persistence_and_fingerprint_include_all_configuration(tmp_path):
    journey = make_journey()
    fitted = TransitTokenizer.fit([journey], resolutions=(200.25, 200.75), periods=(300, 900))
    path = tmp_path / "nested" / "tokenizer.json"
    fitted.save(path)
    restored = TransitTokenizer.load(path)
    assert restored.to_dict() == fitted.to_dict()
    assert len(restored.fingerprint) == 64
    assert restored.fingerprint == fitted.fingerprint
    for before, after in zip(fitted.encode(journey), restored.encode(journey), strict=True):
        np.testing.assert_array_equal(before, after)
    assert fitted.encode(journey)[0].shape == (6, 2)
    assert fitted.feature_dim == 26
    labels = fitted.inspect(journey)["tokens"][0]
    assert labels[0].startswith("200.25:") and labels[1].startswith("200.75:")
    changed = TransitTokenizer.from_dict(fitted.to_dict() | {"origin_lat": fitted.origin_lat + .01})
    assert changed.fingerprint != fitted.fingerprint
    payload = fitted.to_dict()
    payload["vocab"].clear()
    assert fitted.vocab
    with pytest.raises(TypeError):
        fitted.vocab["new"] = 20
    with pytest.raises(AttributeError):
        fitted.origin_lon = 0


def test_tokenizer_load_rejects_duplicate_json_keys_and_non_object_payload(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text('{"version":1,"version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        TransitTokenizer.load(path)
    path.write_text("null", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        TransitTokenizer.load(path)


def test_vocab_fit_ties_are_deterministic_and_zero_limit_is_supported():
    a, b = make_journey(), make_journey("different-place", lat=43)
    first = TransitTokenizer.fit([a, b], max_vocab=7)
    second = TransitTokenizer.fit([b, a], max_vocab=7)
    assert first.to_dict() == second.to_dict()
    empty_vocab = TransitTokenizer.fit([a], max_vocab=0)
    assert empty_vocab.vocab_size == 2
    assert np.all(empty_vocab.encode(a)[0] == 0)


def test_stop_source_rows_cannot_contradict_metadata_evidence():
    journey = make_journey()
    journey.stops[0].source_row = 999
    payload = journey.model_dump()
    payload["metadata"]["source_rows"] = list(range(1, 8))
    with pytest.raises(ValidationError, match="must agree"):
        Journey.model_validate(payload)


def test_jsonl_save_validates_before_replacing_existing_file(tmp_path):
    path = tmp_path / "journeys.jsonl"
    journey = make_journey()
    save_journeys([journey], path)
    before = path.read_bytes()
    assert load_journeys(path)[0] == journey
    with pytest.raises(ValueError, match="duplicate"):
        save_journeys([journey, journey], path)
    with pytest.raises(ValueError, match="empty"):
        save_journeys([], path)
    journey.modes.pop()
    with pytest.raises(ValidationError):
        save_journeys([journey], path)
    assert path.read_bytes() == before
