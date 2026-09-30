"""Multi-resolution directed transit tokens with continuous geographic fallback.

Vocabulary comes exclusively from training journeys. Unknown cells map to UNK,
but metric coordinates, heading, length, mode and transfer features remain usable.
The tokenizer never removes intermediate stops or silently truncates a journey.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

from djgtok.schema import MODES, Journey, _snapshot_journey


@dataclass(frozen=True)
class TransitTokenizer:
    """Immutable fitted vocabulary and local geographic feature configuration.

    ID 0 denotes unknown/padding and ID 1 denotes an experimental mask token.
    Coordinates use a local equirectangular approximation, not road map matching.
    Geographic similarity tokens do not prove that two canonical stops coincide.
    """

    origin_lat: float
    origin_lon: float
    vocab: Mapping[str, int] = field(default_factory=dict)
    resolutions: tuple[float, ...] = (200.0, 800.0, 3200.0)
    periods: tuple[float, ...] = (300.0, 900.0, 2700.0, 8100.0, 24300.0, 72900.0)
    version: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported tokenizer version")
        for key, limit in (("origin_lat", 90), ("origin_lon", 180)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if not -limit <= value <= limit:
                raise ValueError(f"{key} is outside geographic bounds")
            object.__setattr__(self, key, float(value))
        for key in ("resolutions", "periods"):
            values = getattr(self, key)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
                raise ValueError(f"{key} must be a nonempty sequence")
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not math.isfinite(value) or value <= 0 for value in values):
                raise ValueError(f"{key} must contain finite positive numbers")
            normalized = tuple(float(value) for value in values)
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{key} must not contain duplicates")
            object.__setattr__(self, key, normalized)
        if not isinstance(self.vocab, Mapping):
            raise ValueError("vocab must be an object mapping tokens to integer IDs")
        vocab = dict(self.vocab)
        if any(not isinstance(key, str) or not key or type(value) is not int for key, value in vocab.items()):
            raise ValueError("vocab must map nonempty strings to integer IDs")
        if sorted(vocab.values()) != list(range(2, len(vocab) + 2)):
            raise ValueError("vocab IDs must be unique and contiguous starting at 2")
        allowed_resolutions = {_resolution_label(value) for value in self.resolutions}
        for key in vocab:
            parts = key.split(":")
            if (len(parts) != 5 or parts[0] not in allowed_resolutions or parts[4] not in MODES
                    or not _integer_text(parts[1]) or not _integer_text(parts[2])
                    or parts[3] not in {str(value) for value in range(16)}):
                raise ValueError(f"Invalid vocabulary token: {key!r}")
        object.__setattr__(self, "vocab", MappingProxyType(vocab))

    @classmethod
    def fit(
        cls, journeys: Sequence[Journey], max_vocab: int = 30000, *,
        resolutions: Sequence[float] = (200.0, 800.0, 3200.0),
        periods: Sequence[float] = (300.0, 900.0, 2700.0, 8100.0, 24300.0, 72900.0),
    ) -> TransitTokenizer:
        if type(max_vocab) is not int or max_vocab < 0:
            raise ValueError("max_vocab must be a nonnegative integer")
        if not journeys:
            raise ValueError("Cannot fit tokenizer on empty journeys")
        journeys = [_snapshot_journey(journey) for journey in journeys]
        stops = [stop for journey in journeys for stop in journey.stops]
        # fsum prevents input-order summation drift from moving a grid origin.
        obj = cls(math.fsum(stop.lat for stop in stops) / len(stops),
                  math.fsum(stop.lon for stop in stops) / len(stops),
                  resolutions=resolutions, periods=periods)
        counts: Counter[str] = Counter()
        for journey in journeys:
            keys, _ = obj._features(journey)
            counts.update(key for row in keys for key in row)
        # UNK/PAD=0; MASK=1. Tie ordering stable across operating systems.
        vocab = {key: i + 2 for i, (key, _) in enumerate(
            sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_vocab])}
        return cls(obj.origin_lat, obj.origin_lon, vocab, obj.resolutions, obj.periods)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 2

    @property
    def feature_dim(self) -> int:
        return len(self.periods) * 4 + 11 + len(MODES)

    def _features(self, journey: Journey) -> tuple[list[list[str]], np.ndarray]:
        journey = _snapshot_journey(journey)
        coords = np.array([(s.lon, s.lat) for s in journey.stops], dtype=np.float64)
        x = (coords[:, 0] - self.origin_lon) * (111320 * math.cos(math.radians(self.origin_lat)))
        y = (coords[:, 1] - self.origin_lat) * 111320
        dx, dy = np.diff(x), np.diff(y)
        length = np.hypot(dx, dy)
        ux, uy = dx / np.maximum(length, 1), dy / np.maximum(length, 1)
        cx, cy = (x[1:] + x[:-1]) / 2, (y[1:] + y[:-1]) / 2
        angle = np.arctan2(dy, dx)
        heading = np.floor((angle + np.pi) / (2 * np.pi) * 16).astype(int) % 16
        keys = []
        for i in range(len(dx)):
            # Location and oriented travel, shared across line IDs; line IDs are not
            # labels to memorize. A new line may share an existing physical corridor.
            try:
                keys.append([
                    f"{_resolution_label(r)}:{math.floor(float(cx[i])/r)}:"
                    f"{math.floor(float(cy[i])/r)}:{heading[i]}:{journey.modes[i]}"
                    for r in self.resolutions
                ])
            except (OverflowError, ValueError) as exc:
                raise ValueError("resolutions are too small for these coordinates") from exc
        columns = []
        for period in self.periods:
            for value in (cx, cy):
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        angle = value / period * 2 * np.pi
                        columns += [np.sin(angle), np.cos(angle)]
                except FloatingPointError as exc:
                    raise ValueError("periods are too small for these coordinates") from exc
        turn_dot = np.r_[1.0, ux[1:] * ux[:-1] + uy[1:] * uy[:-1]]
        turn_cross = np.r_[0.0, ux[:-1] * uy[1:] - uy[:-1] * ux[1:]]
        transfer = np.array([0.0] + [float(journey.line_ids[i] != journey.line_ids[i-1] or journey.modes[i] != journey.modes[i-1]) for i in range(1, len(dx))])
        columns += [np.clip(cx/30000, -10, 10), np.clip(cy/30000, -10, 10),
                    np.clip(dx/3000, -5, 5), np.clip(dy/3000, -5, 5), ux, uy,
                    np.log1p(length)/9, np.linspace(0, 1, len(dx)), turn_dot, turn_cross, transfer]
        columns += [np.array([float(mode == m) for mode in journey.modes]) for m in MODES]
        features = np.stack(columns, axis=1).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError("tokenizer produced nonfinite features")
        return keys, features

    def encode(self, journey: Journey) -> tuple[np.ndarray, np.ndarray]:
        keys, features = self._features(journey)
        ids = np.array([[self.vocab.get(k, 0) for k in row] for row in keys], dtype=np.int64)
        return ids, features

    def inspect(self, journey: Journey) -> dict:
        keys, features = self._features(journey)
        return {"tokens": keys, "token_ids": [[self.vocab.get(k, 0) for k in row] for row in keys],
                "unknown_fraction": sum(k not in self.vocab for row in keys for k in row) / (len(keys)*len(self.resolutions)),
                "edge_count": len(keys), "feature_dim": features.shape[1]}

    def to_dict(self) -> dict:
        """Return an independent, JSON-safe versioned configuration and vocabulary."""
        return {"origin_lat": self.origin_lat, "origin_lon": self.origin_lon,
                "vocab": dict(self.vocab), "resolutions": list(self.resolutions),
                "periods": list(self.periods), "version": self.version}

    @property
    def fingerprint(self) -> str:
        """SHA-256 of canonical JSON including every configuration value and token ID."""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """Atomically write portable JSON; no Torch dependency or pickle format."""
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent,
                                             prefix=f".{dest.name}.", delete=False) as handle:
                temporary = handle.name
                json.dump(self.to_dict(), handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, dest)
        finally:
            if temporary is not None and Path(temporary).exists():
                Path(temporary).unlink()

    @classmethod
    def load(cls, path: str | Path) -> TransitTokenizer:
        """Read and validate a versioned tokenizer JSON file."""
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid tokenizer JSON: {path}") from exc
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, obj: dict) -> TransitTokenizer:
        if not isinstance(obj, dict):
            raise ValueError("tokenizer configuration must be an object")
        if obj.get("version") != 1:
            raise ValueError("Unsupported tokenizer version")
        required = {"origin_lat", "origin_lon", "vocab", "resolutions", "periods", "version"}
        if obj.keys() != required:
            raise ValueError("tokenizer configuration requires exactly: " + ", ".join(sorted(required)))
        return cls(**obj)


def _resolution_label(value: float) -> str:
    return str(int(value)) if value.is_integer() else repr(value)


def _integer_text(value: str) -> bool:
    try:
        return str(int(value)) == value
    except ValueError:
        return False


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate tokenizer JSON key: {key}")
        result[key] = value
    return result
