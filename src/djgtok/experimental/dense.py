"""Exact dense retrieval over full journeys and overlapping directed segments.

Scores are cosine similarities, not probabilities. Segment retrieval compares
every query window to every indexed window before aggregating by journey.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, Field

from djgtok.schema import Journey, subjourney

if TYPE_CHECKING:
    from djgtok.experimental.model import EncoderBundle

SearchMode = Literal["whole", "shared"]
_FORMAT_VERSION = 1


class SearchHit(BaseModel):
    journey_id: str
    score: float = Field(description="Cosine similarity; this is not a probability.")
    query_stop_range: tuple[int, int] = Field(
        description=(
            "Half-open query stop indices [start, end). In shared mode, this is the query "
            "window in the highest-scoring cosine pair, not confirmed stop alignment. "
            "In whole mode, this covers the full query."
        )
    )
    document_stop_range: tuple[int, int] = Field(
        description=(
            "Half-open indexed-journey stop indices [start, end). In shared mode, this is "
            "the document window in the highest-scoring cosine pair, not confirmed stop "
            "alignment. In whole mode, this covers the full indexed journey."
        )
    )


def _window_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if not values or any(type(value) is not int or value < 1 for value in values):
        raise ValueError("window_sizes must contain positive integer edge counts")
    return tuple(sorted(set(values)))


def _ranges(stop_count: int, sizes: Sequence[int]) -> list[tuple[int, int]]:
    """Whole journey followed by deduplicated windows, including the tail."""
    if stop_count < 2:
        raise ValueError("A journey must contain at least two stops")
    result = [(0, stop_count)]
    seen = set(result)
    edge_count = stop_count - 1
    for size in sizes:
        if size >= edge_count:
            continue
        starts = list(range(0, edge_count - size + 1, max(1, size // 2)))
        if starts[-1] != edge_count - size:
            starts.append(edge_count - size)
        for start in starts:
            window = (start, start + size + 1)
            if window not in seen:
                seen.add(window)
                result.append(window)
    return result


def _checked_vectors(values: object, rows: int, dimension: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] < 1:
        raise ValueError("Encoder/index vectors have invalid dimensions")
    if dimension is not None and result.shape[1] != dimension:
        raise ValueError("Encoder dimension does not match the index")
    if result.dtype != np.float32:
        raise ValueError("Encoder/index vectors must have dtype float32")
    if not np.isfinite(result).all():
        raise ValueError("Encoder/index vectors contain non-finite values")
    norms = np.linalg.norm(result.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-4):
        raise ValueError("Encoder/index vectors must be unit-normalized")
    return result


def _fingerprint(bundle: EncoderBundle) -> str:
    value = bundle.fingerprint
    if not isinstance(value, str) or not value:
        raise ValueError("Encoder must expose a nonempty checkpoint fingerprint")
    if value.strip().lower() == "unsaved":
        raise ValueError("Save the encoder checkpoint before building or querying an index")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DenseIndex:
    """An immutable corpus and normalized vectors bound to one encoder."""

    def __init__(
        self,
        journeys: Sequence[Journey],
        vectors: np.ndarray,
        chunks: Sequence[tuple[int, int, int]],
        model_fingerprint: str,
        window_sizes: Sequence[int] = (4, 8, 16),
    ) -> None:
        if not journeys:
            raise ValueError("Cannot build or load an empty index")
        self.window_sizes = _window_sizes(window_sizes)
        self._journeys = tuple(journey.model_copy(deep=True) for journey in journeys)
        self._by_id = {journey.id: position for position, journey in enumerate(self._journeys)}
        if len(self._by_id) != len(self._journeys):
            raise ValueError("Journey IDs must be unique")
        expected = [
            (position, start, end)
            for position, journey in enumerate(self._journeys)
            for start, end in _ranges(len(journey.stops), self.window_sizes)
        ]
        if list(chunks) != expected:
            raise ValueError("Index chunk references do not match the corpus and window policy")
        if not isinstance(model_fingerprint, str) or not model_fingerprint:
            raise ValueError("Index model fingerprint is missing")
        if model_fingerprint.strip().lower() == "unsaved":
            raise ValueError("An index requires a saved encoder checkpoint fingerprint")
        self.model_fingerprint = model_fingerprint
        self._chunks = tuple(expected)
        self._vectors = _checked_vectors(vectors, len(expected)).copy()
        self._vectors.setflags(write=False)
        self._full_rows = np.array(
            [
                row
                for row, (position, start, end) in enumerate(self._chunks)
                if start == 0 and end == len(self._journeys[position].stops)
            ],
            dtype=np.int64,
        )

    @property
    def dimension(self) -> int:
        return self._vectors.shape[1]

    @property
    def journey_count(self) -> int:
        return len(self._journeys)

    @property
    def vector_count(self) -> int:
        return len(self._chunks)

    @classmethod
    def build(
        cls,
        journeys: Sequence[Journey],
        bundle: EncoderBundle,
        *,
        window_sizes: Sequence[int] = (4, 8, 16),
        batch_size: int = 128,
    ) -> DenseIndex:
        sizes = _window_sizes(window_sizes)
        model_fingerprint = _fingerprint(bundle)
        corpus = list(journeys)
        if not corpus:
            raise ValueError("Cannot build an empty index")
        if len({journey.id for journey in corpus}) != len(corpus):
            raise ValueError("Journey IDs must be unique")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        chunks = [
            (position, start, end)
            for position, journey in enumerate(corpus)
            for start, end in _ranges(len(journey.stops), sizes)
        ]
        windows = [subjourney(corpus[position], start, end) for position, start, end in chunks]
        vectors = bundle.encode(windows, batch_size=batch_size)
        return cls(corpus, vectors, chunks, model_fingerprint, sizes)

    def assert_compatible(self, bundle: EncoderBundle) -> None:
        if _fingerprint(bundle) != self.model_fingerprint:
            raise ValueError("Checkpoint fingerprint does not match the index; rebuild the index")

    def get_journey(self, journey_id: str) -> Journey | None:
        position = self._by_id.get(journey_id)
        return None if position is None else self._journeys[position].model_copy(deep=True)

    def example_journeys(self, limit: int = 5) -> list[Journey]:
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        return [journey.model_copy(deep=True) for journey in self._journeys[:limit]]

    def search(
        self,
        query: Journey,
        bundle: EncoderBundle,
        *,
        top_k: int = 10,
        mode: SearchMode = "shared",
    ) -> list[SearchHit]:
        self.assert_compatible(bundle)
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if mode not in ("whole", "shared"):
            raise ValueError("mode must be 'whole' or 'shared'")
        query_ranges = (
            [(0, len(query.stops))]
            if mode == "whole"
            else _ranges(len(query.stops), self.window_sizes)
        )
        query_windows = [subjourney(query, start, end) for start, end in query_ranges]
        query_vectors = _checked_vectors(
            bundle.encode(query_windows), len(query_ranges), self.dimension
        )
        rows = self._full_rows if mode == "whole" else np.arange(self.vector_count)
        best: dict[int, tuple[float, int, int]] = {}
        # Bound the temporary similarity matrix by processing corpus windows in blocks.
        for offset in range(0, len(rows), 8192):
            block_rows = rows[offset : offset + 8192]
            similarities = query_vectors @ self._vectors[block_rows].T
            best_queries = similarities.argmax(axis=0)
            best_scores = similarities[best_queries, np.arange(len(block_rows))]
            for row, query_row, score in zip(block_rows, best_queries, best_scores, strict=True):
                document_position = self._chunks[int(row)][0]
                current = best.get(document_position)
                if current is None or float(score) > current[0]:
                    best[document_position] = (float(score), int(query_row), int(row))
        ranked = sorted(best.items(), key=lambda item: (-item[1][0], self._journeys[item[0]].id))
        return [
            SearchHit(
                journey_id=self._journeys[position].id,
                score=float(np.clip(score, -1.0, 1.0)),
                query_stop_range=query_ranges[query_row],
                document_stop_range=self._chunks[row][1:],
            )
            for position, (score, query_row, row) in ranked[:top_k]
        ]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        temporary: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                vector_path = Path(stream.name)
                temporary.append(vector_path)
                np.save(stream, self._vectors, allow_pickle=False)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False
            ) as stream:
                journey_path = Path(stream.name)
                temporary.append(journey_path)
                for journey in self._journeys:
                    stream.write(journey.model_dump_json() + "\n")
            manifest = {
                "format_version": _FORMAT_VERSION,
                "model_fingerprint": self.model_fingerprint,
                "dimension": self.dimension,
                "journey_count": self.journey_count,
                "vector_count": self.vector_count,
                "window_sizes": list(self.window_sizes),
                "chunks": [list(chunk) for chunk in self._chunks],
                "arrays_sha256": _sha256(vector_path),
                "journeys_sha256": _sha256(journey_path),
                "score_kind": "cosine_similarity_not_probability",
            }
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=directory, delete=False
            ) as stream:
                manifest_path = Path(stream.name)
                temporary.append(manifest_path)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            # Commit the checksummed manifest last. Interrupted/overlapping writes
            # produce a rejected index instead of silently mixing artifacts.
            os.replace(vector_path, directory / "arrays.npy")
            os.replace(journey_path, directory / "journeys.jsonl")
            os.replace(manifest_path, directory / "manifest.json")
        finally:
            for path in temporary:
                path.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path, bundle: EncoderBundle) -> DenseIndex:
        directory = Path(directory)
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["format_version"] != _FORMAT_VERSION:
                raise ValueError("Unsupported index format version")
            if manifest["model_fingerprint"] != _fingerprint(bundle):
                raise ValueError("Checkpoint fingerprint does not match the index; rebuild the index")
            for filename, hash_key in (
                ("arrays.npy", "arrays_sha256"),
                ("journeys.jsonl", "journeys_sha256"),
            ):
                if _sha256(directory / filename) != manifest[hash_key]:
                    raise ValueError(f"Index artifact checksum mismatch: {filename}")
            with (directory / "journeys.jsonl").open(encoding="utf-8") as stream:
                journeys = [Journey.model_validate_json(line) for line in stream if line.strip()]
            vectors = np.load(directory / "arrays.npy", allow_pickle=False)
            chunks = manifest["chunks"]
            if not isinstance(chunks, list) or any(
                not isinstance(chunk, list)
                or len(chunk) != 3
                or any(type(value) is not int for value in chunk)
                for chunk in chunks
            ):
                raise ValueError("Index chunk references are malformed")
            index = cls(
                journeys,
                vectors,
                [tuple(chunk) for chunk in chunks],
                manifest["model_fingerprint"],
                manifest["window_sizes"],
            )
            for key in ("dimension", "journey_count", "vector_count"):
                if type(manifest[key]) is not int or manifest[key] != getattr(index, key):
                    raise ValueError(f"Index manifest {key} does not match its artifacts")
            # The fingerprint binds weights/configuration; a probe additionally
            # catches encoders that violate their advertised output dimensions.
            _checked_vectors(bundle.encode([journeys[0]]), 1, index.dimension)
            return index
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load index at {directory}: {error}") from error
