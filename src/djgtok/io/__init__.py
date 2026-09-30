"""Import user-supplied transit files; no data is bundled or downloaded automatically."""

from pathlib import Path

from djgtok.io.gtfs import extract_gtfs
from djgtok.schema import Journey, load_journeys, save_journeys

__all__ = ["load_gtfs", "load_seoul", "load_journeys", "save_journeys"]


def load_gtfs(path: str | Path, *, namespace: str, min_stops: int = 3,
              max_patterns: int | None = None) -> tuple[list[Journey], dict]:
    """Read a permitted local feed into validated planned, single-vehicle patterns.

    Use a stable city/provider namespace to prevent collisions across feeds.
    This does not validate service-date or transfer feasibility.
    """
    if not isinstance(namespace, str) or not namespace.strip() or namespace != namespace.strip():
        raise ValueError("namespace must be a nonblank string without surrounding whitespace")
    records, report = extract_gtfs(path, source=namespace, min_stops=min_stops,
                                   max_patterns=max_patterns)
    return [Journey.model_validate(record) for record in records], report


def load_seoul(path: str | Path) -> tuple[list[Journey], dict]:
    """Read the supported Seoul OA-1095 XLSX format; requires the 'seoul' extra."""
    try:
        from djgtok.io.seoul import extract_seoul
    except ModuleNotFoundError as error:
        if error.name != "openpyxl":
            raise
        raise ImportError("Seoul XLSX import requires installation with DJGtok[seoul]") from error
    records, report = extract_seoul(path)
    return [Journey.model_validate(record) for record in records], report
