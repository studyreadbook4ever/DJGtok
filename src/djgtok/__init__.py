"""DJGtok: validated transit paths, geographic tokens, and exact directed overlap.

The stable API needs NumPy and Pydantic only. Learned encoders and dense retrieval
are optional experiments available under :mod:`djgtok.experimental`.
"""
from djgtok.matching import Match, OverlapHit, OverlapIndex, compare_routes
from djgtok.schema import Journey, Stop, reverse_journey, subjourney
from djgtok.tokenizer import TransitTokenizer

__all__ = [
    "Journey", "Stop", "subjourney", "reverse_journey", "TransitTokenizer",
    "Match", "OverlapHit", "compare_routes", "OverlapIndex",
]
