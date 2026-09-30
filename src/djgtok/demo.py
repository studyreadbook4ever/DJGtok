"""Entirely invented coordinates and identifiers, usable without a transit feed."""

from djgtok.schema import Journey, Stop, reverse_journey, subjourney


def synthetic_corpus() -> tuple[Journey, list[Journey]]:
    stops = [Stop(id=f"demo:{i}", lat=10 + i * 0.001, lon=20 + i * 0.001) for i in range(9)]
    query = Journey(id="demo:query", stops=stops, modes=["bus"] * 8,
                    line_ids=["demo:line-a"] * 8, source="synthetic")
    long = subjourney(query, 0, 8).model_copy(update={"id": "demo:long", "line_ids": ["demo:line-b"] * 7})
    short = subjourney(query, 3, 6).model_copy(update={"id": "demo:short"})
    reverse = reverse_journey(query)
    return query, [short, reverse, long]
