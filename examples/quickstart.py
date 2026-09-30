"""An invented route example. Run after installing DJGtok; no feed is needed."""

from djgtok import Journey, OverlapIndex, Stop, TransitTokenizer, compare_routes

stops = [Stop(id=f"demo:{i}", lat=10 + i * 0.001, lon=20 + i * 0.001) for i in range(6)]
route = Journey(id="demo:route", stops=stops, modes=["bus"] * 5, line_ids=["demo:line"] * 5)
query = Journey(id="demo:query", stops=stops[1:5], modes=["bus"] * 3, line_ids=["demo:other"] * 3)

match = compare_routes(query, route)
assert match.shared_edge_count == 3
assert match.document_stop_range == (1, 5)
print(match.model_dump())
print(OverlapIndex([route]).search(query)[0].journey_id)

tokenizer = TransitTokenizer.fit([route])
ids, features = tokenizer.encode(query)
assert ids.shape == (3, 3)
assert features.shape == (3, 42)
print("Token IDs:", ids.shape, "continuous features:", features.shape)
