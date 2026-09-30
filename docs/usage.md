# Using the library

Install from the versioned GitHub tag, then run `djgtok demo`. See the [README quickstart](../README.md). Python 3.11–3.13 is supported. No data download occurs on import or during the invented demo.

## Core API

| API | Result |
| --- | --- |
| `Journey(...)`, `Stop(...)` | Validated Pydantic models, with `.model_dump()` / `.model_dump_json()` |
| `subjourney(journey, start, end)` | Independent contiguous slice using half-open stop indices |
| `reverse_journey(journey)` | Marked synthetic reversal, without forward service promises |
| `compare_routes(query, document)` | Longest directed canonical-ID-and-mode match |
| `OverlapIndex(journeys)` | Independent in-memory corpus snapshot |
| `index.search(query, top_k=5, min_shared_edges=1, exclude_ids=())` | Ranked positive matches with copied evidence |
| `index.get_journey(id)` | Independent journey copy, or `KeyError` |
| `TransitTokenizer.fit(training_journeys)` | Frozen vocabulary and geographic configuration |
| `tokenizer.encode(journey)` | NumPy token ID and continuous feature arrays |
| `tokenizer.inspect(journey)` | Human-readable tokens and unknown fraction |
| `tokenizer.save(path)` / `TransitTokenizer.load(path)` | Versioned JSON serialization |

The default tokenizer returns 3 token IDs and 42 features per edge. Custom `resolutions` and `periods` change these dimensions. Store the tokenizer with any consumer trained against its IDs. The `.fingerprint` identifies its full mapping and configuration.

`Match` fields are `shared_edge_count`, `query_stop_range`, `document_stop_range`, `query_coverage`, `document_coverage` and `match_kind`. Ranges contain `shared_edge_count + 1` stops. For no overlap, both ranges are `None` and both coverages are zero. `OverlapHit` adds `journey_id`, integer `score`, `score_kind` and `evidence`. Different line IDs can match; different modes cannot. Search includes a matching query ID unless you explicitly exclude it.

Exact matching requires provider-scoped canonical IDs. Do not merge feeds whose bare IDs collide. Coordinates are numeric latitude/longitude, not strings or projected x/y meters. Journeys have 2–512 stops; the importer does not truncate larger ones. Source metadata must contain finite JSON values.

## Local files

Use the high-level GTFS adapter to get `Journey` objects:

```python
from djgtok.io import load_gtfs, load_journeys, save_journeys

journeys, diagnostics = load_gtfs("permitted-feed.zip", namespace="my-city:provider")
save_journeys(journeys, "local-output/journeys.jsonl")
restored = load_journeys("local-output/journeys.jsonl")
```

Each JSONL line is one complete Journey object. A query JSON file contains one Journey object, not a list. Stop-level provider fields and route-level metadata retain evidence locally, so review outputs before sharing them.

The GTFS loader collapses repeated scheduled trips into planned single-vehicle patterns. It validates structural references, sequence uniqueness and time syntax; it does not resolve service calendars for a date, enforce trip accessibility, route around closures, or construct valid transfers. Restrictions and representative times remain metadata for a downstream application to interpret.

For the supported Seoul workbook format, install the extra:

```sh
python -m pip install "DJGtok[seoul] @ git+https://github.com/studyreadbook4ever/DJGtok.git@v0.1.0"
```

```python
from djgtok.io import load_seoul

journeys, diagnostics = load_seoul("my-authorized-local-workbook.xlsx")
```

This adapter expects the OA-1095 workbook's `Data` sheet and eight documented headers. It excludes an entire route if its sequence is incomplete or inconsistent, with reasons in diagnostics. Calling a workbook reader does not independently verify the download's origin. Data access and redistribution follow the provider's terms; the repository includes no actual feed.

## CLI

All commands operate on local files or invented examples. Use `djgtok --help` or `python -m djgtok --help`.

```sh
djgtok demo
djgtok import-gtfs permitted-feed.zip journeys.jsonl --namespace my-city:provider
djgtok import-seoul permitted-workbook.xlsx journeys.jsonl
djgtok compare query.json document.json
djgtok search journeys.jsonl query.json --top-k 5 --min-shared-edges 3
djgtok fit-tokenizer training-journeys.jsonl tokenizer.json
djgtok tokenize tokenizer.json query.json
```

Use `--exclude-id` repeatedly on `search` to omit known journey IDs. File imports overwrite the explicitly supplied output path. Search constructs the in-memory index from JSONL each time; long-running applications should retain an `OverlapIndex` instance. The current exact matcher has no persisted binary index format or built-in web server.

## Optional experiments

Install `DJGtok[torch]` from the GitHub tag using the same direct-URL syntax as the Seoul example. GPU setup is outside core installation; CPU execution is available. Pretrained weights are not distributed. You need your own permitted training data and must evaluate its suitability before publishing derived artifacts.

```sh
python -m djgtok.experimental.train --help
python -m djgtok.experimental.evaluate --help
```

The default learned encoder is a small Transformer with contrastive training. The experimental dense index's `whole` and `shared` modes return cosine similarity, not exact shared length or probabilities. They are separate from `OverlapIndex`; no model is loaded by the core API. Historical evidence did not establish a reliable advantage for shared-corridor search. See [architecture](architecture.md) and [benchmarks](benchmarks.md) before relying on an experiment.

## Using with an existing journey planner

Convert each proposed itinerary into a Journey with stop IDs from one consistent namespace and one mode/line ID per consecutive movement. Keep the planner responsible for valid service dates, transfers, walking connectors, accessibility and arrival times. Use `compare_routes` to explain shared contiguous portions, or fit a tokenizer for separate representation experiments. Do not infer a walk edge or feasible transfer merely because two legs end nearby.
