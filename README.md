# DJGtok — 대중교통 토큰화

**Turn ordered transit routes into reusable tokens and explainable shared corridors.**

[한국어 소개](README.ko.md) · [Architecture](docs/architecture.md) · [Benchmarks](docs/benchmarks.md) · [City extensions](docs/extending-cities.md) · [Provenance](docs/provenance.md)

[![CI](https://github.com/studyreadbook4ever/DJGtok/actions/workflows/ci.yml/badge.svg)](https://github.com/studyreadbook4ever/DJGtok/actions/workflows/ci.yml)

The idea is simple: different transit lines can describe the same movement for part of their journey. Represent those movements as directed segments, then make their relationship inspectable and reusable. A rider, researcher or developer should be able to ask **“which routes travel together, in this direction, across how many consecutive stop pairs?”** and see the matching stops.

DJGtok is an early Python library with two independent core tools:

* **`TransitTokenizer`** produces multi-resolution geographic tokens plus continuous direction, position, mode and transfer features. It preserves every consecutive stop pair and does not require known stop or line identities.
* **`OverlapIndex` / `compare_routes`** find the longest consecutive shared sequence of **canonical stop IDs, in the same direction and mode**. Search ranks by shared edge count and returns the actual matching ranges. These tools require aligned stop IDs and do not use embeddings.

The library can sit beside an existing journey planner: compare its proposed journeys, explain shared portions, or build a research dataset. It does not compute optimal itineraries, validate transfers, check operating dates or predict arrivals. An edge means two consecutive supplied stops; it does not prove that vehicles follow the same road between them.

## Install and run

Python 3.11–3.13. Install from this repository; this release is not published to PyPI.

```sh
python -m pip install "git+https://github.com/studyreadbook4ever/DJGtok.git@v0.1.0"
djgtok demo
```

The invented demo needs no API key, feed, model, GPU or network after installation. Core dependencies are NumPy and Pydantic. Optional extras are `seoul` for local XLSX imports and `torch` for the experimental neural code.

```python
from djgtok import Journey, OverlapIndex, Stop, TransitTokenizer, compare_routes

# Entirely invented IDs and coordinates, not an operating service.
stops = [Stop(id=f"demo:{i}", lat=10 + i * .001, lon=20 + i * .001)
         for i in range(6)]
route = Journey(id="demo:route", stops=stops,
                modes=["bus"] * 5, line_ids=["demo:line"] * 5)
query = Journey(id="demo:query", stops=stops[1:5],
                modes=["bus"] * 3, line_ids=["demo:other"] * 3)

match = compare_routes(query, route)
assert match.shared_edge_count == 3
assert match.document_stop_range == (1, 5)  # Half-open STOP indices.
assert match.query_coverage == 1.0

hit = OverlapIndex([route]).search(query)[0]
print(hit.journey_id, hit.score, hit.evidence)

tokenizer = TransitTokenizer.fit([route])
ids, features = tokenizer.encode(query)
assert ids.shape == (3, 3)       # Edges × spatial resolutions.
assert features.shape == (3, 42)
tokenizer.save("tokenizer.json")
restored = TransitTokenizer.load("tokenizer.json")
```

A score of `3` means **three consecutive shared directed edges**, not a probability. A zero match has `None` ranges. Search omits zero matches. Opposite directions, different modes and disconnected fragments do not count as one corridor. Loops retain repeated stops. Equal-length matches have documented deterministic tie rules. Results include copied source evidence when supplied; they do not independently certify its accuracy.

Use your own permitted GTFS file:

```python
from djgtok import OverlapIndex
from djgtok.io import load_gtfs

routes, report = load_gtfs("my-permitted-feed.zip", namespace="my-city:my-provider")
index = OverlapIndex(routes)
hits = index.search(routes[0], exclude_ids=[routes[0].id], min_shared_edges=3)
```

The importer handles unsorted stop times and preserves loops, service-day times and boarding restrictions. It imports planned single-vehicle patterns, without constructing transfers. See [the API and CLI guide](docs/usage.md) for JSONL, local Seoul XLSX and optional experiments.

## Evidence and the failed approach

The local Seoul snapshot contains **707 accepted bus patterns and 40,883 stop occurrences**. Eleven routes with sequence gaps were excluded as whole routes. The public repository contains aggregate measurements and scripts, without the feed or trained weights. Data source: **Seoul Metropolitan Government**, [OA-1095, 2026-09-02 file](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do), listed under Korea Open Government License Type 1; see the separate [data attribution](docs/provenance.md).

The earlier neural experiment is deliberately documented, including its failure: max-cosine search over small windows placed a route sharing 72 directed edges at rank **27**. A perfect short-window cosine did not express how much of the routes actually traveled together. Its clean known-parent Recall@1 was **66/102 (64.7%)** in the shared-window setting. This is not evidence of a reliable learned corridor search system.

The released default uses an exact longest-sequence criterion instead. It has a different input requirement and objective; this is **not a claim that the neural model improved**. See [measured current results, timings, protocols and historical comparisons](docs/benchmarks.md), including runnable synthetic checks and the limits of same-ID matching. The neural implementation is retained under `djgtok.experimental` for reproducible research, with no pretrained artifact or default role in retrieval.

| Full-route query → comparison route | Shared directed edges | Earlier max-cosine rank | Released exact-match rank |
| --- | ---: | ---: | ---: |
| 750A → 750B | 72 | 27 | 1 |
| 702A → 702B | 57 | 3 | 1 |
| 6411 → 8641 | 46 | 4 | 1 |

Each search uses the 707-pattern corpus with the query itself excluded. These are audited diagnostic cases, not held-out neural generalization claims. The new exact matcher also recovered **102/102 known parents at rank 1** in the historical generated-crop cohort of 103 real routes plus 103 synthetic reversals. That same-ID structural test is deliberately easy for an exact method; it does not establish unknown-ID or cross-city performance. Local warm full-route search measured **12.7 ms median / 15.7 ms p95** over 30 calls on an i7-10700 CPU, excluding file loading and index construction. See the benchmark document for full protocols and caveats.

## Bring this philosophy to your city

**New York, Tokyo, Sydney — or your city: extensions are welcome.** The generic GTFS adapter is a starting point, not a claim that these cities have already been validated. Each provider has its own access conditions, identity rules, formats and service semantics.

You can use **Claude Code, Codex, or another coding assistant** to understand this repository and implement a city adapter. Start with [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) and the [city extension guide](docs/extending-cities.md), which includes official source entry points and a concrete contributor prompt. Small contributions are useful: a synthetic fixture, a documented namespace mapping, an importer diagnostic, or an independently checked benchmark.

```text
Read AGENTS.md, docs/architecture.md and docs/extending-cities.md.
Help me add a <city/provider> adapter using a local feed I am permitted to use.
Preserve directed stop order, mode, loops and source provenance.
Keep real feed data and credentials out of fixtures and commits.
Add synthetic tests, document limitations, and measure results without claiming
that tokenizer similarity proves a shared operational corridor.
```

An assistant can help implement and test an adapter; it cannot establish permission to redistribute a provider's data. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for development and evidence expectations. No paid service or account with this project is needed to use the library.

## How it was made and licensed

This project began as a question about treating transit movement like a vocabulary. The trajectory-representation idea was informed by research such as [t2vec](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf); the current implementation is a separate, AI-assisted project, not the original t2vec implementation or an official successor. [Development history](docs/history.md) explains the neural prototype, the failures discovered and the choices behind the public library. [Architecture](docs/architecture.md) describes the actual data flow and algorithms.

Project-authored code, documentation and invented examples are released under the **[MIT License](LICENSE)**. MIT permits commercial reuse too; the project itself is offered freely. This license does not relicense transit feeds, third-party packages, papers or their figures. Dependencies retain their own licenses. We do not bundle third-party implementation code, feeds, models, map tiles or screenshots. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [provenance](docs/provenance.md) for scope and attribution. We cannot promise the absence of every legal issue; uncertain redistribution rights are a reason to leave the asset out.
