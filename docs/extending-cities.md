# Extending DJGtok to another city

Start with an authorized local feed and the exact overlap matcher. A new city does not require a trained model. The steps below are integration guidance for NYC, Tokyo, and Sydney, not claims that those cities have already been evaluated. Source and policy links were checked on 2026-09-30; recheck the selected dataset when downloading it.

## Obtain the right source

| Region | Official starting point | Scope to resolve before importing |
| --- | --- | --- |
| New York | [MTA developer resources](https://www.mta.info/developers) | Select the relevant subway, bus, or railroad feed; bus feeds and operating companies are separate. Regular and supplemented subway schedules serve different purposes. |
| Tokyo | [ODPT data catalog](https://ckan.odpt.org/) and [developer site](https://developer.odpt.org/) | Choose an operator and dataset. ODPT offers multiple formats and access conditions; do not assume a single Tokyo-wide GTFS feed. |
| Sydney / NSW | [TfNSW Timetables Complete GTFS](https://opendata.transport.nsw.gov.au/dataset/timetables-complete-gtfs) | Decide whether the complete static bundle or a realtime-related subset fits the task; they have different coverage. |

For MTA, read the [data-feed terms](https://www.mta.info/developers/terms-and-conditions). They address hosting, accuracy claims, and preservation of the supplied data, including a restriction on modifying or deleting it. Keep the original download intact and distinguish internal application representations from the feed. Review the intended transformation and distribution against those terms. Feed access does not grant an unrestricted license to MTA maps, logos, or other branding.

For Tokyo, read the selected dataset's license and provider-specific conditions via the [ODPT terms](https://developer.odpt.org/terms). The [Basic License](https://developer.odpt.org/terms/data_basic_license.html) only applies where expressly assigned, and provider-specific terms may prevail. Review its redistribution restrictions before sharing original or recoverable derived data. An openly accessible API does not by itself authorize committing snapshots or a normalized corpus to GitHub.

TfNSW's dataset page labels the complete GTFS bundle as Creative Commons Attribution. Read the applicable dataset license and [portal terms](https://opendata.transport.nsw.gov.au/sites/default/files/2024-09/TfNSW-Open-Data-Portal-Terms.pdf) for attribution and use conditions. For account and API access, follow the official [API basics](https://opendata.transport.nsw.gov.au/developers/api-basics). Keep access keys in your local environment, outside fixtures, URLs in public logs, and source control.

These links are starting points, not a legal clearance for every application. Where permission for including material is unclear, leave that material out of the contribution and use a synthetic fixture. DJGtok's MIT license applies to its original code, not to provider data.

## Preserve route meaning

Use `Journey` as an ordered stop sequence, with one `mode` and `line_id` per directed edge. The available modes are `walk`, `bus`, `rail`, `subway`, `tram`, `ferry`, and `other`.

1. **Namespace IDs by city and provider.** For example, a local GTFS namespace could be `nyc:mta-subway`. Preserve the source ID separately where needed. Identical short IDs from unrelated feeds must not collide. Use a documented crosswalk if two feeds identify the same boarding location differently; matching names or nearby coordinates alone is insufficient.
2. **Preserve explicit order and direction.** Sort GTFS stop times by numeric `stop_sequence`, not file order, coordinates, or stop name. GTFS sequence numbers need not be consecutive. Duplicate sequence positions are ambiguous and should be rejected or reported. Do not infer a return service by reversing a list.
3. **Preserve loops and repeated visits.** `A → B → A → C` contains four visits; collapsing it into distinct stops changes its edges. Use source direction or trip-pattern fields where available. Do not split an apparent loop into operating directions without evidence.
4. **Preserve mode and service context.** Keep pickup/drop-off restrictions, representative times, original trip/route identifiers, and available service references in metadata or a separate local manifest. Times after `24:00:00` belong to a service day and must not be silently wrapped. The core matcher does not check calendars, departures, transfer feasibility, or accessibility.
5. **Validate instead of filling gaps.** Require finite WGS84 coordinates, valid IDs, a justified order, and aligned edge arrays. Report unsupported records and exclusion reasons. Do not invent coordinates, transfers, travel times, or direction labels. Journeys currently contain 2–512 stops; handle longer patterns with explicitly documented contiguous windows or report them as unsupported.
6. **Track source changes.** Retain snapshot hashes, provider terms, parser version, and counts. Reused or changed provider stop IDs require a documented migration; do not assume an ID remains stable forever. Keep all views from the same route group together when evaluating.

The [official GTFS Schedule reference](https://gtfs.org/documentation/schedule/reference/) defines the source fields and their service-day semantics. A pattern extraction is not a timetable router, and a set of routes is not a set of passenger journeys.

## Load a local GTFS feed

Obtain the ZIP through the provider's permitted access process. The high-level loader requires an explicit namespace:

```python
from djgtok import OverlapIndex
from djgtok.io import load_gtfs

journeys, report = load_gtfs(
    "local-feeds/provider.zip",
    namespace="city:provider",
)
print(report)
index = OverlapIndex(journeys)
query = journeys[0]
hits = index.search(query, top_k=5, min_shared_edges=1, exclude_ids=(query.id,))
print(hits)
```

Inspect the report and verify a few retained patterns against the local source before interpreting results. This snippet searches supplied patterns. It does not fetch data, provision an API key, or generate a feasible itinerary.

## Run a fully synthetic example

All IDs, coordinates, and lines below are invented. No transit feed is needed.

```python
from djgtok import Journey, OverlapIndex, Stop, TransitTokenizer, compare_routes

stops = {
    name: Stop(id=f"demo:provider:{name}", lat=0.0, lon=i * 0.001)
    for i, name in enumerate("XABCY")
}


def route(route_id, names):
    return Journey(
        id=f"demo:{route_id}",
        stops=[stops[name] for name in names],
        modes=["bus"] * (len(names) - 1),
        line_ids=[f"demo:line:{route_id}"] * (len(names) - 1),
        source="synthetic",
    )


query = route("query", "ABC")
document = route("document", "XABCY")
match = compare_routes(query, document)
assert match.shared_edge_count == 2
assert match.query_stop_range == (0, 3)
assert match.document_stop_range == (1, 4)
assert compare_routes(query, route("reverse", "CBA")).shared_edge_count == 0

index = OverlapIndex([document])
print(index.search(query, top_k=5, min_shared_edges=1, exclude_ids=()))

tokenizer = TransitTokenizer.fit([document])
print(tokenizer.inspect(query))
```

Ranges are **half-open stop indices**: `document.stops[1:4]` includes three stops and two edges. Matching requires a contiguous run of equal ordered endpoint IDs and modes; line IDs need not match. Nearby stops with different IDs do not match automatically. This definition explains a shared corridor without establishing simultaneous service or a valid transfer.

`TransitTokenizer.fit` builds a local vocabulary and geographic reference from the supplied journeys. It is separate from neural model training. Fit and retain a tokenizer per geographic region or explicitly evaluated regional configuration; a local coordinate approximation is not a global projection. For an evaluation, fit on training data only and test unseen stops and cells. The exact matcher does not use the tokenizer and needs neither fit nor training.

## Work with Codex or Claude Code

Read [AGENTS.md](../AGENTS.md) or [CLAUDE.md](../CLAUDE.md) in your coding session. The following prompts request bounded changes with observable completion criteria. Replace the provider placeholder with a specific dataset you have selected; keep real feeds outside the repository.

**NYC adapter and import report**

> Extend DJGtok's local GTFS import workflow for a selected MTA feed. Read the official developer page and current terms, document the exact feed scope, and use an explicit city/provider namespace. Work from a user-provided local ZIP; add no downloader or raw data to the package. Identify fields the current parser retains or loses. Add synthetic fixtures for unsorted stop_times, nonconsecutive sequences, repeated stops, mode changes, and times after midnight. Report exclusions, preserve source identifiers, and show the exact commands and test results. Do not claim route-planning or NYC performance from these fixture tests.

**Tokyo operator integration**

> Select one named operator dataset from the official ODPT catalog and document its format, access requirements, license, and additional conditions. If it is GTFS, reuse the existing loader; if it is JSON-LD, propose and implement a focused local-file adapter after inspecting its official schema. Do not assume rail line order proves the stopping pattern of every train. Reject records lacking required order or coordinates with an explicit report. Use synthetic fixtures for express/local differences, loops, and provider namespace collisions. Keep provider snapshots and credentials out of Git.

**Sydney feed coverage**

> Add a documented local-import example for the TfNSW Complete GTFS bundle. Compare its official scope with the realtime-related bundle, record the chosen scope, and reuse load_gtfs where possible. Preserve stop identity, travel mode, pickup/drop-off restrictions, service-day times, and available service references. Expose missing information rather than inventing it. Use synthetic multi-operator fixtures and verify that the same bare stop ID in independent namespaces never creates an accidental match. Document unsupported GTFS fields and include no actual feed or key.

**Improve matching speed without changing answers**

> Profile OverlapIndex using deterministic synthetic corpora at several sizes. Keep compare_routes as the exhaustive correctness oracle. Optimize candidate selection only if all returned matches, ranges, exclusions, minimum-edge filters, and ranking remain equivalent. Test repeated edges, ties, reverse direction, distinct modes, and independent namespaces. Report seed, corpus shape, hardware, build time, query latency, and memory methodology. Include regressions and avoid extrapolating synthetic speed to an unmeasured city.

**Explore regional tokens**

> Evaluate a tokenizer change on independently split route groups using authorized local data. Fit only on the training split, retain directions and derived windows in the same group, and audit duplicated subpaths. Add synthetic tests for unknown cells, finite features, reverse direction, save/load consistency, and new-region behavior. Compare with exact directed overlap before introducing a trained model. Publish aggregate measurements with denominators and failures; leave real records, private paths, and checkpoints out of the contribution.

For a contribution to be ready, its synthetic tests should pass offline, its import report should account for retained and excluded records, and every new claim should name the evidence that supports it. Include changes to source attribution and known limitations alongside the code.
