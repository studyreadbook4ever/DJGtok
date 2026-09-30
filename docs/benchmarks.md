# Benchmarks and retained negative results

DJGtok's default matcher answers a specific question: **which supplied route has the longest consecutive sequence of the same directed stop IDs and transport modes?** It uses canonical IDs, not a learned notion of similar geography. The measurements below validate this structural task. They do not establish passenger usefulness, timetable feasibility, route-planning quality, or improved neural retrieval.

The repository contains [synthetic fixtures](../benchmarks/fixtures.py), a [runnable benchmark](../benchmarks/run_benchmark.py), and aggregate JSON reports. It contains no actual stop lists, source journeys, coordinates, feed archives, or trained weights.

## Current exact matcher: different real routes

The local corpus contained **707 complete Seoul bus patterns, 40,883 stop occurrences and 40,176 directed edges**. Each query below was the entire source route. The query's own route was excluded; the two routes in each pair had different full stop sequences.

| Query → target | Consecutive shared edges | Current target rank | Archived neural shared rank |
|---|---:|---:|---:|
| 750A → 750B | 72 | **1** | 27 |
| 702A → 702B | 57 | **1** | 3 |
| 6411 → 8641 | 46 | **1** | 4 |

For each of the three queries, an independent quadratic dynamic-programming oracle scanned every other source route. The current first result had the maximum shared-edge count, and each named target preceded every shorter match. These are three source-selected demonstrations, not a representative or held-out accuracy sample. The archived neural queries were training routes.

The old neural scorer selected the maximum cosine over short windows. Many identical four-edge windows could score near 1, allowing a short shared fragment to outrank a much longer shared corridor. The new default explicitly orders by consecutive shared-edge count. **Changing the criterion and requiring matching IDs is not an improvement to the neural model.**

Full measurements: [seoul-exact.json](../benchmarks/results/seoul-exact.json).

## Generated crop cohort

To make the protocol inspectable, the benchmark replays the earlier route-group partition: 493 train-labeled routes, 111 validation-labeled routes and 103 test-labeled routes. The exact algorithm has **no training stage**; these labels only identify the historical cohort.

One test route with a full coordinate topology duplicated in the other partitions was excluded from query generation. With NumPy seed 42, one strict contiguous crop was sampled from each remaining route: **102 queries**. The search corpus was the 103 test routes plus 103 explicitly synthetic reversed routes. The crop lengths, sampling order and split hash rule are recorded in the script and report.

| Structural measurement | Result |
|---|---:|
| Known parent Recall@1 | 102/102, **100%** |
| Known parent Recall@5 | 102/102, **100%** |
| Own reversed parent outranked the parent, eligible pairs | 0/101 |
| Zero-overlap own reversal returned in the first five, eligible pairs | 0/101 |

These queries retain the original stop IDs, so exact containment is available by construction. Alternative routes can also be correct answers; a parent-identity score is not a complete relevance assessment. Reverse checks exclude the one query whose original direction also appears within its parent's reversal. No human labels or statistical generalization claim are implied.

## Measured local latency

Recorded on Linux x86-64, Intel Core i7-10700, CPython 3.12.13, NumPy 2.5.3 and Pydantic 2.13.5. Three warm-up passes preceded ten measured passes. The p95 is the nearest-rank percentile. Timings include in-process search and hit construction; they exclude input loading and index construction.

| Workload | Search corpus | Measured calls | Median | p95 |
|---|---:|---:|---:|---:|
| Three full-route queries, self excluded, top 5 | 707 routes | 30 | 12.698 ms | 15.654 ms |
| Generated crop queries, top 5 | 103 real + 103 synthetic routes | 1,020 | 4.564 ms | 12.381 ms |

One construction of the 707-route index took 551.581 ms. These observations are not throughput guarantees, all-city scale tests, or public-transport journey-planning latency. The two timing rows use different corpus sizes and query lengths.

## Synthetic adversarial cases

[synthetic.json](../benchmarks/results/synthetic.json) records four passing cases:

- A six-edge shared corridor ranks before a two-edge fragment; reversed, wrong-mode and disconnected alternatives are absent.
- Repeated loop edges retain sequence order; a reordered alternative gets only its genuinely consecutive match.
- Mode changes are part of edge identity.
- Renaming every query stop ID produces no match, even with unchanged coordinates. This deliberately exposes the known-ID requirement.

These small cases check behavior; their timing is not a transit-network performance estimate.

## Archived neural experiment

The optional `djgtok.experimental` code preserves the earlier research direction. Its historical failures are retained in [historical-neural-seoul.json](../benchmarks/results/historical-neural-seoul.json); the default matcher does not depend on it.

The experiment ran 600 optimization steps and selected step 550 by the minimum fixed validation contrastive loss, 0.414277. The clean generated-crop cohort had 102 queries and 206 documents. Three complete query subsequences also appeared in train/validation routes despite route-group separation.

| Archived clean method | Parent Recall@1 | Parent Recall@5 | Own reversal outranked parent, 101 eligible pairs |
|---|---:|---:|---:|
| Learned whole-route vector | 90.2% | 99.0% | 9/101 |
| Learned shared-window vector | **64.7%** | 99.0% | 4/101 |
| Untrained whole-route vector | 66.7% | 99.0% | 29/101 |
| Untrained shared-window vector | 59.8% | 100% | 9/101 |
| Earlier exact contiguous-ID baseline | 99.0% | 100% | 0/101 |

The learned shared method still ranked 750B **27th** for the full 750A query despite their 72 consecutive shared edges. This was a real failure under a longest-corridor requirement, not a result to hide. Whole-vector learning showed improvement over its untrained counterpart on this generated cohort, but neither that result nor the shared result establishes a generally superior neural system.

The archived renamed-ID and 15 m coordinate-noise scenario is a different observation-robustness task. It intentionally disables an exact-ID baseline. Its numbers remain in the JSON and must not be used to claim that the new ID-based matcher solves noisy or unaligned identities.

Historical reports are whitelisted aggregates extracted from locally retained experiment artifacts. Their artifact hashes are recorded; the underlying model and datasets are not distributed here. The neural experiment was not rerun to produce the current exact-matcher result.

## Reproduce without bundled data

Run these commands from a repository checkout. The benchmark does not download anything.

```sh
python -m pip install -e ".[dev,seoul]"
python -m benchmarks.run_benchmark synthetic \
  --output results/synthetic.json --warmup 3 --repeats 20
```

For the frozen Seoul comparison, obtain the permitted 2026-09-02 workbook from the official source yourself, retain it outside the checkout if desired, and import it locally:

```sh
djgtok import-seoul YOUR_LOCAL_WORKBOOK.xlsx YOUR_LOCAL_JOURNEYS.jsonl
python -m benchmarks.run_benchmark seoul \
  --data YOUR_LOCAL_JOURNEYS.jsonl \
  --output results/seoul-exact.json --seed 42 --warmup 3 --repeats 10
```

There is no default personal data path. The frozen comparison rejects a different source hash instead of applying the historical expected counts to a changed dataset. It writes aggregates only. To extract aggregates from your own copies of the archived-format experiment reports:

```sh
python -m benchmarks.run_benchmark historical \
  --evaluation YOUR_EVALUATION.json --raw-audit YOUR_RAW_AUDIT.json \
  --training YOUR_TRAINING_REPORT.json --output results/historical.json
```

## Source and interpretation limits

Source: **Seoul Metropolitan Government**, [Seoul Open Data OA-1095](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do), file `서울시버스노선별정류소정보(20260902).xlsx`, listed under **Korea Open Government License Type 1 (attribution)**. Workbook SHA-256:

```text
b54816e9b7e77bcf49dd7562b993db06a093daa666351f115ea4d1dd0dd8b1df
```

The local source had 718 route IDs and 41,688 rows. Eleven entire routes containing sequence gaps were excluded, leaving the measured 707 routes. Earlier independent local auditing compared every retained ID, name, coordinate and sequence to the workbook. Only aggregate evidence is included here; the raw or processed dataset is not redistributed.

The published sequence can include outbound and return travel; an outbound/return boundary and operating timetable were not supplied. X/Y values were interpreted as longitude/latitude, but the workbook did not declare a coordinate datum. Consecutive matching stops do not prove identical road geometry between them. Matching IDs must refer to the same provider namespace; connecting independently named feeds requires an explicit, validated mapping. None of these benchmarks demonstrates optimal routing, valid transfers, real-time operation, passenger intention, automatic semantic clustering, or a universal ranking of useful journeys.
