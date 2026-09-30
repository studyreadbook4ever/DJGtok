# Provenance and measurement boundaries

DJGtok distributes independently implemented Python code, synthetic fixtures, documentation, and aggregate experiment summaries. Raw transit feeds, extracted real route records, trained checkpoints, and private experiment logs are excluded from both the repository and package releases. See [third-party notices](../THIRD_PARTY_NOTICES.md) for the distinction between the code's MIT license and external rights.

## Research lineage

The starting question was whether the representation-learning idea behind t2vec could help compare directed public-transport journeys. The reference is Li, Zhao, Cong, Jensen, and Wei, *Deep Representation Learning for Trajectory Similarity Computation*, ICDE 2018, [DOI 10.1109/ICDE.2018.00062](https://doi.org/10.1109/ICDE.2018.00062), [author-hosted paper](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf). The paper studies representations of trajectories affected by sampling differences and noise.

Our implementation was written separately. We did not copy the authors' code, weights, figures, or paper text. The public library's primary matcher finds exact contiguous directed stop-ID-and-mode overlap; the optional neural experiments are a separate research path. Neither implementation is a faithful t2vec reproduction, and no result here establishes an improvement over t2vec.

## Seoul source

| Field | Value |
| --- | --- |
| Publisher | Seoul Metropolitan Government, Seoul Open Data Plaza |
| Dataset | [서울시 버스노선별정류소정보, OA-1095](https://data.seoul.go.kr/dataList/OA-1095/L/1/datasetView.do) |
| Source workbook | `서울시버스노선별정류소정보(20260902).xlsx` |
| Snapshot date | 2026-09-02 |
| Workbook SHA-256 | `b54816e9b7e77bcf49dd7562b993db06a093daa666351f115ea4d1dd0dd8b1df` |
| Source sheet | `Data` |
| Source terms | [Korea Open Government License, Type 1: attribution](https://www.kogl.or.kr/info/licenseType1.do), as identified on the dataset page |

Attribution: Seoul Metropolitan Government supplied the source route-stop data. We ordered each route using the published sequence, checked required fields and coordinates, excluded incomplete sequences, and computed the aggregate statistics below. These processing decisions and experiment results are ours. The source license remains applicable to the source data; DJGtok's MIT license does not replace it.

The workbook contained **41,688 rows across 718 routes**. Eleven routes, accounting for 805 rows, were excluded for gaps in their published stop sequence. The retained corpus contained **707 complete route patterns and 40,883 stop occurrences**. Occurrences are rows in route sequences, not a count of distinct physical stops. No missing stops, opposite-direction service, or transfer itineraries were invented.

## What the historical experiment measured

These are aggregate results from a predecessor experiment run on 2026-09-29, not a fresh benchmark of every public DJGtok release.

Routes were divided into 493 training, 111 validation, and 103 test groups before deriving cropped views. The evaluation document set contained 103 test patterns and 103 **synthetically reversed** patterns: **206 documents**. The earlier map demonstration used the larger set of 707 source patterns. Those are different corpus sizes and should not be interchanged.

After excluding one test pattern whose coordinate sequence duplicated a training or validation pattern, evaluation used 102 generated contiguous query crops. Three remaining query subpaths also occurred in training or validation routes. These were generated queries from scheduled route patterns, not observed passenger trips or independently collected search requests.

| Method in that experiment | Source-parent recall at 1 |
| --- | ---: |
| Exact directed stop-ID edge baseline | 101/102 (99.0%) |
| Learned whole-route representation | 92/102 (90.2%) |
| Untrained whole-route representation | 68/102 (66.7%) |
| Learned shared-segment representation | 66/102 (64.7%) |
| Untrained shared-segment representation | 61/102 (59.8%) |

The baseline compared stop-ID edges on this all-bus corpus. The public matcher explicitly includes mode in edge identity. Source-parent recall asks whether the original parent wins; another route sharing the query can still be a useful answer. The exact baseline's single first-place miss therefore does not mean it failed to find a matching corridor.

In the learned shared-segment experiment, a synthetic reverse scored above its parent on **4 of 101 direction-eligible queries** and ranked first on **3**. For the learned whole-route representation, those counts were **9** and **8**. A reversed pattern was a stress test, not evidence of a real service operating in that direction.

These results did **not** establish a useful advantage for learned shared-segment retrieval over exact overlap or its untrained comparison. We do not claim state-of-the-art results, reliable direction handling by the learned model, improved route planning, or successful transfer to another city. Exact overlap is the public core because its result can be inspected as matching edges and ranges; it requires no model training.

## Reproduction and new evidence

The aggregate numbers are historical evidence, not a self-contained reproducibility bundle: the source snapshot, processed routes, and trained checkpoint are intentionally absent. A newer provider download may differ from the recorded hash. Synthetic examples and tests can be run without provider accounts or transit data.

For a new real-data run, obtain an authorized snapshot yourself and record the provider, dataset and download URLs, acquisition time, SHA-256, source terms, parser version, inclusion/exclusion counts, and transformation rules. Keep raw identifiers and row references in local provenance. Keep credentials, private paths, raw records, and checkpoints out of public summaries.

Report the exact package revision, query construction, corpus size, split unit, duplicate checks, denominator, metric definition, seed, and runtime environment with any new measurement. Retain failures and relevant baselines. A claim about NYC, Tokyo, Sydney, noisy GPS, or cross-provider stop matching needs its own evidence.
