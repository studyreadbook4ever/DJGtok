# How DJGtok was made

## The question

The original project asked whether transit movement could be treated like a vocabulary: express where and in which direction a journey moves, then relate different routes that share part of that movement. The intended benefit was reusable understanding of transit corridors alongside an existing public-transport journey service.

Trajectory-representation work such as [t2vec](https://xiucheng.org/assets/pdfs/icde18-t2vec.pdf) helped frame the idea. DJGtok does not claim to have invented trajectory embeddings, tokenization, contrastive learning, Transformers or longest-common-substring matching. Its contribution is the particular transit-oriented representation, validation, evidence interface and reusable combination implemented here. It is neither the original t2vec code nor a reproduction of that paper's reported results.

## The local prototype

The project first developed GTFS ingestion and a trainable geographic representation, then investigated a Seoul bus snapshot. Directed spatial tokens and continuous geometry fed a small contrastive Transformer. Route-group splits, synthetic reverse cases, untrained comparisons and source-row checks were added to test whether the idea worked. AI coding assistance was used throughout; the repository records tests and measurements rather than treating generated code as evidence of correctness.

The original local application also had a map/demo interface. That interface, its third-party map library, tiles, screenshots and local service controls are outside this library release. The release extracts the reusable Python work and keeps examples self-contained.

## What failed

The prototype's shared-window search aggregated the maximum cosine over short windows. That made a highly similar small fragment sufficient for a high score. It did not reliably rank routes by how much movement they shared: a 72-edge corridor appeared at rank 27 in an audited example. Whole-route and shared-window retrieval were also different tasks, so a favorable whole-route score could not establish that shared-corridor retrieval worked.

Untrained and simple baselines remained competitive on some metrics, especially noisy/renamed-ID tests. Consequently the project does not claim a neural state of the art, reliable cross-city generalization, or a learned replacement for routing. The historical implementation remains explicitly experimental so others can inspect the attempt; the failed scoring approach has no default role in the public matcher.

## The public library

For version 0.1.0, the default retrieval criterion became the longest consecutive canonical stop-ID-and-mode match. It expresses a narrower, testable contract: return exactly how many directed edges match and where. Geographic tokenization remains available independently for new experiments.

Release preparation also introduced an importable `djgtok` API, lightweight core dependencies, optional Seoul/PyTorch extras, offline invented examples, JSON tokenizer persistence, deterministic matching, source-evidence ranges, mutation isolation, stronger validation and stale-model/index rejection. Packaging and continuous integration make these checks repeatable outside the author's original workspace.

Existing provider data and trained checkpoints were not copied into the release. Research is cited, dependencies are installed separately, and the project's original work is offered under MIT with separate data provenance. The initial public commit is a curated library snapshot; it is not represented as a complete Git history of the earlier local exploration.

## What comes next

City extensions are welcome when they use permitted source data and preserve identity, order, direction and provenance. Canonical stop matching and geographic similarity have different requirements; a future model should beat suitable untrained, geographic and exact-ID baselines on its intended observation regime before becoming a default feature. That includes held-out route families, explicit duplicate checks, direction tests, unknown-ID tests and measurements of shared length rather than only the maximum short-window score.

The [city guide](extending-cities.md) supports contributors using Claude Code or Codex, but contributions are assessed through their sources, implementation and verification. See [benchmarks](benchmarks.md) for the evidence available now.
