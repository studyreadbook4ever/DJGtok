# Working on DJGtok with Codex or another coding agent

DJGtok is a Python library for directed transit tokenization and explainable shared-corridor matching. Read `README.md`, `pyproject.toml`, and the relevant source/tests before changing an interface. See `docs/extending-cities.md` for concrete contributor tasks and `docs/provenance.md` for the evidence behind published results.

## Contracts to preserve

- The public core exports `Stop`, `Journey`, `TransitTokenizer`, `compare_routes`, and `OverlapIndex` from `djgtok`.
- A journey has ordered stops and exactly one mode and line ID per edge. Preserve repeated visits and direction. Do not invent missing stops, timings, transfers, or an opposite-direction service.
- Exact matching finds the longest contiguous run of directed edges with equal endpoint stop IDs and modes. Similar coordinates or matching stop names do not establish identical stops. Line IDs need not be equal.
- Match ranges are half-open **stop** indices. Two shared edges span three stops. Keep documented tie handling and index ranking deterministic.
- Provider namespaces prevent unrelated IDs from colliding. A provider crosswalk needs evidence and its own documented policy.
- Keep optional Seoul and neural dependencies lazy. Importing and using the core must not require openpyxl, PyTorch, a network connection, API keys, real datasets, or trained weights.

## Implementation and verification

Proceed with ordinary local coding, documentation, and tests within the requested scope. Prefer small changes that can be reviewed and verified. Use existing interfaces where they fit; do not add a city-specific subsystem merely because a generic loader can read its feed.

Install development tools with `python -m pip install -e ".[dev]"`. Add the `seoul` or `torch` extra when testing that optional functionality. Use `python -m pytest -q` and `python -m ruff check .`; run targeted tests while iterating. For distribution changes, also build with `python -m build` and inspect wheel and source-archive contents. State any optional checks that were skipped and why.

Write meaningful synthetic tests for changed behavior. Matching changes should be checked against exhaustive `compare_routes` results, including ranges, loops, ties, reverse direction, modes, namespaces, and filters. Parser changes should cover source order, invalid input, missing fields, and explicit exclusion reports. Do not add network downloads to the ordinary test suite.

Keep experimental neural code in `djgtok.experimental`. Preserve baseline comparisons and split route groups before producing training views. Do not reinterpret a failed experiment as a successful feature. Record metric definitions, denominators, seeds, corpus size, duplicate checks, and runtime environment for performance claims.

## Sources and release contents

Use independently written code and synthetic fixtures in public contributions. Cite research inspiration without copying code, weights, figures, or paper text. Dependencies belong in package metadata; do not vendor them casually.

The repository and distributions exclude raw/processed provider feeds, real route records, checkpoints, private experiment logs, credentials, and personal filesystem paths. Obtain authorized data separately for local integration checks. If redistribution permission is unclear, leave the material out and document the unresolved point. The code's MIT license does not license external data.

Record provider URLs, applicable terms, snapshot identifiers, hashes, transformations, and exclusions when reporting real-data work. Use aggregate public summaries with source attribution. See `THIRD_PARTY_NOTICES.md`; provider branding and endorsement are separate from data access.

Finish by explaining what changed, the commands and results used to verify it, and any remaining limitation. Keep claims proportional to evidence: synthetic tests prove contracts, historical experiments describe their own setup, and an import alone does not demonstrate city-wide accuracy or feasible route planning.
