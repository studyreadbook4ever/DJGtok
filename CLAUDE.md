# DJGtok instructions for Claude Code

Use [AGENTS.md](AGENTS.md) as the shared project guidance. It defines the public API, directed-edge matching contract, validation approach, source policy, and release exclusions. Read it along with the relevant source and tests. This file adds no separate approval process for ordinary local contribution work.

DJGtok's useful starting point is exact matching of ordered stop-ID-and-mode edges. It requires no model training. `TransitTokenizer.fit` builds a regional vocabulary; optional neural experiments remain in `djgtok.experimental`. A shared segment is not proof of a feasible itinerary, operating reverse service, or a valid transfer.

For a first contribution, use one of the bounded prompts in [the city extension guide](docs/extending-cities.md): improve a local-file adapter, strengthen synthetic parser fixtures, verify index equivalence, or evaluate a regional tokenizer change. State which provider format or API contract the change supports before implementing it.

Typical development commands:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
```

Use the `seoul` or `torch` extra for the corresponding optional tests. Core usage must remain independent of them. For packaging changes, run `python -m build` and inspect both artifacts for unintended data or files.

Create synthetic fixtures yourself. Do not add actual feeds, extracted route records, model weights, credentials, private logs, or personal paths. Keep source attribution and uncertainty visible. Report verification results and failures without claiming performance that was not measured. The MIT license covers the project's original code; external data retains its own terms.
