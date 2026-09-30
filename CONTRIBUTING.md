# Contributing to DJGtok

Small, verifiable contributions are welcome: synthetic edge cases, importer diagnostics, city/provider adapters, clearer examples and independently reproduced measurements. Read [architecture](docs/architecture.md), [city extensions](docs/extending-cities.md), and [source provenance](docs/provenance.md) first. The project is young; proposals that preserve a small, reusable library are especially useful.

## Develop locally

```sh
git clone https://github.com/studyreadbook4ever/DJGtok.git
cd DJGtok
python -m venv .venv
# Activate with your operating system's usual command.
python -m pip install -e ".[dev,seoul]"
python -m pytest -q
ruff check .
python examples/quickstart.py
python -m build
python -m twine check --strict dist/*
```

Use Python 3.11–3.13. PyTorch tests skip when the `torch` extra is absent; install it to check neural changes. Core CI intentionally runs without either optional dependency. Tests use invented data and do not download transit feeds.

## Changes and evidence

Describe the concrete behavior changed, the reason, the commands you ran and any limits. Matchers need directed, mode-aware, loop and tie examples. Adapter tests should use synthetic fixtures, retaining source order and explaining exclusions instead of inventing missing records. Benchmarks need denominators, corpus size, seeds, evaluation splits and environment details. Improvements to one task should not be described as improvements to a different task.

For a city extension, first determine whether the generic GTFS loader already works. Add a separate adapter only when a real format/semantics difference requires it. Record official source and terms links. Keep feeds, real route lists, credentials, checkpoints and private local logs out of commits. Do not copy third-party code, documentation, figures or examples whose reuse rights are unclear. Dependencies should normally be installed through package metadata.

Claude Code and Codex are welcome development tools. [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) and the city guide include working instructions. Review generated work and verify it as you would any contribution. An assistant's assertion is not a data license or a benchmark result.

Submit a pull request with project-original code under the repository's MIT license. Cite research inspiration and distinguish external data rights. Please avoid attaching provider data to issues or pull requests; a small synthetic reproduction is easier to review and reuse.
