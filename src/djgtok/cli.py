"""Small offline CLI for validated interchange, tokenization and exact matching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from djgtok import Journey, OverlapIndex, TransitTokenizer, compare_routes
from djgtok.demo import synthetic_corpus
from djgtok.io import load_gtfs, load_journeys, load_seoul, save_journeys


def _journey(path: str) -> Journey:
    return Journey.model_validate_json(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DJGtok: directed transit tokens and exact shared corridors")
    parser.add_argument("--version", action="version", version="DJGtok 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run an invented example without downloads or training")
    compare = commands.add_parser("compare", help="Compare two Journey JSON objects")
    compare.add_argument("query")
    compare.add_argument("document")
    search = commands.add_parser("search", help="Search a Journey JSONL corpus using canonical stop IDs")
    search.add_argument("corpus")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--min-shared-edges", type=int, default=1)
    search.add_argument("--exclude-id", action="append", default=[])
    fit = commands.add_parser("fit-tokenizer", help="Fit on a user-chosen training JSONL corpus")
    fit.add_argument("corpus")
    fit.add_argument("output")
    tokenize = commands.add_parser("tokenize", help="Encode a Journey JSON using a saved tokenizer")
    tokenize.add_argument("tokenizer")
    tokenize.add_argument("query")
    gtfs = commands.add_parser("import-gtfs", help="Import an authorized local GTFS ZIP; no download")
    gtfs.add_argument("feed")
    gtfs.add_argument("output")
    gtfs.add_argument("--namespace", required=True)
    seoul = commands.add_parser("import-seoul", help="Import a local OA-1095 XLSX; needs the seoul extra")
    seoul.add_argument("workbook")
    seoul.add_argument("output")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            query, corpus = synthetic_corpus()
            tokenizer = TransitTokenizer.fit(corpus)
            ids, features = tokenizer.encode(query)
            result = {"data": "invented example, not an operating transit service",
                      "token_ids_shape": list(ids.shape), "features_shape": list(features.shape),
                      "matches": [hit.model_dump() for hit in OverlapIndex(corpus).search(query)]}
        elif args.command == "compare":
            result = compare_routes(_journey(args.query), _journey(args.document)).model_dump()
        elif args.command == "search":
            index = OverlapIndex(load_journeys(args.corpus))
            result = [hit.model_dump() for hit in index.search(
                _journey(args.query), top_k=args.top_k, min_shared_edges=args.min_shared_edges,
                exclude_ids=args.exclude_id)]
        elif args.command == "fit-tokenizer":
            tokenizer = TransitTokenizer.fit(load_journeys(args.corpus))
            tokenizer.save(args.output)
            result = {"vocab_size": tokenizer.vocab_size, "feature_dim": tokenizer.feature_dim,
                      "fingerprint": tokenizer.fingerprint}
        elif args.command == "tokenize":
            tokenizer = TransitTokenizer.load(args.tokenizer)
            ids, features = tokenizer.encode(_journey(args.query))
            result = {"token_ids": ids.tolist(), "features": features.tolist(),
                      "tokenizer_fingerprint": tokenizer.fingerprint}
        else:
            if args.command == "import-gtfs":
                journeys, result = load_gtfs(args.feed, namespace=args.namespace)
            else:
                journeys, result = load_seoul(args.workbook)
            save_journeys(journeys, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(2, f"djgtok: {error}\n")


if __name__ == "__main__":
    main()
