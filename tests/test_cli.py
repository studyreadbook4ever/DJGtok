"""User-facing offline installation workflows and useful input failures."""

import json
import subprocess
import sys

from djgtok.demo import synthetic_corpus
from djgtok.schema import save_journeys


def run(*args):
    return subprocess.run([sys.executable, "-m", "djgtok", *map(str, args)],
                          capture_output=True, text=True, check=False)


def test_offline_demo_and_no_eager_optional_dependencies():
    result = run("demo")
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["token_ids_shape"] == [8, 3]
    assert output["features_shape"] == [8, 42]
    assert [hit["score"] for hit in output["matches"]] == [7, 2]
    check = subprocess.run([sys.executable, "-c",
                            "import sys, djgtok, djgtok.io; "
                            "assert 'torch' not in sys.modules; assert 'openpyxl' not in sys.modules"],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_json_interchange_fit_reload_and_search(tmp_path):
    query, corpus = synthetic_corpus()
    corpus_path, query_path = tmp_path / "corpus.jsonl", tmp_path / "query.json"
    save_journeys(corpus, corpus_path)
    query_path.write_text(query.model_dump_json(), encoding="utf-8")
    tokenizer_path = tmp_path / "tokenizer.json"
    fitted = run("fit-tokenizer", corpus_path, tokenizer_path)
    assert fitted.returncode == 0, fitted.stderr
    encoded = run("tokenize", tokenizer_path, query_path)
    assert encoded.returncode == 0, encoded.stderr
    assert len(json.loads(encoded.stdout)["features"]) == 8
    found = run("search", corpus_path, query_path, "--min-shared-edges", 3)
    assert found.returncode == 0, found.stderr
    hits = json.loads(found.stdout)
    assert len(hits) == 1 and hits[0]["journey_id"] == "demo:long"
    assert json.loads(fitted.stdout)["fingerprint"] == json.loads(encoded.stdout)["tokenizer_fingerprint"]


def test_invalid_user_input_is_an_error_not_an_empty_success(tmp_path):
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{}", encoding="utf-8")
    result = run("compare", invalid, invalid)
    assert result.returncode == 2
    assert "validation" in result.stderr
    assert "Traceback" not in result.stderr
