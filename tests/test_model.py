import numpy as np
import pytest

torch = pytest.importorskip("torch")

from djgtok.experimental.model import EncoderBundle, EncoderConfig, TransitEncoder
from djgtok.schema import Journey, reverse_journey, subjourney
from djgtok.tokenizer import TransitTokenizer
from djgtok.experimental.dense import DenseIndex


def journey(n=12):
    return Journey(id="rail", stops=[{"id": str(i), "lat": 42.0 + i*.001, "lon": -71.0 + i*.002}
                                     for i in range(n)], modes=["rail"]*(n-1), line_ids=["R"]*(n-1))


def test_model_roundtrip_and_padding_invariance(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(2)
    j = journey()
    tokenizer = TransitTokenizer.fit([j])
    bundle = EncoderBundle(TransitEncoder(tokenizer, EncoderConfig(hidden_dim=32, output_dim=24, layers=1)), tokenizer)
    short = subjourney(j, 0, 4)
    together = bundle.encode([j, short])
    alone = bundle.encode([short])
    np.testing.assert_allclose(together[1], alone[0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(together, axis=1), 1, atol=1e-6)
    bundle.save(tmp_path / "model.pt")
    loaded = EncoderBundle.load(tmp_path / "model.pt")
    assert loaded.fingerprint == bundle.fingerprint and loaded.fingerprint != "unsaved"
    np.testing.assert_allclose(loaded.encode([j, short]), together, atol=1e-6)


def test_unseen_stop_ids_do_not_change_geometry_embedding():
    torch.manual_seed(3)
    j = journey()
    tokenizer = TransitTokenizer.fit([j])
    bundle = EncoderBundle(TransitEncoder(tokenizer, EncoderConfig(hidden_dim=32, output_dim=24, layers=1)), tokenizer)
    renamed = j.model_copy(update={"stops": [s.model_copy(update={"id": "new:"+s.id}) for s in j.stops]})
    values = bundle.encode([j, renamed, reverse_journey(j)])
    np.testing.assert_array_equal(values[0], values[1])
    assert float(values[0] @ values[2]) < .9999


def test_saved_index_rejects_changed_weights_and_tokenizer(tmp_path):
    j = journey()
    tokenizer = TransitTokenizer.fit([j])
    bundle = EncoderBundle(TransitEncoder(tokenizer, EncoderConfig(hidden_dim=16, output_dim=8, layers=1)), tokenizer)
    with pytest.raises(ValueError, match="Save"):
        DenseIndex.build([j], bundle)
    bundle.save(tmp_path / "model.pt")
    index = DenseIndex.build([j], bundle)
    assert index.search(j, bundle, mode="whole")[0].journey_id == j.id
    # .data bypasses autograd's version counter; byte-level binding still detects it.
    bundle.model.projection[-1].bias.data.add_(0.1)
    with pytest.raises(ValueError, match="changed after saving"):
        index.search(j, bundle)
    bundle.save(tmp_path / "changed.pt")
    with pytest.raises(ValueError, match="fingerprint"):
        index.search(j, bundle)
    second = DenseIndex.build([j], bundle)
    bundle.tokenizer = TransitTokenizer(tokenizer.origin_lat + 0.1, tokenizer.origin_lon, tokenizer.vocab)
    with pytest.raises(ValueError, match="changed after saving"):
        second.search(j, bundle)


def test_dense_index_roundtrip_and_corruption_rejection(tmp_path):
    j = journey(5)
    tokenizer = TransitTokenizer.fit([j])
    bundle = EncoderBundle(TransitEncoder(tokenizer, EncoderConfig(hidden_dim=16, output_dim=8, layers=1)), tokenizer)
    bundle.save(tmp_path / "model.pt")
    index = DenseIndex.build([j], bundle)
    index.save(tmp_path / "index")
    restored = DenseIndex.load(tmp_path / "index", bundle)
    assert restored.search(j, bundle, mode="whole")[0].score == pytest.approx(1.0)
    path = tmp_path / "index" / "arrays.npy"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        DenseIndex.load(tmp_path / "index", bundle)
