"""Small trainable sequence encoder, used identically for query and corpus."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from djgtok.schema import Journey
from djgtok.tokenizer import TransitTokenizer


@dataclass
class EncoderConfig:
    hidden_dim: int = 96
    output_dim: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1


class TransitEncoder(nn.Module):
    def __init__(self, tokenizer: TransitTokenizer, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(tokenizer.vocab_size, config.hidden_dim, padding_idx=0)
        nn.init.normal_(self.tokens.weight, std=0.02)
        with torch.no_grad():
            self.tokens.weight[0].zero_()
        self.continuous = nn.Linear(tokenizer.feature_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(config.hidden_dim, config.heads, config.hidden_dim*2,
                                            config.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.projection = nn.Sequential(nn.LayerNorm(config.hidden_dim), nn.Linear(config.hidden_dim, config.output_dim))

    def forward(self, ids: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = self.continuous(features) + self.tokens(ids).mean(dim=2)
        values = self.context(values, src_key_padding_mask=~mask)
        pooled = (values * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(self.projection(pooled), p=2, dim=-1)


def collate_encoded(encoded: list[tuple[np.ndarray, np.ndarray]], device: str | torch.device,
                    token_mask_rate: float = 0.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not encoded:
        raise ValueError("Cannot collate an empty batch")
    longest = max(len(ids) for ids, _ in encoded)
    ids = np.zeros((len(encoded), longest, encoded[0][0].shape[1]), dtype=np.int64)
    features = np.zeros((len(encoded), longest, encoded[0][1].shape[1]), dtype=np.float32)
    mask = np.zeros((len(encoded), longest), dtype=bool)
    for i, (item_ids, item_features) in enumerate(encoded):
        n = len(item_ids)
        ids[i, :n], features[i, :n], mask[i, :n] = item_ids, item_features, True
    ti = torch.as_tensor(ids, device=device)
    if token_mask_rate:
        drop = torch.rand(ti.shape, device=device) < token_mask_rate
        ti = ti.masked_fill(drop & (ti != 0), 1)
    return ti, torch.as_tensor(features, device=device), torch.as_tensor(mask, device=device)


class EncoderBundle:
    def __init__(self, model: TransitEncoder, tokenizer: TransitTokenizer, *, device: str = "cpu",
                 fingerprint: str = "unsaved", training_metadata: dict | None = None):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self._fingerprint = fingerprint
        self.training_metadata = training_metadata or {}
        self._saved_state_digest = self._state_digest()

    def _state_digest(self) -> str:
        """Bind configuration, tokenizer and actual tensor bytes, including .data edits."""
        digest = hashlib.sha256(json.dumps(
            {"config": asdict(self.model.config), "tokenizer": self.tokenizer.to_dict()},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode())
        for name, value in sorted(self.model.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        if self._fingerprint != "unsaved" and self._state_digest() != self._saved_state_digest:
            raise ValueError("Encoder or tokenizer changed after saving; save again and rebuild the index")
        return self._fingerprint

    @property
    def dimension(self) -> int:
        return self.model.config.output_dim

    @torch.inference_mode()
    def encode(self, journeys: list[Journey], batch_size: int = 128) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model.eval()
        output = []
        for start in range(0, len(journeys), batch_size):
            items = [self.tokenizer.encode(j) for j in journeys[start:start+batch_size]]
            batch = collate_encoded(items, self.device)
            output.append(self.model(*batch).cpu().numpy())
        result = np.concatenate(output) if output else np.empty((0, self.dimension), dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Encoder produced nonfinite embeddings")
        return result.astype(np.float32, copy=False)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Only tensors/primitives: safe weights_only loading, no serialized modules.
        content = {"format_version": 1, "config": asdict(self.model.config),
                   "tokenizer": self.tokenizer.to_dict(),
                   "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                   "training_metadata": self.training_metadata}
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(content, tmp)
        tmp.replace(path)
        self._fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        self._saved_state_digest = self._state_digest()

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> EncoderBundle:
        path = Path(path)
        content = torch.load(path, map_location="cpu", weights_only=True)
        if content.get("format_version") != 1:
            raise ValueError("Unsupported model format")
        tokenizer = TransitTokenizer.from_dict(content["tokenizer"])
        model = TransitEncoder(tokenizer, EncoderConfig(**content["config"]))
        model.load_state_dict(content["state_dict"], strict=True)
        return cls(model, tokenizer, device=device, fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(),
                   training_metadata=content["training_metadata"])
