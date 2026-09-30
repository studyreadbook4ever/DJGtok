"""Contrastive training on real transit patterns; validation-only selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from djgtok.experimental.model import EncoderBundle, EncoderConfig, TransitEncoder, collate_encoded
from djgtok.schema import (
    Journey,
    assign_split,
    is_parent_pattern,
    load_journeys,
    reverse_journey,
    split_group,
)
from djgtok.tokenizer import TransitTokenizer


def parent_patterns(journeys: list[Journey]) -> list[Journey]:
    return [j for j in journeys if is_parent_pattern(j)]


class ViewSampler:
    def __init__(self, journeys: list[Journey], tokenizer: TransitTokenizer, seed: int):
        self.journeys = journeys
        self.encoded = [tokenizer.encode(j) for j in journeys]
        self.reversed = [tokenizer.encode(reverse_journey(j)) for j in journeys]
        self.edges = [[(a.id, b.id, m) for a, b, m in zip(j.stops, j.stops[1:], j.modes)] for j in journeys]
        self.rng = random.Random(seed)

    @staticmethod
    def crop(encoded: tuple, start: int, end: int) -> tuple:
        ids, feats = encoded
        feats = feats[start:end].copy()
        # Last 18 = numeric geographic/direction/position/turn/transfer + 7 modes.
        offset = feats.shape[1] - 18
        feats[:, offset + 7] = np.linspace(0, 1, len(feats))
        feats[0, offset + 8:offset + 11] = [1, 0, 0]
        return ids[start:end].copy(), feats

    def sample(self, batch_size: int) -> tuple[list, list, list, np.ndarray]:
        anchors, positives, negatives, a_edges, p_edges, n_edges = [], [], [], [], [], []
        for _ in range(batch_size):
            idx = self.rng.randrange(len(self.journeys))
            n = len(self.edges[idx])
            width = self.rng.randint(min(4, n), min(32, n))
            start = self.rng.randint(0, n-width)
            end = start + width
            shift = max(0, int(width * .15))
            ps = max(0, start + self.rng.randint(-shift, shift))
            pe = min(n, end + self.rng.randint(-shift, shift))
            anchors.append(self.crop(self.encoded[idx], start, end))
            positives.append(self.crop(self.encoded[idx], ps, pe))
            negatives.append(self.crop(self.reversed[idx], n-end, n-start))
            a_edges.append(set(self.edges[idx][start:end]))
            p_edges.append(set(self.edges[idx][ps:pe]))
            n_edges.append({(b, a, m) for a, b, m in self.edges[idx][start:end]})
        # Different journeys may share the same corridor: don't label them negative.
        targets = np.zeros((batch_size, batch_size*2), dtype=np.float32)
        for row, a in enumerate(a_edges):
            for col, b in enumerate(p_edges+n_edges):
                overlap = len(a & b) / max(1, min(len(a), len(b)))
                if overlap >= .7:
                    targets[row, col] = overlap
            targets[row, row] = 1.0
        targets /= targets.sum(axis=1, keepdims=True)
        return anchors, positives, negatives, targets


def _loss(model: TransitEncoder, batch: tuple, device: str, *, augment: bool) -> torch.Tensor:
    anchors, positives, negatives, targets = batch
    encoded = model(*collate_encoded(anchors + positives + negatives, device, .15 if augment else 0))
    batch_size = len(anchors)
    scores = encoded[:batch_size] @ encoded[batch_size:].T / .10
    target = torch.as_tensor(targets, device=device)
    return -(target * F.log_softmax(scores, dim=1)).sum(dim=1).mean()


def train(data_path: str | Path, output_dir: str | Path, *, steps: int = 600, batch_size: int = 64,
          seed: int = 42, device: str = "cpu", learning_rate: float = 3e-4) -> dict:
    if steps < 1 or batch_size < 2:
        raise ValueError("steps >= 1 and batch_size >= 2 required")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; choose --device cpu explicitly")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "model.pt").exists():
        raise FileExistsError("Model exists: use a new output directory to retain experiment history")
    patterns = parent_patterns(load_journeys(data_path))
    training = [j for j in patterns if assign_split(j) == "train"]
    validation = [j for j in patterns if assign_split(j) == "val"]
    if not training or not validation:
        raise ValueError("Need nonempty train and validation route groups")
    tokenizer = TransitTokenizer.fit(training)
    model = TransitEncoder(tokenizer, EncoderConfig()).to(device)
    metadata = {"seed": seed, "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
                "dataset_sources": sorted({j.source for j in patterns}),
                "pattern_kinds": sorted({j.metadata.get("kind", "user_supplied") for j in patterns}),
                "train_groups": sorted({split_group(j) for j in training}),
                "val_groups": sorted({split_group(j) for j in validation}),
                "train_pattern_count": len(training), "val_pattern_count": len(validation),
                "split_rule": "sha256 transitvec-v1 route group; 70/15/15 before augmentation",
                "objective": "multi-positive directed corridor contrastive + synthetic reverse negatives",
                "training_steps": 0, "device": device, "selection": "minimum fixed validation contrastive loss",
                "limitations": "planned single-vehicle patterns; no passenger preference or live feasibility labels"}
    bundle = EncoderBundle(model, tokenizer, device=device, training_metadata=metadata)
    bundle.save(output / "untrained.pt")
    sampler = ViewSampler(training, tokenizer, seed)
    val_sampler = ViewSampler(validation, tokenizer, seed + 1)
    val_batches = [val_sampler.sample(batch_size) for _ in range(3)]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    log, best, started = [], float("inf"), time.perf_counter()
    for step in range(steps + 1):
        train_loss = None
        if step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model, sampler.sample(batch_size), device, augment=True)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became nonfinite; experiment failed")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss = float(loss.detach())
        if step % 50 == 0 or step == steps:
            model.eval()
            with torch.inference_mode():
                val_loss = float(np.mean([float(_loss(model, batch, device, augment=False)) for batch in val_batches]))
            if not np.isfinite(val_loss):
                raise RuntimeError("Validation loss became nonfinite; experiment failed")
            record = {"step": step, "train_loss": train_loss, "val_loss": val_loss,
                      "elapsed_s": round(time.perf_counter()-started, 2)}
            log.append(record)
            print(json.dumps(record), flush=True)
            # Require at least one actual optimization step for the learned artifact.
            if step > 0 and val_loss < best:
                best = val_loss
                bundle.training_metadata = {**metadata, "training_steps": step, "validation_loss": val_loss}
                bundle.save(output / "model.pt")
            (output / "training.json").write_text(json.dumps({"metadata": metadata, "history": log,
                "metadata_scope": "initial configuration before optimization",
                "completed_steps": step,
                "best_checkpoint_step": bundle.training_metadata["training_steps"] if step > 0 else None,
                "best_validation_loss": best if np.isfinite(best) else None}, indent=2), encoding="utf-8")
    if not (output / "model.pt").is_file():
        raise RuntimeError("Training ended without a valid learned checkpoint")
    return {"model": str(output / "model.pt"), "best_validation_loss": best,
            "elapsed_s": time.perf_counter()-started, "history": log}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/journeys.jsonl")
    parser.add_argument("--output", default="artifacts/run1")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.data, args.output, steps=args.steps, batch_size=args.batch_size, seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
