"""A small learned action selector for the NCM's two-stage policy.

The full NCM generator is good at copying the arguments of an order but it
collapses the rare leading action character (`l`/`u`) to `m`.  This module
learns just that character from the compact state prefix.  At inference the
NCM can then ask this model *which action* to take and use legal candidate
lines for the arguments, instead of relying on a scripted ferry advisor.

The model is deliberately tiny: deterministic character n-gram features through
an EmbeddingBag and a linear classifier.  It trains in seconds on a GPU/CPU and
keeps the deployed NCM's dependency surface unchanged beyond torch, which the
NCM already uses.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ACTION_CLASSES",
    "ActionModel",
    "CharNgramActionModel",
    "feature_ids",
    "load_jsonl",
    "train_action_model",
]

ACTION_CLASSES: tuple[str, ...] = ("m", "a", "c", "l", "u", "b", "h", "e", "p", "s", "r")
_CLASS_INDEX = {char: index for index, char in enumerate(ACTION_CLASSES)}
_NGRAM = 4
_BUCKETS = 16384


def feature_ids(text: str, *, ngram: int = _NGRAM, buckets: int = _BUCKETS) -> torch.Tensor:
    """Deterministic hashed character n-grams for one state prefix."""
    compact = str(text).replace("\n", " ")
    if len(compact) < ngram:
        grams = [compact or " "]
    else:
        grams = [compact[index : index + ngram] for index in range(len(compact) - ngram + 1)]
    ids = [zlib.crc32(gram.encode("utf-8")) % buckets for gram in grams]
    return torch.tensor(ids, dtype=torch.long)


class CharNgramActionModel(nn.Module):
    def __init__(self, buckets: int = _BUCKETS, dimension: int = 128, classes: int = len(ACTION_CLASSES)):
        super().__init__()
        self.buckets = int(buckets)
        self.dimension = int(dimension)
        self.embedding = nn.EmbeddingBag(buckets, dimension, mode="mean", padding_idx=0)
        self.classifier = nn.Linear(dimension, classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(features))


def action_label(row: dict[str, Any]) -> str:
    """The action class to learn for one training row.

    ``ferry_action`` turns a plain transport move into a stage label (pickup,
    sail or reposition) so the selector can choose the ferry stage, not just a
    generic ``m``.
    """
    target_action = str(row.get("target", ""))[:1] or "m"
    if target_action in ("l", "u", "c"):
        return target_action
    ferry_action = str(row.get("ferry_action") or "")
    if ferry_action == "pickup":
        return "p"
    if ferry_action == "sail":
        return "s"
    if ferry_action == "reposition":
        return "r"
    if ferry_action == "load":
        return "l" if target_action == "l" else "p"
    if ferry_action == "unload":
        return "u"
    return target_action


def _batch(rows: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    features = [feature_ids(row["prefix"]) for row in rows]
    padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0)
    labels = torch.tensor(
        [_CLASS_INDEX.get(action_label(row), 0) for row in rows],
        dtype=torch.long,
    )
    return padded, labels


def load_jsonl(path: Path | str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _class_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = Counter(action_label(row) for row in rows)
    total = sum(counts.values())
    raw = []
    for char in ACTION_CLASSES:
        count = max(1, counts.get(char, 0))
        raw.append(min(5.0, math.sqrt(total / (len(ACTION_CLASSES) * count))))
    return torch.tensor(raw, dtype=torch.float32, device=device)


def train_action_model(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    *,
    out: Path,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    device: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train the action selector and write a small checkpoint."""
    random.seed(seed)
    torch.manual_seed(seed)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = CharNgramActionModel().to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    weights = _class_weights(train_rows, resolved)

    history: list[dict[str, Any]] = []
    for epoch in range(1, max(1, int(epochs)) + 1):
        random.shuffle(train_rows)
        model.train()
        for start in range(0, len(train_rows), batch_size):
            batch = train_rows[start : start + batch_size]
            features, labels = _batch(batch)
            features = features.to(resolved)
            labels = labels.to(resolved)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(features), labels, weight=weights)
            loss.backward()
            optimizer.step()
        metrics = evaluate_action_model(model, validation_rows, device=str(resolved))
        history.append({"epoch": epoch, **metrics})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch {epoch}: accuracy {metrics['accuracy']:.3f}  "
                + "  ".join(
                    f"{char}:{hit}/{total}"
                    for char, (hit, total) in sorted(metrics["per_class"].items())
                )
            )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": model.state_dict(),
            "classes": ACTION_CLASSES,
            "buckets": model.buckets,
            "dimension": model.dimension,
            "ngram": _NGRAM,
        },
        out,
    )
    return {"history": history, "best": history[-1] if history else {}, "path": str(out)}


@torch.no_grad()
def evaluate_action_model(
    model: CharNgramActionModel,
    rows: list[dict[str, Any]],
    *,
    device: str = "cpu",
    batch_size: int = 128,
) -> dict[str, Any]:
    resolved = torch.device(device)
    model.eval()
    correct = 0
    total = 0
    hit: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        features, labels = _batch(batch)
        predictions = model(features.to(resolved)).argmax(dim=-1).cpu()
        for label, prediction in zip(labels.tolist(), predictions.tolist()):
            char = ACTION_CLASSES[label]
            counts[char] += 1
            hit[char] += int(label == prediction)
            correct += int(label == prediction)
            total += 1
    return {
        "accuracy": correct / max(1, total),
        "per_class": {char: (hit[char], counts[char]) for char in sorted(counts)},
        "n": total,
    }


class ActionModel:
    """Loaded inference wrapper used by :class:`NcmAgent`."""

    def __init__(self, path: Path | str, device: str | None = None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.classes = tuple(checkpoint.get("classes") or ACTION_CLASSES)
        self.buckets = int(checkpoint.get("buckets") or _BUCKETS)
        self.dimension = int(checkpoint.get("dimension") or 128)
        self.ngram = int(checkpoint.get("ngram") or _NGRAM)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = CharNgramActionModel(
            buckets=self.buckets,
            dimension=self.dimension,
            classes=len(self.classes),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()

    @torch.no_grad()
    def predict(self, state_text: str) -> str:
        ids = feature_ids(state_text, ngram=self.ngram, buckets=self.buckets).unsqueeze(0)
        index = int(self.model(ids.to(self.device)).argmax(dim=-1).item())
        return self.classes[index]

    @torch.no_grad()
    def probabilities(self, state_text: str) -> dict[str, float]:
        ids = feature_ids(state_text, ngram=self.ngram, buckets=self.buckets).unsqueeze(0)
        probs = torch.softmax(self.model(ids.to(self.device))[0], dim=-1).cpu().tolist()
        return {char: float(prob) for char, prob in zip(self.classes, probs)}
