"""Train the NCM's learned action selector.

    python -m tbs_ncm.train_action_clf `
      --train "..\\saves\\ferry_data_train20.jsonl,tbs_ncm/data/train.jsonl" `
      --validation "..\\saves\\ferry_data_val.jsonl,tbs_ncm/data/validation.jsonl" `
      --out artifacts/tbs/action_clf.pt

The selector is a second, much smaller model: it chooses the order's action
character from the state prefix.  The full NCM generator still supplies the
arguments through legal candidate lines.  This removes the scripted ferry
advisor from the deployed player.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .action_model import load_jsonl, train_action_model
from .common import artifact_path


def _paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="comma-separated JSONL paths")
    parser.add_argument("--validation", required=True, help="comma-separated JSONL paths")
    parser.add_argument("--out", default="action_clf.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="")
    args = parser.parse_args(argv)

    train_rows: list[dict] = []
    for path in _paths(args.train):
        rows = load_jsonl(path)
        print(f"train {path}: {len(rows)} rows")
        train_rows.extend(rows)
    validation_rows: list[dict] = []
    for path in _paths(args.validation):
        rows = load_jsonl(path)
        print(f"validation {path}: {len(rows)} rows")
        validation_rows.extend(rows)

    result = train_action_model(
        train_rows,
        validation_rows,
        out=artifact_path(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device or None,
    )
    print(
        f"wrote {result['path']}; final accuracy "
        f"{result['best'].get('accuracy', 0.0):.3f}; "
        f"per class {result['best'].get('per_class')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
