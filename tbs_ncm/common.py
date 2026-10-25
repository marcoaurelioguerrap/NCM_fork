"""Shared paths and helpers for the TBS/NCM bridge.

The game engine and its agent codecs live in a separate repository.  They are
pure stdlib, so this package can import them from the NCM virtualenv without
pulling in pygame or the UI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
NCM_ROOT = HERE.parent
GAME_REPO = Path(
    os.environ.get(
        "TBS_GAME_REPO",
        r"C:\Users\marco\OneDrive\Documentos\opencode_code\turn_based_strategy_vibe",
    )
)
DATA_DIR = HERE / "data"
ARTIFACT_DIR = NCM_ROOT / "artifacts" / "tbs"
CONFIG_PATH = HERE / "config.yaml"

if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))


def load_config(path: Path | str = CONFIG_PATH):
    return OmegaConf.load(str(path))


def artifact_path(name: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / name


def load_generator_state(net, state: dict) -> None:
    """Load an NCM generator state dict across prefix-length changes.

    ``attn_mask_for_masking`` is derived from ``prefix_len`` and used to be a
    persistent buffer.  Drop it when present so a checkpoint trained at one
    context length can initialize a model at another; every other key must
    match exactly.
    """
    filtered = {key: value for key, value in state.items() if key != "attn_mask_for_masking"}
    net.load_state_dict(filtered, strict=True)
