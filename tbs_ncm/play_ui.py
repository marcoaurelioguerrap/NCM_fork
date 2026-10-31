"""Launch the game UI with the NCM player.

Run from the NCM repository with its virtualenv (torch and pygame live there):

    python -m tbs_ncm.play_ui --no-menu --agent-side B

The launcher switches to the game repository so the UI's `saves/` directory is
the game's, then hands the rest of the command line to `tbs.ui.app`.
"""

from __future__ import annotations

import os
import sys

from .common import ARTIFACT_DIR, GAME_REPO

if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))

os.chdir(GAME_REPO)

from tbs.ui.app import main  # noqa: E402


def _argv_with_checkpoint(argv: list[str]) -> list[str]:
    """Default to the trained checkpoint when the caller asks for a side."""
    args = list(argv)
    has_side = any(arg == "--agent-side" or arg.startswith("--agent-side=") for arg in args)
    has_checkpoint = any(
        arg == "--ncm-checkpoint" or arg.startswith("--ncm-checkpoint=") for arg in args
    )
    if has_side and not has_checkpoint:
        return ["--ncm-checkpoint", str(ARTIFACT_DIR), *args]
    return args


if __name__ == "__main__":
    raise SystemExit(main(_argv_with_checkpoint(sys.argv[1:])))
