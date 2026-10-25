"""Launch the game UI with the NCM player.

Run from the NCM repository with its virtualenv (torch and pygame live there):

    python -m tbs_ncm.play_ui --no-menu --agent-side B

The launcher switches to the game repository so the UI's `saves/` directory is
the game's, then hands the rest of the command line to `tbs.ui.app`.
"""

from __future__ import annotations

import os
import sys

from .common import GAME_REPO

if str(GAME_REPO) not in sys.path:
    sys.path.insert(0, str(GAME_REPO))

os.chdir(GAME_REPO)

from tbs.ui.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
