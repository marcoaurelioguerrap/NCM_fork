"""Legal-order sets and nearest-legal projection for the TBS codec.

The NCM sees a lossy state string, not the engine's option lists, so a sampled
line can be well formed yet disagree with the current view.  This module builds
the canonical set of orders the view would accept and projects a generated line
onto the closest member of that set.

It deliberately stays outside the game repository: the engine and its codecs are
still the authority, this is only an inference-time repair layer.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from .common import GAME_REPO  # noqa: F401  (puts the game repo on sys.path)

from tbs.agents.codec import CodecError, decode_order, encode_order

__all__ = [
    "NearestLegal",
    "legal_lines",
    "line_similarity",
    "nearest_legal_order",
]

ACTION_CODES = frozenset("mabcluh")


@dataclass(frozen=True)
class NearestLegal:
    """The closest canonical legal line and its decoded order."""

    line: str
    order: dict
    score: float


def legal_lines(view: dict) -> list[str]:
    """Every order the engine's own view would accept, as canonical lines.

    Stored next to an example only when ``--record-legal`` is used, so the
    generator evaluation can report "legal on the first try" without replaying
    games: the compact state text is lossy and cannot rebuild ``unit_actions``.
    """
    lines: list[str] = []
    for option in view.get("build_options", []):
        if option.get("affordable"):
            lines.append(
                encode_order(
                    {
                        "action": "build",
                        "site": list(option["site"]),
                        "unit": option["unit"],
                    }
                )
            )
    for unit_id, options in sorted((view.get("unit_actions") or {}).items()):
        if not isinstance(options, dict):
            continue
        for destination in options.get("move_to") or []:
            lines.append(
                encode_order(
                    {"action": "move", "unit": unit_id, "to": list(destination[0])}
                )
            )
        for attack in options.get("attack") or []:
            lines.append(
                encode_order(
                    {
                        "action": "attack",
                        "unit": unit_id,
                        "target": attack[0],
                        "weapon": attack[1],
                    }
                )
            )
        if options.get("capture_remaining_hp") is not None:
            lines.append(encode_order({"action": "capture", "unit": unit_id}))
        for cargo in options.get("load") or []:
            lines.append(
                encode_order({"action": "load", "unit": unit_id, "cargo": cargo})
            )
        for cargo, at in options.get("unload") or []:
            lines.append(
                encode_order(
                    {
                        "action": "unload",
                        "unit": unit_id,
                        "cargo": cargo,
                        "target": list(at),
                    }
                )
            )
        if options.get("hold"):
            lines.append(encode_order({"action": "hold", "unit": unit_id}))
    return sorted(set(lines))


def line_similarity(left: str, right: str) -> float:
    """A character plus token similarity in ``[0, 1]``.

    Character distance catches spelling noise ("tanf" / "tank"); token overlap
    keeps the chosen action, site, unit and coordinate together so a snap does
    not migrate to an unrelated order with the same prefix.
    """
    char_ratio = difflib.SequenceMatcher(None, left, right).ratio()
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if left_tokens or right_tokens:
        token_ratio = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    else:
        token_ratio = 0.0
    return 0.6 * char_ratio + 0.4 * token_ratio


def nearest_legal_order(
    view: dict,
    line: str,
    *,
    allowed: list[str] | tuple[str, ...] | set[str] | None = None,
    min_score: float = 0.45,
) -> NearestLegal | None:
    """Project ``line`` onto the closest canonical order the view accepts.

    ``allowed`` lets a caller reuse one legal set across retries.  When the
    generated line starts with a known action code the search stays inside that
    action, which avoids turning a botched attack into a build because one
    character happened to match.  The result is decoded and rechecked, so a
    canonical line that no longer parses is never returned.
    """
    candidates = list(allowed) if allowed is not None else legal_lines(view)
    line = str(line or "").strip()
    if not line or not candidates:
        return None

    first = line.split()[0] if line.split() else ""
    if first in ACTION_CODES:
        same_action = [candidate for candidate in candidates if candidate.split()[0] == first]
        if same_action:
            candidates = same_action

    best_line = ""
    best_score = -1.0
    for candidate in candidates:
        score = line_similarity(line, candidate)
        if score > best_score:
            best_line = candidate
            best_score = score
    if not best_line or best_score < min_score:
        return None
    try:
        order = decode_order(best_line)
    except CodecError:
        return None
    return NearestLegal(best_line, order, best_score)
