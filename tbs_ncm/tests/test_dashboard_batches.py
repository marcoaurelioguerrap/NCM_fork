"""Tests for evaluation-batch grouping in the dashboard.

Run with::

    python -m unittest discover -s tbs_ncm/tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tbs_ncm.batches import attach_batches, build_index, short_label


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class BatchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.now = time.time()

        _write(
            self.root / "personas" / "league_test" / "report.json",
            {
                "generator": "generator.pt",
                "decode": "legal",
                "maps": ["border_skirmish"],
                "games": [
                    {
                        "seed": 1,
                        "map": "border_skirmish",
                        "pair": ["aggressive", "defensive"],
                    }
                ],
            },
        )
        _write(
            self.root / "personas" / "mixed_league_test" / "report.json",
            {
                "ncm_personas": ["aggressive"],
                "heuristics": [{"name": "heuristic-1", "seed": 900001}],
                "maps": ["border_skirmish"],
                "games": 1,
                "generator": "mixed_generator.pt",
                "decode": "legal",
                "max_turns": 20,
                "seconds": 10,
            },
        )
        self.arena = _write(
            self.root / "vs_ncm" / "run1" / "iter1" / "arena_baseline.json",
            {
                "label": "vs-ncm-run1-i1-baseline",
                "matches": [{"scenario": "border_skirmish"}],
            },
        )
        self.selfplay = _write(
            self.root / "vs_ncm" / "run1" / "iter1" / "selfplay.summary.json",
            {"games": [{"seed": 1}]},
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_exact_league_match_and_public_metadata(self) -> None:
        rows = [
            {
                "seed": 1,
                "scenario": "border_skirmish",
                "learner_persona": "aggressive",
                "opponent_persona": "defensive",
            }
        ]
        batches = attach_batches(rows, root=self.root)
        self.assertEqual(rows[0]["batch"], "personas/league_test")
        self.assertEqual(rows[0]["batch_short"], "test")
        self.assertEqual(batches[0]["assigned"], 1)
        self.assertEqual(batches[0]["expected_games"], 1)
        self.assertEqual(batches[0]["generator"], "generator.pt")

    def test_mixed_league_time_window_and_explicit_batch(self) -> None:
        started = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(self.now - 5)
        )
        rows = [
            {
                "seed": 99,
                "scenario": "border_skirmish",
                "learner_persona": "aggressive",
                "opponent_persona": "heuristic-1",
                "started_at": started,
                "file": "mixed-test.json",
            },
            {
                "seed": 100,
                "scenario": "border_skirmish",
                "learner_persona": "aggressive",
                "opponent_persona": "heuristic-1",
                # Explicit metadata wins even outside the report window.
                "batch": "mixed_league_test",
                "mtime": self.now - 3600,
                "file": "mixed-test-2.json",
            },
        ]
        batches = attach_batches(rows, root=self.root)
        self.assertEqual(rows[0]["batch"], "personas/mixed_league_test")
        self.assertEqual(rows[1]["batch"], "personas/mixed_league_test")
        mixed = next(batch for batch in batches if batch["kind"] == "mixed_league")
        self.assertEqual(mixed["assigned"], 2)

    def test_vs_ncm_and_selfplay_paths(self) -> None:
        rows = [
            {"arena_path": str(self.arena)},
            {"batch_path": str(self.selfplay)},
        ]
        batches = attach_batches(rows, root=self.root)
        self.assertEqual(rows[0]["batch"], "vs_ncm/run1/iter1/baseline")
        self.assertEqual(rows[1]["batch"], "vs_ncm/run1/iter1/selfplay")
        kinds = {batch["id"]: batch["kind"] for batch in batches}
        self.assertEqual(kinds["vs_ncm/run1/iter1/baseline"], "vs_ncm")
        self.assertEqual(kinds["vs_ncm/run1/iter1/selfplay"], "selfplay")

    def test_unmatched_rows_stay_unbatched(self) -> None:
        rows = [{"seed": 123, "scenario": "border_skirmish"}]
        attach_batches(rows, root=self.root)
        self.assertIsNone(rows[0]["batch"])

    def test_short_labels_keep_the_distinguishing_part(self) -> None:
        self.assertEqual(short_label("mixed_league_20261025"), "mixed 20261025")
        self.assertEqual(short_label("mixed_league_all_maps"), "mixed all maps")
        self.assertEqual(short_label("league_legal_20261024_final"), "legal final")
        self.assertEqual(short_label("top_round_robin_40"), "RR 40")

    def test_index_cache_tracks_new_reports(self) -> None:
        first = build_index(self.root)
        self.assertEqual(len(first["batches"]), 4)
        _write(
            self.root / "personas" / "league_new" / "report.json",
            {
                "generator": "generator.pt",
                "decode": "legal",
                "maps": ["border_skirmish"],
                "games": [
                    {
                        "seed": 2,
                        "map": "border_skirmish",
                        "pair": ["aggressive", "defensive"],
                    }
                ],
            },
        )
        second = build_index(self.root)
        self.assertEqual(len(second["batches"]), 5)
        self.assertIn("personas/league_new", second["by_id"])


if __name__ == "__main__":
    unittest.main()
