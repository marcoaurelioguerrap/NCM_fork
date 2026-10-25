"""Unit tests for the legal projection layer and the NCM rerank loop.

Run with::

    python -m unittest discover -s tbs_ncm/tests

They use a synthetic view, so they do not need a checkpoint or a GPU.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tbs_ncm.common  # noqa: F401  (adds the game repo to sys.path)

from tbs.agents.codec import encode_order
from tbs.agents.protocol import AgentPlan
from tbs_ncm.legal import legal_lines, nearest_legal_order
from tbs_ncm.ncm_agent import (
    candidate_for_line,
    order_is_legal,
    repair_in_view,
    resolve_unit_id,
)


def make_view() -> dict:
    return {
        "current_player": "A",
        "build_options": [
            {"site": [1, 1], "unit": "tank", "cost": 700, "affordable": True},
            {"site": [1, 1], "unit": "infantry", "cost": 300, "affordable": True},
            {"site": [2, 2], "unit": "tank", "cost": 700, "affordable": False},
        ],
        "unit_actions": {
            "A1": {
                "move_to": [[[1, 1], 1], [[2, 1], 1]],
                "attack": [["B1", "rifle", 0.5, 4]],
                "capture_remaining_hp": 10,
                "hold": True,
            },
            "A2": {
                "move_to": [[[5, 5], 2], [[5, 6], 2]],
                "hold": True,
            },
            "A3": {"acted": True},
            "A4": {"new_unit": True},
        },
        "units": [
            {"id": "A1", "owner": "A", "type": "infantry", "pos": [0, 0]},
            {"id": "A2", "owner": "A", "type": "tank", "pos": [5, 4]},
            {"id": "B1", "owner": "B", "type": "infantry", "pos": [1, 1]},
        ],
    }


class LegalSetTests(unittest.TestCase):
    def test_legal_lines_skip_unavailable_and_unaffordable(self):
        lines = set(legal_lines(make_view()))
        self.assertIn("h A1", lines)
        self.assertIn("m A2 5,5", lines)
        self.assertIn("b 1,1 tank", lines)
        self.assertNotIn("h A3", lines)
        self.assertNotIn("h A4", lines)
        self.assertNotIn("b 2,2 tank", lines)

    def test_resolve_unit_id_uses_numeric_neighbour(self):
        view = make_view()
        self.assertEqual(resolve_unit_id(view, "A9"), "A2")
        self.assertIsNone(resolve_unit_id(view, "B9"))

    def test_nearest_legal_projects_spelling_and_coordinate(self):
        view = make_view()
        nearest = nearest_legal_order(view, "m A1 9,9")
        self.assertIsNotNone(nearest)
        self.assertTrue(order_is_legal(view, nearest.order))

    def test_repair_same_unit_action_and_unit_snap(self):
        view = make_view()
        move = repair_in_view(view, {"action": "move", "unit": "A1", "to": [9, 9]})
        self.assertEqual(move["action"], "move")
        self.assertEqual(move["unit"], "A1")
        self.assertIn(move["to"], ([1, 1], [2, 1]))
        snapped = repair_in_view(view, {"action": "move", "unit": "A9", "to": [5, 6]})
        self.assertEqual(snapped, {"action": "move", "unit": "A2", "to": [5, 6]})

    def test_repair_action_chain_and_build_spelling(self):
        view = make_view()
        # A2 cannot attack, but it can hold: attack -> capture -> hold -> move.
        repair = repair_in_view(view, {"action": "attack", "unit": "A2", "target": "B1"})
        self.assertEqual(repair, {"action": "hold", "unit": "A2"})
        build = repair_in_view(view, {"action": "build", "site": [1, 1], "unit": "tanf"})
        self.assertEqual(build, {"action": "build", "site": [1, 1], "unit": "tank"})

    def test_candidate_prefers_exact_then_nearest(self):
        view = make_view()
        exact = candidate_for_line(view, "m A1 2,1")
        self.assertEqual(exact.kind, "generated")
        broken = candidate_for_line(view, "m A1 9,9")
        self.assertIn(broken.kind, ("same-unit", "nearest"))
        self.assertTrue(order_is_legal(view, broken.order))
        garbage = candidate_for_line(view, "zz")
        self.assertIsNone(garbage)


class StubSampler:
    prefix_len = 2048

    def __init__(self, lines):
        self.lines = list(lines)

    def fits(self, text):
        return True

    def sample(self, text, temperature=0.0):
        return self.lines.pop(0)


class StubFallback:
    def plan(self, request):
        return AgentPlan(orders=[{"action": "hold", "unit": "A1"}])


class AgentRerankTests(unittest.TestCase):
    def request(self, feedback=None):
        return SimpleNamespace(view=make_view(), feedback=feedback or [], player="A")

    def test_rerank_keeps_sampling_until_exact(self):
        from tbs_ncm.ncm_agent import NcmAgent

        agent = NcmAgent(StubSampler(["zz", "m A1 1,1"]), retries=3, fallback=StubFallback())
        with patch("tbs_ncm.ncm_agent.encode_state", return_value="state"):
            plan = agent.plan(self.request())
        self.assertEqual(plan.orders, [{"action": "move", "unit": "A1", "to": [1, 1]}])
        self.assertEqual(agent.generated, 1)
        self.assertEqual(agent.samples, 2)
        self.assertEqual(agent.fallbacks, 0)

    def test_all_garbage_falls_back_with_reason(self):
        from tbs_ncm.ncm_agent import NcmAgent

        agent = NcmAgent(StubSampler(["zz", "qq"]), retries=1, fallback=StubFallback())
        with patch("tbs_ncm.ncm_agent.encode_state", return_value="state"):
            plan = agent.plan(self.request())
        self.assertEqual(plan.orders, [{"action": "hold", "unit": "A1"}])
        self.assertEqual(agent.fallbacks, 1)
        self.assertEqual(agent.fallback_reasons, ["no-legal-sample"])

    def test_overlong_state_is_reported_before_sampling(self):
        from tbs_ncm.ncm_agent import NcmAgent

        class TooLongSampler(StubSampler):
            def fits(self, text):
                return False

        agent = NcmAgent(TooLongSampler(["h A1"]), fallback=StubFallback())
        with patch("tbs_ncm.ncm_agent.encode_state", return_value="state"):
            plan = agent.plan(self.request())
        self.assertEqual(plan.orders, [{"action": "hold", "unit": "A1"}])
        self.assertEqual(agent.samples, 0)
        self.assertEqual(agent.state_too_long, 1)
        self.assertEqual(agent.fallback_reasons, ["state-too-long"])

    def test_feedback_avoids_the_rejected_order(self):
        from tbs_ncm.ncm_agent import NcmAgent

        agent = NcmAgent(StubSampler(["m A1 1,1", "m A1 1,1", "h A1"]), retries=3, fallback=StubFallback())
        with patch("tbs_ncm.ncm_agent.encode_state", return_value="state"):
            agent.plan(self.request())
            plan = agent.plan(self.request(feedback=["engine rejected the last order"]))
        self.assertEqual(plan.orders, [{"action": "hold", "unit": "A1"}])
        self.assertEqual(agent.rejected, 1)
        self.assertEqual(agent.recovered, 1)
        self.assertEqual(agent.fallbacks, 0)


if __name__ == "__main__":
    unittest.main()
