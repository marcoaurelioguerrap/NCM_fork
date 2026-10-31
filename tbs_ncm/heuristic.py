"""A randomized conventional (scripted) AI opponent for NCM training.

Unlike the NCM player, this agent does not use a neural network.  It reuses the
persona decision policy with a seeded random parameter draw, so every instance
has a stable but different doctrine: production mix, attack appetite, risk,
capture priority, defence radius, standoff and movement bias.

    from tbs_ncm.heuristic import HeuristicAgent
    agent = HeuristicAgent(seed=41)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

from .personas import PersonaTeacher, random_persona

__all__ = ["HeuristicAgent"]


class HeuristicAgent:
    """Conventional AI with randomized parameters and NcmAgent-like counters."""

    name = "heuristic"

    def __init__(self, seed: int, name: str | None = None):
        self.seed = int(seed)
        self.persona = random_persona(self.seed, name=name)
        self.name = self.persona.name
        self.teacher = PersonaTeacher(self.persona, seed=self.seed)
        self.params = asdict(self.persona)
        self._reset_counters()

    def _reset_counters(self) -> None:
        self.steps = 0
        self.samples = 0
        self.generated = 0
        self.repaired = 0
        self.repaired_samples = 0
        self.constrained = 0
        self.tactical = 0
        self.fallbacks = 0
        self.illegal = 0
        self.rejected = 0
        self.recovered = 0
        self.state_too_long = 0
        self.generation_errors = 0
        self.lines: list[str] = []
        self.constrained_lines: list[str] = []
        self.fallback_reasons: list[str] = []
        self.repairs: Counter[str] = Counter()
        self.actions: Counter[str] = Counter()
        self.builds: Counter[str] = Counter()
        self.orders = 0
        self.model_actions: Counter[str] = Counter()
        self.model_builds: Counter[str] = Counter()
        self.model_orders = 0
        self.geometry_samples = 0
        self.enemy_distance = 0.0
        self.home_distance = 0.0

    def plan(self, request):
        plan = self.teacher.plan(request)
        if plan.orders:
            order = plan.orders[0]
            action = str(order.get("action"))
            self.steps += 1
            self.orders += 1
            self.actions[action] += 1
            if action == "build":
                self.builds[str(order.get("unit"))] += 1
        return plan
