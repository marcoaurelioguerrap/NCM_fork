"""Scripted persona teachers for the TBS NCM.

An NCM has no reward function and no persona input: it imitates the policy that
produced its ``(state, order)`` rows.  A persona is therefore a small, explicit
decision policy used to label data.  Each teacher reads exactly the same view the
NCM player sees at inference time and returns one order per call, so the data it
writes is on-policy for the stepwise coordinator.

The policies are deliberately simple and interpretable.  Their purpose is not to
be strong; it is to be *consistently different* so a cloned NCM develops a
recognisable style.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

from tbs.agents.codec import encode_order, encode_state
from tbs.agents.ferry import plan_ferry
from tbs.agents.protocol import AgentPlan, TurnRequest
from tbs.engine.hexgrid import hex_distance

__all__ = [
    "PERSONAS",
    "Persona",
    "PersonaTeacher",
    "get_persona",
    "persona_names",
    "push_persona",
    "random_persona",
]


@dataclass(frozen=True)
class Persona:
    """One style expressed as weights over legal, view-visible choices."""

    name: str
    title: str = ""
    # Production: unit type -> weight.  Zero/absent means never build that type.
    build_weights: dict[str, float] = field(default_factory=dict)
    build_prob: float = 0.55
    # 0 pulls factories back to the HQ, 1 pushes them toward the enemy.
    build_forward: float = 0.5
    # Combat.
    attack_prob: float = 0.88
    attack_threshold: float = 0.0
    risk_aversion: float = 0.35
    target_bonus: dict[str, float] = field(default_factory=dict)
    hq_bonus: float = 0.8
    capture_prob: float = 0.8
    capture_first: bool = False
    capture_unit_types: tuple[str, ...] = ()
    # Defence.
    defense_radius: int = 0
    retreat_hp: float = 0.0
    home_bias: float = 0.0
    hold_prob: float = 0.0
    # Movement.
    enemy_bias: float = 1.0
    enemy_hq_bias: float = 0.0
    site_bias: float = 1.0
    objective_bias: float = 0.0
    standoff: int = 0
    noise: float = 0.15


#: Five doctrines.  Tuned to differ in builds, target choice, and how far the
#: army is willing to leave home.
PERSONAS: dict[str, Persona] = {
    "aggressive": Persona(
        name="aggressive",
        title="Blitz: attack first, push the enemy HQ",
        build_weights={"tank": 1.0, "artillery": 0.7, "aa_missile": 0.2, "infantry": 0.1},
        build_prob=1.0,
        build_forward=0.9,
        attack_prob=1.0,
        risk_aversion=0.12,
        target_bonus={"tank": 1.0, "artillery": 1.4, "infantry": 0.4, "apc": 0.6},
        hq_bonus=1.4,
        capture_prob=1.0,
        enemy_bias=1.5,
        enemy_hq_bias=1.6,
        site_bias=0.5,
        objective_bias=0.5,
        noise=0.05,
    ),
    "defensive": Persona(
        name="defensive",
        title="Anvil: hold the line around the HQ",
        build_weights={"infantry": 1.0, "aa_missile": 0.8, "artillery": 0.6, "tank": 0.2},
        build_prob=1.0,
        build_forward=0.1,
        attack_prob=1.0,
        risk_aversion=0.85,
        target_bonus={"infantry": 0.9, "tank": 0.2, "artillery": 1.1, "apc": 0.4},
        hq_bonus=0.2,
        capture_prob=1.0,
        capture_first=True,
        capture_unit_types=("infantry",),
        defense_radius=7,
        retreat_hp=0.4,
        home_bias=0.45,
        hold_prob=0.0,
        enemy_bias=0.7,
        enemy_hq_bias=0.0,
        site_bias=1.2,
        objective_bias=0.8,
        noise=0.05,
    ),
    "infantry": Persona(
        name="infantry",
        title="Line: mass infantry and take ground",
        build_weights={"infantry": 1.0},
        build_prob=1.0,
        build_forward=0.4,
        attack_prob=1.0,
        risk_aversion=0.55,
        target_bonus={"infantry": 1.1, "artillery": 1.3, "tank": 0.2, "apc": 0.6},
        hq_bonus=0.7,
        capture_prob=1.0,
        capture_first=True,
        capture_unit_types=("infantry",),
        hold_prob=0.0,
        enemy_bias=0.9,
        enemy_hq_bias=1.4,
        site_bias=1.9,
        objective_bias=1.2,
        noise=0.05,
    ),
    "armor": Persona(
        name="armor",
        title="Vanguard: mass tanks and hunt vehicles",
        build_weights={"tank": 1.0, "artillery": 0.45, "aa_missile": 0.25, "infantry": 0.08},
        build_prob=1.0,
        build_forward=0.75,
        attack_prob=1.0,
        risk_aversion=0.3,
        target_bonus={"tank": 1.7, "apc": 1.3, "artillery": 1.4, "infantry": 0.6},
        hq_bonus=1.0,
        capture_prob=1.0,
        enemy_bias=1.4,
        enemy_hq_bias=1.0,
        site_bias=0.7,
        noise=0.05,
    ),
    "artillery": Persona(
        name="artillery",
        title="Siege: bombard from range behind a screen",
        build_weights={"artillery": 1.0, "infantry": 0.65, "aa_missile": 0.3, "tank": 0.1},
        build_prob=1.0,
        build_forward=0.25,
        attack_prob=1.0,
        risk_aversion=0.08,
        target_bonus={"artillery": 1.6, "infantry": 1.1, "tank": 1.2, "apc": 0.8},
        hq_bonus=1.0,
        capture_prob=1.0,
        defense_radius=0,
        home_bias=0.55,
        hold_prob=0.0,
        enemy_bias=1.0,
        enemy_hq_bias=0.7,
        site_bias=0.8,
        standoff=3,
        noise=0.05,
    ),
}


def persona_names() -> list[str]:
    return list(PERSONAS)


def get_persona(name: str) -> Persona:
    try:
        return PERSONAS[name]
    except KeyError:
        raise KeyError(f"unknown persona {name!r}; choose from {persona_names()}") from None


def push_persona(name: str) -> Persona:
    """A pushing variant of a persona: no retreat, no home hold, HQ-focused.

    The build mix and target preferences stay intact, so the aggressive/armor
    styles still mass tanks, infantry still masses infantry, etc.  The changes
    remove the behaviours that made the clones sit on their starting side:
    defensive radius, retreat, home bias, standoff, and voluntary holds.
    """
    persona = get_persona(name)
    return replace(
        persona,
        build_prob=1.0,
        build_forward=0.85,
        attack_prob=1.0,
        attack_threshold=0.0,
        risk_aversion=min(persona.risk_aversion, 0.15),
        hq_bonus=max(persona.hq_bonus, 1.0),
        capture_prob=1.0,
        capture_first=True,
        defense_radius=0,
        retreat_hp=0.0,
        home_bias=0.0,
        hold_prob=0.0,
        enemy_bias=max(persona.enemy_bias, 1.2),
        enemy_hq_bias=max(persona.enemy_hq_bias, 1.4),
        site_bias=max(persona.site_bias, 1.0),
        objective_bias=max(persona.objective_bias, 0.8),
        standoff=0,
        noise=0.05,
    )


def random_persona(seed: int, name: str | None = None) -> Persona:
    """A seeded, randomized conventional-AI policy.

    The same seed always yields the same opponent, which makes training and
    league games reproducible.  The random draw covers production, attack
    appetite, risk, target priorities, capture behaviour, defence radius,
    standoff and movement bias.
    """
    rng = random.Random(seed)
    archetype = rng.choice(("armor", "infantry", "artillery", "air", "balanced"))
    if archetype == "armor":
        base = {"tank": 1.0, "artillery": 0.45, "aa_missile": 0.25}
    elif archetype == "infantry":
        base = {"infantry": 1.0, "artillery": 0.25, "aa_missile": 0.2}
    elif archetype == "artillery":
        base = {"artillery": 1.0, "infantry": 0.5, "aa_missile": 0.2}
    elif archetype == "air":
        base = {"helicopter": 1.0, "attack_plane": 0.8, "bomber": 0.4}
    else:
        types = rng.sample(
            ("infantry", "tank", "artillery", "aa_missile", "apc", "helicopter", "attack_plane"),
            rng.randint(2, 4),
        )
        base = {unit: rng.uniform(0.3, 1.0) for unit in types}
    build_weights = {
        unit: round(weight * rng.uniform(0.6, 1.3), 2)
        for unit, weight in base.items()
    }
    target_bonus = {
        unit: round(rng.uniform(0.0, 1.4), 2)
        for unit in ("infantry", "tank", "artillery", "apc", "hq")
    }
    capture_types = tuple(
        rng.sample(("infantry", "tank", "apc"), rng.randint(0, 2))
    )
    return Persona(
        name=name or f"heuristic-{seed}",
        title=f"Randomized conventional AI ({archetype})",
        build_weights=build_weights,
        build_prob=rng.uniform(0.4, 1.0),
        build_forward=rng.uniform(0.1, 0.9),
        attack_prob=rng.uniform(0.6, 1.0),
        attack_threshold=0.0,
        risk_aversion=rng.uniform(0.0, 0.8),
        target_bonus=target_bonus,
        hq_bonus=rng.uniform(0.0, 2.0),
        capture_prob=rng.uniform(0.6, 1.0),
        capture_first=rng.random() < 0.5,
        capture_unit_types=capture_types,
        defense_radius=rng.choice((0, 0, 4, 5, 6, 7)),
        retreat_hp=rng.uniform(0.0, 0.5),
        home_bias=rng.uniform(0.0, 0.5),
        hold_prob=rng.uniform(0.0, 0.1),
        enemy_bias=rng.uniform(0.6, 1.8),
        enemy_hq_bias=rng.uniform(0.0, 1.8),
        site_bias=rng.uniform(0.4, 1.9),
        objective_bias=rng.uniform(0.0, 1.5),
        standoff=rng.choice((0, 0, 0, 2, 3, 4)),
        noise=rng.uniform(0.05, 0.25),
    )


class PersonaTeacher:
    """A scripted policy with the same one-order-per-step interface as NcmAgent.

    Counters and ``lines`` mirror :class:`tbs_ncm.ncm_agent.NcmAgent`, so this
    object can also be dropped into ``selfplay.collect_game`` when on-policy rows
    against an NCM opponent are wanted.
    """

    name = "persona"

    def __init__(self, persona: Persona | str, seed: int = 0, push: bool = False):
        self.persona = get_persona(persona) if isinstance(persona, str) else persona
        self.name = self.persona.name
        self.push = push
        self.rng = random.Random(seed)
        self._salt = int.from_bytes(
            hashlib.blake2b(self.persona.name.encode("utf-8"), digest_size=8).digest(),
            "big",
        )
        self.generated = 0
        self.repaired = 0
        self.fallbacks = 0
        self.illegal = 0
        self.generation_errors = 0
        self.lines: list[str] = []
        self.fallback_reasons: list[str] = []
        self.actions: Counter[str] = Counter()
        self.builds: Counter[str] = Counter()
        self.orders = 0
        self.geometry_samples = 0
        self.enemy_distance = 0.0
        self.home_distance = 0.0
        self._turn_key: tuple[int, str] | None = None
        self._built = False
        self._used_destinations: set[tuple[int, int]] = set()

    # -- public API ---------------------------------------------------------

    def plan(self, request: TurnRequest) -> AgentPlan:
        order = self.next_order(request)
        if order is None:
            return AgentPlan(orders=[], commentary="persona nothing useful")
        return AgentPlan(orders=[order], commentary=f"persona:{self.persona.name}")

    def next_order(self, request: TurnRequest) -> dict[str, Any] | None:
        view = request.view
        player = request.player
        self._begin_turn(view, player)
        # Make each decision a deterministic function of the observable state.
        # Two games that reach the same view then label the same target, which
        # matters more for behaviour cloning than teacher-to-teacher variety.
        state_text = encode_state(view)
        digest = hashlib.blake2b(state_text.encode("utf-8"), digest_size=8).digest()
        self.rng = random.Random(int.from_bytes(digest, "big") ^ self._salt)
        order = self._choose(view, player)
        if order is None:
            return None
        self.generated += 1
        self.orders += 1
        action = str(order.get("action"))
        self.actions[action] += 1
        if action == "build":
            self.builds[str(order.get("unit"))] += 1
        self.lines.append(encode_order(order))
        self._note_geometry(view, player, order)
        return order

    # -- turn bookkeeping ---------------------------------------------------

    def _begin_turn(self, view: dict[str, Any], player: str) -> None:
        key = (int(view.get("turn", 0)), player)
        if key != self._turn_key:
            self._turn_key = key
            self._built = False
            self._used_destinations = set()

    # -- decision -----------------------------------------------------------

    def _choose(self, view: dict[str, Any], player: str) -> dict[str, Any] | None:
        persona = self.persona
        units = [unit for unit in view.get("units", []) if unit.get("pos") is not None]
        own = [unit for unit in units if unit.get("owner") == player]
        enemies = [unit for unit in units if unit.get("owner") != player]
        by_id = {str(unit["id"]): unit for unit in units}
        own_by_id = {str(unit["id"]): unit for unit in own}
        enemy_positions = [tuple(unit["pos"]) for unit in enemies]
        own_hq = next((unit for unit in own if unit.get("type") == "hq"), None)
        enemy_hq = next((unit for unit in enemies if unit.get("type") == "hq"), None)
        home = tuple(own_hq["pos"]) if own_hq else _centroid(own)
        enemy_home = tuple(enemy_hq["pos"]) if enemy_hq else None
        sites = view.get("sites", [])
        site_at = {tuple(site["pos"]): site for site in sites}
        capturable = [
            tuple(site["pos"])
            for site in sites
            if site.get("owner") != player and site.get("capture_needed") is not None
        ]
        objectives = [tuple(site["pos"]) for site in sites if site.get("objective")]
        actions = view.get("unit_actions", {}) or {}

        # Production first; the coordinator's per-turn view is refreshed after
        # every order, so this cannot build twice.
        if not self._built and self.rng.random() < persona.build_prob:
            order = self._build(view, player, home, enemy_home)
            if order is not None:
                self._built = True
                return order

        # A persona still travels by transport.  The engine's ferry hints are
        # the same ones the greedy teacher and the LLM view use; returning the
        # first staged order lets the stepwise coordinator refresh the view and
        # issue the next move/load/sail/unload in the following step.
        ferry = plan_ferry(view)
        if ferry.orders:
            return dict(ferry.orders[0])

        candidates = [
            unit_id
            for unit_id, summary in actions.items()
            if isinstance(summary, dict)
            and not summary.get("acted")
            and not summary.get("loaded_in")
            and not summary.get("new_unit")
            and unit_id in own_by_id
            and unit_id not in ferry.claimed
        ]
        if not candidates:
            return None
        # Front units first for aggressive styles, wounded first for defensive.
        if persona.defense_radius:
            candidates.sort(
                key=lambda unit_id: (
                    _hp_fraction(own_by_id[unit_id]),
                    _distance_to(own_by_id[unit_id].get("pos"), enemy_positions),
                )
            )
        else:
            candidates.sort(
                key=lambda unit_id: (
                    _distance_to(own_by_id[unit_id].get("pos"), enemy_positions),
                    -_hp_fraction(own_by_id[unit_id]),
                )
            )
        if persona.noise >= 0.2:
            # A little order noise keeps the cloned distribution from collapsing
            # into the same unit every turn.
            if self.rng.random() < 0.25:
                self.rng.shuffle(candidates)

        fallback_hold = None
        for unit_id in candidates:
            summary = actions[unit_id]
            unit = own_by_id[unit_id]
            options = self._unit_options(
                unit, summary, by_id, site_at, home, enemy_home,
                enemy_positions, capturable, objectives,
            )
            if options is not None:
                if self.push and str(options.get("action")) == "hold":
                    if fallback_hold is None:
                        fallback_hold = options
                    continue
                return options
        return fallback_hold

    def _unit_options(
        self,
        unit: dict[str, Any],
        summary: dict[str, Any],
        by_id: dict[str, dict[str, Any]],
        site_at: dict[tuple[int, int], dict[str, Any]],
        home: tuple[int, int],
        enemy_home: tuple[int, int] | None,
        enemy_positions: list[tuple[int, int]],
        capturable: list[tuple[int, int]],
        objectives: list[tuple[int, int]],
    ) -> dict[str, Any] | None:
        persona = self.persona
        unit_id = str(unit["id"])
        pos = tuple(unit["pos"])
        options: dict[str, Any] = summary
        attacks = [row for row in options.get("attack", []) if isinstance(row, list) and len(row) >= 4]
        move_rows = [row for row in options.get("move_to", []) if isinstance(row, list) and row]
        capture_remaining = options.get("capture_remaining_hp")
        hold_available = bool(options.get("hold"))

        # Capture first for capture specialists and for an enemy HQ underfoot.
        capture_order = None
        if capture_remaining is not None:
            site = site_at.get(pos)
            is_enemy_hq = bool(site and site.get("terrain") == "H" and site.get("owner") != unit.get("owner"))
            chance = persona.capture_prob
            if str(unit.get("type")) in persona.capture_unit_types:
                chance = min(1.0, chance + 0.35)
            if is_enemy_hq:
                chance = 1.0
            if self.rng.random() < chance:
                capture_order = {"action": "capture", "unit": unit_id}
        if capture_order is not None and (persona.capture_first or not attacks):
            return capture_order

        if attacks and self.rng.random() < persona.attack_prob:
            attack = self._choose_attack(unit, attacks, by_id, home)
            if attack is not None:
                return {"action": "attack", "unit": unit_id, **attack}

        if capture_order is not None:
            return capture_order

        # Retreat before looking for a forward move.
        if persona.retreat_hp and _hp_fraction(unit) <= persona.retreat_hp and enemy_positions:
            retreat = self._retreat(unit, move_rows, home, enemy_positions)
            if retreat is not None:
                return retreat

        # A deliberate pause is part of the defensive style; it is also new
        # action coverage, because the original greedy dataset never holds.
        if hold_available and self.rng.random() < persona.hold_prob:
            return {"action": "hold", "unit": unit_id}

        # Sea transports are handled by the ferry planner above.  These random
        # load/unload branches remain for land transports (truck/APC), and they
        # add some action coverage the greedy teacher does not produce.
        loads = [
            str(item)
            for item in (options.get("load") or [])
            if by_id.get(str(item), {}).get("move", 0) > 0
        ]
        if loads and self.rng.random() < 0.55:
            cargo = min(
                loads,
                key=lambda item: (
                    _distance_to(by_id.get(item, {}).get("pos"), enemy_positions),
                    item,
                ),
            )
            return {"action": "load", "unit": unit_id, "cargo": cargo}

        unloads = options.get("unload") or []
        if unloads and self.rng.random() < 0.7:
            cargo, at = min(
                unloads,
                key=lambda row: (
                    _distance_to(row[1], enemy_positions),
                    hex_distance(tuple(row[1]), home),
                ),
            )
            return {
                "action": "unload",
                "unit": unit_id,
                "cargo": cargo,
                "target": list(at),
            }

        move = self._choose_move(
            unit, move_rows, home, enemy_home, enemy_positions, capturable, objectives
        )
        if move is not None:
            return move

        if hold_available:
            return {"action": "hold", "unit": unit_id}
        return None

    def _build(
        self,
        view: dict[str, Any],
        player: str,
        home: tuple[int, int],
        enemy_home: tuple[int, int] | None,
    ) -> dict[str, Any] | None:
        persona = self.persona
        affordable = [
            option
            for option in view.get("build_options", [])
            if option.get("affordable") and persona.build_weights.get(option["unit"], 0) > 0
        ]
        if not affordable:
            return None
        weights = [persona.build_weights[option["unit"]] for option in affordable]
        chosen_type = self.rng.choices(affordable, weights=weights, k=1)[0]["unit"]
        candidates = [option for option in affordable if option["unit"] == chosen_type]

        def site_score(option: dict[str, Any]) -> float:
            site = tuple(option["site"])
            forward = hex_distance(site, enemy_home) if enemy_home else 0
            back = hex_distance(site, home) if home else 0
            return persona.build_forward * forward + (1.0 - persona.build_forward) * back

        choice = min(candidates, key=site_score)
        return {
            "action": "build",
            "site": list(choice["site"]),
            "unit": choice["unit"],
        }

    def _choose_attack(
        self,
        unit: dict[str, Any],
        attacks: list[list[Any]],
        by_id: dict[str, dict[str, Any]],
        home: tuple[int, int],
    ) -> dict[str, Any] | None:
        persona = self.persona
        unit_pos = tuple(unit["pos"])
        best: tuple[float, dict[str, Any]] | None = None
        for target, weapon, distance, preview in attacks:
            target_unit = by_id.get(str(target))
            if target_unit is None:
                continue
            target_pos = tuple(target_unit["pos"])
            if persona.defense_radius and hex_distance(target_pos, home) > persona.defense_radius:
                continue
            bonus = persona.target_bonus.get(str(target_unit.get("type")), 0.0)
            if target_unit.get("type") == "hq":
                bonus += persona.hq_bonus
            score = float(preview) * (1.0 + bonus)
            if int(distance) <= 1:
                # Direct fire is counterattacked at half power while the
                # defender lives; approximate the risk from its remaining HP.
                score -= persona.risk_aversion * float(preview) * _hp_fraction(target_unit)
            score -= 0.05 * hex_distance(unit_pos, target_pos)
            score += self.rng.random() * persona.noise
            if score < persona.attack_threshold:
                continue
            order = {"target": target, "weapon": weapon}
            if best is None or score > best[0]:
                best = (score, order)
        return best[1] if best is not None else None

    def _retreat(
        self,
        unit: dict[str, Any],
        move_rows: list[list[Any]],
        home: tuple[int, int],
        enemy_positions: list[tuple[int, int]],
    ) -> dict[str, Any] | None:
        if not move_rows:
            return None
        current = tuple(unit["pos"])
        current_distance = _distance_to(current, enemy_positions)
        best: tuple[float, tuple[int, int]] | None = None
        for row in move_rows:
            destination = tuple(row[0])
            if destination in self._used_destinations:
                continue
            distance = _distance_to(destination, enemy_positions)
            score = distance + 0.25 * hex_distance(destination, home)
            if best is None or score > best[0]:
                best = (score, destination)
        if best is None or best[0] < current_distance + 0.5:
            return None
        self._used_destinations.add(best[1])
        return {"action": "move", "unit": str(unit["id"]), "to": list(best[1])}

    def _choose_move(
        self,
        unit: dict[str, Any],
        move_rows: list[list[Any]],
        home: tuple[int, int],
        enemy_home: tuple[int, int] | None,
        enemy_positions: list[tuple[int, int]],
        capturable: list[tuple[int, int]],
        objectives: list[tuple[int, int]],
    ) -> dict[str, Any] | None:
        persona = self.persona
        if not move_rows:
            return None
        current = tuple(unit["pos"])
        current_home = hex_distance(current, home)
        current_score = self._move_score(
            current, 0.0, home, enemy_home, enemy_positions, capturable, objectives
        )
        best: tuple[float, tuple[int, int]] | None = None
        for row in move_rows:
            destination = tuple(row[0])
            if destination in self._used_destinations:
                continue
            if (
                persona.defense_radius
                and current_home <= persona.defense_radius
                and hex_distance(destination, home) > persona.defense_radius
            ):
                continue
            cost = float(row[1]) if len(row) > 1 else 0.0
            score = self._move_score(
                destination, cost, home, enemy_home, enemy_positions, capturable, objectives
            )
            if best is None or score > best[0]:
                best = (score, destination)
        if best is None:
            return None
        margin = 0.35 if persona.defense_radius else 0.05
        if best[0] <= current_score + margin:
            return None
        self._used_destinations.add(best[1])
        return {"action": "move", "unit": str(unit["id"]), "to": list(best[1])}

    def _move_score(
        self,
        destination: tuple[int, int],
        cost: float,
        home: tuple[int, int],
        enemy_home: tuple[int, int] | None,
        enemy_positions: list[tuple[int, int]],
        capturable: list[tuple[int, int]],
        objectives: list[tuple[int, int]],
    ) -> float:
        persona = self.persona
        d_enemy = _distance_to(destination, enemy_positions)
        d_enemy_hq = hex_distance(destination, enemy_home) if enemy_home else 12.0
        if persona.defense_radius:
            local_sites = [
                spot
                for spot in capturable
                if hex_distance(spot, home) <= persona.defense_radius
            ]
        else:
            local_sites = capturable
        d_site = _distance_to(destination, local_sites)
        d_home = hex_distance(destination, home)
        if persona.standoff:
            score = -abs(d_enemy - persona.standoff) * 2.0
        else:
            score = (
                -persona.enemy_bias * min(d_enemy, 12.0)
                -persona.enemy_hq_bias * min(d_enemy_hq, 20.0)
                -persona.site_bias * min(d_site, 12.0)
            )
            if objectives:
                score -= persona.objective_bias * min(
                    hex_distance(destination, spot) for spot in objectives
                )
        if persona.defense_radius:
            if d_home > persona.defense_radius:
                score -= 6.0 * (d_home - persona.defense_radius)
            else:
                score += 0.15 * (persona.defense_radius - d_home)
        if persona.home_bias:
            score -= persona.home_bias * min(d_home, 12.0)
        score -= 0.05 * cost
        score += self.rng.random() * persona.noise
        return score

    def _note_geometry(self, view: dict[str, Any], player: str, order: dict[str, Any]) -> None:
        units = [unit for unit in view.get("units", []) if unit.get("pos") is not None]
        own = [unit for unit in units if unit.get("owner") == player]
        enemies = [tuple(unit["pos"]) for unit in units if unit.get("owner") != player]
        if not own or not enemies:
            return
        position = None
        unit_id = order.get("unit")
        for unit in own:
            if str(unit["id"]) == str(unit_id):
                if order.get("action") == "move":
                    position = tuple(order.get("to") or unit["pos"])
                else:
                    position = tuple(unit["pos"])
                break
        if position is None:
            return
        own_hq = next((unit for unit in own if unit.get("type") == "hq"), None)
        home = tuple(own_hq["pos"]) if own_hq else _centroid(own)
        self.enemy_distance += _distance_to(position, enemies)
        self.home_distance += hex_distance(position, home)
        self.geometry_samples += 1


def _hp_fraction(unit: dict[str, Any]) -> float:
    maximum = unit.get("max_hp") or 100
    return float(unit.get("hp", 0)) / max(1, float(maximum))


def _distance_to(position: Any, spots: list[tuple[int, int]]) -> float:
    if not spots or position is None:
        return 12.0
    point = tuple(position)
    return float(min(hex_distance(point, spot) for spot in spots))


def _centroid(units: list[dict[str, Any]]) -> tuple[int, int]:
    if not units:
        return (0, 0)
    q = sum(int(unit["pos"][0]) for unit in units) / len(units)
    r = sum(int(unit["pos"][1]) for unit in units) / len(units)
    return (round(q), round(r))
