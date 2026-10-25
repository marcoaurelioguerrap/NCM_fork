"""The NCM player: generate one order line from the compact state prefix."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

from .common import ARTIFACT_DIR, load_config, load_generator_state
from .chartok import CharTokenizer
from .legal import NearestLegal, legal_lines, nearest_legal_order

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from tbs.agents.codec import CodecError, decode_order, encode_order, encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.engine.hexgrid import hex_distance

# Candidate ranking: lower is better.  A nearest-legal global snap outranks a
# unit-snapped *different* action because the latter is more likely to be an
# arbitrary hold, not something the model asked for.
REPAIR_RANK = {
    "generated": 0,
    "same-unit": 1,
    "action": 2,
    "unit": 3,
    "nearest": 4,
    "unit-action": 5,
}


@dataclass(frozen=True)
class Repair:
    """One legal projection of a sampled line plus its provenance."""

    order: dict
    kind: str
    cost: float
    line: str = ""

    @property
    def rank(self) -> int:
        return REPAIR_RANK.get(self.kind, 9)


def _safe_pos(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def _attack_damage(option: list | tuple) -> float:
    if len(option) >= 4:
        try:
            return float(option[3])
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def order_is_legal(view: dict, order: dict) -> bool:
    """Check one decoded order against the same view options an LLM sees.

    This is a pre-filter, not a rules engine: the coordinator still validates
    every order.  It lets the player resample instead of spending a step on a
    move the view already says is impossible.
    """
    if not isinstance(order, dict):
        return False
    action = order.get("action")
    if action == "end_turn":
        return True
    if action == "build":
        site = _safe_pos(order.get("site"))
        unit_type = order.get("unit")
        if site is None:
            return False
        return any(
            _safe_pos(option.get("site")) == site
            and option.get("unit") == unit_type
            and option.get("affordable")
            for option in view.get("build_options", [])
        )

    unit_id = order.get("unit")
    options = (view.get("unit_actions") or {}).get(unit_id)
    if not isinstance(options, dict):
        return False
    if action == "move":
        destination = _safe_pos(order.get("to"))
        if destination is None:
            return False
        return any(
            _safe_pos(option[0]) == destination
            for option in options.get("move_to", [])
        )
    if action == "attack":
        target = order.get("target")
        weapon = order.get("weapon")
        return any(
            option[0] == target and (weapon is None or option[1] == weapon)
            for option in options.get("attack", [])
        )
    if action == "capture":
        return options.get("capture_remaining_hp") is not None
    if action == "hold":
        return bool(options.get("hold"))
    if action == "load":
        return order.get("cargo") in (options.get("load") or [])
    if action == "unload":
        target = _safe_pos(order.get("target"))
        return any(
            option[0] == order.get("cargo") and _safe_pos(option[1]) == target
            for option in options.get("unload", [])
        )
    return False


def _unit_actions(view: dict) -> dict:
    actions = view.get("unit_actions")
    return actions if isinstance(actions, dict) else {}


def _can_act(options: object) -> bool:
    if not isinstance(options, dict):
        return False
    if options.get("acted") or options.get("loaded_in") or options.get("new_unit"):
        return False
    return bool(
        options.get("move_to")
        or options.get("attack")
        or options.get("load")
        or options.get("unload")
        or options.get("hold")
        or options.get("capture_remaining_hp") is not None
    )


def _actionable_units(view: dict, actions: dict | None = None) -> list[str]:
    actions = actions if actions is not None else _unit_actions(view)
    return sorted(unit_id for unit_id, options in actions.items() if _can_act(options))


def resolve_unit_id(
    view: dict,
    requested: object,
    *,
    actions: dict | None = None,
) -> str | None:
    """Map a generated unit id onto an actionable unit in this view.

    Exact ids win.  A mistyped suffix ("A17" when "A12" exists) resolves by
    numeric distance within the same player prefix, not blindly to ``A1``.
    Cross-prefix shapes only pass through a conservative fuzzy match; an enemy
    id is never silently turned into an arbitrary own unit.
    """
    text = str(requested or "").strip()
    if not text:
        return None
    actions = actions if actions is not None else _unit_actions(view)
    candidates = _actionable_units(view, actions)
    if not candidates:
        return None
    by_lower = {candidate.lower(): candidate for candidate in candidates}
    exact = by_lower.get(text.lower())
    if exact is not None:
        return exact

    shape = re.fullmatch(r"([A-Za-z]*)(\d+)", text)
    if shape:
        prefix = shape.group(1).lower()
        number = int(shape.group(2))
        same_prefix: list[tuple[int, str]] = []
        for candidate in candidates:
            match = re.fullmatch(r"([A-Za-z]*)(\d+)", candidate)
            if match and match.group(1).lower() == prefix:
                same_prefix.append((abs(int(match.group(2)) - number), candidate))
        if same_prefix:
            return min(same_prefix)[1]

    close = difflib.get_close_matches(
        text.lower(), list(by_lower), n=1, cutoff=0.55
    )
    if close:
        return by_lower[close[0]]
    return None


def _normalise_name(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def _name_score(requested: str, name: str) -> float:
    """Spelling-tolerant comparison for build unit types."""
    left = _normalise_name(requested)
    right = _normalise_name(name)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    ratio = difflib.SequenceMatcher(None, left, right).ratio()
    if left in right or right in left:
        ratio = max(ratio, 0.72)
    return ratio


def _build_repair(view: dict, order: dict) -> Repair | None:
    requested_site = _safe_pos(order.get("site"))
    requested_unit = str(order.get("unit") or "")
    affordable = [
        option
        for option in view.get("build_options", [])
        if option.get("affordable") and _safe_pos(option.get("site")) is not None
    ]
    if not affordable:
        return None

    def site_distance(option: dict) -> int:
        if requested_site is None:
            return 0
        return hex_distance(_safe_pos(option["site"]), requested_site)

    def pick(options: list[dict]) -> dict:
        return min(
            options,
            key=lambda option: (
                site_distance(option),
                int(option.get("cost", 0)),
                str(option.get("unit", "")),
            ),
        )

    if requested_unit:
        scored = [(_name_score(requested_unit, str(option["unit"])), option) for option in affordable]
        best_score = max(score for score, _ in scored)
        at_site = [
            (score, option)
            for score, option in scored
            if requested_site is not None and _safe_pos(option["site"]) == requested_site
        ]
        if at_site and max(score for score, _ in at_site) >= 0.55:
            choice = pick([option for score, option in at_site if score == max(s for s, _ in at_site)])
        elif best_score >= 0.55:
            names = {score for score, _ in scored}
            best = [
                option for score, option in scored if score == max(names)
            ]
            choice = pick(best)
        else:
            # The line asked for a build but no known type is close.  Keeping
            # the site intent is more useful than dropping the action.
            choice = pick(affordable)
        return Repair(
            order={
                "action": "build",
                "site": list(choice["site"]),
                "unit": choice["unit"],
            },
            kind="same-unit",
            cost=site_distance(choice) / 10.0 + (1.0 - max(best_score, 0.0)) * 0.5,
        )

    choice = pick(affordable)
    return Repair(
        order={
            "action": "build",
            "site": list(choice["site"]),
            "unit": choice["unit"],
        },
        kind="same-unit",
        cost=site_distance(choice) / 10.0,
    )


def _same_action_repair(
    view: dict,
    unit_id: str,
    order: dict,
    actions: dict,
) -> Repair | None:
    action = order.get("action")
    options = actions.get(unit_id)
    if not isinstance(options, dict):
        return None

    if action == "move":
        moves = options.get("move_to") or []
        if not moves:
            return None
        requested = _safe_pos(order.get("to")) or (0, 0)
        choice = min(
            moves,
            key=lambda option: (
                hex_distance(_safe_pos(option[0]) or (0, 0), requested),
                option[1] if len(option) > 1 else 0,
                tuple(option[0]),
            ),
        )
        distance = hex_distance(_safe_pos(choice[0]) or (0, 0), requested)
        return Repair(
            {"action": "move", "unit": unit_id, "to": list(choice[0])},
            "same-unit",
            distance / 10.0,
        )

    if action == "attack":
        attacks = options.get("attack") or []
        if not attacks:
            return None
        target = order.get("target")
        weapon = order.get("weapon")
        same_target = [option for option in attacks if option[0] == target]
        if same_target:
            exact_weapon = [option for option in same_target if option[1] == weapon]
            choice = max(exact_weapon or same_target, key=_attack_damage)
            cost = 0.0 if exact_weapon else 0.1
        else:
            choice = max(attacks, key=lambda option: (_attack_damage(option), str(option[0])))
            cost = 0.5
        return Repair(
            {
                "action": "attack",
                "unit": unit_id,
                "target": choice[0],
                "weapon": choice[1],
            },
            "same-unit",
            cost,
        )

    if action == "capture" and options.get("capture_remaining_hp") is not None:
        return Repair({"action": "capture", "unit": unit_id}, "same-unit", 0.0)
    if action == "hold" and options.get("hold"):
        return Repair({"action": "hold", "unit": unit_id}, "same-unit", 0.0)

    if action == "load":
        loads = [str(cargo) for cargo in (options.get("load") or [])]
        if not loads:
            return None
        requested = str(order.get("cargo") or "")
        exact = [cargo for cargo in loads if cargo == requested]
        close = difflib.get_close_matches(requested, loads, n=1, cutoff=0.55) if requested else []
        cargo = (exact or close or sorted(loads))[0]
        return Repair(
            {"action": "load", "unit": unit_id, "cargo": cargo},
            "same-unit",
            0.0 if exact else 0.2,
        )

    if action == "unload":
        unloads = options.get("unload") or []
        if not unloads:
            return None
        requested_cargo = order.get("cargo")
        requested_at = _safe_pos(order.get("target"))
        same_cargo = [row for row in unloads if row[0] == requested_cargo]
        pool = same_cargo or unloads
        choice = min(
            pool,
            key=lambda row: (
                hex_distance(_safe_pos(row[1]) or (0, 0), requested_at or (0, 0)),
                str(row[0]),
            ),
        )
        return Repair(
            {
                "action": "unload",
                "unit": unit_id,
                "cargo": choice[0],
                "target": list(choice[1]),
            },
            "same-unit",
            0.0 if same_cargo else 0.3,
        )
    return None


def _alternate_action_repair(
    view: dict,
    unit_id: str,
    order: dict,
    actions: dict,
) -> Repair | None:
    """Try a legal action for the same unit: attack, capture, hold, move."""
    options = actions.get(unit_id)
    if not isinstance(options, dict):
        return None

    attacks = options.get("attack") or []
    if attacks:
        choice = max(attacks, key=lambda option: (_attack_damage(option), str(option[0])))
        return Repair(
            {
                "action": "attack",
                "unit": unit_id,
                "target": choice[0],
                "weapon": choice[1],
            },
            "action",
            0.1,
        )
    if options.get("capture_remaining_hp") is not None:
        return Repair({"action": "capture", "unit": unit_id}, "action", 0.2)
    if options.get("hold"):
        return Repair({"action": "hold", "unit": unit_id}, "action", 0.3)

    moves = options.get("move_to") or []
    if moves:
        requested = _safe_pos(order.get("to"))
        choice = min(
            moves,
            key=lambda option: (
                hex_distance(_safe_pos(option[0]) or (0, 0), requested or (0, 0)),
                tuple(option[0]),
            ),
        )
        distance = (
            hex_distance(_safe_pos(choice[0]) or (0, 0), requested)
            if requested is not None
            else 0
        )
        return Repair(
            {"action": "move", "unit": unit_id, "to": list(choice[0])},
            "action",
            0.4 + distance / 10.0,
        )

    loads = options.get("load") or []
    if loads:
        return Repair(
            {"action": "load", "unit": unit_id, "cargo": sorted(str(c) for c in loads)[0]},
            "action",
            0.5,
        )
    unloads = options.get("unload") or []
    if unloads:
        return Repair(
            {
                "action": "unload",
                "unit": unit_id,
                "cargo": unloads[0][0],
                "target": list(unloads[0][1]),
            },
            "action",
            0.5,
        )
    return None


def repair_order(
    view: dict,
    order: dict,
    *,
    actions: dict | None = None,
) -> Repair | None:
    """Project one order onto a legal option, with provenance and a snap cost."""
    if not isinstance(order, dict):
        return None
    if order.get("action") == "build":
        repair = _build_repair(view, order)
        return repair if repair and order_is_legal(view, repair.order) else None
    if order.get("action") == "end_turn":
        return None

    actions = actions if actions is not None else _unit_actions(view)
    requested = str(order.get("unit") or "")
    unit_id = resolve_unit_id(view, requested, actions=actions)
    if unit_id is None:
        return None
    exact_unit = unit_id.lower() == requested.lower()

    repair = _same_action_repair(view, unit_id, order, actions)
    if repair is not None:
        if order_is_legal(view, repair.order):
            if exact_unit:
                return repair
            return Repair(repair.order, "unit", repair.cost + 0.05)
        # The requested action could not be salvaged; fall through.

    # A different action is only meaningful when the model named a real unit.
    # If we already had to invent the unit id, a snap to an unrelated hold is
    # not model intent; let the global nearest-legal pass or the fallback win.
    if not exact_unit:
        return None

    repair = _alternate_action_repair(view, unit_id, order, actions)
    if repair is not None and order_is_legal(view, repair.order):
        return repair
    return None


def repair_in_view(view: dict, order: dict) -> dict | None:
    """Compatibility wrapper: return only the projected order."""
    repair = repair_order(view, order)
    return repair.order if repair is not None else None


def candidate_for_line(
    view: dict,
    line: str,
    *,
    allowed: list[str] | tuple[str, ...] | set[str] | None = None,
) -> Repair | None:
    """Decode and project one generated line, returning the best legal candidate."""
    allowed = list(allowed) if allowed is not None else legal_lines(view)
    order: dict | None = None
    try:
        order = decode_order(line)
    except CodecError:
        order = None

    if order is not None and order_is_legal(view, order):
        return Repair(order, "generated", 0.0, line)

    candidates: list[Repair] = []
    if order is not None:
        repair = repair_order(view, order)
        if repair is not None:
            candidates.append(
                Repair(repair.order, repair.kind, repair.cost, encode_order(repair.order))
            )
    nearest: NearestLegal | None = nearest_legal_order(view, line, allowed=allowed)
    if nearest is not None and order_is_legal(view, nearest.order):
        candidates.append(
            Repair(nearest.order, "nearest", 1.0 - nearest.score, nearest.line)
        )
    if not candidates:
        return None
    return min(candidates, key=lambda candidate: (candidate.rank, candidate.cost))


class NcmSampler:
    """Load the trained tokenizer + generator and sample one order line."""

    def __init__(
        self,
        checkpoint_dir: Path | str = ARTIFACT_DIR,
        device: str | None = None,
        prefix_len: int | None = 2048,
        generator: str | Path | None = None,
    ):
        checkpoint_dir = Path(checkpoint_dir)
        local_config = checkpoint_dir / "config.yaml"
        cfg = load_config(local_config) if local_config.exists() else load_config()
        if prefix_len is not None:
            old_prefix = int(cfg.model.prefix_len)
            headroom = int(cfg.model.context_length) - old_prefix
            if headroom < 1:
                raise ValueError(
                    f"checkpoint context_length {cfg.model.context_length} is not "
                    f"larger than prefix_len {old_prefix}"
                )
            cfg.model.prefix_len = int(prefix_len)
            cfg.model.context_length = int(prefix_len) + headroom
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer_path = checkpoint_dir / str(cfg.checkpoints.tokenizer)
        if generator is None:
            generator_path = checkpoint_dir / str(cfg.checkpoints.generator)
        else:
            candidate = Path(generator)
            generator_path = candidate if candidate.is_absolute() else checkpoint_dir / candidate
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"missing tokenizer checkpoint {tokenizer_path}")
        if not generator_path.exists():
            raise FileNotFoundError(f"missing generator checkpoint {generator_path}")

        self.vqvae = VQVAE(**cfg.tokenizer.vqvae).to(self.device)
        tokenizer_state = torch.load(tokenizer_path, map_location="cpu", weights_only=False)
        self.vqvae.load_state_dict(tokenizer_state["model"], strict=True)
        self.vqvae.eval()
        for parameter in self.vqvae.parameters():
            parameter.requires_grad_(False)

        self.net = NextConceptTransformer(vqvae_local=self.vqvae, config=cfg.model).to(self.device)
        generator_state = torch.load(generator_path, map_location="cpu", weights_only=False)
        load_generator_state(self.net, generator_state["model"])
        self.net.eval()

        self.chartok = CharTokenizer()
        self.gpt2 = HFTokenizer("gpt2", pad_token="<|pad|>").hf_tokenizer
        self.prefix_len = int(cfg.model.prefix_len)
        self.pad_id = int(cfg.model.prefix_pad_token_id)
        if self.gpt2.pad_token_id != self.pad_id:
            raise SystemExit(
                f"prefix pad id {self.gpt2.pad_token_id} does not match config {self.pad_id}"
            )

    def token_count(self, text: str) -> int:
        return len(self.gpt2.encode(text))

    def fits(self, text: str) -> bool:
        return self.token_count(text) <= self.prefix_len

    def prefix_ids(self, text: str) -> list[int]:
        ids = self.gpt2.encode(text)
        if len(ids) > self.prefix_len:
            raise ValueError(
                f"state prefix needs {len(ids)} GPT-2 tokens, the model accepts {self.prefix_len}"
            )
        return ids + [self.pad_id] * (self.prefix_len - len(ids))

    @torch.no_grad()
    def sample(self, state_text: str, temperature: float = 0.0) -> str:
        prefix = torch.tensor([self.prefix_ids(state_text)], device=self.device)
        codes = self.net.generate(
            None,
            prefix,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
        )
        char_ids = self.vqvae.decode(codes).argmax(dim=-1)
        return self.chartok.decode(char_ids[0].tolist()).strip()


class NcmAgent:
    """One generated order per step; a scripted policy recovers if all samples fail."""

    name = "ncm"

    def __init__(
        self,
        sampler: NcmSampler,
        temperature: float = 0.0,
        retries: int = 6,
        retry_temperature: float = 0.5,
        fallback: object | None = None,
    ):
        self.sampler = sampler
        self.temperature = temperature
        self.retries = max(0, int(retries))
        self.retry_temperature = retry_temperature
        self.fallback = fallback if fallback is not None else GreedyAgent()
        self.steps = 0
        self.samples = 0
        self.generated = 0
        self.repaired = 0
        self.repaired_samples = 0
        self.fallbacks = 0
        self.illegal = 0
        self.rejected = 0
        self.recovered = 0
        self.state_too_long = 0
        self.generation_errors = 0
        self.lines: list[str] = []
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
        self._last_line = ""
        self._last_order: dict | None = None

    def plan(self, request) -> AgentPlan:
        self.steps += 1
        view = request.view
        feedback = bool(request.feedback)
        avoid = self._last_line if feedback and self._last_line else ""
        if feedback:
            self.rejected += 1

        try:
            state_text = encode_state(view)
        except CodecError:
            return self._fallback(request, "state-encode-error")

        fits = getattr(self.sampler, "fits", None)
        if fits is not None:
            try:
                if not fits(state_text):
                    self.state_too_long += 1
                    return self._fallback(request, "state-too-long")
            except (RuntimeError, ValueError):
                self.generation_errors += 1
                return self._fallback(request, "generation-error")

        try:
            allowed = legal_lines(view)
        except Exception:
            allowed = []

        best: Repair | None = None
        best_key: tuple[int, float, int] | None = None
        for attempt in range(self.retries + 1):
            temperature = self.temperature if attempt == 0 else self.retry_temperature
            try:
                line = self.sampler.sample(state_text, temperature=temperature)
            except ValueError as exc:
                message = str(exc)
                if "GPT-2 tokens" in message or "prefix needs" in message:
                    self.state_too_long += 1
                    return self._fallback(request, "state-too-long")
                self.generation_errors += 1
                return self._fallback(request, "generation-error")
            except RuntimeError:
                self.generation_errors += 1
                return self._fallback(request, "generation-error")

            self.samples += 1
            self.lines.append(line)
            candidate = candidate_for_line(view, line, allowed=allowed)
            if candidate is None:
                self.illegal += 1
                continue

            if candidate.kind == "generated":
                if avoid and encode_order(candidate.order) == avoid:
                    self.illegal += 1
                    continue
                self.generated += 1
                if feedback:
                    self.recovered += 1
                self._remember(encode_order(candidate.order), candidate.order)
                self._note_order(view, candidate.order)
                return AgentPlan(orders=[candidate.order], commentary="")

            if avoid and encode_order(candidate.order) == avoid:
                self.illegal += 1
                continue
            self.repaired_samples += 1
            key = (candidate.rank, candidate.cost, attempt)
            if best_key is None or key < best_key:
                best = candidate
                best_key = key

        if best is not None:
            self.repaired += 1
            self.repairs[best.kind] += 1
            if feedback:
                self.recovered += 1
            self._remember(encode_order(best.order), best.order)
            self._note_order(view, best.order)
            return AgentPlan(orders=[best.order], commentary="")

        return self._fallback(request, "rejected" if feedback else "no-legal-sample")

    def _remember(self, line: str, order: dict) -> None:
        self._last_line = line or encode_order(order)
        self._last_order = dict(order)

    def _fallback(self, request, reason: str) -> AgentPlan:
        self.fallbacks += 1
        self.fallback_reasons.append(reason)
        plan = self.fallback.plan(request)
        if plan.orders:
            order = plan.orders[0]
            self._remember(encode_order(order), order)
            self._note_order(request.view, order, fallback=True)
            return AgentPlan(orders=[order], commentary=f"fallback:{reason}")
        return AgentPlan(orders=[])

    def _note_order(self, view: dict, order: dict, fallback: bool = False) -> None:
        action = str(order.get("action"))
        self.actions[action] += 1
        self.orders += 1
        if action == "build":
            self.builds[str(order.get("unit"))] += 1
        if not fallback:
            self.model_actions[action] += 1
            self.model_orders += 1
            if action == "build":
                self.model_builds[str(order.get("unit"))] += 1

        units = [unit for unit in view.get("units", []) if unit.get("pos") is not None]
        own = [unit for unit in units if unit.get("owner") == view.get("current_player")]
        enemies = [tuple(unit["pos"]) for unit in units if unit.get("owner") != view.get("current_player")]
        if not own or not enemies:
            return
        position = None
        for unit in own:
            if str(unit["id"]) == str(order.get("unit")):
                if action == "move":
                    position = tuple(order.get("to") or unit["pos"])
                else:
                    position = tuple(unit["pos"])
                break
        if position is None:
            return
        own_hq = next((unit for unit in own if unit.get("type") == "hq"), None)
        if own_hq is None:
            return
        home = tuple(own_hq["pos"])
        self.enemy_distance += min(hex_distance(position, spot) for spot in enemies)
        self.home_distance += hex_distance(position, home)
        self.geometry_samples += 1
