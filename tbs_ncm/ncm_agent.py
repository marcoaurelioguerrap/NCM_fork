"""The NCM player: generate one order line from the compact state prefix."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import ARTIFACT_DIR, load_config, load_generator_state
from .chartok import CharTokenizer
from .legal import NearestLegal, legal_lines, nearest_legal_order, rank_legal_orders

from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE
from tokenization.hf_tokenizer import HFTokenizer

from tbs.agents.codec import CodecError, decode_order, encode_order, encode_state
from tbs.agents.greedy import GreedyAgent
from tbs.agents.protocol import AgentPlan
from tbs.engine.hexgrid import hex_distance

from .action_model import ActionModel

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

    @torch.no_grad()
    def score_lines(self, state_text: str, lines: list[str]) -> list[float]:
        """Mean log-probability of each candidate order under this model.

        This is the same teacher-forced objective used for validation: encode
        each line with the VQ-VAE, run the generator with the state prefix, and
        average the per-code log-probability.  Higher is better.
        """
        if not lines:
            return []
        context = int(self.cfg.tokenizer.vqvae.context_length)
        targets = [self.chartok.encode_padded(line, context) for line in lines]
        target = torch.tensor(targets, dtype=torch.long, device=self.device)
        codes = self.vqvae.encode(target)
        x_target = self.vqvae.vq.idx_to_ncm_input(codes)
        prefix = torch.tensor(
            [self.prefix_ids(state_text)] * len(lines), device=self.device
        )
        l0 = self.net.l0_queries.expand(len(lines), -1, -1)
        x = torch.cat([l0, x_target], dim=1)
        pad_mask = prefix == self.pad_id
        with torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"
        ):
            logits = self.net(x, prefix_ids=prefix, prefix_pad_mask=pad_mask)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            codes.reshape(-1),
            reduction="none",
        ).view(len(lines), -1)
        return [float(-row.mean().item()) for row in losses]


class NcmAgent:
    """One order per step: free sample + repair, or legal-candidate rerank."""

    name = "ncm"

    def __init__(
        self,
        sampler: NcmSampler,
        temperature: float = 0.0,
        retries: int = 6,
        retry_temperature: float = 0.5,
        fallback: object | None = None,
        decode: str = "sample",
        rerank_limit: int = 8,
        action_model: ActionModel | str | Path | None = None,
    ):
        if decode not in ("sample", "legal"):
            raise ValueError(f"decode must be 'sample' or 'legal', got {decode!r}")
        self.sampler = sampler
        self.action_model: ActionModel | None
        if isinstance(action_model, ActionModel):
            self.action_model = action_model
        else:
            path = Path(action_model) if action_model else ARTIFACT_DIR / "action_clf.pt"
            self.action_model = (
                ActionModel(path, device=getattr(sampler, "device", None)) if path.exists() else None
            )
        self.temperature = temperature
        self.retries = max(0, int(retries))
        self.retry_temperature = retry_temperature
        self.fallback = fallback if fallback is not None else GreedyAgent()
        self.decode = decode
        self.rerank_limit = max(1, int(rerank_limit))
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
        self.action_model_orders = 0
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

        # The generator collapses rare action characters to "m".  A small
        # learned action selector picks the action character; the full NCM still
        # supplies the arguments through legal candidate lines.  This is a
        # two-stage model, not the scripted ferry advisor.
        if self.action_model is not None and view.get("ferry"):
            order = self._action_model_order(state_text, view, allowed, request.player)
            if order is not None:
                try:
                    line = encode_order(order)
                except (CodecError, KeyError):
                    line = ""
                if line and not (feedback and self._last_line == line):
                    self.action_model_orders += 1
                    self._remember(line, order)
                    self._note_order(view, order)
                    return AgentPlan(orders=[order], commentary="action-model")

        if self.decode == "legal":
            return self._constrained_plan(
                request, state_text, view, allowed, avoid, feedback
            )

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

    def _constrained_plan(
        self,
        request,
        state_text: str,
        view: dict,
        allowed: list[str],
        avoid: str,
        feedback: bool,
    ) -> AgentPlan:
        """Pick a legal order with the model as the ranker.

        One free sample provides the model's intended action/unit.  The legal
        lines for that unit/action are then scored by the same teacher-forced
        objective used in training, and the highest-scoring legal line is
        returned.  This is a constrained decode: the order is always legal,
        and the NCM chooses among the legal alternatives instead of producing
        an illegal string that needs a string-distance repair.
        """
        if not allowed:
            return self._fallback(request, "no-legal-orders")

        try:
            line = self.sampler.sample(state_text, temperature=self.temperature)
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
        raw_line = (
            candidate.line
            if candidate is not None and candidate.kind == "generated"
            else ""
        )
        raw_action = (
            str(candidate.order.get("action"))
            if candidate is not None and candidate.kind == "generated"
            else (line.split()[0] if line.split() else "")
        )

        raw_ranked = rank_legal_orders(
            view, line, allowed=allowed, limit=self.rerank_limit
        )
        tactical, tactical_scores = self._tactical_candidates(
            view, allowed, limit=self.rerank_limit
        )
        merged: dict[str, NearestLegal] = {item.line: item for item in raw_ranked}
        for item in tactical:
            merged.setdefault(item.line, item)
        if tactical:
            ordered: list[NearestLegal] = list(tactical)
            ordered.extend(
                item for item in raw_ranked if item.line not in tactical_scores
            )
            merged = {item.line: item for item in ordered[: self.rerank_limit * 2]}
        candidates = list(merged.values())
        if not candidates:
            nearest = nearest_legal_order(view, line, allowed=allowed)
            candidates = [nearest] if nearest is not None else []

        best: NearestLegal | None = None
        if candidates:
            try:
                scores = self.sampler.score_lines(
                    state_text, [item.line for item in candidates]
                )
            except (RuntimeError, ValueError):
                scores = []
            if len(scores) == len(candidates):
                combined = [
                    (
                        scores[index]
                        + tactical_scores.get(item.line, 0.0)
                        + (2.0 if raw_line and item.line == raw_line else 0.0),
                        item,
                    )
                    for index, item in enumerate(candidates)
                ]
                combined.sort(key=lambda pair: pair[0], reverse=True)
                for _score, item in combined:
                    if avoid and encode_order(item.order) == avoid:
                        continue
                    best = item
                    break
            if best is None:
                for item in candidates:
                    if avoid and encode_order(item.order) == avoid:
                        continue
                    best = item
                    break

        if best is not None:
            encoded = encode_order(best.order)
            chosen_raw = bool(raw_line and best.line == raw_line)
            if chosen_raw:
                self.generated += 1
            else:
                self.constrained += 1
                self.constrained_lines.append(best.line)
                if (
                    tactical_scores.get(best.line, 0.0) > 0
                    and str(best.order.get("action")) != raw_action
                ):
                    self.tactical += 1
            if feedback:
                self.recovered += 1
            self._remember(encoded, best.order)
            self._note_order(view, best.order)
            return AgentPlan(
                orders=[best.order],
                commentary="constrained-raw" if chosen_raw else "legal-tactical",
            )

        return self._fallback(request, "no-legal-order")

    def _tactical_candidates(
        self, view: dict, allowed: list[str], limit: int
    ) -> tuple[list[NearestLegal], dict[str, float]]:
        """Legal orders ranked by winning progress, not just model style."""
        player = view.get("current_player")
        units = {
            str(unit["id"]): unit
            for unit in view.get("units", [])
            if unit.get("pos") is not None
        }
        sites = {tuple(site["pos"]): site for site in view.get("sites", [])}
        enemy_hq = next(
            (
                unit
                for unit in units.values()
                if unit.get("type") == "hq" and unit.get("owner") != player
            ),
            None,
        )
        own_hq = next(
            (
                unit
                for unit in units.values()
                if unit.get("type") == "hq" and unit.get("owner") == player
            ),
            None,
        )
        ctx = {
            "player": player,
            "units": units,
            "sites": sites,
            "unit_actions": view.get("unit_actions") or {},
            "enemy_positions": [
                tuple(unit["pos"])
                for unit in units.values()
                if unit.get("owner") != player
            ],
            "capturable": {
                tuple(site["pos"])
                for site in view.get("sites", [])
                if site.get("owner") != player
                and site.get("capture_needed") is not None
            },
            "enemy_hq_pos": tuple(enemy_hq["pos"]) if enemy_hq is not None else None,
            "home": tuple(own_hq["pos"]) if own_hq is not None else None,
            "persona": getattr(getattr(self, "fallback", None), "persona", None),
        }
        scored: list[tuple[float, NearestLegal]] = []
        for candidate_line in allowed:
            try:
                order = decode_order(candidate_line)
            except CodecError:
                continue
            score = self._tactical_score(order, ctx)
            if score > 0:
                scored.append((score, NearestLegal(candidate_line, order, score)))
        scored.sort(key=lambda pair: (-pair[0], pair[1].line))
        top = scored[: max(1, limit)]
        return [item for _score, item in top], {
            item.line: float(score) for score, item in top
        }

    def _tactical_score(self, order: dict, ctx: dict) -> float:
        action = str(order.get("action"))
        units = ctx["units"]
        unit = units.get(str(order.get("unit")))
        persona = ctx["persona"]
        home = ctx["home"]
        enemy_hq_pos = ctx["enemy_hq_pos"]
        enemy_positions = ctx["enemy_positions"]

        if action == "capture":
            if unit is None:
                return 0.0
            site = ctx["sites"].get(tuple(unit["pos"]))
            if site is None:
                return 0.0
            if site.get("terrain") == "H" and site.get("owner") != ctx["player"]:
                return 500.0
            score = 30.0
            if persona is not None:
                score += 10.0 * float(getattr(persona, "capture_prob", 1.0))
            return score

        if action == "attack":
            target = units.get(str(order.get("target")))
            if target is None or unit is None:
                return 0.0
            preview = 0.0
            attacks = (ctx["unit_actions"].get(str(unit["id"]), {}) or {}).get(
                "attack", []
            )
            for row in attacks:
                if len(row) < 4 or str(row[0]) != str(target["id"]):
                    continue
                if order.get("weapon") and str(row[1]) != str(order.get("weapon")):
                    continue
                preview = float(row[3])
                break
            score = max(1.0, preview)
            if target.get("type") == "hq":
                score += 25.0
            if preview >= float(target.get("hp", 100)):
                score += 30.0
            if persona is not None:
                bonus = float(
                    getattr(persona, "target_bonus", {}).get(
                        str(target.get("type")), 0.0
                    )
                )
                score += bonus * max(1.0, preview)
                if target.get("type") == "hq":
                    score += 10.0 * float(getattr(persona, "hq_bonus", 0.0))
                radius = int(getattr(persona, "defense_radius", 0) or 0)
                if (
                    radius
                    and home is not None
                    and hex_distance(tuple(target["pos"]), home) > radius
                ):
                    score -= 20.0
            return score

        if action == "move":
            if unit is None or not order.get("to"):
                return 0.0
            pos = tuple(unit["pos"])
            dest = tuple(order["to"])
            if enemy_hq_pos is not None:
                gain = max(
                    0,
                    hex_distance(pos, enemy_hq_pos)
                    - hex_distance(dest, enemy_hq_pos),
                )
            elif enemy_positions:
                gain = max(
                    0,
                    min(hex_distance(pos, spot) for spot in enemy_positions)
                    - min(hex_distance(dest, spot) for spot in enemy_positions),
                )
            else:
                gain = 0
            score = gain * 8.0
            if dest in ctx["capturable"]:
                score += 15.0
            if enemy_hq_pos is not None and dest == enemy_hq_pos:
                score += 20.0
            if persona is not None:
                score *= max(0.2, float(getattr(persona, "enemy_hq_bias", 1.0)) + 0.2)
                if home is not None:
                    home_bias = float(getattr(persona, "home_bias", 0.0))
                    if home_bias:
                        score -= home_bias * min(hex_distance(dest, home), 12)
                    radius = int(getattr(persona, "defense_radius", 0) or 0)
                    if radius and hex_distance(dest, home) > radius:
                        score -= 20.0
                standoff = int(getattr(persona, "standoff", 0) or 0)
                if standoff and enemy_positions:
                    d_enemy = min(hex_distance(dest, spot) for spot in enemy_positions)
                    score += max(0.0, 4.0 - abs(d_enemy - standoff)) * 4.0
            return score

        if action == "build":
            unit_type = str(order.get("unit") or "")
            score = 1.0
            if persona is not None:
                score += 1.0 * float(
                    getattr(persona, "build_weights", {}).get(unit_type, 0.0)
                )
            return score

        if action in ("load", "unload"):
            return 1.0
        return 0.0

    def _action_model_order(
        self,
        state_text: str,
        view: dict,
        allowed: list[str],
        player: str,
    ) -> dict | None:
        """Pick a legal action-model order, or None to let the generator decide."""
        assert self.action_model is not None
        action = self.action_model.predict(state_text)
        if action not in ("l", "u", "c"):
            return None
        candidates = [line for line in allowed if line.startswith(action + " ")]
        if not candidates:
            return None

        units = {str(unit.get("id")): unit for unit in view.get("units", ()) if isinstance(unit, dict)}
        if action == "l":
            def load_key(line: str) -> tuple:
                parts = line.split()
                if len(parts) < 3:
                    return (9, 9, 9, line)
                transport = units.get(parts[1], {})
                cargo = units.get(parts[2], {})
                return (
                    0 if transport.get("type") == "transport_ship" else 1,
                    0 if cargo.get("type") == "infantry" else 1,
                    0 if cargo.get("stranded") else 1,
                    line,
                )

            valid = [
                line
                for line in candidates
                if len(line.split()) >= 3
                and units.get(line.split()[2], {}).get("type") != "hq"
                and units.get(line.split()[2], {}).get("move", 0) > 0
            ]
            if not valid:
                return None
            line = min(valid, key=load_key)
        elif action == "u":
            sites = [
                site
                for site in view.get("sites", ())
                if isinstance(site, dict)
                and site.get("capture_needed") is not None
                and site.get("owner") != player
            ]
            objectives = [site for site in sites if site.get("objective")] or sites

            def unload_key(line: str) -> tuple:
                parts = line.split()
                if len(parts) < 4:
                    return (10**6, line)
                try:
                    target = tuple(int(value) for value in parts[3].split(","))
                except (ValueError, IndexError):
                    return (10**6, line)
                distance = (
                    min(hex_distance(target, tuple(site["pos"])) for site in objectives)
                    if objectives
                    else 0
                )
                return (distance, line)

            line = min(candidates, key=unload_key)
        else:
            line = sorted(candidates)[0]

        try:
            return decode_order(line)
        except CodecError:
            return None

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
