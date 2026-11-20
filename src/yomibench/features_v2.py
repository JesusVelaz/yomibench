# Feature version 2, frozen: the encoder checkpoint 1 and older models were trained with. Do not
# edit, not even to fix its known defect (Ninja's stored_momentum_x isn't mirrored): a model's
# weights mean what they meant in training. ValueNet.load plays version-2 models with this module;
# new models use yomibench.features (version 3).
"""Turn a protocol `state` into network inputs, from one seat's point of view.

Egocentric and mirrored: "me" is the seat being evaluated, and x is flipped so the opponent is
always on the positive side; a position and its mirror image get the same features, and so do
p1's and p2's views of a mirror matchup. Numbers are raw (the model standardizes them with
statistics saved at training time); names (fighter state, stance, character, object kind and
state) become vocabulary ids looked up in learned embeddings. Ids are assigned from training data
and saved with the model; a name never seen in training maps to 0.

Each fighter contributes:
- the protocol's own numbers (hp, position, velocity, meter, burst, combo, feints...),
- the current move's timing (length, ticks left, interruptible-at, hitstun),
- shared simulation variables from `raw` (Sadness penalty, combo proration, air movements,
  blockstun, invulnerability...),
- one block per character of that character's resources (Ninja momentum and bombs, Cowboy
  bullets, Robot armor and fuel, Wizard hover and orb, Mutant juke and install), zero-filled
  for the other characters, so one network covers every matchup.

`raw` holds only values that changed since the match started; anything absent is the
character's start value, learned from records (`Vocab.raw_defaults`).

Projectiles and other objects fill OBJECT_SLOTS slots, nearest to me first, plus counts of
each side's objects (a match can have 16+ caltrops out).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OBJECT_SLOTS = 8
PLAYER_NUMS = ("hp_frac", "x", "y", "vx", "vy", "facing", "state_tick", "hitlag", "busy", "grounded",
               "meter", "meter_partial", "burst", "burst_partial", "combo", "air_options", "feints")
TIMING_NUMS = ("state_length", "state_left", "state_iasa", "hitstun")
SHARED_RAW = (
    "penalty", "sadness_immunity_ticks", "air_movements_left", "combo_proration", "hitstun_decay_combo_count",
    "grounded_hits_taken", "wall_slams", "combo_damage", "blockstun_ticks", "wakeup_throw_immunity_ticks",
    "initiative", "invulnerable", "projectile_invulnerable", "throw_invulnerable", "touching_wall",
    "in_blockstring", "feinting", "used_air_dodge", "counterhit_this_turn", "state_interruptable",
    "state_hit_cancellable", "super_meter_used_recently", "parried", "got_parried",
)
# Per-character resources (variable names from each character's script).
CHARACTER_RAW = {
    "Ninja": ("momentum_stores", "stored_momentum_x", "stored_momentum_y", "stored_speed_1", "stored_speed_2",
              "stored_speed_3", "current_momentum", "sticky_bombs_left", "can_summon_kunai", "can_summon_kick",
              "bomb_thrown", "whip_beam_charged", "skull_shaker_bleed_ticks", "can_divekick_hop",
              "used_grappling_hook"),
    "Cowboy": ("bullets_left", "has_gun", "consecutive_shots", "active_time_bullets", "shifting", "drift_effect"),
    "Robot": ("armor_pips", "armor_active", "fly_ticks_left", "loic_meter", "can_loic", "kill_process_super_level",
              "magnet_ticks_left", "magnet_installed", "can_ground_pound", "orbital_strike_out"),
    "Wizard": ("hover_left", "hovering", "fast_falling", "geyser_charge", "orb_duration", "gusts_in_combo",
               "flame_wave_cooldown", "can_flame_wave", "can_vile_clutch"),
    "Mutant": ("install_ticks", "juke_pips", "juke_ticks", "air_vaults", "bc_charge", "can_air_dash"),
}
CHARACTERS = tuple(CHARACTER_RAW)
OBJECT_NUMS = ("present", "mine", "dx", "dy", "vx", "vy", "state_tick")
GLOBAL_NUMS = ("time_left", "dx", "dy", "my_objects", "their_objects")
PLAYER_WIDTH = len(PLAYER_NUMS) + len(TIMING_NUMS) + len(SHARED_RAW) + sum(len(v) for v in CHARACTER_RAW.values())
N_NUMERIC = 2 * PLAYER_WIDTH + OBJECT_SLOTS * len(OBJECT_NUMS) + len(GLOBAL_NUMS)
# Categorical slots, in order: me state, stance, character; opp state, stance, character; then
# each object's kind and state.
CATEGORY_OF_SLOT = ["state", "stance", "char"] * 2 + ["kind", "ostate"] * OBJECT_SLOTS
N_CATEGORICAL = len(CATEGORY_OF_SLOT)
CATEGORIES = ("state", "stance", "char", "kind", "ostate")


@dataclass
class Vocab:
    """Name -> id per category; 0 is unknown/empty. `raw_defaults` maps a character to its raw
    values at the start of a match, for raw entries a state leaves out."""
    ids: dict[str, dict[str, int]] = field(default_factory=lambda: {c: {} for c in CATEGORIES})
    raw_defaults: dict[str, dict[str, Any]] = field(default_factory=dict)
    frozen: bool = False

    def get(self, cat: str, name: str) -> int:
        table = self.ids.setdefault(cat, {})
        if name not in table:
            if self.frozen:
                return 0
            table[name] = len(table) + 1
        return table[name]

    def sizes(self) -> dict[str, int]:
        return {cat: len(self.ids.get(cat, {})) + 1 for cat in CATEGORIES}

    def learn_defaults(self, state: dict[str, Any]) -> None:
        """Record each character's start values from a match's first state (its `raw_start`)."""
        start = state.get("raw_start")
        if not start:
            return
        for p, raw in zip(state["players"], start):
            self.raw_defaults.setdefault(str(p.get("character", "")), dict(raw))


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:  # fixed-point values arrive as strings
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _player(p: dict[str, Any], flip: float, vocab: Vocab, nums: list[float], cats: list[int]) -> None:
    char = str(p.get("character", ""))
    raw = p.get("raw", {})
    defaults = vocab.raw_defaults.get(char, {})

    def rv(name: str) -> float:
        return _num(raw[name] if name in raw else defaults.get(name, 0))

    length = p.get("state_length") or 0
    nums += [
        p["hp"] / p["max_hp"], p["x"] * flip, p.get("y", 0.0), p.get("vx", 0.0) * flip, p.get("vy", 0.0),
        p.get("facing", 1) * flip, p.get("state_tick", 0), p.get("hitlag", 0), float(p.get("busy_interrupt", False)),
        float(p.get("grounded", True)), p.get("meter", 0), p.get("meter_partial", 0), p.get("burst", 0),
        p.get("burst_partial", 0), p.get("combo", 0), p.get("air_options", 0), p.get("feints", 0),
        length, max(0, length - p.get("state_tick", 0)), _num(p.get("state_iasa")), _num(p.get("hitstun")),
    ]
    nums += [rv(name) for name in SHARED_RAW]
    for c, names in CHARACTER_RAW.items():
        nums += [rv(name) for name in names] if c == char else [0.0] * len(names)
    cats += [vocab.get("state", str(p.get("state", ""))), vocab.get("stance", str(p.get("stance", ""))),
             vocab.get("char", char)]


def encode(state: dict[str, Any], seat: int, vocab: Vocab) -> tuple[list[float], list[int]]:
    """(numeric features, categorical ids) for `seat`."""
    me, opp = state["players"][seat], state["players"][1 - seat]
    flip = 1.0 if me["x"] <= opp["x"] else -1.0
    nums: list[float] = []
    cats: list[int] = []
    _player(me, flip, vocab, nums, cats)
    _player(opp, flip, vocab, nums, cats)
    my_id = seat + 1  # objects' owner is the player id
    objects = state.get("objects", [])
    objs = sorted(objects, key=lambda o: abs(o["x"] - me["x"]) + abs(o["y"] - me.get("y", 0.0)))
    for k in range(OBJECT_SLOTS):
        if k < len(objs):
            o = objs[k]
            nums += [1.0, float(o.get("owner") == my_id), (o["x"] - me["x"]) * flip, o["y"] - me.get("y", 0.0),
                     o.get("vx", 0.0) * flip, o.get("vy", 0.0), o.get("state_tick", 0)]
            cats += [vocab.get("kind", str(o.get("kind", o.get("name", "")))), vocab.get("ostate", str(o.get("state", "")))]
        else:
            nums += [0.0] * len(OBJECT_NUMS)
            cats += [0, 0]
    # Time left rather than the tick, so a different game length doesn't shift the input.
    time_left = state.get("time_left", 3600 - state.get("tick", 0))
    mine = sum(o.get("owner") == my_id for o in objects)
    nums += [time_left, (opp["x"] - me["x"]) * flip, opp.get("y", 0.0) - me.get("y", 0.0), mine, len(objects) - mine]
    return nums, cats
