"""Turn a protocol `state` into network inputs, from one seat's point of view.

Egocentric and mirrored: "me" is the seat being evaluated, and x is flipped so the opponent is
always on the positive side; a position and its mirror image get the same features, and so do
p1's and p2's views of a mirror matchup. Every horizontal quantity is flipped: positions,
velocities, facing, aim and DI vectors, hitbox directions, and the raw fields in MIRRORED_RAW.
Numbers are raw (the model standardizes them with statistics saved at training time); names
(fighter state, stance, character, object kind and state) become vocabulary ids looked up in
learned embeddings. Ids are assigned from training data and saved with the model; a name never
seen in training maps to 0.

Each fighter contributes:
- the protocol's own numbers (hp, position, velocity, meter, burst, combo, feints...),
- the current move's timing (length, ticks left, interruptible-at, hitstun),
- the current move itself (MOVE_NUMS): its submitted data (aim direction, slider amount, parry
  timing and height), the fighter's DI, move staling (`combo_moves_used`), and the move's
  hitboxes (pending or active: when, how hard, which way),
- shared simulation variables from `raw` (Sadness penalty, combo proration, air movements,
  blockstun, invulnerability...),
- one block per character of that character's resources (Ninja momentum and bombs, Cowboy
  bullets, Robot armor and fuel, Wizard hover and orb, Mutant juke and install), zero-filled
  for the other characters, so one network covers every matchup.

`raw` holds only values that changed since the match started; anything absent is the
character's start value, learned from records (`Vocab.raw_defaults`). The move fields
(`state_data`, `di`, `combo_moves_used`, `hitboxes`) arrived with engine state v3; older records
lack them, so they encode as zeros with the `v3` flag off (the loader fills `di` back in from the
recorded picks).

Objects fill OBJECT_SLOTS slots, nearest to either fighter first, each with its hitbox threat.
Every object, not only the slotted ones, feeds a per-side threat summary (THREAT_NUMS): how many,
how many with an active hitbox, their damage, the nearest one to its target, the damage close to
the target, and the soonest one closing in (a match can have 16+ caltrops out).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

OBJECT_SLOTS = 8
PLAYER_NUMS = ("hp_frac", "x", "y", "vx", "vy", "facing", "state_tick", "hitlag", "busy", "grounded",
               "meter", "meter_partial", "burst", "burst_partial", "combo", "air_options", "feints")
TIMING_NUMS = ("state_length", "state_left", "state_iasa", "hitstun")
MOVE_NUMS = (
    "v3",  # 1 when the state carries the move fields below (engine state v3)
    "aim_x", "aim_y", "aim_len", "amount", "parry_timing", "parry_height", "di_x", "di_y",
    "moves_used", "moves_distinct", "moves_max_uses", "this_move_uses",
    "hit_pending", "hit_active", "hit_in", "hit_damage", "hit_dir_x", "hit_dir_y", "hit_knockback", "hit_height",
)
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
              "used_grappling_hook", "quick_slash_move_dir_x", "quick_slash_start_pos_x"),
    "Cowboy": ("bullets_left", "has_gun", "consecutive_shots", "active_time_bullets", "shifting", "drift_effect",
               "lightning_slice_x", "shot_dir_x", "shot_dir_y", "stance_teleport_x", "stance_teleport_y",
               "fatal_cut_move_dir_x", "fatal_cut_move_dir_y", "fatal_cut_start_pos_x", "fatal_cut_start_pos_y"),
    "Robot": ("armor_pips", "armor_active", "fly_ticks_left", "loic_meter", "can_loic", "kill_process_super_level",
              "magnet_ticks_left", "magnet_installed", "can_ground_pound", "orbital_strike_out", "loic_dir"),
    "Wizard": ("hover_left", "hovering", "fast_falling", "geyser_charge", "orb_duration", "gusts_in_combo",
               "flame_wave_cooldown", "can_flame_wave", "can_vile_clutch"),
    "Mutant": ("install_ticks", "juke_pips", "juke_ticks", "air_vaults", "bc_charge", "can_air_dash",
               "juke_dir_x", "juke_dir_y"),
}
# Raw values in world x (positions, directions, signed speeds): flipped with x. Audited against
# every raw field seen in records of all five characters (fields whose values take both signs,
# names with x/dir/vel/force/pos/facing) and against each character's script. Cowboy's
# lightning_slice_x is an x quantity kept relative to the fighter's facing (always positive in
# records, whichever way the Cowboy faces), so it is not flipped. Tests check every CHARACTER_RAW
# name that looks horizontal is in one of the two sets.
MIRRORED_RAW = frozenset({"stored_momentum_x", "quick_slash_move_dir_x", "quick_slash_start_pos_x",
                          "shot_dir_x", "stance_teleport_x", "fatal_cut_move_dir_x", "fatal_cut_start_pos_x",
                          "loic_dir", "juke_dir_x"})
FACING_RELATIVE_RAW = frozenset({"lightning_slice_x"})
CHARACTERS = tuple(CHARACTER_RAW)
OBJECT_NUMS = ("present", "mine", "dx", "dy", "vx", "vy", "state_tick",
               "hit_active", "hit_damage", "hit_dir_x", "hit_dir_y")
THREAT_NUMS = ("count", "active", "active_damage", "nearest", "damage_near", "soonest")
GLOBAL_NUMS = ("time_left", "dx", "dy") + tuple(f"{side}_{n}" for side in ("mine", "theirs") for n in THREAT_NUMS)
PLAYER_WIDTH = (len(PLAYER_NUMS) + len(TIMING_NUMS) + len(MOVE_NUMS) + len(SHARED_RAW)
                + sum(len(v) for v in CHARACTER_RAW.values()))
N_NUMERIC = 2 * PLAYER_WIDTH + OBJECT_SLOTS * len(OBJECT_NUMS) + len(GLOBAL_NUMS)
# A name per numeric input, in encode()'s order (for logs and diagnostics): "me."/"opp." player fields (a
# character's own raw fields as "me.Wizard.hover_left"), "obj<k>." object fields, "global." fields.
NUMERIC_NAMES = tuple(
    [f"{side}.{n}" for side in ("me", "opp")
     for n in (*PLAYER_NUMS, *TIMING_NUMS, *MOVE_NUMS, *SHARED_RAW,
               *(f"{c}.{r}" for c, raws in CHARACTER_RAW.items() for r in raws))]
    + [f"obj{k}.{n}" for k in range(OBJECT_SLOTS) for n in OBJECT_NUMS]
    + [f"global.{n}" for n in GLOBAL_NUMS])
assert len(NUMERIC_NAMES) == N_NUMERIC
# Categorical slots, in order: me state, stance, character; opp state, stance, character; then
# each object's kind and state.
CATEGORY_OF_SLOT = ["state", "stance", "char"] * 2 + ["kind", "ostate"] * OBJECT_SLOTS
N_CATEGORICAL = len(CATEGORY_OF_SLOT)
CATEGORIES = ("state", "stance", "char", "kind", "ostate")

FAR = 1000.0       # "no such object" distance, about the stage width
NEAR = 80.0        # an active hitbox this close to its target counts as about to hit
SOON = 120.0       # "never" for ticks until an object reaches its target
NO_HIT_YET = 60.0  # "no pending hitbox" for ticks until a fighter's next hitbox


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
        """Record each character's start values from a match's first state (its `raw_start`).

        Directional (MIRRORED_RAW) start values are stored as 0. A default stands in for a raw
        field the state leaves out, and encode multiplies a mirrored field by the view's flip; a
        world-x start value (Cowboy's shot_dir_x starts at 100: aiming right, as the left player
        does) would then read +100 from one seat's view and -100 from the other's on every state,
        a constant that tells the net which seat it plays. The set-4 sim leaves those fields out
        of every state, so this is the only value they ever take. 0 is the same from both views."""
        start = state.get("raw_start")
        if not start:
            return
        for p, raw in zip(state["players"], start):
            self.raw_defaults.setdefault(str(p.get("character", "")),
                                         {k: (0 if k in MIRRORED_RAW else v) for k, v in raw.items()})


def _num(v: Any) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    try:  # fixed-point values arrive as strings
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _move_data(data: Any) -> tuple[float, float, float, float, float]:
    """(aim x, aim y, amount, parry timing, parry height) from a move's submitted data, the same
    dicts candidates carry: {"x", "y"} aims (also nested, e.g. QuickSlash's "Direction"), {"x"}
    sliders (dash distance, nested as "Distance"), parry {"Melee Parry Timing": {"count"},
    "Block Height": {"x", "y"}}. Anything else (flags, strings, plain ints) gives zeros."""
    out = {"aim": None, "amount": 0.0, "timing": 0.0, "height": 0.0}

    def walk(d: Any, key: str) -> None:
        if not isinstance(d, dict):
            return
        if key == "Block Height":
            out["height"] = _num(d.get("y"))
        elif "x" in d and "y" in d:
            if out["aim"] is None:
                out["aim"] = (_num(d["x"]), _num(d["y"]))
        elif "x" in d:
            out["amount"] = _num(d["x"])
        elif "count" in d:
            out["timing"] = _num(d["count"])
        else:
            for k, v in d.items():
                walk(v, k)

    walk(data, "")
    ax, ay = out["aim"] or (0.0, 0.0)
    return ax, ay, out["amount"], out["timing"], out["height"]


def _main_hitbox(hbs: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The hardest-hitting of `hbs` (first on ties)."""
    return max(hbs, key=lambda h: _num(h.get("damage")), default=None)


def _move(p: dict[str, Any], flip: float) -> list[float]:
    if "hitboxes" not in p:  # a pre-v3 state: no move fields (`di` may have been filled in)
        di = p.get("di") or {}
        return [0.0] * 7 + [_num(di.get("x")) * flip, _num(di.get("y"))] + [0.0] * (len(MOVE_NUMS) - 9)
    ax, ay, amount, timing, height = _move_data(p.get("state_data"))
    aim_len = math.hypot(ax, ay)
    if aim_len:
        ax, ay = ax / aim_len, ay / aim_len
    di = p.get("di") or {}
    used = {k: _num(v) for k, v in (p.get("combo_moves_used") or {}).items()}
    tick = _num(p.get("state_tick"))
    pending = [h for h in p["hitboxes"]
               if h.get("active") or tick < _num(h.get("start_tick")) + _num(h.get("active_ticks"))]
    main = _main_hitbox(pending)
    # A fighter's hitboxes point the way the fighter faces (a hitbox's own facing is only kept up
    # to date while it is active).
    facing = _num(p.get("facing", 1)) or 1.0
    return [
        1.0, ax * flip, ay, aim_len, amount, timing, height, _num(di.get("x")) * flip, _num(di.get("y")),
        sum(used.values()), len(used), max(used.values(), default=0.0), used.get(str(p.get("state", "")), 0.0),
        len(pending), float(any(h.get("active") for h in pending)),
        min((max(0.0, _num(h.get("start_tick")) - tick) for h in pending), default=NO_HIT_YET),
        _num(main.get("damage")) if main else 0.0,
        _num(main.get("dir_x")) * facing * flip if main else 0.0, _num(main.get("dir_y")) if main else 0.0,
        _num(main.get("knockback")) if main else 0.0, _num(main.get("hit_height")) if main else 0.0,
    ]


def _player(p: dict[str, Any], flip: float, vocab: Vocab, nums: list[float], cats: list[int]) -> None:
    char = str(p.get("character", ""))
    raw = p.get("raw", {})
    defaults = vocab.raw_defaults.get(char, {})

    def rv(name: str) -> float:
        v = _num(raw[name] if name in raw else defaults.get(name, 0))
        return v * flip if name in MIRRORED_RAW else v

    length = p.get("state_length") or 0
    nums += [
        p["hp"] / p["max_hp"], p["x"] * flip, p.get("y", 0.0), p.get("vx", 0.0) * flip, p.get("vy", 0.0),
        p.get("facing", 1) * flip, p.get("state_tick", 0), p.get("hitlag", 0), float(p.get("busy_interrupt", False)),
        float(p.get("grounded", True)), p.get("meter", 0), p.get("meter_partial", 0), p.get("burst", 0),
        p.get("burst_partial", 0), p.get("combo", 0), p.get("air_options", 0), p.get("feints", 0),
        length, max(0, length - p.get("state_tick", 0)), _num(p.get("state_iasa")), _num(p.get("hitstun")),
    ]
    nums += _move(p, flip)
    nums += [rv(name) for name in SHARED_RAW]
    for c, names in CHARACTER_RAW.items():
        nums += [rv(name) for name in names] if c == char else [0.0] * len(names)
    cats += [vocab.get("state", str(p.get("state", ""))), vocab.get("stance", str(p.get("stance", ""))),
             vocab.get("char", char)]


def _object_hit(o: dict[str, Any], flip: float) -> tuple[float, float, float, float]:
    """(active, damage, dir x, dir y) of an object's hardest-hitting active hitbox."""
    main = _main_hitbox([h for h in o.get("hitboxes", ()) if h.get("active")])
    if main is None:
        return 0.0, 0.0, 0.0, 0.0
    facing = _num(main.get("facing", 1)) or 1.0
    return 1.0, _num(main.get("damage")), _num(main.get("dir_x")) * facing * flip, _num(main.get("dir_y"))


def _threats(objs: list[dict[str, Any]], hits: list[tuple[float, ...]], target: dict[str, Any]) -> list[float]:
    """THREAT_NUMS for one side's objects against `target` (a fighter)."""
    tx, ty = target["x"], target.get("y", 0.0)
    active = damage = near = 0.0
    nearest, soonest = FAR, SOON
    for o, (on, dmg, _, _) in zip(objs, hits):
        if not on:
            continue
        rx, ry = tx - o["x"], ty - o["y"]
        dist = math.hypot(rx, ry)
        active += 1
        damage += dmg
        nearest = min(nearest, dist)
        if dist < NEAR:
            near += dmg
        closing = (rx * o.get("vx", 0.0) + ry * o.get("vy", 0.0)) / dist if dist else 0.0
        if closing > 0:
            soonest = min(soonest, dist / closing)
    return [len(objs), active, damage, nearest, near, soonest]


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
    mx, my = me["x"], me.get("y", 0.0)
    ox, oy = opp["x"], opp.get("y", 0.0)

    def closeness(o: dict[str, Any]) -> float:
        return min(abs(o["x"] - mx) + abs(o["y"] - my), abs(o["x"] - ox) + abs(o["y"] - oy))
    objs = sorted(objects, key=closeness)
    hits = [_object_hit(o, flip) for o in objs]
    for k in range(OBJECT_SLOTS):
        if k < len(objs):
            o = objs[k]
            nums += [1.0, float(o.get("owner") == my_id), (o["x"] - mx) * flip, o["y"] - my,
                     o.get("vx", 0.0) * flip, o.get("vy", 0.0), o.get("state_tick", 0), *hits[k]]
            cats += [vocab.get("kind", str(o.get("kind", o.get("name", "")))), vocab.get("ostate", str(o.get("state", "")))]
        else:
            nums += [0.0] * len(OBJECT_NUMS)
            cats += [0, 0]
    # Time left rather than the tick, so a different game length doesn't shift the input.
    time_left = state.get("time_left", 3600 - state.get("tick", 0))
    nums += [time_left, (ox - mx) * flip, oy - my]
    mine = [k for k, o in enumerate(objs) if o.get("owner") == my_id]
    theirs = [k for k, o in enumerate(objs) if o.get("owner") != my_id]
    nums += _threats([objs[k] for k in mine], [hits[k] for k in mine], opp)
    nums += _threats([objs[k] for k in theirs], [hits[k] for k in theirs], me)
    return nums, cats
