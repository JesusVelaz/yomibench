"""Seat symmetry of the features: p2's view of a state must be p1's view of its mirror image.

mirror(s) swaps the players and reflects x about the stage centre (x = 0): positions, velocities,
facing, DI x, aims, directional raw fields (MIRRORED_RAW), objects (owner ids swap) and hitbox
facing. Any column that differs is information about which seat the net plays, which an
antisymmetric value net can still learn from (round 3's first candidates did: a Cowboy start
value read +100 from one seat and -100 from the other)."""
import copy

import pytest

from yomibench import features as F
from yomibench.features import Vocab, encode


def _flip_aims(d):
    if isinstance(d, dict):
        if "x" in d and "y" in d:
            d["x"] = -F._num(d["x"])
        else:
            for v in d.values():
                _flip_aims(v)


def _mirror_player(p):
    q = copy.deepcopy(p)
    q["x"], q["vx"], q["facing"] = -p["x"], -F._num(p.get("vx", 0)), -F._num(p.get("facing", 1))
    if isinstance(q.get("di"), dict):
        q["di"]["x"] = -F._num(q["di"].get("x"))
    for k in list(q.get("raw") or {}):
        if k in F.MIRRORED_RAW:
            q["raw"][k] = -F._num(q["raw"][k])
    _flip_aims(q.get("state_data"))
    for h in q.get("hitboxes") or []:
        h["facing"] = -F._num(h.get("facing", 1))
    return q


def mirror(s):
    m = copy.deepcopy(s)
    m["players"] = [_mirror_player(s["players"][1]), _mirror_player(s["players"][0])]
    for o in m.get("objects", []):
        o["x"], o["vx"] = -o["x"], -F._num(o.get("vx", 0))
        if o.get("owner") in (1, 2):
            o["owner"] = 3 - o["owner"]
        for h in o.get("hitboxes") or []:
            h["facing"] = -F._num(h.get("facing", 1))
    if "raw_start" in m:  # each player's start values travel with the player
        m["raw_start"] = [dict(s["raw_start"][1]), dict(s["raw_start"][0])]
        for r in m["raw_start"]:
            for k in list(r):
                if k in F.MIRRORED_RAW:
                    r[k] = -F._num(r[k])
    return m


def _fighter(char, x, facing, raw=None, state="QuickSlash"):
    """A set-4 fighter as the sim sends it, with every v3 move field."""
    return {"character": char, "hp": 1100, "max_hp": 1500, "x": x, "y": -20, "vx": 2.5 * facing, "vy": 1.0,
            "facing": facing, "state": state, "state_tick": 3, "state_length": 20, "state_iasa": 13, "hitstun": None,
            "stance": "Normal", "meter": 1, "meter_partial": 43, "burst": 1, "combo": 2, "air_options": 1, "feints": 2,
            "raw": dict(raw or {}), "state_data": {"Direction": {"x": 35 * facing, "y": -35}, "Distance": {"x": 60}},
            "di": {"x": str(100 * facing), "y": "0"}, "combo_moves_used": {"QuickSlash": 2, "Uppercut": 1},
            "hitboxes": [{"active": False, "start_tick": 5, "active_ticks": 3, "damage": 80, "dir_x": "1.0",
                          "dir_y": "-0.5", "knockback": "9.0", "hit_height": 1, "facing": facing}]}


def _object(owner, x, vx, facing):
    return {"name": "3", "kind": "Kunai", "owner": owner, "x": x, "y": -40, "vx": vx, "vy": 0.5, "state": "Default",
            "state_tick": 4, "hitboxes": [{"active": True, "start_tick": 1, "active_ticks": 5, "damage": 70,
                                           "dir_x": "1.0", "dir_y": "-1.0", "knockback": "5.0", "hit_height": 1,
                                           "facing": facing}]}


def _states(a, b):
    """Positions of an a-v-b match: either side left, objects of both owners, raw fields given and left out."""
    raw = {"Ninja": {"stored_momentum_x": "7.5", "quick_slash_move_dir_x": "-0.7", "sticky_bombs_left": 2},
           "Cowboy": {"bullets_left": 5, "shot_dir_x": -100, "fatal_cut_start_pos_x": -364, "has_gun": True}}
    out = []
    for xs in ((-125, 125), (240, -60), (-300, -290)):
        for with_raw in (True, False):
            p1 = _fighter(a, xs[0], 1 if xs[0] < xs[1] else -1, raw[a] if with_raw else {})
            p2 = _fighter(b, xs[1], -1 if xs[0] < xs[1] else 1, raw[b] if with_raw else {}, state="Idle")
            out.append({"tick": 120, "time_left": 3400, "players": [p1, p2],
                        "objects": [_object(1, xs[0] + 40, 6.0, 1), _object(2, xs[1] - 30, -4.0, -1),
                                    _object(2, -10, 0.0, 1)]})
    return out


PAIRS = [("Ninja", "Ninja"), ("Ninja", "Cowboy"), ("Cowboy", "Ninja"), ("Cowboy", "Cowboy")]


@pytest.mark.parametrize("pair", PAIRS)
def test_every_column_of_both_views_is_a_true_mirror(pair):
    vocab = Vocab()
    # Start values as a set-4 match sends them: each fighter's own, directional ones in world x
    # (the left Cowboy aims right, +100; the right one left, -100).
    start = {"tick": 0, "players": [_fighter(pair[0], -125, 1), _fighter(pair[1], 125, -1)], "objects": [],
             "raw_start": [{"shot_dir_x": 100, "bullets_left": 6, "stored_momentum_x": 0, "has_gun": True},
                           {"shot_dir_x": -100, "bullets_left": 6, "stored_momentum_x": 0, "has_gun": True}]}
    vocab.learn_defaults(start)
    for s in [start] + _states(*pair):
        m = mirror(s)
        for seat in (0, 1):
            a, b = encode(s, seat, vocab), encode(m, 1 - seat, vocab)
            assert a[1] == b[1], ("categorical", seat)
            diff = [i for i, (x, y) in enumerate(zip(a[0], b[0])) if abs(x - y) > 1e-9]
            assert not diff, (seat, diff)


def test_learned_directional_start_values_are_seat_neutral():
    v = Vocab()
    v.learn_defaults({"players": [{"character": "Cowboy"}, {"character": "Cowboy"}],
                      "raw_start": [{"shot_dir_x": 100, "bullets_left": 6}, {"shot_dir_x": -100, "bullets_left": 6}]})
    assert v.raw_defaults["Cowboy"] == {"shot_dir_x": 0, "bullets_left": 6}
    # What it guards against: a nonzero directional default reads differently from the two seats.
    s = {"tick": 5, "players": [_fighter("Cowboy", -125, 1), _fighter("Cowboy", 125, -1)], "objects": []}
    biased = Vocab(raw_defaults={"Cowboy": {"shot_dir_x": 100}})
    assert encode(s, 1, biased)[0] != encode(mirror(s), 0, biased)[0]
    assert encode(s, 1, v)[0] == encode(mirror(s), 0, v)[0]


@pytest.mark.xfail(strict=True, reason="equal x: both views keep flip +1 (encode's `me.x <= opp.x`), so a tie state's "
                                       "views aren't mirrors; ~0.7% of round-3 states. Fixing it changes the encoder "
                                       "(and the simulator plugin's native copy), so it waits for features v4.")
def test_equal_positions_are_mirrors_too():
    s = {"tick": 5, "players": [_fighter("Ninja", -40, 1), _fighter("Ninja", -40, -1)], "objects": []}
    vocab = Vocab()
    assert encode(s, 1, vocab) == encode(mirror(s), 0, vocab)
