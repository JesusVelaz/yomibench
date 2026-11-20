"""The state-rarity boost for least-tried exploration (settings "state_boost"): per-seat p scaled by how
rarely the seat's own engine state is reached, tapered per state class like move types, retired per
character, off by default. Generic: no move, state or character is named by the code."""
import collections
import json
import random

import pytest

from test_least_tried import cand, play, strip_timing, write_counts

from yomibench import least_tried
from yomibench.least_tried import LeastTried


def lt(states, state_taper=None, cap=10.0, p=0.03, counts=None):
    return LeastTried(p, 2, 100, counts or {"X": {"A": 5, "B": 9}}, {}, "0" * 64, 0, {}, 1.0, False, cap,
                      states, state_taper or {})


def test_settings_and_header():
    assert least_tried.parse_settings({"state_boost": 10}) == {"p": 0.03, "k": 2, "window_games": 20000,
                                                               "state_boost": 10.0}
    assert least_tried.parse_settings({"state_boost": 1}) == {"p": 0.03, "k": 2, "window_games": 20000}
    for bad in (0.5, 11, True, "x"):
        with pytest.raises(ValueError):
            least_tried.parse_settings({"state_boost": bad})
    assert "state_boost" not in lt({}, cap=0).header() and lt({}, cap=10).header()["state_boost"] == 10


def test_boost_rule_caps_decays_and_tapers():
    states = {"X": {"common1": 9000, "common2": 8000, "mid": 4000, "rare": 10, "never": 0}}
    b = lt(states)
    assert b.boost("X", "common1") == 1.0  # at or above the median: no boost
    assert b.boost("X", "rare") == pytest.approx(1 + 9 * (1 - 10 / 2000))  # capped at 10, decayed a little
    assert b.boost("X", "unseen") == pytest.approx(1 + 9 * 1.0)  # a class never counted: the cap
    # continuous decay as the count rises toward the minimum positions, never a cliff
    vals = [lt({"X": {**states["X"], "rare": n}}).boost("X", "rare") for n in (10, 100, 500, 1000, 1500, 1999)]
    assert all(a >= c for a, c in zip(vals, vals[1:])) and vals[-1] > 1
    # enough positions but the error not normal yet: the floor share of the excess remains
    assert lt({"X": {**states["X"], "rare": 2500}}).boost("X", "rare") == pytest.approx(
        1 + (min(10, (4000 / 2500) ** 0.5) - 1) * least_tried.STATE_DECAY_FLOOR)
    # caught up (coverage state_taper): exactly 1; a retired character: 1 everywhere
    assert lt(states, {"X": {"caught": ["rare"], "retired": False}}).boost("X", "rare") == 1.0
    assert lt(states, {"X": {"caught": [], "retired": True}}).boost("X", "never") == 1.0
    # off, without counts, or without a state: 1
    assert lt(states, cap=0).boost("X", "rare") == 1.0 and lt({}).boost("X", "rare") == 1.0
    assert lt(states).boost("X", None) == 1.0 and lt(states).boost("Y", "rare") == 1.0


def test_choose_is_unchanged_with_every_boost_one_and_boosts_a_rare_state():
    cands = [[cand("A"), cand("B")], [cand("A"), cand("B")]]
    plain = LeastTried(0.03, 2, 100, {"X": {"A": 5, "B": 9}}, {}, "0" * 64, 0)
    flat = lt({"X": {"s": 100, "t": 100}})  # every class at the median: boost 1
    for seed in range(300):
        a = plain.choose(cands, ("X", "X"), [0, 0], [True, True], random.Random(seed))
        b = flat.choose(cands, ("X", "X"), [0, 0], [True, True], random.Random(seed), None, ["s", "t"])
        if a is None:
            assert b is None
        else:
            info = dict(b[3])
            assert info.pop("state") in ("s", "t") and info.pop("boost") == 1.0
            assert a == (b[0], b[1], b[2], info)
    # a seat in a rare state explores up to cap x p as often; its pick records the boost
    rare = lt({"X": {"common": 9000, "other": 9000, "rare": 1}})
    hits = collections.Counter()
    for seed in range(4000):
        got = rare.choose(cands, ("X", "X"), [0, 0], [True, True], random.Random(seed), None, ["rare", "common"])
        if got:
            hits[got[0]] += 1
            if got[0] == 0:
                assert got[3]["state"] == "rare" and got[3]["boost"] > 9
    assert hits[0] > 5 * max(1, hits[1])  # seat 0's p ~0.3 vs seat 1's 0.03
    # k still caps picks per seat
    assert rare.choose(cands, ("X", "X"), [2, 2], [True, True], random.Random(1), None, ["rare", "rare"]) is None


def test_seat_states_and_counting_by_state():
    line = json.dumps({"turn": 1, "state": {"tick": 3, "players": [
        {"character": "X", "hp": 1, "state": "Alpha", "raw": {"state": "nested"}},
        {"character": "Y", "hp": 2, "state": "Beta"}]}, "candidates": [[{"action": "A"}, {"action": "B"}],
                                                                       [{"action": "A"}]], "picks": [1, 0]},
                      separators=(",", ":"))
    assert least_tried.seat_states(line) == ["Alpha", "Beta"]
    counts, states = {}, {}
    least_tried.count_record_lines([json.dumps({"characters": ["X", "Y"]}), line], counts, None, states)
    assert counts == {"X": {"B": 1}} and states == {"X": {"Alpha": 1}}  # seat 1 had one candidate: not counted


def test_state_taper_from_coverage_and_counts_file(tmp_path):
    cov = {"state_taper": {"X": {"caught": ["b", "a"], "retired": False, "behind": ["c"]}, "Y": "bad"}}
    assert least_tried.state_taper_from_coverage(cov) == {"X": {"caught": ["a", "b"], "retired": False}}
    assert least_tried.state_taper_from_coverage(None) == {}
    doc = {"format": least_tried.COUNTS_FORMAT, "window_games": 100, "games": 7, "tars": [], "counts": {},
           "states": {"X": {"a": 3}}, "state_taper": {"X": {"caught": ["a"], "retired": True}}, "taper": {},
           "created": 0}
    path = tmp_path / "c.json"
    least_tried.write_counts(doc, path)
    loaded = LeastTried.load({"p": 0.03, "k": 2, "window_games": 100, "state_boost": 10.0}, path)
    assert loaded.states == {"X": {"a": 3}} and loaded.state_taper["X"]["retired"] is True and loaded.state_boost == 10


def test_coverage_state_taper_retires_and_counts_boosted_picks():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("cov", Path(__file__).resolve().parents[1] / "tools" / "move_type_coverage.py")
    cov = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cov)
    sstats = {"X": {"a": {"positions": 3000, "mse_n": 3000, "mse_sum": 900.0},
                    "b": {"positions": 2500, "mse_n": 2500, "mse_sum": 800.0},
                    "c": {"positions": 40, "mse_n": 40, "mse_sum": 12.0}},
              "Y": {"a": {"positions": 2100, "mse_n": 2100, "mse_sum": 600.0}}}
    t = cov._state_taper(sstats, {"X": {"least_tried_boosted": 7}}, 2000, 1.2, 50)
    assert t["X"]["caught"] == ["a", "b"] and t["X"]["behind"] == ["c"] and t["X"]["retired"] is False
    assert t["X"]["boosted_picks"] == 7 and t["Y"]["retired"] is True


def test_selfplay_with_the_boost_on_and_no_state_counts_plays_as_without_it(tmp_path):
    counts = {"Toy": {"Wait": 900, "Walk": 800, "Jab": 700, "Heavy": 5, "Grab": 600, "Block": 400}}
    path, _ = write_counts(tmp_path, counts)
    plain = LeastTried.load({"p": 0.5, "k": 2, "window_games": 100}, path)
    boosted = LeastTried.load({"p": 0.5, "k": 2, "window_games": 100, "state_boost": 10.0}, path)
    a = play(tmp_path / "a", games=4, explore_least_tried=plain)
    b = play(tmp_path / "b", games=4, explore_least_tried=boosted)
    for x, y in zip(a, b):
        assert y[0]["explore_least_tried"].pop("state_boost") == 10.0
        for tx, ty in zip(strip_timing(x), strip_timing(y)):
            if "explore_pick" in ty:
                ty["explore_pick"].pop("state")
                assert ty["explore_pick"].pop("boost") == 1.0  # no state counts: boost 1
            assert tx == ty
