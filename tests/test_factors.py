"""Candidate set 4 on the client: factors, `expand`, refinement near the support, records."""

import copy
import random

from yomibench import toy
from yomibench.arena import play_game
from yomibench.bots import MatrixBot, expand_near_support, make_bot, turn_candidates
from yomibench.engine import Candidate, Engine, InProcess
from yomibench.search import evaluate
from yomibench.selfplay import play_recorded


class FactorToy(toy.ToyEngine):
    """The toy game with one set-4 factor: attacks can be feinted, and a feinted attack goes
    through Block for full damage. `expand` appends overridden copies to this decision's list,
    as the real engine does."""

    def __init__(self):
        super().__init__()
        self.added = [[], []]
        self.msgs = []

    def _cands(self, state, seat):
        out = []
        for c in toy.candidates(state, seat):
            if c["action"] in toy.ATTACKS:
                c = {**c, "factors": {"feint": True}, "values": {"feint": False}}
            out.append(c)
        return out + self.added[seat]

    def _obs(self, state, with_candidates=True):
        obs = super()._obs(state, with_candidates)
        if with_candidates and not obs["done"]:
            obs["candidates"] = [self._cands(state, i) for i in (0, 1)]
        return obs

    def _picks(self, state, ids):
        out = []
        for seat, cid in enumerate(ids):
            cands = self._cands(state, seat)
            if not isinstance(cid, int) or not 0 <= cid < len(cands):
                raise ValueError(f"seat {seat}: no candidate {cid!r}")
            out.append(cands[cid])
        a, b = out
        # A feinted attack against Block lands as if the defender had done nothing.
        for i, (me, them) in enumerate(((a, b), (b, a))):
            if (me.get("extra") or {}).get("feint") and them["action"] == "Block":
                out[1 - i] = {**them, "action": "Wait"}
        return out[0], out[1]

    def handle(self, msg):
        self.msgs.append(msg)
        if msg.get("cmd") == "new_match":
            self.added = [[], []]
        if msg.get("cmd") == "step":
            out = super().handle(msg)
            self.added = [[], []]
            return {**out, "candidates": [self._cands(self.state, i) for i in (0, 1)]} if "candidates" in out else out
        if msg.get("cmd") == "expand":
            results = []
            for spec in msg["specs"]:
                seat, i, ov = spec["seat"], spec["candidate"], spec["extra_override"]
                cands = self._cands(self.state, seat)
                base = cands[i]
                if set(ov) - set(base.get("factors") or {}):
                    results.append({"error": f"{base['action']} has no factor in {sorted(ov)}"})
                    continue
                c = {**copy.deepcopy(base), "id": len(cands), "base": i,
                     "values": {**base["values"], **ov}, "extra": {"feint": ov.get("feint", False)}}
                self.added[seat].append(c)
                results.append({"id": c["id"], "candidate": c})
            return {"results": results}
        return super().handle(msg)


def _engine():
    t = FactorToy()
    return Engine(InProcess(t)), t


def test_expand_appends_candidates_and_reports_refusals():
    engine, _ = _engine()
    obs = engine.new_match("Toy", "Toy", 0)
    jab = next(c for c in obs.candidates[0] if c.action == "Jab")
    got = engine.expand([{"seat": 0, "candidate": jab.id, "extra_override": {"feint": True}},
                         {"seat": 0, "candidate": 0, "extra_override": {"feint": True}}])
    assert isinstance(got[0], Candidate) and got[0].base == jab.id and got[0].values == {"feint": True}
    assert got[0].id == len(obs.candidates[0]) and got[0].record()["base"] == jab.id
    assert isinstance(got[1], str) and "no factor" in got[1]
    engine.step(got[0].id, 0)  # an expanded id is a normal candidate id for this decision


def test_neighbours_are_asked_only_around_the_support_and_once():
    engine, t = _engine()
    obs = engine.new_match("Toy", "Toy", 0)
    n0 = len(obs.candidates[0])
    jab = next(c.id for c in obs.candidates[0] if c.action == "Jab")
    heavy = next(c.id for c in obs.candidates[0] if c.action == "Heavy")
    row = [0.0] * n0
    row[jab], row[0] = 0.5, 0.5  # Jab and Wait in the support; Wait has no factors
    col = [1.0] + [0.0] * (len(obs.candidates[1]) - 1)
    assert expand_near_support(obs, engine, (row, col)) == 1
    specs = [s for m in t.msgs if m.get("cmd") == "expand" for s in m["specs"]]
    assert specs == [{"seat": 0, "candidate": jab, "extra_override": {"feint": True}}]
    assert expand_near_support(obs, engine, (row, col)) == 0  # already asked
    assert heavy not in [s["candidate"] for s in specs]


def test_search_refines_factors_and_certifies_over_them():
    from yomibench.search import evaluate, exploitability

    engine, _ = _engine()
    bot = MatrixBot(iterations=400, search="oracle")
    obs = engine.new_match("Toy", "Toy", 0)
    refined = 0
    rng = random.Random(0)
    while not obs.done:
        row, col, _ = bot.search_turn(obs, engine)
        cands = turn_candidates(obs, engine)
        assert (len(row), len(col)) == (len(cands[0]), len(cands[1]))
        info = engine.turn_cache["oracle"]
        refined += info.get("refined", 0)
        if info["converged"]:  # certified against every candidate, expanded ones included
            pv = engine.preview([(a.id, b.id) for a in cands[0] for b in cands[1]], candidates=False)
            m = [[evaluate(pv[i * len(cands[1]) + j].state, 0) for j in range(len(cands[1]))]
                 for i in range(len(cands[0]))]
            assert max(exploitability(m, row, col)) <= bot.tolerance + 1e-9
        obs = engine.step(*[bot.choose(obs, seat, engine, rng) for seat in (0, 1)])
    assert refined > 0


def test_records_keep_expanded_candidates():
    engine, _ = _engine()
    lines = play_recorded(engine, MatrixBot(iterations=300, search="oracle"), ("Toy", "Toy"), 3,
                          engine.hello(), max_turns=20)
    turns = lines[1:-1]
    expanded = [c for t in turns for seat in t["candidates"] for c in seat if "base" in c]
    assert expanded and all(c["values"]["feint"] for c in expanded)
    for t in turns:
        assert all(p < len(seat) for p, seat in zip(t["picks"], t["candidates"]))
        if t["strategies"]:
            assert [len(s) for s in t["strategies"]] == [len(s) for s in t["candidates"]]


def test_arena_asks_each_seat_for_its_bots_candidate_set():
    engine, t = _engine()
    a, b = make_bot("random@4"), make_bot("random@2")
    assert (a.name, b.name) == ("random@4", "random@2")
    play_game(engine, (b, a), ("Toy", "Toy"), 1)
    (nm,) = [m for m in t.msgs if m.get("cmd") == "new_match"]
    assert nm["candidate_sets"] == [2, 4]


def test_records_preserve_base_candidate_factor_values():
    engine, _ = _engine()
    obs = engine.new_match("Toy", "Toy", 0)
    jab = next(c for c in obs.candidates[0] if c.action == "Jab")
    assert jab.record()["factors"] == {"feint": True}
    assert jab.record()["values"] == {"feint": False}


def test_refinement_can_change_the_played_action():
    class SmallGame(FactorToy):
        def _cands(self, state, seat):
            allowed = ("Wait", "Jab") if seat == 0 else ("Wait", "Block")
            return [{**c, "id": i} for i, c in enumerate(
                c for c in super()._cands(state, seat) if c["action"] in allowed)]

    t = SmallGame()
    engine = Engine(InProcess(t))
    engine.new_match("Toy", "Toy", 0)
    t.state["players"][1]["x"] = 12  # Jab is in range; Block makes its unrefined version worse.
    obs = engine.observe()
    bot = MatrixBot(iterations=400, search="oracle")
    row, _, _ = bot.search_turn(obs, engine)
    cands = turn_candidates(obs, engine)[0]
    expanded = [i for i, c in enumerate(cands) if c.base is not None]
    assert expanded and sum(row[i] for i in expanded) > 0.99
    chosen = bot.choose(obs, 0, engine, random.Random(0))
    assert next(c for c in cands if c.id == chosen).extra["feint"]


def test_candidates_replay_tool_reproduces_and_catches_differences(tmp_path):
    import gzip
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import candidates_replay

    engine, _ = _engine()  # replay needs only new_match, expand and step
    lines = play_recorded(engine, MatrixBot(iterations=300, search="oracle"), ("Toy", "Toy"), 3,
                          engine.hello(), max_turns=15)
    assert any("base" in c for t in lines[1:-1] for seat in t["candidates"] for c in seat)
    path = tmp_path / "g.jsonl.gz"
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    assert candidates_replay.check(engine, path) is None
    lines[3]["candidates"][1][1]["data"] = {"Distance": 99}  # a list the engine would not produce
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    assert "decision 2" in candidates_replay.check(engine, path)
    for seed in range(3, 40):  # a game that played an expanded candidate
        lines = play_recorded(engine, MatrixBot(iterations=300, search="oracle"), ("Toy", "Toy"), seed,
                              engine.hello(), max_turns=15, opening=(15, 1.0, 1.0))  # uniform picks reach them
        hits = [(t, q) for t in lines[1:-1] for q in (0, 1) if "base" in t["candidates"][q][t["picks"][q]]]
        if hits:
            break
    t, q = hits[0]
    t["candidates"][q][t["picks"][q]]["extra"] = {"feint": "tampered"}  # not what expand rebuilds
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    assert "rebuilt differently" in candidates_replay.check(engine, path)


def test_di_neighbours_only_for_the_most_likely_support_moves():
    class DiToy(FactorToy):
        def _cands(self, state, seat):
            out = super()._cands(state, seat)
            return [{**c, "factors": {"di": 3}, "values": {"di": 0}} if "base" not in c else c for c in out]

    engine = Engine(InProcess(DiToy()))
    obs = engine.new_match("Toy", "Toy", 0)
    n = len(obs.candidates[0])
    row = [0.0] * n
    row[1], row[2], row[3] = 0.5, 0.3, 0.2  # three support moves, most likely first
    col = [1.0] + [0.0] * (len(obs.candidates[1]) - 1)
    t = engine.transport.handler
    expand_near_support(obs, engine, (row, col), di_moves=2)
    specs = [s for m in t.msgs if m.get("cmd") == "expand" for s in m["specs"]]
    assert {s["candidate"] for s in specs if s["seat"] == 0} == {1, 2}  # not the third
    assert {s["candidate"] for s in specs if s["seat"] == 1} == {0}


def test_set4_refinement_keeps_mains_seeded_starts_and_certified_fallback(monkeypatch):
    import yomibench.bots as bots
    from yomibench.search import double_oracle, evaluate, exploitability

    starts = []
    real = bots.double_oracle

    def capped(*a, **k):
        starts.append(k.get("start"))
        r = double_oracle(*a, **k)
        r.converged = False  # force the full-table fallback after refinement
        return r

    monkeypatch.setattr(bots, "double_oracle", capped)
    engine, _ = _engine()
    engine.new_match("Toy", "Toy", 11)
    obs = engine.step(4, 1)
    bot = MatrixBot(iterations=200, search="oracle")
    row, col, _ = bot.search_turn(obs, engine)
    info = engine.turn_cache["oracle"]
    cands = turn_candidates(obs, engine)
    assert info["fallback"]
    pv = engine.preview([(a.id, b.id) for a in cands[0] for b in cands[1]], candidates=False)
    m = [[evaluate(pv[i * len(cands[1]) + j].state, 0) for j in range(len(cands[1]))] for i in range(len(cands[0]))]
    assert max(exploitability(m, row, col)) <= bot.tolerance + 1e-9
    monkeypatch.setattr(bots, "double_oracle", real)
    e2, _ = _engine()
    first = starts[0]
    starts.clear()
    monkeypatch.setattr(bots, "double_oracle", lambda *a, **k: starts.append(k.get("start")) or real(*a, **k))
    MatrixBot(iterations=200, search="oracle").search_turn(e2.new_match("Toy", "Toy", 12), e2)
    assert starts[0] != first  # per-match seed (main's #19), not one start for every match


def test_factor_exploration_is_seeded_recorded_and_off_by_default():
    from yomibench.bots import FactorExplore

    def run(explore, seed=3):
        engine, t = _engine()
        obs = engine.new_match("Toy", "Toy", seed)
        MatrixBot(iterations=300, search="oracle", explore=explore).search_turn(obs, engine)
        specs = [s for m in t.msgs if m.get("cmd") == "expand" for s in m["specs"]]
        return engine.turn_cache["oracle"], specs

    info, specs = run(None)
    assert "explore" not in info
    info2, specs2 = run(FactorExplore(margin=10.0, random_k=1, all_prob=0.0))
    assert info2["explore"]["margin"][0] > 0 and len(specs2) > len(specs)  # off-support candidates refined
    assert run(FactorExplore(margin=10.0, random_k=1, all_prob=0.0)) == (info2, specs2)  # seeded
    info3, _ = run(FactorExplore(margin=None, random_k=0, all_prob=1.0))
    assert info3["explore"]["all"] is True and "gap" in info3["explore"]


def test_softmax_decision_exploration_marks_one_seat():
    engine, _ = _engine()
    lines = play_recorded(engine, MatrixBot(iterations=300, search="oracle"), ("Toy", "Toy"), 2,
                          engine.hello(), max_turns=12, explore_cfg=(1.0, 0.05))
    assert lines[0]["explore"] == {"epsilon": 1.0, "temperature": 0.05}
    searched = [ln for ln in lines[1:-1] if ln["strategies"] is not None]
    assert searched and all(ln.get("explore_mode") == "softmax" and sum(ln["explore"]) == 1 for ln in searched)
    plain = play_recorded(engine, MatrixBot(iterations=300, search="oracle"), ("Toy", "Toy"), 2, engine.hello(),
                          max_turns=12)
    assert not any("explore" in ln for ln in plain[1:-1])


def _two_bot_game(first_searcher, seed=3, turns=25):
    """A seat-swapped-style game between two different bots on FactorToy, with `first_searcher`
    (0 or 1) running its search first each turn. Returns the picked ids per turn."""
    engine, _ = _engine()
    bots = (MatrixBot(iterations=300, search="oracle"), MatrixBot(iterations=301, search="oracle"))
    obs = engine.new_match("Toy", "Toy", seed)
    rngs = (random.Random(f"{seed}:0"), random.Random(f"{seed}:1"))
    picks = []
    for _ in range(turns):
        if obs.done:
            break
        order = (first_searcher, 1 - first_searcher)
        for seat in order:
            bots[seat].search_turn(obs, engine)
        ids = [bots[seat].choose(obs, seat, engine, rngs[seat]) for seat in (0, 1)]
        picks.append(ids)
        obs = engine.step(*ids)
    return picks


def test_a_two_bot_game_does_not_depend_on_which_seat_searches_first():
    for seed in range(4):
        assert _two_bot_game(0, seed) == _two_bot_game(1, seed)


def test_each_bot_sees_only_its_own_expansions_and_shares_their_results():
    engine, t = _engine()
    obs = engine.new_match("Toy", "Toy", 0)
    a, b = MatrixBot(iterations=300, search="oracle"), MatrixBot(iterations=301, search="oracle")
    a.search_turn(obs, engine)
    view_a = [list(c) for c in turn_candidates(obs, engine, a._key())]
    b.search_turn(obs, engine)
    assert turn_candidates(obs, engine, a._key()) == view_a  # b's search never reaches a's lists
    base = [len(c) for c in obs.candidates]
    for view in (a._key(), b._key()):  # each view: the engine's base lists plus its own additions
        lists = turn_candidates(obs, engine, view)
        assert [c.id for c in lists[0][:base[0]]] == [c.id for c in obs.candidates[0]]
        assert all(c in turn_candidates(obs, engine)[s] for s in (0, 1) for c in lists[s])
    specs = [(s["seat"], s["candidate"], tuple(sorted(s["extra_override"].items())))
             for m in t.msgs if m.get("cmd") == "expand" for s in m["specs"]]
    assert len(specs) == len(set(specs))  # a spec both bots wanted was built once


def test_a_lone_bot_searches_the_full_lists_as_before():
    engine, _ = _engine()
    bot = MatrixBot(iterations=300, search="oracle")
    obs = engine.new_match("Toy", "Toy", 1)
    for _ in range(15):
        if obs.done:
            break
        bot.search_turn(obs, engine)
        mine, full = turn_candidates(obs, engine, bot._key()), turn_candidates(obs, engine)
        assert [[c.id for c in s] for s in mine] == [[c.id for c in s] for s in full]
        rng = random.Random(0)
        obs = engine.step(bot.choose(obs, 0, engine, rng), bot.choose(obs, 1, engine, rng))


def _seat_preflight(monkeypatch, decisions=10):
    """pool_eval's seat preflight on FactorToy, between two copies of one bot (distinct value
    objects, as two loads of one net are)."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("pool_eval", Path(__file__).parents[1] / "tools/pool_eval.py")
    pool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pool)
    monkeypatch.setattr(pool, "connect", lambda spec, candidate_set: _engine()[0])
    monkeypatch.setattr(pool, "make_bot", lambda spec: MatrixBot(iterations=300, search="oracle",
                                                                 value=lambda state, seat: evaluate(state, seat)))
    return pool, lambda: pool.seat_preflight("copy", "toy", 4, ["Toy"], 3, decisions)


def test_the_gate_seat_preflight_passes_when_each_bot_searches_its_own_view(monkeypatch):
    _, run = _seat_preflight(monkeypatch)
    assert run() == 10


def test_the_gate_seat_preflight_fails_when_bots_share_their_expansions(monkeypatch):
    import pytest

    from yomibench import bots

    shared = bots.turn_candidates  # the pre-#54 search: one candidate list per seat for both bots
    monkeypatch.setattr(bots, "turn_candidates", lambda obs, engine, view=None: shared(obs, engine))
    pool, run = _seat_preflight(monkeypatch)
    with pytest.raises(pool.SeatPreflightFailed, match="searches first"):
        run()


def test_list_factors_ask_the_engines_child_values_as_given():
    """Candidate set 5 offers `aim` and `di_fine` as lists of child values around a candidate (not a
    count of indices): refinement asks exactly those values, and a candidate without the factor in
    its values defaults to 0."""
    class ListToy(FactorToy):
        def _cands(self, state, seat):
            out = super()._cands(state, seat)
            return [{**c, "factors": {"aim": [-15, 15], "di_fine": [3, 7]}, "values": {}} if "base" not in c else c
                    for c in out]

        def handle(self, msg):
            if msg.get("cmd") == "expand":
                results = []
                for spec in msg["specs"]:
                    seat, i, ov = spec["seat"], spec["candidate"], spec["extra_override"]
                    cands = self._cands(self.state, seat)
                    c = {**copy.deepcopy(cands[i]), "id": len(cands), "base": i, "values": dict(ov)}
                    self.added[seat].append(c)
                    results.append({"id": c["id"], "candidate": c})
                self.msgs.append(msg)
                return {"results": results}
            return super().handle(msg)

    engine = Engine(InProcess(ListToy()))
    obs = engine.new_match("Toy", "Toy", 0)
    row = [0.0] * len(obs.candidates[0])
    row[1] = 1.0
    col = [0.0] * len(obs.candidates[1])
    col[2] = 1.0
    t = engine.transport.handler
    added = expand_near_support(obs, engine, (row, col))
    specs = [s for m in t.msgs if m.get("cmd") == "expand" for s in m["specs"]]
    seat0 = sorted((tuple(sorted(s["extra_override"].items()))) for s in specs if s["seat"] == 0)
    assert seat0 == sorted([
        (("aim", -15), ("di_fine", 0)), (("aim", 15), ("di_fine", 0)),
        (("aim", 0), ("di_fine", 3)), (("aim", 0), ("di_fine", 7))])
    assert added == 8  # 4 per seat


def test_count_and_boolean_factors_are_unchanged():
    from yomibench.bots import _factor_values
    assert _factor_values("di", 3) == [0, 1, 2]
    assert _factor_values("extra_set", 2) == [0, 1]
    assert _factor_values("feint", True) == [False, True]
    assert _factor_values("reverse", False) == [False]
    assert _factor_values("aim", [-15, 15]) == [-15, 15]
