import importlib.util
from pathlib import Path

import pytest

from yomibench.engine import Candidate, EngineError, Observation

spec = importlib.util.spec_from_file_location("render_record", Path(__file__).parents[1] / "tools/render_record.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)


def cand(i, action, data=None, base=None, values=None):
    return Candidate(i, action, data, {}, {"di": 17} if base is None else None, values, base)


def rec(c):
    return c.record()


class FakeEngine:
    """Plays a fixed script of states; records the commands it gets."""

    def __init__(self, states, rules="r1", expand_result=None):
        self.states, self.rules, self.expand_result = states, rules, expand_result
        self.k, self.steps, self.expands, self.settings = 0, [], [], []

    def obs(self):
        done = self.k == len(self.states) - 1
        st = self.states[self.k]
        cands = (cand(0, "Wait"), cand(1, "Fireball", {"x": 45, "y": 0}))
        return Observation(self.k, done, 1 if done else None, st, (cands, cands))

    def hello(self):
        return {"candidate_rules": self.rules}

    def new_match(self, p1, p2, seed, max_candidates=None, settings=None):
        if settings and "no_such_setting" in settings:
            raise EngineError("bad_request", "unknown setting no_such_setting")
        self.settings.append(settings)
        self.k = 0
        return self.obs()

    def expand(self, specs):
        self.expands.append(specs)
        return [self.expand_result]

    def step(self, a, b):
        self.steps.append((a, b))
        self.k += 1
        return self.obs()


def state(hp1, hp2, x=0):
    return {"tick": 0, "players": [{"hp": hp1, "x": x}, {"hp": hp2, "x": 0}]}


def record_for(states, picks):
    base = [rec(cand(0, "Wait")), rec(cand(1, "Fireball", {"x": 45, "y": 0}))]
    head = {"characters": ["Ninja", "Cowboy"], "seed": 7, "engine": {"candidate_rules": "r1", "max_candidates": 0}}
    turns = [{"turn": k, "state": s, "candidates": [list(base), list(base)], "picks": list(p)}
             for k, (s, p) in enumerate(zip(states, picks))]
    result = {"done": True, "winner": 1, "turns": len(states) - 1,
              "hp": [p["hp"] for p in states[-1]["players"]]}
    return head, turns, result


def test_state_diff_names_fields_and_ignores_engine_only_keys():
    want = {"tick": 3, "players": [{"hp": 10, "x": 1}], "objects": []}
    have = {"tick": 3, "players": [{"hp": 9, "x": 1, "di": {"x": 0}}], "objects": [], "extra": 1}
    assert rr.state_diff(want, have) == ["state.players[0].hp: 10 vs 9"]
    assert rr.state_diff(want, {"tick": 3, "objects": []}) == ["state.players: missing from the engine's state"]
    assert rr.state_diff({"objects": [1, 2]}, {"objects": [1]}) == ["state.objects: 2 items vs 1"]
    assert rr.state_diff(want, want) == []


def test_replay_plays_recorded_picks_stepwise():
    states = [state(10, 10), state(5, 10), state(0, 10)]
    head, turns, result = record_for(states[:2] + [states[2]], [(0, 1), (1, 0), (0, 0)])
    eng = FakeEngine(states)
    rr.replay(eng, head, turns[:2], result)
    assert eng.steps == [(0, 1), (1, 0)]


def test_replay_stops_on_first_state_mismatch():
    states = [state(10, 10), state(5, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 1), (1, 0), (0, 0)])
    turns[1]["state"] = state(5, 10, x=4)  # the record says p1 moved; the engine disagrees
    eng = FakeEngine(states)
    with pytest.raises(rr.Divergence, match=r"decision 1 \(turn 1\): state differs:\n  state.players\[0\].x: 4 vs 0"):
        rr.replay(eng, head, turns[:2], result)
    assert eng.steps == [(0, 1)]


def test_replay_checks_rules_and_result():
    states = [state(10, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 1), (0, 0)])
    with pytest.raises(rr.Divergence, match="candidate_rules differ"):
        rr.replay(FakeEngine(states, rules="other"), head, turns[:1], result)
    result["hp"] = [0, 9]
    with pytest.raises(rr.Divergence, match="result differs"):
        rr.replay(FakeEngine(states), head, turns[:1], result)


def test_expanded_pick_is_rebuilt_with_expand():
    aimed = cand(2, "Fireball", {"x": 45, "y": 0}, base=1, values={"di": 3})
    decision = {"candidates": [[rec(cand(0, "Wait")), rec(cand(1, "Fireball")), rec(aimed)],
                               [rec(cand(0, "Wait"))]], "picks": [2, 0]}
    eng = FakeEngine([state(1, 1)], expand_result=Candidate(5, "Fireball", {"x": 45, "y": 0}, {}, None, {"di": 3}, 1))
    assert rr.pick_ids(eng, decision) == [5, 0]
    assert eng.expands == [[{"seat": 0, "candidate": 1, "extra_override": {"di": 3}}]]
    eng.expand_result = "no such factor"
    with pytest.raises(rr.Divergence, match="expand refused"):
        rr.pick_ids(eng, decision)


def test_a_playout_capped_decision_replays_its_picks_by_spec():
    states = [state(10, 10), state(5, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 1), (0, 1), (0, 0)])
    # Turn 1 was cheap: seat 0 searched only the Fireball (position 0 of its capped list, engine id 1).
    turns[1]["search"] = "cheap"
    turns[1]["candidates"][0] = [rec(cand(1, "Fireball", {"x": 45, "y": 0}))]
    eng = FakeEngine(states)
    rr.replay(eng, head, turns[:2], result)
    assert eng.steps == [(0, 1), (1, 1)]


def test_a_capped_list_must_be_the_engines_with_candidates_left_out():
    states = [state(10, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 0), (0, 0)])
    turns[0]["search"] = "cheap"
    turns[0]["candidates"][0] = [rec(cand(0, "Fireball", {"x": 45, "y": 0})), rec(cand(1, "Wait"))]  # out of order
    with pytest.raises(rr.Divergence, match=r"seat 0: capped candidate 1 of 2 isn't in the engine's 2"):
        rr.replay(FakeEngine(states), head, turns[:1], result)
    turns[0]["candidates"][0] = [rec(cand(0, "Teleport"))]
    with pytest.raises(rr.Divergence, match="capped candidate 0 of 1"):
        rr.replay(FakeEngine(states), head, turns[:1], result)


def test_an_uncapped_decision_still_needs_the_engines_exact_lists():
    states = [state(10, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 0), (0, 0)])
    turns[0]["candidates"][0] = [rec(cand(0, "Wait"))]  # a subset, but the turn isn't marked cheap
    with pytest.raises(rr.Divergence, match="seat 0: candidates differ"):
        rr.replay(FakeEngine(states), head, turns[:1], result)
    turns[0]["search"] = "full"
    with pytest.raises(rr.Divergence, match="seat 0: candidates differ"):
        rr.replay(FakeEngine(states), head, turns[:1], result)


def test_a_playout_capped_selfplay_record_replays_on_the_engine(tmp_path, monkeypatch):
    from yomibench import selfplay
    from yomibench.bots import MatrixBot
    from yomibench.engine import connect

    # Capped lists that keep the last candidates, so positions in them are never engine ids.
    monkeypatch.setattr(selfplay, "cap_candidates", lambda cands, n, *_: list(cands)[-n:])
    engine = connect("toy")
    lines = selfplay.play_recorded(engine, MatrixBot(iterations=30, search="oracle"), ("Toy", "Toy"), 3,
                                   engine.hello(), max_turns=40, playout_cap=(0.5, 3))
    path = selfplay.write_record(tmp_path, 3, lines)
    head, turns, result = rr.read_record(path)
    cheap = [t for t in turns if t["search"] == "cheap"]
    assert cheap and any(len(c) < 9 for t in cheap for c in t["candidates"])
    obs = rr.replay(connect("toy"), head, turns, result)
    assert obs.turn == result["turns"]


def test_replay_starts_the_match_on_the_records_settings():
    states = [state(10, 10), state(0, 10)]
    head, turns, result = record_for(states, [(0, 1), (0, 0)])
    eng = FakeEngine(states)
    rr.replay(eng, head, turns[:1], result)
    assert eng.settings == [None]  # no settings in the record: the server's defaults
    head["engine"]["match_settings"] = {"stage_width": 600, "game_length": 3000, "starting_meter": "1.5"}
    eng = FakeEngine(states)
    rr.replay(eng, head, turns[:1], result)
    assert eng.settings == [{"stage_width": 600, "game_length": 3000, "starting_meter": "1.5"}]
    head["engine"]["match_settings"] = {"no_such_setting": 1}
    with pytest.raises(rr.Divergence, match="refused the record's match settings"):
        rr.replay(FakeEngine(states), head, turns[:1], result)


def test_docker_engine_serves_the_records_rules_on_one_cpu():
    cmd = rr.docker_command("7a624558f81b", Path("/game"), "yomibench-engine", 7790, "r")
    assert cmd[-2:] == ["--candidate-rules", "7a624558f81b"]
    assert "--cpus=1" in cmd and "127.0.0.1:7790:7777" in cmd
    i = cmd.index("yomibench-engine")
    assert cmd[cmd.index("--entrypoint") + 1] == "nice" and cmd[i + 1:i + 3] == ["-n", "19"]


def test_only_nonstandard_match_settings_are_asked_for():
    """Records carry the full settings dict even for standard games: only the settings that differ from the
    engine's standard ones are passed, so a standard record replays on an engine that plays only those (a Batch)."""
    std = {"stage_width": 800, "game_length": 3600, "starting_meter": "0", "turn_time": 30}
    hello = {"match_settings": std}
    assert rr._nonstandard(None, hello) is None and rr._nonstandard({}, hello) is None
    assert rr._nonstandard(dict(std), hello) is None
    assert rr._nonstandard({**std, "stage_width": 600, "starting_meter": "1.5"}, hello) == {"stage_width": 600,
                                                                                           "starting_meter": "1.5"}
    # an engine already set to other settings still reports its standard ones as default_match_settings
    assert rr._nonstandard(dict(std), {"match_settings": {**std, "stage_width": 600}, "default_match_settings": std}) is None
    assert rr._nonstandard({**std, "ceiling_height": 500}, hello) == {"ceiling_height": 500}  # unknown to the engine


def test_a_set_the_game_cannot_serve_replays_on_exact_inputs():
    """A set the game does not serve (e.g. a future set 7): no candidate check, the picks' own inputs through step_raw, state still
    checked at every decision."""
    calls = []

    class RawEngine:
        def hello(self):
            return {"candidate_rules": "game-default"}

        def new_match(self, *a, **k):
            return Observation(0, False, None, {"tick": 0, "players": []}, ((), ()))

        def step_raw(self, inputs):
            calls.append(inputs)
            return Observation(1, True, 0, {"tick": 10, "players": []}, ((), ()))

    head = {"characters": ["Cowboy", "Cowboy"], "seed": 1, "engine": {"candidate_set": "7", "candidate_rules": "x"}}
    c = {"action": "Shoot", "data": {"a": 1}, "extra": {"DI": {"x": 0, "y": 0}}}
    turns = [{"turn": 0, "state": {"tick": 0, "players": []}, "candidates": [[c], [dict(c, action="Grab")]], "picks": [0, 0]}]
    rr.replay(RawEngine(), head, turns, {"done": True, "winner": 0, "turns": 1, "hp": []})
    assert calls == [[c, dict(c, action="Grab")]]
    assert rr.exact_inputs(head) and not rr.exact_inputs({"engine": {"candidate_set": "5"}}) and not rr.exact_inputs({})
