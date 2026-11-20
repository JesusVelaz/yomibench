"""tools/explain_game.py: decision explanations from recorded or re-searched values."""

import gzip
import json
import sys
from pathlib import Path

from yomibench.arena import run_round_robin
from yomibench.bots import MatrixBot, make_bot
from yomibench.engine import connect
from yomibench.selfplay import play_recorded, write_record

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explain_game as eg  # noqa: E402


def _cand(action, di=0):
    return {"action": action, "data": None, "extra": {"DI": {"x": di, "y": 0}}}


def _decision(values, strategy, actions, pick, state="Wait", **kw):
    cands = [_cand(a, di) for a, di in actions]
    player = {"character": "Ninja", "hp": 1000, "max_hp": 1500, "x": 0, "state": state}
    return {"turn": 7, "state": {"tick": 70, "players": [player, {**player, "x": 120}]}, "actionable": [True, False],
            "candidates": [cands, [_cand("Continue")]], "picks": [pick, 0], **kw}, \
        {"strategy": strategy, "opponent": [1.0], "values": values, "value": 0.1}


def _seat(values, strategy, actions, pick, **kw):
    d, search = _decision(values, strategy, actions, pick, **kw)
    return eg.explain_seat(d, 0, search, 1200.0)


def test_a_flat_tie_compares_the_best_other_move_outside_the_mix_not_the_mix_or_its_variants():
    acts = [("Wait", 0), ("Taunt", 0), ("Wait", 50), ("Jump", 0)]
    tie = _seat([0.1, 0.1, 0.1, 0.097], [0.5, 0.5, 0.0, 0.0], acts, 0)
    assert tie["flags"]["flat_tie"] and not tie["flags"]["confident"] and tie["category"] == "flat_tie"
    assert tie["best_outside"]["label"] == "Jump" and tie["variant_gap"] == 0.0  # Wait DI(50,0) is a variant
    assert tie["support"] == 2 and tie["waste"] == "wait" and tie["flags"]["in_mix"]
    assert tie["best_outside"]["gap_hp"] == 3.6 and tie["reading"].startswith("near-tie")
    far = _seat([0.1, 0.1, 0.1, 0.07], [0.5, 0.5, 0.0, 0.0], acts, 1)
    assert far["flags"]["confident"] and not far["flags"]["flat_tie"] and far["waste"] == "taunt"
    assert far["category"] == "confident" and far["reading"].startswith("confident mix")
    middle = _seat([0.1, 0.05, 0.05, 0.09], [1.0, 0.0, 0.0, 0.0], acts, 0)
    assert middle["category"] == "moderate" and middle["top"][0]["hp"] == 0.0 and middle["top"][1]["hp"] == -12.0


def test_quick_explored_and_idle_continue_marks():
    acts = [("Continue", 0), ("Jump", 0)]
    q = _seat([0.1, 0.0], [1.0, 0.0], acts, 0, search="cheap")
    assert q["flags"]["search"] == "quick" and q["category"] == "quick" and q["waste"] == "idle_continue"
    e = _seat([0.1, 0.0], [1.0, 0.0], acts, 1, explore=[True, False], explore_mode="softmax")
    assert e["flags"]["explored"] and e["flags"]["explore_mode"] == "softmax" and not e["flags"]["in_mix"]
    assert e["category"] == "explored" and e["pick"]["hp"] == -120.0
    busy = _seat([0.1, 0.0], [1.0, 0.0], acts, 0, state="DashForward")
    assert busy["waste"] is None


def test_hp_per_value_comes_from_the_nets_pricing():
    pricing = {"nets": {"g6": "abc"}, "results": {"g6": {"hp+100": {"mean": 0.08}}}}
    assert eg.hp_per_value("abc", pricing)[0] == 1250.0
    assert eg.hp_per_value("other", pricing)[0] == eg.DEFAULT_HP_PER_VALUE


def test_a_self_play_record_is_explained_from_its_recorded_search(tmp_path):
    engine = connect("toy")
    lines = play_recorded(engine, MatrixBot(search="oracle"), ("Toy", "Toy"), 3, engine.hello(), max_turns=25)
    path = write_record(tmp_path, 3, lines)
    e = eg.explain(path, net_dirs=[])
    assert e["kind"] == "selfplay" and e["values"] == "recorded" and e["decisions"]
    for d in e["decisions"]:
        for x in d["seats"]:
            if x:
                assert x["flags"]["in_mix"] and x["pick"]["p"] > 0 and x["top"]
                assert any(t["picked"] for t in x["top"])


def _strip(path, out):
    """An arena record as written before arena recorded searches."""
    lines = [json.loads(x) for x in gzip.open(path, "rt")]
    for d in lines[1:]:
        if "picks" in d:
            for k in ("strategies", "opponent_strategies", "candidate_values", "seat_values"):
                d.pop(k, None)
            d.update(strategies=None, value=None, candidate_values=None)
    with gzip.open(out, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)
    return out


def test_an_old_gate_record_is_re_searched_to_the_values_the_game_recorded(tmp_path):
    rec = tmp_path / "rec"
    run_round_robin(lambda: connect("toy"), [make_bot("matrix-do"), make_bot("matrix")], ["Toy"], 1, 5,
                    max_turns=30, record_dir=rec)
    for path in sorted(rec.glob("g*.jsonl.gz")):
        recorded = eg.explain(path, net_dirs=[])
        assert recorded["values"] == "recorded" and recorded["decisions"]
        old = _strip(path, tmp_path / path.name)
        redone = eg.explain(old, net_dirs=[], make_engine=lambda bots: connect("toy"))
        rc = redone["recompute"]
        assert redone["values"] == "recomputed" and rc["picks_reproduced"] == rc["seat_decisions"] > 0
        assert rc["states_checked"] == recorded["decisions_total"]
        for a, b in zip(recorded["decisions"], redone["decisions"], strict=True):
            for s in (0, 1):
                if b["seats"][s]:
                    assert b["seats"][s].pop("reproduced") is True
            assert a == b


def test_main_writes_one_file_per_record_and_skips_finished_ones(tmp_path, capsys):
    engine = connect("toy")
    path = write_record(tmp_path, 4, play_recorded(engine, MatrixBot(), ("Toy", "Toy"), 4, engine.hello(),
                                                   max_turns=10))
    out = tmp_path / "out"
    assert eg.main([str(path), "--out", str(out), "--net-dir", str(tmp_path)]) == 0
    (written,) = out.glob("*.explain.json")
    assert json.loads(written.read_text())["record"] == path.name
    assert eg.main([str(path), "--out", str(out), "--net-dir", str(tmp_path)]) == 0
    assert "exists" in capsys.readouterr().out
