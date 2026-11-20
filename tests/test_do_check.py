"""tools/do_check.py's replay mode: picking decisions and replaying records' picks."""

import sys
from pathlib import Path

from yomibench.bots import MatrixBot
from yomibench.engine import connect
from yomibench.selfplay import read_record, run_selfplay

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import do_check  # noqa: E402


def _records(tmp_path, monkeypatch):
    from yomibench import selfplay

    monkeypatch.setattr(selfplay, "cap_candidates", lambda cands, n, *_: list(cands)[::-1][:n])  # positions != ids
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30, search="oracle"), [("Toy", "Toy")],
                 range(6), tmp_path, playout_cap=(0.5, 3))
    return sorted(tmp_path.glob("g*.jsonl.gz"))


def test_picks_of_playout_capped_records_replay_by_identity(tmp_path, monkeypatch):
    for path in _records(tmp_path, monkeypatch):
        lines = read_record(path)
        head, turns, result = lines[0], lines[1:-1], lines[-1]["result"]
        assert any(t.get("search") == "cheap" for t in turns)
        engine = connect("toy")
        obs = engine.new_match(*head["characters"], head["seed"])
        for t in turns:
            assert obs.state == t["state"]
            obs = engine.step(*do_check._pick_ids(engine, obs, t))
        assert (obs.turn, obs.winner, [p["hp"] for p in obs.state["players"]]) == \
            (result["turns"], result["winner"], result["hp"])


def test_sampling_takes_full_searched_turns_only(tmp_path, monkeypatch):
    paths = _records(tmp_path, monkeypatch)
    chosen = do_check._sample(paths, 10, 2, seed=3)
    assert 0 < sum(len(v) for v in chosen.values()) <= 10 and all(len(v) <= 2 for v in chosen.values())
    for path, ks in chosen.items():
        turns = read_record(path)[1:-1]
        assert all(turns[k]["strategies"] and turns[k].get("search", "full") == "full" for k in ks)
    assert chosen == do_check._sample(paths, 10, 2, seed=3)  # seeded
