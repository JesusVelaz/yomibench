"""train --window-budget-games: a recency-weighted sample of the training games, the same from both loaders."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yomibench import cache, train  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.selfplay import run_selfplay  # noqa: E402

HOUR = 3600.0
BASE = 1.8e9


def _keys(cset="5", ceiling=800):
    game = json.dumps({"game_version": "v", "state_version": 3, "features_version": 1,
                       "match_settings": {"ceiling_height": ceiling}}, sort_keys=True)
    return game, cset


def _synthetic(n_train=20, n_val=4):
    """Games g00..: game i finished at hour i (validation games interleaved among the newest)."""
    times, keys = {}, {}
    train_files = [f"g{i:02d}" for i in range(n_train)]
    val_files = [f"v{i:02d}" for i in range(n_val)]
    for i, p in enumerate(train_files):
        times[p] = BASE + i * HOUR
        keys[p] = _keys("6" if i % 2 else "5", 500 if i % 5 == 0 else 800)
    for i, p in enumerate(val_files):
        times[p] = BASE + (n_train - n_val + i) * HOUR + 0.5 * HOUR
        keys[p] = _keys()
    return {"train": train_files, "val": val_files, "eval": []}, times, keys


def _cut(budget, hours, seed, **kw):
    chunks, times, keys = _synthetic(**kw)
    logs = []
    s = train.sample_window(chunks, times.get, keys.get, hours, budget, seed, logs.append)
    return chunks, s, logs, times


def test_recent_games_kept_in_full_older_sampled_to_the_budget_deterministically():
    chunks, s, logs, times = _cut(budget=10, hours=5.5, seed=1)
    newest = BASE + 19.5 * HOUR  # the newest validation game sets "now"
    recent = [f"g{i:02d}" for i in range(20) if times[f"g{i:02d}"] >= newest - 5.5 * HOUR]
    assert recent == ["g14", "g15", "g16", "g17", "g18", "g19"]
    assert set(recent) <= set(chunks["train"]) and len(chunks["train"]) == 10
    assert chunks["train"] == sorted(chunks["train"])  # the split's own order
    assert (s["recent_kept"], s["older"], s["older_sampled"], s["kept"]) == (6, 14, 4, 10)
    assert chunks["val"] == ["v00", "v01", "v02", "v03"] and s["val_games"] == 4 and not s["val_sampled"]
    again, _, _, _ = _cut(budget=10, hours=5.5, seed=1)
    assert again["train"] == chunks["train"]  # deterministic per seed
    others = {tuple(_cut(budget=10, hours=5.5, seed=k)[0]["train"]) for k in range(2, 8)}
    assert len(others | {tuple(chunks["train"])}) > 1  # and the seed matters
    # counts, logged and returned
    assert sum(s["kept_by_candidate_set"].values()) == 10 and sum(s["kept_by_ceiling"].values()) == 10
    (line,) = logs
    assert "kept 6 recent + 4 older sampled (of 14 older) = 10 of 20 training games" in line
    assert "kept by candidate set: " in line and "kept by ceiling: " in line
    assert "validation kept complete (4 games)" in line


def test_a_budget_smaller_than_the_recent_games_keeps_the_most_recent():
    chunks, s, logs, _ = _cut(budget=3, hours=5.5, seed=1)
    assert chunks["train"] == ["g17", "g18", "g19"]
    assert (s["recent"], s["recent_kept"], s["older_sampled"], s["kept"], s["recent_over_budget"]) == (6, 3, 0, 3, True)
    assert "the 6 recent games exceed the budget: kept the most recent 3" in logs[0]


def test_a_budget_above_the_window_keeps_everything():
    chunks, s, _, _ = _cut(budget=100, hours=2, seed=0)
    assert chunks["train"] == [f"g{i:02d}" for i in range(20)] and s["kept"] == 20


def _records(d, seeds):
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], seeds, d)
    for i, p in enumerate(sorted(d.glob("g*.jsonl.gz"))):  # game i finished at hour i
        os.utime(p, (BASE + i * HOUR, BASE + i * HOUR))


@pytest.fixture
def capture(monkeypatch):
    """The training and validation file lists each loader ends up with."""
    seen = []
    real = train.sample_window

    def spy(chunks, *a, **kw):
        out = real(chunks, *a, **kw)
        seen.append({k: list(v) for k, v in chunks.items()})
        return out
    monkeypatch.setattr(train, "sample_window", spy)
    return seen


def test_both_loaders_select_the_same_games_without_moving_any_between_splits(tmp_path, capture):
    d = tmp_path / "g"
    _records(d, range(16))
    window = {"recent_hours": 4.0, "budget": 7, "seed": 3}
    full = train.load_samples([d], train.Vocab(), 0.25, 0, 0.5, workers=1)
    direct = train.load_samples([d], train.Vocab(), 0.25, 0, 0.5, workers=1, window=window)
    cached = cache.open_cached([d], train.Vocab(), 0.25, 0, 0.5, workers=1, window=window)
    mem_files, cache_files = capture
    assert mem_files == cache_files  # the same games from both paths
    splits, _, _ = train.select_files([d], 0.25, 0, None, [], None, 0)
    assert set(mem_files["train"]) < set(splits["train"]) and len(mem_files["train"]) == 7
    assert mem_files["val"] == splits["val"]  # no game changed split; validation complete
    assert direct["sample_window"] == cached["sample_window"]
    assert direct["sample_window"]["kept"] == 7 and direct["sample_window"]["time"] == "source file mtime"
    newest = max(os.stat(p).st_mtime for p in [*splits["train"], *splits["val"]])
    recent = {p for p in splits["train"] if os.stat(p).st_mtime >= newest - 4 * HOUR}
    assert recent and recent <= set(mem_files["train"])
    assert direct["train"]["games"] == cached["train"]["games"] == 7
    np.testing.assert_array_equal(direct["train"]["outcome"], cached["train"]["outcome"])
    np.testing.assert_array_equal(direct["train"]["y"].cpu() if hasattr(direct["train"]["y"], "cpu")
                                  else direct["train"]["y"], np.asarray(cached["train"]["y"]))
    for split in ("val",):  # validation is the unwindowed run's, row for row
        assert direct[split]["games"] == full[split]["games"] == cached[split]["games"]
        np.testing.assert_array_equal(direct[split]["outcome"], full[split]["outcome"])
        np.testing.assert_array_equal(cached[split]["outcome"], full[split]["outcome"])
    assert direct["heldout"][str(d)]["sha256"] == full["heldout"][str(d)]["sha256"]


def _train(d, out, *extra, ledger=None):
    train.main([str(d), "--out", str(out), "--epochs", "2", "--hidden", "8", "--workers", "1", "--value-weight", "0.5",
                *(["--ledger", str(ledger)] if ledger else ["--no-ledger"]), *extra])
    z = np.load(out, allow_pickle=False)
    return {k: z[k] for k in z.files if k != "meta"}, json.loads(str(z["meta"]))


def test_counts_reach_the_log_the_meta_and_the_ledger(tmp_path, capsys):
    d = tmp_path / "g"
    _records(d, range(12))
    led = tmp_path / "ledger.jsonl"
    _, meta = _train(d, tmp_path / "m.npz", "--cache", str(tmp_path / "c"), "--window-budget-games", "5",
                     "--window-recent-hours", "3", "--window-seed", "2", ledger=led)
    s = meta["sample_window"]
    assert (s["budget_games"], s["recent_hours"], s["seed"], s["kept"]) == (5, 3.0, 2, 5)
    assert meta["train_games"] <= 5 and s["kept_by_candidate_set"] and s["kept_by_ceiling"]
    assert "sample window (recent 3h, budget 5 training games, seed 2" in capsys.readouterr().out
    (row,) = [json.loads(x) for x in led.read_text().splitlines()]
    assert row["metrics"]["sample_window"] == s
    assert row["provenance"]["settings"]["window_budget_games"] == 5
    assert row["provenance"]["counts"]["excluded_by_reason"]["sample_window"] == s["candidates"] - 5


def test_off_is_unchanged_and_a_budget_above_the_window_trains_the_same_model(tmp_path):
    d = tmp_path / "g"
    _records(d, range(10))
    led = tmp_path / "ledger.jsonl"
    off, meta_off = _train(d, tmp_path / "off.npz", "--cache", str(tmp_path / "c"), ledger=led)
    on, meta_on = _train(d, tmp_path / "on.npz", "--cache", str(tmp_path / "c"), "--window-budget-games", "1000",
                         "--window-recent-hours", "1")
    assert "sample_window" not in meta_off and meta_on["sample_window"]["kept"] == meta_on["sample_window"]["candidates"]
    assert off.keys() == on.keys()
    for k in off:
        np.testing.assert_array_equal(off[k], on[k], err_msg=k)
    (row,) = [json.loads(x) for x in led.read_text().splitlines()]  # off: no trace in the ledger row
    assert "sample_window" not in row["metrics"]
    assert not any(k.startswith("window_") for k in row["provenance"]["settings"])
    with pytest.raises(SystemExit):
        train.main([str(d), "--out", str(tmp_path / "x.npz"), "--no-ledger", "--window-recent-hours", "12"])
