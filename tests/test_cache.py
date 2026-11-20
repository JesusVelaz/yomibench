"""yomibench.cache: incremental builds, invalidation, and training equal to the in-memory path."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from yomibench import cache, train  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.selfplay import run_selfplay  # noqa: E402


def _records(tmp_path, seeds, explore=False):
    """Toy self-play records; with `explore`, opening noise marks early picks exploratory."""
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], seeds, tmp_path,
                 opening=(3, 1.0, 0.5) if explore else None)


def test_builds_incrementally_and_rebuilds_when_the_encoder_changes(tmp_path, monkeypatch):
    _records(tmp_path, range(4))
    cdir = cache.build(tmp_path, workers=1)
    m = json.loads((cdir / "manifest.json").read_text())
    assert len(m["files"]) == 4 and len(m["shards"]) == 1
    assert np.load(cdir / "s00000" / "nums0.npy", mmap_mode="r").dtype == np.float16
    assert np.load(cdir / "s00000" / "cats0.npy").dtype == np.uint8
    cache.build(tmp_path, workers=1)  # nothing new: untouched
    assert json.loads((cdir / "manifest.json").read_text()) == m
    _records(tmp_path, range(4, 6))
    cache.build(tmp_path, workers=1)
    m2 = json.loads((cdir / "manifest.json").read_text())
    assert len(m2["files"]) == 6 and len(m2["shards"]) == 2  # only the new games were encoded
    monkeypatch.setattr(cache, "encoder_hash", lambda: "different")
    cache.build(tmp_path, workers=1)
    m3 = json.loads((cdir / "manifest.json").read_text())
    assert len(m3["shards"]) == 1 and len(m3["files"]) == 6 and not (cdir / "s00001").exists()


def test_a_changed_source_file_rebuilds_and_a_root_keeps_the_data_folder_clean(tmp_path):
    data = tmp_path / "data"
    _records(data, range(3))
    cdir = cache.build(data, tmp_path / "caches", workers=1)
    assert not (data / ".cache").exists() and cdir.parent == tmp_path / "caches"
    victim = sorted(data.glob("g*.jsonl.gz"))[0]
    victim.write_bytes(victim.read_bytes())  # new mtime
    import os
    os.utime(victim, (1, 1))
    logs = []
    cache.build(data, tmp_path / "caches", workers=1, log=logs.append)
    assert any("changed or disappeared" in x for x in logs)


def _mark_v3(path, every=2):
    """Give every `every`-th state of a record the engine-v3 `hitboxes` field (as real-engine v3
    states carry), so state_v3_fraction is neither 0 nor 1."""
    import gzip

    lines = [json.loads(x) for x in gzip.open(path, "rt")]
    k = 0
    for ln in lines:
        if "state" in ln:
            if k % every == 0:
                ln["state"]["players"][0]["hitboxes"] = []
            k += 1
    with gzip.open(path, "wt") as f:
        f.writelines(json.dumps(x) + "\n" for x in lines)


def test_training_from_the_cache_matches_the_in_memory_path(tmp_path, monkeypatch):
    # Two generations: the first explored (value-only targets, unscored in validation), the second
    # half engine-v3 states; weights, a per-generation cap.
    old, new = tmp_path / "gen0", tmp_path / "gen1"
    _records(old, range(7), explore=True)
    _records(new, range(7, 14))
    for path in sorted(new.glob("g*.jsonl.gz"))[::2]:
        _mark_v3(path)
    real_encode = train.encode

    def f16(state, seat, vocab):  # the in-memory path on float16-rounded inputs, as the cache stores them
        n, c = real_encode(state, seat, vocab)
        return [float(np.float16(x)) for x in n], c

    common = [str(old), str(new), "--epochs", "3", "--hidden", "16", "--value-weight", "0.5", "--workers", "1",
              "--games-per-gen", "6", "--gen-weights", "0.5,1", "--explore-targets", "all", "--no-ledger"]
    monkeypatch.setattr(train, "encode", f16)
    train.main(common + ["--out", str(tmp_path / "mem.npz"), "--in-memory"])
    monkeypatch.setattr(train, "encode", real_encode)
    train.main(common + ["--out", str(tmp_path / "cache.npz")])
    a, b = (np.load(tmp_path / f, allow_pickle=False) for f in ("mem.npz", "cache.npz"))
    ma, mb = json.loads(str(a["meta"])), json.loads(str(b["meta"]))
    assert ma["vocab"] == mb["vocab"] and ma["raw_defaults"] == mb["raw_defaults"]
    assert ma["state_v3_fraction"] == mb["state_v3_fraction"] and 0 < mb["state_v3_fraction"] < 1
    assert (ma["train_games"], ma["val_games"], ma["best_epoch"]) == (mb["train_games"], mb["val_games"],
                                                                      mb["best_epoch"])
    for k in a.files:
        if k != "meta":
            np.testing.assert_allclose(a[k], b[k], rtol=0, atol=1e-6, err_msg=k)


@pytest.mark.parametrize("value", [65505.0, -65505.0, float("inf"), float("nan")])
def test_cache_rejects_nonfinite_or_overflowing_numeric_inputs(tmp_path, monkeypatch, value):
    from yomibench import features
    _records(tmp_path, range(1))
    encode = features.encode
    def invalid(state, seat, vocab):
        n, c = encode(state, seat, vocab)
        n[0] = value
        return n, c
    monkeypatch.setattr(features, "encode", invalid)
    with pytest.raises(ValueError, match="out-of-range float16"):
        cache.build(tmp_path, workers=1)


def test_cached_rows_equal_in_memory_with_only_float16_rounding(tmp_path):
    old, new, ev = tmp_path / "gen0", tmp_path / "gen1", tmp_path / "eval"
    _records(old, range(6), explore=True)
    _records(new, range(6, 12))
    _records(ev, range(12, 15))
    for path in sorted(new.glob("g*.jsonl.gz"))[::2]:
        _mark_v3(path)
    kw = dict(workers=1, games_per_gen=5, gen_weights=[0.5, 1.0], eval_dirs=[ev])
    for targets in ("uniform-only", "all", "none"):
        direct = train.load_samples([old, new], train.Vocab(), 0.25, 0, 0.5, explore_targets=targets, **kw)
        vocab = train.Vocab()
        cached = cache.open_cached([old, new], vocab, 0.25, 0, 0.5, explore_targets=targets, **kw)
        drop_added = lambda h: {k: {f: v for f, v in m.items() if f != "added"} for k, m in h.items()}  # noqa: E731
        assert drop_added(direct["heldout"]) == drop_added(cached["heldout"])  # same frozen manifests
        assert targets != "all" or direct["train"]["value_only"] > 0
        for name in ("train", "val", "eval"):
            d, c = direct[name], cached[name]
            for key in ("games", "explored", "value_only", "v3_fraction", "v3_fraction_by_gen", "outcome_coverage"):
                assert d.get(key) == c.get(key), (targets, name, key)
            for field in ("nums0", "nums1"):
                got = c[field][0:len(c[field])]
                np.testing.assert_array_equal(got, d[field].astype(np.float16).astype(np.float32))
                assert np.isfinite(got).all()
            for field in ("cats0", "cats1"):
                np.testing.assert_array_equal(c[field][0:len(c[field])], d[field], err_msg=f"{targets} {name} {field}")
            for field in ("y", "w", "gen", "outcome", "phase", "baseline", "matchup"):
                np.testing.assert_array_equal(np.asarray(c[field]), d[field], err_msg=f"{targets} {name} {field}")
