"""The candidate heads (--aux-cand-value-weight / --aux-mix-weight): side losses on what the search
records at full turns (each candidate's search value, the solved mix). Off, training is unchanged;
on, play still reads only the value head, and the main cache is untouched."""

import gzip
import json
import math
import random

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from yomibench import cache, train  # noqa: E402
from yomibench.netvalue import BatchValue, Preview, net_key  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.features import Vocab, encode  # noqa: E402
from yomibench.policy import encode_candidate, flip_columns, view_flip  # noqa: E402
from yomibench.search import terminal_value  # noqa: E402
from yomibench.search_record import cap_positions, turn_candidates  # noqa: E402
from yomibench.selfplay import SlimTurn, read_slim, run_selfplay  # noqa: E402
from yomibench.valuenet import CAND, SEARCH_W, ValueNet, candidate_heads, trunk  # noqa: E402


def test_cap_keeps_the_support_and_samples_the_rest():
    strat = [0.0] * 40
    strat[3], strat[17], strat[39] = 0.5, 0.3, 0.2
    for seed in range(20):
        keep = cap_positions(strat, 8, random.Random(seed))
        assert len(keep) == 8 and {3, 17, 39} <= set(keep) and keep == sorted(keep)
    assert cap_positions(strat, 8, random.Random(1)) == cap_positions(strat, 8, random.Random(1))
    # a support larger than K: its K likeliest (ties by position), still in list order
    big = [0.1, 0.0, 0.3, 0.1, 0.2, 0.3]
    assert cap_positions(big, 3, random.Random(0)) == [2, 4, 5]
    assert cap_positions(big, 10, random.Random(0)) == [0, 1, 2, 3, 4, 5]  # short lists keep everything


def _cand(i):
    return {"action": f"Move{i % 3}", "data": {"x": i, "y": 1} if i % 2 else None, "extra": {}}


def test_turn_targets_are_the_recorded_values_and_mix():
    state = {"players": [{"x": 50}, {"x": -50}]}  # p1 stands right: its view is mirrored
    cands = [[_cand(i) for i in range(30)], [_cand(i) for i in range(2)]]
    strat = [[0.0] * 30, [0.25, 0.75]]
    strat[0][5], strat[0][20] = 0.6, 0.4
    values = [[round(i / 30 - 0.5, 4) for i in range(30)], [0.1, None]]
    turn = SlimTurn(state, 0.3, (False, False), values, None, False, strat, cands, 7)
    seats = turn_candidates(turn, 4, "g1.jsonl.gz:7")
    (f0, e0, v0, p0), (f1, e1, v1, p1) = seats
    keep = cap_positions(strat[0], 4, random.Random("g1.jsonl.gz:7:0"))
    assert {5, 20} <= set(keep) and f0 == view_flip(state, 0) == -1.0 and f1 == 1.0
    assert e0 == [encode_candidate(cands[0][i]) for i in keep]
    assert v0 == [values[0][i] for i in keep] and p0 == [strat[0][i] for i in keep]
    assert v1[0] == 0.1 and math.isnan(v1[1]) and p1 == [0.25, 0.75]
    # quick turns, and turns without a solved mix, have no candidate targets
    assert turn_candidates(turn._replace(cheap=True), 4, "x") is None
    assert turn_candidates(turn._replace(strategies=None), 4, "x") is None


# --- loaders on synthetic search records -------------------------------------------------------

@pytest.fixture(scope="module")
def games(tmp_path_factory):
    """Toy self-play records whose full turns carry known synthetic search records: candidate i of
    seat s at turn t has value (t + i) % 7 / 10 - 0.3 for s 0 (its negative for s 1), and the mix puts
    0.7 on the last candidate and 0.3 on the first. Quick turns (playout cap) are left as recorded."""
    src = tmp_path_factory.mktemp("src")
    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(8), src,
                 playout_cap=(0.4, 2), opening=(2, 0.5, 1.0))
    d = tmp_path_factory.mktemp("cand")
    for p in sorted(src.glob("g*.jsonl.gz")):
        lines = [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8")]
        for t, line in enumerate(lines):
            if "candidates" not in line or line.get("search") == "cheap" or not line.get("strategies"):
                continue
            n = [len(c) for c in line["candidates"]]
            line["candidate_values"] = [[(t + i) % 7 / 10 - 0.3 for i in range(n[0])],
                                        [-((t + i) % 7 / 10 - 0.3) for i in range(n[1])]]
            line["strategies"] = [[1.0] if k == 1 else [0.3] + [0.0] * (k - 2) + [0.7] for k in n]
        with gzip.open(d / p.name, "wt", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
    return d


def _synthetic(t, i, seat):
    v = (t + i) % 7 / 10 - 0.3
    return v if seat == 0 else -v


def test_loader_targets_are_exactly_the_records(games, tmp_path):
    kw = dict(val_fraction=0.3, seed=0, value_weight=0.5, workers=1, heldout_root=tmp_path / "h")
    mem = train.load_samples([games], Vocab(), cand_k=3, **kw)
    cac = cache.open_cached([games], Vocab(), root=tmp_path / "c", cand_k=3, **kw)
    tm, tc = mem["cand_table"], cac["cand_table"]
    assert tm["vocab"] == tc["vocab"]
    total = 0
    for split in ("train", "val"):
        a, b = mem[split], cac[split]
        assert np.array_equal(a["cand_slot"], b["cand_slot"]) and (a["cand_slot"] >= 0).any()
        for k in ("cand_flip", "cand_prob"):
            assert np.array_equal(a[k], b[k])
        assert np.array_equal(a["cand_val"], b["cand_val"], equal_nan=True)
        # the same candidates (as table content) in both loaders
        for t_, s_ in ((tm, a), (tc, b)):
            s_["_nums"] = t_["unums"][np.maximum(s_["cand_uid"], 0)] * (s_["cand_uid"] >= 0)[..., None]
            s_["_act"] = t_["uact"][np.maximum(s_["cand_uid"], 0)] * (s_["cand_uid"] >= 0)
        assert np.array_equal(a["_nums"], b["_nums"]) and np.array_equal(a["_act"], b["_act"])
    assert "cand_slot" not in mem["eval"] and "cand_slot" not in cac["eval"]
    # targets: the synthetic values and mixes, mix support always kept
    files = sorted(games.glob("g*.jsonl.gz"))
    for p in files:
        _, _, turns = read_slim(p, policy=True)
        for k, t in enumerate(turns):
            seats = turn_candidates(t, 3, f"{p.name}:{k}")
            if seats is None:
                assert t.cheap or not t.strategies
                continue
            total += 1
            for s, (_, encs, vals, probs) in enumerate(seats):
                n = len(t.candidates[s])
                keep = cap_positions(t.strategies[s], 3, random.Random(f"{p.name}:{k}:{s}"))
                assert vals == pytest.approx([_synthetic(k + 1, i, s) for i in keep])  # line 0 is the header
                if n > 1:
                    assert {0, n - 1} <= set(keep) and probs == pytest.approx([0.3 if i == 0 else 0.7 if i == n - 1
                                                                              else 0.0 for i in keep])
    assert total
    # what the loader holds per row is that turn's list
    n_train = int((mem["train"]["cand_slot"] >= 0).sum() + (mem["val"]["cand_slot"] >= 0).sum())
    assert 0 < n_train <= total


def test_main_cache_is_untouched_by_the_candidate_side_cache(games, tmp_path):
    h = cache.encoder_hash()
    cdir = cache.build(games, tmp_path / "c", workers=1)
    before = (cdir / "manifest.json").read_bytes()
    shards = sorted(x.name for x in cdir.iterdir())
    kw = dict(val_fraction=0.3, seed=0, workers=1, heldout_root=tmp_path / "h", root=tmp_path / "c")
    cache.open_cached([games], Vocab(), **kw)
    assert not list(cdir.glob("cands-*"))  # off: never built
    cache.open_cached([games], Vocab(), cand_k=4, **kw)
    assert (cdir / "manifest.json").read_bytes() == before and cache.encoder_hash() == h
    assert sorted(x.name for x in cdir.iterdir()) == sorted([*shards, "cands-k4"])
    side = json.loads((cdir / "cands-k4" / "manifest.json").read_text())
    assert side["k"] == 4 and side["encoder"] == cache.cand_hash(4) != cache.cand_hash(5)
    mtime = (cdir / "cands-k4" / "manifest.json").stat().st_mtime_ns
    cache.open_cached([games], Vocab(), cand_k=4, **kw)  # up to date: nothing rebuilt
    assert (cdir / "cands-k4" / "manifest.json").stat().st_mtime_ns == mtime


# --- training ---------------------------------------------------------------------------------

def _train(games, tmp_path, name, *flags, loader=()):
    out = tmp_path / f"{name}.npz"
    train.main([str(games), "--out", str(out), "--epochs", "2", "--hidden", "16", "--workers", "1",
                "--val-fraction", "0.3", "--value-weight", "0.5", "--select", "final", "--cache", str(tmp_path / "cache"),
                "--ledger", str(tmp_path / f"{name}.jsonl"), *loader, *flags])
    (row,) = [json.loads(x) for x in (tmp_path / f"{name}.jsonl").read_text().splitlines()]
    return ValueNet.load(out), row


@pytest.mark.parametrize("loader", [(), ("--in-memory",)])
def test_switch_off_trains_identical_params(games, tmp_path, loader):
    a, _ = _train(games, tmp_path, "a", loader=loader)
    b, _ = _train(games, tmp_path, "b", "--aux-cand-value-weight", "0", "--aux-mix-weight", "0", "--aux-cand-k", "5",
                  loader=loader)
    assert set(a.p) == set(b.p) and all(np.array_equal(a.p[k], b.p[k]) for k in a.p)
    assert not any(k.startswith(CAND) for k in a.p) and not any(k.startswith("aux_cand") for k in a.meta)
    assert not list((tmp_path / "cache").rglob("cands-*"))


def _states(games):
    return [t.state for p in sorted(games.glob("g*.jsonl.gz")) for t in read_slim(p)[2]]


def _previews(model, states):
    out = []
    for s in states:
        p = Preview(s)
        enc = [encode(s, seat, model.vocab) for seat in (0, 1)]
        p.feats = {net_key(model): (np.array([e[0] for e in enc], np.float32), np.array([e[1] for e in enc]))}
        t0 = terminal_value(s, 0)
        p.done, p.winner = t0 is not None, -1 if not t0 else (0 if t0 > 0 else 1)
        out.append(p)
    return out


@pytest.mark.parametrize("loader", [(), ("--in-memory",)])
def test_heads_train_with_the_other_switches_and_never_change_play(games, tmp_path, loader):
    net, row = _train(games, tmp_path, "cand", "--aux-cand-value-weight", "0.1", "--aux-mix-weight", "0.05",
                      "--aux-cand-k", "3", "--aux-search-heads", "1,4", "--aux-hp-weight", "0.2", "--target-nstep", "4",
                      loader=loader)
    m = net.meta
    assert m["aux_cand_value_weight"] == 0.1 and m["aux_mix_weight"] == 0.05 and m["aux_cand_k"] == 3
    assert m["aux_search_heads"] == [1, 4] and m["aux_hp_weight"] == 0.2 and m["target_nstep"] == 4
    assert m["val_aux_cand_value_mse"] is not None and m["val_aux_mix_ce"] is not None
    assert 0 <= m["val_aux_mix_top1"] <= 1 and set(m["aux_cand_vocab"]) == {"action", "dkey"}
    tb = row["provenance"]["target_blend"]
    assert tb["aux_cand_value_weight"] == 0.1 and tb["aux_mix_weight"] == 0.05 and tb["aux_cand_k"] == 3
    assert {SEARCH_W, "w3_aux", f"{CAND}value_w", f"{CAND}logit_w", f"{CAND}emb_action"} <= set(net.p)
    # play: the same values with or without the candidate heads' keys (numpy and Batch engine paths)
    states = _states(games)[:60]
    stripped = ValueNet({k: v for k, v in net.p.items() if not k.startswith(CAND)}, net.vocab,
                        {k: v for k, v in net.meta.items() if not k.startswith("aux_cand")})
    for seat in (0, 1):
        main = list(net.batch(states, seat))
        assert main == list(stripped.batch(states, seat))
        assert list(BatchValue(net).batch(_previews(net, states), seat)) == main
    with pytest.raises(ValueError, match="no candidate heads"):
        stripped.candidate_heads(states[0], 0, [])
    # the diagnostics path scores a recorded turn's candidates: values in [-1, 1], a mix summing to 1
    p = sorted(games.glob("g*.jsonl.gz"))[0]
    t = next(t for t in read_slim(p, policy=True)[2] if t.strategies and not t.cheap)
    v, mix = net.candidate_heads(t.state, 0, t.candidates[0])
    assert len(v) == len(mix) == len(t.candidates[0]) and np.all(np.abs(v) <= 1) and mix.sum() == pytest.approx(1)


def test_torch_candidate_heads_match_numpy():
    sizes = {"state": 3, "stance": 2, "char": 2, "kind": 2, "ostate": 2}
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    cmean, cstd = rng.normal(0, 1, train.N_CAND).astype(np.float32), rng.uniform(0.5, 2, train.N_CAND).astype(np.float32)
    cstd[3] = 1e-3  # a constant input enters as 0
    t = train.Net(sizes, np.zeros(train.N_NUMERIC, np.float32), np.ones(train.N_NUMERIC, np.float32), 8,
                  cand=({"action": 4, "dkey": 3}, cmean, cstd))
    nums = rng.normal(0, 1, (5, train.N_NUMERIC)).astype(np.float32)
    cats = rng.integers(0, 2, (5, train.N_CATEGORICAL))
    cnums = rng.normal(0, 1, (5, 6, train.N_CAND)).astype(np.float32)
    cact, cdkey = rng.integers(0, 4, (5, 6)), rng.integers(0, 3, (5, 6))
    with torch.no_grad():
        *_, h = t.outputs(torch.tensor(nums), torch.tensor(cats))
        v, lg = t.cand(h, torch.tensor(cnums), torch.tensor(cact), torch.tensor(cdkey))
    p = t.export()
    h_np = trunk(p, nums, cats)
    assert np.allclose(h.numpy(), h_np, atol=1e-6)
    v_np, lg_np = candidate_heads(p, h_np, cnums, cact, cdkey)
    assert np.allclose(v.numpy(), v_np, atol=1e-5) and np.allclose(lg.numpy(), lg_np, atol=1e-5)
    # the same net without the candidate heads draws the same initial weights for everything else
    torch.manual_seed(0)
    plain = train.Net(sizes, np.zeros(train.N_NUMERIC, np.float32), np.ones(train.N_NUMERIC, np.float32), 8).export()
    assert all(np.array_equal(plain[k], p[k]) for k in plain)


def test_flags_are_checked(games, tmp_path):
    for bad in (["--aux-cand-value-weight", "-1"], ["--aux-mix-weight", "-0.1"], ["--aux-cand-k", "0"]):
        with pytest.raises(SystemExit):
            train.main([str(games), "--out", str(tmp_path / "x.npz"), "--no-ledger", *bad])


def test_flip_columns_mirror_view():
    x = np.zeros((1, 2, 3, train.N_CAND), np.float32)
    x[..., train.X_COLUMNS[0]] = 1.0
    y = flip_columns(x, np.array([[1.0, -1.0]], np.float32))
    assert (y[0, 0, :, train.X_COLUMNS[0]] == 1).all() and (y[0, 1, :, train.X_COLUMNS[0]] == -1).all()
