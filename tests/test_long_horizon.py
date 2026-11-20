"""Training-only recipe switches: multi-step search targets (--target-nstep / --target-lambda),
auxiliary search-value heads (--aux-search-heads) and depth (--layers). Off, training is unchanged;
on, play still reads only the value head."""

import json
import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from yomibench import train  # noqa: E402
from yomibench.netvalue import BatchValue, Preview, net_key  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.cache import open_cached  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.features import Vocab, encode  # noqa: E402
from yomibench.search import terminal_value  # noqa: E402
from yomibench.selfplay import SlimTurn, read_slim, run_selfplay  # noqa: E402
from yomibench.train import select_rows  # noqa: E402
from yomibench.valuenet import SEARCH_B, SEARCH_W, ValueNet, forward  # noqa: E402

NAN = float("nan")


def _t(value, cheap=False, pick=None):
    """A slim turn: `pick` "least-tried" / "softmax" marks p1's pick exploratory in that mode."""
    return SlimTurn(None, value, (pick is not None, False), None, pick, cheap)


# 0 full 0.2 | 1 quick | 2 full 0.4 | 3 full -0.6, least-tried pick | 4 full 0.8 | 5 quick | 6 full 1.0
GAME = [_t(0.2), _t(0.9, cheap=True), _t(0.4), _t(-0.6, pick="least-tried"), _t(0.8), _t(0.0, cheap=True), _t(1.0)]


def _ys(rows):
    return {k: round(y, 6) for k, y, _ in rows}


def test_nstep_and_lambda_targets_by_hand():
    """Windows stop at the least-tried pick (its own value included); quick turns' values never count."""
    kw = dict(split="train", explore_targets="none", value_weight=0.5, z=1.0)  # every row on the blend
    # N=2 means: 0.3, -0.1, -0.1, -0.6 (window stops at 3), 0.9, 1.0, 1.0
    rows, _ = select_rows(GAME, target_nstep=2, **kw)
    assert _ys(rows) == {0: 0.65, 1: 1.0, 2: 0.45, 3: 0.2, 4: 0.95, 5: 1.0, 6: 1.0}  # quick turns: outcome
    rows, _ = select_rows(GAME, target_nstep=2, cheap_target="next-full", **kw)
    assert _ys(rows)[1] == 0.45 and _ys(rows)[5] == 1.0  # quick turns: the mean from the next full turn
    # lambda 0.5: k0 (0.2 + 0.5*0.4 + 0.25*-0.6) / 1.75; k2 (0.4 - 0.3) / 1.5; k3 -0.6; k4 (0.8 + 0.5) / 1.5
    rows, _ = select_rows(GAME, target_lambda=0.5, **kw)
    want = {0: 0.25 / 1.75, 2: 0.1 / 1.5, 3: -0.6, 4: 1.3 / 1.5, 6: 1.0}
    assert _ys(rows) == {**{k: round(0.5 + 0.5 * v, 6) for k, v in want.items()}, 1: 1.0, 5: 1.0}
    # Before the pick (--explore-targets uniform-only): value-only targets on the same means; quick turn 1
    # (no value of its own with the outcome cheap target) is skipped, as without the switch.
    rows, counts = select_rows(GAME, "train", "uniform-only", 0.5, 1.0, target_nstep=2)
    assert _ys(rows) == {0: 0.3, 2: -0.1, 3: -0.6, 4: 0.95, 5: 1.0, 6: 1.0} and counts["value_only"] == 3


def test_one_step_reproduces_the_single_turn_target():
    for split in ("train", "val"):
        for explore in ("uniform-only", "none"):
            base, _ = select_rows(GAME, split, explore, 0.5, -1.0)
            for kw in ({"target_nstep": 1}, {"target_lambda": 0.0}):
                assert select_rows(GAME, split, explore, 0.5, -1.0, **kw)[0] == base
    # validation and evaluation targets never change
    for split in ("val", "eval"):
        assert select_rows(GAME, split, "none", 0.5, 1.0, target_nstep=3)[0] == select_rows(GAME, split, "none", 0.5, 1.0)[0]
    with pytest.raises(ValueError):
        select_rows(GAME, "train", "none", 0.5, 1.0, target_nstep=2, target_lambda=0.5)


def test_aux_search_targets_by_hand():
    """Nearest full turn at or after k+d; NaN past a least-tried pick or past the game's end."""
    rows, counts = select_rows(GAME, "train", "uniform-only", 0.5, 1.0, aux_offsets=(1, 4, 16))
    got = {k: a for (k, _, _), a in zip(rows, counts["aux"])}
    want = {0: (0.4, NAN, NAN),  # +4 would read turn 4, after the pick at 3
            2: (-0.6, NAN, NAN),  # the pick's own value is in the window
            3: (NAN, NAN, NAN),  # +1 is after the pick
            4: (1.0, NAN, NAN),  # quick turn 5 skipped; the game ends before +4
            5: (1.0, NAN, NAN), 6: (NAN, NAN, NAN)}
    assert set(got) == set(want)
    for k, w in want.items():
        assert all((math.isnan(a) and math.isnan(b)) or a == pytest.approx(b) for a, b in zip(got[k], w)), (k, got[k])
    # a softmax pick doesn't cut the window by default (it keeps outcome targets), but does with "all"
    soft = [_t(0.1), _t(0.2, pick="softmax"), _t(0.3)]
    assert select_rows(soft, "train", "uniform-only", 0.5, 1.0, aux_offsets=(2,))[1]["aux"][0] == (0.3,)
    assert math.isnan(select_rows(soft, "train", "all", 0.5, 1.0, aux_offsets=(2,))[1]["aux"][2][0])
    # without the switch there are no aux targets, and the rows don't change
    plain, c = select_rows(GAME, "train", "uniform-only", 0.5, 1.0)
    assert "aux" not in c and plain == rows


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    d = tmp_path_factory.mktemp("lh")
    # quick turns, and opening exploration (uniform picks) to cut windows
    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(8), d,
                 playout_cap=(0.5, 2), opening=(2, 0.5, 1.0))
    return d


def test_cache_and_in_memory_loaders_agree_on_the_new_targets(games, tmp_path):
    kw = dict(val_fraction=0.3, seed=0, value_weight=0.5, workers=1, heldout_root=tmp_path / "h")
    for q in ({"target_nstep": 4, "aux_offsets": [1, 4, 16]}, {"target_lambda": 0.7, "cheap_target": "next-full"}):
        mem = train.load_samples([games], Vocab(), quick_turns=q, **kw)
        cac = open_cached([games], Vocab(), root=tmp_path / "c", quick_turns=q, **kw)
        for split in ("train", "val"):
            assert np.allclose(np.asarray(mem[split]["y"]), np.asarray(cac[split]["y"]), atol=1e-6)
            if "aux_offsets" in q:
                a, b = np.asarray(mem[split]["aux_search"]), np.asarray(cac[split]["aux_search"])
                assert a.shape == b.shape == (len(mem[split]["y"]), 3)
                assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)])
                assert np.isnan(a[:, 2]).sum() > np.isnan(a[:, 0]).sum()  # +16 runs past games' ends more often


def _train(games, tmp_path, name, *flags, loader=()):
    out = tmp_path / f"{name}.npz"
    train.main([str(games), "--out", str(out), "--epochs", "2", "--hidden", "16", "--workers", "1",
                "--val-fraction", "0.3", "--value-weight", "0.5", "--select", "final", "--cache", str(tmp_path / "cache"),
                "--ledger", str(tmp_path / f"{name}.jsonl"), *loader, *flags])
    (row,) = [json.loads(x) for x in (tmp_path / f"{name}.jsonl").read_text().splitlines()]
    return ValueNet.load(out), row


@pytest.mark.parametrize("loader", [(), ("--in-memory",)])
def test_switches_off_train_identical_params(games, tmp_path, loader):
    """The defaults, spelled out, train the very same model as no flags."""
    a, _ = _train(games, tmp_path, "a", loader=loader)
    b, _ = _train(games, tmp_path, "b", "--layers", "2", "--aux-search-weight", "0.1", loader=loader)
    assert set(a.p) == set(b.p) and all(np.array_equal(a.p[k], b.p[k]) for k in a.p)
    assert not {"layers", "target_nstep", "target_lambda", "aux_search_heads"} & set(a.meta)


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


def test_aux_search_heads_are_saved_and_never_change_play(games, tmp_path):
    net, row = _train(games, tmp_path, "aux", "--aux-search-heads", "1,4,16", "--aux-search-weight", "0.2",
                      "--target-nstep", "4")
    assert net.p[SEARCH_W].shape == (16, 3) and net.p[SEARCH_B].shape == (3,)
    assert net.meta["aux_search_heads"] == [1, 4, 16] and net.meta["aux_search_weight"] == 0.2
    assert set(net.meta["val_aux_search_mse"]) == {"1", "4", "16"} and net.meta["target_nstep"] == 4
    tb = row["provenance"]["target_blend"]
    assert tb["target_nstep"] == 4 and tb["aux_search_heads"] == [1, 4, 16]
    states = _states(games)[:60]
    stripped = ValueNet({k: v for k, v in net.p.items() if k not in (SEARCH_W, SEARCH_B)}, net.vocab, dict(net.meta))
    for seat in (0, 1):
        main = list(net.batch(states, seat))
        assert main == list(stripped.batch(states, seat))  # numpy path
        assert list(BatchValue(net).batch(_previews(net, states), seat)) == main  # Batch engine path
    assert np.array_equal(net.search_heads(states, 1), -net.search_heads(states, 0))
    with pytest.raises(ValueError, match="no auxiliary search"):
        stripped.search_heads(states, 0)


def test_three_hidden_layers_train_save_and_play(games, tmp_path):
    net, _ = _train(games, tmp_path, "deep", "--layers", "3")
    assert net.meta["layers"] == 3
    assert {"w1", "w2", "wh3", "w_out", "b_out"} <= set(net.p) and "w3" not in net.p  # older code: KeyError
    states = _states(games)[:40]
    for seat in (0, 1):
        assert list(BatchValue(net).batch(_previews(net, states), seat)) == list(net.batch(states, seat))
    # the torch net and the numpy forward compute the same value head
    sizes = {"state": 3, "stance": 2, "char": 2, "kind": 2, "ostate": 2}
    torch.manual_seed(0)
    t = train.Net(sizes, np.zeros(train.N_NUMERIC, np.float32), np.ones(train.N_NUMERIC, np.float32), 8, layers=4,
                  aux_search=2)
    rng = np.random.default_rng(0)
    nums = rng.normal(0, 1, (5, train.N_NUMERIC)).astype(np.float32)
    cats = rng.integers(0, 2, (5, train.N_CATEGORICAL))
    with torch.no_grad():
        v, _, s = t.all_heads(torch.tensor(nums), torch.tensor(cats))
    p = t.export()
    assert np.allclose(v.numpy(), forward(p, nums, cats), atol=1e-6)
    assert np.allclose(s.numpy(), forward(p, nums, cats, head="search"), atol=1e-6)
    # fine-tuning needs the same depth; a deep model fine-tunes from itself
    with pytest.raises(SystemExit):
        train.main([str(games), "--out", str(tmp_path / "x.npz"), "--no-ledger", "--hidden", "16",
                    "--init-from", str(tmp_path / "deep.npz")])
    again, _ = _train(games, tmp_path, "deep2", "--layers", "3", "--init-from", str(tmp_path / "deep.npz"), "--lr", "0",
                      "--epochs", "1")
    assert all(np.array_equal(again.p[k], net.p[k]) for k in net.p if not k.startswith("emb_"))


def test_flags_are_checked(games, tmp_path):
    for bad in (["--target-nstep", "2", "--target-lambda", "0.5"], ["--target-nstep", "0"], ["--target-lambda", "1.5"],
                ["--aux-search-heads", "0,4"], ["--aux-search-heads", "4,4"], ["--aux-search-heads", "x"],
                ["--layers", "1"], ["--aux-search-heads", "1", "--aux-search-weight", "-1"]):
        with pytest.raises(SystemExit):
            train.main([str(games), "--out", str(tmp_path / "x.npz"), "--no-ledger", *bad])
