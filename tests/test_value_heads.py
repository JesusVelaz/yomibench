"""Training options for the value net: which epoch is kept (--select), the auxiliary hp-margin head
(--aux-hp-weight) and the residual on the hp lead (--residual-hp), and that models without them
load and score exactly as before."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from yomibench import train  # noqa: E402
from yomibench.netvalue import BatchValue, Preview, net_key  # noqa: E402
from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.features import Vocab, encode  # noqa: E402
from yomibench.search import evaluate, terminal_value  # noqa: E402
from yomibench.selfplay import final_hp_margin, read_hp_margin, read_slim, run_selfplay  # noqa: E402
from yomibench.valuenet import ValueNet, forward, hp_lead  # noqa: E402


@pytest.fixture(scope="module")
def games(tmp_path_factory):
    d = tmp_path_factory.mktemp("games")
    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(8), d)
    return d


def _states(games):
    return [t.state for p in sorted(games.glob("g*.jsonl.gz")) for t in read_slim(p)[2]]


def _train(games, tmp_path, *flags):
    out, ledger = tmp_path / "m.npz", tmp_path / "ledger.jsonl"
    train.main([str(games), "--out", str(out), "--epochs", "3", "--hidden", "16", "--workers", "1",
                "--val-fraction", "0.3", "--cache", str(tmp_path / "cache"), "--ledger", str(ledger), *flags])
    (row,) = [json.loads(x) for x in ledger.read_text().splitlines()]
    return ValueNet.load(out), row["metrics"]


def _plain(net, states, seat):
    """A model's value as scored before these options: (f(seat view) - f(other view)) / 2, terminal exact."""
    out = []
    for view in (seat, 1 - seat):
        enc = [encode(s, view, net.vocab) for s in states]
        out.append(forward(net.p, np.array([e[0] for e in enc], np.float32), np.array([e[1] for e in enc], np.int64)))
    values = (out[0] - out[1]) / 2
    return [terminal_value(s, seat) if terminal_value(s, seat) is not None else v for s, v in zip(states, values)]


# --- --select ---------------------------------------------------------------------------------

@pytest.mark.parametrize("select", ["best", "final"])
def test_select_keeps_the_best_or_the_last_epoch_and_records_it(games, tmp_path, select):
    net, m = _train(games, tmp_path, *(["--select", select] if select == "final" else []))
    meta = net.meta
    assert meta["select"] == m["select"] == select
    assert meta["kept_epoch"] == m["kept_epoch"] == (2 if select == "final" else meta["best_epoch"])
    assert meta["val_mse"] == m["val_mse"]
    assert meta["best_val_mse"] <= meta["val_mse"]
    if select == "best":
        assert meta["val_mse"] == meta["best_val_mse"]
    # The saved weights are the kept epoch's: rescoring them gives the MSE recorded for that epoch.
    assert meta["val_mse_by_phase"]["all"] == pytest.approx(meta["val_mse"], abs=2e-4)
    assert m["outcome_positions"] == m["train_positions"] - m["value_only_positions"]


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "tools" / "round3_loop.py").exists(),
                    reason="the fleet loop isn't part of this build")
def test_the_round3_trainer_trains_a_fixed_three_epochs_and_keeps_the_last():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from round3_loop import TRAIN_SETTINGS

    s = TRAIN_SETTINGS
    assert s[s.index("--epochs") + 1] == "3" and s[s.index("--select") + 1] == "final"


# --- auxiliary hp-margin head -----------------------------------------------------------------

def test_final_hp_margin_comes_from_the_result(games):
    p = sorted(games.glob("g*.jsonl.gz"))[0]
    _, result, turns = read_slim(p)
    res = result["result"]
    players = res["final_state"]["players"]
    want = res["hp"][0] / players[0]["max_hp"] - res["hp"][1] / players[1]["max_hp"]
    assert final_hp_margin(result) == pytest.approx(want) and read_hp_margin(p) == pytest.approx(want)
    # max hp from the first state when the result has no final state; NaN when nothing says
    no_final = {"result": {k: v for k, v in res.items() if k != "final_state"}}
    assert final_hp_margin(no_final, turns[0].state) == pytest.approx(want)
    assert np.isnan(final_hp_margin(no_final)) and np.isnan(final_hp_margin({"result": {"winner": 0}}))


def test_aux_head_is_trained_saved_and_ignored_by_play(games, tmp_path):
    net, m = _train(games, tmp_path, "--aux-hp-weight", "0.5")
    assert {"w3_aux", "b3_aux"} <= set(net.p) and net.meta["aux_hp_weight"] == m["aux_hp_weight"] == 0.5
    assert net.meta["val_aux_hp_mse"] is not None
    states = _states(games)[:40]
    without = ValueNet({k: v for k, v in net.p.items() if not k.endswith("_aux")}, net.vocab, dict(net.meta))
    for seat in (0, 1):
        assert list(net.batch(states, seat)) == list(without.batch(states, seat)) == _plain(net, states, seat)
        # the aux head itself is antisymmetric too
    assert list(net.hp_margin(states, 1)) == list(-net.hp_margin(states, 0))
    with pytest.raises(ValueError, match="no auxiliary"):
        without.hp_margin(states, 0)


def test_torch_aux_head_matches_numpy_and_is_antisymmetric():
    sizes = {"state": 3, "stance": 2, "char": 2, "kind": 2, "ostate": 2}
    torch.manual_seed(0)
    t = train.Net(sizes, np.zeros(train.N_NUMERIC, np.float32), np.ones(train.N_NUMERIC, np.float32), 8, aux=True)
    rng = np.random.default_rng(0)
    nums = [rng.normal(0, 1, (5, train.N_NUMERIC)).astype(np.float32) for _ in range(2)]
    cats = [rng.integers(0, 2, (5, train.N_CATEGORICAL)) for _ in range(2)]
    with torch.no_grad():
        (v0, a0), (v1, a1) = (t.heads(torch.tensor(n), torch.tensor(c)) for n, c in zip(nums, cats))
        assert torch.equal(v0, t(torch.tensor(nums[0]), torch.tensor(cats[0])))
    p = t.export()
    assert np.allclose(a0.numpy(), forward(p, nums[0], cats[0], head="aux"), atol=1e-6)
    assert np.allclose(v0.numpy(), forward(p, nums[0], cats[0]), atol=1e-6)
    assert torch.equal((a0 - a1) / 2, -((a1 - a0) / 2))


def test_cached_and_in_memory_hp_margins_agree_and_old_caches_are_filled_in(games, tmp_path):
    from yomibench import cache

    mem = train.load_samples([games], Vocab(), 0.3, 0, workers=1, heldout_root=tmp_path / "h")
    cdir = cache.build(games, tmp_path / "c", workers=1)
    manifest = json.loads((cdir / "manifest.json").read_text())
    assert all("hp_margin" in f for f in manifest["files"].values())
    for f in manifest["files"].values():  # a cache built before the margin was kept
        del f["hp_margin"]
    (cdir / "manifest.json").write_text(json.dumps(manifest))
    lacking = cache.open_cached([games], Vocab(), 0.3, 0, workers=1, heldout_root=tmp_path / "h", root=tmp_path / "c")
    assert np.isnan(lacking["train"]["hp_margin"]).all()
    filled = cache.open_cached([games], Vocab(), 0.3, 0, workers=1, heldout_root=tmp_path / "h", root=tmp_path / "c",
                               hp_margin=True)
    for split in ("train", "val"):
        a, b = np.asarray(mem[split]["hp_margin"]), np.asarray(filled[split]["hp_margin"])
        assert len(a) and np.isfinite(a).all() and np.array_equal(a, b)


# --- residual on the hp lead -------------------------------------------------------------------

def test_residual_value_is_the_clipped_hp_lead_plus_the_net(games, tmp_path):
    net, m = _train(games, tmp_path, "--residual-hp", "--select", "final")
    assert net.meta["residual_hp"] is True and m["residual_hp"] is True and net.residual_hp
    states = _states(games)
    plain = ValueNet(net.p, net.vocab, {k: v for k, v in net.meta.items() if k != "residual_hp"})
    assert not plain.residual_hp
    for seat in (0, 1):
        got = net.batch(states, seat)
        corr = plain.batch(states, seat)
        for s, g, c in zip(states, got, corr):
            t = terminal_value(s, seat)
            assert g == (t if t is not None else pytest.approx(np.clip(evaluate(s, seat) + c, -1, 1), abs=1e-6))
    assert list(net.batch(states, 1)) == list(-net.batch(states, 0))  # antisymmetric, exactly
    # the saved model's validation score is that of the clipped residual value
    assert net.meta["val_mse_by_phase"]["all"] == pytest.approx(net.meta["val_mse"], abs=2e-4)


def test_hp_lead_from_features_is_evaluate():
    from yomibench import features

    for hp in [(900, 900), (1, 900), (450, 333), (0, 10)]:
        s = {"players": [{"hp": hp[0], "max_hp": 900, "x": -50}, {"hp": hp[1], "max_hp": 900, "x": 50}], "objects": []}
        for seat in (0, 1):
            nums = np.array([encode(s, seat, Vocab())[0]], np.float32)
            assert hp_lead(nums, features)[0] == pytest.approx(hp[seat] / 900 - hp[1 - seat] / 900, abs=1e-7)


def test_batch_value_scores_a_residual_net_like_the_handler(games, tmp_path):
    """BatchValue on previews carrying the encoder's features gives ValueNet.batch's values, for a
    residual net (the hp lead read from the features) and a plain one (no simulator needed)."""
    net, _ = _train(games, tmp_path, "--residual-hp")
    states = _states(games)
    for model in (net, ValueNet(net.p, net.vocab, {**net.meta, "residual_hp": False})):
        previews = []
        for s in states:
            p = Preview(s)
            enc = [encode(s, seat, model.vocab) for seat in (0, 1)]
            p.feats = {net_key(model): (np.array([e[0] for e in enc], np.float32), np.array([e[1] for e in enc]))}
            t0 = terminal_value(s, 0)
            p.done, p.winner = t0 is not None, -1 if not t0 else (0 if t0 > 0 else 1)
            previews.append(p)
        for seat in (0, 1):
            assert list(BatchValue(model).batch(previews, seat)) == list(model.batch(states, seat))


def test_models_saved_before_these_options_load_and_score_as_before(games, tmp_path):
    net, _ = _train(games, tmp_path)
    old_meta = {k: v for k, v in net.meta.items() if k not in ("select", "kept_epoch", "aux_hp_weight", "residual_hp",
                                                               "val_mse")}
    ValueNet(net.p, net.vocab, old_meta).save(tmp_path / "old.npz")
    old = ValueNet.load(tmp_path / "old.npz")
    assert not old.residual_hp and "w3_aux" not in old.p
    states = _states(games)
    for seat in (0, 1):
        assert list(old.batch(states, seat)) == _plain(old, states, seat)
