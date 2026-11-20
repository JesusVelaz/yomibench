"""Training-data rules: which positions get which targets, and which records may be mixed."""

import pytest

torch = pytest.importorskip("torch")

from yomibench.bots import MatrixBot  # noqa: E402
from yomibench.engine import connect  # noqa: E402
from yomibench.features import Vocab  # noqa: E402
from yomibench.selfplay import read_slim, run_selfplay  # noqa: E402


def _split_files(dirs, *_):
    """The (train, val) record files load_samples used: each folder's held-out manifest decides."""
    import json

    train_files, val_files = [], []
    for d in dirs:
        held = set(json.loads((d / ".heldout.json").read_text())["heldout"])
        for p in sorted(d.glob("g*.jsonl.gz")):
            (val_files if p.name in held else train_files).append(p)
    return train_files, val_files


def test_positions_up_to_the_last_exploratory_pick_train_on_the_value_alone(tmp_path):
    from yomibench import train

    # Epsilon 1: each seat's first 3 decisions with a choice explore.
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50, search="oracle"), [("Toy", "Toy")],
                 range(4), tmp_path, opening=(3, 1.0, 1.0))
    data = train.load_samples([tmp_path], Vocab(), 0.25, 0, value_weight=0.25, workers=1)
    train_files, val_files = _split_files([tmp_path], 0.25, 0)

    want_y, explored = [], 0
    for path in train_files:
        _, result, turns = read_slim(path)
        z = {0: 1.0, 1: -1.0, None: 0.0}[result["result"]["winner"]]
        last = max(k for k, t in enumerate(turns) if any(t.explore))  # epsilon 1: every game explores
        for k, t in enumerate(turns):
            if k <= last:
                explored += 1
                if t.value is not None:
                    want_y.append(t.value)  # the search value only, whatever the value weight
            else:
                want_y.append(0.75 * z + 0.25 * (z if t.value is None else t.value))
    assert data["train"]["y"] == pytest.approx(want_y, abs=1e-6)
    assert data["train"]["explored"] == explored > 0

    # Validation scores against the outcome, so it keeps only positions after the last exploratory pick.
    kept = dropped = 0
    for path in val_files:
        turns = read_slim(path)[2]
        last = max(k for k, t in enumerate(turns) if any(t.explore))
        kept, dropped = kept + len(turns) - last - 1, dropped + last + 1
    assert val_files and len(data["val"]["outcome"]) == kept and data["val"]["explored"] == dropped


def test_games_without_exploration_keep_outcome_targets(tmp_path):
    from yomibench import train

    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50, search="oracle"), [("Toy", "Toy")],
                 range(3), tmp_path)
    data = train.load_samples([tmp_path], Vocab(), 0.3, 0, workers=1)
    train_files, _ = _split_files([tmp_path], 0.3, 0)
    assert data["train"]["explored"] == data["val"]["explored"] == 0
    assert (data["train"]["y"] == data["train"]["outcome"]).all()  # value weight 0: the outcome everywhere
    assert len(data["train"]["y"]) == sum(len(read_slim(p)[2]) for p in train_files)


def _expected(path, explore_targets, value_weight):
    """What load_samples should give for one record: (train targets, explored count, value-only
    count, validation positions kept), from the record's own marks."""
    from yomibench.selfplay import read_record

    lines = read_record(path)
    z = {0: 1.0, 1: -1.0, None: 0.0}[lines[-1]["result"]["winner"]]
    marks = [(any(ln.get("explore", [False, False])), ln.get("explore_mode")) for ln in lines[1:-1]]
    uniform = [m and mode != "softmax" for m, mode in marks]
    last_uniform = max((k for k, u in enumerate(uniform) if u), default=-1)
    last_any = max((k for k, (m, _) in enumerate(marks) if m), default=-1)
    cut = {"uniform-only": last_uniform, "all": last_any, "none": -1}[explore_targets]
    ys, value_only = [], 0
    for k, ln in enumerate(lines[1:-1]):
        v = ln["value"]
        if k <= cut:
            if v is not None:
                ys.append(v)
                value_only += 1
        else:
            ys.append((1 - value_weight) * z + value_weight * (z if v is None else v))
    return ys, cut + 1, value_only, len(marks) - last_uniform - 1


def _explored_games(tmp_path, opening, explore_cfg, seeds=range(4)):
    """Toy self-play with the given exploration, plus set 4's factor-exploration notes in
    `oracle.explore` on every searched decision (they change no pick, so they must not count)."""
    import gzip
    import json

    from yomibench.selfplay import read_record

    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50, search="oracle"), [("Toy", "Toy")],
                 seeds, tmp_path, opening=opening, explore_cfg=explore_cfg)
    modes = []
    for path in sorted(tmp_path.glob("g*.jsonl.gz")):
        lines = read_record(path)
        for ln in lines[1:-1]:
            if "explore" in ln:
                modes.append(ln.get("explore_mode", "opening"))
            if "oracle" in ln:
                ln["oracle"]["explore"] = {"margin": [1, 0], "random": [2, 2]}
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in lines))
    return modes


@pytest.mark.parametrize("explore_targets", ["uniform-only", "all", "none"])
def test_explore_targets_choose_which_picks_cut_outcome_targets(tmp_path, explore_targets):
    from yomibench import train

    # Opening noise (uniform) in each seat's first 2 decisions, then set 4's softmax exploration.
    modes = _explored_games(tmp_path, (2, 1.0, 1.0), (0.3, 0.1))
    assert "opening" in modes and "softmax" in modes
    data = train.load_samples([tmp_path], Vocab(), 0.25, 0, value_weight=0.25, workers=1,
                              explore_targets=explore_targets)
    train_files, val_files = _split_files([tmp_path], 0.25, 0)
    want_y, explored, value_only = [], 0, 0
    for path in train_files:
        ys, e, vo, _ = _expected(path, explore_targets, 0.25)
        want_y += ys
        explored += e
        value_only += vo
    assert data["train"]["y"] == pytest.approx(want_y, abs=1e-6)
    assert data["train"]["explored"] == explored and data["train"]["value_only"] == value_only
    # Validation drops only the positions at or before the last uniform pick, whatever the setting.
    kept = sum(_expected(path, explore_targets, 0.25)[3] for path in val_files)
    assert val_files and len(data["val"]["outcome"]) == kept
    assert data["val"]["value_only"] == 0


def test_softmax_exploration_keeps_outcome_targets_and_validation_by_default(tmp_path):
    from yomibench import train

    modes = _explored_games(tmp_path, None, (0.3, 0.1))
    assert modes and set(modes) == {"softmax"}
    data = train.load_samples([tmp_path], Vocab(), 0.25, 0, workers=1)  # default: uniform-only
    assert data["train"]["explored"] == data["train"]["value_only"] == data["val"]["explored"] == 0
    assert (data["train"]["y"] == data["train"]["outcome"]).all()  # value weight 0: the outcome everywhere
    train_files, val_files = _split_files([tmp_path], 0.25, 0)
    assert len(data["val"]["outcome"]) == sum(len(read_slim(path)[2]) for path in val_files)
    old = train.load_samples([tmp_path], Vocab(), 0.25, 0, workers=1, explore_targets="all")
    assert old["train"]["value_only"] > 0  # the previous rule cut them off
    assert len(old["val"]["outcome"]) == len(data["val"]["outcome"])  # same validation positions
    with pytest.raises(ValueError, match="explore_targets"):
        train.load_samples([tmp_path], Vocab(), 0.25, 0, workers=1, explore_targets="softmax")


def test_marks_without_a_mode_are_uniform():
    from yomibench.selfplay import SlimTurn, uniform_explore
    from yomibench.train import explore_cutoffs

    t = lambda e, mode=None: SlimTurn({}, 0.0, e, None, mode if any(e) else None)  # noqa: E731
    turns = [t((True, False), "uniform"), t((False, False)), t((False, True), "softmax"), t((False, False))]
    assert [uniform_explore(x) for x in turns] == [True, False, False, False]
    assert explore_cutoffs(turns) == (0, 0)
    assert explore_cutoffs(turns, "all") == (2, 0)
    assert explore_cutoffs(turns, "none") == (-1, 0)
    assert uniform_explore(t((True, True), "some-new-mode"))  # anything but softmax counts as off-policy


def test_train_reports_and_records_the_value_only_share(tmp_path, capsys):
    import json

    import numpy as np

    from yomibench import train

    _explored_games(tmp_path / "g", (2, 1.0, 1.0), (0.3, 0.1))
    out, metrics = tmp_path / "m.npz", tmp_path / "ledger.jsonl"
    data = train.load_samples([tmp_path / "g"], Vocab(), 0.25, 0, workers=1)
    train.main([str(tmp_path / "g"), "--out", str(out), "--epochs", "1", "--hidden", "8", "--workers", "1",
                "--val-fraction", "0.25", "--ledger", str(metrics)])
    share = data["train"]["value_only"] / len(data["train"]["y"])
    assert 0 < share < 1
    assert f"({share:.1%})" in capsys.readouterr().out
    meta = json.loads(str(np.load(out)["meta"]))
    assert meta["explore_targets"] == "uniform-only" and meta["value_only_target_fraction"] == round(share, 4)
    (row,) = [json.loads(x) for x in metrics.read_text().splitlines()]
    assert row["metrics"]["explore_targets"] == "uniform-only"
    assert row["metrics"]["value_only_target_fraction"] == round(share, 4)
    assert row["metrics"]["value_only_positions"] == data["train"]["value_only"]
    assert row["provenance"]["target_blend"]["explore_targets"] == "uniform-only"


def _hello(engine="yomisim", version="2", game="1.10.0-steam", cset="2", **settings):
    return {"engine": {"engine": engine, "version": version, "game_version": game, "candidate_set": cset,
                       "match_settings": {"stage_width": 800, **settings}}}


def test_real_game_and_simulator_records_of_the_same_game_mix():
    from yomibench.train import check_compatible

    check_compatible([_hello("yomi"), _hello("yomisim"), _hello("yomisim")])


@pytest.mark.parametrize("other", [_hello(version="1"), _hello(game="1.11.0-steam"), _hello(stage_width=900)])
def test_records_of_different_games_are_refused(other):
    from yomibench.train import check_compatible

    with pytest.raises(SystemExit, match="different game versions, state versions or match settings"):
        check_compatible([_hello(), other])
    with pytest.raises(SystemExit):  # the candidate-set override doesn't cover a different game
        check_compatible([_hello(), other], allow_mixed_candidate_sets=True)


def test_mixed_candidate_sets_need_the_flag():
    from yomibench.train import check_compatible

    with pytest.raises(SystemExit, match=r"1 x set 2, 1 x set 3\); pass --allow-mixed-candidate-sets"):
        check_compatible([_hello(cset="2"), _hello("yomi", cset="3")])
    check_compatible([_hello(cset="2"), _hello(cset="3")], allow_mixed_candidate_sets=True)


def _rewrite_headers(folder, **engine):
    """Give every record in `folder` these engine hello fields."""
    import gzip
    import json

    for path in folder.glob("g*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        header = json.loads(lines[0])
        header["engine"].update(engine)
        lines[0] = json.dumps(header)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def test_train_refuses_mixed_candidate_sets_before_parsing_unless_allowed(tmp_path):
    from yomibench import train

    a, b = tmp_path / "a", tmp_path / "b"
    for d, seeds in ((a, range(2)), (b, range(2, 4))):
        run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], seeds, d)
    _rewrite_headers(a, engine="yomi", candidate_set="2")
    _rewrite_headers(b, engine="yomisim", candidate_set="2")
    data = train.load_samples([a, b], Vocab(), 0.25, 0, workers=1)  # real game and simulator: one game
    assert data["train"]["games"] + data["val"]["games"] == 4
    _rewrite_headers(b, candidate_set="3")
    with pytest.raises(SystemExit, match="different candidate sets"):
        train.main([str(a), str(b), "--out", str(tmp_path / "m.npz"), "--epochs", "1", "--workers", "1"])
    assert not (tmp_path / "m.npz").exists()
    train.main([str(a), str(b), "--out", str(tmp_path / "m.npz"), "--epochs", "1", "--hidden", "8", "--workers", "1",
                "--allow-mixed-candidate-sets"])
    assert (tmp_path / "m.npz").exists()


def _generations(tmp_path, n=3, games=4):
    dirs = []
    for gen in range(n):
        d = tmp_path / f"gen{gen}"
        run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")],
                     range(gen * 100, gen * 100 + games), d)
        dirs.append(d)
    return dirs


def test_generation_caps_and_weights(tmp_path):
    from yomibench import train

    dirs = _generations(tmp_path)
    data = train.load_samples(dirs, Vocab(), 0.2, 0, workers=1, games_per_gen=2, gen_weights=[0.25, 0.5, 1.0])
    gens = list(data["train"]["gen"]) + list(data["val"]["gen"])
    assert data["train"]["games"] + data["val"]["games"] == 6  # two records from each generation
    assert set(gens) == {0, 1, 2}
    assert list(data["train"]["w"]) == [[0.25, 0.5, 1.0][int(g)] for g in data["train"]["gen"]]
    again = train.load_samples(dirs, Vocab(), 0.2, 0, workers=1, games_per_gen=2, gen_weights=[0.25, 0.5, 1.0])
    assert (again["train"]["y"] == data["train"]["y"]).all()  # the sample is seeded
    full = train.load_samples(dirs, Vocab(), 0.2, 0, workers=1)
    assert full["train"]["games"] + full["val"]["games"] == 12 and (full["train"]["w"] == 1).all()
    with pytest.raises(SystemExit, match="2 generation weights for 3 folders"):
        train.load_samples(dirs, Vocab(), 0.2, 0, workers=1, gen_weights=[1.0, 1.0])


def test_train_on_a_window_of_the_last_generations(tmp_path, capsys):
    import numpy as np

    from yomibench import train

    dirs = _generations(tmp_path, n=4, games=3)
    out = tmp_path / "w.npz"
    train.main([*map(str, dirs), "--window", "3", "--games-per-gen", "2", "--gen-weights", "0.5,1,2",
                "--out", str(out), "--epochs", "1", "--hidden", "8", "--workers", "1", "--val-fraction", "0.5"])
    text = capsys.readouterr().out
    meta = __import__("json").loads(str(np.load(out)["meta"]))
    assert meta["sources"] == [str(d) for d in dirs[1:]]  # the oldest generation is outside the window
    assert meta["games_per_gen"] == 2 and meta["gen_weights"] == [0.5, 1.0, 2.0]
    assert meta["train_games"] + meta["val_games"] == 6
    assert any(line.strip().startswith("gen gen") for line in text.splitlines())  # validation by generation
    with pytest.raises(SystemExit):
        train.main([*map(str, dirs), "--window", "2", "--gen-weights", "1,1,1", "--out", str(out)])


def _add_v3_move_fields(folder):
    """Make every recorded state in `folder` look like an engine-v3 state (move fields present)."""
    import gzip
    import json

    for path in folder.glob("g*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        for line in lines[1:-1]:
            for p in line["state"]["players"]:
                p["hitboxes"] = []
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("".join(json.dumps(line) + "\n" for line in lines))


def test_v3_state_fraction_counts_every_included_generation(tmp_path):
    import json

    import numpy as np

    from yomibench import train

    old, new = tmp_path / "old", tmp_path / "new"
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(4), old)
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30, search="oracle"), [("Toy", "Toy")],
                 range(100, 104), new, opening=(2, 1.0, 1.0))  # some positions skipped
    _add_v3_move_fields(new)
    data = train.load_samples([old, new], Vocab(), 0.25, 0, workers=1)
    for split in ("train", "val"):
        gens = data[split]["gen"]
        assert data[split]["v3_fraction"] == pytest.approx(float(np.mean(gens == 1)))  # over the samples kept
        assert data[split]["v3_fraction_by_gen"] == [0.0, 1.0]
    out = tmp_path / "m.npz"
    train.main([str(old), str(new), "--out", str(out), "--epochs", "1", "--hidden", "8", "--workers", "1",
                "--val-fraction", "0.25"])
    meta = json.loads(str(np.load(out)["meta"]))
    assert meta["state_v3_fraction"] == pytest.approx(data["train"]["v3_fraction"])
    train.main([str(old), str(new), "--window", "1", "--out", str(out), "--epochs", "1", "--hidden", "8",
                "--workers", "1"])
    assert json.loads(str(np.load(out)["meta"]))["state_v3_fraction"] == 1.0  # only the newest generation


def test_validated_games_never_drift_into_training(tmp_path):
    """The split is the folder's frozen manifest, not a reshuffle: a later run with more games and
    another --seed and --val-fraction still validates on every game validated before."""
    from yomibench import train

    d = tmp_path / "gen"
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(6), d)
    first = train.load_samples([d], Vocab(), 0.3, 0, workers=1)
    _, val1 = _split_files([d])
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(6, 12), d)
    second = train.load_samples([d], Vocab(), 0.6, 5, workers=1)
    train2, val2 = _split_files([d])
    assert set(val1) <= set(val2) and not set(val1) & set(train2)
    assert second["heldout"][str(d)]["added"] == 6 and first["heldout"][str(d)]["added"] == 6
    assert second["heldout"][str(d)]["sha256"] != first["heldout"][str(d)]["sha256"]
    assert second["train"]["games"] == len(train2) and second["val"]["games"] == len(val2)


def test_eval_dir_games_are_scored_but_never_trained_on(tmp_path, capsys):
    import json

    import numpy as np

    from yomibench import train

    gen, ev, root = tmp_path / "gen", tmp_path / "eval", tmp_path / "manifests"
    # 8 games, not 4: the held-out split hashes record names (which hash wall-clock content), so with
    # 4 games at 0.25 all of them landed in validation about 0.4% of runs and training had no rows.
    _explored_games(gen, (2, 1.0, 1.0), (0.3, 0.1), seeds=range(8))
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30, search="oracle"), [("Toy", "Toy")],
                 range(50, 53), ev)  # exploration off
    data = train.load_samples([gen], Vocab(), 0.25, 0, workers=1, eval_dirs=[ev], heldout_root=root)
    assert data["eval"]["games"] == 3 and (data["eval"]["gen"] == -1).all()
    assert len(data["eval"]["outcome"]) == sum(len(read_slim(p)[2]) for p in ev.glob("g*.jsonl.gz"))
    assert not (gen / ".heldout.json").exists() and not list(ev.glob(".heldout*"))  # eval needs no manifest
    out, metrics = tmp_path / "m.npz", tmp_path / "ledger.jsonl"
    train.main([str(gen), "--eval-dir", str(ev), "--heldout-root", str(root), "--out", str(out), "--epochs", "2",
                "--hidden", "8", "--workers", "1", "--ledger", str(metrics)])
    text = capsys.readouterr().out
    assert "eval mse" in text and "evaluation (--eval-dir, never trained on)" in text
    meta = json.loads(str(np.load(out)["meta"]))
    assert meta["eval_games"] == 3 and meta["eval_mse"] is not None and meta["eval_dirs"] == [str(ev)]
    assert meta["heldout"][str(gen)]["sha256"] == data["heldout"][str(gen)]["sha256"]
    (row,) = [json.loads(x) for x in metrics.read_text().splitlines()]
    m, p = row["metrics"], row["provenance"]
    assert m["eval_games"] == 3 and m["eval_mse"] == meta["eval_mse"] and m["eval_phases"]["all"]["n"] > 0
    assert p["heldout"][str(gen)]["sha256"] == data["heldout"][str(gen)]["sha256"]
    assert p["eval_set"]["games_per_dir"] == {str(ev): 3}
    with pytest.raises(SystemExit, match="exploratory picks"):  # explored games can't be evaluation games
        train.load_samples([ev], Vocab(), 0.25, 0, workers=1, eval_dirs=[gen], heldout_root=root)
    with pytest.raises(SystemExit, match="also a training folder"):
        train.load_samples([gen], Vocab(), 0.25, 0, workers=1, eval_dirs=[gen], heldout_root=root)


def test_outcome_target_coverage_by_phase(tmp_path):
    from yomibench import train

    _explored_games(tmp_path, (3, 1.0, 1.0), (0.3, 0.1))  # opening noise: early decisions lose outcome targets
    for mode, check in (("none", lambda c: all(v == 1.0 for v in c.values())),
                        ("uniform-only", lambda c: c["early"] < c["late"] == 1.0 and 0 < c["all"] < 1),
                        ("all", lambda c: c["all"] <= cov_uniform["all"])):
        data = train.load_samples([tmp_path], Vocab(), 0.25, 0, workers=1, explore_targets=mode)
        cov = data["train"]["outcome_coverage"]
        assert set(cov) == {"early", "mid", "late", "all"} and check(cov), (mode, cov)
        if mode == "uniform-only":
            cov_uniform = cov
            train_files, _ = _split_files([tmp_path])
            decisions = sum(len(read_slim(p)[2]) for p in train_files)
            assert cov["all"] == round((len(data["train"]["y"]) - data["train"]["value_only"]) / decisions, 4)


@pytest.mark.parametrize("search", ["oracle", "full"])
def test_slim_and_full_records_of_one_game_train_identically(tmp_path, search):
    """The same game recorded with and without previews: identical read_slim output, and identical samples
    from the in-memory loader and from the packed training cache."""
    import numpy as np

    from yomibench import train
    from yomibench.selfplay import RecordWriter, play_recorded, read_record

    volatile = {"ms", "previewed", "seconds", "perf", "started", "previews"}  # timings, and the dropped payload

    def strip(d):
        return {k: strip(v) if isinstance(v, dict) else v for k, v in d.items() if k not in volatile}
    for seed in range(3):
        dirs = {}
        for full in (True, False):
            d = dirs[full] = tmp_path / f"{search}{seed}-{'full' if full else 'slim'}"
            d.mkdir()
            engine = connect("toy")  # a fresh engine: the same game both times
            writer = RecordWriter(d, seed)
            play_recorded(engine, MatrixBot(iterations=100, search=search), ("Toy", "Toy"), seed, engine.hello(),
                          full_previews=full, out=writer, opening=(2, 1.0, 0.5), explore_cfg=(0.3, 0.1))
            writer.finish()
        (full_path,), (slim_path,) = (sorted(dirs[f].glob("g*.jsonl.gz")) for f in (True, False))
        full_lines, slim_lines = read_record(full_path), read_record(slim_path)
        assert full_lines[0]["previews"] == "full" and slim_lines[0]["previews"] == "slim"
        assert any("previewed" in ln for ln in full_lines[1:-1]) and not any("previewed" in ln for ln in slim_lines)
        assert [strip(ln) for ln in full_lines] == [strip(ln) for ln in slim_lines]
        assert any(ln.get("candidate_values") for ln in slim_lines[1:-1])
        (h_full, r_full, t_full), (h_slim, r_slim, t_slim) = read_slim(full_path), read_slim(slim_path)
        assert t_full == t_slim and strip(h_full) == strip(h_slim) and strip(r_full) == strip(r_slim)
        # A held-out fraction near 0: a lone game is never held out, so both folders train on it.
        got = [train.load_samples([dirs[f]], Vocab(), 1e-9, 0, workers=1)["train"] for f in (True, False)]
        assert got[0]["games"] == got[1]["games"] == 1  # one game per folder: all training
        for k in train.DTYPES:
            assert np.array_equal(got[0][k], got[1][k]), k
        # The packed training cache (yomibench.cache), built from each record: identical samples too.
        from yomibench import cache

        packed = [cache.open_cached([dirs[f]], Vocab(), 1e-9, 0, workers=1, root=tmp_path / "cache",
                                    log=lambda *a: None)["train"] for f in (True, False)]
        assert packed[0]["games"] == packed[1]["games"] == 1
        for key in ("explored", "value_only", "v3_fraction", "outcome_coverage"):
            assert packed[0][key] == packed[1][key], key
        for k in ("nums0", "nums1", "cats0", "cats1"):  # memory-mapped columns
            assert np.array_equal(packed[0][k][0:len(packed[0][k])], packed[1][k][0:len(packed[1][k])]), k
        for k in ("y", "w", "gen", "outcome", "phase", "baseline", "matchup"):
            assert np.array_equal(np.asarray(packed[0][k]), np.asarray(packed[1][k])), k
        assert len(packed[0]["y"]) == len(got[0]["y"]) > 0  # and the same rows as the in-memory loader


def test_quick_turn_targets_next_full_blend_and_full_turns_only():
    """--cheap-target next-full blends a quick turn toward the next full turn's search value (outcome
    alone after the last full turn); --cheap-value-weight sets that blend; --full-turns-only drops
    quick turns; validation and evaluation keep the default targets either way."""
    from yomibench.selfplay import SlimTurn
    from yomibench.train import select_rows

    def t(v, cheap):
        return SlimTurn(None, v, (False, False), None, None, cheap)
    turns = [t(0.4, False), t(0.9, True), t(0.1, True), t(-0.2, False), t(0.7, True)]
    ys = lambda rows: {k: round(y, 6) for k, y, _ in rows}  # noqa: E731
    default, _ = select_rows(turns, "train", "uniform-only", 0.5, 1.0)
    assert ys(default) == {0: 0.7, 1: 1.0, 2: 1.0, 3: 0.4, 4: 1.0}  # quick turns: outcome alone
    nf, _ = select_rows(turns, "train", "uniform-only", 0.5, 1.0, cheap_target="next-full")
    assert ys(nf) == {0: 0.7, 1: 0.4, 2: 0.4, 3: 0.4, 4: 1.0}  # toward turn 3's -0.2; none after turn 4
    nf75, _ = select_rows(turns, "train", "uniform-only", 0.5, 1.0, cheap_target="next-full", cheap_value_weight=0.75)
    assert ys(nf75)[1] == 0.1 and ys(nf75)[0] == 0.7  # 0.25 x 1 + 0.75 x -0.2; full turns keep --value-weight
    full, _ = select_rows(turns, "train", "uniform-only", 0.5, 1.0, full_turns_only=True)
    assert sorted(ys(full)) == [0, 3]
    for split in ("val", "eval"):  # unchanged by either option
        a, _ = select_rows(turns, split, "uniform-only", 0.5, 1.0)
        b, _ = select_rows(turns, split, "uniform-only", 0.5, 1.0, cheap_target="next-full", full_turns_only=False)
        c, _ = select_rows(turns, split, "uniform-only", 0.5, 1.0, full_turns_only=True)
        assert ys(a) == ys(b) == ys(c)
    with pytest.raises(ValueError):
        select_rows(turns, "train", "uniform-only", 0.5, 1.0, cheap_target="nope")


@pytest.mark.parametrize("cheap", [{"cheap_target": "next-full"}, {"cheap_target": "next-full", "cheap_value_weight": 0.75},
                                   {"full_turns_only": True}])
def test_quick_turn_options_give_the_cache_and_the_in_memory_loader_the_same_targets(tmp_path, cheap):
    import numpy as np

    from yomibench import train
    from yomibench.cache import open_cached

    d = tmp_path / "g"
    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(6), d,
                 playout_cap=(0.5, 2))
    assert any(t.cheap for p in d.glob("g*.jsonl.gz") for t in read_slim(p)[2])  # the games have quick turns
    kw = dict(val_fraction=0.3, seed=0, value_weight=0.5, workers=1, explore_targets="uniform-only")
    mem = train.load_samples([d], Vocab(), quick_turns=cheap, **kw)
    cac = open_cached([d], Vocab(), root=tmp_path / "cache", quick_turns=cheap, **kw)
    base = train.load_samples([d], Vocab(), **kw)
    for split in ("train", "val"):
        assert np.allclose(np.asarray(mem[split]["y"]), np.asarray(cac[split]["y"]), atol=1e-6)
    assert np.allclose(np.asarray(mem["val"]["y"]), np.asarray(base["val"]["y"]))  # validation unchanged
    if cheap.get("full_turns_only"):
        assert len(mem["train"]["y"]) < len(base["train"]["y"])
    else:  # (the toy's games are draws with zero search values, so the blend's targets equal the default's;
        # test_quick_turn_targets_next_full_blend_and_full_turns_only checks the blend itself)
        assert len(mem["train"]["y"]) == len(base["train"]["y"])


def test_train_records_the_quick_turn_settings_and_refuses_contradictions(tmp_path):
    import json

    import numpy as np

    from yomibench import train

    d = tmp_path / "g"
    run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(4), d,
                 playout_cap=(0.5, 2))
    out = tmp_path / "m.npz"
    led = tmp_path / "ledger.jsonl"
    train.main([str(d), "--out", str(out), "--epochs", "1", "--hidden", "8", "--workers", "1", "--ledger", str(led),
                "--cache", str(tmp_path / "cache"), "--cheap-target", "next-full", "--cheap-value-weight", "0.75"])
    meta = json.loads(str(np.load(out, allow_pickle=True)["meta"]))
    assert (meta["cheap_target"], meta["cheap_value_weight"], meta["full_turns_only"]) == ("next-full", 0.75, False)
    (row,) = [json.loads(x) for x in led.read_text().splitlines()]
    tb = row["provenance"]["target_blend"]
    assert (tb["cheap_target"], tb["cheap_value_weight"], tb["full_turns_only"]) == ("next-full", 0.75, False)
    for bad in (["--full-turns-only", "--cheap-target", "next-full"], ["--cheap-value-weight", "0.5"],
                ["--cheap-target", "next-full", "--cheap-value-weight", "1.5"]):
        with pytest.raises(SystemExit):
            train.main([str(d), "--out", str(tmp_path / "x.npz"), "--no-ledger", *bad])
