"""The frozen held-out split: assigned once per game file, never reassigned, appended for new games."""
import json

import pytest

from yomibench import heldout


def _games(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / f"g{n:012d}-abc.jsonl.gz").write_bytes(b"")


def test_assignment_is_frozen_and_new_games_are_appended(tmp_path):
    d = tmp_path / "gen"
    _games(d, range(40))
    tr, va, info = heldout.split(d, 0.25, seed=3)
    m = json.loads((d / ".heldout.json").read_text())
    assert (m["seed"], m["fraction"]) == (3, 0.25) and info["added"] == 40
    assert {p.name for p in va} == set(m["heldout"]) and len(tr) + len(va) == 40 and 0 < len(va) < 40
    # A later run asking for another fraction and seed changes nothing already assigned.
    tr2, va2, info2 = heldout.split(d, 0.9, seed=99)
    assert (tr2, va2) == (tr, va) and info2["added"] == 0 and info2["sha256"] == info["sha256"]
    # New games are assigned once, with the manifest's own seed and fraction, and appended.
    _games(d, range(40, 80))
    tr3, va3, info3 = heldout.split(d, 0.9, seed=99)
    assert set(va) <= set(va3) and set(tr) <= set(tr3) and info3["added"] == 40 and len(va3) < 40
    m3 = json.loads((d / ".heldout.json").read_text())
    assert (m3["seed"], m3["fraction"]) == (3, 0.25) and info3["sha256"] != info["sha256"]
    # A game that disappears and comes back keeps its side.
    gone = va3[0]
    gone.unlink()
    assert gone not in heldout.split(d)[1]
    gone.write_bytes(b"")
    assert gone in heldout.split(d)[1]


def test_a_small_folder_still_holds_one_game_out(tmp_path):
    for seed in range(20):
        d = tmp_path / f"s{seed}"
        _games(d, range(3))
        tr, va, _ = heldout.split(d, 0.01, seed=seed)
        assert len(va) == 1 and len(tr) == 2
    d = tmp_path / "one"
    _games(d, [0])
    assert heldout.split(d, 0.01)[1] == []  # one game: nothing to hold out yet


def test_manifests_can_live_outside_read_only_folders(tmp_path):
    a, b, root = tmp_path / "x" / "gen1", tmp_path / "y" / "gen1", tmp_path / "manifests"
    _games(a, range(10))
    _games(b, range(10))
    ia, ib = heldout.split(a, root=root)[2], heldout.split(b, root=root)[2]
    assert not (a / ".heldout.json").exists() and ia["manifest"] != ib["manifest"]  # same name, own manifests
    assert {p.parent for p in map(__import__("pathlib").Path, (ia["manifest"], ib["manifest"]))} == {root}
    with pytest.raises(SystemExit, match="between 0 and 1"):
        heldout.split(tmp_path / "z", 1.5)
