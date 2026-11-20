"""Human-play games are a held-out human test: training, the training cache, gate eval sets and the
round-3 trainer's collection never take them (yomibench.compat's held-out guards)."""

from __future__ import annotations

import gzip
import io
import json
import tarfile

import pytest

from yomibench import compat, evalset

HUMAN = {"type": "header", "arm": "human-play", "train": False, "seed": 7, "engine": {"candidate_set": "4"}}
SELF = {"seed": 8, "engine": {"candidate_set": "4"}}


def gz(*lines):
    buf = io.BytesIO()
    with gzip.open(buf, "wt", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return buf.getvalue()


def test_held_out_headers_and_folders(tmp_path):
    assert compat.held_out_arm(HUMAN) == "human-play" and compat.held_out_arm(SELF) is None
    assert compat.held_out_gz(gz(HUMAN, {"turn": 0})) == "human-play" and compat.held_out_gz(gz(SELF)) is None
    assert compat.held_out_gz(b"not gzip") is None
    assert compat.held_out_folder(tmp_path) is None
    compat.mark_held_out(tmp_path / "human-play", "human-play")
    assert compat.held_out_folder(tmp_path / "human-play" / "games" / "x")  # the marker covers subfolders
    assert compat.held_out_folder(tmp_path) is None
    with pytest.raises(SystemExit, match="held-out"):
        compat.refuse_held_out([("a", SELF), ("b", HUMAN)])
    compat.refuse_held_out([("a", SELF)])


def test_eval_sets_from_gate_tars_skip_human_play_records(tmp_path):
    tar = tmp_path / "r3-gate-x-s0-records.tar"
    with tarfile.open(tar, "w") as tf:
        for name, data in (("g000000000007-aa.jsonl.gz", gz(HUMAN)), ("g000000000008-bb.jsonl.gz", gz(SELF))):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    assert evalset.freeze([tar], tmp_path / "eval", 2) is None  # only one ordinary record: too few
    m = evalset.freeze([tar], tmp_path / "eval", 1)
    assert list(m["records"]) == ["g000000000008-bb.jsonl.gz"]


def test_the_training_cache_refuses_a_human_play_record(tmp_path):
    pytest.importorskip("numpy")
    from yomibench import cache

    path = tmp_path / "g000000000007-aa.jsonl.gz"
    path.write_bytes(gz(HUMAN, {"result": {"winner": 0}}))
    with pytest.raises(ValueError, match="held-out"):
        cache.encode_record(str(path))


def test_training_refuses_human_play_folders_and_records(tmp_path):
    pytest.importorskip("torch")
    from yomibench import train
    from yomibench.features import Vocab

    marked = tmp_path / "human-play" / "games"
    marked.mkdir(parents=True)
    compat.mark_held_out(tmp_path / "human-play", "human-play")
    with pytest.raises(SystemExit, match="held out"):
        train.select_files([marked], 0.15, 0, None, [], None, 0)
    plain = tmp_path / "records"
    plain.mkdir()
    for name, header in (("g000000000007-aa.jsonl.gz", HUMAN), ("g000000000008-bb.jsonl.gz", SELF)):
        (plain / name).write_bytes(gz(header))
    with pytest.raises(SystemExit, match="held-out record"):
        train.load_samples([plain], Vocab(), 0.15, 0, workers=1)
