import gzip
import json
import tarfile

from yomibench.bots import MatrixBot
from yomibench.engine import connect
from yomibench.selfplay import run_selfplay

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("coverage_report", Path(__file__).parents[1] / "tools/coverage_report.py")
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)


def test_coverage_counts_offered_and_picked_moves_per_character(tmp_path):
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=20), [("Toy", "Toy")], range(3), tmp_path / "r")
    cov = cr.coverage(cr._records([str(tmp_path / "r")]))
    toy = cov["characters"]["Toy"]
    assert cov["games"] == 3 and toy["turns"] > 0
    picked = sum(m.get("picked", 0) for m in toy["moves"].values())
    assert picked == toy["turns"]  # one pick per actionable seat-turn
    assert all(m.get("offered", 0) >= m.get("picked", 0) for m in toy["moves"].values())
    # the same counts from a chunk tar
    tar = tmp_path / "c.tar"
    with tarfile.open(tar, "w") as tf:
        for f in sorted((tmp_path / "r").glob("g*.jsonl.gz")):
            tf.add(f, arcname=f.name)
    assert cr.coverage(cr._records([str(tar)])) == cov
    assert "== Toy" in cr.report(cov) and "offered but never picked" in cr.report(cov)
