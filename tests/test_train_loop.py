"""tools/train_loop.py: cycles run their stages in order, a promote writes best.json (generation, model, sha256),
and a rerun resumes at the next unfinished cycle without replaying finished stages."""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_loop", ROOT / "tools" / "train_loop.py")
tl = importlib.util.module_from_spec(spec)
sys.modules["train_loop"] = tl
spec.loader.exec_module(tl)


def fake_stages(monkeypatch, gate_codes):
    calls = []

    def call(cmd, **_kw):
        args = [str(c) for c in cmd]
        if "selfplay" in args:
            out = Path(args[args.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            for i in range(int(args[args.index("--games") + 1])):
                (out / f"g{i}.jsonl.gz").write_bytes(b"")
            calls.append("selfplay")
        elif "yomibench.train" in args:
            Path(args[args.index("--out") + 1]).write_bytes(f"net {len(calls)}".encode())
            calls.append("train")
        else:
            calls.append("gate")
            return gate_codes.pop(0)
        return 0

    monkeypatch.setattr(tl.subprocess, "call", call)
    return calls


def test_cycles_promote_and_resume(tmp_path, monkeypatch):
    calls = fake_stages(monkeypatch, [tl.PROMOTE, tl.REJECT])
    assert tl.main(["--out", str(tmp_path), "--engine", "toy", "--characters", "Toy", "--games-per-cycle", "3",
                    "--cycles", "2"]) == 0
    assert calls == ["selfplay", "train", "gate"] * 2
    best = json.loads((tmp_path / "best.json").read_text())
    assert (best["generation"], best["cycle"]) == (1, 1)
    assert Path(best["model"]).name == f"{best['sha256']}.npz" and Path(best["model"]).exists()
    # rerun with a higher cap: cycles 1-2 are done, cycle 3 plays with the generation-1 net and promotes it to 2
    calls[:] = []
    fake_stages(monkeypatch, [tl.PROMOTE])
    assert tl.main(["--out", str(tmp_path), "--engine", "toy", "--characters", "Toy", "--games-per-cycle", "3",
                    "--cycles", "3"]) == 0
    assert json.loads((tmp_path / "best.json").read_text())["generation"] == 2
    assert not (tmp_path / "cycle-4").exists()


def test_an_interrupted_cycle_skips_its_finished_stages(tmp_path, monkeypatch):
    calls = fake_stages(monkeypatch, [tl.CLOSE])
    c1 = tmp_path / "cycle-1"
    (c1 / "selfplay").mkdir(parents=True)
    for i in range(3):
        (c1 / "selfplay" / f"g{i}.jsonl.gz").write_bytes(b"")
    (c1 / "model.npz").write_bytes(b"trained before the interruption")
    assert tl.main(["--out", str(tmp_path), "--engine", "toy", "--characters", "Toy", "--games-per-cycle", "3",
                    "--cycles", "1"]) == 0
    assert calls == ["gate"] and not (tmp_path / "best.json").exists()


def test_a_promotion_is_not_repeated_after_a_restart(tmp_path, monkeypatch):
    calls = fake_stages(monkeypatch, [tl.PROMOTE])
    args = ["--out", str(tmp_path), "--engine", "toy", "--characters", "Toy", "--games-per-cycle", "3", "--cycles", "1"]
    assert tl.main(args) == 0
    (tmp_path / "cycle-1" / "done").unlink()  # interrupted after the promotion, before the cycle was marked done
    calls[:] = []
    fake_stages(monkeypatch, [])
    assert tl.main(args) == 0
    best = json.loads((tmp_path / "best.json").read_text())
    assert (best["generation"], best["cycle"]) == (1, 1) and calls == []
