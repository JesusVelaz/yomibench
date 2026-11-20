"""Train a bot on one PC: self-play, train, gate against the current best, promote. Repeats; resumable.

    python tools/train_loop.py --out ~/.yomibench/train [--engine game-pool:auto] [--cycles 0]

Each cycle K (OUT/cycle-K/) runs four stages, each skipped when its output exists, so rerunning the same command
resumes where it stopped:
  1. selfplay  --games-per-cycle games by the current best (plain matrix-do search before the first promotion), on
               --engine (default game-pool:auto: headless copies of your own game, sized to this PC) and candidate
               set --candidate-set -> cycle-K/selfplay/
  2. train     a value net on every cycle's records so far -> cycle-K/model.npz
  3. gate      tools/pool_eval.py --quick: the new net against the current best, seat-swapped pairs over every
               pairing of --characters (--pairs each) -> cycle-K/gate.jsonl (exit 0 promote, 1 reject, 2 too close)
  4. promote   on a promote: the net is copied to OUT/models/<sha256>.npz and OUT/best.json names it (generation,
               model, sha256, cycle). Human play reads best.json too (YOMIBENCH_BEST or ~/.yomibench/best.json).
--cycles N stops after N cycles (0: keep going). Speed on one PC is tens to low hundreds of games an hour; starting
from published weights (--start-from MODEL.npz) skips the slow first cycles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TRAIN = ["--hidden", "64", "--weight-decay", "0.01", "--lr", "3e-4", "--value-weight", "0.5"]
PROMOTE, REJECT, CLOSE = 0, 1, 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_best(out: Path) -> dict | None:
    p = out / "best.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_best(out: Path, best: dict) -> None:
    tmp = out / "best.json.tmp"
    tmp.write_text(json.dumps(best, indent=1), encoding="utf-8")
    tmp.replace(out / "best.json")


def promote(out: Path, model: Path, cycle: int, prev: dict | None) -> dict:
    sha = sha256(model)
    dst = out / "models" / f"{sha}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(model, dst)
    best = {"generation": (prev or {}).get("generation", 0) + 1, "model": str(dst), "sha256": sha, "cycle": cycle,
            "promoted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_best(out, best)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path.home() / ".yomibench" / "train")
    ap.add_argument("--engine", default="game-pool:auto", help="engine spec for self-play and the gate")
    ap.add_argument("--candidate-set", type=int, default=6)
    ap.add_argument("--characters", default="Ninja,Cowboy,Wizard,Mutant,Robot")
    ap.add_argument("--games-per-cycle", type=int, default=200)
    ap.add_argument("--pairs", type=int, default=30, help="gate pairs per pairing")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--cycles", type=int, default=0, help="stop after this many cycles (0: keep going)")
    ap.add_argument("--procs", type=int, default=1, help="self-play and gate processes")
    ap.add_argument("--start-from", type=Path, help="a model .npz to start as generation 1 (e.g. published weights)")
    ap.add_argument("--first-seed", type=int, default=1_000_000)
    ap.add_argument("--below-normal", action="store_true", help="run every stage at below-normal priority (Windows)")
    a = ap.parse_args(argv)
    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "ledger.jsonl"
    chars = [c.strip() for c in a.characters.split(",") if c.strip()]
    matchups = ",".join(f"{x}:{y}" for x in chars for y in chars)
    flags = 0x00004000 if a.below_normal and sys.platform == "win32" else 0
    py = sys.executable
    # Legality from the game's own action panel, whatever else this checkout carries.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO / "src"), os.environ.get("PYTHONPATH")])),
           "YOMIBENCH_SERVER_ARGS": os.environ.get("YOMIBENCH_SERVER_ARGS", "--yomi-legality=game"),
           "YOMIBENCH_NO_BUILD_WARNING": "1"}  # warned once, here, not again by every stage
    if a.engine.startswith("game"):
        sys.path.insert(0, str(REPO / "src"))
        from yomibench import game

        try:
            game.warn_if_unsupported(game.find_game_dir())
        except FileNotFoundError:
            pass  # self-play says where it looked

    def run(stage, cmd, ok=(0,)):
        print(f"=== {stage}: {' '.join(map(str, cmd))}", flush=True)
        code = subprocess.call([str(c) for c in cmd], cwd=REPO, env=env, creationflags=flags)
        if code not in ok:
            raise SystemExit(f"{stage} failed (exit {code}); fix it and rerun: finished stages are skipped")
        return code

    if a.start_from and read_best(out) is None:
        print(f"generation 1: {promote(out, a.start_from, 0, None)['sha256'][:12]} (from {a.start_from})")
    cycle = 1
    while (out / f"cycle-{cycle}" / "done").exists():
        cycle += 1
    while a.cycles == 0 or cycle <= a.cycles:
        cdir = out / f"cycle-{cycle}"
        records, model, gate = cdir / "selfplay", cdir / "model.npz", cdir / "gate.jsonl"
        best = read_best(out)
        bot = f"matrix-do:{best['model']}" if best else "matrix-do"
        tags = ["--ledger", ledger, "--tag", "experiment=train-loop", "--tag", f"cycle={cycle}"]
        done = len(list(records.glob("g*.jsonl.gz"))) if records.exists() else 0
        if done < a.games_per_cycle:
            run("selfplay", [py, "-m", "yomibench.cli", "selfplay", "--engine", a.engine, "--bot", bot,
                             "--matchups", matchups, "--games", a.games_per_cycle,
                             "--first-seed", a.first_seed + cycle * 100_000, "--procs", a.procs, "--out", records,
                             "--candidate-set", a.candidate_set, *tags, "--tag", "stage=selfplay"])
        if not model.exists():
            dirs = [d / "selfplay" for d in sorted(out.glob("cycle-*")) if (d / "selfplay").exists()]
            run("train", [py, "-m", "yomibench.train", *dirs, "--out", model, *TRAIN, "--epochs", a.epochs,
                          "--heldout-root", out / "heldout", "--cache", out / "cache", "--name", f"cycle-{cycle}",
                          *tags, "--tag", "stage=train"])
        decided = cdir / "gate.exit"
        if decided.exists():
            code = int(decided.read_text())
        else:
            gate.unlink(missing_ok=True)  # an interrupted gate replays whole
            code = run("gate", [py, REPO / "tools" / "pool_eval.py", "--quick", "--candidate", f"matrix-do:{model}",
                                "--previous-best", bot, "--characters", ",".join(chars), "--pairs", a.pairs,
                                "--engine", a.engine, "--candidate-set", a.candidate_set, "--procs", a.procs,
                                "--results", gate, *tags, "--tag", "stage=gate"], ok=(PROMOTE, REJECT, CLOSE))
            decided.write_text(str(code))
        if code == PROMOTE and best and best.get("cycle") == cycle:  # promoted before a restart: not again
            print(f"cycle {cycle}: already promoted, generation {best['generation']}", flush=True)
        elif code == PROMOTE:
            best = promote(out, model, cycle, best)
            print(f"cycle {cycle}: promoted, generation {best['generation']} ({best['sha256'][:12]})", flush=True)
        else:
            print(f"cycle {cycle}: {'rejected' if code == REJECT else 'too close to call'}; the best stays", flush=True)
        (cdir / "done").write_text(json.dumps({"gate_exit": code}), encoding="utf-8")
        cycle += 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
