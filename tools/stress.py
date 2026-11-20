"""Stress checks for the real-game server: matchups, soak, scaling.

    uv run python tools/stress.py matchups --processes 8 --games 2 --out runs/matchups.json
    uv run python tools/stress.py soak --processes 8 --minutes 65 --out runs/soak.json
    uv run python tools/stress.py scaling --counts 1,4,8,12,16 --seconds 60 --out runs/scaling.json

All start their own servers (tools/serverpool.py) and play seeded random/greedy games.

- matchups: every ordered pair of the five characters (25, mirrors included). Each turn previews
  the pair about to be played (detail) and compares it with the real step. A game "halts" when the
  server crashes, answers with an error, or the game doesn't end within --max-turns.
- soak: N servers in continuous play (random play with 64-pair preview batches and a `continue`
  rollout every few turns) for M minutes; samples working set, engine object/node counts, crashes.
- scaling: previews/s at several process counts, with process and system CPU utilization.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serverpool import Pool  # noqa: E402

from yomibench.engine import EngineCrashed  # noqa: E402

CHARS = ["Ninja", "Cowboy", "Wizard", "Robot", "Mutant"]


def req(engine, msg: dict) -> dict:
    return engine.transport.request(msg)


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def system_cpu_times() -> tuple[float, float]:
    """(idle, total) seconds across all logical CPUs (Windows)."""
    from ctypes import wintypes

    idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
    ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7  # noqa: E731
    return ft(idle), ft(kernel) + ft(user)   # kernel time includes idle


# --- matchups --------------------------------------------------------------------------------------

def play_checked(engine, c1: str, c2: str, seed: int, rng: random.Random, max_turns: int) -> dict:
    """One random (p1) vs greedy (p2) game; every played pair is previewed first and compared."""
    r = {"turns": 0, "divergences": 0, "examples": [], "done": False, "winner": None, "error": None}
    obs = req(engine, {"cmd": "new_match", "p1": c1, "p2": c2, "seed": seed})
    if "error" in obs:
        r["error"] = obs["error"]
        return r
    while not obs["done"] and r["turns"] < max_turns:
        a = rng.randrange(len(obs["candidates"][0]))
        # Greedy p2: preview each of its moves against p1's default, take the best hp swing.
        n2 = len(obs["candidates"][1])
        res = req(engine, {"cmd": "preview", "pairs": [[0, j] for j in range(n2)], "candidates": False})
        if "error" in res:
            r["error"] = res["error"]
            return r
        hp = [p["hp"] for p in obs["state"]["players"]]
        def gain(x):  # noqa: E306
            pl = x["state"]["players"]
            return (hp[0] - pl[0]["hp"]) - (hp[1] - pl[1]["hp"])
        scores = [gain(x) for x in res["results"]]
        best = max(scores)
        b = rng.choice([j for j, s in enumerate(scores) if s == best]) if rng.random() < 0.8 else rng.randrange(n2)
        pv = req(engine, {"cmd": "preview", "pairs": [[a, b]], "candidates": False, "detail": True})
        nxt = req(engine, {"cmd": "step", "actions": [a, b], "detail": True})
        if "error" in nxt or "error" in pv:
            r["error"] = nxt.get("error") or pv.get("error")
            return r
        d = diff(pv["results"][0]["detail"], nxt["detail"])
        if d:
            r["divergences"] += 1
            if len(r["examples"]) < 3:
                r["examples"].append({"turn": obs["turn"], "moves": [obs["candidates"][0][a]["action"],
                                      obs["candidates"][1][b]["action"]], "diff": d[:6]})
        r["turns"] += 1
        obs = nxt
    r["done"] = obs["done"]
    r["winner"] = obs.get("winner")
    return r


def cmd_matchups(args: argparse.Namespace) -> None:
    jobs = [(c1, c2, g) for c1 in CHARS for c2 in CHARS for g in range(args.games)]
    results: dict[str, dict] = {}
    lock = threading.Lock()
    with Pool(args.processes, base_port=args.base_port) as pool:
        def work(i: int) -> None:
            engine = pool.engine(i, timeout=300)
            rng = random.Random(i)
            while True:
                with lock:
                    if not jobs:
                        break
                    c1, c2, g = jobs.pop(0)
                t0 = time.monotonic()
                try:
                    r = play_checked(engine, c1, c2, 5000 + g * 31 + CHARS.index(c1) * 5 + CHARS.index(c2), rng, args.max_turns)
                except (EngineCrashed, OSError, ValueError) as e:
                    r = {"turns": 0, "divergences": 0, "examples": [], "done": False, "error": f"crash: {e}",
                         "exit_code": pool.exit_code(i)}
                    pool.restart(i)
                    engine = pool.engine(i, timeout=300)
                r["seconds"] = round(time.monotonic() - t0, 1)
                key = f"{c1} v {c2}"
                with lock:
                    results.setdefault(key, {"games": []})["games"].append(r)
                print(key, g, {k: r[k] for k in ("turns", "divergences", "done", "error", "seconds")}, flush=True)
            engine.close()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(args.processes)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        out = {"results": results, "restarts": pool.restarts, "max_turns": args.max_turns}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


# --- soak ------------------------------------------------------------------------------------------

def cmd_soak(args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.minutes * 60
    with Pool(args.processes, base_port=args.base_port) as pool:
        counters = [{"steps": 0, "previews": 0, "rollouts": 0, "games": 0, "crashes": 0, "errors": 0}
                    for _ in range(args.processes)]
        samples: list[dict] = []
        engines_stats: list[list] = [[] for _ in range(args.processes)]
        crash_log: list[dict] = []
        lock = threading.Lock()

        def work(i: int) -> None:
            engine = pool.engine(i, timeout=300)
            rng = random.Random(1000 + i)
            c = counters[i]
            g = 0
            last_stats = 0.0
            while time.monotonic() < deadline:
                try:
                    c1, c2 = CHARS[(i + g) % 5], CHARS[(i + 2 * g + 1) % 5]
                    obs = req(engine, {"cmd": "new_match", "p1": c1, "p2": c2, "seed": i * 10000 + g})
                    g += 1
                    c["games"] += 1
                    while not obs["done"] and time.monotonic() < deadline:
                        pairs = [[rng.randrange(len(obs["candidates"][0])), rng.randrange(len(obs["candidates"][1]))]
                                 for _ in range(args.batch)]
                        res = req(engine, {"cmd": "preview", "pairs": pairs, "candidates": False})
                        if "error" in res:
                            c["errors"] += 1
                            break
                        c["previews"] += len(pairs)
                        if c["steps"] % 5 == 0:
                            leaf = next((x for x in res["results"] if not x["done"]), None)
                            if leaf:
                                req(engine, {"cmd": "continue", "handle": leaf["handle"], "pair": None,
                                             "until": {"ticks": 120}, "candidates": False})
                                c["rollouts"] += 1
                        obs = req(engine, {"cmd": "step", "actions": [rng.randrange(len(obs["candidates"][0])),
                                                                     rng.randrange(len(obs["candidates"][1]))]})
                        if "error" in obs:
                            c["errors"] += 1
                            break
                        c["steps"] += 1
                        if time.monotonic() - last_stats > 60:
                            last_stats = time.monotonic()
                            st = req(engine, {"cmd": "stats"})
                            with lock:
                                engines_stats[i].append({"t": round(time.monotonic() - t0), **st["engine"]})
                except (EngineCrashed, OSError, ValueError) as e:
                    c["crashes"] += 1
                    with lock:
                        crash_log.append({"server": i, "t": round(time.monotonic() - t0), "why": str(e),
                                          "exit_code": pool.exit_code(i), "chars": [c1, c2], "seed": i * 10000 + g - 1})
                    pool.restart(i)
                    engine = pool.engine(i, timeout=300)
            engine.close()

        t0 = time.monotonic()
        threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(args.processes)]
        for t in threads:
            t.start()
        cpu0 = pool.cpu_seconds()
        while any(t.is_alive() for t in threads):
            time.sleep(args.sample)
            mem = pool.memory_mb()
            sample = {"t": round(time.monotonic() - t0), "ws_mb": [round(m[0], 1) if m else None for m in mem],
                      "peak_mb": [round(m[1], 1) if m else None for m in mem],
                      "previews": sum(c["previews"] for c in counters), "steps": sum(c["steps"] for c in counters)}
            samples.append(sample)
            print(json.dumps(sample), flush=True)
        wall = time.monotonic() - t0
        out = {"processes": args.processes, "minutes": round(wall / 60, 1), "counters": counters,
               "restarts": pool.restarts, "startup_failures": pool.startup_failures, "crashes": crash_log,
               "samples": samples, "engine": engines_stats,
               "cpu_seconds_start": cpu0}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


# --- scaling ---------------------------------------------------------------------------------------

def cmd_scaling(args: argparse.Namespace) -> None:
    rows = []
    for n in [int(x) for x in args.counts.split(",")]:
        with Pool(n, base_port=args.base_port) as pool:
            done = [0] * n
            stop = time.monotonic() + args.warmup + args.seconds
            measure_from = time.monotonic() + args.warmup
            counted = [0] * n

            def work(i: int) -> None:
                engine = pool.engine(i, timeout=300)
                rng = random.Random(i)
                g = 0
                while time.monotonic() < stop:
                    obs = req(engine, {"cmd": "new_match", "p1": CHARS[(i + g) % 5], "p2": CHARS[(i + g + 1) % 5], "seed": g})
                    g += 1
                    while not obs["done"] and time.monotonic() < stop:
                        pairs = [[rng.randrange(len(obs["candidates"][0])), rng.randrange(len(obs["candidates"][1]))]
                                 for _ in range(args.batch)]
                        req(engine, {"cmd": "preview", "pairs": pairs, "candidates": False})
                        if time.monotonic() >= measure_from:
                            counted[i] += len(pairs)
                        done[i] += len(pairs)
                        obs = req(engine, {"cmd": "step", "actions": [rng.randrange(len(obs["candidates"][0])),
                                                                     rng.randrange(len(obs["candidates"][1]))]})
                engine.close()

            threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            time.sleep(args.warmup)
            idle0, tot0 = system_cpu_times()
            cpu0 = pool.cpu_seconds()
            t0 = time.monotonic()
            for t in threads:
                t.join()
            wall = time.monotonic() - t0
            idle1, tot1 = system_cpu_times()
            cpu1 = pool.cpu_seconds()
            proc_cpu = [round((b - a) / wall, 2) for a, b in zip(cpu0, cpu1) if a is not None and b is not None]
            row = {"processes": n, "seconds": round(wall, 1), "previews_per_s": round(sum(counted) / wall, 1),
                   "per_process": [round(c / wall, 1) for c in counted],
                   "server_cpu_cores_busy": round(sum(proc_cpu), 2), "per_process_cpu": proc_cpu,
                   "system_cpu_util": round(1 - (idle1 - idle0) / (tot1 - tot0), 3),
                   "peak_ws_mb": [round(m[1]) if m else None for m in pool.memory_mb()]}
            print(json.dumps(row), flush=True)
            rows.append(row)
    Path(args.out).write_text(json.dumps({"logical_cpus": os.cpu_count(), "rows": rows}, indent=1), encoding="utf-8")


# --- report ----------------------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    """Markdown sections from the JSON outputs of matchups / soak / scaling (whichever exist)."""
    L: list[str] = []
    w = L.append
    mp = Path(args.matchups)
    if mp.exists():
        m = json.loads(mp.read_text(encoding="utf-8"))
        w("## Matchups (5x5, mirrors included)\n")
        w(f"Each game: p1 random, p2 greedy (one preview per own candidate against p1's default, best hp swing "
          f"80% of the time), at most {m['max_turns']} turns. Every turn the pair about to be played is previewed "
          "with `detail` and compared with the real step. *halt*: crash, error answer, or no result within the turn cap.\n")
        w("| p1 \\ p2 | " +" | ".join(CHARS) + " |")
        w("|---|" + "---|" * len(CHARS))
        tot = {"games": 0, "turns": 0, "div": 0, "halts": 0, "crashes": 0}
        for c1 in CHARS:
            cells = []
            for c2 in CHARS:
                gs = m["results"].get(f"{c1} v {c2}", {}).get("games", [])
                turns = sum(g["turns"] for g in gs)
                div = sum(g["divergences"] for g in gs)
                halts = sum(1 for g in gs if g.get("error") or not g.get("done"))
                crashes = sum(1 for g in gs if str(g.get("error", "")).startswith("crash"))
                tot["games"] += len(gs); tot["turns"] += turns; tot["div"] += div; tot["halts"] += halts; tot["crashes"] += crashes
                cells.append(f"{len(gs)}g {turns}t, div {div}, halt {halts}")
            w(f"| {c1} | " + " | ".join(cells) + " |")
        w(f"\nTotal: {tot['games']} games, {tot['turns']} turns, {tot['div']} preview-vs-step divergences "
          f"({100 * tot['div'] / max(tot['turns'], 1):.2f}%), {tot['halts']} halts ({tot['crashes']} crashes), "
          f"server restarts {sum(m['restarts'])}.\n")
        ex = [(k, e) for k, v in m["results"].items() for g in v["games"] for e in g.get("examples", [])]
        if ex:
            w("<details><summary>Divergence examples</summary>\n")
            for k, e in ex[:20]:
                w(f"- {k} turn {e['turn']} `{e['moves']}`: {'; '.join(e['diff'][:3])[:300]}")
            w("\n</details>\n")
        errs = [(k, g["error"]) for k, v in m["results"].items() for g in v["games"] if g.get("error")]
        for k, e in errs:
            w(f"- halt: {k}: {str(e)[:300]}")
        w("")
    sp = Path(args.soak)
    if sp.exists():
        d = json.loads(sp.read_text(encoding="utf-8"))
        S = d["samples"]
        w(f"## Soak ({d['processes']} servers, {d['minutes']} min of continuous play)\n")
        c = d["counters"]
        w(f"Random play, 64-pair preview batch per turn, a 120-tick `continue` rollout every 5th turn. Totals: "
          f"{sum(x['games'] for x in c)} games, {sum(x['steps'] for x in c)} turns, {sum(x['previews'] for x in c)} previews, "
          f"{sum(x['rollouts'] for x in c)} rollouts; crashes {sum(x['crashes'] for x in c)}, error answers "
          f"{sum(x['errors'] for x in c)}, failed startups {len(d.get('startup_failures', []))}.\n")
        def avg(rows, i):
            vals = [r["ws_mb"][i] for r in rows if r["ws_mb"][i] is not None]
            return sum(vals) / len(vals) if vals else float("nan")
        first = [r for r in S if r["t"] <= 600]
        last = [r for r in S if r["t"] >= S[-1]["t"] - 600]
        mid = [r for r in S if 600 < r["t"] <= 1200]
        w("| server | working set, first 10 min (MB) | 10-20 min | last 10 min | peak | growth after 10 min |")
        w("|---|---|---|---|---|---|")
        for i in range(d["processes"]):
            peak = max((r["peak_mb"][i] or 0) for r in S)
            w(f"| {i} | {avg(first, i):.0f} | {avg(mid, i):.0f} | {avg(last, i):.0f} | {peak:.0f} | {avg(last, i) - avg(mid, i):+.0f} |")
        w("")
        eng = d.get("engine", [])
        if eng and any(eng):
            w("Engine counts per server (first and last `stats` sample): " + "; ".join(
                f"{i}: nodes {e[0]['nodes']:.0f} -> {e[-1]['nodes']:.0f}, objects {e[0]['objects']:.0f} -> {e[-1]['objects']:.0f}, "
                f"orphans {e[-1]['orphan_nodes']:.0f}" for i, e in enumerate(eng) if e) + "\n")
        for cr in d.get("crashes", []):
            w(f"- crash: server {cr['server']} at {cr['t']} s, {cr['chars']} seed {cr['seed']}, exit {cr['exit_code']}: {cr['why'][:150]}")
        w("")
    cp = Path(args.scaling)
    if cp.exists():
        d = json.loads(cp.read_text(encoding="utf-8"))
        w(f"## Scaling ({d['logical_cpus']} logical CPUs)\n")
        w("Random play, 64-pair batches, `candidates: false`, one client thread per server, after a warm-up.\n")
        w("| processes | previews/s | per process (min-max) | server CPU (cores busy) | system CPU | peak working set (MB) |")
        w("|---|---|---|---|---|---|")
        for r in d["rows"]:
            pp = r["per_process"]
            w(f"| {r['processes']} | {r['previews_per_s']:.0f} | {min(pp):.0f}-{max(pp):.0f} | {r['server_cpu_cores_busy']:.1f} | "
              f"{100 * r['system_cpu_util']:.0f}% | {max(x for x in r['peak_ws_mb'] if x)} |")
        w("")
    out = Path(args.md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("matchups")
    m.add_argument("--processes", type=int, default=8)
    m.add_argument("--games", type=int, default=2)
    m.add_argument("--max-turns", type=int, default=250)
    m.add_argument("--out", default="runs/matchups.json")
    m.add_argument("--base-port", type=int, default=7820)
    m.set_defaults(fn=cmd_matchups)
    s = sub.add_parser("soak")
    s.add_argument("--processes", type=int, default=8)
    s.add_argument("--minutes", type=float, default=65)
    s.add_argument("--batch", type=int, default=64)
    s.add_argument("--sample", type=float, default=30)
    s.add_argument("--out", default="runs/soak.json")
    s.add_argument("--base-port", type=int, default=7840)
    s.set_defaults(fn=cmd_soak)
    c = sub.add_parser("scaling")
    c.add_argument("--counts", default="1,4,8,12,16")
    c.add_argument("--seconds", type=float, default=60)
    c.add_argument("--warmup", type=float, default=10)
    c.add_argument("--batch", type=int, default=64)
    c.add_argument("--out", default="runs/scaling.json")
    c.add_argument("--base-port", type=int, default=7860)
    c.set_defaults(fn=cmd_scaling)
    r = sub.add_parser("report")
    r.add_argument("--matchups", default="runs/matchups.json")
    r.add_argument("--soak", default="runs/soak.json")
    r.add_argument("--scaling", default="runs/scaling.json")
    r.add_argument("--md", default="runs/stress.md")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
