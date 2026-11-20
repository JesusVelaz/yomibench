"""Checks and measurements for a protocol engine (made for the real-game server in mod/YomiBench).

    uv run python tools/engine_check.py agreement --engine tcp:127.0.0.1:7777 --turns 300
    uv run python tools/engine_check.py determinism --engine tcp:127.0.0.1:7777
    uv run python tools/engine_check.py speed --engine tcp:127.0.0.1:7777
    uv run python tools/engine_check.py parallel --processes 4 --seconds 60   # starts its own servers
    uv run python tools/engine_check.py matrix --engine tcp:127.0.0.1:7777 --turns 10
    uv run python tools/engine_check.py rollout --engine tcp:127.0.0.1:7777 --ticks 120

Moves are picked by a seeded RNG, so every run is reproducible. `detail: true` asks the server
for a fuller snapshot (object list, state ticks, hitlag) than the protocol's `state`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import threading
import time
from typing import Any

from yomibench.engine import Engine, connect

CHARS = ["Ninja", "Cowboy", "Wizard", "Robot", "Mutant"]


def call(engine: Engine, msg: dict[str, Any]) -> dict[str, Any]:
    return engine._call(msg)


def pick(obs: dict[str, Any], rng: random.Random) -> list[int]:
    return [rng.randrange(len(obs["candidates"][s])) for s in (0, 1)]


def diff(a: Any, b: Any, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: len {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def cmd_agreement(args: argparse.Namespace) -> None:
    """Preview the pair about to be played, then play it; compare the two snapshots."""
    engine = connect(args.engine)
    rng = random.Random(args.seed)
    checked = mismatched = 0
    by_char: dict[str, list[int]] = {}
    game = 0
    while checked < args.turns:
        c1, c2 = CHARS[game % 5], CHARS[(game * 2 + 1) % 5]
        obs = call(engine, {"cmd": "new_match", "p1": c1, "p2": c2, "seed": args.seed + game})
        while not obs["done"] and checked < args.turns:
            pair = pick(obs, rng)
            names = [obs["candidates"][s][pair[s]]["action"] for s in (0, 1)]
            # Other previews first, so the checked one runs on a ghost dirtied by earlier previews.
            warm = [pick(obs, rng) for _ in range(args.warm)]
            ghost = call(engine, {"cmd": "preview", "pairs": warm + [pair], "detail": True, "candidates": False})["results"][-1]["detail"]
            obs = call(engine, {"cmd": "step", "actions": pair, "detail": True})
            real = obs["detail"]
            d = diff(real, ghost)
            checked += 1
            stats = by_char.setdefault(f"{c1} v {c2}", [0, 0])
            stats[0] += 1
            if d:
                mismatched += 1
                stats[1] += 1
                if args.verbose or mismatched <= 5:
                    print(f"MISMATCH {c1} v {c2} turn {obs['turn']} {names}: {d[:6]}")
        game += 1
    for k, (n, m) in by_char.items():
        print(f"  {k:<18} {m}/{n}")
    print(f"agreement: {checked - mismatched}/{checked} previews matched the real step "
          f"({100 * mismatched / max(checked, 1):.2f}% diverged) over {game} games")


def play_record(engine: Engine, c1: str, c2: str, seed: int, rng_seed: int, max_turns: int) -> tuple[list, dict]:
    rng = random.Random(rng_seed)
    obs = call(engine, {"cmd": "new_match", "p1": c1, "p2": c2, "seed": seed})
    actions = []
    while not obs["done"] and obs["turn"] < max_turns:
        pair = pick(obs, rng)
        actions.append(pair)
        obs = call(engine, {"cmd": "step", "actions": pair, "detail": True})
    return actions, obs.get("detail", obs["state"])


def replay(engine: Engine, c1: str, c2: str, seed: int, actions: list) -> dict:
    obs = call(engine, {"cmd": "new_match", "p1": c1, "p2": c2, "seed": seed, "detail": True})
    for pair in actions:
        obs = call(engine, {"cmd": "step", "actions": pair, "detail": True})
    return obs.get("detail", obs["state"])


def cmd_determinism(args: argparse.Namespace) -> None:
    """Same seed + same actions, replayed on this engine (and on --other if given)."""
    engine = connect(args.engine)
    others = [connect(s) for s in args.other]
    ok = total = 0
    for g in range(args.games):
        c1, c2 = CHARS[g % 5], CHARS[(g + 2) % 5]
        actions, final = play_record(engine, c1, c2, 1000 + g, g, args.max_turns)
        for target in [engine, *others]:
            again = replay(target, c1, c2, 1000 + g, actions)
            d = diff(final, again)
            total += 1
            ok += not d
            print(f"{c1} v {c2}: {len(actions)} turns, tick {final['tick']}, hp "
                  f"{[p['hp'] for p in final['players']]} -> {'same' if not d else 'DIFFERENT ' + str(d[:4])}")
    print(f"determinism: {ok}/{total} replays reproduced the final state exactly")


def cmd_speed(args: argparse.Namespace) -> None:
    """Steps and previews per second on one process, with random play."""
    engine = connect(args.engine)
    rng = random.Random(args.seed)
    step_s = prev_s = 0.0
    steps = previews = 0
    ticks = 0
    counts: list[int] = []
    t_end = time.perf_counter() + args.seconds
    g = 0
    while time.perf_counter() < t_end:
        obs = call(engine, {"cmd": "new_match", "p1": CHARS[g % 5], "p2": CHARS[(g + 1) % 5], "seed": g})
        g += 1
        while not obs["done"] and time.perf_counter() < t_end:
            counts += [len(obs["candidates"][0]), len(obs["candidates"][1])]
            if args.batch:
                pairs = [pick(obs, rng) for _ in range(args.batch)]
                t = time.perf_counter()
                call(engine, {"cmd": "preview", "pairs": pairs, "candidates": args.candidates})
                prev_s += time.perf_counter() - t
                previews += len(pairs)
            tick0 = obs["state"]["tick"]
            t = time.perf_counter()
            obs = call(engine, {"cmd": "step", "actions": pick(obs, rng)})
            step_s += time.perf_counter() - t
            steps += 1
            ticks += obs["state"]["tick"] - tick0
    server = call(engine, {"cmd": "stats"})
    print(json.dumps({
        "server_commands": {k: {"n": v["n"], "handle_ms": round(v["handle_us"] / 1000 / v["n"], 2),
                                "json_ms": round(v["json_us"] / 1000 / v["n"], 2), "kb": round(v["bytes"] / 1024 / v["n"], 1)}
                            for k, v in server["commands"].items()},
        "games_started": g, "steps": steps, "steps_per_s": round(steps / step_s, 1),
        "ticks_per_turn": round(ticks / max(steps, 1), 1),
        "previews": previews, "previews_per_s": round(previews / prev_s, 1) if previews else None,
        "ms_per_preview": round(1000 * prev_s / previews, 2) if previews else None,
        "candidates_per_seat": {"mean": round(statistics.mean(counts), 1), "median": statistics.median(counts),
                                "max": max(counts), "min": min(counts)},
    }))


def peak_memory_mb(pid: int) -> float | None:
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    h = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
    if not h:
        return None
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    try:
        if not ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return round(pmc.PeakWorkingSetSize / 2**20, 1)
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def cmd_parallel(args: argparse.Namespace) -> None:
    """Start N servers, drive each from a thread with random play + previews, sum throughput."""
    from yomibench import game

    run_dir = game.build_run_dir(game.find_game_dir(args.game_dir))
    ports = [args.base_port + i for i in range(args.processes)]
    procs = [game.launch(run_dir, p) for p in ports]
    try:
        for p, proc in zip(ports, procs):
            game.wait_ready(p, proc)
        results: list[dict] = [{} for _ in ports]

        def work(i: int) -> None:
            engine = connect(f"tcp:127.0.0.1:{ports[i]}")
            rng = random.Random(i)
            steps = previews = 0
            t_end = time.perf_counter() + args.seconds
            g = 0
            while time.perf_counter() < t_end:
                obs = call(engine, {"cmd": "new_match", "p1": CHARS[(i + g) % 5], "p2": CHARS[(i + g + 1) % 5], "seed": g})
                g += 1
                while not obs["done"] and time.perf_counter() < t_end:
                    pairs = [pick(obs, rng) for _ in range(args.batch)]
                    call(engine, {"cmd": "preview", "pairs": pairs, "candidates": args.candidates})
                    previews += len(pairs)
                    obs = call(engine, {"cmd": "step", "actions": pick(obs, rng)})
                    steps += 1
            results[i] = {"steps": steps, "previews": previews}
            engine.close()

        threads = [threading.Thread(target=work, args=(i,)) for i in range(len(ports))]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        mem = [peak_memory_mb(p.pid) for p in procs]
        total_prev = sum(r["previews"] for r in results)
        total_steps = sum(r["steps"] for r in results)
        print(json.dumps({
            "processes": len(ports), "seconds": round(wall, 1),
            "total_previews_per_s": round(total_prev / wall, 1),
            "per_process_previews_per_s": [round(r["previews"] / wall, 1) for r in results],
            "total_steps": total_steps, "peak_working_set_mb": mem,
        }))
    finally:
        for p in procs:
            p.kill()
            p.wait()


def cmd_matrix(args: argparse.Namespace) -> None:
    """Seconds per turn for a full payoff matrix (what the matrix and greedy bots preview)."""
    engine = connect(args.engine)
    rng = random.Random(args.seed)
    obs = call(engine, {"cmd": "new_match", "p1": args.p1, "p2": args.p2, "seed": args.seed})
    rows = []
    for _ in range(args.turns):
        if obs["done"]:
            break
        n1, n2 = len(obs["candidates"][0]), len(obs["candidates"][1])
        pairs = [[i, j] for i in range(n1) for j in range(n2)]
        t = time.perf_counter()
        call(engine, {"cmd": "preview", "pairs": pairs, "candidates": False})
        dt = time.perf_counter() - t
        rows.append((n1, n2, dt))
        print(f"turn {obs['turn']}: {n1}x{n2} = {n1 * n2} previews in {dt:.2f}s")
        obs = call(engine, {"cmd": "step", "actions": pick(obs, rng)})
    print(f"mean {statistics.mean(r[2] for r in rows):.2f}s per full matrix, "
          f"mean size {statistics.mean(r[0] * r[1] for r in rows):.0f}")


def cmd_rollout(args: argparse.Namespace) -> None:
    """Cost of `continue`: from each of K preview leaves, roll out the default continuation for
    at least N ticks. Leaves other than the last one previewed have to be replayed first."""
    engine = connect(args.engine)
    rng = random.Random(args.seed)
    obs = call(engine, {"cmd": "new_match", "p1": args.p1, "p2": args.p2, "seed": args.seed})
    roll_s = 0.0
    rolls = ticks = 0
    for _ in range(args.turns):
        if obs["done"]:
            break
        leaves = call(engine, {"cmd": "preview", "pairs": [pick(obs, rng) for _ in range(args.leaves)],
                               "candidates": False})["results"]
        for leaf in leaves:
            if leaf["done"]:
                continue
            t = time.perf_counter()
            end = call(engine, {"cmd": "continue", "handle": leaf["handle"], "pair": None,
                                "until": {"ticks": args.ticks}, "candidates": False})
            roll_s += time.perf_counter() - t
            rolls += 1
            ticks += end["state"]["tick"] - leaf["state"]["tick"]
        obs = call(engine, {"cmd": "step", "actions": pick(obs, rng)})
    print(json.dumps({"rollouts": rolls, "ms_per_rollout": round(1000 * roll_s / max(rolls, 1), 1),
                      "ticks_per_rollout": round(ticks / max(rolls, 1), 1),
                      "ticks_per_s_in_rollouts": round(ticks / roll_s, 0) if roll_s else None}))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("agreement", cmd_agreement), ("determinism", cmd_determinism), ("speed", cmd_speed),
                     ("parallel", cmd_parallel), ("matrix", cmd_matrix), ("rollout", cmd_rollout)]:
        s = sub.add_parser(name)
        s.set_defaults(fn=fn)
        s.add_argument("--engine", default="tcp:127.0.0.1:7777")
        s.add_argument("--seed", type=int, default=1)
    sub.choices["agreement"].add_argument("--turns", type=int, default=300)
    sub.choices["agreement"].add_argument("--verbose", action="store_true")
    sub.choices["agreement"].add_argument("--warm", type=int, default=8, help="random previews before the checked one")
    sub.choices["determinism"].add_argument("--games", type=int, default=5)
    sub.choices["determinism"].add_argument("--max-turns", type=int, default=400)
    sub.choices["determinism"].add_argument("--other", action="append", default=[], help="another engine spec to replay on")
    sub.choices["speed"].add_argument("--seconds", type=float, default=60)
    sub.choices["speed"].add_argument("--batch", type=int, default=64, help="previews per turn (0: steps only)")
    sub.choices["speed"].add_argument("--candidates", action="store_true", help="previews return candidate lists")
    sub.choices["parallel"].add_argument("--processes", type=int, default=4)
    sub.choices["parallel"].add_argument("--seconds", type=float, default=60)
    sub.choices["parallel"].add_argument("--batch", type=int, default=64)
    sub.choices["parallel"].add_argument("--candidates", action="store_true")
    sub.choices["parallel"].add_argument("--base-port", type=int, default=7801)
    sub.choices["parallel"].add_argument("--game-dir")
    sub.choices["matrix"].add_argument("--turns", type=int, default=10)
    sub.choices["matrix"].add_argument("--p1", default="Ninja")
    sub.choices["matrix"].add_argument("--p2", default="Cowboy")
    for name, default in (("turns", 10), ("leaves", 16), ("ticks", 120)):
        sub.choices["rollout"].add_argument(f"--{name}", type=int, default=default)
    sub.choices["rollout"].add_argument("--p1", default="Ninja")
    sub.choices["rollout"].add_argument("--p2", default="Cowboy")
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
