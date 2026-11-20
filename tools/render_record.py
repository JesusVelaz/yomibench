"""Render a sim game record as a video by replaying its recorded picks in the real game.

    python tools/render_record.py ENGINE RECORD.jsonl.gz [...] --out DIR [--render]

The simulator is bit-exact with the real game, so a video needs no live play: for each record (round-3
self-play, a gate game, a sim-played showcase), `new_match` with the record's characters, seed and
candidate budget, then feed every decision's recorded picks with no search, one engine command at a
time (see tools/candidates_replay.py: the real game's server crashed when one command played many
steps). Expanded picks (set 4 factor overrides, which carry `base`) are rebuilt with `expand` from
their recorded base and values, since expand ids aren't stable.

Every decision is checked before its picks are played: the engine's state must equal the record's
`state` (every recorded field; fields only the engine sends are ignored) and its base candidate
lists must equal the record's. A playout-capped decision (`"search": "cheap"`, selfplay
--cheap-fraction) records only its capped lists: they must be the engine's lists with candidates
left out (same order, compared by action, data and extra), and its picks are found in the
engine's lists by that spec, since a position in a capped list isn't an engine id. At the end the result (winner, turns, hp) must match. The first
mismatch stops that record with the decision, turn and differing fields, and no replay is saved.

ENGINE is a real-game server (e.g. tcp:127.0.0.1:7777) running the record's candidate set; the
record's `candidate_rules` must equal the server's (start it with `serve --candidate-rules <hash>`,
which picks set 4 or the set-5 switch combination the hash names). ENGINE `docker` starts that
server itself: one yomibench-engine container per distinct `candidate_rules` among the records
(1 CPU, nice 19; --game-dir, --image, --port), stopped when its records are done. The match is
started with the record's own match settings (`engine.match_settings` in its header: stage width,
game length, starting meter, ceiling, ...), so a ranked-preset game replays and renders at the
ranked preset; a server that refuses them fails the record. Records without settings start on the server's defaults.
A record whose candidate set the real game doesn't serve (GAME_SETS: set 6 is the simulator's only) is checked
on exact inputs instead: a server on its default set, and at each decision the recorded picks' own
{action, data, extra} go in through `step_raw`. Its state is still checked at every decision and its result at the
end; only the candidate lists can't be (the game can't generate that set's).
Matching records are saved as DIR/<stem>.replay.
`--render` then encodes them with `yomibench render` (needs the game and a display; on a Mac, run
docker/render.sh on the saved replays instead).
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates_replay import _base, _first_diff  # noqa: E402

from yomibench.engine import Candidate, EngineError, connect  # noqa: E402


class Divergence(Exception):
    """The real game left the record."""


def state_diff(want: Any, have: Any, path: str = "state", limit: int = 12) -> list[str]:
    """Fields where `have` differs from `want`, as 'path: recorded vs engine' lines (at most
    `limit`). Every field of `want` must be present and equal; keys only in `have` are ignored."""
    out: list[str] = []

    def walk(w: Any, h: Any, p: str) -> None:
        if len(out) >= limit:
            return
        if isinstance(w, dict):
            if not isinstance(h, dict):
                out.append(f"{p}: {w!r} vs {h!r}")
                return
            for k, v in w.items():
                if k not in h:
                    out.append(f"{p}.{k}: missing from the engine's state")
                else:
                    walk(v, h[k], f"{p}.{k}")
        elif isinstance(w, list):
            if not isinstance(h, list) or len(w) != len(h):
                out.append(f"{p}: {len(w)} items vs {len(h) if isinstance(h, list) else repr(h)}")
                return
            for i, (a, b) in enumerate(zip(w, h)):
                walk(a, b, f"{p}[{i}]")
        elif w != h:
            out.append(f"{p}: {w!r} vs {h!r}")

    walk(want, have, path)
    return out


def read_record(path: Path) -> tuple[dict, list[dict], dict | None]:
    lines = [json.loads(x) for x in gzip.open(path, "rt")]
    turns = [x for x in lines[1:] if "picks" in x]
    result = next((x["result"] for x in lines[1:] if "result" in x), None)
    return lines[0], turns, result


def _spec(c: dict) -> dict:
    return {k: c[k] for k in ("action", "data", "extra")}


def capped_subset(want: list[dict], have: list[dict]) -> str | None:
    """None if `want` is `have` with some entries left out (order kept), else what doesn't fit."""
    k = 0
    for i, c in enumerate(want):
        while k < len(have) and have[k] != c:
            k += 1
        if k == len(have):
            return f"capped candidate {i} of {len(want)} isn't in the engine's {len(have)} (in order): {c}"
        k += 1
    return None


def pick_ids(engine, decision: dict, obs=None) -> list[int]:
    """Engine ids for the decision's recorded picks; expanded picks are rebuilt with `expand`. A
    base pick's id is its position in the record's list, or, with `obs` (a playout-capped
    decision, whose list leaves candidates out), the id of the engine candidate with its spec."""
    ids = []
    for seat in (0, 1):
        c = decision["candidates"][seat][decision["picks"][seat]]
        if "base" not in c:
            if obs is None:
                ids.append(decision["picks"][seat])
                continue
            found = [e.id for e in obs.candidates[seat] if _spec(e.record()) == _spec(c)]
            if not found:
                raise Divergence(f"seat {seat}: the recorded pick isn't among the engine's candidates: {_spec(c)}")
            ids.append(found[0])
            continue
        (got,) = engine.expand([{"seat": seat, "candidate": c["base"], "extra_override": c["values"]}])
        if not isinstance(got, Candidate):
            raise Divergence(f"seat {seat}: expand refused the recorded pick ({got})")
        if got.record() != c:
            raise Divergence(f"seat {seat}: the recorded pick rebuilt differently: {c} vs {got.record()}")
        ids.append(got.id)
    return ids


def _nonstandard(settings: dict | None, hello: dict) -> dict | None:
    """The record's match settings that differ from the engine's standard ones (None if none do).
    Records carry the full settings dict even for standard games; an engine that plays only the
    standard settings (a Batch) must not be asked for them."""
    if not settings:
        return None
    std = hello.get("default_match_settings") or hello.get("match_settings") or {}
    diff = {k: v for k, v in settings.items() if k not in std or str(std[k]) != str(v)}
    return diff or None


GAME_SETS = ("2", "4", "5", "6")  # candidate sets the real game's server can generate


def exact_inputs(head: dict) -> bool:
    """Check this record on exact inputs (step_raw): its candidate set is one the real game can't serve."""
    cs = head.get("engine", {}).get("candidate_set")
    return cs is not None and str(cs) not in GAME_SETS  # older records name no set: the game's


def replay(engine, head: dict, turns: list[dict], result: dict | None):
    """Play the record on `engine`, checking state (and, on a set the game serves, candidates) at every decision
    and the result at the end. Returns the final observation; raises Divergence on the first mismatch."""
    raw = exact_inputs(head)
    want_rules = head.get("engine", {}).get("candidate_rules")
    have_rules = engine.hello().get("candidate_rules")
    if not raw and want_rules != have_rules:
        raise Divergence(f"candidate_rules differ: record {want_rules}, engine {have_rules}")
    settings = _nonstandard(head.get("engine", {}).get("match_settings"), engine.hello())
    try:
        obs = engine.new_match(*head["characters"], head["seed"], head.get("engine", {}).get("max_candidates"),
                               settings=settings)
    except EngineError as e:
        raise Divergence(f"the engine refused the record's match settings: {e}") from None
    for k, t in enumerate(turns):
        where = f"decision {k} (turn {t['turn']})"
        if obs.done:
            raise Divergence(f"{where}: the real game already ended, {len(turns) - k} decisions early")
        diff = state_diff(t["state"], obs.state)
        if diff:
            raise Divergence(f"{where}: state differs:\n  " + "\n  ".join(diff))
        if raw:
            try:
                obs = engine.step_raw([_spec(t["candidates"][s][t["picks"][s]]) for s in (0, 1)])
            except EngineError as e:
                raise Divergence(f"{where}: the real game refused the recorded inputs: {e}") from None
            continue
        capped = t.get("search") == "cheap"
        for seat in (0, 1):
            want = _base(t["candidates"][seat])
            have = [{"action": c.action, "data": c.data, "extra": c.extra} for c in obs.candidates[seat]]
            if capped:
                why = capped_subset(want, have)
                if why:
                    raise Divergence(f"{where} seat {seat}: {why}")
            elif want != have:
                raise Divergence(f"{where} seat {seat}: candidates differ: {_first_diff(want, have)}")
        try:
            ids = pick_ids(engine, t, obs if capped else None)
        except Divergence as e:
            raise Divergence(f"{where} {e}") from None
        obs = engine.step(*ids)
    if result is not None:
        have = {"done": obs.done, "winner": obs.winner, "turns": obs.turn,
                "hp": [p["hp"] for p in obs.state["players"]]}
        want = {k: result.get(k) for k in have}
        if want != have:
            raise Divergence(f"result differs: record {want}, engine {have}")
    return obs


def docker_command(rules: str, game_dir: Path, image: str, port: int, name: str) -> list[str]:
    """`docker run` for one engine server on the candidate set `rules` names, on 127.0.0.1:port:
    1 CPU, nice 19 inside the container."""
    checkout = Path(__file__).resolve().parents[1]
    return ["docker", "run", "-d", "--rm", "--name", name, "--platform", "linux/amd64", "--cpus=1",
            "-v", f"{checkout}:/yomibench:ro", "-v", f"{game_dir}:/game:ro", "-p", f"127.0.0.1:{port}:7777",
            "--entrypoint", "nice", image, "-n", "19", "/bin/sh", "/yomibench/docker/entrypoint.sh",
            "--count", "1", "--port", "7777", *(["--candidate-rules", rules] if rules else [])]


@contextlib.contextmanager
def docker_engine(rules: str, game_dir: Path, image: str, port: int, wait: float = 300.0):
    """A real-game server in a container, serving the candidate set `rules` names; stopped on exit."""
    name = f"yomibench-render-{port}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    subprocess.run(docker_command(rules, game_dir, image, port, name), check=True, capture_output=True)
    try:
        deadline = time.monotonic() + wait
        while True:
            try:
                engine = connect(f"tcp:127.0.0.1:{port}")
                hello = engine.hello()
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(2)
        if rules and hello.get("candidate_rules") != rules:
            raise RuntimeError(f"the container serves rules {hello.get('candidate_rules')}, not {rules}")
        yield engine
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@contextlib.contextmanager
def native_engine(rules: str | None, game_dir: Path):
    """A real-game server on this machine (Windows: the game's own Windows build), headless, muted and windowless
    at below-normal priority, serving the candidate set `rules` names; stopped on exit."""
    from yomibench.game import GameServer

    run_dir = Path.home() / ".yomibench" / "render-check-run"  # its own scratch copy (never the shared run dir)
    server = GameServer(game_dir=str(game_dir), run_dir=run_dir,
                        extra_args=[f"--yomi-candidate-rules={rules}"] if rules else [], below_normal=True)
    try:
        from yomibench.game import connect_server

        engine = connect_server(server.port)
        hello = engine.hello()
        if rules and hello.get("candidate_rules") != rules:
            raise RuntimeError(f"the server serves rules {hello.get('candidate_rules')}, not {rules}")
        yield engine
    finally:
        server.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("engine", help="tcp:HOST:PORT, docker (start a container per candidate_rules), or native (start "
                                   "a headless server of the game in --game-dir on this machine per candidate_rules)")
    ap.add_argument("records", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True, help="folder for the replays (and videos)")
    ap.add_argument("--render", action="store_true", help="encode the replays with yomibench render")
    ap.add_argument("--game-dir", type=Path, default=Path.home() / ".yomibench" / "game-linux",
                    help="ENGINE docker: the game's Linux build")
    ap.add_argument("--image", default="yomibench-engine", help="ENGINE docker: the image")
    ap.add_argument("--port", type=int, default=7790, help="ENGINE docker: host port")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    failed = 0
    if args.engine in ("docker", "native"):
        groups: dict[str, list[Path]] = {}
        for path in args.records:
            head = read_record(path)[0]
            rules = None if exact_inputs(head) else head.get("engine", {}).get("candidate_rules")  # None: default set
            groups.setdefault(rules, []).append(path)
        for rules, paths in groups.items():
            ctx = (native_engine(rules, args.game_dir.resolve()) if args.engine == "native"
                   else docker_engine(rules, args.game_dir.resolve(), args.image, args.port))
            with ctx as engine:
                failed += replay_all(engine, paths, args.out, saved)
    else:
        failed = replay_all(connect(args.engine), args.records, args.out, saved)
    if args.render and saved:
        from yomibench.render import render
        render(saved, args.out, game_dir=str(args.game_dir) if args.engine == "native" else None)
    return 1 if failed else 0


def replay_all(engine, paths: list[Path], out: Path, saved: list[Path]) -> int:
    """Replay each record on `engine`, saving the matching ones' replays; returns the failures."""
    failed = 0
    for path in paths:
        head, turns, result = read_record(path)
        stem = path.name.split(".")[0]
        t0, start = time.monotonic(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            obs = replay(engine, head, turns, result)
        except Divergence as e:
            print(f"{path.name}: DIVERGED {e}", flush=True)
            failed += 1
            continue
        replay_path = out / f"{stem}.replay"
        info = engine.save_replay(replay_path)
        print(json.dumps({"record": path.name, "characters": head["characters"], "seed": head["seed"],
                          "decisions": len(turns), "turns": obs.turn, "winner": obs.winner,
                          "replay": str(replay_path), "check": ("every decision's state matched (exact inputs)"
                                                                if exact_inputs(head) else "every decision matched"),
                          "check_start_utc": start, "check_s": round(time.monotonic() - t0, 1), **info}), flush=True)
        saved.append(replay_path)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
