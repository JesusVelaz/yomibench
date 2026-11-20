"""Exhaustive move and parameter coverage for the real-game server (mod/YomiBench).

    uv run python tools/move_coverage.py run --processes 16 --minutes 60 --out runs/coverage
    uv run python tools/move_coverage.py report --out runs/coverage --md docs/testing/move-coverage.md

The bot only sees a discretized candidate set. This tool checks that every move and every widget
value the game's UI can produce is safe and faithful, using the server's debug commands `moves`
(every menu move, legality, widgets) and `probe` (a candidate from chosen widget values, set
through the widgets themselves). Per server thread it plays seeded matches (random, greedy and
super-seeking play, so fighters end up grounded, airborne, cornered, in hitstun, with meter and
burst, after supers, with projectiles out), and at each decision:

- builds a dense sweep per legal move (XY plots: 32 angles x 4 magnitudes, limit edges, snapped
  mouse input near snap angles; 8-way: all nine directions; sliders/counts: every step; checkboxes,
  toggles, dropdowns: every value; widgets of one move as a cartesian product when it has <= 200
  combinations, else one at a time plus pairwise), plus extras (PlayerExtra widgets, DI, feint,
  reverse, prediction);
- probes a capped, coverage-first sample of it and previews each probe against the opponent's
  Continue (detail: states entered, so a move that silently didn't happen shows);
- watches the server's engine log for script errors after each batch;
- steps a sampled pair (often a probe) and compares its preview with the real step;
- every 8th decision sweeps one parameterized move completely next to the bot's own candidates
  for that move (discretization analysis).

A crash (server gone, connection dropped, timeout) is recorded with the request, the match seed
and the action history, the server is restarted, and the crash is replayed on the fresh server
one probe at a time to find the exact move + data. Output: <out>/coverage.json (aggregates),
<out>/events.jsonl (crashes, divergences, script errors, refusals), <out>/discretization.jsonl.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serverpool import LogTail, Pool  # noqa: E402

from yomibench.engine import EngineCrashed, EngineError  # noqa: E402

CHARS = ["Ninja", "Cowboy", "Wizard", "Robot", "Mutant"]
WAY_DIRS = ["E", "NE", "SE", "N", "S", "W", "NW", "SW", "Neutral"]
XY_ANGLES = [i * 360 / 32 for i in range(32)]
XY_MAGS = [0.1, 0.5, 0.9, 1.0]
COMBO_LIMIT = 200
PER_SEAT_CAP = 160          # unique probe candidates previewed per seat per decision
FULL_SWEEP_EVERY = 8
ERROR_MARKERS = ("SCRIPT ERROR", "ERROR:", "Invalid call", "Invalid get", "Invalid set", "panicked")


# --- sweep generation ----------------------------------------------------------------------------

def xy_values(w: dict, dense: bool = True) -> list[dict]:
    """XY plot values as fractions of the radius, screen axes (y down)."""
    out = [{"x": 0.0, "y": 0.0}]
    angles = XY_ANGLES if dense else XY_ANGLES[::2]
    mags = XY_MAGS if dense else [0.5, 1.0]
    for a in angles:
        r = math.radians(a)
        for m in mags:
            out.append({"x": round(math.cos(r) * m, 4), "y": round(-math.sin(r) * m, 4)})
    if w.get("limit_angle"):
        c = math.radians(w["limit_center_degrees"])
        cx, cy = math.cos(c) * w.get("facing", 1), math.sin(c)
        center = math.atan2(cy, cx)
        half = math.radians(w["limit_range_degrees"]) / 2
        for edge in (center - half, center + half):
            for off in (0.0, math.radians(1.0) * (1 if edge > center else -1)):
                a = edge + off
                out.append({"x": round(math.cos(a), 4), "y": round(math.sin(a), 4)})
                if w.get("limit_symmetrical"):
                    out.append({"x": round(-math.cos(a), 4), "y": round(-math.sin(a), 4)})
    if dense and w.get("snap_angles"):
        # Mouse input: snapping happens within ~5.7 degrees of a snap angle.
        n = w["snap_angles"]
        for i in range(n):
            base = 2 * math.pi * i / n
            for off in (3.0, 7.0):
                a = base + math.radians(off)
                for m in (0.6, 1.0):
                    out.append({"x": round(math.cos(a) * m, 4), "y": round(math.sin(a) * m, 4), "snap": True})
    return out


def range_values(w: dict) -> list[float]:
    lo, hi, step = w["min"], w["max"], w.get("step") or 0
    if step and (hi - lo) / step <= 100:
        n = int(round((hi - lo) / step))
        vals = [lo + i * step for i in range(n + 1)]
    else:
        vals = [lo + (hi - lo) * i / 10 for i in range(11)]
    return vals + [lo - (step or 1), hi + (step or 1)]   # out of range: the Range clamps


def widget_values(w: dict, dense: bool = True) -> list[Any]:
    k = w["kind"]
    if k == "xy":
        return xy_values(w, dense)
    if k == "8way":
        return list(w.get("enabled", WAY_DIRS))
    if k in ("slider", "count"):
        return range_values(w)
    if k in ("check", "toggle", "height"):
        return [] if w.get("disabled") else [True, False]
    if k == "option":
        return list(range(len(w["items"])))
    return []


def reduce(vals: list[Any], n: int) -> list[Any]:
    if len(vals) <= n:
        return vals
    step = (len(vals) - 1) / (n - 1)
    return [vals[round(i * step)] for i in range(n)]


def widget_combos(widgets: list[dict], dense: bool = True) -> list[dict[str, Any]]:
    """Widget assignments for one move: full product if small, else singles plus pairwise."""
    idx = [i for i, w in enumerate(widgets) if w.get("visible", True)]
    lists = {i: widget_values(widgets[i], dense) for i in idx}
    lists = {i: v for i, v in lists.items() if v}
    if not lists:
        return [{}]
    keys = list(lists)
    total = math.prod(len(lists[k]) for k in keys)
    out: list[dict[str, Any]] = [{}]
    if total <= COMBO_LIMIT:
        for combo in itertools.product(*(lists[k] for k in keys)):
            out.append({str(k): v for k, v in zip(keys, combo)})
        return out
    for k in keys:
        out += [{str(k): v} for v in lists[k]]
    for a, b in itertools.combinations(keys, 2):
        for va in reduce(lists[a], 10):
            for vb in reduce(lists[b], 10):
                out.append({str(a): va, str(b): vb})
    return out


def vkey(v: Any) -> str:
    return json.dumps(v, sort_keys=True)


def sweep_for_seat(desc: dict, rng: random.Random) -> list[tuple[dict, list[tuple]]]:
    """[(probe spec, [coverage keys])] for one seat at one decision."""
    out: list[tuple[dict, list[tuple]]] = []
    legal = [m for m in desc["moves"] if m["legal"]]
    for m in legal:
        name = m["name"]
        # Gate checks: values the menu shows disabled must be refused.
        for i, w in enumerate(m["widgets"]):
            if w["kind"] == "8way" and w.get("visible", True):
                for d in WAY_DIRS:
                    if d not in w["enabled"]:
                        out.append(({"action": name, "widgets": {str(i): d}, "gate": True}, [("gate", name, w["name"], d)]))
        for combo in widget_combos(m["widgets"]):
            keys = [("w", name, int(k), m["widgets"][int(k)]["name"], vkey(v)) for k, v in combo.items()]
            out.append(({"action": name, "widgets": combo}, keys or [("w", name, -1, "", "default")]))
        if m.get("feintable"):
            out.append(({"action": name, "feint": True}, [("feint", name)]))
        if m.get("reversible"):
            out.append(({"action": name, "reverse": True}, [("reverse", name)]))
            for i, w in enumerate(m["widgets"]):
                if w["kind"] == "xy" and w.get("visible", True):
                    for v in reduce(xy_values(w, dense=False), 8):
                        out.append(({"action": name, "reverse": True, "widgets": {str(i): v}},
                                    [("reverse", name), ("w", name, i, w["name"], vkey(v))]))
    # Extras: each visible extra widget, with Continue and with a random legal move.
    targets = ["Continue"] + ([rng.choice(legal)["name"]] if legal else [])
    for i, w in enumerate(desc["extra_widgets"]):
        if w["visible"] and w["kind"] == "toggle" and w.get("disabled"):
            out.append(({"action": "Continue", "extra": {str(i): not w["pressed"]}, "gate": True},
                        [("gate", "extra", w["name"], "disabled")]))
        if not w["visible"]:
            out.append(({"action": "Continue", "extra": {str(i): widget_values(w, False)[0] if widget_values(w, False) else True},
                         "gate": True}, [("gate", "extra", w["name"], "hidden")]))
            continue
        for v in widget_values(w, dense=False):
            for t in targets:
                out.append(({"action": t, "extra": {str(i): v}}, [("extra", w["name"], vkey(v))]))
    toggles = [i for i, w in enumerate(desc["extra_widgets"]) if w["visible"] and w["kind"] == "toggle" and not w.get("disabled")]
    for a, b in itertools.combinations(toggles, 2):
        out.append(({"action": "Continue", "extra": {str(a): True, str(b): True}},
                    [("extra_pair", desc["extra_widgets"][a]["name"], desc["extra_widgets"][b]["name"])]))
    # DI (any action; matters when hit) and prediction.
    di_vals = xy_values({"snap_angles": 8}, dense=desc["hurt"])
    for v in di_vals:
        out.append(({"action": "Continue", "di": v}, [("di", vkey(v))]))
    if legal:
        for v in reduce(di_vals, 6):
            t = rng.choice(legal)["name"]
            out.append(({"action": t, "di": v}, [("di", vkey(v))]))
    for p in desc["predictions"]:
        out.append(({"action": "Continue", "prediction": p}, [("prediction", p)]))
    return out


# --- shared results ------------------------------------------------------------------------------

def nested() -> dict:
    return defaultdict(nested)


class Coverage:
    def __init__(self, out: Path):
        self.lock = threading.RLock()   # events are logged while holding it
        self.out = out
        self.events = open(out / "events.jsonl", "a", encoding="utf-8")
        self.disc = open(out / "discretization.jsonl", "a", encoding="utf-8")
        self.chars: dict[str, dict] = {}
        self.tested: dict[str, set] = defaultdict(set)     # char -> coverage keys
        self.counters = Counter()

    def char(self, c: str) -> dict:
        if c not in self.chars:
            self.chars[c] = {
                "moves": {}, "extra_widgets": {}, "situations": Counter(), "decisions": 0,
                "previews": 0, "probes": 0, "refusals": Counter(), "steps": 0, "probe_steps": 0,
                "divergences": 0, "crashes": 0, "script_errors": 0, "not_entered": Counter(),
                "di_values": set(), "predictions": set(),
            }
        return self.chars[c]

    def move(self, c: str, m: dict) -> dict:
        moves = self.char(c)["moves"]
        if m["name"] not in moves:
            moves[m["name"]] = {
                "type": m["type"], "grab": m["grab"], "air_type": m["air_type"], "legal": 0, "seen": 0,
                "widgets": {}, "probes": 0, "previews": 0, "accepted": 0, "unique_data": set(),
                "feint": 0, "reverse": 0, "not_entered": 0, "entered_first": Counter(), "refusals": Counter(),
                "illegal_usable": Counter(), "divergences": 0, "crashes": 0, "stepped": 0,
            }
        return moves[m["name"]]

    def event(self, kind: str, **kw: Any) -> None:
        with self.lock:
            self.events.write(json.dumps({"kind": kind, "time": time.time(), **kw}, default=str) + "\n")
            self.events.flush()
            self.counters[kind] += 1

    def disc_record(self, rec: dict) -> None:
        with self.lock:
            self.disc.write(json.dumps(rec) + "\n")
            self.disc.flush()

    def dump(self) -> None:
        def conv(o: Any, key: str = "") -> Any:
            if isinstance(o, set):
                if key == "unique_data" or len(o) > 400:
                    return {"count": len(o)}
                return sorted(o, key=str)
            if isinstance(o, Counter):
                return dict(o.most_common(40))
            if isinstance(o, dict):
                return {str(k): conv(v, str(k)) for k, v in o.items()}
            return o
        with self.lock:
            data = {"chars": conv(self.chars), "counters": dict(self.counters),
                    "tested_keys": {c: len(s) for c, s in self.tested.items()}}
        (self.out / "coverage.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")


# --- one worker (one server) ---------------------------------------------------------------------

class Crash(Exception):
    pass


class Worker:
    def __init__(self, idx: int, pool: Pool, cov: Coverage, deadline: float, seed: int):
        self.idx = idx
        self.pool = pool
        self.cov = cov
        self.deadline = deadline
        self.rng = random.Random(seed)
        self.engine = None
        self.log = None
        self.connect()

    def connect(self) -> None:
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
        self.engine = self.pool.engine(self.idx, timeout=300)
        self.log = LogTail(self.pool.log_path(self.idx))

    def call(self, msg: dict) -> dict:
        try:
            return self.engine.transport.request(msg)
        except (EngineCrashed, OSError, TimeoutError, ValueError) as e:
            raise Crash(str(e)) from e

    def errors(self) -> list[str]:
        return [ln for ln in self.log.read() if any(m in ln for m in ERROR_MARKERS) and "DRAWDBG" not in ln]

    # A situation label set for the seat's fighter.
    def situations(self, obs: dict, desc: dict, seat: int, last_moves: list[str], move_types: dict) -> list[str]:
        p = obs["state"]["players"][seat]
        o = obs["state"]["players"][1 - seat]
        s = ["grounded" if desc["grounded"] else "airborne"]
        if abs(p["x"]) >= 420 and abs(p["x"]) > abs(o["x"]) and (p["x"] > 0) == (o["x"] > 0):
            s.append("cornered")
        if desc["hurt"]:
            s.append("hitstun")
        if desc["busy"]:
            s.append("busy")
        if p["meter"] >= 1:
            s.append("meter")
        if p["meter"] >= 3:
            s.append("meter3")
        if p["burst"] >= 1:
            s.append("burst")
        if any(move_types.get(m) == "Super" for m in last_moves):
            s.append("after_super")
        if obs.get("detail", {}).get("objects"):
            s.append("projectiles")
        if o["combo"] > 0:
            s.append("comboing")
        return s

    def run(self) -> None:
        game = 0
        while time.monotonic() < self.deadline:
            c1 = CHARS[(self.idx + game) % 5]
            c2 = self.rng.choice(CHARS)
            seed = self.idx * 100000 + game
            game += 1
            try:
                self.play(c1, c2, seed)
            except Crash as e:
                self.on_crash(str(e))
            except EngineError as e:
                self.cov.event("engine_error", worker=self.idx, error=str(e))

    def on_crash(self, why: str) -> None:
        code = self.pool.exit_code(self.idx)
        ctx = getattr(self, "ctx", {})
        tail = []
        try:
            tail = self.log.read()[-40:]
        except Exception:
            pass
        self.cov.event("crash", worker=self.idx, why=why, exit_code=code, log_tail=tail, **ctx)
        with self.cov.lock:
            for c in ctx.get("chars", []):
                self.cov.char(c)["crashes"] += 1
        self.pool.restart(self.idx)
        self.connect()
        if ctx.get("phase") == "preview" and ctx.get("pairs"):
            self.bisect(ctx)

    def bisect(self, ctx: dict) -> None:
        """Replay the match to the crashing decision on the fresh server and preview the batch's
        candidates one at a time to find the exact move + data."""
        try:
            self.replay_to(ctx)
            found = None
            for spec_seat, spec in ctx["specs"]:
                r = self.call({"cmd": "probe", "probes": [dict(spec, seat=spec_seat)]})["results"][0]
                if "id" not in r:
                    continue
                pair = [r["id"], 0] if spec_seat == 0 else [0, r["id"]]
                try:
                    self.call({"cmd": "preview", "pairs": [pair], "candidates": False, "detail": True})
                except Crash:
                    found = {"seat": spec_seat, "spec": spec, "candidate": r["candidate"]}
                    break
            self.cov.event("crash_bisect", worker=self.idx, found=found, seed=ctx.get("seed"),
                           chars=ctx.get("chars"), turn=ctx.get("turn"))
        except Crash as e:
            self.cov.event("crash_bisect", worker=self.idx, found=None, note=f"replay crashed: {e}",
                           seed=ctx.get("seed"), chars=ctx.get("chars"), turn=ctx.get("turn"))
        if self.pool.exit_code(self.idx) is not None or not self._alive():
            self.pool.restart(self.idx)
            self.connect()

    def isolate(self, pairs: list, meta: list) -> list[dict]:
        """Preview a batch's pairs one at a time; return those that log engine errors."""
        out = []
        for pair, (s, spec, cand) in zip(pairs, meta):
            if spec.get("bot"):
                continue
            self.ctx.update(phase="isolate", pairs=[pair], specs=[(s, spec)])
            self.call({"cmd": "preview", "pairs": [pair], "candidates": False, "detail": True})
            errs = self.errors()
            if errs:
                out.append({"seat": s, "spec": spec, "candidate": cand, "pair": pair, "lines": errs[:4]})
        return out

    def _alive(self) -> bool:
        try:
            self.call({"cmd": "hello"})
            return True
        except Crash:
            return False

    def replay_to(self, ctx: dict) -> None:
        c1, c2 = ctx["chars"]
        self.call({"cmd": "new_match", "p1": c1, "p2": c2, "seed": ctx["seed"]})
        for h in ctx["history"]:
            ids = list(h["ids"])
            for seat in (0, 1):
                if h["probe"][seat] is not None:
                    r = self.call({"cmd": "probe", "probes": [dict(h["probe"][seat], seat=seat)]})["results"][0]
                    ids[seat] = r["id"]
            self.call({"cmd": "step", "actions": ids})

    def play(self, c1: str, c2: str, seed: int) -> None:
        cov = self.cov
        chars = [c1, c2]
        self.ctx = {"chars": chars, "seed": seed, "phase": "new_match", "history": []}
        obs = self.call({"cmd": "new_match", "p1": c1, "p2": c2, "seed": seed, "detail": True})
        history: list[dict] = []
        last_moves: list[str] = []
        decision = 0
        while not obs["done"] and time.monotonic() < self.deadline and obs["turn"] < 400:
            decision += 1
            self.ctx = {"chars": chars, "seed": seed, "turn": obs["turn"], "history": list(history), "phase": "moves"}
            desc = self.call({"cmd": "moves"})
            if "error" in desc:
                cov.event("moves_error", error=desc["error"], chars=chars, seed=seed)
                break
            seats = desc["seats"]
            move_types = {}
            for seat in (0, 1):
                for m in seats[seat]["moves"]:
                    move_types[m["name"]] = m["type"]
            # Record legality and situations.
            sit = [self.situations(obs, seats[s], s, last_moves, move_types) for s in (0, 1)]
            with cov.lock:
                for s in (0, 1):
                    ch = cov.char(chars[s])
                    ch["decisions"] += 1
                    ch["situations"].update(sit[s])
                    for m in seats[s]["moves"]:
                        mv = cov.move(chars[s], m)
                        mv["seen"] += 1
                        if m["legal"]:
                            mv["legal"] += 1
                            for i, w in enumerate(m["widgets"]):
                                mv["widgets"].setdefault(f"{i}:{w['name']}", {"kind": w["kind"], "values": set(), "desc": w})
                        else:
                            mv["illegal_usable"][str(m.get("usable"))] += 1
                    for w in seats[s]["extra_widgets"]:
                        ch["extra_widgets"].setdefault(w["name"], {"kind": w["kind"], "values": set(), "visible": 0})
                        if w["visible"]:
                            ch["extra_widgets"][w["name"]]["visible"] += 1
            # Build sweeps and pick a coverage-first sample per seat.
            full = decision % FULL_SWEEP_EVERY == 0
            chosen: list[tuple[int, dict, list[tuple]]] = []
            full_moves: dict[int, str] = {}
            for s in (0, 1):
                sw = sweep_for_seat(seats[s], self.rng)
                with cov.lock:
                    tested = cov.tested[chars[s]]
                    fresh = [x for x in sw if any(k not in tested for k in x[1])]
                    old = [x for x in sw if all(k in tested for k in x[1])]
                self.rng.shuffle(fresh)
                self.rng.shuffle(old)
                pick = (fresh + old)[:PER_SEAT_CAP]
                if full:
                    param = [m for m in seats[s]["moves"] if m["legal"] and any(w.get("visible", True) for w in m["widgets"])]
                    if param:
                        fm = self.rng.choice(param)
                        full_moves[s] = fm["name"]
                        pick += [x for x in sw if x[0]["action"] == fm["name"] and not x[0].get("reverse")
                                 and not x[0].get("extra") and "di" not in x[0] and x not in pick]
                chosen += [(s, spec, keys) for spec, keys in pick]
            self.ctx.update(phase="probe", specs=[(s, spec) for s, spec, _ in chosen])
            res = self.call({"cmd": "probe", "probes": [dict(spec, seat=s) for s, spec, _ in chosen]})["results"]
            # Dedupe candidates; refusals are counted (they show where the UI gates a widget).
            uniq: dict[tuple, tuple[int, int]] = {}
            pairs = []
            meta = []
            with cov.lock:
                for (s, spec, keys), r in zip(chosen, res):
                    ch = cov.char(chars[s])
                    ch["probes"] += 1
                    mvname = spec["action"]
                    mv = ch["moves"].get(mvname)
                    if spec.get("gate"):
                        cov.tested[chars[s]].update(keys)
                        ch["gates"] = ch.get("gates", 0) + 1
                        if "error" not in r:
                            ch["gate_failures"] = ch.get("gate_failures", 0) + 1
                            cov.event("gate_failure", chars=chars, seed=seed, turn=obs["turn"], seat=s, spec=spec,
                                      candidate=r["candidate"])
                        continue
                    if "error" in r:
                        reason = r["error"].split(":")[0] if "illegal" in r["error"] else r["error"]
                        ch["refusals"][reason[:80]] += 1
                        if mv is not None:
                            mv["refusals"][reason[:80]] += 1
                        cov.event("refusal", chars=chars, seed=seed, turn=obs["turn"], seat=s, spec=spec, error=r["error"])
                        continue
                    cand = r["candidate"]
                    k = (s, vkey([cand["action"], cand["data"], cand["extra"]]))
                    cov.tested[chars[s]].update(keys)
                    for key in keys:
                        if key[0] == "w" and mv is not None and key[2] >= 0:
                            wd = mv["widgets"].get(f"{key[2]}:{key[3]}")
                            if wd is not None:
                                wd["values"].add(key[4])
                        elif key[0] == "extra":
                            ch["extra_widgets"].setdefault(key[1], {"kind": "?", "values": set(), "visible": 0})["values"].add(key[2])
                        elif key[0] == "di":
                            ch["di_values"].add(vkey(cand["extra"]["DI"]))
                        elif key[0] == "prediction":
                            ch["predictions"].add(key[1])
                    if mv is not None:
                        mv["probes"] += 1
                        mv["unique_data"].add(vkey([cand["data"], {k2: v for k2, v in cand["extra"].items() if k2 != "DI"}]))
                        if spec.get("feint"):
                            mv["feint"] += 1
                        if spec.get("reverse"):
                            mv["reverse"] += 1
                    if k in uniq:
                        continue
                    uniq[k] = (s, r["id"])
                    pairs.append([r["id"], 0] if s == 0 else [0, r["id"]])
                    meta.append((s, spec, cand))
            # The bot's own candidates for the fully swept moves (discretization comparison).
            disc_index = []
            for s, mname in full_moves.items():
                for c in obs["candidates"][s]:
                    if c["action"] == mname:
                        pairs.append([c["id"], 0] if s == 0 else [0, c["id"]])
                        meta.append((s, {"action": mname, "bot": True}, c))
                        disc_index.append(len(pairs) - 1)
            self.ctx.update(phase="preview", pairs=pairs,
                            specs=[(s, spec) for s, spec, _ in meta if not spec.get("bot")])
            results = self.call({"cmd": "preview", "pairs": pairs, "candidates": False, "detail": True})["results"] if pairs else []
            errs = self.errors()
            culprit = None
            if errs:
                culprits = self.isolate(pairs, meta)
                cov.event("script_error", chars=chars, seed=seed, turn=obs["turn"], lines=errs[:6],
                          situations=sit, history=history, culprits=culprits[:10], n_culprits=len(culprits))
                with cov.lock:
                    for c in chars:
                        cov.char(c)["script_errors"] += 1
                    for cu in culprits:
                        mv = cov.char(chars[cu["seat"]])["moves"].get(cu["spec"]["action"])
                        if mv is not None:
                            mv["script_errors"] = mv.get("script_errors", 0) + 1
                if culprits:
                    culprit = culprits[0]
            # Acceptance: did the fighter enter the move's state?
            p0 = obs["state"]["players"]
            with cov.lock:
                for (s, spec, cand), r in zip(meta, results):
                    ch = cov.char(chars[s])
                    ch["previews"] += 1
                    mv = ch["moves"].get(spec["action"])
                    if mv is None:
                        continue
                    mv["previews"] += 1
                    entered = r["detail"]["entered"][s]
                    if spec["action"] in entered:
                        mv["accepted"] += 1
                    else:
                        mv["not_entered"] += 1
                        ch["not_entered"][spec["action"]] += 1
                    mv["entered_first"][entered[0] if entered else "-"] += 1
            # Discretization records.
            for s, mname in full_moves.items():
                rows = []
                for (ss, spec, cand), r in zip(meta, results):
                    if ss != s or spec["action"] != mname:
                        continue
                    pl = r["state"]["players"]
                    rows.append({
                        "bot": bool(spec.get("bot")), "data": cand["data"],
                        "out": [pl[s]["x"], pl[s]["y"], pl[1 - s]["x"], pl[1 - s]["y"],
                                p0[1 - s]["hp"] - pl[1 - s]["hp"], p0[s]["hp"] - pl[s]["hp"]],
                        "tick": r["state"]["tick"] - obs["state"]["tick"],
                    })
                cov.disc_record({"char": chars[s], "move": mname, "seat": s, "turn": obs["turn"],
                                 "facing": seats[s]["facing"], "situations": sit[s], "rows": rows})
            # Choose the step: often a probe, else random / greedy / super-seeking.
            ids = [0, 0]
            probe_used: list[dict | None] = [None, None]
            by_seat: dict[int, list] = defaultdict(list)
            for (s, spec, cand), r, pr in zip(meta, results, pairs):
                if not spec.get("bot"):
                    by_seat[s].append((spec, pr[s], r))
            for s in (0, 1):
                mode = self.rng.random()
                cands = obs["candidates"][s]
                if culprit is not None:
                    # Step the error-causing probe for real: does the real game log it too?
                    if culprit["seat"] == s:
                        ids[s], probe_used[s] = culprit["pair"][s], culprit["spec"]
                elif mode < 0.4 and by_seat[s]:
                    spec, cid, _ = self.rng.choice(by_seat[s])
                    ids[s], probe_used[s] = cid, spec
                elif mode < 0.6 and by_seat[s]:
                    def score(item: tuple) -> float:
                        pl = item[2]["state"]["players"]
                        return (p0[1 - s]["hp"] - pl[1 - s]["hp"]) - (p0[s]["hp"] - pl[s]["hp"])
                    spec, cid, _ = max(by_seat[s], key=score)
                    ids[s], probe_used[s] = cid, spec
                elif mode < 0.75:
                    supers = [c for c in cands if move_types.get(c["action"]) in ("Super", "Special")]
                    ids[s] = self.rng.choice(supers or cands)["id"]
                else:
                    ids[s] = self.rng.choice(cands)["id"]
            # Preview exactly the stepped pair, then step and compare.
            self.ctx.update(phase="step_preview", pairs=[ids], specs=[])
            pv = self.call({"cmd": "preview", "pairs": [ids], "candidates": False, "detail": True})["results"][0]
            self.ctx.update(phase="step", step_ids=ids, step_probes=probe_used)
            pv_errs = self.errors()
            nxt = self.call({"cmd": "step", "actions": ids, "detail": True})
            errs = self.errors()
            if culprit is not None or errs:
                cov.event("step_error_check", chars=chars, seed=seed, turn=obs["turn"], step=ids, probes=probe_used,
                          culprit_stepped=culprit is not None, preview_errors=pv_errs[:4], step_errors=errs[:6])
            history.append({"ids": ids, "probe": probe_used})
            last_moves = []
            names = []
            for s in (0, 1):
                act = probe_used[s]["action"] if probe_used[s] else obs["candidates"][s][ids[s]]["action"]
                names.append(act)
                last_moves.append(act)
            same = pv["detail"] == nxt["detail"]
            with cov.lock:
                for s in (0, 1):
                    ch = cov.char(chars[s])
                    ch["steps"] += 1
                    if probe_used[s]:
                        ch["probe_steps"] += 1
                    mv = ch["moves"].get(names[s])
                    if mv is not None:
                        mv["stepped"] += 1
                        if not same:
                            mv["divergences"] += 1
                    if not same:
                        ch["divergences"] += 1
            if not same:
                diffs = diff(pv["detail"], nxt["detail"])
                cov.event("divergence", chars=chars, seed=seed, turn=obs["turn"], moves=names, probes=probe_used,
                          situations=sit, diffs=diffs[:12])
            obs = nxt


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
    return [] if a == b else [f"{path}: preview {a!r} step {b!r}"]


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cov = Coverage(out)
    deadline = time.monotonic() + args.minutes * 60
    with Pool(args.processes, base_port=args.base_port) as pool:
        workers = [Worker(i, pool, cov, deadline, args.seed + i) for i in range(args.processes)]
        threads = [threading.Thread(target=w.run, daemon=True) for w in workers]
        for t in threads:
            t.start()
        t0 = time.monotonic()
        while any(t.is_alive() for t in threads):
            time.sleep(30)
            cov.dump()
            with cov.lock:
                prev = sum(c["previews"] for c in cov.chars.values())
                dec = sum(c["decisions"] for c in cov.chars.values()) // 2
            print(f"[{(time.monotonic() - t0) / 60:5.1f} min] decisions {dec} previews {prev} "
                  f"events {dict(cov.counters)} restarts {pool.restarts}", flush=True)
        cov.dump()
        (out / "run.json").write_text(json.dumps({"processes": args.processes, "minutes": args.minutes,
                                                  "restarts": pool.restarts}), encoding="utf-8")


# --- crash reproduction ---------------------------------------------------------------------------

def cmd_repro(args: argparse.Namespace) -> None:
    """Replay a recorded crash (events.jsonl line --event) on a fresh server: the match up to the
    decision, the same probes in the same order (so ids match), then the same preview batch. If it
    crashes, bisect the batch by prefix (restart + replay each time) to the first crashing pair."""
    events = [json.loads(line) for line in open(Path(args.out) / "events.jsonl", encoding="utf-8")]
    crashes = [e for e in events if e["kind"] == "crash"]
    ev = crashes[args.event]
    print("crash:", ev["chars"], "seed", ev["seed"], "turn", ev.get("turn"), "phase", ev.get("phase"),
          "pairs", len(ev.get("pairs") or []))
    with Pool(1, base_port=args.base_port) as pool:
        def attempt(n_pairs: int | None, tries: int = 1) -> bool:
            """True if the server crashed."""
            for _ in range(tries):
                engine = pool.engine(0, timeout=120)
                call = engine.transport.request
                try:
                    c1, c2 = ev["chars"]
                    call({"cmd": "new_match", "p1": c1, "p2": c2, "seed": ev["seed"]})
                    for h in ev["history"]:
                        ids = list(h["ids"])
                        for seat in (0, 1):
                            if h["probe"][seat] is not None:
                                ids[seat] = call({"cmd": "probe", "probes": [dict(h["probe"][seat], seat=seat)]})["results"][0]["id"]
                        call({"cmd": "step", "actions": ids})
                    # The recorded specs are the batch's (deduplicated) probes, one per pair, in pair
                    # order, followed by bot candidates; re-probe and rebuild the pairs from the ids.
                    specs = ev.get("specs") or []
                    pairs = list(ev["pairs"])
                    if specs:
                        res = call({"cmd": "probe", "probes": [dict(sp, seat=s) for s, sp in specs]})["results"]
                        for k, ((s, _), r) in enumerate(zip(specs, res)):
                            pairs[k] = [r["id"], 0] if s == 0 else [0, r["id"]]
                    pairs = pairs if n_pairs is None else pairs[:n_pairs]
                    call({"cmd": "preview", "pairs": pairs, "candidates": False, "detail": True})
                    engine.close()
                except (EngineCrashed, OSError, ValueError):
                    pool.restart(0)
                    return True
            return False
        if not attempt(None, tries=args.tries):
            print(f"not reproduced in {args.tries} replays of the whole batch")
            return
        lo, hi = 0, len(ev["pairs"])   # crash with hi pairs, none with lo
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if attempt(mid, tries=2):
                hi = mid
            else:
                lo = mid
        specs = ev.get("specs") or []
        last = specs[hi - 1] if hi - 1 < len(specs) else ev["pairs"][hi - 1]
        print(f"first crashing prefix: {hi} of {len(ev['pairs'])} pairs; last: {json.dumps(last)}")


# --- report ----------------------------------------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> None:
    from move_coverage_report import write_report

    write_report(Path(args.out), Path(args.md))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--processes", type=int, default=8)
    r.add_argument("--minutes", type=float, default=30)
    r.add_argument("--out", default="runs/coverage")
    r.add_argument("--base-port", type=int, default=7900)
    r.add_argument("--seed", type=int, default=1)
    r.set_defaults(fn=cmd_run)
    rr = sub.add_parser("repro")
    rr.add_argument("--out", default="runs/coverage")
    rr.add_argument("--event", type=int, default=0, help="index among the crash events")
    rr.add_argument("--tries", type=int, default=3)
    rr.add_argument("--base-port", type=int, default=7990)
    rr.set_defaults(fn=cmd_repro)
    rp = sub.add_parser("report")
    rp.add_argument("--out", default="runs/coverage")
    rp.add_argument("--md", default="docs/testing/move-coverage.md")
    rp.set_defaults(fn=cmd_report)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
