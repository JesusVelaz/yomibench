"""Checks the real game's `action_space` description against the candidate grids.

    python tools/action_space_check.py --port 7841 [--compare-port 7842] --games GAMES.json --out OUT.json

Plays the games in GAMES.json ([{"p1", "p2", "seed", "set": "5"|"4", "settings": "standard"|"ranked",
"turns": N}]) with random picks on a running server, and at every decision asks `action_space` and
checks:
  (a) every candidate (the base list plus a few `expand`s) lies inside the described space:
      its move is listed (or Continue), its data fits the move's widgets, its extras fit the panel
      (DI inside the wheel, reverse / feint only where the menu allows them);
  (b) what the space offers that the candidates never use: widgets never moved off their default,
      extra controls never set, moves missing from the list;
  (c) the space's legal moves equal the generator's (`moves`: Candidates.legal_states).
With --compare-port, a second server plays the same games without `action_space` and every
observation must match: the description is read-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import socket
import sys
import time
from collections import defaultdict

RANKED = {"stage_width": 600, "game_length": 3000, "starting_meter": "1.5", "ceiling_height": 500, "has_ceiling": True}
TOL = 1.5  # percent: rounding of pixel positions to integer percents


class Client:
    def __init__(self, port: int):
        for _ in range(50):
            self.s = socket.create_connection(("127.0.0.1", port))
            self.f = self.s.makefile("rb")
            if "error" not in self.call({"cmd": "hello"}):
                return
            self.s.close()
            time.sleep(0.2)
        raise RuntimeError(f"server on {port} stays busy")

    def call(self, msg):
        self.s.sendall((json.dumps(msg) + "\n").encode())
        return json.loads(self.f.readline())


def _xy_inside(v, w, mirror: bool) -> str | None:
    if not isinstance(v, dict) or set(v) != {"x", "y"}:
        return f"not an {{x, y}} vector: {v}"
    x, y = v["x"], v["y"]
    if mirror:
        x = -x
    n = math.hypot(x, y)
    # The plot panel bounds the mouse (a half plot's origin sits one pixel inside its edge); a bound
    # near the circle is the circle itself (defaults and keyboard nudges reach full length).
    b = {k: (math.copysign(100.0, v) if abs(v) > 90 else v) for k, v in w["box"].items()}
    if n > 100 + TOL:
        return f"length {n:.1f} > 100"
    if not (b["x_min"] - TOL <= x <= b["x_max"] + TOL and b["y_min"] - TOL <= y <= b["y_max"] + TOL):
        return f"({x}, {y}) outside the panel {b}"
    if w["always_max"] and n < 100 - TOL and n > 0:
        return f"length {n:.1f}, the widget is always at full length"
    if w["min_length"] > 0 and n < w["min_length"] * 100 - TOL:
        return f"length {n:.1f} < min_length {w['min_length']}"
    if w["limit_angle"] and n > 0:
        c = w["limit_center_screen"]
        half = w["limit_range_degrees"] / 2 + 1.0  # + rounding
        a = math.degrees(math.atan2(y, x))
        d = abs((a - c["degrees"] + 180) % 360 - 180)
        if w["limit_symmetrical"]:
            d = min(d, abs((a - c["degrees"]) % 360 - 180))
        if d > half:
            return f"angle {a:.1f} is {d:.1f} from the limit centre {c['degrees']:.1f} (range {w['limit_range_degrees']})"
    return None


def widget_inside(v, w, mirror: bool) -> str | None:
    k = w["kind"]
    if k == "xy":
        return _xy_inside(v, w, mirror)
    if k == "8way":
        vals = [dict(x) for x in w["values"]]
        if mirror and isinstance(v, dict):
            v = {"x": -v["x"], "y": v["y"]}
        return None if v in vals or v == "Neutral" else f"{v} not an enabled direction {w['enabled_screen']}"
    if k in ("slider", "count"):
        if not isinstance(v, dict) or list(v) != [w["key"]]:
            return f"{v} is not {{{w['key']}: n}}"
        x = v[w["key"]]
        return None if w["min"] <= x <= w["max"] else f"{x} outside [{w['min']}, {w['max']}]"
    if k == "check":
        return None if isinstance(v, bool) else f"{v} is not a bool"
    if k == "option":
        ok = [i["id"] for i in w["items"] if not i["disabled"]]
        return None if isinstance(v, dict) and v.get("id") in ok else f"{v} not an enabled item {ok}"
    if k == "height":
        return None if v in w["values"] else f"{v} not high/low {w['values']}"
    return f"unknown widget kind {k}"


def data_inside(data, spec, mirror: bool) -> list[str]:
    if spec is None:
        return [] if data is None else [f"data {data} for a move without a data scene"]
    if spec["shape"] == "single":
        e = widget_inside(data, spec["widget"], mirror)
        return [e] if e else []
    if spec["shape"] == "fields":
        if not isinstance(data, dict) or set(data) != set(spec["fields"]):
            return [f"data keys {list(data) if isinstance(data, dict) else data} != {list(spec['fields'])}"]
        out = []
        for k, w in spec["fields"].items():
            e = widget_inside(data[k], w, mirror)
            if e:
                out.append(f"{k}: {e}")
        return out
    # custom scenes: the same keys as the scene's default; vectors inside their widget when names match
    d0 = spec["default"]
    if isinstance(d0, dict):
        if not isinstance(data, dict) or set(data) != set(d0):
            return [f"custom data keys {list(data) if isinstance(data, dict) else data} != {list(d0)}"]
    elif type(data) is not type(d0) and not (d0 is None):
        return [f"custom data {data} of another type than {d0}"]
    return []


def extra_inside(extra, move, seat_space) -> list[str]:
    out = []
    ex = seat_space["extra"]
    keys = set(ex["keys"]) | {"DI", "reverse", "feint", "prediction"}
    if set(extra) != keys:
        out.append(f"extra keys differ: {sorted(set(extra) ^ keys)}")
    di = extra.get("DI")
    if seat_space["di"] is not None:
        e = _xy_inside(di, seat_space["di"], False)
        if e:
            out.append(f"DI {e}")
    elif di not in ({"x": 0, "y": 0}, None):
        out.append(f"DI {di} with DI disabled")
    if move is not None:
        if extra.get("reverse") and not move["reversible"]:
            out.append("reverse where the menu disables it")
        if extra.get("feint") and not move["feint"]:
            out.append("feint where the menu disables it")
    elif extra.get("reverse") or extra.get("feint"):
        out.append("reverse/feint on Continue")
    for k, wname in ex["key_widget"].items():
        v = extra.get(k)
        if wname and isinstance(v, dict) and set(v) == {"x", "y"} and ex["widgets"].get(wname, {}).get("kind") == "xy":
            e = _xy_inside(v, ex["widgets"][wname], False)
            if e:
                out.append(f"extra {k} ({wname}): {e}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--compare-port", type=int)
    ap.add_argument("--games", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", help="write the first action_space answer per character here")
    a = ap.parse_args()
    games = json.load(open(a.games))
    c = Client(a.port)
    cb = Client(a.compare_port) if a.compare_port else None
    stats = {"decisions": 0, "seat_decisions": 0, "candidates": 0, "expanded": 0, "outside": 0,
             "outside_examples": [], "legal_mismatch": 0, "legal_examples": [], "readonly_mismatch": 0,
             "space_moves": 0, "space_moves_only_with_extra": 0}
    # (b): (character, move, widget) -> {"seen": decisions visible, "values": set of values candidates sent}
    widgets: dict = defaultdict(lambda: {"seen": 0, "values": set(), "default": None})
    extras: dict = defaultdict(lambda: {"seen": 0, "values": set(), "default": None, "controllable": 0, "control": None})
    controls: dict = defaultdict(lambda: {"visible": 0, "kind": None, "requires": None, "shown_by": set()})
    missing_moves: dict = defaultdict(int)
    samples = {}
    kinds: dict = defaultdict(int)
    t0 = time.time()
    for g in games:
        rng = random.Random(g["seed"])
        msg = {"cmd": "new_match", "p1": g["p1"], "p2": g["p2"], "seed": g["seed"], "candidate_sets": [g["set"]] * 2}
        if g["settings"] == "ranked":
            msg["settings"] = RANKED
        o = c.call(msg)
        if cb:
            ob = cb.call(msg)
            stats["readonly_mismatch"] += ob != o
        chars = [g["p1"], g["p2"]]
        for _ in range(g["turns"]):
            if "error" in o or o["done"]:
                break
            space = c.call({"cmd": "action_space"})["seats"]
            moves = c.call({"cmd": "moves"})["seats"]
            stats["decisions"] += 1
            for seat in range(2):
                ss = space[seat]
                ch = chars[seat]
                if ch not in samples and o["turn"] >= 20:  # a mid-game decision, not the opening
                    samples[ch] = {"turn": o["turn"], "state": o["state"]["players"][seat], "space": ss}
                stats["seat_decisions"] += 1
                by_name = {m["name"]: m for m in ss["moves"]}
                base_legal = sorted(m["name"] for m in ss["moves"] if "requires_extra" not in m)
                stats["space_moves"] += len(ss["moves"])
                stats["space_moves_only_with_extra"] += len(ss["moves"]) - len(base_legal)
                gen_legal = sorted(m["name"] for m in moves[seat]["moves"] if m["legal"])
                if base_legal != gen_legal:
                    stats["legal_mismatch"] += 1
                    if len(stats["legal_examples"]) < 10:
                        stats["legal_examples"].append({"char": ch, "turn": o["turn"], "space": base_legal, "generator": gen_legal})
                cands = list(o["candidates"][seat])
                # a few expands on random candidates, so factor values (DI, aim, ...) are covered too
                specs = []
                for cd in rng.sample(cands, min(4, len(cands))):
                    fs = cd.get("factors") or {}
                    if fs:
                        k = rng.choice(sorted(fs))
                        v = fs[k]
                        val = rng.choice(v) if isinstance(v, list) and v else (not cd.get("values", {}).get(k, False))
                        specs.append({"seat": seat, "candidate": cd["id"], "extra_override": {k: val}})
                if specs:
                    for r in c.call({"cmd": "expand", "specs": specs})["results"]:
                        if "candidate" in r:
                            cands.append(r["candidate"])
                            stats["expanded"] += 1
                used_moves = set()
                for cd in cands:
                    stats["candidates"] += 1
                    mv = by_name.get(cd["action"])
                    errs = []
                    if cd["action"] != "Continue" and mv is None:
                        errs.append(f"move {cd['action']} not in the space")
                    else:
                        used_moves.add(cd["action"])
                        mirror = bool(cd["extra"].get("reverse"))
                        if mv is not None:
                            errs += data_inside(cd["data"], mv["data"], mirror)
                        errs += extra_inside(cd["extra"], mv, ss)
                    if errs:
                        stats["outside"] += 1
                        kinds[f"{ch} {cd['action']}: {errs[0].split(':')[0][:60]}"] += 1
                        if len(stats["outside_examples"]) < 25:
                            stats["outside_examples"].append({"char": ch, "turn": o["turn"], "action": cd["action"],
                                                              "data": cd["data"], "extra": cd["extra"], "why": errs})
                    # (b) coverage
                    if mv is not None and mv["data"] is not None:
                        spec = mv["data"]
                        if spec["shape"] == "single":
                            widgets[(ch, cd["action"], spec["widget"]["name"])]["values"].add(json.dumps(cd["data"], sort_keys=True))
                        elif spec["shape"] == "fields" and isinstance(cd["data"], dict):
                            for k in spec["fields"]:
                                widgets[(ch, cd["action"], k)]["values"].add(json.dumps(cd["data"].get(k), sort_keys=True))
                        else:
                            widgets[(ch, cd["action"], "<custom>")]["values"].add(json.dumps(cd["data"], sort_keys=True))
                    for k, v in cd["extra"].items():
                        extras[(ch, k)]["values"].add(json.dumps(v, sort_keys=True))
                for m in ss["moves"]:
                    if m["name"] not in used_moves:
                        missing_moves[(ch, m["name"], m.get("requires_extra"))] += 1
                    if m["data"] is not None:
                        spec = m["data"]
                        ws = ([spec["widget"]] if spec["shape"] == "single" else
                              list(spec["fields"].values()) if spec["shape"] == "fields" else [{"name": "<custom>", "visible": True}])
                        for w in ws:
                            if w.get("visible", True):  # a hidden widget offers nothing
                                widgets[(ch, m["name"], w["name"])]["seen"] += 1
                for k, v in ss["extra"]["keys"].items():
                    extras[(ch, k)]["seen"] += 1
                    extras[(ch, k)]["default"] = json.dumps(v, sort_keys=True)
                    wname = ss["extra"]["key_widget"].get(k)
                    if wname:
                        extras[(ch, k)]["controllable"] += 1
                        extras[(ch, k)]["control"] = wname
                for wname, w in ss["extra"]["widgets"].items():
                    cc = controls[(ch, wname)]
                    cc["visible"] += 1
                    cc["kind"] = w["kind"]
                    cc["requires"] = ss["extra"]["requires"].get(wname, cc["requires"])
                    cc["shown_by"].update(ss["extra"]["shown_by"].get(wname, []))
            n0, n1 = len(o["candidates"][0]), len(o["candidates"][1])
            pair = [rng.randrange(n0), rng.randrange(n1)]
            o = c.call({"cmd": "step", "actions": pair})
            if cb:
                ob = cb.call({"cmd": "step", "actions": pair})
                stats["readonly_mismatch"] += ob != o
        print(f"{g['p1']}-{g['p2']} set {g['set']} {g['settings']}: {stats['decisions']} decisions, "
              f"outside {stats['outside']}, legal mismatches {stats['legal_mismatch']}, "
              f"read-only mismatches {stats['readonly_mismatch']}, {time.time() - t0:.0f}s", flush=True)
    never_varied = sorted(
        ({"char": k[0], "move": k[1], "widget": k[2], "decisions_visible": v["seen"], "distinct_values": len(v["values"])}
         for k, v in widgets.items() if v["seen"] and len(v["values"]) <= 1),
        key=lambda r: (r["char"], r["move"]))
    extra_unused = sorted(
        ({"char": k[0], "key": k[1], "decisions": v["seen"], "distinct_values": len(v["values"]),
          "decisions_a_visible_control_drove_it": v["controllable"], "control": v["control"]}
         for k, v in extras.items() if v["seen"] and len(v["values"]) <= 1 and k[1] not in ("prediction",)),
        key=lambda r: (r["char"], r["key"]))
    stats["b_widgets_never_varied"] = never_varied
    stats["b_extra_keys_never_set"] = extra_unused
    stats["b_moves_never_listed"] = [{"char": k[0], "move": k[1], "requires_extra": k[2], "decisions": n}
                                     for k, n in sorted(missing_moves.items(), key=lambda kv: -kv[1])]
    stats["outside_by_kind"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))
    stats["extra_controls_seen"] = [
        {"char": k[0], "control": k[1], "kind": v["kind"], "decisions_visible": v["visible"], "requires": v["requires"],
         "shown_by_moves": len(v["shown_by"]), "shown_with_continue": "Continue" in v["shown_by"]}
        for k, v in sorted(controls.items())]
    stats["seconds"] = round(time.time() - t0)
    json.dump(stats, open(a.out, "w"), indent=1)
    if a.sample:
        json.dump(samples, open(a.sample, "w"), indent=1)
    print(json.dumps({k: v for k, v in stats.items() if not isinstance(v, list)}))


if __name__ == "__main__":
    sys.exit(main())
