"""Markdown report for a tools/move_coverage.py run (coverage.json, events.jsonl, discretization.jsonl).

    uv run python tools/move_coverage.py report --out runs/coverage --md docs/testing/move-coverage.md
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

AIR = {0: "grounded", 1: "aerial", 2: "both"}   # CharacterState.AirType: Grounded, Aerial, Both


def nvals(w: dict) -> int:
    v = w.get("values")
    if isinstance(v, dict):
        return v.get("count", 0)
    return len(v or [])


def possible(w: dict) -> str:
    d = w.get("desc", {})
    k = w["kind"]
    if k == "xy":
        return "disk"
    if k == "8way":
        return str(len(d.get("enabled", [])) or 9)
    if k in ("slider", "count"):
        step = d.get("step") or 1
        return str(int(round((d.get("max", 0) - d.get("min", 0)) / step)) + 1)
    if k in ("check", "toggle", "height"):
        return "2"
    if k == "option":
        return str(len(d.get("items", [])))
    return "?"


def find_xy(data):
    """The first {x, y} pair in a move's data (XY plot or 8-way)."""
    if isinstance(data, dict):
        if "x" in data and "y" in data and isinstance(data["x"], (int, float)):
            return data["x"], data["y"]
        for v in data.values():
            r = find_xy(v)
            if r:
                return r
    return None


def leaves(data, path: str = "") -> dict:
    """Numeric slider-like leaves ({"x": n} without a "y"), keyed by path."""
    out = {}
    if isinstance(data, dict):
        if "x" in data and "y" not in data and isinstance(data["x"], (int, float)) and len(data) == 1:
            out[path] = data["x"]
        elif "count" in data and isinstance(data["count"], (int, float)) and len(data) == 1:
            out[path] = data["count"]
        else:
            for k, v in data.items():
                out.update(leaves(v, f"{path}.{k}"))
    return out


def never_legal_reason(m: dict) -> str:
    u = m.get("illegal_usable", {})
    t, f = u.get("True", 0), u.get("False", 0)
    if t == 0 and f > 0:
        return "is_usable() was false at every decision: needs a resource or condition never met (meter, an object out, a stance or install)"
    if t > 0:
        return f"usable at {t} of {t + f} decisions but never in the cancel list of a state we reached"
    return "not seen"


def discretization(path: Path) -> tuple[list[dict], dict]:
    per: dict[tuple, list] = defaultdict(list)
    best_angles: dict[tuple, Counter] = defaultdict(Counter)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        bot = [x for x in r["rows"] if x["bot"]]
        dense = [x for x in r["rows"] if not x["bot"]]
        if not bot or len(dense) < 5:
            continue
        key = (r["char"], r["move"])
        # The full 7-direction set (Candidates.XY_ANGLES at full length), taken from the dense rows
        # (their 32-angle grid contains those angles), plus the bot's default variant.
        # Sliders/counts: the generator's min / midpoint / max (of the values seen in the sweep).
        sl: dict[str, set] = defaultdict(set)
        for d in dense:
            for path, v in leaves(d["data"]).items():
                sl[path].add(v)
        disc = {p: {min(v), max(v), min(v) + round((max(v) - min(v)) / 2)} for p, v in sl.items()}
        seven = [bot[0]]
        for d in dense:
            xy = find_xy(d["data"])
            if xy:
                ang = math.degrees(math.atan2(-xy[1], xy[0] * r["facing"])) % 360
                if math.hypot(*xy) < 95 or not any(min(abs(ang - a % 360), 360 - abs(ang - a % 360)) < 3
                                                   for a in (0, 45, 90, 180, -45, 135, -90)):
                    continue
            lv = leaves(d["data"])
            if lv and all(v in disc[p] for p, v in lv.items()):
                seven.append(d)
            elif xy and not lv:
                seven.append(d)
        net = lambda x: x["out"][4] - x["out"][5]  # noqa: E731
        bmax = max(net(x) for x in bot)
        smax = max(net(x) for x in seven)
        dmax = max(net(x) for x in dense)

        def gap(ref):
            return [min(math.hypot(d["out"][0] - b["out"][0], d["out"][1] - b["out"][1]) for b in ref) for d in dense]
        g, g7 = gap(bot), gap(seven)
        per[key].append({"extra": dmax - bmax, "extra7": dmax - smax, "gap_max": max(g), "gap7_max": max(g7),
                         "n_bot": len(bot), "n_seven": len(seven), "n_dense": len(dense)})
        if dmax > bmax:
            for d in dense:
                if d["out"][4] - d["out"][5] == dmax:
                    xy = find_xy(d["data"])
                    if xy and (xy[0] or xy[1]):
                        ang = math.degrees(math.atan2(-xy[1], xy[0] * r["facing"]))
                        best_angles[key][int(round(ang / 22.5)) * 22.5 % 360] += 1
    rows = []
    for key, recs in per.items():
        missed = [x for x in recs if x["extra"] > 0]
        missed7 = [x for x in recs if x["extra7"] > 0]
        rows.append({
            "missed7_pct": 100 * len(missed7) / len(recs),
            "extra7_dmg": statistics.mean(x["extra7"] for x in missed7) if missed7 else 0,
            "gap7_median": statistics.median(x["gap7_max"] for x in recs),
            "n_seven": statistics.mean(x["n_seven"] for x in recs),
            "char": key[0], "move": key[1], "states": len(recs),
            "missed_pct": 100 * len(missed) / len(recs),
            "extra_dmg": statistics.mean(x["extra"] for x in missed) if missed else 0,
            "gap_median": statistics.median(x["gap_max"] for x in recs),
            "gap_p90": sorted(x["gap_max"] for x in recs)[int(0.9 * (len(recs) - 1))],
            "n_bot": statistics.mean(x["n_bot"] for x in recs),
            "n_dense": statistics.mean(x["n_dense"] for x in recs),
            "angles": [a for a, _ in best_angles[key].most_common(4)],
        })
    rows.sort(key=lambda r: (-r["missed_pct"], -r["gap_median"]))
    return rows, {}


def write_report(out: Path, md: Path) -> None:
    cov = json.loads((out / "coverage.json").read_text(encoding="utf-8"))
    run = json.loads((out / "run.json").read_text(encoding="utf-8")) if (out / "run.json").exists() else {}
    events = [json.loads(l) for l in open(out / "events.jsonl", encoding="utf-8")]
    chars = cov["chars"]
    L: list[str] = []
    w = L.append
    w("# Move and parameter coverage (real-game engine server)\n")
    w(f"Run: `tools/move_coverage.py run --processes {run.get('processes', '?')} --minutes {run.get('minutes', '?')}` "
      f"(server restarts {sum(run.get('restarts', []) or [0])}). Every probe is built by the move's own data scene "
      "(`probe` sets its widgets; data is the scene's `get_data()`), previewed against the opponent's Continue with "
      "the states entered recorded, and the engine log is checked after each batch. A sampled pair is stepped each "
      "decision (often a probe) and its preview compared with the real step.\n")
    # --- summary table
    w("## Coverage per character\n")
    w("| character | decisions | moves | ever legal | swept | widget values tested | extra-widget values | DI values | previews | probe steps | gate checks (failed) | crashes | divergences | script-error batches |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(chars):
        ch = chars[c]
        moves = ch["moves"]
        legal = [m for m in moves.values() if m["legal"] > 0]
        swept = [m for m in moves.values() if m["accepted"] > 0]
        wv = sum(nvals(x) for m in moves.values() for x in m["widgets"].values())
        ev = sum(nvals(x) for x in ch["extra_widgets"].values())
        di = nvals({"values": ch["di_values"]})
        w(f"| {c} | {ch['decisions']} | {len(moves)} | {len(legal)} | {len(swept)} | {wv} | {ev} | {di} | {ch['previews']} | "
          f"{ch['probe_steps']} | {ch.get('gates', 0)} ({ch.get('gate_failures', 0)}) | {ch['crashes']} | {ch['divergences']} | {ch['script_errors']} |")
    w("")
    w("*moves*: every move in the action menu. *swept*: legal and previewed with at least one probe that entered the "
      "move's state. *widget values tested*: distinct values set per widget, summed over all widgets of all moves. "
      "*gate checks*: values the menu shows disabled or hidden (8-way directions, toggles), which the server must refuse.\n")
    # --- situations
    w("## Situations reached (decisions per label, both seats)\n")
    labels = ["grounded", "airborne", "cornered", "hitstun", "busy", "comboing", "meter", "meter3", "burst", "after_super", "projectiles"]
    w("| character | " + " | ".join(labels) + " |")
    w("|---|" + "---|" * len(labels))
    for c in sorted(chars):
        s = chars[c]["situations"]
        w(f"| {c} | " + " | ".join(str(s.get(x, 0)) for x in labels) + " |")
    w("")
    # --- findings
    crashes = [e for e in events if e["kind"] == "crash"]
    bis = [e for e in events if e["kind"] == "crash_bisect"]
    serr = [e for e in events if e["kind"] == "script_error"]
    steps = [e for e in events if e["kind"] == "step_error_check"]
    divs = [e for e in events if e["kind"] == "divergence"]
    gates = [e for e in events if e["kind"] == "gate_failure"]
    w("## Crashes, engine errors, divergences\n")
    w(f"- **Crashes**: {len(crashes)}." + ("" if not crashes else " Bisected: " + "; ".join(
        f"{b['chars']} turn {b['turn']}: {json.dumps(b['found'])[:300] if b.get('found') else b.get('note', 'not reproduced')}" for b in bis)))
    w(f"- **Gate failures** (a disabled/hidden widget value accepted): {len(gates)}.")
    culprits: Counter = Counter()
    for e in serr:
        for cu in e.get("culprits", []):
            culprits[(e["chars"][cu["seat"]], cu["spec"]["action"], cu["lines"][0][:90] if cu.get("lines") else e["lines"][0][:90])] += 1
    w(f"- **Engine errors after a batch**: {len(serr)} batches. Isolated culprits (character, move, first error line, batches):")
    for (c, mv, line), n in culprits.most_common(20):
        w(f"  - {c} `{mv}`: `{line}` ({n})")
    real_too = sum(1 for e in steps if e.get("culprit_stepped") and e.get("step_errors"))
    w(f"  - The culprit was then stepped for real {sum(1 for e in steps if e.get('culprit_stepped'))} times; the real "
      f"step logged the same errors {real_too} times (so these are the game's own behavior, not preview artifacts).")
    by_moves: Counter = Counter()
    for e in divs:
        by_moves[" / ".join(e["moves"])] += 1
    total_steps = sum(ch["steps"] for ch in chars.values()) // 2
    w(f"- **Preview vs step divergences**: {len(divs)} of {total_steps} stepped pairs "
      f"({100 * len(divs) / max(total_steps, 1):.2f}%). By moves played: " +
      ", ".join(f"{k} ({n})" for k, n in by_moves.most_common(15)))
    w("")
    if divs:
        w("<details><summary>Divergence examples</summary>\n")
        for e in divs[:15]:
            w(f"- {e['chars']} seed {e['seed']} turn {e['turn']} `{e['moves']}` probes {json.dumps(e.get('probes'))[:200]}: "
              f"{'; '.join(e['diffs'][:4])[:400]}")
        w("\n</details>\n")
    # --- never legal
    w("## Moves never legal in testing\n")
    w("| character | move | type | air | reason |")
    w("|---|---|---|---|---|")
    for c in sorted(chars):
        for name, m in sorted(chars[c]["moves"].items()):
            if m["legal"] == 0:
                w(f"| {c} | {name} | {m['type']} | {AIR.get(m['air_type'], m['air_type'])} | {never_legal_reason(m)} |")
    w("")
    # --- legal but some probes didn't enter
    w("## Probes that did not enter the move's state\n")
    w("Most are moves whose first state is a router or that are replaced on entry (the first state entered is shown).\n")
    w("| character | move | previews | not entered | first states entered |")
    w("|---|---|---|---|---|")
    for c in sorted(chars):
        for name, m in sorted(chars[c]["moves"].items()):
            if m["not_entered"]:
                w(f"| {c} | {name} | {m['previews']} | {m['not_entered']} | {', '.join(f'{k} {v}' for k, v in list(m['entered_first'].items())[:4])} |")
    w("")
    # --- discretization
    if (out / "discretization.jsonl").exists():
        rows, _ = discretization(out / "discretization.jsonl")
        w("## Does the bot's discretization miss outcomes?\n")
        w("At every 8th decision one parameterized move per seat was swept completely next to the bot's own candidates for "
          "that move (which the per-seat budget may have cut to fewer variants). Outcome = state at the next decision. "
          "*missed*: share of states where some dense value did more net damage (dealt minus taken) than every bot "
          "variant; *extra*: mean extra net damage then; *gap*: largest distance from a dense outcome's own position to the "
          "nearest bot variant's (px, median and p90 over states); *best angles*: relative to facing (0 = forward, 90 = up) "
          "of the dense values that found the extra damage.\n")
        w("Two references: the bot's candidates as served (after the 30-per-seat budget) and the full 7-direction set "
          "(for XY moves; default + the 7 angles at full length, as if the budget allowed every variant).\n")
        w("| character | move | states | bot variants | missed (bot) | extra dmg | gap median (bot) | 7-dir variants | missed (7-dir) | extra dmg | gap median (7-dir) | dense values | best angles |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            if r["states"] < 3:
                continue
            w(f"| {r['char']} | {r['move']} | {r['states']} | {r['n_bot']:.1f} | {r['missed_pct']:.0f}% | {r['extra_dmg']:.0f} | "
              f"{r['gap_median']:.0f} | {r['n_seven']:.1f} | {r['missed7_pct']:.0f}% | {r['extra7_dmg']:.0f} | {r['gap7_median']:.0f} | "
              f"{r['n_dense']:.0f} | {', '.join(str(int(a)) for a in r['angles'])} |")
        w("")
    # --- per character detail
    w("## Per-character detail\n")
    for c in sorted(chars):
        ch = chars[c]
        w(f"### {c}\n")
        w("| move | type | legal | probes | previews | distinct data | widgets (values tested / possible) | feint | reverse | not entered | divergences | script errors |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, m in sorted(ch["moves"].items(), key=lambda kv: (kv[1]["type"], kv[0])):
            ws = ", ".join(f"{k.split(':', 1)[1]} {x['kind']} {nvals(x)}/{possible(x)}" for k, x in m["widgets"].items()) or "-"
            ud = m["unique_data"]["count"] if isinstance(m["unique_data"], dict) else m["unique_data"]
            tag = " (grab)" if m.get("grab") else ""
            w(f"| {name}{tag} | {m['type']} | {m['legal']}/{m['seen']} | {m['probes']} | {m['previews']} | {ud} | {ws} | "
              f"{m['feint']} | {m['reverse']} | {m['not_entered']} | {m['divergences']} | {m.get('script_errors', 0)} |")
        w("")
        ex = ", ".join(f"{k} ({x['kind']}) {nvals(x)} values, visible at {x.get('visible', 0)} decisions"
                       for k, x in ch["extra_widgets"].items())
        w(f"Extras: {ex or '-'}. Predictions tried: {ch['predictions']}. Refusals (UI-gated values): "
          f"{json.dumps(ch['refusals'])[:400]}\n")
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {md}")
