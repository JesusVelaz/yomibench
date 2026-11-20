"""Per-character x move-type (and x option x value) coverage of a set of records, and the least-tried
taper rule.

    python tools/move_type_coverage.py RECORDS... [--out PREFIX] [--procs 2] [--hub URL]
        [--min-positions 2000] [--error-ratio 1.2] [--floor 0.015] [--min-error-positions 50]

RECORDS are chunk tars, record folders, .jsonl.gz files or globs (the set is "the training window").
For each character and move type (a candidate's action; yomibench.least_tried's grouping), over the
seat-decisions where that seat had a choice:
  legal     full-search decisions where the type was in the engine's full candidate list (a quick
            turn records only its capped list, so legality is counted on full turns);
  offered   decisions (full and quick) where it was in the list the search considered;
  picked    picks by the search (not exploration);
  explored  exploratory picks of any mode; least-tried counts `explore_mode: "least-tried"` alone;
  after     positions right after a pick of the type (the game's next decision);
  mse       the recorded value's squared error against the game's outcome (p1's +1/-1/0) on those
            positions with a full-search value. The value comes from the net that played the game,
            which never trained on it, so it is a held-out error. `x common` divides it by the pooled
            error after the character's common types (those with at least --min-positions after;
            every type when none has).
Option values (yomibench.least_tried.candidate_options: each data widget, a bare XY data, each extra;
generic over every widget) get the same columns per character x move type x option x value, where
`legal` counts full-search decisions on which the value was available and its option had more than
one value among the type's candidates (a real choice), and `after`/`mse` the positions after a pick of
a candidate with that value (no `offered`: quick turns' option lists aren't read).
Taper rule: a type or option value has caught up when `after` >= --min-positions and, on at least
--min-error-positions valued positions, its mse <= --error-ratio x common (the character's common
types). DI (`extra.DI`, a fixed grid offered with many types) is judged once per character per DI value, its
counts pooled over every move type, instead of per type x value. When every legal type, legal option value and DI value of a character has caught up, the JSON's `taper[character]` is {"tapered": true, "floor": --floor}; the
round-3 worker reads it from the hub (`--hub URL` PUTs the JSON as r3-move-coverage.json) and a
tapered character keeps least-tried exploration in a seeded share `floor` of its games, never zero.
Outputs PREFIX.md and PREFIX.json (default: the markdown on stdout). Outcomes are reported, never
used as targets.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yomibench.least_tried import (COVERAGE_HUB_NAME, MODE, TAPER_DEFAULTS, candidate_options, decision,  # noqa: E402
                                   seat_states)

FIELDS = ("legal", "offered", "picked", "explored", "least_tried", "after", "mse_n", "mse_sum")
# DI is a fixed grid of directions offered with many move types: for the taper each DI value is judged once per
# character, its counts pooled over every move type, not per move type x value (user decision 2026-11-18).
DI_OPTION = "extra.DI"


def _sources(paths: Iterable[str]) -> list[str]:
    out: list[str] = []
    for a in paths:
        for h in sorted(glob.glob(a)) if any(ch in a for ch in "*?[") else [a]:
            p = Path(h)
            if p.is_dir():
                files = sorted(str(f) for f in p.rglob("g*.jsonl.gz"))
                out += [";".join(files[i:i + 250]) for i in range(0, len(files), 250)]  # batches of files
            elif p.suffix == ".tar" or p.name.endswith(".jsonl.gz"):
                out.append(str(p))
    return out


def _record_texts(source: str) -> Iterator[str]:
    if source.endswith(".tar"):
        with tarfile.open(source) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".jsonl.gz"):
                    with tf.extractfile(m) as raw:
                        yield gzip.decompress(raw.read()).decode("utf-8")
    else:
        for f in source.split(";"):
            with gzip.open(f, "rt", encoding="utf-8") as g:
                yield g.read()


def _row(d: dict, *keys: str) -> dict:
    for k in keys:
        d = d.setdefault(k, {})
    if not d:
        d.update(dict.fromkeys(FIELDS, 0))
    return d


def _add(stats: dict, char: str, move: str, key: str, n: float = 1) -> None:
    _row(stats, char, move)[key] += n


def record_stats(text: str, stats: dict, totals: dict, ostats: dict | None = None, sstats: dict | None = None) -> None:
    """Add one record (its file's text) to stats[character][move type], totals[character] and, with
    `ostats`, ostats[character][move type][option][value]; with `sstats`, sstats[character][engine state]
    (the state-rarity boost's classes: positions with a choice in that state and their held-out error)."""
    lines = text.splitlines()
    if len(lines) < 2:
        return
    header, last = json.loads(lines[0]), json.loads(lines[-1])
    chars = header["characters"]
    res = last.get("result") or {}
    z = None if not res or res.get("outcome") == "truncated" else (
        0.0 if res.get("winner") is None else (1.0 if res["winner"] == 0 else -1.0))
    after: list[tuple | None] = [None, None]  # each seat's previous pick: (type, its options)
    for line in lines[1:]:
        d = decision(line, middle=True, picked=ostats is not None)
        if d is None:
            continue
        full = d.get("search", "full") == "full" and d.get("strategies") is not None
        sts = seat_states(line) if sstats is not None else (None, None)
        valued = full and d.get("value") is not None and z is not None
        err = (d["value"] - z) ** 2 if valued else 0.0
        cands = None
        if full and ostats is not None:
            cands = json.loads("{" + line[line.rfind('"candidates":'):])["candidates"]
        for seat in (0, 1):
            char = chars[seat]
            t = totals.setdefault(char, dict.fromkeys(("decisions", "full_decisions", "positions", "mse_n", "mse_sum"), 0))
            t["positions"] += 1
            t["mse_n"] += valued
            t["mse_sum"] += err
            if after[seat] is not None:
                move, opts = after[seat]
                rows = [_row(stats, char, move)]
                if ostats is not None:
                    rows += [_row(ostats, char, move, k, v) for k, v in opts.items()]
                for row in rows:
                    row["after"] += 1
                    row["mse_n"] += valued
                    row["mse_sum"] += err
            acts = d["actions"][seat]
            if len(acts) <= 1:
                after[seat] = None
                continue
            t["decisions"] += 1
            t["full_decisions"] += full
            if sstats is not None and sts[seat]:
                r = sstats.setdefault(char, {}).setdefault(sts[seat], {"positions": 0, "mse_n": 0, "mse_sum": 0.0})
                r["positions"] += 1
                r["mse_n"] += valued
                r["mse_sum"] += err
            for move in set(acts):
                _add(stats, char, move, "offered")
                if full:
                    _add(stats, char, move, "legal")
            if cands is not None:  # option values with a real choice among each type's candidates
                by_move: dict[str, list[dict]] = {}
                for c in cands[seat]:
                    by_move.setdefault(c["action"], []).append(candidate_options(c))
                for move, opts in by_move.items():
                    for k in set().union(*opts):
                        values = {o[k] for o in opts if k in o}
                        if len(values) > 1:
                            for v in values:
                                _row(ostats, char, move, k, v)["legal"] += 1
            move = acts[d["picks"][seat]]
            opts = candidate_options(d["picked"][seat]) if ostats is not None else {}
            explored = bool((d.get("explore") or [False, False])[seat])
            least = explored and d.get("explore_mode") == MODE
            rows = [_row(stats, char, move)]
            if ostats is not None:
                rows += [_row(ostats, char, move, k, v) for k, v in opts.items()]
            for row in rows:
                row["explored" if explored else "picked"] += 1
                row["least_tried"] += least
            pick = d.get("explore_pick") or {}
            if least and pick.get("seat") == seat and float(pick.get("boost") or 1) > 1:
                t["least_tried_boosted"] = t.get("least_tried_boosted", 0) + 1
            after[seat] = (move, opts)


def source_stats(source: str) -> dict:
    stats: dict = {}
    totals: dict = {}
    ostats: dict = {}
    sstats: dict = {}
    games = 0
    for text in _record_texts(source):
        record_stats(text, stats, totals, ostats, sstats)
        games += 1
    return {"games": games, "stats": stats, "totals": totals, "ostats": ostats, "sstats": sstats}


def _sum_into(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict):
            _sum_into(dst.setdefault(k, {}), v)
        else:
            dst[k] = dst.get(k, 0) + v


def merge(parts: Iterable[dict]) -> dict:
    out: dict = {"games": 0, "stats": {}, "totals": {}, "ostats": {}, "sstats": {}}
    for p in parts:
        out["games"] += p["games"]
        for key in ("stats", "totals", "ostats", "sstats"):
            _sum_into(out[key], p.get(key) or {})
    return out


def summarize(raw: dict, min_positions: int = TAPER_DEFAULTS["min_positions"],
              error_ratio: float = TAPER_DEFAULTS["error_ratio"], floor: float = TAPER_DEFAULTS["floor"],
              min_error_positions: int = TAPER_DEFAULTS["min_error_positions"]) -> dict:
    """The coverage document: per character its totals, the common reference error and each move
    type's counts, mse, ratio and caught-up flag; `taper` per character (see the module docstring)."""
    chars: dict = {}
    taper: dict = {}
    for char in sorted(raw["stats"]):
        moves, tot = raw["stats"][char], raw["totals"].get(char, {})
        common = sorted(m for m, r in moves.items() if r["after"] >= min_positions)
        pool = common or sorted(moves)
        n = sum(moves[m]["mse_n"] for m in pool)
        common_mse = sum(moves[m]["mse_sum"] for m in pool) / n if n else None
        def judge(r: dict) -> dict:
            mse = r["mse_sum"] / r["mse_n"] if r["mse_n"] else None
            if mse is None or common_mse is None:
                ratio = None
            else:  # no error after common moves: only an error-free type matches it
                ratio = mse / common_mse if common_mse > 0 else (1.0 if mse == 0 else None)
            caught = (r["after"] >= min_positions and r["mse_n"] >= min_error_positions
                      and ratio is not None and ratio <= error_ratio)
            return {**{k: r[k] for k in FIELDS if k != "mse_sum"},
                    "mse": None if mse is None else round(mse, 5),
                    "ratio": None if ratio is None else round(ratio, 3), "caught_up": caught}

        types = {m: judge(r) for m, r in sorted(moves.items(), key=lambda kv: (-kv[1]["legal"], kv[0]))}
        options: dict = {}
        o_legal = o_behind = 0
        behind_opts: list[tuple[int, str]] = []
        di_pool: dict[str, dict] = {}  # DI value -> its rows summed over every move type
        for m, by_opt in sorted((raw.get("ostats") or {}).get(char, {}).items()):
            for k, by_val in sorted(by_opt.items()):
                if not any(r["legal"] for r in by_val.values()):
                    continue  # never a choice: not an option of this type in practice
                if k == DI_OPTION:  # a fixed grid: judged per character per value, pooled over moves (below)
                    for v, r in by_val.items():
                        if r["legal"]:
                            _sum_into(di_pool.setdefault(v, dict.fromkeys(FIELDS, 0)), r)
                    continue
                for v, r in sorted(by_val.items()):
                    if not r["legal"]:
                        continue
                    row = judge(r)
                    del row["offered"]  # quick turns' option lists aren't read
                    options.setdefault(m, {}).setdefault(k, {})[v] = row
                    o_legal += 1
                    if not row["caught_up"]:
                        o_behind += 1
                        behind_opts.append((r["after"], f"{m} {k}={v}"))
        di = {v: judge(r) for v, r in sorted(di_pool.items())}
        for row in di.values():
            del row["offered"]
        di_behind = sorted((r["after"], v) for v, r in di.items() if not r["caught_up"])
        legal = [m for m, t in types.items() if t["legal"] > 0]
        behind = [m for m in legal if not types[m]["caught_up"]]
        chars[char] = {"decisions": tot.get("decisions", 0), "full_decisions": tot.get("full_decisions", 0),
                       "positions": tot.get("positions", 0),
                       "mse_all": round(tot["mse_sum"] / tot["mse_n"], 5) if tot.get("mse_n") else None,
                       "common_types": common, "common_mse": None if common_mse is None else round(common_mse, 5),
                       "legal_types": len(legal), "caught_up": len(legal) - len(behind),
                       "legal_option_values": o_legal, "option_values_caught_up": o_legal - o_behind,
                       "legal_di_values": len(di), "di_values_caught_up": len(di) - len(di_behind),
                       "types": types, "options": options, "di": di}
        taper[char] = {"tapered": bool(legal) and not behind and not o_behind and not di_behind, "floor": floor,
                       "caught_up": len(legal) - len(behind), "legal_types": len(legal),
                       "option_values_caught_up": o_legal - o_behind, "legal_option_values": o_legal,
                       "di_values_caught_up": len(di) - len(di_behind), "legal_di_values": len(di),
                       "behind": sorted(behind, key=lambda m: types[m]["after"]),
                       "options_behind": [name for _, name in sorted(behind_opts)[:50]],
                       "di_behind": [v for _, v in di_behind]}
    state_taper = _state_taper(raw.get("sstats") or {}, raw["totals"], min_positions, error_ratio, min_error_positions)
    return {"format": 1, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "games": raw["games"],
            "state_taper": state_taper,
            "state_boost_retired": bool(state_taper) and all(t["retired"] for t in state_taper.values()),
            "rule": {"min_positions": min_positions, "error_ratio": error_ratio, "floor": floor,
                     "min_error_positions": min_error_positions},
            "characters": chars, "taper": taper,
            "summary": {c: {"legal_types": t["legal_types"], "caught_up": t["caught_up"],
                            "legal_option_values": t["legal_option_values"],
                            "option_values_caught_up": t["option_values_caught_up"],
                            "legal_di_values": t["legal_di_values"], "di_values_caught_up": t["di_values_caught_up"],
                            "tapered": t["tapered"]}
                        for c, t in taper.items()}}


def _state_taper(sstats: dict, totals: dict, min_positions: int, error_ratio: float, min_error_positions: int) -> dict:
    """Per character, the state-rarity boost's taper (the move-type rule on state classes): a class has
    caught up with >= min_positions positions and, on >= min_error_positions valued ones, an error <=
    error_ratio x that of the common classes (those with >= min_positions; every class when none has).
    `retired` once every class of the character has caught up (its boost is then off). Also each
    character's least-tried picks made with a boost > 1 (`boosted_picks`), to watch the boost decay."""
    out: dict = {}
    for char in sorted(sstats):
        rows = sstats[char]
        common = [k for k, r in rows.items() if r["positions"] >= min_positions] or list(rows)
        n = sum(rows[k]["mse_n"] for k in common)
        cm = sum(rows[k]["mse_sum"] for k in common) / n if n else None
        caught, behind = [], []
        for k, r in sorted(rows.items(), key=lambda kv: kv[1]["positions"]):
            mse = r["mse_sum"] / r["mse_n"] if r["mse_n"] else None
            ok = (r["positions"] >= min_positions and r["mse_n"] >= min_error_positions and mse is not None
                  and cm is not None and (mse <= error_ratio * cm if cm > 0 else mse == 0))
            (caught if ok else behind).append(k)
        out[char] = {"caught": sorted(caught), "behind": behind, "classes": len(rows), "retired": bool(rows) and not behind,
                     "boosted_picks": int((totals.get(char) or {}).get("least_tried_boosted", 0)),
                     "positions": {k: rows[k]["positions"] for k in sorted(rows)}}
    return out


def _f(x: Any, nd: int = 3) -> str:
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else f"{x:,}")


def markdown(doc: dict, sources: int) -> str:
    r = doc["rule"]
    out = [f"# Move-type coverage", "",
           f"{doc['games']:,} games from {sources} sources, {doc['created_utc']}. A type has caught up with "
           f"≥ {r['min_positions']:,} positions after it and a held-out value error ≤ {r['error_ratio']}× that "
           f"of common types (on ≥ {r['min_error_positions']} valued positions); a character whose legal types "
           f"have all caught up keeps least-tried exploration in {r['floor'] * 100:g}% of its games.", "",
           "| character | types caught up | option values caught up | DI values caught up | tapered | types still behind (fewest positions first) |",
           "|---|---:|---:|---:|---|---|"]
    for c, t in doc["taper"].items():
        out.append(f"| {c} | {t['caught_up']} of {t['legal_types']} | {t['option_values_caught_up']:,} of "
                   f"{t['legal_option_values']:,} | {t['di_values_caught_up']} of {t['legal_di_values']} | "
                   f"{'yes' if t['tapered'] else 'no'} | "
                   f"{', '.join(t['behind'][:8])}{' …' if len(t['behind']) > 8 else ''} |")
    st = doc.get("state_taper") or {}
    if st:
        out += ["", "State-rarity boost taper (state classes: each seat's own engine state; the same rule as move types)"
                + (": **retired for every character**" if doc.get("state_boost_retired") else "") + ".", "",
                "| character | classes caught up | retired | boosted least-tried picks | classes still behind (fewest positions first) |",
                "|---|---:|---|---:|---|"]
        for c, t in st.items():
            out.append(f"| {c} | {len(t['caught'])} of {t['classes']} | {'yes' if t['retired'] else 'no'} | "
                       f"{t['boosted_picks']:,} | {', '.join(t['behind'][:8])}{' …' if len(t['behind']) > 8 else ''} |")
    for c, ch in doc["characters"].items():
        out += ["", f"## {c}", "",
                f"{ch['decisions']:,} seat-decisions with a choice ({ch['full_decisions']:,} full-search); "
                f"value MSE vs outcome {_f(ch['mse_all'])} on all positions, {_f(ch['common_mse'])} after common "
                f"types ({len(ch['common_types'])}).", "",
                "| move type | legal | offered | picked | explored | least-tried | after | MSE after (n) | × common | caught up |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for m, t in ch["types"].items():
            out.append(f"| {m} | {_f(t['legal'])} | {_f(t['offered'])} | {_f(t['picked'])} | {_f(t['explored'])} | "
                       f"{_f(t['least_tried'])} | {_f(t['after'])} | {_f(t['mse'])} ({t['mse_n']:,}) | "
                       f"{_f(t['ratio'], 2)} | {'yes' if t['caught_up'] else 'no'} |")
        rows = sorted(((t["after"], m, k, v, t) for m, by_opt in ch["options"].items() for k, by_val in by_opt.items()
                       for v, t in by_val.items()), key=lambda r: (r[0], r[1], r[2], r[3]))
        if rows:
            out += ["", f"Option values ({ch['option_values_caught_up']:,} of {ch['legal_option_values']:,} caught up); "
                        f"the {min(40, len(rows))} with the fewest positions after them (all in the JSON):", "",
                    "| move type | option | value | legal | picked | explored | least-tried | after | MSE after (n) | × common |",
                    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for _, m, k, v, t in rows[:40]:
                out.append(f"| {m} | {k} | `{v}` | {_f(t['legal'])} | {_f(t['picked'])} | {_f(t['explored'])} | "
                           f"{_f(t['least_tried'])} | {_f(t['after'])} | {_f(t['mse'])} ({t['mse_n']:,}) | "
                           f"{_f(t['ratio'], 2)} |")
        if ch.get("di"):
            out += ["", f"DI values, pooled over every move type ({ch['di_values_caught_up']} of "
                        f"{ch['legal_di_values']} caught up):", "",
                    "| DI value | legal | picked | explored | least-tried | after | MSE after (n) | × common | caught up |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
            for v, t in sorted(ch["di"].items(), key=lambda kv: (kv[1]["after"], kv[0])):
                out.append(f"| `{v}` | {_f(t['legal'])} | {_f(t['picked'])} | {_f(t['explored'])} | "
                           f"{_f(t['least_tried'])} | {_f(t['after'])} | {_f(t['mse'])} ({t['mse_n']:,}) | "
                           f"{_f(t['ratio'], 2)} | {'yes' if t['caught_up'] else 'no'} |")
    return "\n".join(out) + "\n"


def put_hub(url: str, data: bytes) -> None:
    url = url if url.endswith("/") else url + "/"
    req = urllib.request.Request(url + COVERAGE_HUB_NAME, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=60) as resp:
        answer = resp.read().decode("utf-8", "replace")
    if hashlib.sha256(data).hexdigest() not in answer:
        raise RuntimeError(f"hub answered {answer[:120]!r}, not the sha256")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="+", help="chunk tars, record folders, .jsonl.gz files or globs")
    ap.add_argument("--out", type=Path, help="write PREFIX.md and PREFIX.json (default: markdown on stdout)")
    ap.add_argument("--procs", type=int, default=2, help="worker processes (default 2)")
    ap.add_argument("--hub", help="also PUT the JSON to this hub as " + COVERAGE_HUB_NAME + " (the worker's taper)")
    ap.add_argument("--min-positions", type=int, default=TAPER_DEFAULTS["min_positions"])
    ap.add_argument("--error-ratio", type=float, default=TAPER_DEFAULTS["error_ratio"])
    ap.add_argument("--floor", type=float, default=TAPER_DEFAULTS["floor"])
    ap.add_argument("--min-error-positions", type=int, default=TAPER_DEFAULTS["min_error_positions"])
    a = ap.parse_args(argv)
    if not 0 < a.floor <= 1:
        ap.error("--floor must be in (0, 1]: the taper never stops exploration")
    sources = _sources(a.records)
    if not sources:
        ap.error("no records found")
    t0 = time.monotonic()
    if a.procs > 1 and len(sources) > 1:
        with ProcessPoolExecutor(min(a.procs, len(sources))) as pool:
            raw = merge(pool.map(source_stats, sources))
    else:
        raw = merge(map(source_stats, sources))
    doc = summarize(raw, a.min_positions, a.error_ratio, a.floor, a.min_error_positions)
    doc["sources"] = len(sources)
    doc["seconds"] = round(time.monotonic() - t0, 1)
    md = markdown(doc, len(sources))
    data = json.dumps(doc, indent=1).encode("utf-8")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(a.out) + ".md").write_text(md, encoding="utf-8")
        Path(str(a.out) + ".json").write_bytes(data)
        print(f"{doc['games']} games, {len(sources)} sources, {doc['seconds']}s: wrote {a.out}.md and {a.out}.json")
        for c, t in doc["taper"].items():
            print(f"  {c}: {t['caught_up']}/{t['legal_types']} legal types, {t['option_values_caught_up']}/"
                  f"{t['legal_option_values']} option values and {t['di_values_caught_up']}/{t['legal_di_values']} "
                  f"DI values caught up; tapered: {t['tapered']}")
    else:
        sys.stdout.write(md)
    if a.hub:
        put_hub(a.hub, data)
        print(f"PUT {COVERAGE_HUB_NAME} to {a.hub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
