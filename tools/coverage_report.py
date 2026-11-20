"""Per-character move coverage in self-play or gate records: for each character and move, the
share of that character's actionable seat-turns where the move was offered (any candidate with that
action, base or expanded), how often it was picked, and the pick rate when offered, on all turns and
on full-search turns. Lists moves offered but never picked. Reads record folders, globs or tars.

    python tools/coverage_report.py RECORDS... [--json OUT.json] [--min-offered PCT]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable, Iterator


def _records(paths: Iterable[str]) -> Iterator[list[dict[str, Any]]]:
    for a in paths:
        hits = glob.glob(a) if any(ch in a for ch in "*?[") else [a]
        for h in hits:
            p = Path(h)
            if p.is_dir():
                for f in sorted(p.rglob("g*.jsonl.gz")):
                    yield [json.loads(x) for x in gzip.open(f, "rt")]
            elif p.suffix == ".tar":
                with tarfile.open(p) as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.endswith(".jsonl.gz"):
                            with gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt") as f:
                                yield [json.loads(x) for x in f]
            elif p.name.endswith(".jsonl.gz"):
                yield [json.loads(x) for x in gzip.open(p, "rt")]


def coverage(records: Iterable[list[dict[str, Any]]]) -> dict[str, Any]:
    """{"games": N, "characters": {char: {"turns", "full_turns", "moves": {move: {offered, picked,
    offered_full, picked_full}}}}} over actionable seat-turns with candidates."""
    out: dict[str, Any] = {"games": 0, "characters": {}}
    for lines in records:
        out["games"] += 1
        chars = lines[0]["characters"]
        for t in lines[1:-1]:
            if "picks" not in t:
                continue
            full = t.get("search") == "full"
            for seat in (0, 1):
                if not (t.get("actionable") or [True, True])[seat] or not t["candidates"][seat]:
                    continue
                c = out["characters"].setdefault(chars[seat], {"turns": 0, "full_turns": 0, "moves": {}})
                c["turns"] += 1
                c["full_turns"] += full
                cands = t["candidates"][seat]
                picked = cands[t["picks"][seat]]["action"]
                for name in {x["action"] for x in cands}:
                    m = c["moves"].setdefault(name, collections.Counter())
                    m["offered"] += 1
                    m["offered_full"] += full
                m = c["moves"].setdefault(picked, collections.Counter())
                m["picked"] += 1
                m["picked_full"] += full
    for c in out["characters"].values():
        c["moves"] = {k: dict(v) for k, v in c["moves"].items()}
    return out


def report(cov: dict[str, Any], min_offered: float = 0.0) -> str:
    rows = [f"{cov['games']} games"]
    for name, c in sorted(cov["characters"].items()):
        n = c["turns"]
        rows += ["", f"== {name}: {n} actionable seat-turns ({c['full_turns']} full-search)",
                 f"{'move':24s} {'offered%':>8s} {'picked%':>8s} {'pick|off%':>9s}"]
        moves = sorted(c["moves"].items(), key=lambda kv: -kv[1].get("offered", 0))
        for move, m in moves:
            o, p = m.get("offered", 0), m.get("picked", 0)
            if 100 * o / n < min_offered:
                continue
            rows.append(f"{move:24s} {100 * o / n:8.1f} {100 * p / n:8.2f} {100 * p / o if o else 0:9.1f}")
        never = sorted(k for k, m in c["moves"].items() if m.get("offered") and not m.get("picked"))
        rows.append(f"offered but never picked ({len(never)}): {', '.join(never) or '-'}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="+", help="record folders, .jsonl.gz files, chunk/gate .tar files or globs")
    ap.add_argument("--json", type=Path, help="also write the counts as JSON")
    ap.add_argument("--min-offered", type=float, default=0.0, help="hide moves offered on fewer than this %% of turns")
    a = ap.parse_args(argv)
    cov = coverage(_records(a.records))
    print(report(cov, a.min_offered))
    if a.json:
        a.json.write_text(json.dumps(cov, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
