"""Count what a value net's vocabulary doesn't know in game records (bug catcher for candidate set 5).

    python tools/vocab_check.py --model NET.npz RECORDS_DIR_OR_FILE ... [--json OUT.json]

A value net embeds state names (players' states and stances, objects' kinds and states) through
its vocabulary; a name it never trained on encodes as 0 (unknown). Set 5 adds candidates (Cowboy's
Draw entries and the like) whose actions, once picked, become states the current nets may not know.
For every record this counts, per candidate set and rules hash (the header's engine.candidate_set
and candidate_rules):

  candidates      candidate entries offered at decisions, and how many have an action whose name
                  isn't in the net's `state` vocabulary (with the distinct names); "Continue" (keep
                  doing the current move) is never a state, so it isn't counted
  picks           the same for the candidates actually played
  states          state names in the recorded states, per category, unknown to the net

It reads records only (no engine) and prints one line per group, plus JSON with --json. The exit
code is 0 either way: the counts are for the log, not a gate.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yomibench.valuenet import ValueNet  # noqa: E402


def record_files(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(p.glob("*.jsonl.gz")) if p.is_dir() else [p]
    return out


def state_names(state: dict) -> list[tuple[str, str]]:
    """(category, name) of every name a state gives the value net's embeddings (features.encode's
    categorical slots: each player's state, stance and character, each object's kind and state)."""
    out = []
    for p in state.get("players") or []:
        out += [("state", str(p.get("state", ""))), ("stance", str(p.get("stance", ""))),
                ("char", str(p.get("character", "")))]
    for o in state.get("objects") or []:
        out += [("kind", str(o.get("kind", o.get("name", "")))), ("ostate", str(o.get("state", "")))]
    return [(c, n) for c, n in out if n]


NOT_STATES = frozenset({"Continue"})  # candidate actions that never become a state name


def check(files: list[Path], vocab_ids: dict[str, dict[str, int]]) -> dict:
    known = {c: set(t) for c, t in vocab_ids.items()}
    groups: dict[str, dict] = defaultdict(lambda: {
        "records": 0, "decisions": 0, "candidates": 0, "unknown_candidates": 0, "picks": 0, "unknown_picks": 0,
        "unknown_candidate_actions": Counter(), "unknown_states": Counter()})
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            eng = header.get("engine") or {}
            g = groups[f"set {eng.get('candidate_set')} rules {eng.get('candidate_rules')}"]
            g["records"] += 1
            for line in fh:
                turn = json.loads(line)
                if "candidates" not in turn:
                    continue
                g["decisions"] += 1
                for seat, cands in enumerate(turn["candidates"]):
                    pick = (turn.get("picks") or [None, None])[seat]
                    for i, c in enumerate(cands):
                        action = str(c.get("action"))
                        unknown = action not in NOT_STATES and action not in known.get("state", set())
                        g["candidates"] += 1
                        g["unknown_candidates"] += unknown
                        if unknown:
                            g["unknown_candidate_actions"][action] += 1
                        if i == pick:
                            g["picks"] += 1
                            g["unknown_picks"] += unknown
                for cat, name in state_names(turn.get("state") or {}):
                    if name not in known.get(cat, set()):
                        g["unknown_states"][f"{cat}:{name}"] += 1
    return {k: {**v, "unknown_candidate_actions": dict(v["unknown_candidate_actions"].most_common()),
                "unknown_states": dict(v["unknown_states"].most_common())} for k, v in sorted(groups.items())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="the value net whose vocabulary is checked")
    ap.add_argument("records", type=Path, nargs="+", help="record files or folders of them")
    ap.add_argument("--json", type=Path, help="also write the counts here")
    args = ap.parse_args(argv)
    net = ValueNet.load(args.model)
    out = check(record_files(args.records), net.vocab.ids)
    for k, g in out.items():
        top = ", ".join(f"{n} x{c}" for n, c in list(g["unknown_candidate_actions"].items())[:8])
        print(f"{k}: {g['records']} records, {g['decisions']} decisions; candidates unknown to {args.model.name}'s "
              f"vocabulary: {g['unknown_candidates']} of {g['candidates']} ({len(g['unknown_candidate_actions'])} "
              f"names{': ' + top if top else ''}); picks {g['unknown_picks']} of {g['picks']}; unknown state names "
              f"{sum(g['unknown_states'].values())} ({len(g['unknown_states'])} distinct)")
    if args.json:
        args.json.write_text(json.dumps({"model": str(args.model), "groups": out}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
