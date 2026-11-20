"""Conformance check for candidate set 4: replay a self-play record on an engine, one engine command
per decision, and compare every decision's base candidate lists with the record's.

    python tools/candidates_replay.py tcp:127.0.0.1:7900 RECORD.jsonl.gz [...]

For each record: `new_match` with the record's characters, seed and budget; then for every
recorded decision, compare both seats' base candidates (action, data, extra; expanded candidates,
which carry `base`, are left out) with the engine's list, rebuild each expanded pick with
`expand` from its recorded `base` and `values` (the rebuilt candidate must equal the recorded
one), and `step` the recorded picks. The record's `engine.candidate_rules` must equal the engine's.
Run it against the real game and against a port: both must reproduce the recorded lists exactly.
Exits non-zero on the first record that differs, printing the first difference.

Replay stays one command per decision on purpose: the real game's server crashed (segfault) when
one command played many steps (deferred deletions only run between commands), so don't fold the
replay into a single server command.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yomibench.engine import Candidate, connect  # noqa: E402


def _base(seat_cands: list[dict]) -> list[dict]:
    return [{k: c[k] for k in ("action", "data", "extra")} for c in seat_cands if "base" not in c]


def _first_diff(want: list[dict], have: list[dict]) -> str:
    i = next((i for i, (a, b) in enumerate(zip(want, have)) if a != b), min(len(want), len(have)))
    return (f"{len(want)} recorded vs {len(have)} replayed; first difference at {i}: "
            f"{want[i] if i < len(want) else None} vs {have[i] if i < len(have) else None}")


def check(engine, path: Path) -> str | None:
    lines = [json.loads(x) for x in gzip.open(path, "rt")]
    head, turns = lines[0], [x for x in lines[1:] if "picks" in x]
    rules, have_rules = head.get("engine", {}).get("candidate_rules"), engine.hello().get("candidate_rules")
    if rules != have_rules:
        return f"candidate_rules differ: record {rules}, engine {have_rules}"
    obs = engine.new_match(*head["characters"], head["seed"], head.get("max_candidates"))
    for k, t in enumerate(turns):
        if obs.done:
            return f"replay ended after {k} of {len(turns)} decisions"
        for seat in (0, 1):
            want = _base(t["candidates"][seat])
            have = [{"action": c.action, "data": c.data, "extra": c.extra} for c in obs.candidates[seat]]
            if want != have:
                return f"decision {k} (turn {t['turn']}) seat {seat}: {_first_diff(want, have)}"
        ids = []
        for seat in (0, 1):
            c = t["candidates"][seat][t["picks"][seat]]
            if "base" in c:
                (got,) = engine.expand([{"seat": seat, "candidate": c["base"], "extra_override": c["values"]}])
                if not isinstance(got, Candidate):
                    return f"decision {k} (turn {t['turn']}) seat {seat}: expand refused the recorded pick: {got}"
                if got.record() != c:
                    return (f"decision {k} (turn {t['turn']}) seat {seat}: the recorded pick rebuilt differently: "
                            f"{c} vs {got.record()}")
                ids.append(got.id)
            else:
                ids.append(t["picks"][seat])
        obs = engine.step(*ids)
    return None


def main() -> int:
    engine = connect(sys.argv[1])
    for p in sys.argv[2:]:
        err = check(engine, Path(p))
        print(f"{p}: {'OK' if err is None else err}", flush=True)
        if err:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
