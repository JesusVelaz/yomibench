"""Candidate rules hashes, as the server reports them in hello["candidate_rules"].

    python tools/rules_hash.py [MOD_DIR]          # set 4's hash
    python tools/rules_hash.py --all [MOD_DIR]    # set 4's, then each of set 5's 16 switch combinations
    python tools/rules_hash.py --raw ...          # the same before rules_aliases.json

md5 over a rules JSON followed by Candidates4.gd (set 4's frozen generator), each reduced to its
tokens: '#' comments (the .gd only) and every whitespace character outside string literals removed;
the first 12 hex digits. So comment and formatting edits keep the hash; any change to code, strings
or values changes it. A port computes the same function (YomiServer._rules_text, a simulator's
src/rules.rs).

Set 4: candidate_rules_v4.json. Set 5: candidate_rules_v5.json with the switches in its `set5`
object set to the combination played (`"draw_pairs": true` -> false, ...), so each combination has
its own hash; the generator text is still Candidates4.gd, as the simulator hashes it.

Aliases: rules_aliases.json maps a computed hash to the frozen one it stands for, when the files
changed without changing a single list. Candidates4.gd lost its copy of the game's legality loop
(now the game's own ActionButtons panel), so its tokens and
the computed hashes changed while the rules did not; every set-4 and set-5 hash keeps its value.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SET5_KEYS = ("draw_pairs", "gun_aim_fine", "bullet_di", "draw_base")  # bit k of a combination


def strip(text: str, comments: bool) -> str:
    out, quote, i = [], None, 0
    while i < len(text):
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'":
            quote = c
            out.append(c)
        elif comments and c == "#":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        elif not c.isspace():
            out.append(c)
        i += 1
    return "".join(out)


RAW = False  # --raw: skip rules_aliases.json


def aliases(mod_dir: Path) -> dict[str, str]:
    f = mod_dir / "rules_aliases.json"
    return json.loads(f.read_text(encoding="utf-8"))["aliases"] if f.exists() else {}


def _hash(rules_tokens: str, mod_dir: Path) -> str:
    text = rules_tokens + strip((mod_dir / "Candidates4.gd").read_text(encoding="utf-8"), True)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    return h if RAW else aliases(mod_dir).get(h, h)


def rules_hash(mod_dir: Path) -> str:
    """Set 4's hash."""
    return _hash(strip((mod_dir / "candidate_rules_v4.json").read_text(encoding="utf-8"), False), mod_dir)


def set5_switches(bits: int) -> dict[str, bool]:
    return {k: bool(bits >> n & 1) for n, k in enumerate(SET5_KEYS)}


def rules_hash5(mod_dir: Path, switches: dict[str, bool]) -> str:
    """Set 5's hash with these switches (keys not given stay on)."""
    text = strip((mod_dir / "candidate_rules_v5.json").read_text(encoding="utf-8"), False)
    for k in SET5_KEYS:
        on = f'"{k}":true'
        assert text.count(on) == 1, f"candidate_rules_v5.json holds {on} once"
        text = text.replace(on, f'"{k}":{"true" if switches.get(k, True) else "false"}')
    return _hash(text, mod_dir)


if __name__ == "__main__":
    args = sys.argv[1:]
    every = "--all" in args
    RAW = "--raw" in args
    args = [a for a in args if a not in ("--all", "--raw")]
    mod = Path(args[0]) if args else Path(__file__).resolve().parents[1] / "mod" / "YomiBench"
    print(rules_hash(mod) if not every else f"4 {rules_hash(mod)}")
    if every:
        for b in range(16):
            on = [k for k, v in set5_switches(b).items() if v]
            print(f"5 {rules_hash5(mod, set5_switches(b))} {','.join(on) or 'none'}")
