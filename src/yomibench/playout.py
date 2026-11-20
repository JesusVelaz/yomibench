"""Playout cap randomization (KataGo's): which decisions are searched cheaply, and the capped
candidate lists a cheap search sees. Self-play uses it per game (`selfplay --cheap-fraction`), a
bot per its own decisions (`make_bot("matrix-do@cheap=P:N")`, for arenas and gates)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from yomibench.engine import Candidate


def cheap_turn(seed: int, turn: int, fraction: float) -> bool:
    """Whether decision `turn` of game `seed` is searched cheaply under playout cap randomization:
    a fixed hash of the seed and turn under `fraction`, like `selfplay.full_previews`, so the split depends on
    nothing else (not the process, the bot, or earlier picks' random draws)."""
    return fraction >= 1 or (fraction > 0 and int(hashlib.sha256(f"cheap:{seed}:{turn}".encode()).hexdigest()[:15], 16)
                             / 16 ** 15 < fraction)


CAP_POLICIES = ("list", "options-first", "rotate", "options-rotate", "options-shuffle")


def _h(*parts: object) -> int:
    return int(hashlib.sha256(":".join(map(str, ("cap",) + parts)).encode()).hexdigest()[:15], 16)


def _leaves(value: Any, path: tuple = ()) -> list[tuple[tuple, Any]]:
    """A candidate's data (or extra) as (path, widget value) leaves: dicts by key, except the dicts a
    data scene's widgets return themselves (an XY plot or 8-way direction {x, y}, a slider {x} or
    {count}, a dropdown {id, name}), which are one leaf each."""
    if isinstance(value, dict) and value and set(value) not in ({"x", "y"}, {"x"}, {"count"}, {"id", "name"}):
        return [leaf for k in sorted(value) for leaf in _leaves(value[k], path + (k,))]
    return [(path, value)]


def _is_mode(path: tuple, values: list[Any]) -> bool:
    """Whether one widget leaf of a move is a mode widget (set 4's split: checkbox, dropdown, block
    height), from the values its variants show: a checkbox is a bool, a dropdown {id, name} or a
    string, a block height {x: 0, y: 0 | 1} under a key of a multi-widget dict. An XY plot, an 8-way
    direction or a slider is not (a top-level {x, y} is always an aim)."""
    if all(isinstance(v, (bool, str)) or v is None for v in values):
        return any(v is not None for v in values)
    if all(isinstance(v, dict) and set(v) == {"id", "name"} for v in values):
        return True
    return bool(path) and all(isinstance(v, dict) and set(v) == {"x", "y"} and v["x"] == 0 and v["y"] in (0, 1)
                              for v in values)


def option_keys(cands: list[Candidate]) -> list[str]:
    """Each candidate's option within its move: its values of the mode widgets (`_is_mode`, judged over
    the move's variants in this list) that vary among those variants, in its data and its extra, as a
    string. Candidates of one move that differ only in aims, directions or sliders share an option."""
    by_move: dict[str, list[int]] = {}
    for i, c in enumerate(cands):
        by_move.setdefault(c.action, []).append(i)
    out = [""] * len(cands)
    for idx in by_move.values():
        leaves = [dict(_leaves(cands[i].data, ("data",)) + _leaves(cands[i].extra, ("extra",))) for i in idx]
        paths = sorted({p for lv in leaves for p in lv}, key=str)
        mode = [p for p in paths if _is_mode(p[1:], [lv.get(p) for lv in leaves])
                and len({json.dumps(lv.get(p), sort_keys=True) for lv in leaves}) > 1]
        for i, lv in zip(idx, leaves):
            out[i] = json.dumps([[list(p), lv.get(p)] for p in mode], sort_keys=True)
    return out


def cap_candidates(cands: Iterable[Candidate], cap: int, policy: str = "list", salt: str = "") -> list[Candidate]:
    """One seat's candidates cut to `cap`. Every Continue variant and one variant of each move are
    always kept (even over the cap); then further variants one per move per pass until the cap.
    Kept candidates stay in list order. A move is a run of candidates with one action. Policies:

    list           (default) the engine's per-seat budget (set 4's rules, `budget`): each move's
                   first variant, then its next variants in list order, passes over the moves in
                   list order, so a cheap decision sees about what `max_candidates` would give. The
                   engine round-robins over move categories (Attack, Special, ...) before moves;
                   candidates don't carry their category, so here every move is its own turn of the
                   round-robin: the same count and the same moves, possibly other data variants.
    options-first  as list, but first the passes add each move's other options (`option_keys`: a
                   distinct value of its mode widgets, e.g. a checkbox's other value), each option's
                   first variant in list order, one per move per pass; only then do list's passes add
                   further variants (other aims, directions, sliders) until the cap.
    rotate         each move's kept variant is `h(salt, move) mod n` of its n variants instead of
                   the first, its next ones follow cyclically, and each pass starts at move
                   `h(salt) mod moves` instead of the list start.
    options-rotate options-first with rotation: each move's always-kept variant is the first variant
                   of its option `h(salt, move) mod options` (not always the first option), its other
                   options follow cyclically, and every pass starts at move `h(salt) mod moves`
                   instead of the list start. A move's further variants still come in list order.
    options-shuffle options-first's order of moves, options and passes, but the variant that stands for
                   each option (and the order its other variants come in) is a seeded shuffle of that
                   option's variants (`h(salt, move, option, variant)`) instead of list order. The
                   kept options are exactly options-first's; only which aim, direction or slider value
                   represents each one changes, uniformly over the option's variants.

    `salt` (the game seed, turn and seat for a cheap decision) makes the rotation a fixed function
    of the decision, like `cheap_turn`; list and options-first ignore it."""
    if policy not in CAP_POLICIES:
        raise ValueError(f"cap policy must be one of {CAP_POLICIES}: {policy!r}")
    cands = list(cands)
    if cap <= 0 or len(cands) <= cap:
        return cands
    moves: list[list[int]] = []
    for i, c in enumerate(cands):
        if moves and cands[moves[-1][0]].action == c.action:
            moves[-1].append(i)
        else:
            moves.append([i])
    rotating = policy in ("rotate", "options-rotate")
    # Per move: the variants its option passes take (first: the one always kept), then the rest.
    phases: list[tuple[list[int], list[int]]] = []
    for m in moves:
        if policy == "options-shuffle":
            phases.append(_shuffled_options(cands, m, salt))
        elif policy in ("options-first", "options-rotate"):
            keys = option_keys([cands[i] for i in m])
            heads = [i for k, i in enumerate(m) if keys[k] not in keys[:k]]
            if policy == "options-rotate":
                r = _h(salt, cands[m[0]].action) % len(heads)
                heads = heads[r:] + heads[:r]
            phases.append((heads, [i for i in m if i not in heads]))
        else:
            r = _h(salt, cands[m[0]].action) % len(m) if policy == "rotate" else 0
            order = m[r:] + m[:r]
            phases.append((order[:1], order[1:]))
    keep = {h[0] for h, _ in phases} | {i for m in moves if cands[m[0]].action == "Continue" for i in m}
    start = _h(salt) % len(moves) if rotating else 0
    phases = phases[start:] + phases[:start]
    for runs in ([h[1:] for h, _ in phases], [rest for _, rest in phases]):
        depth = 0
        while len(keep) < cap and any(len(r) > depth for r in runs):
            for r in runs:
                if len(keep) >= cap:
                    break
                if len(r) > depth:
                    keep.add(r[depth])
            depth += 1
    return [c for i, c in enumerate(cands) if i in keep]


def _shuffled_options(cands: list[Candidate], m: list[int], salt: str) -> tuple[list[int], list[int]]:
    """options-shuffle's phases for one move (its candidate indices `m`): the options in list order of
    their first variant, each represented by the first of its variants in a seeded shuffle; then the
    rest, one per option per round in that order, each option's in its shuffled order."""
    keys = option_keys([cands[i] for i in m])
    groups: dict[str, list[int]] = {}
    for k, i in zip(keys, m):
        groups.setdefault(k, []).append(i)
    action = cands[m[0]].action
    order = [sorted(g, key=lambda i, k=k: _h(salt, action, k, json.dumps([cands[i].data, cands[i].extra], sort_keys=True)))
             for k, g in groups.items()]
    rest = [g[r] for r in range(1, max(map(len, order))) for g in order if r < len(g)]
    return [g[0] for g in order], rest
