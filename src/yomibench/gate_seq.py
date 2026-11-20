"""Preregistered, staged gates: a candidate plays stages of seat-swapped pairs, with
early-reject bounds fixed before the gate starts.

A gate job carries its whole plan before any game is played: every stage's seed, pairs, looks and
thresholds. "Core" is the candidate against the previous best (`mode: quick`), pooled over the
character pairings; the matchups are the pairings. Each pair's score is the candidate's points in
its two seat-swapped games / 2.

Candidate stages (early reject; early accept only where the stage preregisters `accept_z`):
  At look k (25/50/75/100% of the core pairs; the first look also needs >= 8 complete pairs in
  every pairing), with mean m_k over n_k complete pairs and SE_k = sd(pair scores) / sqrt(n_k):
  reject when m_k + z_k * SE_k < 0.5, z = 4.048, 2.862, 2.337, 2.024 (O'Brien-Fleming boundaries
  for 4 looks, two-sided alpha 0.05). Efficacy bound (full search on both sides only, so no
  cheap-vs-full bias; never at the first look): accept (pass) at look k when m_k - a_k * SE_k > 0.5
  and the per-matchup guard holds, a_k from the stage's `accept_z` (null at a look with no accept;
  the whole field null in quick-turn screens, which stay reject-only). At the last look the stage's
  own rule decides: pass on a core point estimate >= pass_min, with the per-matchup guard (no core
  matchup's upper 95% bound below guard_upper) where the stage has one.

Plans:
  two_stage  a quick-turn screen (both bots @cheap=0.75:32, 50 pairs x 4, the looks above), then a
             full-search confirmation on fresh seeds for survivors (25 pairs x 4, one look, promote
             on core >= 0.50)
  full_core  the fallback while quick/full parity is unverified: one full-search stage, 50 pairs x
             4, the looks above, early accept at looks 2 and 3 (accept_z null, 2.862, 2.337, null),
             promote on core >= 0.50 with the guard

Anchor regression checks at promotion (early accept and early reject): per
anchor, on the new best's seat-swapped pair scores against that anchor, at look k (only once every
pairing has >= 8 complete pairs against it): pass when mean - z_k * SE > 0.5, fail (the revert rule)
when mean + z_k * SE < 0.5, else open; the same O'Brien-Fleming z as candidate gates (anchor_verdict()).
Games are correlated within a pair and draws are fractional, so the test is on pairs, not games.

A look is evaluated once, the first time the complete pairs reach it; if pairs arrive past several
looks at once, only the latest reached look is evaluated.

Per-pairing pair counts (focus-weighted gates): a stage may preregister `pairs_by_pairing`
("P1:P2" -> pairs, every pairing of the gate's characters), e.g. Cowboy:Cowboy 60, Cowboy:Ninja 60,
Ninja:Cowboy 60, Ninja:Ninja 32. Its `pairs` is then the total (212), each look is at its fraction of
that total (and still needs min_pairs_per_pairing complete pairs in every pairing), the score at a
look is the mean over all complete pairs, so each pairing weighs by its count (the focus-weighted
mean), with that mean's SE, and the per-matchup guard still applies to every pairing. A stage
without the field is unchanged. pool_eval --pairs-by-pairing plays such a schedule (see
yomibench.arena.pair_schedule for its seeds and the order microshards split it in).

Scores in reasons and printed summaries are win percentages (won()): "won 64% of games (58–71%)" is
the candidate's points in those games, a draw counting half a win, as a whole percentage, with the
95% interval. JSON fields keep the scores as fractions (score, ci95, mean, se).

The Cowboy gate (plan "cowboy", cowboy()): one full-search stage, against the previous best, in
which the candidate plays Cowboy in every game. Its cells:
  Cowboy mirror   `mirror_pairs` ordinary seat-swapped Cowboy:Cowboy pairs (candidate P1, then P2)
  Cowboy v X      for each X in `cowboy_vs` (X -> pairs; Ninja, Wizard, Mutant or Robot), that many
                  candidate-character pairs: the candidate plays Cowboy and the previous best plays X
                  in both games, the candidate P1 in the first (Cowboy:X) and P2 in the second
                  (X:Cowboy), both on the same seed (yomibench.arena.cell_schedule). A pair stays
                  whole: its score is the candidate's points in its two games / 2, as in any pair.
No Ninja mirror. The pairs are interleaved over the whole schedule, so every look has each cell in
proportion. Looks are at 25/50/75/100% of all the stage's pairs; the first also needs
min_pairs_per_pairing (8) complete Cowboy mirror pairs. With mean +- z x SE over whole pairs
(pair_stats), "pooled" being every gate pair (mirror and every Cowboy-vs-X cell, each pair weighted
equally), at look k:
  pass if   (a) mirror mean - a x SE > 0.5, or
            (b) pooled mean - a x SE > 0.5 and mirror mean + g x SE >= 0.5 (the mirror not worse)
                and mirror mean >= mirror_floor (0.47; user 2026-11-18: no pass carried by the Cowboy-
                vs-X cells while the mirror itself loses; a job preregistered without mirror_floor
                keeps the old rule)
            a = accept_z[k] at looks 2 and 3 (2.862, 2.337; none at look 1), final_z (2.024) at the
            final look; g = guard_z (default 1.96)
  block     a Cowboy-vs-X cell with >= 8 complete pairs and mean + g x SE < 0.40 blocks a pass: an
            early accept plays on, the final look rejects. No early accept before every Cowboy-vs-X
            cell has 8 complete pairs (it plays on); at the final look a cell short of 8 is reported
            unguarded and does not block
  reject    early, only when the mirror is clearly worse: mirror mean + reject_z[k] x SE < 0.5;
            never on the pooled score or a Cowboy-vs-X cell alone. Floor futility (floor_futility;
            user 2026-11-18): at looks 2 and 3 a gate also stops and rejects when the conditional
            power of a final mirror mean >= mirror_floor is < cp_min (0.10): the mirror's remaining
            scheduled pairs playing at the current mirror mean, with the pairs' sample SD (floor_cp);
            a job preregistered without floor_futility never stops on it. The final look rejects unless it
            passes (inconclusive only with no complete mirror pair)
Every bound is in the stage dict (plan, mirror_pairs, cowboy_vs, cells, looks, reject_z, accept_z,
final_z, guard_z, cell_block_upper, not_worse, mirror_floor, floor_futility, min_pairs_per_pairing, pooled,
rule).

The per-character floor's extension (plan "cowboy_extension", cowboy_extension(); user decision
2026-11-19): a Cowboy gate job may preregister cell_floor = {floor, extension_pairs} (cell_floor_rule).
When that gate passes, every Cowboy-vs-X cell whose point score is below `floor` plays
`extension_pairs` more candidate-character pairs (both seats, as in the gate), in that cell only, in one
full-search stage with a single, final look (no early stop). Its game numbers continue the gate's
schedule on the gate's seed: after the gate's last candidate-character pair, a block of
`extension_pairs` per X in the gate's cowboy_vs order (extension_first_games), so they are disjoint
from the gate's games and don't depend on which cells are extended. Each extended cell is then judged
over all its complete pairs (the gate's and the extension's; halted or truncated games drop their
pair): below `floor` blocks the promotion (any one cell), else it promotes (extension_verdict).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

LOOKS = (0.25, 0.5, 0.75, 1.0)
REJECT_Z = (4.048, 2.862, 2.337, 2.024)  # O'Brien-Fleming, 4 looks, two-sided alpha 0.05
# full_core's efficacy bound: the same boundaries at looks 2 and 3; none at look 1, and the last look
# decides by the stage's own rule
FULL_CORE_ACCEPT_Z = (None, 2.862, 2.337, None)
MIN_PAIRS_PER_PAIRING = 8


def stage(name: str, turns: str, seed: int, pairs: int | None, looks: Sequence[float] = (1.0,),
          reject_z: Sequence[float] | None = None, min_pairs_per_pairing: int = MIN_PAIRS_PER_PAIRING,
          pass_min: float = 0.5, guard_upper: float | None = None, final: bool = False,
          accept_z: Sequence[float | None] | None = None, pairs_by_pairing: Mapping[str, int] | None = None,
          characters: Sequence[str] | None = None) -> dict[str, Any]:
    """One preregistered stage (validated). `accept_z`: the efficacy bound per look (None where a
    look can't accept); full-search stages only, and never at the first look. `pairs_by_pairing`
    ("P1:P2" -> pairs, needs `characters`): per-pairing pair counts, every pairing of `characters`,
    each enough for min_pairs_per_pairing complete pairs by the first look; `pairs` is then their
    total (None, or that total). Without it the stage is as before."""
    looks = [float(t) for t in looks]
    if turns not in ("cheap", "full"):
        raise ValueError(f"turns is cheap or full, got {turns!r}")
    if not looks or looks[0] <= 0 or abs(looks[-1] - 1) > 1e-12 or any(b <= a for a, b in zip(looks, looks[1:])):
        raise ValueError(f"looks must increase from above 0 to 1, got {looks}")
    if reject_z is not None and len(reject_z) != len(looks):
        raise ValueError("one rejection z per look")
    if accept_z is not None:
        if len(accept_z) != len(looks):
            raise ValueError("one acceptance z (or None) per look")
        if turns != "full":
            raise ValueError("early accept needs full search on both sides (a quick-turn stage stays reject-only)")
        if accept_z[0] is not None:
            raise ValueError("no early accept at the first look")
    if pairs_by_pairing is not None:
        counts = check_pairs_by_pairing(pairs_by_pairing, characters)
        short = {p: c for p, c in counts.items() if c * looks[0] < min_pairs_per_pairing - 1e-9}
        if short:
            raise ValueError(f"too few pairs to reach {min_pairs_per_pairing} complete pairs per pairing by the "
                             f"first look ({looks[0]:g} of each pairing's pairs): {short}")
        if pairs is not None and pairs != sum(counts.values()):
            raise ValueError(f"pairs {pairs} is not the total of pairs_by_pairing ({sum(counts.values())})")
        pairs = sum(counts.values())
    elif pairs is None:
        raise ValueError("pairs is required without pairs_by_pairing")
    if pairs < 1 or not 0 < seed < 10 ** 7:
        raise ValueError(f"pairs must be positive and seed in 1..9999999, got {pairs}, {seed}")
    out = {"name": name, "turns": turns, "seed": int(seed), "pairs": int(pairs), "looks": looks,
           "reject_z": list(reject_z) if reject_z is not None else None,
           "accept_z": list(accept_z) if accept_z is not None else None,
           "min_pairs_per_pairing": int(min_pairs_per_pairing), "pass_min": pass_min, "guard_upper": guard_upper,
           "final": final}
    if pairs_by_pairing is not None:
        out["pairs_by_pairing"] = counts
    return out


def check_pairs_by_pairing(pairs_by_pairing: Mapping[str, int], characters: Sequence[str] | None) -> dict[str, int]:
    """`pairs_by_pairing` ("P1:P2" -> pairs) checked against the gate's characters: exactly their
    pairings, each a positive count. Returns it in pairing order (P1 major, as games are scheduled)."""
    if not characters:
        raise ValueError("pairs_by_pairing needs the gate's characters")
    want = [f"{a}:{b}" for a in characters for b in characters]
    missing, extra = [p for p in want if p not in pairs_by_pairing], sorted(set(pairs_by_pairing) - set(want))
    if missing or extra:
        raise ValueError(f"pairs_by_pairing must name every pairing of {','.join(characters)} once"
                         + (f"; missing {missing}" if missing else "") + (f"; not a pairing {extra}" if extra else ""))
    bad = {p: c for p, c in pairs_by_pairing.items() if not isinstance(c, int) or isinstance(c, bool) or c < 1}
    if bad:
        raise ValueError(f"pairs_by_pairing counts must be positive integers: {bad}")
    return {p: int(pairs_by_pairing[p]) for p in want}


def parse_pairs_by_pairing(spec: str) -> dict[str, int]:
    """'Cowboy:Cowboy=60,Cowboy:Ninja=60,...' -> {"Cowboy:Cowboy": 60, ...} (a CLI flag's value)."""
    out: dict[str, int] = {}
    for part in filter(None, (x.strip() for x in spec.split(","))):
        pairing, eq, count = part.partition("=")
        if not eq or pairing.count(":") != 1 or pairing in out:
            raise ValueError(f"want P1:P2=PAIRS,... with each pairing once, got {spec!r}")
        try:
            out[pairing] = int(count)
        except ValueError:
            raise ValueError(f"{pairing}: {count!r} is not a number of pairs") from None
    return out


def scheduled_pairs(st: Mapping[str, Any], n_pairings: int) -> int:
    """The total pairs a stage schedules: its per-pairing counts' total (or the Cowboy gate's or its
    extension's cells' total), else `pairs` per pairing."""
    if st.get("pairs_by_pairing") or is_cowboy(st) or is_extension(st):
        return int(st["pairs"])
    return n_pairings * int(st["pairs"])


def two_stage(screen_seed: int, confirm_seed: int, screen_pairs: int = 50, confirm_pairs: int = 25) -> list[dict]:
    """The round-4 default: a quick-turn screen with early reject, then a full-search confirmation."""
    if screen_seed == confirm_seed:
        raise ValueError("the confirmation plays fresh seeds")
    return [stage("screen", "cheap", screen_seed, screen_pairs, LOOKS, reject_z=REJECT_Z, guard_upper=0.5),
            stage("confirm", "full", confirm_seed, confirm_pairs, (1.0,), final=True)]


def full_core(seed: int, pairs: int = 50, pairs_by_pairing: Mapping[str, int] | None = None,
              characters: Sequence[str] | None = None) -> list[dict]:
    """The fallback while quick/full parity is unverified: one full-search stage with early reject
    and, at looks 2 and 3, early accept (full search on both sides). With `pairs_by_pairing` (and
    the gate's `characters`), those per-pairing counts instead of `pairs` in every pairing."""
    if pairs_by_pairing is not None:
        return [stage("core", "full", seed, None, LOOKS, reject_z=REJECT_Z, guard_upper=0.5, final=True,
                      accept_z=FULL_CORE_ACCEPT_Z, pairs_by_pairing=pairs_by_pairing, characters=characters)]
    return [stage("core", "full", seed, pairs, LOOKS, reject_z=REJECT_Z, guard_upper=0.5, final=True,
                  accept_z=FULL_CORE_ACCEPT_Z)]


# --- the Cowboy gate (plan "cowboy") ------------------------------------------------------------

COWBOY = "Cowboy"
COWBOY_VS = ("Ninja", "Wizard", "Mutant", "Robot")  # the opponents' characters a Cowboy-vs-X cell may name
MIRROR = "Cowboy mirror"
GUARD_Z = 1.96  # the "not worse" and "clearly worse" checks (95%)
FINAL_Z = REJECT_Z[-1]  # the efficacy z at the final look: the O'Brien-Fleming final bound
CELL_BLOCK_UPPER = 0.40  # a Cowboy-vs-X cell with mean + guard_z x SE below this blocks a pass
NOT_WORSE = 0.5  # (b) needs the Cowboy mirror's mean + guard_z x SE at least this
MIRROR_FLOOR = 0.47  # (b) also needs the Cowboy mirror's point estimate (mean) at least this
# looks (0-based: looks 2 and 3) at which a gate stops and rejects when the conditional power of a final mirror
# mean >= mirror_floor is below cp_min (floor_cp)
FLOOR_FUTILITY = {"cp_min": 0.10, "looks": [1, 2]}


def is_cowboy(st: Mapping[str, Any]) -> bool:
    return st.get("plan") == "cowboy"


def parse_cowboy_vs(spec: str) -> dict[str, int]:
    """'Ninja=16,Wizard=16' -> {"Ninja": 16, "Wizard": 16} (a CLI flag's value; checked by cowboy())."""
    out: dict[str, int] = {}
    for part in filter(None, (x.strip() for x in spec.split(","))):
        x, eq, count = part.partition("=")
        if not eq or not x or x in out:
            raise ValueError(f"want X=PAIRS,... with each character once, got {spec!r}")
        try:
            out[x] = int(count)
        except ValueError:
            raise ValueError(f"{x}: {count!r} is not a number of pairs") from None
    return out


def cowboy(seed: int, mirror_pairs: int, vs: Mapping[str, int], guard_z: float = GUARD_Z,
           min_pairs: int = MIN_PAIRS_PER_PAIRING, strict: bool = True) -> list[dict]:
    """The Cowboy gate: one full-search stage (see the module docstring) of `mirror_pairs`
    seat-swapped Cowboy:Cowboy pairs and, per opponent character X in `vs` (X -> pairs, in the order
    given, which fixes their game numbers), that many candidate-character pairs in which the
    candidate plays Cowboy in both games. Every bound is preregistered in the stage. `strict`
    (default) also refuses sizes that can't reach `min_pairs` complete Cowboy mirror pairs by the
    first look, or Cowboy-vs-X cells under `min_pairs` pairs (which could never be guarded);
    pool_eval's unstaged decision on a small schedule skips that."""
    def count(n: Any) -> bool:
        return isinstance(n, int) and not isinstance(n, bool) and 0 < n < 100_000
    if not count(mirror_pairs):
        raise ValueError(f"the Cowboy mirror needs a positive number of pairs, got {mirror_pairs!r}")
    if not vs:
        raise ValueError("the Cowboy gate needs at least one Cowboy-vs-X cell")
    bad = {x: n for x, n in vs.items() if x not in COWBOY_VS or not count(n)}
    if bad:
        raise ValueError(f"Cowboy-vs-X cells are X=PAIRS with X one of {', '.join(COWBOY_VS)} and a positive count, "
                         f"got {bad}")
    if not (isinstance(guard_z, (int, float)) and 0 < guard_z < 10):
        raise ValueError(f"guard_z must be a positive z, got {guard_z!r}")
    if strict and mirror_pairs * LOOKS[0] < min_pairs - 1e-9:
        raise ValueError(f"the Cowboy mirror's {mirror_pairs} pairs can't reach {min_pairs} complete pairs by the "
                         f"first look ({LOOKS[0]:g} of them)")
    short = {x: n for x, n in vs.items() if n < min_pairs}
    if strict and short:
        raise ValueError(f"Cowboy-vs-X cells need >= {min_pairs} pairs each to be guarded: {short}")
    st = stage("cowboy", "full", seed, mirror_pairs + sum(vs.values()), LOOKS, reject_z=REJECT_Z,
               min_pairs_per_pairing=min_pairs, final=True, accept_z=FULL_CORE_ACCEPT_Z)
    del st["pass_min"], st["guard_upper"]  # the Cowboy rule below replaces them
    vs = {x: int(n) for x, n in vs.items()}
    cells = {MIRROR: {"pairs": mirror_pairs, "games": ["Cowboy:Cowboy, candidate P1", "Cowboy:Cowboy, candidate P2"]}}
    cells.update({f"Cowboy v {x}": {"pairs": n, "games": [f"Cowboy:{x}, candidate P1 as Cowboy",
                                                          f"{x}:Cowboy, candidate P2 as Cowboy"]}
                  for x, n in vs.items()})
    st.update(plan="cowboy", mirror_pairs=mirror_pairs, cowboy_vs=vs, cells=cells, final_z=FINAL_Z,
              guard_z=float(guard_z), cell_block_upper=CELL_BLOCK_UPPER, not_worse=NOT_WORSE,
              mirror_floor=MIRROR_FLOOR, floor_futility=dict(FLOOR_FUTILITY, looks=list(FLOOR_FUTILITY["looks"])),
              pooled="every gate pair weighted equally (the Cowboy mirror and every Cowboy-vs-X cell)",
              rule=("pass if (a) mirror mean - a x SE > 0.5, or (b) pooled mean - a x SE > 0.5 and mirror mean + "
                    "guard_z x SE >= not_worse and mirror mean >= mirror_floor; a = accept_z at looks 2 and 3, final_z at the last look; any "
                    "Cowboy-vs-X cell with >= min_pairs_per_pairing complete pairs and mean + guard_z x SE < "
                    "cell_block_upper blocks a pass (an early one plays on, the final one rejects); no early accept "
                    "before every Cowboy-vs-X cell has min_pairs_per_pairing complete pairs; early reject only when "
                    "mirror mean + reject_z x SE < 0.5, or at floor_futility's looks when the conditional power of a "
                    "final mirror mean >= mirror_floor (the remaining mirror pairs at the current mirror mean, the "
                    "pairs' sample SD) is < floor_futility cp_min; the last look rejects unless it passes"))
    return [st]


def floor_cp(st: Mapping[str, Any], mirror: Mapping[str, Any]) -> float | None:
    """The conditional power of finishing a Cowboy gate with the mirror's mean >= mirror_floor: its
    remaining scheduled pairs (mirror_pairs - n) play at the current mirror mean, their mean ~ Normal
    with the pairs' sample SD / sqrt(remaining). `mirror`: pair_stats of the complete mirror pairs.
    None without a floor, below 2 pairs, or with no pair left to play."""
    floor = st.get("mirror_floor")
    n, mean, se = mirror["n"], mirror["mean"], mirror["se"]
    rest = int(st["mirror_pairs"]) - n
    if floor is None or se is None or n < 2 or rest <= 0:
        return None
    need = (floor * int(st["mirror_pairs"]) - mean * n) / rest
    sd = se * math.sqrt(n)
    if sd == 0:
        return 1.0 if mean >= need else 0.0
    z = (need - mean) / (sd / math.sqrt(rest))
    return 0.5 * math.erfc(z / math.sqrt(2))  # P(Normal > need)


def cowboy_cells(st: Mapping[str, Any]) -> list[tuple[str, list[tuple[str, str]], int]]:
    """(cell, its matchups as (P1, P2), scheduled pairs) for a Cowboy gate stage (or {mirror_pairs,
    cowboy_vs}): the Cowboy mirror first, then each Cowboy-vs-X cell in its preregistered order."""
    out = [(MIRROR, [(COWBOY, COWBOY)], int(st["mirror_pairs"]))]
    return out + [(f"Cowboy v {x}", [(COWBOY, x), (x, COWBOY)], int(n)) for x, n in st["cowboy_vs"].items()]


def cowboy_cell(characters: Sequence[str]) -> str:
    """The Cowboy gate cell a game's matchup belongs to ("Cowboy mirror", "Cowboy v Ninja")."""
    a, b = characters
    return MIRROR if a == b == COWBOY else f"Cowboy v {b if a == COWBOY else a}"


def cowboy_characters(st: Mapping[str, Any]) -> list[str]:
    """Every character a Cowboy gate plays: Cowboy, then each X."""
    return [COWBOY, *st["cowboy_vs"]]


def cowboy_verdict(st: Mapping[str, Any], look: int, cells: Sequence[Mapping[str, Any]],
                   pooled: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Look `look` of a Cowboy gate stage: continue, reject, pass or inconclusive (no complete
    Cowboy mirror pair at the last look). `cells`: per cell (cowboy_cells' names) {cell, scores: the
    complete pairs' scores, and for reporting score, ci95, complete_pairs, halted_pairs};
    `pooled`: {score, ci95} of every gate pair (reporting only). Statistics are mean +- z x SE over
    whole pairs (pair_stats); the reason reports win percentages (won())."""
    by = {c["cell"]: c for c in cells}
    missing = [name for name, _, _ in cowboy_cells(st) if name not in by]
    if missing:
        raise ValueError(f"the verdict needs every cell's pairs: {missing} missing")
    k, last = look, look == len(st["looks"]) - 1
    need, g, block, floor = st["min_pairs_per_pairing"], st["guard_z"], st["cell_block_upper"], st["not_worse"]
    mfloor = st.get("mirror_floor")  # None: a job preregistered before the floor (the old rule (b))
    a = st["final_z"] if last else st["accept_z"][k]
    zr = st["reject_z"][k]
    rows = []
    for name, _, sched in cowboy_cells(st):
        c = by[name]
        s = pair_stats(list(c["scores"]))
        x = name != MIRROR
        guarded = x and s["n"] >= need and s["se"] is not None
        hi = s["mean"] + g * s["se"] if s["se"] is not None else None
        rows.append({"cell": name, "scheduled_pairs": sched, "score": c.get("score"), "ci95": c.get("ci95"),
                     "complete_pairs": c.get("complete_pairs", s["n"]), "halted_pairs": c.get("halted_pairs"),
                     "mean": s["mean"], "se": s["se"], "n": s["n"], "upper_guard": hi,
                     "guarded": guarded, "broken": guarded and hi < block})
    cc = rows[0]
    pool = pair_stats([v for c in cells for v in c["scores"]])
    pooled = dict(pooled or {})

    def label(r):
        return f"{r['cell']}: {won(r['score'] if r['score'] is not None else r['mean'], r['ci95'])}"
    all_label = f"pooled (every gate pair): {won(pool['mean'] if pooled.get('score') is None else pooled['score'], pooled.get('ci95'))}"
    ok = cc["se"] is not None and pool["se"] is not None
    worse = ok and zr is not None and cc["mean"] + zr * cc["se"] < 0.5
    via_a = ok and a is not None and cc["mean"] - a * cc["se"] > 0.5
    not_worse = ok and cc["mean"] + g * cc["se"] >= floor
    above_floor = mfloor is None or (cc["n"] > 0 and cc["mean"] >= mfloor)
    via_b = ok and a is not None and pool["mean"] - a * pool["se"] > 0.5 and not_worse and above_floor
    path = "a" if via_a else "b" if via_b else None
    fut = st.get("floor_futility")
    cp = floor_cp(st, cc) if fut and not last and k in fut.get("looks", []) else None
    futile = cp is not None and cp < fut["cp_min"] and path is None
    broken = [r for r in rows if r["broken"]]
    unready = [r for r in rows[1:] if not r["guarded"]]
    blocked = "; ".join(f"blocked by {label(r)}, {_bound(r['mean'], '+', g, r['se'])} < {block:.0%}" for r in broken)
    if path == "a":
        rule = f"(a) {label(cc)}; {_bound(cc['mean'], '-', a, cc['se'])} > 50%"
    elif path == "b":
        rule = (f"(b) {all_label}; {_bound(pool['mean'], '-', a, pool['se'])} > 50%, and the Cowboy mirror is not "
                f"worse: {label(cc)}; {_bound(cc['mean'], '+', g, cc['se'])} >= {floor:.0%}"
                + ("" if mfloor is None else f", mean {cc['mean']:.1%} >= {mfloor:.0%}"))
    if cc["n"] == 0 and last:
        v, why = "inconclusive", "no complete Cowboy mirror pairs"
    elif worse:
        v, why = "reject", f"the Cowboy mirror is clearly worse: {label(cc)}; {_bound(cc['mean'], '+', zr, cc['se'])} < 50%"
    elif futile:
        rest = int(st["mirror_pairs"]) - cc["n"]
        need = (mfloor * int(st["mirror_pairs"]) - cc["mean"] * cc["n"]) / rest
        v, why = "reject", (f"floor futility: {label(cc)}, mean {cc['mean']:.1%} over {cc['n']} pairs; the remaining "
                            f"{rest} mirror pairs need a mean of {need:.1%} for a final mean >= {mfloor:.0%}; "
                            f"conditional power {cp:.1%} < {fut['cp_min']:.0%}")
    elif path and not last and unready:
        v, why = "continue", (f"would accept via {rule}, but no early accept before every Cowboy-vs-X cell has {need} "
                              "complete pairs (" + ", ".join(f"{r['cell']} {r['n']}" for r in unready) + "); play on")
    elif path and broken:
        v, why = ("continue" if not last else "reject"), f"would pass via {rule}, but {blocked}" + (
            "; play on" if not last else "")
    elif path:
        v, why = "pass", ("early accept" if not last else "pass") + f" via {rule}; no Cowboy-vs-X cell broken"
        if unready:
            why += "; unguarded (< {} complete pairs): {}".format(need, ", ".join(f"{r['cell']} {r['n']}"
                                                                                for r in unready))
    elif not last:
        v, why = "continue", f"{label(cc)}, {all_label}; no early stop"
    else:
        why_not = (f"(a) {label(cc)}; " + (f"{_bound(cc['mean'], '-', a, cc['se'])} <= 50%" if ok else "too few pairs")
                   + f"; (b) {all_label}; " + (f"{_bound(pool['mean'], '-', a, pool['se'])}" if ok else "too few pairs")
                   + ("" if not ok else " <= 50%" if pool["mean"] - a * pool["se"] <= 0.5 else
                      f" > 50% but the Cowboy mirror is worse: {_bound(cc['mean'], '+', g, cc['se'])} < {floor:.0%}"
                      if not not_worse else
                      f" > 50% but the Cowboy mirror is under its floor: mean {cc['mean']:.1%} < {mfloor:.0%}"))
        v, why = "reject", f"no pass: {why_not}"
    return {"stage": st["name"], "look": look, "fraction": st["looks"][look], "verdict": v, "reason": why,
            "core": {**pool, "score": pooled.get("score"), "ci95": pooled.get("ci95")},
            "cowboy": {"cells": rows, "pooled": {**pool, "score": pooled.get("score"), "ci95": pooled.get("ci95")},
                       "path": path if v == "pass" else None, "would_pass": path,
                       "blocked_by": [r["cell"] for r in broken] if path else [],
                       "unguarded": [r["cell"] for r in unready], "efficacy_z": a, "reject_z": zr, "guard_z": g,
                       "floor_cp": cp, "floor_futile": futile}}


# --- the per-character floor's extension (plan "cowboy_extension") ------------------------------

EXTENSION = "cowboy_extension"
CELL_FLOOR = 0.47  # the per-character floor (user decision 2026-11-19)
EXTENSION_PAIRS = 32  # pairs a low cell plays in its extension


def is_extension(st: Mapping[str, Any]) -> bool:
    return st.get("plan") == EXTENSION


def _floor_value(f: Any) -> bool:
    return isinstance(f, (int, float)) and not isinstance(f, bool) and 0 <= f < 1


def cell_floor_rule(floor: Any = CELL_FLOOR, extension_pairs: int = EXTENSION_PAIRS,
                    basis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The per-character floor a Cowboy gate job preregisters (its `cell_floor`): a passing gate's
    Cowboy-vs-X cell with a point score below its floor plays `extension_pairs` more pairs and is judged
    over all of them (0: it is held instead, no extension). `floor` is one score for every cell, or a
    map {"Cowboy v X": score} (user decision 11/19 9:28 pm ET: the control-relative floor, each cell's
    equal-net control score minus 0.10; a cell missing from the map has no floor). `basis` records where
    the floors came from (e.g. the control's per-cell numbers), kept with the rule."""
    if isinstance(floor, Mapping):
        bad = {k: v for k, v in floor.items() if not str(k).startswith("Cowboy v ") or not _floor_value(v)}
        if not floor or bad:
            raise ValueError(f"per-cell floors are {{'Cowboy v X': score in [0, 1)}}, got {dict(floor)!r}")
        floor = {str(k): float(v) for k, v in floor.items()}
    elif not (_floor_value(floor) and floor > 0):
        raise ValueError(f"the cell floor is a score in (0, 1), got {floor!r}")
    else:
        floor = float(floor)
    if not (isinstance(extension_pairs, int) and not isinstance(extension_pairs, bool) and 0 <= extension_pairs < 10_000):
        raise ValueError(f"extension pairs must be a count >= 0, got {extension_pairs!r}")
    return {"floor": floor, "extension_pairs": int(extension_pairs), **({"basis": dict(basis)} if basis else {})}


def floor_for(floor: Any, cell: str) -> float | None:
    """A cell's floor under a rule's `floor` (one score, or a per-cell map; None: no floor for it)."""
    if isinstance(floor, Mapping):
        f = floor.get(cell)
        return None if f is None else float(f)
    return float(floor)


def check_extension(cowboy_vs: Mapping[str, int], extension_vs: Mapping[str, int]) -> dict[str, int]:
    """An extension's cells ({X: pairs}) checked against its gate's cowboy_vs: at least one, each a
    gate cell, one positive count for all. Returns them in the gate's cell order."""
    if not extension_vs:
        raise ValueError("an extension plays at least one Cowboy-vs-X cell")
    bad = sorted(x for x in extension_vs if x not in cowboy_vs)
    if bad:
        raise ValueError(f"extension cells {bad} are not cells of the gate ({', '.join(cowboy_vs)})")
    counts = set(extension_vs.values())
    n = next(iter(counts))
    if len(counts) != 1 or not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"extension cells play one positive number of pairs each, got {dict(extension_vs)}")
    return {x: n for x in cowboy_vs if x in extension_vs}


def extension_offsets(cowboy_vs: Mapping[str, int], extension_vs: Mapping[str, int]) -> dict[str, int]:
    """Each extended X's first game number, counted from the gate's first candidate-character game
    (arena.HELD_GAMES): past every gate cell's pairs, then a block of the extension's pairs per X in
    the gate's cell order, extended or not, so a cell's extension seeds don't depend on which others
    are extended, and none is a gate game's."""
    ext = check_extension(cowboy_vs, extension_vs)
    n, base = next(iter(ext.values())), sum(int(v) for v in cowboy_vs.values())
    return {x: base + j * n for j, x in enumerate(cowboy_vs) if x in ext}


def cowboy_extension(main: Mapping[str, Any], low: Sequence[Mapping[str, Any]],
                     rule: Mapping[str, Any]) -> dict[str, Any]:
    """The extension stage of passing Cowboy gate stage `main` for its `low` cells ([{cell, score,
    complete_pairs}], the cells below the floor) under its preregistered cell_floor `rule`: one
    full-search stage on the gate's seed, `extension_pairs` candidate-character pairs per low cell,
    one final look. The gate's numbers of each low cell are kept in the stage (main_cells) to judge
    the cell over all its pairs (extension_verdict)."""
    if not is_cowboy(main):
        raise ValueError("an extension extends a Cowboy gate stage")
    rule = cell_floor_rule(rule["floor"], rule["extension_pairs"], rule.get("basis"))
    n = rule["extension_pairs"]
    if n < 1:
        raise ValueError("this gate's cell floor has no extension (extension_pairs 0)")
    xs = {}
    for c in low:
        if not str(c["cell"]).startswith("Cowboy v ") or c["cell"][len("Cowboy v "):] not in main["cowboy_vs"]:
            raise ValueError(f"{c['cell']!r} is not a Cowboy-vs-X cell of the gate")
        xs[c["cell"][len("Cowboy v "):]] = c
    ext = check_extension(main["cowboy_vs"], {x: n for x in xs})
    main_cells = {}
    for x in ext:
        score, pairs = float(xs[x]["score"]), int(xs[x]["complete_pairs"] or 0)
        ci = xs[x].get("ci95")
        # pair scores are quarters: the sum is exact although the reported score is rounded
        main_cells[f"Cowboy v {x}"] = {"score": score, "complete_pairs": pairs, "pair_points": round(score * pairs * 4) / 4,
                                       **({"se": (float(ci[1]) - float(ci[0])) / 3.92} if ci else {})}
    st = stage("floorext", "full", int(main["seed"]), n * len(ext), (1.0,), min_pairs_per_pairing=0, final=True)
    del st["pass_min"], st["guard_upper"]
    st.update(plan=EXTENSION, extends=main["name"], mirror_pairs=int(main["mirror_pairs"]),
              cowboy_vs=dict(main["cowboy_vs"]), extension_vs=ext, extension_pairs=n, cell_floor=rule["floor"],
              **({"cell_floor_basis": rule["basis"]} if rule.get("basis") else {}),
              main_cells=main_cells,
              cells={f"Cowboy v {x}": {"pairs": n, "games": [f"Cowboy:{x}, candidate P1 as Cowboy",
                                                              f"{x}:Cowboy, candidate P2 as Cowboy"]} for x in ext},
              rule=("each extended Cowboy-vs-X cell plays extension_pairs candidate-character pairs on the gate's seed "
                    "(game numbers after the gate's: extension_offsets), all of them (one final look); its point score "
                    "over every complete pair, the gate's (main_cells) and the extension's, below cell_floor blocks the "
                    "promotion (any one cell); otherwise it promotes"))
    return st


def control_test(basis: Mapping[str, Any] | None, cell: str, main: Mapping[str, Any], ext_scores: Sequence[float],
                 total: float | None, n: int) -> dict[str, Any] | None:
    """The control-relative floor's noise check (coordination 11/19 9:50 pm ET): with a basis
    {"control": {cell: {score, se, pairs}}, "z": z}, the shortfall (candidate - control) is real only if its
    upper z bound is below 0. The candidate's SE pools the gate cell's (from its 95% interval) and the
    extension pairs' variance. None without a control basis for the cell (then the point floor alone decides)."""
    ctl = ((basis or {}).get("control") or {}).get(cell)
    if not ctl or total is None or n < 2:
        return None
    z = float((basis or {}).get("z", 1.96))
    nm, ne = int(main.get("complete_pairs") or 0), len(ext_scores)
    var_m = (float(main["se"]) ** 2) * nm if main.get("se") is not None and nm else None
    var_e = None
    if ne >= 2:
        mu = sum(ext_scores) / ne
        var_e = sum((x - mu) ** 2 for x in ext_scores) / (ne - 1)
    parts = [(v, k) for v, k in ((var_m, nm), (var_e, ne)) if v is not None and k > 1]
    if not parts:
        return None
    var = sum(v * (k - 1) for v, k in parts) / max(sum(k - 1 for _, k in parts), 1)
    se_c = math.sqrt(var / n)
    diff = total - float(ctl["score"])
    se_d = math.sqrt(se_c ** 2 + float(ctl["se"]) ** 2)
    return {"control": float(ctl["score"]), "control_se": float(ctl["se"]), "control_pairs": int(ctl.get("pairs") or 0),
            "diff": diff, "se_diff": se_d, "upper": diff + z * se_d, "z": z, "real": diff + z * se_d < 0}


def extension_cells(st: Mapping[str, Any]) -> list[tuple[str, list[tuple[str, str]], int]]:
    """(cell, its matchups as (P1, P2), scheduled pairs) for an extension stage's cells, in the gate's order."""
    return [(f"Cowboy v {x}", [(COWBOY, x), (x, COWBOY)], int(n)) for x, n in st["extension_vs"].items()]


def extension_verdict(st: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """An extension stage's decision: pass (promote) unless an extended cell's point score over all
    its complete pairs (the gate's, from main_cells, and the extension's) is below cell_floor, which
    blocks. `cells`: per extended cell {cell, scores: the extension's complete pairs' scores, and for
    reporting score, ci95, complete_pairs, halted_pairs}."""
    by = {c["cell"]: c for c in cells}
    missing = [name for name, _, _ in extension_cells(st) if name not in by]
    if missing:
        raise ValueError(f"the verdict needs every extended cell's pairs: {missing} missing")
    floor, rows = st["cell_floor"], []
    for name, _, sched in extension_cells(st):
        c, m = by[name], st["main_cells"][name]
        f = floor_for(floor, name)
        scores = list(c["scores"])
        n = m["complete_pairs"] + len(scores)
        total = (m["pair_points"] + sum(scores)) / n if n else None
        test = control_test(st.get("cell_floor_basis"), name, m, scores, total, n)
        rows.append({"cell": name, "main": {"score": m["score"], "complete_pairs": m["complete_pairs"]},
                     "extension": {"score": c.get("score"), "ci95": c.get("ci95"), "scheduled_pairs": sched,
                                   "complete_pairs": len(scores), "halted_pairs": c.get("halted_pairs")},
                     "score": total, "complete_pairs": n, "floor": f, **({"control_test": test} if test else {}),
                     "below_floor": f is not None and (total is None or total < f) and (test is None or test["real"])})
    blocked = [r for r in rows if r["below_floor"]]

    def label(r):
        ext = r["extension"]
        return (f"{r['cell']}: gate {won(r['main']['score'])} over {r['main']['complete_pairs']} pairs, extension "
                f"{won(ext['score'])} over {ext['complete_pairs']} of {ext['scheduled_pairs']}; all "
                + (f"{r['complete_pairs']} pairs {r['score']:.1%}" if r["score"] is not None else "no pairs"))
    if blocked:
        v, why = "blocked", "blocked by the per-character floor: " + "; ".join(
            f"{label(r)} < {r['floor']:.1%}" for r in blocked)
        ok = [r for r in rows if not r["below_floor"]]
        if ok:
            why += " (clears it: " + "; ".join(f"{label(r)} >= {r['floor']:.1%}" for r in ok if r["floor"] is not None) + ")"
    else:
        v, why = "pass", "every extended cell clears the per-character floor: " + "; ".join(
            f"{label(r)} >= {r['floor']:.1%}" for r in rows if r["floor"] is not None)
    return {"stage": st["name"], "look": 0, "fraction": 1.0, "verdict": v, "reason": why, "cells": rows,
            "blocked_by": [r["cell"] for r in blocked], "cell_floor": floor}


def look_due(st: dict[str, Any], complete_pairs: int, scheduled_pairs: int, done: Sequence[int],
             per_pairing: Mapping[Any, int] | None = None) -> int | None:
    """The index of the latest look reached by `complete_pairs` that hasn't been evaluated (`done`),
    or None. Looks skipped over aren't evaluated. No look before every pairing has the stage's
    minimum of complete pairs (`per_pairing`: complete pairs by pairing)."""
    if per_pairing is not None and min(per_pairing.values(), default=0) < st.get("min_pairs_per_pairing", 0):
        return None
    reached = [k for k, t in enumerate(st["looks"]) if complete_pairs >= math.ceil(t * scheduled_pairs)]
    if not reached or reached[-1] in done:
        return None
    return reached[-1]


def pair_stats(scores: Sequence[float]) -> dict[str, Any]:
    """{mean, se, n} of pair scores: se = sample sd / sqrt(n) (None below 2 pairs)."""
    n = len(scores)
    if n == 0:
        return {"mean": None, "se": None, "n": 0}
    mean = sum(scores) / n
    se = math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1) / n) if n > 1 else None
    return {"mean": mean, "se": se, "n": n}


def won(score: float | None, ci95: Sequence[float | None] | None = None) -> str:
    """A score as people read it: "won 64% of games (58–71%)", the candidate's points in the games
    counted (a draw is half a win) as a whole percentage, with its 95% interval when there is one."""
    if score is None:
        return "no complete pairs"
    lo, hi = (list(ci95) if ci95 else [None, None])[:2]
    return f"won {score * 100:.0f}% of games" + (f" ({lo * 100:.0f}–{hi * 100:.0f}%)" if None not in (lo, hi) else "")


def _bound(mean: float, sign: str, z: float, se: float) -> str:
    """'mean - 2.862 x SE = 52.3%' (one decimal: a bound is compared with 50% or 40%)."""
    return f"mean {sign} {z} x SE = {mean + (z if sign == '+' else -z) * se:.1%}"


def verdict(st: dict[str, Any], look: int, core: dict[str, Any], matchups: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Look `look` of stage `st`: continue, reject, pass (a final stage's pass promotes) or
    inconclusive (no data at the last look). `core`: {mean, se, n} of the pair scores and the gate's
    {score, ci95}; `matchups`: per pairing {characters, score, ci95}. The reason reports scores as
    win percentages (won())."""
    last = look == len(st["looks"]) - 1
    mean, se = core.get("mean"), core.get("se")
    score, ci = core.get("score"), (core.get("ci95") or [None, None])
    z = (st.get("reject_z") or [None] * len(st["looks"]))[look]
    za = (st.get("accept_z") or [None] * len(st["looks"]))[look]  # jobs before the field: reject-only
    bad = [m for m in matchups if st["guard_upper"] is not None and (m.get("ci95") or [None, None])[1] is not None
           and m["ci95"][1] < st["guard_upper"]]
    so_far = won(score if score is not None else mean, ci)
    guard = ", ".join(f"{':'.join(m['characters'])} {won(m['score'], m['ci95'])}, upper < {st['guard_upper']:.0%}"
                      for m in bad)
    if z is not None and mean is not None and se is not None and mean + z * se < 0.5:
        v, why = "reject", f"look {look + 1}: {so_far}; {_bound(mean, '+', z, se)} < 50%"
    elif not last and za is not None and mean is not None and se is not None and mean - za * se > 0.5:
        if bad:
            v, why = "continue", f"look {look + 1}: {so_far}; {_bound(mean, '-', za, se)} > 50% but the guard fails ({guard})"
        else:
            v, why = "pass", f"look {look + 1}: early accept, {so_far}; {_bound(mean, '-', za, se)} > 50%, guard holds"
    elif not last:
        v, why = "continue", (f"look {look + 1}: {so_far}, no early stop" if mean is not None else "no pairs")
    elif score is None:
        v, why = "inconclusive", "no complete pairs"
    else:
        if bad:
            v, why = "reject", f"guard: {guard}"
        elif score >= st["pass_min"]:
            v, why = "pass", f"core {won(score, ci)}, >= {st['pass_min']:.0%}"
        else:
            v, why = "reject", f"core {won(score, ci)}, < {st['pass_min']:.0%}"
    return {"stage": st["name"], "look": look, "fraction": st["looks"][look], "verdict": v, "reason": why,
            "core": {"mean": mean, "se": se, "n": core.get("n"), "score": score, "ci95": ci}}


def anchor_verdict(look: int, scores: Sequence[float], pairs_per_pairing: Mapping[Any, int],
                   min_pairs: int = MIN_PAIRS_PER_PAIRING, z: Sequence[float] = REJECT_Z) -> dict[str, Any]:
    """One anchor's regression check at look `look` on the new best's pair scores against it: pass,
    fail or open (never before every pairing has `min_pairs` complete pairs)."""
    st = pair_stats(scores)
    ready = bool(pairs_per_pairing) and min(pairs_per_pairing.values()) >= min_pairs
    m, se, zk = st["mean"], st["se"], z[look]
    if not ready or m is None or se is None:
        v = "open"
    else:
        v = "pass" if m - zk * se > 0.5 else "fail" if m + zk * se < 0.5 else "open"
    return {"look": look, "z": zk, "verdict": v, "ready": ready, **st}
