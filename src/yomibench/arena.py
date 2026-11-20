"""Seat-swapped matches, a JSONL ledger and Bradley-Terry ratings."""

from __future__ import annotations

import itertools
import json
import math
import platform
import random
import re
import threading
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping

from yomibench.bots import Bot
from yomibench.engine import Engine, EngineCrashed, EngineError


UNRATED = ("halt", "truncated")  # outcomes left out of scores, ratings and the win matrix


@dataclass
class GameResult:
    bots: tuple[str, str]  # (seat 0, seat 1)
    characters: tuple[str, str]
    seed: int
    outcome: str  # "p1", "p2", "draw", "truncated" (hit the turn limit unfinished), "halt" (engine failed)
    turns: int
    hp: tuple[int, int] | None
    seconds: float
    detail: str = ""

    def score(self, name: str) -> float | None:
        """1 win, 0.5 draw, 0 loss for bot `name`; None if not rated (engine halt, or a game the
        turn limit cut off: nobody won it, and it is not a draw either)."""
        if self.outcome in UNRATED:
            return None
        if self.outcome == "draw":
            return 0.5
        winner = self.bots[0] if self.outcome == "p1" else self.bots[1]
        return 1.0 if winner == name else 0.0


def play_game(
    engine: Engine, bots: tuple[Bot, Bot], characters: tuple[str, str], seed: int, max_turns: int = 1000,
    record_dir: Path | None = None,
) -> GameResult:
    """One game between two bots. With `record_dir`, also writes the game as a format-3 record
    (`_ArenaRecord`); the bots' decisions are the same either way."""
    rec = _ArenaRecord(engine, bots, characters, seed, max_turns, record_dir) if record_dir else None
    try:
        result = _play(engine, bots, characters, seed, max_turns, rec)
    except BaseException:
        if rec:
            rec.abort()
        raise
    if rec:
        rec.finish()
    return result


def _play(engine: Engine, bots: tuple[Bot, Bot], characters: tuple[str, str], seed: int, max_turns: int,
          rec: "_ArenaRecord | None") -> GameResult:
    start = time.perf_counter()
    rngs = (random.Random(f"{seed}:0"), random.Random(f"{seed}:1"))
    sets = tuple(getattr(b, "candidate_set", None) for b in bots)
    if any(sets):  # an A/B arena between candidate sets: each seat gets its bot's set
        obs = engine.new_match(characters[0], characters[1], seed, candidate_sets=sets)
    else:
        obs = engine.new_match(characters[0], characters[1], seed)
    if rec:
        rec.start(obs)
    while not obs.done and obs.turn < max_turns:
        t = time.perf_counter()
        a = bots[0].choose(obs, 0, engine, rngs[0])
        b = bots[1].choose(obs, 1, engine, rngs[1])
        if rec:
            rec.decision(obs, (a, b), t)
        obs = engine.step(a, b)
    if not obs.done:  # cut off by max_turns: no result
        outcome = "truncated"
    else:
        outcome = "draw" if obs.winner is None else ("p1" if obs.winner == 0 else "p2")
    hp = tuple(p["hp"] for p in obs.state["players"])
    seconds = round(time.perf_counter() - start, 3)
    if rec:
        rec.result(obs, outcome, seconds)
    return GameResult(
        (bots[0].name, bots[1].name), characters, seed, outcome, obs.turn, hp,  # type: ignore[arg-type]
        seconds,
    )


def seat_rules(engine: Any, bots: tuple[Any, Any], hello: dict[str, Any]) -> list[dict[str, Any]]:
    """The candidate rules each seat played this match: [{candidate_set, candidate_rules}] for P1 and
    P2. From the engine when it says (`BatchEngine.seat_rules`: the slot's per-seat sets and hashes),
    else each seat's bot's set (or the engine's) with hello's rules hash when that is hello's set
    (None when the hash for a seat's set isn't known)."""
    got = getattr(engine, "seat_rules", None)
    if callable(got):
        rules = got()
        if rules:
            return rules
    default = hello.get("candidate_set")
    out = []
    for b in bots:
        cset = getattr(b, "candidate_set", None)
        cset = default if cset is None else str(cset)
        out.append({"candidate_set": None if cset is None else str(cset),
                    "candidate_rules": hello.get("candidate_rules") if str(cset) == str(default) else None})
    return out


class _ArenaRecord:
    """An arena game as a format-3 record (selfplay's format and writer; slim previews): the header
    names both bots (each seat's config, `bot` being its arena name) and each seat's candidate rules
    (`seat_rules`: [{candidate_set, candidate_rules}] for P1 and P2; see seat_rules), every decision has its state,
    each seat's candidate list and both picks (indices into those lists), and the last line the
    result.

    The bots search separately, so each seat's search fields come from its own bot (`MatrixBot.decision`,
    read after both picks, so recording changes no decision), over the record's lists: `strategies[s]`
    is the mix seat s played, `opponent_strategies[s]` the mix seat s's bot solved for the other seat,
    `candidate_values[s]` its value of each of its candidates against that mix (its own side, rounded
    to 4 places, null where its search never evaluated one or the candidate isn't in its view, such as
    the other bot's expansions) and `seat_values[s]` its guaranteed value (its own side). A seat whose
    bot doesn't search (random, greedy) has nulls. `value` stays null: the two bots' values differ, and
    the trainer reads `value` as a target when it evaluates on gate records (`train.select_rows`), so
    a search value there would move those targets toward one net's opinion.

    A seat's list is the engine's candidates followed by the candidates `expand` added to that seat
    this decision (set 4 factor refinement), seen by wrapping `engine.expand` for the game, so the
    record doesn't depend on how bots keep their candidate views. Engine ids are positions in those
    lists, so a pick is the position of the chosen id, as in self-play records (tools/render_record.py
    replays them). The file is `g<seed>-<seat 0>-vs-<seat 1>-<sha1>.jsonl.gz`."""

    def __init__(self, engine: Engine, bots: tuple[Bot, Bot], characters: tuple[str, str], seed: int,
                 max_turns: int, out_dir: Path):
        from yomibench.search import SOLVER
        from yomibench.selfplay import FORMAT, RecordWriter, git_revision

        self.engine, self.bots = engine, bots
        safe = lambda name: re.sub(r"[^A-Za-z0-9._-]+", "_", name)  # noqa: E731
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.writer = RecordWriter(Path(out_dir), seed, f"g{seed:012d}-{safe(bots[0].name)}-vs-{safe(bots[1].name)}")
        configs = [{**(b.config() if hasattr(b, "config") else {}), "bot": b.name} for b in bots]  # type: ignore[attr-defined]
        self.header = {"format": FORMAT, "characters": list(characters), "seed": seed, "bots": configs,
                       "engine": engine.hello(), "max_candidates": None, "max_turns": max_turns, "opening": None,
                       "solver": SOLVER, "previews": "slim", "full_previews_fraction": None, "explore": None,
                       "playout_cap": None, "arena": True, "yomibench": git_revision(),
                       "platform": platform.system(), "started": time.time()}
        self.expanded: list[list[Any]] = [[], []]
        self._expand, self._own = engine.expand, "expand" in vars(engine)

        def expand(specs: list[dict[str, Any]]) -> list[Any]:
            got = self._expand(specs)
            for spec, c in zip(specs, got):
                if not isinstance(c, str):
                    self.expanded[spec["seat"]].append(c)
            return got
        engine.expand = expand  # type: ignore[method-assign]  (restored by finish / abort)

    def start(self, obs: Any) -> None:
        self.header["seat_rules"] = seat_rules(self.engine, self.bots, self.header["engine"])
        self.writer.write(self.header)
        self.expanded = [[], []]

    def decision(self, obs: Any, ids: tuple[int, int], t: float) -> None:
        seats = [list(obs.candidates[s]) + sorted(self.expanded[s], key=lambda c: c.id) for s in (0, 1)]
        picks = [next(i for i, c in enumerate(seats[s]) if c.id == ids[s]) for s in (0, 1)]
        self.writer.write({"turn": obs.turn, "state": obs.state, "actionable": list(obs.actionable),
                           **self._searches(obs, seats), "value": None,
                           "previews": len(self.engine.turn_cache.get("previewed", {})),
                           "candidates": [[c.record() for c in seat] for seat in seats], "picks": picks,
                           "ms": round((time.perf_counter() - t) * 1000)})
        self.expanded = [[], []]

    def _searches(self, obs: Any, seats: list[list[Any]]) -> dict[str, Any]:
        """Each seat's search fields from its own bot (see the class docstring), over `seats`."""
        out: dict[str, list[Any]] = {k: [None, None] for k in
                                     ("strategies", "opponent_strategies", "candidate_values", "seat_values")}
        for s, bot in enumerate(self.bots):
            if not hasattr(bot, "decision"):
                continue
            d = bot.decision(obs, s, self.engine)
            pos = [{c.id: k for k, c in enumerate(seat)} for seat in seats]

            def onto(side: int, xs: Any, fill: Any) -> list[Any]:
                out_ = [fill] * len(seats[side])
                for c, x in zip(d["lists"][side], xs):
                    out_[pos[side][c.id]] = x
                return out_
            out["strategies"][s] = onto(s, [round(p, 5) for p in d["strategies"][s]], 0.0)
            out["opponent_strategies"][s] = onto(1 - s, [round(p, 5) for p in d["strategies"][1 - s]], 0.0)
            if d["replies"] is not None:
                out["candidate_values"][s] = onto(
                    s, [round(x, 4) if math.isfinite(x) else None for x in d["replies"][s]], None)
            out["seat_values"][s] = round(d["value"] if s == 0 else -d["value"], 5)
        if all(x is None for x in out["strategies"]):
            return {k: None for k in out}
        return out

    def result(self, obs: Any, outcome: str, seconds: float) -> None:
        from yomibench.selfplay import _end_reason

        self.writer.write({"result": {
            "winner": obs.winner, "outcome": outcome, "end": _end_reason(obs), "turns": obs.turn, "done": obs.done,
            "hp": [p["hp"] for p in obs.state["players"]], "final_state": obs.state, "seconds": seconds}})

    def _restore(self) -> None:
        if self._own:
            self.engine.expand = self._expand  # type: ignore[method-assign]
        else:
            del self.engine.expand  # back to the class's method

    def finish(self) -> Path:
        self._restore()
        return self.writer.finish()

    def abort(self) -> None:
        self._restore()
        self.writer.abort()


def pair_schedule(
    characters: list[str], pairs: int, pairs_by_pairing: Mapping[tuple[str, str], int] | None = None
) -> list[tuple[tuple[str, str], int]]:
    """One bot pair's seat-swapped pairs in schedule order: (pairing, game number), the game number
    giving the seed. `pairs` per pairing, pairing by pairing (P1 major). With `pairs_by_pairing`
    (pairing -> pairs, every pairing): pair i < m (m the smallest count) of pairing j keeps the
    `pairs`=m schedule's game number j*m + i, so those games are the same whatever the larger
    counts are; the rest get the numbers after, pairing by pairing. The order interleaves the
    pairings (pair i of a c-pair pairing at (i + 1/2) / c), so any stretch of the schedule, and any
    run of shards splitting it k mod n, holds every pairing in proportion to its count."""
    pairings = list(itertools.product(characters, repeat=2))
    if pairs_by_pairing is None:
        return [(pr, j * pairs + i) for j, pr in enumerate(pairings) for i in range(pairs)]
    counts = [pairs_by_pairing[pr] for pr in pairings]
    m = min(counts)
    extra = list(itertools.accumulate((c - m for c in counts), initial=len(pairings) * m))
    order = sorted((Fraction(2 * i + 1, 2 * c), j, i) for j, c in enumerate(counts) for i in range(c))
    return [(pairings[j], j * m + i if i < m else extra[j] + i - m) for _, j, i in order]


HELD_GAMES = 500_000  # candidate-character pairs' game numbers start here, clear of ordinary pairs'


def cell_schedule(
    ordinary: Mapping[tuple[str, str], int], held: Mapping[tuple[str, str], int],
    held_first: Mapping[tuple[str, str], int] | None = None,
) -> list[tuple[tuple[str, str], int, bool]]:
    """One bot pair's schedule of two kinds of pairs, as (pairing, game number, held): `ordinary`
    (pairing -> pairs) are seat-swapped pairs as pair_schedule's (both games at the pairing, bot a
    P1 then bot b P1), numbered from 0 pairing by pairing in the order given; `held` (pairing (P, Q)
    -> pairs) are candidate-character pairs: bot a plays P and bot b plays Q in both games, a at P1
    in the first (P:Q) and at P2 in the second (Q:P), on the same seed, numbered from HELD_GAMES
    pairing by pairing in the order given (so adding a pairing at the end keeps the others' seeds).
    The order interleaves every pairing of both kinds (pair i of a c-pair pairing at (i + 1/2) / c,
    ties in the order given, ordinary first), so any stretch of the schedule, and any run of shards
    splitting it k mod n, holds every pairing in proportion to its count. `held_first` (held pairing
    -> its first game number, counted from HELD_GAMES) replaces that numbering for every held pairing
    (a Cowboy gate's cell extension: gate_seq.extension_offsets); the ranges must not overlap."""
    groups = [(pr, c, False) for pr, c in ordinary.items()] + [(pr, c, True) for pr, c in held.items()]
    if sum(ordinary.values()) > HELD_GAMES:
        raise ValueError(f"more than {HELD_GAMES} ordinary pairs")
    start, nxt = [], {False: 0, True: HELD_GAMES}
    for pr, c, h in groups:
        start.append(nxt[h] if held_first is None or not h else HELD_GAMES + held_first[pr])
        nxt[h] += c
    if held_first is not None:
        spans = sorted((s, s + c) for s, (_, c, h) in zip(start, groups) if h)
        if (spans and spans[0][0] < HELD_GAMES) or any(b > s2 for (_, b), (s2, _) in zip(spans, spans[1:])):
            raise ValueError(f"held pairings' game numbers overlap or start below HELD_GAMES: {spans}")
    order = sorted((Fraction(2 * i + 1, 2 * c), j, i) for j, (_, c, _) in enumerate(groups) for i in range(c))
    return [(groups[j][0], start[j] + i, groups[j][2]) for _, j, i in order]


def run_round_robin(
    connect: Callable[[], Engine] | list[Callable[[], Engine]],
    bots: list[Bot],
    characters: list[str],
    pairs_per_matchup: int,
    seed: int,
    ledger: Path | None = None,
    on_result: Callable[[GameResult], None] | None = None,
    max_turns: int = 1000,
    shard: tuple[int, int] | None = None,
    record_dir: Path | None = None,
    pairs_by_pairing: Mapping[tuple[str, str], int] | None = None,
    schedule: list[tuple[tuple[str, str], int, bool]] | None = None,
) -> list[GameResult]:
    """Every bot pair plays `pairs_per_matchup` seat-swapped game pairs per character pairing.

    With a list of engine factories, games run in parallel, one worker thread per engine.
    Each game's seed depends only on its place in the schedule, so results don't depend on
    how many engines played them (only the ledger's line order does). `shard=(k, n)` plays only
    the pairs whose place in the schedule is k mod n (both games of a pair stay together), so n
    processes can split one schedule with the same seeds. With `record_dir`, every game is also
    written there as a format-3 record (see play_game); a game the engine halted leaves none.
    `pairs_by_pairing` (pairing -> pairs) replaces `pairs_per_matchup` per pairing (pair_schedule:
    seeds and order; shards split the schedule order k mod n). `schedule` (cell_schedule) replaces
    both: its held pairs keep each bot on its character, the matchup reversed with the seats."""
    jobs: list[tuple[tuple[Bot, Bot], tuple[str, str], int]] = []
    sched = schedule if schedule is not None else [
        (chars, game_no, False) for chars, game_no in pair_schedule(characters, pairs_per_matchup, pairs_by_pairing)]
    for combo, (a, b) in enumerate(itertools.combinations(bots, 2)):
        for place, (chars, game_no, held) in enumerate(sched, combo * len(sched)):
            if shard is not None and place % shard[1] != shard[0]:
                continue
            game_seed = seed * 1_000_003 + combo * len(sched) + game_no
            jobs.append(((a, b), chars, game_seed))
            jobs.append(((b, a), (chars[1], chars[0]) if held else chars, game_seed))

    connects = connect if isinstance(connect, list) else [connect]
    results: list[GameResult | None] = [None] * len(jobs)
    out = ledger.open("a") if ledger else None
    lock = threading.Lock()
    queue = iter(range(len(jobs)))

    errors: list[BaseException] = []

    def worker(make: Callable[[], Engine]) -> None:
        engine: Engine | None = None
        try:
            engine = make()
            while True:
                with lock:
                    i = next(queue, None)
                if i is None:
                    return
                seats, chars, game_seed = jobs[i]
                crashed = False
                try:
                    r = play_game(engine, seats, chars, game_seed, max_turns, record_dir)
                except (EngineCrashed, EngineError) as e:
                    r = GameResult((seats[0].name, seats[1].name), chars, game_seed, "halt", 0, None, 0.0, str(e))
                    crashed = isinstance(e, EngineCrashed)
                with lock:
                    results[i] = r
                    if out:
                        out.write(json.dumps(asdict(r)) + "\n")
                        out.flush()
                    if on_result:
                        on_result(r)
                if crashed:
                    engine.close()
                    engine = None
                    engine = make()
        except BaseException as e:  # surfaced to the caller after all workers stop
            with lock:
                errors.append(e)
        finally:
            if engine is not None:
                engine.close()

    try:
        if len(connects) == 1:
            worker(connects[0])
        else:
            threads = [threading.Thread(target=worker, args=(c,)) for c in connects]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if errors:
            raise errors[0]
    finally:
        if out:
            out.close()
    return [r for r in results if r is not None]


def _pair_key(r: GameResult) -> tuple:
    return (tuple(sorted(r.bots)), r.characters, r.seed)


def _held_key(r: Any) -> tuple:
    return (tuple(sorted(zip(r.bots, r.characters))), r.seed)


def pair_games(results: list[Any]) -> list[list[Any]]:
    """Games (GameResults, or anything with bots, characters and seed) grouped into their
    seat-swapped pairs, in the order of each pair's first game; a game whose partner is missing is
    a group of one. A game pairs with the game on its seed between the same bots at the same
    matchup (an ordinary pair: each bot plays each side once); a game left without one pairs with
    the game on its seed in which each bot plays the same character from the other seat (a
    candidate-character pair, cell_schedule's held pairs: the matchup reversed with the seats)."""
    groups: dict[tuple, list[Any]] = {}
    for r in results:
        groups.setdefault(_pair_key(r), []).append(r)
    out: list[list[Any]] = []
    held: dict[tuple, list[Any]] = {}
    for g in groups.values():
        if len(g) != 1:
            out.append(g)
        elif _held_key(g[0]) in held:
            held[_held_key(g[0])].append(g[0])
        else:
            out.append(held.setdefault(_held_key(g[0]), g))
    return out


def rated_pairs(results: list[GameResult]) -> list[list[GameResult]]:
    """Seat-swapped pairs (pair_games) with both games finished. A halt or a truncated game in
    either seat drops the whole pair, so no bot gets credit for only the seat it happened to play."""
    return [g for g in pair_games(results) if len(g) == 2 and all(r.outcome not in UNRATED for r in g)]


def bradley_terry(results: list[GameResult], anchor: str, iterations: int = 500) -> dict[str, float]:
    """Elo-scale ratings with `anchor` at 1000, over complete seat-swapped pairs. Draws count
    half; one virtual draw per matchup keeps unbeaten or winless bots finite."""
    return _bt([r for pair in rated_pairs(results) for r in pair], anchor, iterations)


def bradley_terry_ci(
    results: list[GameResult], anchor: str, samples: int = 200, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% intervals from a paired bootstrap: resample whole seat-swapped pairs."""
    pairs = rated_pairs(results)
    names = sorted({n for r in results for n in r.bots})
    if not pairs:
        return {n: (math.nan, math.nan) for n in names}
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(samples):
        sample = [r for _ in pairs for r in rng.choice(pairs)]
        ratings = _bt(sample, anchor, 200)
        for n in names:
            draws[n].append(ratings.get(n, math.nan))
    out = {}
    for n, xs in draws.items():
        xs = sorted(x for x in xs if not math.isnan(x))
        out[n] = (xs[int(0.025 * (len(xs) - 1))], xs[int(0.975 * (len(xs) - 1))]) if xs else (math.nan, math.nan)
    return out


def win_matrix(results: list[GameResult]) -> dict[tuple[str, str], tuple[float, int]]:
    """(a, b) -> (a's score rate against b, games), over complete pairs. Shows non-transitive
    results that a single rating hides."""
    acc: dict[tuple[str, str], list[float]] = {}
    for pair in rated_pairs(results):
        for r in pair:
            x, y = r.bots
            s = r.score(x)
            assert s is not None
            acc.setdefault((x, y), []).append(s)
            acc.setdefault((y, x), []).append(1 - s)
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


def _bt(results: list[GameResult], anchor: str, iterations: int) -> dict[str, float]:
    names = sorted({n for r in results for n in r.bots})
    wins = {n: 0.0 for n in names}
    games: dict[tuple[str, str], float] = {}
    for r in results:
        s = r.score(r.bots[0])
        if s is None:
            continue
        x, y = r.bots
        wins[x] += s
        wins[y] += 1 - s
        key = (min(x, y), max(x, y))
        games[key] = games.get(key, 0.0) + 1
    for (x, y) in list(games):
        games[(x, y)] += 1
        wins[x] += 0.5
        wins[y] += 0.5
    strength = {n: 1.0 for n in names}
    for _ in range(iterations):
        new = {}
        for n in names:
            denom = sum(
                g / (strength[x] + strength[y])
                for (x, y), g in games.items()
                if n in (x, y)
            )
            new[n] = wins[n] / denom if denom else strength[n]
        strength = new
    base = strength.get(anchor) or 1.0
    return {n: round(1000 + 400 * math.log10(strength[n] / base), 1) for n in names}
