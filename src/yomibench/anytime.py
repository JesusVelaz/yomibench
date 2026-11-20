"""Anytime search for human play: answer before the game's turn timer, and (opt-in) think until
the human locks in.

**The turn deadline** (every search mode, every match): a decision has one turn of the match's
turn timer (`turn_seconds`; with no timer running, the match's `turn_time`), or less if
`GameTimer`, which follows the game's own clock for the bot's seat (ui/UILayer.gd: per-turn,
chess bank or increment), has less left, and must be answered `margin` before that runs out:
max(DEADLINE_MARGIN_S, DEADLINE_FRACTION x that time). First a `quick` answer is computed, production's quick
turn (`cap_candidates` 32 per seat, options-shuffle, no refinement: about a tenth of a full search),
as the guaranteed fallback; then the production search. If the deadline cuts the production
search, the quick answer is played; if it cuts even that, P2 holds (Continue). When the
deadline doesn't hit, the answer is exactly the plain search's.

**Think until I lock in** (opt-in): the bot answers a decision with the deepest search it finished before it had to stop. Stages run
in order; each one's answer replaces the stored one only when it completes and is certified
(converged), so a stop mid-stage keeps the last finished answer:

0. `quick` (only with a turn timer): the fallback above, in a private turn cache.
1. `production`: the bot's own search, exactly as human play runs it without this option (the
   same bot, settings and candidate view). Only a hard stop (the turn timer's, or the decision
   being superseded) can cut it short; a lock-in never does, so the answer is never weaker than
   the plain bot's.
2. `widen`: the same search with more factor refinement: DI neighbours for every support move (no
   3-move cap), near-optimal off-support moves expanded too (within 0.02 of the value), 4
   refinement rounds instead of 2. Its own candidate view; previews are shared with stage 1.
3. `two-decision` on the top K support pairs of the stored answer by
   r[i]*c[j] (K = 4, then 8): each pair's previewed state is a live branch (the preview's handle),
   its next decision is solved by double oracle with the evaluator at the leaves (base candidate
   lists, no expand), and that subgame's value replaces the pair's payoff; then the first decision
   is re-solved, and pairs that newly join the top K are branched (at most 2 rounds). Exact: a
   child is the handle's `continue` with that pair, the same clone-and-step a preview is.
4. `two-decision` with K = 16 and up to 3 re-solve rounds.

The plan proposed the Batch's spare slots for the branches. Human play's sim is a Handler (it must
follow the human's off-grid inputs with `step_raw`, which the Batch doesn't have), and the
Handler's preview handle plus `continue` is the same exact branch with no engine change.

Every engine and evaluator call goes through a stop check (`Clock`): previews in chunks of
`PREVIEW_CHUNK` pairs, value batches in chunks of `VALUE_CHUNK` states (stage 1 checks before each
batch but never splits one, so its numbers are the production search's bit for bit), and one check
per `continue`. A lock-in therefore lands within one chunk, about 50 ms.

Nothing here is used by self-play, gates or the arena; with the option off, human play only uses
its `Clock` (a few attributes per decision) and never runs a stage.
"""

from __future__ import annotations

import copy
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from yomibench.bots import FactorExplore, MatrixBot, preview_states, turn_candidates
from yomibench.engine import Candidate, Engine, Observation
from yomibench.playout import cap_candidates
from yomibench.search import double_oracle

PREVIEW_CHUNK = 128  # pairs per preview request in a stoppable stage (~350 us a pair on a loaded Mac)
VALUE_CHUNK = 256    # states per value-net batch in stages 2-4
# A decision is answered max(DEADLINE_MARGIN_S, DEADLINE_FRACTION x its time) before the turn timer
# runs out: a 15 s turn is answered by 12.75 s. Both are configurable (human_play --deadline-margin,
# --deadline-fraction).
DEADLINE_MARGIN_S = 2.0
DEADLINE_FRACTION = 0.15
QUICK_CAP, QUICK_POLICY = 32, "options-shuffle"  # production's quick turn (self-play's cheap_cap / cheap_cap_policy)
# The match settings that are the game's turn clock, not the simulation (ui/UILayer.gd and main.gd
# read them; game.gd never does): kept for the deadline, never sent to the sim.
TIMER_KEYS = ("timer_mode", "turn_time", "turn_min_length", "increment_starting_time", "increment_per_turn",
              "increment_max_time")


class Stopped(Exception):
    """A stop check fired; `why` is "lock-in", "deadline" or "superseded"."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


def new_work() -> dict[str, int]:
    """A stage's work tally: pairs previewed in the sim, positions the evaluator scored, value
    batches (one per oracle round or refinement step in matrix-do), expands and continues."""
    return {"previews": 0, "positions": 0, "value_batches": 0, "expands": 0, "continues": 0}


class Clock:
    """The stop signals of one decision. A soft stop (the human locked in) ends the extra stages;
    a hard stop (the turn timer's deadline, a `deadline` message, or the decision superseded by
    the next request) ends any stage, the production search too. Thread-safe: the transport's
    reader thread signals, the search thread checks."""

    def __init__(self, turn: int | None = None, hard_at: float | None = None,
                 superseded: Callable[[], bool] | None = None):
        self.turn, self.hard_at, self.superseded = turn, hard_at, superseded
        self.t0 = time.monotonic()
        self._soft, self._hard = threading.Event(), threading.Event()
        self._soft_why = self._hard_why = ""
        self.soft_at: float | None = None  # when the lock-in arrived (monotonic)
        # The running stage's work so far (reset by `run` per stage; Guarded/GuardedValue add to it after
        # each call returns): logged per stage, finished or cut. Counting only: no extra compute.
        self.work: dict[str, int] = new_work()

    def signal(self, kind: str) -> None:
        """A notification for this decision: `locked_in` (soft) or `deadline` (hard)."""
        if kind == "deadline":
            self.stop("deadline")
        else:
            if not self._soft.is_set():
                self._soft_why, self.soft_at = "lock-in", time.monotonic()
            self._soft.set()

    def stop(self, why: str) -> None:
        if not self._hard.is_set():
            self._hard_why = why
        self._hard.set()

    def hard(self) -> str | None:
        if self._hard.is_set():
            return self._hard_why
        if self.hard_at is not None and time.monotonic() >= self.hard_at:
            return "deadline"
        if self.superseded is not None and self.superseded():
            return "superseded"
        return None

    def soft(self) -> str | None:
        return self.hard() or (self._soft_why if self._soft.is_set() else None)

    def check_hard(self) -> None:
        why = self.hard()
        if why:
            raise Stopped(why)

    def check_soft(self) -> None:
        why = self.soft()
        if why:
            raise Stopped(why)


def turn_seconds(timer_settings: dict[str, Any]) -> float | None:
    """The seconds one turn has on the match's turn timer, which every answer must beat (user
    decision: the bot always answers before the game's turn timer, in every mode):
    `chess` its per-turn minimum `turn_min_length` (ranked: 15 s), `increment` its
    `increment_per_turn`, and otherwise `turn_time` seconds (`default`, and with no timer running,
    e.g. singleplayer's `timer_mode` none, the match's `turn_time`: the standard lobby's 30 s).
    None only when the settings name no positive per-turn time at all."""
    mode = timer_settings.get("timer_mode")
    key = {"chess": "turn_min_length", "increment": "increment_per_turn"}.get(mode, "turn_time")
    v = _num(timer_settings.get(key))
    if (v is None or v <= 0) and key != "turn_time":
        v = _num(timer_settings.get("turn_time")) if mode != "chess" else None
    return v if v is not None and v > 0 else None


def answer_within(available: float, margin_s: float = DEADLINE_MARGIN_S,
                  fraction: float = DEADLINE_FRACTION) -> float:
    """How long a decision with `available` seconds on the clock may take: the time less
    max(margin_s, fraction x the time), never below 0."""
    return max(0.0, available - max(margin_s, fraction * available))


def _num(v: Any) -> float | None:
    return None if isinstance(v, bool) or not isinstance(v, (int, float)) else float(v)


class GameTimer:
    """The game's turn timer for the bot's seat, as the real game runs it (ui/UILayer.gd,
    on_player_actionable and _apply_increment; main.gd sets it up):

    - `default`: `turn_time` seconds every turn.
    - `chess`: a bank of `turn_time` minutes; each turn after the first, a bank under
      `turn_min_length` seconds is topped up to it. Ranked: a 15-minute bank, 15 s minimum.
    - `increment`: a bank of max(`increment_starting_time`, `increment_per_turn`) seconds, plus
      `increment_per_turn` each turn after the first, capped at `increment_max_time` minutes (0: none).

    `available()` is the time this decision has; `spend(s)` charges what answering took (the bot's
    clock runs from the decision to its answer). `of(settings)` is None for no timer
    (`timer_mode` "none" or absent: the singleplayer default)."""

    def __init__(self, mode: str, turn_time: float, turn_min: float, inc: float, inc_start: float, inc_max_min: float):
        self.mode, self.turn_time, self.turn_min = mode, turn_time, turn_min
        self.inc, self.inc_max = inc, inc_max_min * 60
        self.first = True
        if mode == "chess":
            self.bank = turn_time * 60
        elif mode == "increment":
            self.bank = self._cap(max(inc_start, inc))
        else:
            self.bank = turn_time

    @staticmethod
    def of(settings: dict[str, Any]) -> "GameTimer | None":
        mode = settings.get("timer_mode")
        if not isinstance(mode, str) or mode == "none":
            return None
        tt, tmin = _num(settings.get("turn_time")), _num(settings.get("turn_min_length"))
        inc, inc0, incmax = (_num(settings.get(k)) for k in ("increment_per_turn", "increment_starting_time",
                                                               "increment_max_time"))
        if mode == "increment":
            inc = 10.0 if inc is None else inc  # main.gd's defaults
            return GameTimer(mode, 0.0, 0.0, inc, 60.0 if inc0 is None else inc0, incmax or 0.0)
        if tt is None or tt <= 0:
            return None  # main.gd starts no timer without turn_time
        # default, chess, and anything else the game would run like default (seconds per turn)
        return GameTimer(mode if mode == "chess" else "default", tt, tmin or 0.0, 0.0, 0.0, 0.0)

    def _cap(self, bank: float) -> float:
        return min(bank, self.inc_max) if self.inc_max > 0 else bank

    def available(self) -> float:
        if self.mode == "default":
            return self.turn_time
        if not self.first:
            if self.mode == "chess" and self.bank < self.turn_min:
                self.bank = self.turn_min
            elif self.mode == "increment":
                self.bank = self._cap(self.bank + self.inc)
        self.first = False
        return self.bank

    def spend(self, seconds: float) -> None:
        if self.mode != "default":
            self.bank = max(0.0, self.bank - seconds)

    def state(self) -> dict[str, Any]:
        return {"mode": self.mode, "bank_s": round(self.bank, 3)}


class Guarded:
    """An Engine with a stop check before every preview chunk, expand and continue. Attribute
    reads and writes (turn_cache, stats, match_seed) go to the engine itself."""

    def __init__(self, engine: Engine, check: Callable[[], None], chunk: int = PREVIEW_CHUNK,
                 work: dict[str, int] | None = None):
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_chunk", max(1, chunk))
        object.__setattr__(self, "_work", work if work is not None else new_work())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._engine, name, value)

    def preview(self, pairs: list[tuple[int, int]], candidates: bool = True) -> list[Observation]:
        out: list[Observation] = []
        for k in range(0, len(pairs), self._chunk):
            self._check()
            chunk = pairs[k:k + self._chunk]
            out += self._engine.preview(chunk, candidates)
            self._work["previews"] += len(chunk)
        return out

    def expand(self, specs: list[dict[str, Any]]) -> list[Candidate | str]:
        self._check()
        out = self._engine.expand(specs)
        self._work["expands"] += len(specs)
        return out

    def continue_from(self, handle: int, pair: tuple[int, int] | None = None, until: Any = "decision",
                      candidates: bool = True) -> Observation:
        self._check()
        out = self._engine.continue_from(handle, pair, until, candidates)
        self._work["continues"] += 1
        return out


class GuardedValue:
    """An evaluator (a function, or a value net with `batch`) with a stop check before each call.
    `chunk` splits batches (stages 2-4); None keeps each batch whole (stage 1: bit-identical)."""

    def __init__(self, value: Callable[[dict[str, Any], int], float], check: Callable[[], None],
                 chunk: int | None = None, work: dict[str, int] | None = None):
        self.value, self.check, self.chunk = value, check, chunk
        self.work = work if work is not None else new_work()
        self.__name__ = getattr(value, "__name__", repr(value))
        if getattr(value, "batch", None) is not None:
            self.batch = self._batch

    def __call__(self, state: dict[str, Any], seat: int) -> float:
        self.check()
        out = self.value(state, seat)
        self.work["positions"] += 1
        return out

    def _batch(self, states: list[dict[str, Any]], seat: int) -> Any:
        if not self.chunk:
            self.check()
            res = self.value.batch(states, seat)
            self.work["positions"] += len(states)
            self.work["value_batches"] += 1
            return res
        out: list[float] = []
        for k in range(0, len(states), self.chunk):
            self.check()
            part = states[k:k + self.chunk]
            out += map(float, self.value.batch(part, seat))
            self.work["positions"] += len(part)
        self.work["value_batches"] += 1
        return out

    def many(self, states: list[dict[str, Any]], seat: int = 0) -> list[float]:
        if not states:
            return []
        if getattr(self, "batch", None) is not None:
            return list(map(float, self.batch(states, seat)))
        out = [float(self(st, seat)) for st in states]
        self.work["value_batches"] += 1
        return out


@dataclass
class Answer:
    """One finished stage: both seats' strategies over `lists` (a bot's candidate view), p1's value."""
    stage: int
    name: str
    lists: list[list[Candidate]]
    row: list[float]
    col: list[float]
    value: float
    view: Any = None
    pairs: int = 0          # two-decision: first-decision pairs whose payoff is a solved subgame
    info: dict[str, Any] = field(default_factory=dict)
    ms: float = 0.0

    def strategy(self, seat: int) -> tuple[list[float], float]:
        return (self.row, self.value) if seat == 0 else (self.col, -self.value)

    def label(self) -> str:
        return {"quick": "quick search", "production": "full search",
                "widen": "widened"}.get(self.name, f"2-decision x{self.pairs}")


def widened(bot: MatrixBot) -> MatrixBot:
    """Stage 2's bot: the production bot with more factor refinement (see the module doc)."""
    b = copy.copy(bot)
    b.refine_rounds = max(4, bot.refine_rounds)
    b.di_refine_moves = None
    b.explore = FactorExplore(margin=0.02, random_k=0, all_prob=0.0)
    return b


class Deepener:
    """The stages for one decision of `bot` (the production MatrixBot) as `seat` on `engine`."""

    TWO_DECISION = ((3, 4, 2), (3, 8, 2), (4, 16, 3))  # (stage, K, re-solve rounds)

    def __init__(self, bot: MatrixBot, obs: Observation, seat: int, engine: Engine, clock: Clock):
        if not isinstance(bot, MatrixBot):
            raise ValueError("thinking until lock-in needs a matrix bot")
        self.bot, self.obs, self.seat, self.engine, self.clock = bot, obs, seat, engine, clock
        self.depth2: dict[tuple[int, int], float] = {}  # (id p1, id p2) -> solved subgame value (p1's)
        self.leaf: dict[tuple[Any, int, int], float] = {}  # (view, i, j) -> evaluator value (p1's)
        self.unbranchable: set[tuple[int, int]] = set()
        self.child_previews = 0
        # A bot's search key holds id(value): keep every stage's evaluator alive for the decision,
        # so no later stage's evaluator can reuse an id and meet an earlier stage's cached search.
        self.keep: list[Any] = []

    def stages(self, quick: bool = False, deep: bool = True) -> list[Stage]:
        """`quick`: the quick fallback first (a turn timer is on); `deep`: the extra stages after
        production (think until lock-in). The quick and production stages stop only on a hard stop."""
        out: list[Stage] = [(0, lambda _b: self.quick(), False)] if quick else []
        out.append((1, lambda _b: self.production(), False))
        if deep:
            out.append((2, self.widen, True))
            for stage, k, rounds in self.TWO_DECISION:
                out.append((stage, (lambda best, s=stage, k=k, r=rounds: self.two_decision(best, s, k, r)), True))
        return out

    # --- stage 0: the quick fallback ---------------------------------------------------------

    def quick(self) -> Answer:
        """Production's quick turn: each seat's candidates capped at QUICK_CAP (QUICK_POLICY),
        no factor refinement, searched in a private turn cache so nothing of it reaches the
        production search's lists or previews."""
        check = self.clock.check_hard
        b = self.bot.cheap()
        b.value = GuardedValue(self.bot.value, check, VALUE_CHUNK, self.clock.work)
        self.keep.append(b.value)
        eng = Guarded(self.engine, check, work=self.clock.work)
        base = [cap_candidates(c, QUICK_CAP, QUICK_POLICY, f"{self.engine.match_seed}:{self.obs.turn}:{s}")
                for s, c in enumerate(self.obs.candidates)]
        shared, private = self.engine.turn_cache, {"base": base}
        self.engine.turn_cache = private
        try:
            row, col, value = b.search_turn(self.obs, eng, self.seat)  # type: ignore[arg-type]
            lists = turn_candidates(self.obs, eng, b._key(self.seat))  # type: ignore[arg-type]
        finally:
            self.engine.turn_cache = shared
        info = {**dict(private.get("oracle") or {}), "cap": QUICK_CAP, "cap_policy": QUICK_POLICY}
        return Answer(0, "quick", lists, row, col, value, info=info)

    # --- stages 1 and 2: the matrix bot's own search -----------------------------------------

    def _search(self, bot: MatrixBot, stage: int, name: str, check: Callable[[], None],
                chunk: int | None) -> Answer:
        b = copy.copy(bot)
        b.value = GuardedValue(bot.value, check, chunk, self.clock.work)
        self.keep.append(b.value)
        eng = Guarded(self.engine, check, work=self.clock.work)
        row, col, value = b.search_turn(self.obs, eng, self.seat)  # type: ignore[arg-type]
        key = b._key(self.seat)
        lists = turn_candidates(self.obs, eng, key)  # type: ignore[arg-type]
        info = dict(self.engine.turn_cache.get("oracle") or {})
        return Answer(stage, name, lists, row, col, value, view=key, info=info)

    def production(self) -> Answer:
        return self._search(self.bot, 1, "production", self.clock.check_hard, None)

    def widen(self, _best: Answer | None) -> Answer | None:
        if self.bot.mode != "oracle":
            return None  # a full-table bot has nothing to refine
        return self._search(widened(self.bot), 2, "widen", self.clock.check_soft, VALUE_CHUNK)

    # --- stages 3 and 4: two-decision search on the stored answer ----------------------------

    def _leaf_values(self, base: Answer, value: GuardedValue, eng: Guarded):
        lists = base.lists

        def evaluate(pairs: list[tuple[int, int]]) -> list[float]:
            missing = [p for p in dict.fromkeys(pairs) if (base.view, *p) not in self.leaf]
            if missing:
                states = preview_states(self.obs, eng, missing, base.view)  # type: ignore[arg-type]
                for p, v in zip(missing, value.many(states)):
                    self.leaf[(base.view, *p)] = v
            out = []
            for i, j in pairs:
                ids = (lists[0][i].id, lists[1][j].id)
                out.append(self.depth2.get(ids, self.leaf[(base.view, i, j)]))
            return out
        return evaluate

    def branch(self, ids: tuple[int, int], value: GuardedValue, eng: Guarded) -> float | None:
        """The solved next decision after the pair `ids` (p1's value), or None (match over, no
        choice to solve, or the child search didn't converge: the pair keeps its one-decision
        payoff)."""
        (h,) = eng.preview([ids], candidates=True)
        handles = [h.handle] if h.handle is not None else []
        try:
            if h.done or h.handle is None:
                return None
            c0, c1 = h.candidates
            if not c0 or not c1 or len(c0) * len(c1) == 1:
                return None

            def child(pairs: list[tuple[int, int]]) -> list[float]:
                out: list[float] = []
                for k in range(0, len(pairs), PREVIEW_CHUNK):
                    obs = [eng.continue_from(h.handle, (c0[a].id, c1[b].id), candidates=False)
                           for a, b in pairs[k:k + PREVIEW_CHUNK]]
                    self.child_previews += len(obs)
                    self.engine.release([o.handle for o in obs if o.handle is not None])
                    out += value.many([o.state for o in obs])
                return out

            seed = f"{self.engine.match_seed}:{self.obs.turn}:{ids[0]}:{ids[1]}:depth2"
            start = tuple([0] + random.Random(f"{seed}:{s}").sample(range(1, n), min(self.bot.oracle_start, n - 1))
                          for s, n in enumerate((len(c0), len(c1))))
            r = double_oracle(len(c0), len(c1), child, self.bot.iterations, self.bot.tolerance, start=start)
            return r.value if r.converged else None
        finally:
            if handles:
                self.engine.release(handles)

    def two_decision(self, base: Answer | None, stage: int, k: int, rounds: int) -> Answer | None:
        if base is None:
            return None
        check = self.clock.check_soft
        value = GuardedValue(self.bot.value, check, VALUE_CHUNK, self.clock.work)
        eng = Guarded(self.engine, check, work=self.clock.work)
        lists, row, col = base.lists, base.row, base.col
        evaluate = self._leaf_values(base, value, eng)
        n0, n1 = len(lists[0]), len(lists[1])
        result, refused = None, 0
        for rnd in range(rounds):
            ranked = sorted(((row[i] * col[j], i, j) for i in range(n0) if row[i] > 0 for j in range(n1) if col[j] > 0),
                            key=lambda t: (-t[0], t[1], t[2]))[:k]
            todo = [(lists[0][i].id, lists[1][j].id) for _, i, j in ranked]
            todo = [ids for ids in todo if ids not in self.depth2 and ids not in self.unbranchable]
            if not todo and result is not None:
                break
            for ids in todo:
                v = self.branch(ids, value, eng)
                if v is None:
                    self.unbranchable.add(ids)
                    refused += 1
                else:
                    self.depth2[ids] = v
            support = ([i for i in range(n0) if row[i] > 0], [j for j in range(n1) if col[j] > 0])
            result = double_oracle(n0, n1, evaluate, self.bot.iterations, self.bot.tolerance, start=support)
            if not result.converged:
                return None  # never store an uncertified answer
            row, col = result.row, result.col
        assert result is not None
        pairs = len(self.depth2)
        info = {"rounds": result.rounds, "converged": True, "depth2_pairs": pairs, "unbranchable": refused,
                "child_previews": self.child_previews, "k": k}
        return Answer(stage, "two-decision", lists, row, col, result.value, view=base.view, pairs=pairs, info=info)


Stage = tuple[int, Callable[[Answer | None], "Answer | None"], bool]  # (number, stage, does a lock-in stop it)


@dataclass
class Outcome:
    best: Answer | None              # the deepest finished answer (None: the hard stop cut everything)
    stopped: str                     # "lock-in", "deadline", "superseded" or "done"
    trace: list[dict[str, Any]]      # per stage tried: stage, name, ms, and how it ended
    production: Answer | None        # the production search's answer, if it finished


def run(stages: list[Stage], clock: Clock) -> Outcome:
    """Run `stages` in order, keeping the last one that finished with an answer. A stop ends the
    run; a stage stopped part-way changes nothing. A stage whose flag is False (quick, production)
    runs whatever the soft stop says: only a hard stop ends it, or keeps it from starting."""
    best: Answer | None = None
    production: Answer | None = None
    trace: list[dict[str, Any]] = []
    stopped = "done"
    for n, stage, soft in stages:
        why = clock.soft() if soft else clock.hard()
        if why:
            stopped = why
            break
        t = time.perf_counter()
        clock.work.update(new_work())  # in place: the stage's guards hold this dict
        try:
            ans = stage(best)
        except Stopped as e:  # how far the stage got before the stop: its work so far
            trace.append({"stage": n, "ms": round((time.perf_counter() - t) * 1000, 1), "stopped": e.why,
                          "work": dict(clock.work)})
            stopped = e.why
            break
        ms = round((time.perf_counter() - t) * 1000, 1)
        if ans is None:
            trace.append({"stage": n, "ms": ms, "kept": False, "work": dict(clock.work)})
            continue
        ans.ms = ms
        trace.append({"stage": ans.stage, "name": ans.name, "ms": ms, "kept": True, "pairs": ans.pairs,
                      "value": round(ans.value, 6), "n_candidates": [len(ans.lists[0]), len(ans.lists[1])],
                      "work": dict(clock.work)})
        best = ans
        if ans.name == "production":
            production = ans
    return Outcome(best, stopped, trace, production)


def think(bot: MatrixBot, obs: Observation, seat: int, engine: Engine, clock: Clock, *, deep: bool = True,
          quick: bool = False) -> Outcome:
    """`bot`'s answer for `seat`: production's search (after the quick fallback with `quick`),
    then, with `deep`, deeper stages until the clock stops it or every stage is done."""
    return run(Deepener(bot, obs, seat, engine, clock).stages(quick=quick, deep=deep), clock)
