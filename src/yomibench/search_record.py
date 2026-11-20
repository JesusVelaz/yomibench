"""What the search already records about each full turn, as training targets for the value net's
candidate heads (yomibench.train --aux-cand-value-weight / --aux-mix-weight): each seat's candidates,
the search's value of each against the other seat's mix (`candidate_values`, the seat's own side),
and the solved mixed strategy (`strategies`).

A turn qualifies when it was searched in full (not playout-capped) and records both seats'
`candidates` and `strategies`, list for list. Full turns can list thousands of candidates (Robot), so
each seat keeps at most K of them (cap_positions): the strategy's whole support (probability > 0;
the K likeliest when the support alone is larger), then a seeded uniform sample of the rest, in list
order. The kept strategy is renormalized over the kept candidates; the mix head's softmax runs over
the same list, so it learns the mix against sampled alternatives.

A candidate is encoded as yomibench.policy encodes it (action name, data shape, CAND_NUMS in world
x), and mirrored into its seat's view (policy.view_flip / flip_columns) when it is read. Encoded
candidates repeat across turns, so loaders keep one table of unique ones (CandTable) and the per-turn
lists hold table rows.

No torch here: record workers import this module.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from yomibench.policy import CAND_CATEGORIES, N_CAND, encode_candidate, view_flip

DEFAULT_K = 16  # candidates kept per seat per turn (train --aux-cand-k)


def cap_positions(strategy: list[float], k: int, rng: random.Random) -> list[int]:
    """Positions kept from a seat's list of len(strategy): the support (probability > 0), the `k`
    likeliest if it is larger (ties by position), else all of it plus a sample of the rest drawn by
    `rng` up to `k`; in list order."""
    support = [i for i, p in enumerate(strategy) if p > 0]
    if len(support) >= k:
        return sorted(sorted(support, key=lambda i: (-strategy[i], i))[:k])
    rest = [i for i, p in enumerate(strategy) if not p > 0]
    return sorted(support + rng.sample(rest, min(k - len(support), len(rest))))


def turn_candidates(turn: Any, k: int, key: str) -> list[tuple[float, list, list[float], list[float]]] | None:
    """For a full turn read with read_slim(policy=True), per seat: (view flip, encoded kept
    candidates, their search values (NaN where the search has none), their strategy probabilities
    renormalized over the kept ones). None when the turn doesn't qualify (see the module docstring).
    `key` seeds the sample (the record's name and the turn's index, so every loader keeps the same)."""
    if turn.cheap or not turn.strategies or not turn.candidates or turn.state is None:
        return None
    out = []
    for seat in (0, 1):
        cands, strat = turn.candidates[seat], turn.strategies[seat]
        if not cands or len(cands) != len(strat):
            return None
        cv = (turn.candidate_values or [None, None])[seat] or []
        keep = cap_positions([float(p) for p in strat], k, random.Random(f"{key}:{seat}"))
        probs = [float(strat[i]) for i in keep]
        total = sum(probs)
        vals = [float(cv[i]) if i < len(cv) and cv[i] is not None and math.isfinite(cv[i]) else math.nan for i in keep]
        out.append((view_flip(turn.state, seat), [encode_candidate(cands[i]) for i in keep], vals,
                    [p / total for p in probs] if total > 0 else probs))
    return out


def table_key(enc: tuple[str, str, tuple[float, ...]]) -> tuple[str, str, bytes]:
    """An encoded candidate's identity in a CandTable: its numbers as stored (float32)."""
    return enc[0], enc[1], np.asarray(enc[2], np.float32).tobytes()


class CandTable:
    """Unique encoded candidates: rows of `nums` (float32, world x) with their action and data-shape
    names."""

    def __init__(self):
        self.index: dict[tuple[str, str, bytes], int] = {}
        self.nums: list[np.ndarray] = []
        self.names: dict[str, list[str]] = {c: [] for c in CAND_CATEGORIES}

    def add(self, enc: tuple[str, str, tuple[float, ...]]) -> int:
        key = table_key(enc)
        u = self.index.get(key)
        if u is None:
            u = self.index[key] = len(self.nums)
            self.nums.append(np.frombuffer(key[2], np.float32))
            self.names["action"].append(enc[0])
            self.names["dkey"].append(enc[1])
        return u

    def unums(self) -> np.ndarray:
        return np.stack(self.nums) if self.nums else np.zeros((0, N_CAND), np.float32)


class SplitCands:
    """One split's candidate lists, built row by row: `slot` per row (-1: none) and per slot both
    seats' padded lists (uid -1 / value NaN / probability 0 where padded)."""

    def __init__(self, k: int):
        self.k, self.slot, self.flip, self.uid, self.val, self.prob = k, [], [], [], [], []

    def add_row(self, seats: list | None, table: CandTable) -> None:
        if seats is None:
            self.slot.append(-1)
            return
        self.slot.append(len(self.flip))
        uid = np.full((2, self.k), -1, np.int32)
        val = np.full((2, self.k), np.nan, np.float32)
        prob = np.zeros((2, self.k), np.float32)
        for s, (_, encs, vals, probs) in enumerate(seats):
            n = len(encs)
            uid[s, :n] = [table.add(e) for e in encs]
            val[s, :n], prob[s, :n] = vals, probs
        self.flip.append([seats[0][0], seats[1][0]])
        self.uid.append(uid)
        self.val.append(val)
        self.prob.append(prob)

    def arrays(self) -> dict[str, np.ndarray]:
        return split_arrays(np.asarray(self.slot, np.int64), self.flip, self.uid, self.val, self.prob, self.k)


def split_arrays(slot, flip, uid, val, prob, k: int) -> dict[str, np.ndarray]:
    """A split's candidate columns (train's `cand_*`), from per-slot pieces."""
    return {"cand_slot": np.asarray(slot, np.int64),
            "cand_flip": np.asarray(flip, np.float32).reshape(-1, 2),
            "cand_uid": np.asarray(uid, np.int32).reshape(-1, 2, k),
            "cand_val": np.asarray(val, np.float32).reshape(-1, 2, k),
            "cand_prob": np.asarray(prob, np.float32).reshape(-1, 2, k)}


def finish_table(unums: np.ndarray, names: dict[str, list[str]], train_uid: np.ndarray) -> dict[str, Any]:
    """The run's candidate table: `unums`, and per row its action and data-shape ids in the run's
    candidate vocabulary: the names the training split's candidates use, sorted (ids from 1; a name
    only validation uses is 0, the unknown row), so both loaders assign the same ids."""
    used = np.unique(train_uid[train_uid >= 0])
    vocab = {c: {n: i + 1 for i, n in enumerate(sorted({names[c][u] for u in used}))} for c in CAND_CATEGORIES}
    ids = {c: np.array([vocab[c].get(n, 0) for n in names[c]], np.int64) for c in CAND_CATEGORIES}
    return {"unums": np.asarray(unums, np.float32).reshape(-1, N_CAND), "uact": ids["action"], "udkey": ids["dkey"],
            "vocab": vocab}
