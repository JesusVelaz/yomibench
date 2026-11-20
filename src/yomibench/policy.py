"""A policy network prototype: score a turn's candidates from one seat's view, so the search can
look at the likely moves first (or only).

The target is the search's own answer: each turn's mixed strategy for a seat over that seat's
candidate list (records' `strategies`). Pruning acts on the engine's base list, the candidates a
decision starts with, so the target is folded onto it: a set-4 candidate the search added with
`expand` (a factor neighbour, recorded with `base`) gives its probability to the base candidate it
overrides. Set 2 and older have no expansions; there the base list is the whole list.

A candidate is encoded as its action name and the shape of its `data` (two vocabulary ids), plus
numbers (CAND_NUMS): the aim, slider amount and parry fields features.py reads from move data, a
hashed bag of every other data leaf, the `extra` fields (DI, feint, reverse, prediction, the
per-character flags, Ninja's release direction), the set-4 factors the candidate allows and its own
factor values. Horizontal quantities are in world x, mirrored per view like features.py (columns in
X_COLUMNS flip with the seat's view). The state is features.encode for that seat. The net scores
each candidate with a small MLP over (state embedding, candidate embedding, their product), and a
softmax over the turn's list is the policy; training (yomibench.train_policy, torch) minimises
cross-entropy against the folded strategy.

Inference here is plain numpy (no torch), like yomibench.valuenet, so a bot can rank candidates:

    net = PolicyNet.load("policy.npz")
    scores = net.scores(state, seat, candidates)   # one score per candidate, higher first

`game_samples` (record -> compact samples) runs in training's worker processes, so this module must
not import torch.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from pathlib import Path
from typing import Any

import numpy as np

from yomibench import features
from yomibench.features import Vocab, _move_data, _num

FEATURES = 1  # bump when encode_candidate changes; training writes this version
SUPPORT = 0.01  # a candidate is in the search's support when its (folded) probability exceeds this
DATA_BUCKETS = 16
EXTRA_FLAGS = ("reverse", "feint", "explode", "pull", "detach", "release", "detonate", "drift", "gun_cancel",
               "hindsight", "input_aerial", "input_grounded", "shift")
FACTORS = ("di", "feint", "reverse", "extra_set")
CAND_NUMS = (
    ("aim_x", "aim_y", "aim_len", "amount", "parry_timing", "parry_height")
    + tuple(f"data_{k}" for k in range(DATA_BUCKETS))
    + ("di_x", "di_y", "prediction", "release_dir_x", "release_dir_y", "extra_other")
    + tuple(f"extra_{f}" for f in EXTRA_FLAGS)
    + ("has_factors",) + tuple(f"allows_{f}" for f in FACTORS) + tuple(f"value_{f}" for f in FACTORS)
)
N_CAND = len(CAND_NUMS)
X_COLUMNS = tuple(CAND_NUMS.index(n) for n in ("aim_x", "di_x", "release_dir_x"))
CAND_CATEGORIES = ("action", "dkey")
EMBED = {"action": 16, "dkey": 4}
# Bags hash data leaves named "x" with their sign, so they would need flipping too; they are
# mostly the same aims `_move_data` reads (already in aim_x), so the bag drops their sign instead.


def _bucket(key: str) -> int:
    return zlib.crc32(key.encode()) % DATA_BUCKETS


def _data_bag(data: Any, out: list[float], path: str = "") -> None:
    """Every leaf of a candidate's `data`, hashed by its path (and value, for strings and flags)."""
    if isinstance(data, dict):
        for k in sorted(data):
            _data_bag(data[k], out, f"{path}/{k}")
    elif isinstance(data, bool) or data is None or isinstance(data, str):
        if data not in (False, None):
            out[_bucket(f"{path}={data}")] += 1.0
    elif isinstance(data, (int, float)):
        v = abs(float(data)) if path.endswith("/x") else float(data)
        out[_bucket(path)] += v / 100.0
    else:
        out[_bucket(f"{path}:{type(data).__name__}")] += 1.0


def data_key(data: Any) -> str:
    """The shape of a candidate's data: its top-level keys, or its type."""
    if isinstance(data, dict):
        return ",".join(sorted(data))
    return type(data).__name__


def encode_candidate(c: dict[str, Any]) -> tuple[str, str, tuple[float, ...]]:
    """(action name, data shape, CAND_NUMS in world x) for a recorded candidate (Candidate.record())."""
    data, extra = c.get("data"), c.get("extra") or {}
    ax, ay, amount, timing, height = _move_data(data)
    aim_len = math.hypot(ax, ay)
    if aim_len:
        ax, ay = ax / aim_len, ay / aim_len
    bag = [0.0] * DATA_BUCKETS
    _data_bag(data, bag)
    di = extra.get("DI") or {}
    rd = extra.get("release_dir") or {}
    known = {"DI", "release_dir", "prediction", *EXTRA_FLAGS}
    other = sum(1.0 for k, v in extra.items() if k not in known and v not in (False, None, 0, -1))
    factors = c.get("factors") or {}
    values = c.get("values") or {}
    nums = ([ax, ay, aim_len, amount, timing, height] + bag
            + [_num(di.get("x")), _num(di.get("y")), _num(extra.get("prediction", -1)), _num(rd.get("x")),
               _num(rd.get("y")), other]
            + [float(bool(extra.get(f))) for f in EXTRA_FLAGS]
            + [float(bool(factors))] + [_num(factors.get(f, 0)) for f in FACTORS]
            + [_num(values.get(f, 0)) for f in FACTORS])
    return str(c.get("action", "")), data_key(data), tuple(float(x) for x in nums)


def view_flip(state: dict[str, Any], seat: int) -> float:
    """features.encode's mirror for `seat`: 1 when it stands left of the opponent, else -1."""
    me, opp = state["players"][seat], state["players"][1 - seat]
    return 1.0 if me["x"] <= opp["x"] else -1.0


def flip_columns(nums: np.ndarray, flip: np.ndarray | float) -> np.ndarray:
    """Candidate numbers (rows of CAND_NUMS) mirrored into a view: X_COLUMNS times `flip`
    (a scalar, or one per row)."""
    out = np.array(nums, dtype=np.float32, copy=True)
    f = np.asarray(flip, dtype=np.float32)
    out[..., list(X_COLUMNS)] *= f.reshape(f.shape + (1,) * (out.ndim - f.ndim))
    return out


def base_positions(cands: list[dict[str, Any]]) -> list[int]:
    """Positions of the engine's own candidates in a recorded list (not added by `expand`)."""
    return [i for i, c in enumerate(cands) if c.get("base") is None]


def fold_to_base(cands: list[dict[str, Any]], strategy: list[float]) -> tuple[list[int], list[float]]:
    """(base positions, each one's probability with its expansions' folded in). An expansion's
    `base` is the id (= position) of the candidate it overrides."""
    base = base_positions(cands)
    where = {i: k for k, i in enumerate(base)}
    probs = [0.0] * len(base)
    for i, (c, p) in enumerate(zip(cands, strategy)):
        b = c.get("base")
        k = where.get(i if b is None else int(b))
        if k is not None:
            probs[k] += float(p)
    total = sum(probs)
    return base, [p / total for p in probs] if total > 0 else probs


def random_starts(header: dict[str, Any], turn: int, tick: Any, sizes: tuple[int, int]) -> list[set[int]]:
    """The base candidates (positions) a recorded double-oracle search started from at random, per
    seat, besides Continue: MatrixBot draws `oracle_start` of them, seeded, before anything is
    solved. In tied positions they tend to stay in the support whatever the position is, so they
    are noise in a policy target. `sizes` is both seats' base list lengths (the search's lists
    before any expansion). Two seeding schemes exist: since 5d5d029 (which also added the header's
    `opening` key) each seat draws from its own stream keyed by the match seed; before it, one
    stream keyed by turn and tick drew seat 0's, then seat 1's. Empty for non-oracle searches."""
    bot = (header.get("bots") or [{}])[0]
    k = int(bot.get("oracle_start") or 0)
    if bot.get("search") != "oracle" or k <= 0:
        return [set(), set()]
    if "opening" in header:
        seed = header.get("seed")
        return [set(random.Random(f"{seed}:{turn}:{tick}:{s}").sample(range(1, n), min(k, n - 1))) if n > 1 else set()
                for s, n in enumerate(sizes)]
    rng = random.Random(f"{turn}:{tick}")
    return [set(rng.sample(range(1, n), min(k, n - 1))) if n > 1 else set() for n in sizes]


def pairing(characters: list[str], seat: int) -> str:
    """'Me v Opponent' from `seat`'s side."""
    return f"{characters[seat]} v {characters[1 - seat]}"


def game_samples(path: Path, cap: int | None = None, seed: int = 0) -> dict[str, Any]:
    """One record's policy samples (a worker-process job: no torch). A sample is one seat at one
    searched decision (full, not playout-capped) with at least two base candidates. `cap` keeps a
    seeded sample of at most that many decisions per game (neighbouring decisions are near copies,
    so a cap loses little). Returns {"characters", "candidate_set", "first_state", "samples":
    [{turn, seat, state, cands: [(action, dkey, nums)], target, target_full, values}]}.
    `target_full` is the folded strategy; `target` is the same with the search's random start
    moves (random_starts) removed and renormalized (all zeros when only they had mass: such a
    sample trains on nothing). `values` is each base candidate's own search value against the
    other seat's mix (None if not evaluated; `candidate_values`)."""
    from yomibench.selfplay import read_slim

    header, _, turns = read_slim(Path(path), policy=True)
    chars = [str(c) for c in header.get("characters", ("?", "?"))]
    # A playout-capped (cheap) decision searched a capped list without refinement: a weak target.
    eligible = [k for k, t in enumerate(turns) if t.strategies and t.candidates and not t.cheap]
    if cap is not None and len(eligible) > cap:
        eligible = sorted(random.Random(f"{seed}:{Path(path).name}").sample(eligible, cap))
    samples = []
    for k in eligible:
        t = turns[k]
        sizes = tuple(len(base_positions(c)) for c in t.candidates)
        starts = random_starts(header, t.turn if t.turn is not None else k, t.state.get("tick"), sizes)
        for seat in (0, 1):
            cands, strat = t.candidates[seat], t.strategies[seat]
            if len(cands) != len(strat):
                continue
            base, full = fold_to_base(cands, strat)
            if len(base) < 2 or sum(full) <= 0:
                continue
            kept = [0.0 if i in starts[seat] else p for i, p in enumerate(full)]
            total = sum(kept)
            target = [p / total for p in kept] if total > 0 else kept
            cv = (t.candidate_values or [None, None])[seat] or []
            values = [cv[i] if i < len(cv) else None for i in base]
            samples.append({"turn": k, "seat": seat, "state": t.state,
                            "cands": [encode_candidate(cands[i]) for i in base], "target": target,
                            "target_full": full, "values": values})
    cset = (header.get("engine") or {}).get("candidate_set")
    return {"characters": chars, "candidate_set": None if cset is None else str(cset),
            "first_state": turns[0].state if turns else None, "samples": samples}


# --- offline metrics -------------------------------------------------------------------------

def topk_metrics(order: list[int] | np.ndarray, target: np.ndarray, k: int) -> tuple[float, float, float]:
    """(recall of the support, probability mass, whole support covered) for the first `k` of
    `order` (candidate positions, best first) against `target` (folded strategy)."""
    support = target > SUPPORT
    s = int(support.sum())
    top = np.asarray(order[:k], dtype=np.int64)
    hit = int(support[top].sum())
    return (hit / s if s else 1.0), float(target[top].sum()), float(hit == s)


def random_metrics(target: np.ndarray, k: int) -> tuple[float, float, float]:
    """topk_metrics' expectation for `k` candidates drawn uniformly without replacement."""
    n = len(target)
    s = int((target > SUPPORT).sum())
    m = min(k, n)
    if s == 0:
        return 1.0, m / n, 1.0
    cover = 0.0
    if m >= s:  # P(all s support moves among the m drawn) = C(n-s, m-s) / C(n, m)
        cover = math.exp(math.lgamma(n - s + 1) - math.lgamma(m - s + 1) - math.lgamma(n - m + 1)
                         - math.lgamma(n + 1) + math.lgamma(m + 1) + math.lgamma(n - m + 1))
    return m / n, m / n, cover


# --- numpy inference -------------------------------------------------------------------------

def _mlp(x: np.ndarray, p: dict[str, np.ndarray], prefix: str, layers: int) -> np.ndarray:
    for i in range(1, layers + 1):
        x = x @ p[f"{prefix}w{i}"] + p[f"{prefix}b{i}"]
        if i < layers:
            x = np.maximum(x, 0)
    return x


CONSTANT_STD = 1e-3 * (1 + 1e-4)  # as valuenet: inputs constant in training enter as 0


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.where(std <= CONSTANT_STD, 0.0, (x - mean) / std).astype(np.float32)


def forward(p: dict[str, np.ndarray], nums: np.ndarray, cats: np.ndarray, cnums: np.ndarray,
            cact: np.ndarray, cdkey: np.ndarray) -> np.ndarray:
    """Scores for one state's candidates: the same computation as train_policy.Net.forward, in
    numpy. `nums`/`cats` are one state view (features.encode), `cnums` the candidates' numbers
    already mirrored into that view, `cact`/`cdkey` their vocabulary ids."""
    x = [_standardize(nums[None, :], p["mean"], p["std"])]
    for slot, cat in enumerate(features.CATEGORY_OF_SLOT):
        table = p[f"emb_{cat}"]
        x.append(table[min(int(cats[slot]), len(table) - 1)][None, :])
    h = np.maximum(_mlp(np.concatenate(x, axis=1), p, "s", 2), 0)  # (1, H)
    ta, td = p["emb_action"], p["emb_dkey"]
    c = np.concatenate([_standardize(cnums, p["cmean"], p["cstd"]),
                        ta[np.minimum(cact, len(ta) - 1)], td[np.minimum(cdkey, len(td) - 1)]], axis=1)
    c = np.maximum(_mlp(c, p, "c", 2), 0)  # (n, H)
    hb = np.broadcast_to(h, c.shape)
    return _mlp(np.concatenate([hb, c, hb * c], axis=1), p, "o", 2)[:, 0]


class PolicyNet:
    def __init__(self, params: dict[str, np.ndarray], vocab: Vocab, cvocab: dict[str, dict[str, int]],
                 meta: dict[str, Any]):
        self.p = params
        self.vocab = vocab
        self.vocab.frozen = True
        self.cvocab = cvocab
        self.meta = meta

    @staticmethod
    def load(path: str | Path) -> "PolicyNet":
        with np.load(path, allow_pickle=False) as f:
            params = {k: f[k] for k in f.files if k != "meta"}
            meta = json.loads(str(f["meta"]))
        if meta.get("policy_features") != FEATURES or meta.get("features") != 3:
            raise ValueError(f"{path}: policy feature version {meta.get('policy_features')} / state version "
                             f"{meta.get('features')}; this code plays {FEATURES} / 3")
        vocab = Vocab(ids=meta.pop("vocab"), raw_defaults=meta.pop("raw_defaults", {}))
        return PolicyNet(params, vocab, meta.pop("cand_vocab"), meta)

    def save(self, path: str | Path) -> None:
        meta = {**self.meta, "policy_features": FEATURES, "features": 3, "vocab": self.vocab.ids,
                "raw_defaults": self.vocab.raw_defaults, "cand_vocab": self.cvocab}
        np.savez(path, meta=np.array(json.dumps(meta)), **self.p)

    def cand_ids(self, encoded: list[tuple[str, str, tuple[float, ...]]]) -> tuple[np.ndarray, np.ndarray]:
        a, d = self.cvocab.get("action", {}), self.cvocab.get("dkey", {})
        return (np.array([a.get(e[0], 0) for e in encoded], dtype=np.int64),
                np.array([d.get(e[1], 0) for e in encoded], dtype=np.int64))

    def scores(self, state: dict[str, Any], seat: int, candidates: list[Any]) -> np.ndarray:
        """One score per candidate (records' dicts or engine Candidates); higher is likelier."""
        recs = [c if isinstance(c, dict) else c.record() for c in candidates]
        encoded = [encode_candidate(c) for c in recs]
        nums, cats = features.encode(state, seat, self.vocab)
        cact, cdkey = self.cand_ids(encoded)
        cnums = flip_columns(np.array([e[2] for e in encoded], dtype=np.float32), view_flip(state, seat))
        return forward(self.p, np.asarray(nums, np.float32), np.asarray(cats, np.int64), cnums, cact, cdkey)
