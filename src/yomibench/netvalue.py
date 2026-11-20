"""Scoring previewed positions with value nets, on features an engine computed for each preview (an engine that
returns features rather than states, such as a batch engine plugin). `BatchValue` gives the same numbers as
`ValueNet.batch` on the states, bit for bit."""
from __future__ import annotations

import json

import numpy as np

from yomibench import features
from yomibench.valuenet import ValueNet, forward, hp_lead, residual_value


def net_key(net: ValueNet) -> str:
    """What a net's features depend on: its feature version, vocabulary and raw defaults."""
    return json.dumps([net.version, net.vocab.ids, net.vocab.raw_defaults], sort_keys=True)


class Preview(dict):
    """One previewed pair. As a dict it is the next state when the engine was asked for states
    (else empty); `feats[net_key(net)]` is (nums (2, n_numeric), cats (2, n_categorical)), both
    seats' value-net features of that state for each net the engine serves; `winner` is the
    Batch's (0, 1, or -1 for none) and `done` whether the game is over there."""

    __slots__ = ("feats", "done", "winner")

    def terminal(self, seat: int) -> float | None:
        """`search.terminal_value(state, seat)`: the exact result if the game is over (KO or time:
        the Batch's `done` is the same test), else None."""
        if not self.done:
            return None
        return 0.0 if self.winner < 0 else (1.0 if self.winner == seat else -1.0)


def v3_strip_columns() -> tuple[np.ndarray, np.ndarray]:
    """The columns of a v3 encoding that `valuenet.strip_v3_state` changes, and the values they
    take there (features.py: a state without the move fields encodes `_move` as zeros but for
    DI; objects without hitboxes have no `_object_hit` and so no threats but their count)."""
    f = features
    cols: list[int] = []
    vals: list[float] = []
    move_at = len(f.PLAYER_NUMS) + len(f.TIMING_NUMS)
    for player_at in (0, f.PLAYER_WIDTH):
        for k, name in enumerate(f.MOVE_NUMS):
            if name not in ("di_x", "di_y"):
                cols.append(player_at + move_at + k)
                vals.append(0.0)
    objects_at = 2 * f.PLAYER_WIDTH
    for slot in range(f.OBJECT_SLOTS):
        for k, name in enumerate(f.OBJECT_NUMS):
            if name.startswith("hit_"):
                cols.append(objects_at + slot * len(f.OBJECT_NUMS) + k)
                vals.append(0.0)
    no_threat = {"active": 0.0, "active_damage": 0.0, "nearest": f.FAR, "damage_near": 0.0, "soonest": f.SOON}
    globals_at = objects_at + f.OBJECT_SLOTS * len(f.OBJECT_NUMS)
    for k, name in enumerate(f.GLOBAL_NUMS):
        side, _, what = name.partition("_")
        if side in ("mine", "theirs") and what in no_threat:
            cols.append(globals_at + k)
            vals.append(no_threat[what])
    return np.asarray(cols), np.asarray(vals, dtype=np.float32)


class BatchValue:
    """A `ValueNet` that scores `Preview`s: the same numbers as `ValueNet.batch` on their states
    (same encoder inputs, bit for bit, through the same `forward`, in the same batch shapes). A
    net that plays on states without the v3 move fields (`ValueNet.mask_v3_state`) gets the
    Batch's v3 features with those fields' columns set as a stripped state encodes them. A
    `residual_hp` net adds the hp lead read from the same features, as `ValueNet.batch` does."""

    def __init__(self, net: ValueNet):
        self.net = net
        self.key = net_key(net)
        self.__name__ = net.__name__  # bot configs (and records) name the value function by it
        self.strip = v3_strip_columns() if getattr(net, "mask_v3_state", False) else None

    def batch(self, previews: list[Preview], seat: int) -> np.ndarray:
        nums = np.stack([p.feats[self.key][0] for p in previews])
        if self.strip is not None:
            nums[:, :, self.strip[0]] = self.strip[1]
        cats = np.stack([p.feats[self.key][1] for p in previews]).astype(np.int64)
        out = [forward(self.net.p, nums[:, view], cats[:, view], self.net.encoder.CATEGORY_OF_SLOT)
               for view in (seat, 1 - seat)]
        values = (out[0] - out[1]) / 2
        if getattr(self.net, "residual_hp", False):
            values = residual_value(values, hp_lead(nums[:, seat], self.net.encoder))
        for k, p in enumerate(previews):  # a finished game has an exact value; the net never decides it
            t = p.terminal(seat)
            if t is not None:
                values[k] = t
        return values

    def __call__(self, state: Preview, seat: int) -> float:
        return float(self.batch([state], seat)[0])
