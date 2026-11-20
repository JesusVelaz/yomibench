"""Average the weights of value nets from one lineage into one model (weight averaging; KataGo gates a weight average
of snapshots). A candidate per training cycle: the cycle's warm-start fits, all fine-tuned from the same best.

    python tools/average_models.py OUT.npz MODEL.npz MODEL.npz [...] [--name NAME]

What can be averaged: models with the same architecture (the same param keys and shapes, apart from the optional
side heads, which are dropped unless every model has them), the same standardization (mean, std) and the same feature
version and raw defaults. Embedding rows are matched BY NAME: each fit numbers its vocabulary itself, so row i of one
model can be another name in the next. The output uses the first model's vocabulary; a name another model lacks enters
from that model's unknown row 0. Anything else refuses with the reason, so a cycle never screens a meaningless average.
The output's meta records the sources (name, sha256) and `averaged_from`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SIDE_PREFIXES = ("aux_search_", "aux_cand_", "aux_mix_", "w3_aux", "b3_aux")  # side heads: play never reads them


def _load(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return {k: f[k] for k in f.files if k != "meta"}, meta


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(paths: list[Path], name: str | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """(params, meta) of the average of `paths`' models; ValueError if they can't be averaged."""
    if len(paths) < 2:
        raise ValueError("averaging needs at least two models")
    models = [_load(p) for p in paths]
    p0, m0 = models[0]
    side = {k for k in p0 if k.startswith(SIDE_PREFIXES)}
    core = sorted(set(p0) - side)
    for (p, m), path in zip(models[1:], paths[1:]):
        missing = [k for k in core if k not in p]
        extra = sorted(set(p) - set(core) - {k for k in p if k.startswith(SIDE_PREFIXES)})
        if missing or extra:
            raise ValueError(f"{path.name}: different architecture (missing {missing[:4]}, extra {extra[:4]})")
        bad = [k for k in core if p[k].shape != p0[k].shape]
        if bad:
            raise ValueError(f"{path.name}: different shapes for {bad[:4]}")
        for k in ("mean", "std"):
            if k in p0 and not np.array_equal(p[k], p0[k]):
                raise ValueError(f"{path.name}: different standardization ({k}); not one lineage")
        for k in ("features", "raw_defaults"):
            if m.get(k) != m0.get(k):
                raise ValueError(f"{path.name}: different {k}")
        if set(m["vocab"]) != set(m0["vocab"]):
            raise ValueError(f"{path.name}: different embedding categories")
    keep_side = sorted(k for k in side if all(k in p and p[k].shape == p0[k].shape for p, _ in models[1:]))
    out: dict[str, np.ndarray] = {}
    for k in core + keep_side:
        if k in ("mean", "std"):
            out[k] = p0[k]
            continue
        if k.startswith("emb_"):
            cat = k[len("emb_"):]
            v0 = m0["vocab"][cat]
            acc = np.zeros_like(p0[k], dtype=np.float64)
            for p, m in models:
                v = m["vocab"][cat]
                rows = np.zeros(len(p0[k]), dtype=np.int64)  # row 0: unknown
                for nm, i in v0.items():
                    j = v.get(nm, 0)
                    rows[i] = j if j < len(p[k]) else 0
                acc += p[k][rows]
            out[k] = (acc / len(models)).astype(p0[k].dtype)
            continue
        out[k] = (sum(p[k].astype(np.float64) for p, _ in models) / len(models)).astype(p0[k].dtype)
    meta = dict(m0)
    meta["name"] = name or (m0.get("name", "model") + "-avg")
    meta["averaged_from"] = [{"name": m.get("name"), "path": str(pth), "sha256": _sha(pth)}
                             for (_, m), pth in zip(models, paths)]
    for k in ("best_val_mse", "val_mse", "eval_mse", "val_mse_by_phase", "val_mse_by_matchup", "best_epoch",
              "kept_epoch"):
        meta.pop(k, None)  # a fit's own validation numbers don't describe the average
    if not keep_side:
        for k in ("aux_search_heads", "aux_search_weight"):
            meta.pop(k, None)
        if "w3_aux" not in out:
            meta["aux_hp_weight"] = 0.0
    return out, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("models", type=Path, nargs="+")
    ap.add_argument("--name")
    a = ap.parse_args(argv)
    try:
        params, meta = average(a.models, a.name)
    except ValueError as e:
        print(f"average_models: not averaged: {e}", file=sys.stderr)
        return 2
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **params)
    tmp.replace(a.out)
    print(f"averaged {len(a.models)} models into {a.out} ({_sha(a.out)[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
