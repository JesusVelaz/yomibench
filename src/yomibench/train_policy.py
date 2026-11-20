"""Train and check the policy network prototype (yomibench.policy; needs torch).

    python -m yomibench.train_policy RECORD_DIR [RECORD_DIR ...] --out policy.npz \\
        --heldout-root DIR [--cap 60] [--report report.md]

Each folder's games split by its frozen held-out manifest (yomibench.heldout, as yomibench.train):
held-out games are the offline check and are never trained on. A seeded `--dev-fraction` of the
training games picks the best epoch, so the held-out games choose nothing.

A sample is one seat at one searched decision; the target is that seat's mixed strategy from the
search, folded onto the base candidate list, without the moves the search started from at random
(yomibench.policy.random_starts; a sample whose only support they were trains on nothing). Training keeps at most `--cap`
decisions per game (neighbouring decisions are near copies); the check uses every decision of
every held-out game (`--eval-cap` to cap it).

The offline check, per dataset (candidate set) and character pairing (the deciding seat's
character v the opponent's), for each K in `--ks`: the share of the search's support (base
candidates with folded probability > policy.SUPPORT) found in the policy's top K (recall), the
probability mass the top K covers, and the share of decisions whose whole support is in the top K
(cover). Baselines: K uniformly random candidates (in expectation) and the first K in list order
(what a candidate cap keeps). It is reported twice: against the training target (random start
moves dropped; decisions with no other support left out) and against the search's whole folded
strategy.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __name__ != "__mp_main__":  # record workers only need yomibench.policy (see yomibench.train)
    import torch
    from torch import nn
else:
    nn = SimpleNamespace(Module=object)

from yomibench import features, heldout
from yomibench.features import CATEGORY_OF_SLOT, N_NUMERIC, Vocab
from yomibench.policy import (CAND_CATEGORIES, CONSTANT_STD, EMBED as CEMBED, N_CAND, PolicyNet, flip_columns,
                              game_samples, random_metrics, topk_metrics, view_flip)
from yomibench.valuenet import EMBED

KS = (4, 8, 16)


# --- data ------------------------------------------------------------------------------------

class Table:
    """Unique candidates across every split: (action, dkey, nums) -> row of `nums`/`act`/`dkey`."""

    def __init__(self):
        self.index: dict[tuple, int] = {}
        self.nums: list[tuple[float, ...]] = []
        self.act: list[int] = []
        self.dkey: list[int] = []
        self.cvocab: dict[str, dict[str, int]] = {c: {} for c in CAND_CATEGORIES}
        self.frozen = False

    def _id(self, cat: str, name: str) -> int:
        table = self.cvocab[cat]
        if name not in table:
            if self.frozen:
                return 0
            table[name] = len(table) + 1
        return table[name]

    def add(self, enc: tuple[str, str, tuple[float, ...]]) -> int:
        u = self.index.get(enc)
        if u is None:
            u = self.index[enc] = len(self.nums)
            self.nums.append(enc[2])
            self.act.append(self._id("action", enc[0]))
            self.dkey.append(self._id("dkey", enc[1]))
        return u

    def arrays(self) -> dict[str, np.ndarray]:
        return {"unums": np.array(self.nums, dtype=np.float32).reshape(-1, N_CAND),
                "uact": np.array(self.act, dtype=np.int64), "udkey": np.array(self.dkey, dtype=np.int64)}


def _empty_split() -> dict[str, list]:
    return {k: [] for k in ("nums", "cats", "flip", "len", "group", "game", "uid", "target", "target_full", "cval")}


def _add_game(split: dict[str, list], game: dict[str, Any], game_id: int, vocab: Vocab, table: Table,
              groups: dict[str, int]) -> None:
    dataset = f"set {game['candidate_set'] or '?'}"
    for s in game["samples"]:
        state, seat = s["state"], s["seat"]
        nums, cats = features.encode(state, seat, vocab)
        split["nums"].append(np.asarray(nums, dtype=np.float32))
        split["cats"].append(np.asarray(cats, dtype=np.int32))
        split["flip"].append(view_flip(state, seat))
        split["len"].append(len(s["cands"]))
        name = f"{dataset} | {game['characters'][seat]} v {game['characters'][1 - seat]}"
        split["group"].append(groups.setdefault(name, len(groups)))
        split["game"].append(game_id)
        split["uid"].append(np.array([table.add(c) for c in s["cands"]], dtype=np.int32))
        split["target"].append(np.asarray(s["target"], dtype=np.float32))
        split["target_full"].append(np.asarray(s["target_full"], dtype=np.float32))
        split["cval"].append(np.array([np.nan if v is None else v for v in s["values"]], dtype=np.float32))


def _finish(split: dict[str, list]) -> dict[str, np.ndarray]:
    n = len(split["len"])
    out = {"nums": np.stack(split["nums"]) if n else np.zeros((0, N_NUMERIC), np.float32),
           "cats": np.stack(split["cats"]) if n else np.zeros((0, len(CATEGORY_OF_SLOT)), np.int32),
           "flip": np.array(split["flip"], dtype=np.float32), "group": np.array(split["group"], dtype=np.int32),
           "game": np.array(split["game"], dtype=np.int32)}
    out["off"] = np.concatenate([[0], np.cumsum(split["len"])]).astype(np.int64)
    for k in ("uid", "target", "target_full", "cval"):
        out[k] = np.concatenate(split[k]) if n else np.zeros(0, np.float32)
    return out


def load(dirs: list[Path], heldout_root: Path | None, cap: int | None, eval_cap: int | None, dev_fraction: float,
         seed: int, workers: int, fraction: float = 0.15, heldout_seed: int = 0) -> dict[str, Any]:
    """Every split's arrays (train, dev, test), the unique candidate table, both vocabularies,
    group names and the held-out manifests."""
    files: dict[str, list[Path]] = {"train": [], "dev": [], "test": []}
    manifests = {}
    for d in dirs:
        tr, te, manifests[str(d)] = heldout.split(Path(d), fraction, heldout_seed, heldout_root)
        dev = {p for p in tr if random.Random(f"{seed}:dev:{p.name}").random() < dev_fraction}
        files["train"] += [p for p in tr if p not in dev]
        files["dev"] += sorted(dev)
        files["test"] += te
    vocab, table, groups = Vocab(), Table(), {}
    out: dict[str, Any] = {"manifests": manifests, "games": {}}
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with ProcessPoolExecutor(workers) as pool:
        for name in ("train", "dev", "test"):
            vocab.frozen = table.frozen = name != "train"  # names first seen outside training map to 0
            fn = partial(game_samples, cap=cap if name != "test" else eval_cap, seed=seed)
            split, t0, n = _empty_split(), time.time(), 0
            for gid, game in enumerate(pool.map(fn, files[name], chunksize=4)):
                if name == "train" and game["first_state"] is not None:
                    vocab.learn_defaults(game["first_state"])
                _add_game(split, game, gid, vocab, table, groups)
                n += 1
                if n % 500 == 0:
                    print(f"  {name}: {n}/{len(files[name])} games, {len(split['len'])} samples, "
                          f"{time.time() - t0:.0f}s", flush=True)
            out[name] = _finish(split)
            out["games"][name] = n
            print(f"{name}: {n} games, {len(out[name]['flip'])} samples, {len(out[name]['uid'])} candidate rows",
                  flush=True)
    vocab.frozen = table.frozen = False
    out.update(table.arrays())
    out["vocab"], out["cvocab"], out["groups"] = vocab, table.cvocab, [g for g, _ in sorted(groups.items(), key=lambda x: x[1])]
    return out


def save_cache(path: Path, data: dict[str, Any]) -> None:
    arrays = {f"{s}/{k}": v for s in ("train", "dev", "test") for k, v in data[s].items()}
    arrays.update({k: data[k] for k in ("unums", "uact", "udkey")})
    meta = {"vocab": data["vocab"].ids, "raw_defaults": data["vocab"].raw_defaults, "cvocab": data["cvocab"],
            "groups": data["groups"], "manifests": data["manifests"], "games": data["games"]}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(tmp, path)


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        data: dict[str, Any] = {s: {} for s in ("train", "dev", "test")}
        for k in f.files:
            if "/" in k:
                s, name = k.split("/", 1)
                data[s][name] = f[k]
            elif k != "meta":
                data[k] = f[k]
    data["vocab"] = Vocab(ids=meta["vocab"], raw_defaults=meta["raw_defaults"])
    data.update(cvocab=meta["cvocab"], groups=meta["groups"], manifests=meta["manifests"], games=meta["games"])
    return data


def batch(data: dict[str, Any], split: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Samples `idx` padded to their longest candidate list (`mask` marks real candidates)."""
    off = split["off"]
    starts, lens = off[idx], off[idx + 1] - off[idx]
    width = int(lens.max())
    mask = np.arange(width)[None, :] < lens[:, None]
    pos = np.where(mask, starts[:, None] + np.arange(width)[None, :], 0)
    uid = split["uid"][pos]
    return {"nums": split["nums"][idx], "cats": split["cats"][idx].astype(np.int64),
            "cnums": flip_columns(data["unums"][uid], split["flip"][idx]),
            "cact": data["uact"][uid], "cdkey": data["udkey"][uid], "mask": mask,
            "target": np.where(mask, split["target"][pos], 0.0).astype(np.float32),
            "cval": np.where(mask, split["cval"][pos], np.nan).astype(np.float32)}


def batches(split: dict[str, np.ndarray], size: int, rng: np.random.Generator | None):
    """Index batches, each from one group (so a batch pads to lists of similar length); shuffled
    when `rng` is given."""
    out = []
    for g in np.unique(split["group"]):
        idx = np.flatnonzero(split["group"] == g)
        if rng is not None:
            idx = rng.permutation(idx)
        out += [idx[i:i + size] for i in range(0, len(idx), size)]
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def standardization(x: np.ndarray, sample: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    rows = x if len(x) <= sample else x[rng.choice(len(x), sample, replace=False)]
    rows = rows.astype(np.float64)
    return rows.mean(axis=0).astype(np.float32), (rows.std(axis=0) + 1e-3).astype(np.float32)


# --- model -----------------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, sizes: dict[str, int], csizes: dict[str, int], stats: dict[str, np.ndarray],
                 hidden: int = 256, width: int = 128):
        super().__init__()
        for k in ("mean", "std", "cmean", "cstd"):
            self.register_buffer(k, torch.tensor(stats[k]))
        self.emb = nn.ModuleDict({cat: nn.Embedding(sizes[cat], EMBED[cat]) for cat in EMBED})
        self.cemb = nn.ModuleDict({cat: nn.Embedding(csizes[cat], CEMBED[cat]) for cat in CEMBED})
        s_in = N_NUMERIC + sum(EMBED[c] for c in CATEGORY_OF_SLOT)
        self.s1, self.s2 = nn.Linear(s_in, hidden), nn.Linear(hidden, width)
        self.c1, self.c2 = nn.Linear(N_CAND + sum(CEMBED.values()), hidden // 2), nn.Linear(hidden // 2, width)
        self.o1, self.o2 = nn.Linear(3 * width, width), nn.Linear(width, 2)  # [score, candidate value]

    @staticmethod
    def _std(x, mean, std):
        return torch.where(std <= CONSTANT_STD, torch.zeros_like(x), (x - mean) / std)

    def forward(self, nums, cats, cnums, cact, cdkey):
        x = [self._std(nums, self.mean, self.std)]
        for slot, cat in enumerate(CATEGORY_OF_SLOT):
            x.append(self.emb[cat](cats[:, slot]))
        h = torch.relu(self.s2(torch.relu(self.s1(torch.cat(x, dim=1)))))
        c = torch.cat([self._std(cnums, self.cmean, self.cstd), self.cemb["action"](cact), self.cemb["dkey"](cdkey)],
                      dim=-1)
        c = torch.relu(self.c2(torch.relu(self.c1(c))))
        hb = h[:, None, :].expand_as(c)
        out = self.o2(torch.relu(self.o1(torch.cat([hb, c, hb * c], dim=-1))))
        return out[..., 0], out[..., 1]

    def export(self) -> dict[str, np.ndarray]:
        g = lambda t: t.detach().cpu().numpy().copy()  # noqa: E731
        p = {k: g(getattr(self, k)) for k in ("mean", "std", "cmean", "cstd")}
        for cat in EMBED:
            p[f"emb_{cat}"] = g(self.emb[cat].weight)
        for cat in CEMBED:
            p[f"emb_{cat}"] = g(self.cemb[cat].weight)
        for prefix, layers in (("s", (self.s1, self.s2)), ("c", (self.c1, self.c2)), ("o", (self.o1, self.o2))):
            for i, layer in enumerate(layers, start=1):
                w, b = g(layer.weight).T.copy(), g(layer.bias)
                if layer is self.o2:  # inference reads the score only
                    w, b = w[:, :1].copy(), b[:1].copy()
                p[f"{prefix}w{i}"], p[f"{prefix}b{i}"] = w, b
        return p


def _tensors(b: dict[str, np.ndarray], device: str) -> dict[str, Any]:
    return {k: torch.from_numpy(v).to(device) for k, v in b.items()}


def losses(model: Net, t: dict[str, Any]) -> tuple[Any, Any, int]:
    """(cross-entropy against the target, MSE on the candidate values, samples with a target):
    means over the batch's samples that have a target (an all-zero target trains on nothing)."""
    score, value = model(t["nums"], t["cats"], t["cnums"], t["cact"], t["cdkey"])
    logp = torch.log_softmax(score.masked_fill(~t["mask"], -1e9), dim=1)
    has = (t["target"].sum(dim=1) > 0).float()
    n = int(has.sum().item())
    ce = (-(t["target"] * logp).sum(dim=1) * has).sum() / max(n, 1)
    known = ~torch.isnan(t["cval"])
    mse = ((value - torch.nan_to_num(t["cval"])) ** 2)[known].mean() if known.any() else ce * 0
    return ce, mse, n


def predict(model: Net, data: dict[str, Any], split: dict[str, np.ndarray], device: str, size: int = 1024) -> np.ndarray:
    """Every candidate row's score, aligned with split['uid']."""
    out = np.zeros(len(split["uid"]), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for idx in batches(split, size, None):
            b = batch(data, split, idx)
            score, _ = model(*(torch.from_numpy(b[k]).to(device) for k in ("nums", "cats", "cnums", "cact", "cdkey")))
            score = score.cpu().numpy()
            for row, i in enumerate(idx):
                lo, hi = split["off"][i], split["off"][i + 1]
                out[lo:hi] = score[row, :hi - lo]
    return out


def evaluate_split(model: Net, data: dict[str, Any], split: dict[str, np.ndarray], device: str) -> float:
    model.eval()
    total = n = 0.0
    with torch.no_grad():
        for idx in batches(split, 1024, None):
            ce, _, m = losses(model, _tensors(batch(data, split, idx), device))
            total += float(ce) * m
            n += m
    return total / max(n, 1)


# --- offline check ---------------------------------------------------------------------------

def offline_check(split: dict[str, np.ndarray], scores: np.ndarray, groups: list[str],
                  ks: tuple[int, ...] = KS, field: str = "target") -> dict[str, Any]:
    """Per group and K: mean recall / mass / cover for the policy, first K and random K, against
    `field` ("target" or "target_full"); decisions whose target is all zeros are left out."""
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))
    info: dict[str, dict] = defaultdict(lambda: {"n": 0, "candidates": 0, "support": 0, "games": set()})
    for i in range(len(split["flip"])):
        lo, hi = split["off"][i], split["off"][i + 1]
        target, s = split[field][lo:hi], scores[lo:hi]
        if not target.sum() > 0:
            continue
        order = np.argsort(-s, kind="stable")
        first = np.arange(len(target))
        for name in (groups[split["group"][i]], "all"):
            g = info[name]
            g["n"] += 1
            g["candidates"] += len(target)
            g["support"] += int((target > 0.01).sum())
            g["games"].add(int(split["game"][i]))
            for k in ks:
                acc[name][("policy", k)] += topk_metrics(order, target, k)
                acc[name][("first", k)] += topk_metrics(first, target, k)
                acc[name][("random", k)] += random_metrics(target, k)
    out = {}
    for name, g in info.items():
        n = g["n"]
        out[name] = {"decisions": n, "games": len(g["games"]), "mean_candidates": g["candidates"] / n,
                     "mean_support": g["support"] / n,
                     "k": {str(k): {m: dict(zip(("recall", "mass", "cover"), (acc[name][(m, k)] / n).round(4).tolist()))
                                    for m in ("policy", "first", "random")} for k in ks}}
    return out


def report_table(check: dict[str, Any], ks: tuple[int, ...] = KS) -> str:
    def order(name):
        return (name == "all", name)
    lines = ["| dataset \\| pairing | games | decisions | cands | support | K | recall policy / first / random "
             "| mass policy / first / random | cover policy / first / random |",
             "|---|---:|---:|---:|---:|---:|---|---|---|"]
    for name in sorted(check, key=order):
        g = check[name]
        for k in ks:
            r = g["k"][str(k)]
            cells = [" / ".join(f"{r[m][metric]:.3f}" for m in ("policy", "first", "random"))
                     for metric in ("recall", "mass", "cover")]
            lines.append(f"| {name} | {g['games']} | {g['decisions']} | {g['mean_candidates']:.1f} | "
                         f"{g['mean_support']:.2f} | {k} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- main ------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="yomibench.train_policy")
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--heldout-root", type=Path, default=None,
                    help="keep held-out manifests here instead of in each record folder (for read-only folders)")
    ap.add_argument("--cap", type=int, default=60, help="decisions per training game (a seeded sample)")
    ap.add_argument("--eval-cap", type=int, default=None, help="decisions per held-out game (default: all)")
    ap.add_argument("--dev-fraction", type=float, default=0.05, help="training games that pick the best epoch")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="weight of the candidate-value regression (records' candidate_values) in the loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    ap.add_argument("--device", default=None, help="torch device (default: mps when available, else cpu)")
    ap.add_argument("--cache", type=Path, default=None, help="save the loaded arrays here, or load them if present")
    ap.add_argument("--report", type=Path, default=None, help="write the offline check as Markdown here")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.time()
    if args.cache and args.cache.exists():
        data = load_cache(args.cache)
        print(f"loaded {args.cache}", flush=True)
    else:
        data = load(args.dirs, args.heldout_root, args.cap, args.eval_cap, args.dev_fraction, args.seed, args.workers)
        if args.cache:
            save_cache(args.cache, data)
    load_s = time.time() - t0
    tr, dev, te = data["train"], data["dev"], data["test"]
    rng = np.random.default_rng(args.seed)
    # Candidate statistics over the rows training sees, mirrored into their views.
    sample = rng.choice(len(tr["uid"]), min(len(tr["uid"]), 2_000_000), replace=False)
    sample_owner = np.searchsorted(tr["off"], sample, side="right") - 1
    cn = flip_columns(data["unums"][tr["uid"][sample]], tr["flip"][sample_owner])
    cmean, cstd = standardization(cn, len(cn), rng)
    mean, std = standardization(tr["nums"], 500_000, rng)
    sizes = data["vocab"].sizes()
    csizes = {c: len(data["cvocab"][c]) + 1 for c in CAND_CATEGORIES}
    model = Net(sizes, csizes, {"mean": mean, "std": std, "cmean": cmean, "cstd": cstd}, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    best, best_state, history = float("inf"), None, []
    t1 = time.time()
    for epoch in range(args.epochs):
        model.train()
        tot = tot_mse = n = 0.0
        for idx in batches(tr, args.batch, rng):
            ce, mse, _ = losses(model, _tensors(batch(data, tr, idx), device))
            loss = ce + args.aux_weight * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += ce.item() * len(idx)
            tot_mse += mse.item() * len(idx)
            n += len(idx)
        sched.step()
        dev_ce = evaluate_split(model, data, dev, device) if len(dev["flip"]) else tot / n
        history.append({"epoch": epoch + 1, "train_ce": round(tot / n, 4), "train_value_mse": round(tot_mse / n, 5),
                        "dev_ce": round(dev_ce, 4)})
        print(f"epoch {epoch + 1}: train ce {tot / n:.4f}  value mse {tot_mse / n:.5f}  dev ce {dev_ce:.4f}  "
              f"({time.time() - t1:.0f}s)", flush=True)
        if dev_ce < best:
            best, best_state = dev_ce, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_s = time.time() - t1
    test_ce = evaluate_split(model, data, te, device)
    scores = predict(model, data, te, device)
    check = offline_check(te, scores, data["groups"])
    check_full = offline_check(te, scores, data["groups"], field="target_full")
    table = (f"Against the training target (random start moves dropped):\n\n{report_table(check)}\n\n"
             f"Against the search's whole folded strategy:\n\n{report_table(check_full)}")
    print(f"held-out ce {test_ce:.4f}\n{table}", flush=True)
    meta = {"name": args.out.stem, "dirs": [str(d) for d in args.dirs], "games": data["games"],
            "samples": {s: int(len(data[s]["flip"])) for s in ("train", "dev", "test")},
            "heldout": data["manifests"], "cap": args.cap, "eval_cap": args.eval_cap, "epochs": args.epochs,
            "aux_weight": args.aux_weight, "history": history, "best_dev_ce": round(best, 4),
            "heldout_ce": round(test_ce, 4), "offline_check": check, "offline_check_full": check_full, "load_seconds": round(load_s),
            "train_seconds": round(train_s), "device": device}
    PolicyNet(model.export(), data["vocab"], data["cvocab"], meta).save(args.out)
    print(f"saved {args.out}", flush=True)
    if args.report:
        args.report.write_text(f"held-out cross-entropy {test_ce:.4f} (best dev {best:.4f})\n\n{table}\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
