"""Why is the value net's validation error flat? Load a record set once, then compare models and
baselines on the same split.

    python tools/value_diagnose.py GEN_DIR [GEN_DIR ...] --games-per-gen 3000 --out diag.json

Exploration: this diagnostic loads with load_samples' default `--explore-targets uniform-only`
and value weight 1, then sweeps a blend over every training position, so positions before a
uniform pick are blended here where train.py would put them on the value alone. Validation drops
the positions at or before a game's last uniform pick, as in train.py; report explored counts
and treat the sweep as a target-policy ablation when they are nonzero.

Reports, by game phase and by generation, against the final outcome:
  - baselines: a constant (the training mean), the hand-written `evaluate` (hp lead), the same
    rescaled by a least-squares factor fitted on training positions, and the search's own
    recorded value (what --value-weight blends in);
  - for each hidden size (value weight 0, as checkpoint 1 trained): train and validation MSE per
    epoch, so memorizing game outcomes shows up as the two curves parting;
  - for each value weight w: held-out MSE against the outcome and against the blended target
    (1-w)*outcome + w*value that training actually fit, so the gap between them is visible.

A game is one outcome bit shared by its ~110 positions, so outcome-trained nets can learn games
rather than positions. Everything is printed and, with --out, saved as JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def mse(a, b, m=None):
    if m is not None:
        a, b = a[m], b[m]
    return float(np.mean((a - b) ** 2)) if len(a) else float("nan")


def groups(split, gens):
    from yomibench.train import PHASES

    out = [("all", np.ones(len(split["outcome"]), bool))]
    out += [(name, (split["phase"] >= lo) & (split["phase"] < hi)) for lo, hi, name in PHASES]
    if len(gens) > 1:
        out += [(g, split["gen"] == k) for k, g in enumerate(gens)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--games-per-gen", type=int, default=None)
    ap.add_argument("--hidden", default="64,256,512")
    ap.add_argument("--weights", default="0,0.25,0.5,0.75")
    ap.add_argument("--weights-hidden", type=int, default=256, help="hidden size for the value-weight A/B")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout-root", type=Path, default=None,
                    help="keep held-out manifests here, not in the record folders (see yomibench.heldout)")
    ap.add_argument("--save-models", type=Path, default=None,
                    help="folder to save each value-weight run's best model (for arenas)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    import torch

    from yomibench.features import Vocab
    from yomibench.train import Net, _standardization, load_samples
    from yomibench.valuenet import ValueNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    vocab = Vocab()
    # value_weight 1: y is the search value (the outcome where a turn had no search), so any blend
    # is (1-w)*outcome + w*y.
    data = load_samples(args.dirs, vocab, 0.15, args.seed, value_weight=1.0, workers=args.workers, device=device,
                        games_per_gen=args.games_per_gen, heldout_root=args.heldout_root)
    tr, va = data["train"], data["val"]
    gens = [d.name for d in args.dirs]
    n_tr = len(tr["y"])
    print(f"loaded {tr['games']} train games ({n_tr} positions), {va['games']} val games ({len(va['y'])}) "
          f"in {time.perf_counter() - t0:.0f}s on {device}", flush=True)
    print("states with engine-v3 move fields (state_v3_fraction): train "
          + ", ".join(f"{g} {'-' if f is None else f'{f:.1%}'}" for g, f in zip(gens, tr["v3_fraction_by_gen"]))
          + f", all {tr['v3_fraction']:.1%}; val all {va['v3_fraction']:.1%}", flush=True)
    results = {"settings": vars(args) | {"dirs": [str(d) for d in args.dirs],
                      "out": str(args.out), "save_models": str(args.save_models)},
               "source_sha": _git_sha(),
               "device": device, "torch": torch.__version__, "platform": platform.platform(),
               "command": sys.argv, "dirs": [str(d) for d in args.dirs], "games_per_gen": args.games_per_gen,
               "train_games": tr["games"], "val_games": va["games"], "train_positions": n_tr,
               "val_positions": len(va["y"]), "explored": [tr["explored"], va["explored"]],
               "state_v3_fraction": {"train": tr["v3_fraction"], "val": va["v3_fraction"],
                                     "train_by_gen": dict(zip(gens, tr["v3_fraction_by_gen"])),
                                     "val_by_gen": dict(zip(gens, va["v3_fraction_by_gen"]))}}

    z_tr = tr["outcome"].cpu().numpy().astype(np.float64)
    v_tr = tr["y"].cpu().numpy().astype(np.float64)
    z_va, v_va = va["outcome"].astype(np.float64), va["y"].astype(np.float64)
    g_va = groups(va, gens)

    # Baselines.
    c = float(z_tr.mean())
    scale = float(np.dot(tr["baseline"], z_tr) / max(1e-12, np.dot(tr["baseline"], tr["baseline"])))
    base = {"zero": np.zeros_like(z_va), "constant": np.full_like(z_va, c), "evaluate": va["baseline"], "evaluate_scaled": scale * va["baseline"],
            "search_value": v_va}
    results["baselines"] = {"constant": c, "evaluate_scale": scale, "draw_rate_val": float(np.mean(z_va == 0)),
                            "mse": {name: {g: mse(p, z_va, m) for g, m in g_va} for name, p in base.items()}}
    results["baselines"]["train_mse"] = {"zero": mse(np.zeros_like(z_tr), z_tr), "constant": mse(np.full_like(z_tr, c), z_tr),
                                         "evaluate": mse(tr["baseline"], z_tr),
                                         "search_value": mse(v_tr, z_tr)}
    print(f"\nbaselines vs outcome (val; constant {c:+.3f}, evaluate x{scale:.2f} when scaled):")
    print(_table({k: v for k, v in results["baselines"]["mse"].items()}, [g for g, _ in g_va]))

    rng = np.random.default_rng(args.seed)
    probe = torch.from_numpy(rng.choice(n_tr, min(n_tr, 100_000), replace=False)).to(device)
    mean, std = _standardization(tr["nums0"], tr["nums1"])
    v = {k: torch.from_numpy(va[k]).to(device) for k in ("nums0", "cats0", "nums1", "cats1")}
    zt, vt = tr["outcome"], tr["y"]

    def run(hidden, w, tag):
        torch.manual_seed(args.seed)
        net = Net(vocab.sizes(), mean, std, hidden).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        target = (1 - w) * zt + w * vt

        def value(d, idx=slice(None)):
            return (net(d["nums0"][idx], d["cats0"][idx]) - net(d["nums1"][idx], d["cats1"][idx])) / 2

        def predict(d, n, idx=None):
            out = []
            with torch.no_grad():
                for i in range(0, n, 65536):
                    sl = slice(i, min(n, i + 65536)) if idx is None else idx[i:i + 65536]
                    out.append(value(d, sl).double().cpu().numpy())
            return np.concatenate(out)
        curve, best = [], None
        y_va = (1 - w) * z_va + w * v_va
        for epoch in range(args.epochs):
            net.train()
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, args.batch):
                idx = perm[i:i + args.batch]
                loss = torch.mean((value(tr, idx) - target[idx]) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            p_tr = predict(tr, len(probe), probe)
            p_va = predict(v, len(z_va))
            pz, pt = z_tr[probe.cpu().numpy()], target[probe].double().cpu().numpy()
            row = {"epoch": epoch, "train_vs_target": mse(p_tr, pt), "train_vs_outcome": mse(p_tr, pz),
                   "val_vs_outcome": mse(p_va, z_va), "val_vs_target": mse(p_va, y_va)}
            curve.append(row)
            if best is None or row["val_vs_outcome"] < best[0]["val_vs_outcome"]:
                best = (row, {g: mse(p_va, z_va, m) for g, m in g_va}, {g: mse(p_va, y_va, m) for g, m in g_va},
                        net.export())
        print(f"\n{tag} hidden={hidden} w={w}: per epoch train(target) / train(outcome) / val(outcome) / val(target)")
        print("  " + "  ".join(f"{r['train_vs_target']:.3f}/{r['train_vs_outcome']:.3f}/{r['val_vs_outcome']:.3f}/"
                               f"{r['val_vs_target']:.3f}" for r in curve))
        print(f"  best epoch {best[0]['epoch']}: by group, val vs outcome | val vs target")
        print(_table({"vs outcome": best[1], "vs target": best[2]}, [g for g, _ in g_va]))
        if args.save_models and tag == "weight":
            args.save_models.mkdir(parents=True, exist_ok=True)
            name = f"diag-h{hidden}-w{w:g}"
            ValueNet(best[3], vocab, {"name": name, "hidden": hidden, "epochs": args.epochs,
                                      "best_epoch": best[0]["epoch"], "value_weight": w,
                                      "train_games": tr["games"], "val_games": va["games"],
                                      "sources": results["dirs"], "created": time.time()}).save(
                args.save_models / f"{name}.npz")
        return {"hidden": hidden, "w": w, "curve": curve, "best_epoch": best[0]["epoch"],
                "best_val_vs_outcome": best[1], "best_val_vs_target": best[2]}

    def save():  # after every run, so a stopped job keeps what it finished
        if args.out:
            results["seconds"] = time.perf_counter() - t0
            args.out.write_text(json.dumps(results, indent=1))
    save()
    results["hidden"] = []
    for h in (int(h) for h in args.hidden.split(",") if h):
        results["hidden"].append(run(h, 0.0, "hidden"))
        save()
    results["weights"] = []
    for w in (float(x) for x in args.weights.split(",") if x):
        done = next((r for r in results["hidden"] if r["hidden"] == args.weights_hidden and r["w"] == w), None)
        results["weights"].append(done if done and not args.save_models else run(args.weights_hidden, w, "weight"))
        save()
    print("\nvalue-weight A/B, best held-out MSE vs outcome (all positions):")
    for r in results["weights"]:
        print(f"  w={r['w']:<5} {r['best_val_vs_outcome']['all']:.4f} (vs its blended target "
              f"{r['best_val_vs_target']['all']:.4f}, best epoch {r['best_epoch']})")
    save()
    if args.out:
        print(f"saved {args.out}")


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _table(rows: dict[str, dict[str, float]], cols: list[str]) -> str:
    head = " " * 18 + "".join(f"{c[:9]:>10}" for c in cols)
    return "\n".join([head] + [f"  {name:<16}" + "".join(f"{vals[c]:>10.4f}" for c in cols)
                               for name, vals in rows.items()])


if __name__ == "__main__":
    sys.exit(main())
