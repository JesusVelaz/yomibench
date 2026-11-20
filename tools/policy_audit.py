"""Offline audit of policy pruning on recorded decisions: how much value the pruned search's answer
loses against a full-width game.

    python tools/policy_audit.py --policy policy.npz --records DIR --manifest DIR.heldout.json \\
        --out audit.jsonl [--variants both:16,own:16] [--games-per-matchup 50] [--decisions-per-game 5] \\
        [--random-k 4] [--procs 2] [--seed 0]
    python tools/policy_audit.py ... --rerun OLD.jsonl [--threshold 0.02]    # only OLD's flagged decisions
    python tools/policy_audit.py --summarize audit-seed0.jsonl audit-seed1.jsonl   # pooled

Decisions come from held-out games (the manifest's `heldout` list; `--turns` picks full-search or
quick decisions): a seeded sample of
`--games-per-matchup` games per character matchup and `--decisions-per-game` searched decisions in
each (`--seed` picks both). Each game is replayed on the in-process sim (`new_match` with the
record's seed, then the recorded picks: expanded picks re-expanded from their `base` and `values`,
quick decisions' picks found by content, since a capped list's positions aren't the engine's),
and every replayed state must equal the record's (a game that diverges is skipped and logged).

At each sampled decision, all in private turn caches:
- an independent unpruned search (MatrixBot's double oracle with its own factor refinement; gate
  settings, no factor exploration, unless `--explore MARGIN:RANDOM:ALL` gives self-play's
  FactorExplore, which every search then uses);
- each pruned variant, `SCOPE:K` with K the policy's top K (plus `--random-k` random and
  Continue): `both` (self-play: both seats pruned, one search) and `own` (gates and play against
  people: one search per seat, that seat pruned, every opponent reply kept);
- with `--turns quick` (decisions self-play played capped, `search: cheap`), the quick-turn
  variants `cap:N` (playout.cap_candidates' fixed per-move rule, as self-play caps) and `pcap:N`
  (the policy picks the same number of candidates per seat: Continue and its top others), each
  searched as self-play searches a quick turn (MatrixBot.cheap: no factor refinement or
  exploration), both seats capped, scored where either list is longer than N;
- for each pruned answer, each seat's best reply against it over the candidates of both searches
  (bots.best_response_gains) and that reply's class (below), its value, kept counts, previews and
  seconds; and the control: the same best replies against the full search's own answer, over the
  same candidates.

**Value lost** (the criterion) for a pruned seat s: the full search's value from s's side minus what
s's pruned strategy guarantees against every known reply (its pruned value minus the opponent's
best-response gain). The secondary **missed improvement** is s's own best-response gain against the
opponent's pruned-game mix: what a better move of s's would earn if the opponent didn't adapt (it
overstates the cost; it is what this tool reported first). The **control** is what the full
search's own strategy loses against the same replies (refinement noise: either search may build a
reply the other lacks); **pruning loss** is value lost minus the control, the difference between the
two strategies' guarantees. A seat-decision is scored only when pruning is active there: for `own`,
the pruned seat has more than K + random-k + 1 base moves; for `both`, either seat does. For `own`,
only the pruned seat counts; for `both`, each seat. Values are in the value function's units
(`evaluate`, or `--value`).

A best reply's class, for the seat that plays it: `d` the pruned search had it; `a` a base move
outside that seat's kept set; `b` a factor variant (set 4 `expand`) of a kept base move that only the
full search built; `c` a factor variant of a base move outside the kept set. Value lost has two
parts: the **own** part, the full value minus the pruned value (s lacked a better move of its own;
s's own best reply against the opponent's pruned-game mix names it), and the **opponent** part, the
opponent's best-response gain against s's pruned strategy net of the control (the opponent's best
reply names it). Each loss is attributed to its larger part.

One JSON line per decision goes to `--out` as games finish (rerunning skips games already there).
`--summarize` (one or more files, pooled) reports pruning loss per variant and pairing (the
deciding seat's character v the opponent's) with by-game bootstrap 95% intervals (games resampled,
every scored seat-decision of a game kept together), the raw value lost, control and missed
improvement, the best-reply classes behind pruning losses over 0.02, and the worst cases.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

THRESHOLD = 0.02
DEFAULT_K = 16


QUICK = ("cap", "pcap")  # quick-turn variants: both seats, one capped search, no refinement


def variant_names(scope: str, k: int) -> list[str]:
    """Row keys of a variant: `both`/`own0`/`own1` at the default K (the first audit's names), else
    with `@K`; the quick-turn variants always carry `@N`."""
    if scope in QUICK:
        return [f"{scope}@{k}"]
    suffix = "" if k == DEFAULT_K else f"@{k}"
    return [f"both{suffix}"] if scope == "both" else [f"own0{suffix}", f"own1{suffix}"]


def parse_variants(text: str) -> list[tuple[str, int]]:
    out = []
    for item in text.split(","):
        scope, _, k = item.strip().partition(":")
        if scope not in ("both", "own") + QUICK or not k.isdigit():
            raise SystemExit(f"--variants takes SCOPE:K items with SCOPE both, own, cap or pcap: {item!r}")
        out.append((scope, int(k)))
    return out


def _both_seats(name: str) -> bool:
    return not name.startswith("own")


def value_lost(row: dict, name: str) -> dict[int, tuple[float, float]]:
    """{pruned seat: (value lost, missed improvement)} for one variant of one audit row."""
    v = row[name]
    seats = (0, 1) if _both_seats(name) else (int(name[3]),)
    out = {}
    for s in seats:
        sign = 1 if s == 0 else -1
        out[s] = (sign * row["full"]["value"] - (sign * v["value"] - v["gain"][1 - s]), v["gain"][s])
    return out


def pruning_loss(row: dict, name: str) -> dict[int, dict[str, float]]:
    """{scored seat: {loss, lost, control, missed}} for one variant of one audit row: value lost
    minus the control, for the seats where pruning is active (see the module docstring)."""
    v = row[name]
    k, r = v.get("k", DEFAULT_K), v.get("random_k", 4)
    above = v.get("active_above", k + r + 1)  # a list longer than this is pruned (capped)
    active = [n > above for n in row["n_base"]]
    out = {}
    for s, (lost, missed) in value_lost(row, name).items():
        if not (any(active) if _both_seats(name) else active[s]):
            continue
        control = v["control_gain"][1 - s] if "control_gain" in v else 0.0
        out[s] = {"loss": lost - control, "lost": lost, "control": control, "missed": missed,
                  "own": lost - v["gain"][1 - s], "opp": v["gain"][1 - s] - control}
    return out


def _variant_keys(row: dict) -> list[str]:
    return [k for k in row if re.fullmatch(r"(both|own[01]|cap|pcap)(@\d+)?", k)]


def _same(c, rec) -> bool:
    return c.action == rec["action"] and c.data == rec.get("data") and c.extra == rec.get("extra")


def _replay_id(engine, obs, seat, i, rec):
    """The engine id of a recorded pick (position `i` in the record's list for `seat`). A quick
    (playout-capped) decision records its capped list, so its positions aren't the engine's: a base
    pick is found by its content."""
    if rec.get("base") is None:
        cands = obs.candidates[seat]
        if i < len(cands) and _same(cands[i], rec):
            return cands[i].id
        for c in cands:
            if _same(c, rec):
                return c.id
        raise RuntimeError(f"seat {seat} pick {i}: {rec['action']} is not in the engine's list")
    got = engine.expand([{"seat": seat, "candidate": rec["base"], "extra_override": rec["values"]}])[0]
    if isinstance(got, str):
        raise RuntimeError(f"seat {seat} pick {i}: the engine refused its expansion ({got})")
    return got.id


def _classify(obs, cid, seat, by_id, pruned_ids, kept):
    """A best reply's class and what it is (see the module docstring). Engine ids are positions in
    each seat's own list, so `by_id` and `pruned_ids` are per seat."""
    c = by_id[seat][cid]
    base = c.id if c.base is None else c.base
    if cid in pruned_ids[seat]:
        cls = "d"
    elif c.base is None:
        cls = "a"
    else:
        cls = "b" if base in kept[seat] else "c"
    return {"seat": seat, "class": cls, "base_action": obs.candidates[seat][base].action,
            "variant": None if c.base is None else c.values, "base_kept": base in kept[seat]}


def audit_decision(engine, obs, net, value, variants, random_k, explore=None):
    """The independent search and each pruned variant at one decision (see the module docstring)."""
    from yomibench.bots import MatrixBot, PolicyPrune, _states_by_id, best_response_gains, turn_candidates

    def run(bot, seat=None, seed_previews=None, base=None):
        base = base or obs.candidates
        cache = {"base": base}
        if seed_previews:  # the full cache's base positions are engine ids; this cache's base may be a subset
            pos = [{c.id: k for k, c in enumerate(b)} for b in base]
            cache["previewed"] = {(pos[0][i], pos[1][j]): st for (i, j), st in seed_previews.items()
                                  if i in pos[0] and j in pos[1]}
        engine.turn_cache = cache
        p0, t = engine.stats["preview_pairs"], time.perf_counter()
        result = bot.search_turn(obs, engine, seat)
        out = {"previews": engine.stats["preview_pairs"] - p0, "seconds": round(time.perf_counter() - t, 3)}
        lists = turn_candidates(obs, engine, bot._key(seat))
        info = engine.turn_cache.get("oracle", {})
        return result, lists, _states_by_id(cache, turn_candidates(obs, engine)), cache.get("previewed", {}), out, info

    plain = MatrixBot(search="oracle", value=value, explore=explore)
    full, full_lists, memo, full_prev, full_out, _ = run(plain)
    row = {"n_base": [len(c) for c in obs.candidates], "full": {**full_out, "value": round(full[2], 6),
                                                               "candidates": [len(c) for c in full_lists]}}
    key = f"{engine.match_seed}:{obs.turn}:{obs.state.get('tick')}"
    for scope, k in variants:
        if scope in QUICK:
            row.update(_quick_variant(scope, k, obs, engine, net, value, run, full, full_lists, full_prev, memo))
            continue
        for name, seat in zip(variant_names(scope, k), (None,) if scope == "both" else (0, 1)):
            prune = PolicyPrune(net, k, random_k, audit=0.0, scope=scope)
            bot = MatrixBot(search="oracle", value=value, explore=explore, policy=prune)
            res, lists, m, _, out, info = run(bot, seat, full_prev)
            memo.update(m)
            g = best_response_gains(engine, value, lists, full_lists, res, memo)
            ctrl = best_response_gains(engine, value, full_lists, lists, full, memo)  # same replies, full answer
            kept, _ = prune.keep(obs, [list(c) for c in obs.candidates], key, prune.prune_seats(seat))
            by_id = [{c.id: c for c in list(full_lists[s]) + list(lists[s])} for s in (0, 1)]
            # What the pruned search could play: its kept base moves and its expansions (all of a
            # kept move). Its view lists also hold every base move, kept or not (turn_candidates).
            pruned_ids = [{c.id for c in lists[s] if c.base is not None or c.id in set(kept[s])} for s in (0, 1)]
            row[name] = {**out, "gain": [round(x, 6) for x in g["gain"]], "value": round(res[2], 6), "k": k,
                         "random_k": random_k, "control_gain": [round(x, 6) for x in ctrl["gain"]],
                         "kept": info.get("policy", {}).get("kept"), "universe": g["candidates"],
                         "best": [_classify(obs, g["best"][s], s, by_id, pruned_ids, [set(x) for x in kept])
                                  for s in (0, 1)]}
    engine.turn_cache = {}
    return row


def _quick_variant(scope, n, obs, engine, net, value, run, full, full_lists, full_prev, memo):
    """A quick-turn search (as self-play plays a playout-capped decision: MatrixBot.cheap, no factor
    refinement or exploration) over a capped list per seat: `cap`, playout.cap_candidates(N);
    `pcap`, the same number of candidates chosen by the policy (Continue and its top others, in
    list order)."""
    from yomibench.bots import MatrixBot, best_response_gains
    from yomibench.playout import cap_candidates

    capped = [cap_candidates(obs.candidates[s], n) for s in (0, 1)]
    if scope == "cap":
        base = capped
    else:
        base = []
        for s in (0, 1):
            cands, m = obs.candidates[s], len(capped[s])
            if m >= len(cands):
                base.append(list(cands))
                continue
            scores = net.scores(obs.state, s, cands)
            top = set(sorted((i for i in range(1, len(cands))), key=lambda i: (-scores[i], i))[:m - 1])
            base.append([c for i, c in enumerate(cands) if i == 0 or i in top])
    bot = MatrixBot(search="oracle", value=value).cheap()
    res, lists, m_, _, out, _ = run(bot, None, full_prev, base)
    memo.update(m_)
    g = best_response_gains(engine, value, lists, full_lists, res, memo)
    ctrl = best_response_gains(engine, value, full_lists, lists, full, memo)
    by_id = [{c.id: c for c in list(full_lists[s]) + list(lists[s])} for s in (0, 1)]
    kept = [{c.id for c in b} for b in base]
    return {f"{scope}@{n}": {**out, "gain": [round(x, 6) for x in g["gain"]], "value": round(res[2], 6), "k": n,
                             "active_above": n, "control_gain": [round(x, 6) for x in ctrl["gain"]],
                             "kept": [len(b) for b in base], "universe": g["candidates"],
                             "best": [_classify(obs, g["best"][s], s, by_id, kept, kept) for s in (0, 1)]}}


def audit_game(job):
    path, n_decisions, turns_wanted, policy_path, value_path, variants, random_k, seed, explore_cfg, quick = job
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    from yomibench.engine import connect
    from yomibench.policy import PolicyNet
    from yomibench.search import evaluate

    net = PolicyNet.load(policy_path)
    explore = None
    if explore_cfg:
        from yomibench.bots import FactorExplore
        explore = FactorExplore(margin=explore_cfg[0], random_k=int(explore_cfg[1]), all_prob=explore_cfg[2])
    value = evaluate
    if value_path:
        from yomibench.valuenet import ValueNet
        value = ValueNet.load(value_path)
    lines = [json.loads(x) for x in gzip.open(path, "rt", encoding="utf-8")]
    header, turns = lines[0], [d for d in lines[1:] if "turn" in d]
    if turns_wanted is not None:
        targets = set(turns_wanted)
    else:
        eligible = [d["turn"] for d in turns if d.get("strategies") and (d.get("search") == "cheap") == quick
                    and all(sum("base" not in c for c in s) >= 2 for s in d["candidates"])]
        targets = set(random.Random(f"{seed}:{Path(path).name}").sample(eligible, min(n_decisions, len(eligible))))
    engine = connect("sim", 4)
    chars = header["characters"]
    obs = engine.new_match(chars[0], chars[1], header["seed"])
    rows = []
    try:
        for d in turns:
            if not targets - {r["turn"] for r in rows}:
                break
            if obs.turn != d["turn"] or obs.state != d["state"]:
                raise RuntimeError(f"turn {d['turn']}: replay diverged from the record")
            if d["turn"] in targets:
                t = time.perf_counter()
                row = audit_decision(engine, obs, net, value, variants, random_k, explore)
                rows.append({"game": Path(path).name, "seed": header["seed"], "characters": chars, "turn": d["turn"],
                             "seconds": round(time.perf_counter() - t, 2), "sample_seed": seed,
                             "value_fn": Path(value_path).stem if value_path else "evaluate",
                             "explore": list(explore_cfg) if explore_cfg else None, **row})
            ids = [_replay_id(engine, obs, s, d["picks"][s], d["candidates"][s][d["picks"][s]]) for s in (0, 1)]
            obs = engine.step(*ids)
    except Exception as e:  # a diverging or failing game is skipped, not fatal to the run
        return {"game": Path(path).name, "error": f"{type(e).__name__}: {e}", "rows": rows}
    finally:
        getattr(engine, "close", lambda: None)()
    return {"game": Path(path).name, "rows": rows}


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))]


def _boot(by_game: dict[str, list[float]], stat, draws: int = 1000, seed: int = 0) -> tuple[float, float]:
    """A by-game bootstrap 95% interval of `stat` over the pooled values: games are resampled with
    replacement, each with all its scored seat-decisions."""
    keys = list(by_game)
    rng = random.Random(seed)
    stats = sorted(stat([x for g in (rng.choice(keys) for _ in keys) for x in by_game[g]]) for _ in range(draws))
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def _share(t):
    return lambda xs: sum(x > t for x in xs) / len(xs)


def summarize(paths: list[Path]) -> str:
    rows = [json.loads(x) for path in paths for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    errors = [r for r in rows if "error" in r]
    rows = [r for r in rows if "full" in r]
    scored: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    considered: dict[str, int] = defaultdict(int)  # seat-decisions looked at, scored or not
    classes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    moves: dict[str, Counter] = defaultdict(Counter)
    worst: dict[str, list] = defaultdict(list)
    previews: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        c = r["characters"]
        previews["full"].append(r["full"]["previews"])
        for name in _variant_keys(r):
            group = re.sub(r"own[01]", "own", name)
            previews[group].append(r[name]["previews"])
            considered[group] += 2 if _both_seats(name) else 1
            for s, x in pruning_loss(r, name).items():
                pairing = f"{c[s]} v {c[1 - s]}"
                scored[(group, pairing)][r["game"]].append(x)
                best = r[name].get("best")
                part = "own" if x["own"] >= x["opp"] else "opp"
                b = (best[s] if part == "own" else best[1 - s]) if best else None
                if x["loss"] > THRESHOLD and b:
                    classes[(group, pairing)][(part, b["class"])] += 1
                    moves[group][(part, b["base_action"], b["class"])] += 1
                worst[group].append((x["loss"], r["game"], r["turn"], pairing, s, x, b, part))
    out = [f"{len(rows)} decisions from {len({r['game'] for r in rows})} held-out games"
           + (f" ({len(errors)} games skipped: {sorted({e['error'][:60] for e in errors})})" if errors else ""),
           "", "Pruning loss = value lost minus the control (the full search's own strategy against the same "
           "replies); only seat-decisions where pruning is active. [ ] = by-game bootstrap 95% interval.", "",
           "| variant | pairing | games | seat-decisions | median | p99 [CI] | max | share > 0.02 [CI] "
           "| share > 0.1 [CI] | raw value lost p99 | control p99 | missed impr. p99 |",
           "|---|---|---:|---:|---:|---|---:|---|---|---:|---:|---:|"]
    order = sorted({g for g, _ in scored}, key=lambda g: (g.split("@")[0], int(g.split("@")[1]) if "@" in g else DEFAULT_K))
    for group in order:
        pairings = sorted(p for g, p in scored if g == group)
        for pairing in pairings + ["all"]:
            if pairing == "all":
                by_game: dict[str, list] = defaultdict(list)
                for (g, _), games in scored.items():
                    if g == group:
                        for k, v in games.items():
                            by_game[k] += v
            else:
                by_game = scored[(group, pairing)]
            xs = [x for v in by_game.values() for x in v]
            loss = {k: [x["loss"] for x in v] for k, v in by_game.items()}
            flat = [x["loss"] for x in xs]
            ci = {name: _boot(loss, f) for name, f in (("p99", lambda v: _pct(v, 0.99)), ("s2", _share(THRESHOLD)),
                                                       ("s10", _share(0.1)))}
            out.append(
                f"| {group} | {pairing} | {len(by_game)} | {len(xs)} | {statistics.median(flat):.4f} | "
                f"{_pct(flat, 0.99):.4f} [{ci['p99'][0]:.4f}, {ci['p99'][1]:.4f}] | {max(flat):.4f} | "
                f"{_share(THRESHOLD)(flat):.3f} [{ci['s2'][0]:.3f}, {ci['s2'][1]:.3f}] | "
                f"{_share(0.1)(flat):.4f} [{ci['s10'][0]:.4f}, {ci['s10'][1]:.4f}] | "
                f"{_pct([x['lost'] for x in xs], 0.99):.4f} | {_pct([x['control'] for x in xs], 0.99):.4f} | "
                f"{_pct([x['missed'] for x in xs], 0.99):.4f} |")
    n_scored = {g: sum(len(v) for (gg, _), games in scored.items() if gg == g for v in games.values()) for g in order}
    out.append("\nShare of seat-decisions scored (pruning active): "
               + ", ".join(f"{g} {n_scored[g] / max(considered[g], 1):.2f}" for g in order))
    out.append("Previews per decision (mean; `own` is per pruned seat's search): "
               + ", ".join(f"{k} {statistics.mean(v):.0f}" for k, v in previews.items()))
    if classes:
        out += ["\nPruning losses > 0.02 by their larger part and that part's best reply class "
                "(a: base outside kept, b: variant of kept base, c: variant of pruned base, d: pruned search had it):",
                "| variant | pairing | own move missing: a | b | c | d | opponent reply: a | b | c | d |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (group, pairing), cnt in sorted(classes.items()):
            out.append(f"| {group} | {pairing} | " + " | ".join(str(cnt[(p, k)]) for p in ("own", "opp") for k in "abcd") + " |")
        for group, cnt in sorted(moves.items()):
            out.append(f"\nTop base moves behind them ({group}): "
                       + ", ".join(f"{a} ({p} {c}) {n}" for (p, a, c), n in cnt.most_common(10)))
    for group in order:
        top = sorted(worst[group], key=lambda w: -w[0])[:8]
        out += [f"\nWorst cases ({group}):", "| game | turn | pairing | seat | pruning loss | own part | opponent part "
                "| control | attributed best reply |", "|---|---:|---|---:|---:|---:|---:|---:|---|"]
        for loss, game, turn, pairing, s, x, b, part in top:
            reply = (f"{b['base_action']} ({b['class']}; base {'kept' if b.get('base_kept') else 'pruned'}"
                     f"{'; variant ' + json.dumps(b['variant']) if b.get('variant') else ''})") if b else "?"
            reply = f"{'own' if part == 'own' else 'opponent'}: {reply}"
            out.append(f"| {game[:20]} | {turn} | {pairing} | {s} | {loss:.4f} | {x['own']:.4f} | {x['opp']:.4f} | "
                       f"{x['control']:.4f} | {reply} |")
    return "\n".join(out)


def flagged(path: Path, threshold: float) -> dict[str, list[int]]:
    """{game: [turn, ...]} for an audit's decisions where some variant's pruned seat lost more than
    `threshold`."""
    out: dict[str, set[int]] = defaultdict(set)
    for x in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(x) if x.strip() else {}
        if "full" in r and any(lost > threshold for name in _variant_keys(r) for lost, _ in value_lost(r, name).values()):
            out[r["game"]].add(r["turn"])
    return {g: sorted(t) for g, t in out.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", type=Path, nargs="+", default=None, metavar="AUDIT.jsonl")
    ap.add_argument("--policy", type=Path)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--manifest", type=Path, help="the records' held-out manifest (yomibench.heldout)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--value", type=Path, default=None, help="value net for the searches (default: evaluate)")
    ap.add_argument("--variants", default=f"both:{DEFAULT_K},own:{DEFAULT_K}",
                    help="pruned searches to audit, SCOPE:K items (default both:16,own:16)")
    ap.add_argument("--rerun", type=Path, default=None, metavar="OLD.jsonl",
                    help="audit only the decisions where OLD has a pruned seat losing more than --threshold")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--games-per-matchup", type=int, default=50)
    ap.add_argument("--decisions-per-game", type=int, default=5)
    ap.add_argument("--random-k", type=int, default=4)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0, help="which games and decisions are sampled")
    ap.add_argument("--turns", choices=("full", "quick"), default="full",
                    help="sample full-search decisions (default) or quick (playout-capped, `search: cheap`) ones")
    ap.add_argument("--explore", default=None, metavar="MARGIN:RANDOM:ALL",
                    help="self-play's factor exploration for every search (FactorExplore), e.g. 0.02:2:0.005")
    args = ap.parse_args(argv)
    if args.summarize:
        print(summarize(args.summarize))
        return 0
    if not (args.policy and args.records and args.out and (args.manifest or args.rerun)):
        ap.error("--policy, --records, --out and --manifest (or --rerun) are needed (or --summarize)")
    variants = parse_variants(args.variants)
    explore = tuple(float(x) for x in args.explore.split(":")) if args.explore else None
    if args.rerun:
        wanted = flagged(args.rerun, args.threshold)
        order = [args.records / g for g in sorted(wanted)]
        print(f"{sum(map(len, wanted.values()))} flagged decisions in {len(wanted)} games of {args.rerun}", flush=True)
    else:
        wanted = None
        held = set(json.loads(args.manifest.read_text(encoding="utf-8"))["heldout"])
        by_matchup: dict[tuple, list[Path]] = defaultdict(list)
        for p in sorted(args.records.glob("g*.jsonl.gz")):
            if p.name in held:
                with gzip.open(p, "rt", encoding="utf-8") as f:
                    by_matchup[tuple(json.loads(f.readline())["characters"])].append(p)
        picked = {m: random.Random(f"{args.seed}:{m}").sample(ps, min(args.games_per_matchup, len(ps)))
                  for m, ps in sorted(by_matchup.items())}
        order = [p for rnd in zip(*picked.values()) for p in rnd]  # interleaved: partial runs cover every matchup
        order += [p for ps in picked.values() for p in ps if p not in set(order)]
        print(f"{len(order)} games picked ({', '.join(f'{m[0]}:{m[1]} {len(v)}' for m, v in picked.items())})", flush=True)
    done = set()
    if args.out.exists():
        done = {json.loads(x)["game"] for x in args.out.read_text(encoding="utf-8").splitlines() if x.strip()}
    jobs = [(str(p), args.decisions_per_game, wanted[p.name] if wanted else None, str(args.policy),
             str(args.value) if args.value else None, variants, args.random_k, args.seed, explore,
             args.turns == "quick")
            for p in order if p.name not in done]
    print(f"{len(done)} games done, {len(jobs)} to go; variants {variants}", flush=True)
    t0, n = time.time(), 0
    with ProcessPoolExecutor(args.procs) as pool, args.out.open("a", encoding="utf-8") as out:
        futures = [pool.submit(audit_game, j) for j in jobs]
        for f in as_completed(futures):
            r = f.result()
            for row in r["rows"]:
                out.write(json.dumps(row) + "\n")
            if "error" in r:
                out.write(json.dumps({"game": r["game"], "error": r["error"]}) + "\n")
            out.flush()
            n += len(r["rows"])
            print(f"{r['game']}: {len(r['rows'])} decisions{' (' + r['error'] + ')' if 'error' in r else ''}; "
                  f"{n} so far, {(time.time() - t0) / 60:.0f} min", flush=True)
    print(summarize([args.out]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
