import math

import numpy as np
import pytest

from yomibench import gate_seq


def test_the_plans_are_preregistered_as_the_round4_plan_says():
    screen, confirm = gate_seq.two_stage(1234567, 7654321)
    assert (screen["turns"], screen["pairs"], screen["looks"]) == ("cheap", 50, [0.25, 0.5, 0.75, 1.0])
    assert screen["reject_z"] == [4.048, 2.862, 2.337, 2.024] and screen["min_pairs_per_pairing"] == 8
    assert screen["guard_upper"] == 0.5 and not screen["final"]
    assert (confirm["turns"], confirm["pairs"], confirm["looks"], confirm["final"]) == ("full", 25, [1.0], True)
    assert confirm["reject_z"] is None and confirm["pass_min"] == 0.5
    (core,) = gate_seq.full_core(42)
    assert (core["turns"], core["pairs"], core["reject_z"], core["final"]) == ("full", 50, screen["reject_z"], True)
    assert core["accept_z"] == [None, 2.862, 2.337, None]  # efficacy bound at looks 2 and 3 only
    assert screen["accept_z"] is None and confirm["accept_z"] is None  # the quick-turn screen stays reject-only
    with pytest.raises(ValueError):
        gate_seq.two_stage(5, 5)


def test_the_rejection_z_values_are_obrien_fleming_for_four_looks():
    """z_k = c / sqrt(t_k) with one c (2.024 for 4 looks, two-sided 0.05)."""
    cs = [z * math.sqrt(t) for z, t in zip(gate_seq.REJECT_Z, gate_seq.LOOKS)]
    assert max(cs) - min(cs) < 0.001 and cs[-1] == pytest.approx(2.024, abs=0.001)


def test_stages_refuse_bad_plans():
    for kw in ({"turns": "quick"}, {"looks": [0.5, 0.5, 1.0]}, {"looks": [0.5, 0.9]}, {"pairs": 0}, {"seed": 0},
               {"seed": 10 ** 7}, {"looks": [0.5, 1.0], "reject_z": [3.0]},
               {"looks": [0.5, 1.0], "accept_z": [None, 2.0]},  # early accept needs full search on both sides
               {"turns": "full", "looks": [0.5, 1.0], "accept_z": [2.0]},  # one per look
               {"turns": "full", "looks": [0.5, 1.0], "accept_z": [2.0, None]}):  # never at the first look
        with pytest.raises(ValueError):
            gate_seq.stage(**{"name": "s", "turns": "cheap", "seed": 1, "pairs": 10, **kw})


def test_looks_are_due_once_and_only_with_enough_pairs_in_every_pairing():
    st = gate_seq.two_stage(1, 2)[0]
    full = {"NN": 12, "NC": 12, "CN": 13, "CC": 13}
    assert gate_seq.look_due(st, 49, 200, [], full) is None
    assert gate_seq.look_due(st, 50, 200, [], full) == 0
    assert gate_seq.look_due(st, 50, 200, [], {**full, "CC": 7}) is None  # a pairing under 8 complete pairs
    assert gate_seq.look_due(st, 50, 200, [0], full) is None
    assert gate_seq.look_due(st, 160, 200, [0], full) == 2  # 50% skipped over: only 75% is evaluated
    assert gate_seq.look_due(st, 200, 200, [0, 2], full) == 3


def _core(scores, score=None, ci=None):
    st = gate_seq.pair_stats(scores)
    return {**st, "score": st["mean"] if score is None else score, "ci95": ci or [None, None]}


def _m(score, lo, hi, chars=("Ninja", "Ninja")):
    return {"characters": list(chars), "score": score, "ci95": [lo, hi]}


def test_look_verdicts():
    screen, confirm = gate_seq.two_stage(1, 2)
    bad = [0.0, 0.5, 0.0, 0.0] * 13  # mean 0.125: mean + 4.048 SE < 0.5 at the first look
    v = gate_seq.verdict(screen, 0, _core(bad), [])
    assert v["verdict"] == "reject" and "4.048" in v["reason"]
    meh = [0.0, 0.5, 0.5, 1.0, 0.0, 0.5] * 9  # mean 0.42: inside the first look's bound
    assert gate_seq.verdict(screen, 0, _core(meh), [])["verdict"] == "continue"
    great = [1.0] * 50
    assert gate_seq.verdict(screen, 0, _core(great), [])["verdict"] == "continue"  # never an early accept
    assert gate_seq.verdict(screen, 1, _core([]), [])["verdict"] == "continue"
    ok = [_m(0.6, 0.45, 0.75), _m(0.55, 0.4, 0.7, ("Cowboy", "Ninja"))]
    fine = [0.5, 0.5, 1.0, 0.0, 0.5, 0.75] * 34
    assert gate_seq.verdict(screen, 3, _core(fine, 0.52, [0.47, 0.57]), ok)["verdict"] == "pass"
    guard = ok + [_m(0.35, 0.2, 0.48, ("Cowboy", "Cowboy"))]
    v = gate_seq.verdict(screen, 3, _core(fine, 0.52, [0.47, 0.57]), guard)
    assert v["verdict"] == "reject" and "Cowboy:Cowboy" in v["reason"]
    assert gate_seq.verdict(screen, 3, _core(fine, 0.49, [0.44, 0.54]), ok)["verdict"] == "reject"
    assert gate_seq.verdict(confirm, 0, _core(fine, 0.50, [0.4, 0.6]), guard)["verdict"] == "pass"  # no guard
    assert gate_seq.verdict(confirm, 0, _core([]), [])["verdict"] == "inconclusive"


def test_the_anchor_check_is_pair_level_with_the_candidate_z():
    """#85: pass when mean - z_k SE > 0.5, fail when mean + z_k SE < 0.5, >= 8 pairs per pairing."""
    ready = {"NN": 8, "NC": 8, "CN": 8, "CC": 8}
    strong = [1.0] * 27 + [0.5] * 5  # g3 vs plain/ckpt1-like: far above 0.5
    assert gate_seq.anchor_verdict(0, strong, ready)["verdict"] == "pass"
    assert gate_seq.anchor_verdict(0, strong, {**ready, "CC": 7})["verdict"] == "open"  # not before 8 everywhere
    weak = [0.0] * 27 + [0.5] * 5
    assert gate_seq.anchor_verdict(0, weak, ready)["verdict"] == "fail"
    even = [0.0, 1.0, 0.5, 0.5] * 8
    assert gate_seq.anchor_verdict(3, even, ready)["verdict"] == "open"
    mid = [1.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.5, 1.0] * 4  # mean 0.6875: passes late, not at the first look
    v0, v3 = gate_seq.anchor_verdict(0, mid, ready), gate_seq.anchor_verdict(3, mid, ready)
    assert (v0["verdict"], v3["verdict"]) == ("open", "pass") and v0["z"] == 4.048 and v3["z"] == 2.024


def _run_stage(st, p, sims, rng, pairings=4):
    """Pass rates of `st` on simulated seat-swapped pairs (each game won with probability p) through
    gate_seq.look_due / gate_seq.verdict, normal-approximation 95% bounds standing in for
    pool_eval's paired bootstrap."""
    n = st["pairs"] * pairings
    scores = (rng.random((sims, n, 2)) < p).mean(axis=2)
    by_matchup = np.arange(n) % pairings  # pairings interleave, as a microshard split interleaves pairs

    def summary(x, idx):
        def ci(v):
            m = float(v.mean())
            se = float(v.std(ddof=1)) / math.sqrt(len(v)) if len(v) > 1 else 0.0
            return {"score": round(m, 4), "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]}
        return ({**gate_seq.pair_stats(list(x)), **ci(x)},
                [{"characters": [str(j), str(j)], **ci(x[idx == j])} for j in range(pairings)])
    passed = np.zeros(sims, bool)
    for i in range(sims):
        done = []
        for m in range(1, n + 1):
            per = {j: int((by_matchup[:m] == j).sum()) for j in range(pairings)}
            k = gate_seq.look_due(st, m, n, done, per)
            if k is None:
                continue
            done.append(k)
            core, matchups = summary(scores[i, :m], by_matchup[:m])
            v = gate_seq.verdict(st, k, core, matchups)["verdict"]
            if v != "continue":
                passed[i] = v == "pass"
                break
    return passed


def test_operating_characteristics():
    """A true 0.53 passes the confirmation about 80% (the plan says 79%); a candidate at 0.53 quick /
    0.45 full passes both stages about 8% (the plan says 5%: the confirmation's >= 0.50 counts a
    100-pair mean of exactly 0.500, which discrete pair scores hit often); a clearly worse candidate
    rarely survives the screen."""
    rng = np.random.default_rng(7)
    screen, confirm = gate_seq.two_stage(1, 2)
    sims = 1000
    assert _run_stage(confirm, 0.53, sims, rng).mean() == pytest.approx(0.80, abs=0.05)
    both = (_run_stage(screen, 0.53, sims, rng) & _run_stage(confirm, 0.45, sims, rng)).mean()
    assert both == pytest.approx(0.08, abs=0.035)
    assert _run_stage(screen, 0.40, 400, rng).mean() < 0.02


def _pairs(n_win, n_split, n_loss):
    return [1.0] * n_win + [0.5] * n_split + [0.0] * n_loss


def _look(st, k, scores, matchups=()):
    return gate_seq.verdict(st, k, {**gate_seq.pair_stats(scores), "score": None, "ci95": None}, list(matchups))


def test_full_core_accepts_early_at_looks_2_and_3_only_and_only_with_the_guard():
    (core,) = gate_seq.full_core(42)
    strong = _pairs(45, 0, 5)  # 50 pairs at 0.90: far past the bound already
    assert _look(core, 0, strong)["verdict"] == "continue"  # no early accept at look 1
    v = _look(core, 1, strong * 2)
    assert v["verdict"] == "pass" and "early accept" in v["reason"]
    losing = [{"characters": ["Ninja", "Cowboy"], "score": 0.3, "ci95": [0.1, 0.45]}]
    blocked = _look(core, 1, strong * 2, losing)
    assert blocked["verdict"] == "continue" and "guard fails" in blocked["reason"]  # the guard must hold
    (screen, _) = gate_seq.two_stage(1, 2)
    assert _look(screen, 1, strong * 2)["verdict"] == "continue"  # quick-turn screens stay reject-only
    old = {k: v for k, v in core.items() if k != "accept_z"}  # a job preregistered before the field
    assert _look(old, 1, strong * 2)["verdict"] == "continue"


def _early_accepts(st, mean, split, sims, rng):
    """Whether each simulated gate accepts at an interim look (2 or 3): pair scores 1 / 0.5 / 0 with
    P(split) = `split` and the rest set so the true mean is `mean`; looks at 25/50/75% of the 200
    pairs; the reject bound applies too (a rejected gate never accepts)."""
    n = st["pairs"] * 4
    win = mean - split / 2
    u = rng.random((sims, n))
    scores = np.where(u < win, 1.0, np.where(u < win + split, 0.5, 0.0))
    out = np.zeros(sims, bool)
    for i in range(sims):
        for k, t in enumerate(st["looks"][:-1]):
            v = _look(st, k, list(scores[i, :math.ceil(t * n)]))["verdict"]
            if v != "continue":
                out[i] = v == "pass"
                break
    return out


def test_a_true_062_full_core_gate_stops_at_look_2():
    """A 0.62 candidate with t12-like pair outcomes (45% won 2-0, 34% split, 21% lost 0-2): look 2's
    100 pairs have mean 0.62, SE 0.039, and 0.62 - 2.862 x 0.039 = 0.509 > 0.5, so it promotes there
    instead of playing all 400 games."""
    (core,) = gate_seq.full_core(42)
    look1, look2 = _pairs(22, 17, 11), _pairs(45, 34, 21)
    assert gate_seq.pair_stats(look2)["mean"] == pytest.approx(0.62)
    assert _look(core, 0, look1)["verdict"] == "continue"
    assert _look(core, 1, look2)["verdict"] == "pass"
    assert _early_accepts(core, 0.62, 0.34, 2000, np.random.default_rng(11)).mean() > 0.6  # most stop early


def test_the_false_accept_rate_at_a_true_05_stays_within_the_nominal_alpha():
    """At a true 0.5 the efficacy bound accepts early no more often than one-sided alpha 0.025 (the
    O'Brien-Fleming boundaries spend only part of it by look 3; the rest belongs to the final look)."""
    (core,) = gate_seq.full_core(42)
    assert _early_accepts(core, 0.5, 0.35, 6000, np.random.default_rng(5)).mean() <= 0.025


FOCUS = {"Cowboy:Cowboy": 60, "Cowboy:Ninja": 60, "Ninja:Cowboy": 60, "Ninja:Ninja": 32}
CHARS = ["Cowboy", "Ninja"]


def test_a_stage_without_per_pairing_counts_is_unchanged():
    """No pairs_by_pairing: the stage dicts are byte-identical to the ones preregistered before it."""
    import json

    (core,) = gate_seq.full_core(42)
    assert json.dumps(core) == json.dumps(
        {"name": "core", "turns": "full", "seed": 42, "pairs": 50, "looks": [0.25, 0.5, 0.75, 1.0],
         "reject_z": [4.048, 2.862, 2.337, 2.024], "accept_z": [None, 2.862, 2.337, None],
         "min_pairs_per_pairing": 8, "pass_min": 0.5, "guard_upper": 0.5, "final": True})
    screen, confirm = gate_seq.two_stage(1, 2)
    assert "pairs_by_pairing" not in screen and "pairs_by_pairing" not in confirm
    assert [gate_seq.scheduled_pairs(s, 4) for s in (core, screen, confirm)] == [200, 200, 100]


def test_a_focus_weighted_stage_preregisters_its_counts_and_looks_at_fractions_of_their_total():
    (core,) = gate_seq.full_core(42, pairs_by_pairing=dict(reversed(FOCUS.items())), characters=CHARS)
    (plain,) = gate_seq.full_core(42)
    assert core["pairs_by_pairing"] == FOCUS and list(core["pairs_by_pairing"]) == list(FOCUS)  # pairing order
    assert core["pairs"] == 212 and gate_seq.scheduled_pairs(core, 4) == 212
    assert {k: v for k, v in core.items() if k not in ("pairs", "pairs_by_pairing")} == {
        k: v for k, v in plain.items() if k != "pairs"}  # looks, z values, guard: all as before
    early = {"Cowboy:Cowboy": 16, "Cowboy:Ninja": 15, "Ninja:Cowboy": 16, "Ninja:Ninja": 8}
    assert gate_seq.look_due(core, 52, 212, [], early) is None  # 25% of 212 is 53 pairs
    assert gate_seq.look_due(core, 53, 212, [], early) == 0
    assert gate_seq.look_due(core, 53, 212, [], {**early, "Ninja:Ninja": 7}) is None  # the mirror under 8
    assert gate_seq.parse_pairs_by_pairing(
        "Cowboy:Cowboy=60,Cowboy:Ninja=60,Ninja:Cowboy=60,Ninja:Ninja=32") == FOCUS


def test_per_pairing_counts_refuse_a_missing_pairing_or_too_few_pairs_for_the_first_look():
    def core(counts, chars=CHARS, **kw):
        return gate_seq.full_core(42, pairs_by_pairing=counts, characters=chars, **kw)
    bad = [{k: v for k, v in FOCUS.items() if k != "Ninja:Ninja"},  # a pairing missing
           {**FOCUS, "Ninja:Ninja": 31},  # 31 x 25% < 8 complete pairs at the first look
           {**FOCUS, "Toy:Toy": 40},  # not a pairing of the gate's characters
           {**FOCUS, "Cowboy:Cowboy": 0}, {**FOCUS, "Cowboy:Cowboy": 60.0}]
    for counts in bad:
        with pytest.raises(ValueError):
            core(counts)
    with pytest.raises(ValueError):
        core(FOCUS, chars=None)  # the pairings are checked against the gate's characters
    with pytest.raises(ValueError, match="missing"):
        gate_seq.full_core(42, pairs_by_pairing={"Cowboy:Cowboy": 60}, characters=CHARS)
    with pytest.raises(ValueError):  # a stated total that isn't the counts' total
        gate_seq.stage("core", "full", 1, 200, gate_seq.LOOKS, pairs_by_pairing=FOCUS, characters=CHARS)
    assert gate_seq.stage("c", "full", 1, None, (1.0,), pairs_by_pairing={**FOCUS, "Ninja:Ninja": 8},
                          characters=CHARS)["pairs"] == 188  # one look at 100%: 8 pairs suffice
    for spec in ("Cowboy:Cowboy", "Cowboy=60", "Cowboy:Cowboy=x", "Cowboy:Cowboy=1,Cowboy:Cowboy=2"):
        with pytest.raises(ValueError):
            gate_seq.parse_pairs_by_pairing(spec)
