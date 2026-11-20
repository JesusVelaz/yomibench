import itertools
import math
import random

import numpy as np
import pytest

from yomibench import policy
from yomibench.bots import MatrixBot, PolicyPrune, turn_candidates
from yomibench.engine import connect
from yomibench.selfplay import read_slim, run_selfplay


def _cand(action="Jab", di=(0, 0), base=None, data=None, **values):
    c = {"action": action, "data": data, "extra": {"DI": {"x": di[0], "y": di[1]}, "feint": False, "reverse": False},
         "factors": {"di": 17, "feint": True}, "values": {"di": 0, "feint": False, **values}}
    if base is not None:
        c["base"] = base
    return c


def test_expansions_fold_onto_the_candidate_they_override():
    cands = [_cand("Continue"), _cand("Jab"), _cand("Grab"), _cand("Jab", di=(1, 0), base=1, di_=3),
             _cand("Grab", base=2, feint=True)]
    base, probs = policy.fold_to_base(cands, [0.1, 0.2, 0.0, 0.5, 0.2])
    assert base == [0, 1, 2]
    assert probs == pytest.approx([0.1, 0.7, 0.2])


def test_candidate_numbers_mirror_with_the_view():
    c = _cand("Roll", di=(3, -2), data={"x": 1, "y": 0})
    _, _, nums = policy.encode_candidate(c)
    nums = np.array([nums], dtype=np.float32)
    flipped = policy.flip_columns(nums, -1.0)
    x = list(policy.X_COLUMNS)
    assert nums[0, policy.CAND_NUMS.index("di_x")] == 3 and nums[0, policy.CAND_NUMS.index("aim_x")] == 1
    assert np.array_equal(flipped[0, x], -nums[0, x])
    others = [k for k in range(policy.N_CAND) if k not in x]
    assert np.array_equal(flipped[0, others], nums[0, others])
    # Per-row flips over a padded (batch, list, numbers) block.
    block = np.stack([nums, nums])
    out = policy.flip_columns(block, np.array([1.0, -1.0]))
    assert np.array_equal(out[0], nums) and np.array_equal(out[1], flipped)


def test_candidates_differing_only_in_a_factor_encode_differently():
    a = policy.encode_candidate(_cand("Jab"))
    b = policy.encode_candidate(_cand("Jab", feint=True))
    c = policy.encode_candidate({**_cand("Jab"), "extra": {"DI": {"x": 0, "y": 0}, "feint": True}})
    assert len({a, b, c}) == 3


def test_topk_metrics():
    target = np.array([0.5, 0.0, 0.3, 0.195, 0.005])  # support: 0, 2, 3
    recall, mass, cover = policy.topk_metrics([2, 1, 0, 3, 4], target, 2)
    assert (recall, mass, cover) == (pytest.approx(1 / 3), pytest.approx(0.3), 0.0)
    assert policy.topk_metrics([2, 0, 3, 1, 4], target, 3) == (1.0, pytest.approx(0.995), 1.0)


def test_random_metrics_match_enumeration():
    target = np.array([0.4, 0.3, 0.0, 0.3, 0.0, 0.0])
    for k in (1, 2, 3, 4, 8):
        draws = list(itertools.combinations(range(len(target)), min(k, len(target))))
        expect = np.mean([policy.topk_metrics(list(d), target, k) for d in draws], axis=0)
        assert policy.random_metrics(target, k) == pytest.approx(tuple(expect))


@pytest.fixture(scope="module")
def toy_records(tmp_path_factory):
    out = tmp_path_factory.mktemp("records")
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50), [("Toy", "Toy")], range(6), out)
    return out


def test_game_samples_read_targets_from_records(toy_records):
    path = sorted(toy_records.glob("g*.jsonl.gz"))[0]
    game = policy.game_samples(path)
    _, _, turns = read_slim(path, policy=True)
    searched = sum(bool(t.strategies) for t in turns)
    assert game["samples"] and len(game["samples"]) <= 2 * searched
    for s in game["samples"]:
        assert sum(s["target"]) == pytest.approx(1) and len(s["target"]) == len(s["cands"]) >= 2
    capped = policy.game_samples(path, cap=2)
    assert len({s["turn"] for s in capped["samples"]}) <= 2
    assert read_slim(path)[2][0].strategies is None  # the value net's loader stays slim


def test_playout_capped_decisions_are_not_policy_targets(tmp_path):
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50), [("Toy", "Toy")], range(1), tmp_path,
                 playout_cap=(1.0, 3))
    (path,) = tmp_path.glob("g*.jsonl.gz")
    assert any(t.strategies for t in read_slim(path, policy=True)[2])
    assert policy.game_samples(path)["samples"] == []


def test_training_end_to_end_and_numpy_inference_matches_torch(toy_records, tmp_path):
    torch = pytest.importorskip("torch")
    from yomibench import train_policy

    out = tmp_path / "p.npz"
    train_policy.main([str(toy_records), "--out", str(out), "--heldout-root", str(tmp_path / "ho"), "--epochs", "2",
                       "--workers", "1", "--device", "cpu", "--dev-fraction", "0.2", "--report", str(tmp_path / "r.md"),
                       "--cache", str(tmp_path / "c.npz")])
    net = policy.PolicyNet.load(out)
    check = net.meta["offline_check"]["all"]
    assert check["decisions"] > 0 and set(check["k"]) == {"4", "8", "16"}
    assert (tmp_path / "r.md").read_text().count("| all |") == 6  # against both targets
    # The numpy forward is the torch forward's score.
    data = train_policy.load_cache(tmp_path / "c.npz")
    stats = {k: net.p[k] for k in ("mean", "std", "cmean", "cstd")}
    model = train_policy.Net(data["vocab"].sizes(), {c: len(data["cvocab"][c]) + 1 for c in policy.CAND_CATEGORIES},
                             stats)
    p = model.export()
    split = data["test"]
    b = train_policy.batch(data, split, np.arange(min(3, len(split["flip"]))))
    with torch.no_grad():
        score, _ = model(*(torch.from_numpy(b[k]) for k in ("nums", "cats", "cnums", "cact", "cdkey")))
    for row in range(len(b["nums"])):
        n = int(b["mask"][row].sum())
        ref = policy.forward(p, b["nums"][row], b["cats"][row], b["cnums"][row, :n], b["cact"][row, :n],
                             b["cdkey"][row, :n])
        assert np.allclose(ref, score[row, :n].numpy(), atol=1e-4)
    # And PolicyNet.scores encodes a recorded decision the way training did.
    path = sorted(toy_records.glob("g*.jsonl.gz"))[0]
    _, _, turns = read_slim(path, policy=True)
    t = next(t for t in turns if t.strategies)
    assert len(net.scores(t.state, 1, t.candidates[1])) == len(t.candidates[1])


class _ReverseNet:
    """Ranks later candidates first."""
    meta = {"name": "reverse"}

    def scores(self, state, seat, cands):
        return np.arange(len(cands), dtype=np.float32)


def _search(bot, seed=3, turns=4, engine=None):
    engine = engine or connect("toy")
    obs = engine.new_match("Toy", "Toy", seed)
    engine.match_seed = seed
    rng = random.Random(seed)
    out = []
    for _ in range(turns):
        row, col, value = bot.search_turn(obs, engine)
        out.append((row, col, value, dict(engine.turn_cache.get("oracle", {})), turn_candidates(obs, engine)))
        picks = [rng.choices(range(len(s)), weights=s)[0] for s in (row, col)]
        obs = engine.step(*(turn_candidates(obs, engine)[k][i].id for k, i in enumerate(picks)))
        if obs.done:
            break
    return out


def _prune(k=2, random_k=1, audit=0.0, scope="both", net=None):
    return PolicyPrune(net or _ReverseNet(), k=k, random_k=random_k, audit=audit, scope=scope)


def test_policy_pruning_is_off_by_default_and_wide_k_changes_nothing():
    assert "policy" not in MatrixBot(search="oracle").config()
    plain = _search(MatrixBot(search="oracle", iterations=200))
    wide = _search(MatrixBot(search="oracle", iterations=200, policy=_prune(k=50)))
    for (r0, c0, v0, *_), (r1, c1, v1, *_) in zip(plain, wide):
        assert (r0, c0, v0) == (r1, c1, v1)


def test_both_scope_keeps_each_seats_top_k():
    bot = MatrixBot(search="oracle", iterations=200, policy=_prune())
    for row, col, value, info, cands in _search(bot):
        for s, strat in enumerate((row, col)):
            n = len(cands[s])
            assert info["policy"]["kept"][s] == 4 and info["policy"]["of"][s] == n
            top = {0, n - 1, n - 2}
            assert sum(p > 0 for i, p in enumerate(strat) if i not in top) <= 1  # besides, only the random one
        assert "audited" not in info["policy"]


def test_own_scope_prunes_only_the_deciding_seat():
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 3)
    engine.match_seed = 3
    bot = MatrixBot(search="oracle", iterations=200, policy=_prune(scope="own"))
    with pytest.raises(ValueError, match="own-scope"):
        bot.search_turn(obs, engine)  # self-play's seatless search needs scope "both"
    for seat in (0, 1):
        bot.search(obs, seat, engine)
        info = engine.turn_cache["oracle"]["policy"]
        n = [len(c) for c in obs.candidates]
        assert info["kept"][seat] == 4 and info["kept"][1 - seat] == n[1 - seat]  # every opponent reply kept


def test_the_audit_is_an_independent_search_that_leaves_the_shared_lists_alone():
    from test_factors import _engine  # FactorToy: set-4 style expansions

    def run(audit, k=2):
        engine, _ = _engine()
        return _search(MatrixBot(search="oracle", iterations=300, policy=_prune(k=k, audit=audit)), seed=1,
                       turns=6, engine=engine)

    quiet, audited = run(0.0), run(1.0)
    for (r0, c0, v0, i0, l0), (r1, c1, v1, i1, l1) in zip(quiet, audited):
        assert (r0, c0, v0) == (r1, c1, v1)  # the audit only measures: the pruned answer is played
        assert [[c.id for c in s] for s in l0] == [[c.id for c in s] for s in l1]  # nor touches the record's lists
        p = i1["policy"]
        assert p["audited"] and len(p["audit_gain"]) == 2 and min(p["audit_gain"]) >= -1e-6
        assert {"audit_value", "audit_full_value", "audit_candidates"} <= set(p)
    # Without pruning the audited answer is the full search's own: nothing gains against it.
    for *_, info, _ in run(1.0, k=50):
        assert max(info["policy"]["audit_gain"]) <= 2e-3


@pytest.fixture(scope="module")
def toy_policy(toy_records, tmp_path_factory):
    pytest.importorskip("torch")
    from yomibench import train_policy

    out = tmp_path_factory.mktemp("policy") / "toy-policy.npz"
    train_policy.main([str(toy_records), "--out", str(out), "--heldout-root", str(out.parent / "ho"),
                       "--epochs", "1", "--workers", "1", "--device", "cpu", "--dev-fraction", "0.2"])
    return out


def test_prune_spec_parses_and_leaves_other_specs_alone(toy_policy, tmp_path):
    from yomibench.bots import make_bot

    bot = make_bot(f"matrix-do@prune=16:4:0.05:{toy_policy}")
    assert (bot.policy.k, bot.policy.random_k, bot.policy.audit, bot.policy.scope) == (16, 4, 0.05, "own")
    assert bot.name == "matrix-do@prune=16:4:0.05:own:toy-policy"
    assert bot.config()["policy"] == {"net": "toy-policy", "k": 16, "random_k": 4, "audit": 0.05, "scope": "own"}
    assert make_bot(f"matrix-do@prune=8:0:0:both:{toy_policy}").policy.scope == "both"
    odd = tmp_path / "C:" / "models"  # everything after the scope is the path, colons and all
    odd.mkdir(parents=True)
    (odd / "p.npz").write_bytes(toy_policy.read_bytes())
    assert make_bot(f"matrix-do@cheap=0.5:8@prune=8:2:0.1:own:{odd / 'p.npz'}").policy.k == 8
    for bad in ("matrix-do@prune=16:4", "matrix-do@prune=16:4:0.05:own:", f"matrix@prune=16:4:0:{toy_policy}",
                f"matrix-do@prune=16:4:2:{toy_policy}"):
        with pytest.raises(ValueError):
            make_bot(bad)
    for spec in ("matrix-do", "matrix", "matrix-do@cheap=0.5:8", "greedy"):  # unchanged without @prune
        b = make_bot(spec)
        assert b.name == spec and "policy" not in getattr(b, "config", dict)()


def test_an_arena_game_prunes_only_the_pruned_bots_own_turns(toy_policy):
    from test_factors import _engine

    from yomibench.arena import play_game
    from yomibench.bots import make_bot

    spec = f"matrix-do@prune=2:1:0.5:{toy_policy}"
    pruned, plain = make_bot(spec), MatrixBot(search="oracle", iterations=300)
    # One game on two engines, the pruned bot searching first on one and last on the other: the
    # full bot's every answer is the same, so nothing of the pruned search (its kept lists,
    # expansions or audits) reaches it, and the pruned bot keeps every reply of its opponent.
    games = []
    for first in ("pruned", "plain"):
        engine, _ = _engine()
        games.append({"engine": engine, "obs": engine.new_match("Toy", "Toy", 4), "first": first,
                      "rngs": [random.Random(0), random.Random(1)]})
    for _ in range(12):
        if games[0]["obs"].done:
            break
        answers = []
        for g in games:
            e, obs = g["engine"], g["obs"]
            order = [0, 1] if g["first"] == "pruned" else [1, 0]
            for seat in order:
                (pruned, plain)[seat].search(obs, seat, e)
            answers.append(plain.search(obs, 1, e))
            g["picks"] = [pruned.choose(obs, 0, e, g["rngs"][0]), plain.choose(obs, 1, e, g["rngs"][1])]
        assert answers[0] == answers[1]
        for g in games:
            g["obs"] = g["engine"].step(*g["picks"])
        assert games[0]["obs"].state == games[1]["obs"].state
    engine, _ = _engine()
    obs = engine.new_match("Toy", "Toy", 4)
    pruned.search(obs, 0, engine)
    info = engine.turn_cache["oracle"]["policy"]
    assert info["kept"][0] < info["of"][0] and info["kept"][1] == info["of"][1]
    plain.search(obs, 1, engine)
    assert "policy" not in engine.turn_cache["oracle"]
    # And it plays through the arena from either seat.
    for bots in ((make_bot(spec), make_bot("matrix-do")), (make_bot("matrix-do"), make_bot(spec))):
        engine, _ = _engine()
        assert play_game(engine, bots, ("Toy", "Toy"), 2, max_turns=10).turns > 0


def test_random_starts_are_what_the_search_started_from(tmp_path):
    import gzip
    import json

    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(search="oracle", iterations=50), [("Toy", "Toy")],
                 range(2), tmp_path, full_previews_fraction=1.0)
    checked = 0
    for path in tmp_path.glob("g*.jsonl.gz"):
        lines = [json.loads(x) for x in gzip.open(path, "rt")]
        for d in lines[1:-1]:
            if not d.get("strategies"):
                continue
            sizes = tuple(len(c) for c in d["candidates"])
            starts = policy.random_starts(lines[0], d["turn"], d["state"].get("tick"), sizes)
            previewed = {(i, j) for i, j, _ in d["previewed"]}
            assert all((i, j) in previewed for i in starts[0] | {0} for j in starts[1] | {0})
            checked += 1
    assert checked
    # Records from before per-seat seeding (no `opening` in the header) drew both seats from one stream.
    old = {"bots": [{"search": "oracle", "oracle_start": 2}], "seed": 9}
    rng = random.Random("4:120")
    assert policy.random_starts(old, 4, 120, (10, 7)) == [set(rng.sample(range(1, 10), 2)), set(rng.sample(range(1, 7), 2))]
    assert policy.random_starts({"bots": [{"search": "full"}]}, 4, 120, (10, 7)) == [set(), set()]


def test_policy_audit_reports_value_lost_and_reply_classes(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("policy_audit", Path(__file__).parent.parent / "tools" / "policy_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    assert audit.parse_variants("both:16,own:32") == [("both", 16), ("own", 32)]
    assert audit.variant_names("own", 32) == ["own0@32", "own1@32"] and audit.variant_names("both", 16) == ["both"]
    best = [{"seat": 0, "class": "d", "base_action": "Wait"}, {"seat": 1, "class": "a", "base_action": "Jab"}]
    row = {"game": "g1", "turn": 3, "characters": ["Ninja", "Cowboy"], "n_base": [30, 30],
           "full": {"value": 0.10, "previews": 100},
           # seat 0 pruned: its pruned value 0.15, but Cowboy's best reply gains 0.08 against it
           "own0": {"value": 0.15, "gain": [0.30, 0.08], "previews": 40, "best": best},
           "own1": {"value": 0.10, "gain": [0.0, 0.0], "previews": 40, "best": best},
           "both": {"value": 0.10, "gain": [0.0, 0.0], "previews": 30, "best": best}}
    lost = audit.value_lost(row, "own0")
    assert lost[0] == pytest.approx((0.10 - (0.15 - 0.08), 0.30))  # 0.03 lost; the missed improvement overstates
    assert audit.value_lost(row, "own1")[1] == pytest.approx((0.0, 0.0))
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(row) + "\n")
    text = audit.summarize([path])
    assert "| own | Ninja v Cowboy | 1 | 1 | 0.0300 |" in text  # one game, one scored seat-decision
    # The loss is the opponent part (0.08 of it; own part 0.10 - 0.15 + ... = -0.05): Cowboy's reply, class a.
    assert "| own | Ninja v Cowboy | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |" in text
    # The control (the full answer against the same replies) comes off; a seat with a short list isn't scored.
    row["own0"]["control_gain"] = [0.0, 0.05]
    assert audit.pruning_loss(row, "own0")[0]["loss"] == pytest.approx(0.03 - 0.05)
    row["n_base"] = [21, 30]
    assert audit.pruning_loss(row, "own0") == {} and set(audit.pruning_loss(row, "both")) == {0, 1}
    assert audit.flagged(path, 0.02) == {"g1": [3]}


def test_policy_audit_classes_each_seats_reply_from_its_own_list():
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from yomibench.engine import Candidate

    spec = importlib.util.spec_from_file_location("policy_audit", Path(__file__).parent.parent / "tools" / "policy_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    base = (tuple(Candidate(i, f"P1Move{i}", None, None) for i in range(3)),
            tuple(Candidate(i, f"P2Move{i}", None, None) for i in range(6)))
    obs = SimpleNamespace(candidates=base)
    expanded = Candidate(5, "P1Move1", None, None, values={"feint": True}, base=1)  # seat 0's id 5; seat 1 has an id 5 too
    by_id = [{c.id: c for c in base[0] + (expanded,)}, {c.id: c for c in base[1]}]
    kept = [{0, 1}, {0, 1, 2, 3, 4, 5}]
    pruned = [{0, 1}, {0, 1, 2, 3, 4, 5}]
    got = audit._classify(obs, 5, 0, by_id, pruned, kept)
    assert (got["class"], got["base_action"], got["variant"]) == ("b", "P1Move1", {"feint": True})
    assert audit._classify(obs, 5, 1, by_id, pruned, kept)["base_action"] == "P2Move5"
    assert audit._classify(obs, 2, 0, by_id, pruned, kept)["class"] == "a"


def test_policy_audit_blames_the_pruned_seats_own_missing_move_when_that_is_the_loss(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("policy_audit", Path(__file__).parent.parent / "tools" / "policy_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    best = [{"seat": 0, "class": "a", "base_action": "Teleport"}, {"seat": 1, "class": "d", "base_action": "Continue"}]
    row = {"game": "g2", "turn": 9, "characters": ["Ninja", "Ninja"], "n_base": [40, 40],
           "full": {"value": 0.30, "previews": 100},
           # seat 0 pruned: the full game is worth 0.30 to it, its pruned search only 0.10, and nothing exploits it
           "own0": {"value": 0.10, "gain": [0.25, 0.0], "control_gain": [0.0, 0.0], "previews": 40, "best": best},
           "own1": {"value": 0.30, "gain": [0.0, 0.0], "control_gain": [0.0, 0.0], "previews": 40, "best": best}}
    x = audit.pruning_loss(row, "own0")[0]
    assert (x["loss"], x["own"], x["opp"]) == (pytest.approx(0.20), pytest.approx(0.20), pytest.approx(0.0))
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(row) + "\n")
    text = audit.summarize([path])
    assert "| own | Ninja v Ninja | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |" in text  # its own Teleport, outside the kept set
    assert "own: Teleport (a" in text


def _audit_tool():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("policy_audit", Path(__file__).parent.parent / "tools" / "policy_audit.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    return audit


def test_policy_audit_replays_capped_picks_by_content_and_scores_quick_variants():
    from types import SimpleNamespace

    from yomibench.engine import Candidate

    audit = _audit_tool()
    cands = tuple(Candidate(i, a, {"x": i}, None) for i, a in enumerate(["Continue", "Jab", "Jab", "Grab"]))
    obs = SimpleNamespace(candidates=(cands, cands))
    # A capped list kept Continue, Jab#1 and Grab: record position 2 is engine id 3.
    assert audit._replay_id(None, obs, 0, 2, {"action": "Grab", "data": {"x": 3}, "extra": None}) == 3
    assert audit._replay_id(None, obs, 0, 1, {"action": "Jab", "data": {"x": 1}, "extra": None}) == 1
    with pytest.raises(RuntimeError):
        audit._replay_id(None, obs, 0, 1, {"action": "Kick", "data": None, "extra": None})
    assert audit.parse_variants("cap:32,pcap:32") == [("cap", 32), ("pcap", 32)]
    assert audit.variant_names("pcap", 32) == ["pcap@32"]
    row = {"n_base": [40, 20], "full": {"value": 0.2},
           "cap@32": {"value": 0.1, "gain": [0.0, 0.05], "control_gain": [0.0, 0.0], "k": 32, "active_above": 32}}
    got = audit.pruning_loss(row, "cap@32")  # both seats scored: one list is over the cap
    assert set(got) == {0, 1} and got[0]["loss"] == pytest.approx(0.2 - (0.1 - 0.05))
    row["n_base"] = [30, 20]
    assert audit.pruning_loss(row, "cap@32") == {}


def test_policy_audit_counts_both_seats_of_a_quick_variant(tmp_path):
    import json

    audit = _audit_tool()
    row = {"game": "g", "turn": 1, "characters": ["Ninja", "Cowboy"], "n_base": [40, 40], "full": {"value": 0.0, "previews": 9},
           "cap@32": {"value": 0.0, "gain": [0.0, 0.0], "control_gain": [0.0, 0.0], "k": 32, "active_above": 32, "previews": 3}}
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert "cap@32 1.00" in audit.summarize([path])  # both seats scored, of two


def test_policy_bot_plays_without_previews_and_parses_its_spec(toy_policy, tmp_path):
    from yomibench.arena import play_game
    from yomibench.bots import PolicyBot, make_bot
    from yomibench.engine import connect

    bot = make_bot(f"policy:{toy_policy}")
    assert isinstance(bot, PolicyBot) and bot.name == "policy:toy-policy" and bot.temperature == 1.0
    cold = make_bot(f"policy:{toy_policy}@temp=0.25@2")
    assert (cold.temperature, cold.candidate_set, cold.name) == (0.25, 2, "policy:toy-policy@temp=0.25@2")
    odd = tmp_path / "C:" / "a@b"  # options are read from the right: the path keeps its colons and @s
    odd.mkdir(parents=True)
    (odd / "p.npz").write_bytes(toy_policy.read_bytes())
    assert make_bot(f"policy:{odd / 'p.npz'}@temp=2").temperature == 2.0
    for bad in ("policy:", f"policy:{toy_policy}@temp=0", f"policy:{toy_policy}@temp=-1"):
        with pytest.raises(ValueError):
            make_bot(bad)

    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 3)
    for seat in (0, 1):
        s = bot.strategy(obs, seat)
        assert len(s) == len(obs.candidates[seat]) and abs(sum(s) - 1) < 1e-9 and min(s) >= 0
        assert max(cold.strategy(obs, seat)) >= max(s)  # a colder policy is more peaked

    calls = {"preview": 0}
    real = engine.preview

    def counting(*a, **kw):
        calls["preview"] += 1
        return real(*a, **kw)

    engine.preview = counting
    result = play_game(engine, (bot, make_bot("random")), ("Toy", "Toy"), seed=5)
    assert result.turns > 0 and calls["preview"] == 0  # no search: it never looks ahead
