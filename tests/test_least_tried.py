"""Least-tried move-type exploration (yomibench.least_tried) and the move-type coverage table
(tools/move_type_coverage.py). Generic: nothing here names a real move or character."""
import collections
import gzip
import hashlib
import importlib.util
import io
import json
import random
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from yomibench import least_tried
from yomibench.bots import MatrixBot
from yomibench.engine import connect
from yomibench.least_tried import LeastTried
from yomibench.selfplay import read_slim, run_selfplay

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("move_type_coverage", ROOT / "tools/move_type_coverage.py")
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

TIMING = {"ms", "seconds", "started", "perf", "yomibench"}


def cand(action, i=0, data=None, extra=None):
    return SimpleNamespace(action=action, data={"i": i} if data is None else data, extra=extra)


def config(counts=None, p=1.0, k=2, taper=None, sha="0" * 64):
    return LeastTried(p, k, 20000, counts or {}, taper or {}, sha, 0)


def write_counts(tmp_path, counts, taper=None, name="counts.json"):
    doc = {"format": least_tried.COUNTS_FORMAT, "window_games": 100, "games": 7, "tars": [], "counts": counts,
           "taper": taper or {}, "created": 0}
    path = tmp_path / name
    return path, least_tried.write_counts(doc, path)


def play(out, games=6, **kw):
    kw.setdefault("explore_cfg", None)
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(games), out,
                 full_previews_fraction=0.0, **kw)
    out_games = []
    for f in sorted(out.glob("g*.jsonl.gz")):
        with gzip.open(f, "rt") as g:
            out_games.append([json.loads(x) for x in g.read().splitlines()])
    return out_games


def strip_timing(lines):
    def clean(d):
        return {k: (clean(v) if isinstance(v, dict) else v) for k, v in d.items() if k not in TIMING}
    return [clean(x) for x in lines]


# --- settings and the pick rule -------------------------------------------------------------------

def test_settings_default_validate_and_absent_means_off():
    assert least_tried.parse_settings(None) is None and least_tried.parse_settings("") is None
    assert least_tried.parse_settings("{}") is None and least_tried.parse_settings({}) is None
    assert least_tried.parse_settings('{"p": 0.03, "k": 2, "window_games": 20000}') == \
        {"p": 0.03, "k": 2, "window_games": 20000}
    assert least_tried.parse_settings({"p": 0.5}) == {"p": 0.5, "k": 2, "window_games": 20000}
    for bad in ('{"p": 0}', '{"p": 1.5}', '{"k": 0}', '{"k": 1.5}', '{"window_games": 0}', '{"move": "Jab"}', "[1]"):
        with pytest.raises(ValueError):
            least_tried.parse_settings(bad)
    assert least_tried.settings_tag({"p": 0.03, "k": 2, "window_games": 20000}) == "p:0.03,k:2,window_games:20000"


def test_least_tried_picks_the_minimum_count_legal_type_with_seeded_ties():
    cands = [[cand("A", 0), cand("A", 1), cand("B"), cand("C", 0), cand("C", 1), cand("D")], [cand("A"), cand("B")]]
    counts = {"X": {"A": 50, "B": 3, "C": 3, "D": 9, "E": 0}}  # E isn't legal here: never chosen
    lt = config(counts)
    seen, variants = collections.Counter(), collections.Counter()
    for s in range(400):
        rng = random.Random(f"{s}:least-tried")
        seat, idx, move, info = lt.choose(cands, ("X", "Y"), [0, 2], [True, True], rng)  # seat 1 used k=2
        count = info["counts_at_pick"]["move_type"]
        assert seat == 0 and move in ("B", "C") and count == 3 and cands[0][idx].action == move
        seen[move] += 1
        variants[idx] += move == "C"
        again = lt.choose(cands, ("X", "Y"), [0, 2], [True, True], random.Random(f"{s}:least-tried"))
        assert again == (seat, idx, move, info)  # seeded: the same draw for the same game
    assert seen["B"] > 120 and seen["C"] > 120  # ties at random
    assert variants[3] > 40 and variants[4] > 40  # the variant uniform within the type
    # a character with no counts: every type ties at 0
    assert lt.choose(cands, ("Z", "Y"), [0, 2], [True, True], random.Random(1))[3]["counts_at_pick"]["move_type"] == 0


def test_candidate_options_are_generic_over_widgets():
    opts = least_tried.candidate_options
    assert opts({"action": "M", "data": {"Speed": "fast", "Aim": {"y": 1, "x": -2}}, "extra": {"t": True, "DI": {"x": 0, "y": 0}}}) \
        == {"data.Speed": '"fast"', "data.Aim": '{"x":-2,"y":1}', "extra.t": "true", "extra.DI": '{"x":0,"y":0}'}
    assert opts({"action": "M", "data": {"x": 28, "y": -71}, "extra": None}) == {"data": '{"x":28,"y":-71}'}  # one XY widget
    assert opts({"action": "M", "data": "left"}) == {"data": '"left"'}
    assert opts({"action": "M", "data": True, "extra": {}}) == {"data": "true"}
    assert opts({"action": "M", "data": None}) == {}
    assert opts(cand("M", data={"n": 3}, extra={"t": False})) == {"data.n": "3", "extra.t": "false"}


def test_least_tried_option_values_within_the_type():
    """Per option with a choice, the least-picked value; then a candidate matching all of them, or
    else the ones matching the most."""
    T = [cand("T", data={"Speed": s, "Aim": a}, extra={"t": t}) for s in ("slow", "fast") for a in (1, 2)
         for t in (False, True)]
    counts = {"X": {"T": 0, "U": 9}}
    options = {"X": {"T": {"data.Speed": {'"fast"': 100, '"slow"': 1}, "extra.t": {"false": 50},
                           "data.Aim": {"1": 4, "2": 4}}}}
    lt = LeastTried(1.0, 2, 100, counts, {}, "0" * 64, 0, options)
    cands = [T + [cand("U")], [cand("U"), cand("T")]]
    aims = collections.Counter()
    for s in range(100):
        seat, idx, move, info = lt.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(s))
        c = cands[0][idx]
        assert move == "T" and c.data["Speed"] == "slow" and c.extra["t"] is True
        assert info["options"]["data.Speed"] == '"slow"' and info["options"]["extra.t"] == "true"
        assert info["counts_at_pick"] == {"move_type": 0, "options": {"data.Speed": 1, "extra.t": 0, "data.Aim": 4}}
        assert info["matched"] == info["of"] == 3
        aims[c.data["Aim"]] += 1
    assert aims[1] > 25 and aims[2] > 25  # tied option values at random
    # no candidate has both least-tried values: one of those matching the most is played
    partial = [c for c in T if not (c.data["Speed"] == "slow" and c.extra["t"])]
    for s in range(30):
        seat, idx, move, info = lt.choose([partial + [cand("U")], [cand("U")]], ("X", "X"), [0, 2], [True, True], random.Random(s))
        assert info["of"] == 3 and info["matched"] == 2
        assert partial[idx].data["Aim"] == int(info["options"]["data.Aim"])


def test_p_k_and_eligibility():
    cands = [[cand("A"), cand("B")], [cand("A"), cand("B")]]
    lt = config(p=0.25, k=1000)
    rng = random.Random(7)
    hits = sum(lt.choose(cands, ("X", "X"), [0, 0], [True, True], rng) is not None for _ in range(4000))
    assert 0.22 < hits / 4000 < 0.28
    lt = config(p=1.0, k=1)
    assert lt.choose(cands, ("X", "X"), [1, 1], [True, True], rng) is None  # both seats at k
    assert lt.choose(cands, ("X", "X"), [0, 0], [False, False], rng) is None  # tapered off this game
    assert lt.choose([[cand("A"), cand("A")], [cand("B")]], ("X", "X"), [0, 0], [True, True], rng) is None  # one type


# --- in self-play --------------------------------------------------------------------------------

def test_absent_flag_leaves_records_identical(tmp_path):
    base = play(tmp_path / "base")
    off = play(tmp_path / "off", explore_least_tried=None)
    assert len(base) == len(off) == 6
    for a, b in zip(base, off):
        assert strip_timing(a) == strip_timing(b)
        assert "explore_least_tried" not in b[0] and "least_tried" not in b[-1]["result"]["perf"]
        assert not any("explore_pick" in t for t in b[1:-1])


def test_selfplay_least_tried_respects_p_k_marks_and_tags(tmp_path):
    counts = {"Toy": {"Wait": 900, "Walk": 800, "Jab": 700, "Heavy": 5, "Grab": 600, "Block": 400}}
    path, sha = write_counts(tmp_path, counts)
    lt = LeastTried.load({"p": 1.0, "k": 2, "window_games": 100}, path)
    assert lt.counts_sha256 == sha == hashlib.sha256(path.read_bytes()).hexdigest()
    games = play(tmp_path / "arm", games=8, explore_least_tried=lt)
    picks = 0
    for lines in games:
        assert lines[0]["explore_least_tried"] == {"p": 1.0, "k": 2, "window_games": 100, "counts_sha256": sha,
                                                   "counts_games": 7, "taper": {}}
        per_seat = [0, 0]
        for t in lines[1:-1]:
            if t.get("explore_mode") != "least-tried":
                assert "explore_pick" not in t
                continue
            info = t["explore_pick"]
            seat = info["seat"]
            assert t["explore"] == [seat == 0, seat == 1] and t.get("search", "full") == "full"
            picked = t["candidates"][seat][t["picks"][seat]]["action"]
            legal = {c["action"] for c in t["candidates"][seat]}
            assert info["kind"] == "least-tried" and info["move_type"] == picked
            n = info["counts_at_pick"]["move_type"]
            assert n == counts["Toy"].get(picked, 0) == min(counts["Toy"].get(m, 0) for m in legal)
            assert info["matched"] == info["of"] == len(info["options"])  # the toy's options are independent
            played = least_tried.candidate_options(t["candidates"][seat][t["picks"][seat]])
            assert all(played[k] == v for k, v in info["options"].items())
            per_seat[seat] += 1
        assert max(per_seat) <= 2  # k
        assert lines[-1]["result"]["perf"]["least_tried"] == sum(per_seat)
        picks += sum(per_seat)
    assert picks >= 8  # p = 1: every game explores (the toy offers Heavy on every decision)


def test_never_on_quick_or_opening_turns_and_p_is_the_rate(tmp_path):
    path, _ = write_counts(tmp_path, {})
    lt = LeastTried.load({"p": 0.3, "k": 1000, "window_games": 100}, path)
    games = play(tmp_path / "arm", games=8, explore_least_tried=lt, playout_cap=(0.5, 3),
                 opening=(2, 1.0, 0.0))
    eligible = explored = 0
    for lines in games:
        for t in lines[1:-1]:
            mode = t.get("explore_mode")
            if mode == "least-tried":
                assert t["search"] == "full" and not t.get("opening")
                explored += 1
            if t.get("strategies") and t["search"] == "full" and not t.get("opening"):
                eligible += 1
            if t["search"] == "cheap" or t.get("opening"):
                assert mode != "least-tried"
    assert eligible > 40 and 0.15 < explored / eligible < 0.45


def test_a_tapered_character_explores_in_only_its_floor_share_of_games(tmp_path):
    lt = config(taper={"Toy": 0.05})
    on = [lt.seat_on(s, seat, "Toy") for s in range(4000) for seat in (0, 1)]
    assert 0.035 < sum(on) / len(on) < 0.065  # about the floor, never zero
    assert all(config().seat_on(s, 0, "Toy") for s in range(50))  # untapered: always
    assert all(lt.seat_on(s, 0, "Other") for s in range(50))
    path, _ = write_counts(tmp_path, {}, taper={"Toy": 1e-9})
    games = play(tmp_path / "arm", games=4, explore_least_tried=LeastTried.load(
        {"p": 1.0, "k": 2, "window_games": 100}, path))
    assert all(g[0]["explore_least_tried"]["taper"] == {"Toy": 1e-9} for g in games)
    assert not any(t.get("explore_mode") == "least-tried" for g in games for t in g[1:-1])


def test_value_only_labels_cut_before_least_tried_picks(tmp_path):
    """Positions up to a game's last least-tried pick train on the search value alone (the D-uo path,
    --explore-targets uniform-only), later ones on the outcome blend; validation drops them."""
    pytest.importorskip("torch")
    from yomibench.selfplay import uniform_explore
    from yomibench.train import explore_cutoffs, select_rows

    path, _ = write_counts(tmp_path, {})
    games = play(tmp_path / "arm", games=4, explore_least_tried=LeastTried.load(
        {"p": 0.5, "k": 2, "window_games": 100}, path))
    checked = 0
    for f in sorted((tmp_path / "arm").glob("g*.jsonl.gz")):
        header, result, turns = read_slim(f)
        marks = [k for k, t in enumerate(turns) if t.explore_mode == "least-tried"]
        if not marks:
            continue
        assert all(uniform_explore(turns[k]) for k in marks)
        cut, val_cut = explore_cutoffs(turns, "uniform-only")
        assert cut == val_cut == marks[-1]
        z = 0.0 if result["result"]["winner"] is None else (1.0 if result["result"]["winner"] == 0 else -1.0)
        rows, counts = select_rows(turns, "train", "uniform-only", 0.5, z)
        for k, y, _ in rows:
            if k <= cut:
                assert y == turns[k].value  # value only: the outcome is no target here
            else:
                assert y == pytest.approx(0.5 * z + 0.5 * (z if turns[k].value is None else turns[k].value))
        val_rows, _ = select_rows(turns, "val", "uniform-only", 0.5, z)
        assert all(k > cut for k, _, _ in val_rows)
        none_cut, _ = explore_cutoffs(turns, "none")
        assert none_cut == -1
        checked += 1
    assert checked >= 1 and games


def test_the_cache_keeps_least_tried_marks(tmp_path):
    pytest.importorskip("torch")
    from yomibench.cache import encode_record

    path, _ = write_counts(tmp_path, {})
    play(tmp_path / "arm", games=3, explore_least_tried=LeastTried.load({"p": 1.0, "k": 2, "window_games": 100}, path))
    modes = [m for f in (tmp_path / "arm").glob("g*.jsonl.gz") for m in encode_record(str(f))["modes"]]
    assert "least-tried" in modes


def test_cli_flags_reach_the_records_and_need_a_counts_file(tmp_path):
    from yomibench import cli

    path, sha = write_counts(tmp_path, {"Toy": {"Wait": 3}})
    out = tmp_path / "out"
    assert cli.main(["selfplay", "--engine", "toy", "--games", "2", "--out", str(out), "--bot", "matrix",
                     "--explore-epsilon", "0", "--no-ledger", "--explore-least-tried", '{"p": 1.0, "k": 1}',
                     "--least-tried-counts", str(path)]) in (0, None)
    for f in out.glob("g*.jsonl.gz"):
        with gzip.open(f, "rt") as g:
            h = json.loads(g.readline())
        assert h["explore_least_tried"]["counts_sha256"] == sha and h["explore_least_tried"]["k"] == 1
    with pytest.raises(SystemExit):
        cli.main(["selfplay", "--engine", "toy", "--games", "1", "--out", str(tmp_path / "x"), "--no-ledger",
                  "--explore-least-tried", '{"p": 1.0}'])


# --- counts refresh ----------------------------------------------------------------------------------

def make_tar(games):
    """A chunk tar of toy-shaped records: games = [[(actions0, actions1, picks), ...], ...]."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for n, decisions in enumerate(games):
            lines = [{"format": 3, "characters": ["P", "Q"], "seed": n}]
            for a0, a1, picks in decisions:
                lines.append({"turn": 1, "state": {"players": [{"candidates": "not this"}, {}]}, "actionable": [1, 1],
                              "strategies": None, "value": None,
                              "candidates": [[{"action": a, "data": None} for a in a0],
                                             [{"action": a, "data": None} for a in a1]], "picks": picks, "ms": 1})
            lines.append({"result": {"winner": 0, "outcome": "p1"}})
            data = gzip.compress("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines).encode())
            info = tarfile.TarInfo(f"g{n:012d}-abc.jsonl.gz")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Files(dict):
    """A hub's files; `get` counts downloads."""

    def __init__(self):
        super().__init__()
        self.gets = collections.Counter()

    def add(self, name, data, sidecar=None):
        self[name] = data
        self[name + ".sha256"] = ((sidecar or hashlib.sha256(data).hexdigest()) + "\n").encode()

    def get(self, name):
        if not name.endswith(".sha256"):
            self.gets[name] += 1
        return dict.get(self, name)


def test_count_tar_counts_picks_where_a_seat_had_a_choice():
    data = make_tar([[(["A", "B", "B"], ["A"], [1, 0]), (["A", "B"], ["A", "C"], [0, 1])], [(["A", "B"], ["C", "C"], [1, 1])]])
    got = least_tried.count_tar(data)
    assert got["games"] == 2 and got["counts"] == {"P": {"A": 1, "B": 2}, "Q": {"C": 2}}


def test_count_tar_counts_option_values_of_the_candidate_played():
    buf = io.BytesIO()
    lines = [{"format": 3, "characters": ["P", "Q"], "seed": 1}]
    for pick in (0, 1, 1):
        lines.append({"turn": 1, "state": {}, "actionable": [1, 1], "candidates": [
            [{"action": "M", "data": {"Speed": "slow"}, "extra": {"t": False}},
             {"action": "M", "data": {"Speed": "fast"}, "extra": {"t": True}}],
            [{"action": "N", "data": None, "extra": None}, {"action": "O", "data": {"x": 1, "y": 0}, "extra": None}]],
            "picks": [pick, 1], "ms": 1})
    lines.append({"result": {"winner": 0}})
    data = gzip.compress("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in lines).encode())
    with tarfile.open(fileobj=buf, mode="w") as tf:
        info = tarfile.TarInfo("g000000000001-abc.jsonl.gz")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    got = least_tried.count_tar(buf.getvalue())
    assert got["counts"] == {"P": {"M": 3}, "Q": {"O": 3}}
    assert got["options"] == {"P": {"M": {"data.Speed": {'"slow"': 1, '"fast"': 2}, "extra.t": {"false": 1, "true": 2}}},
                              "Q": {"O": {"data": {'{"x":1,"y":0}': 3}}}}


def test_refresh_downloads_each_tar_once_keeps_the_window_and_retries_bad_ones(tmp_path):
    hub = Files()
    one = [[(["A", "B"], ["A", "B"], [0, 1])]]
    for c in range(4):  # 4 production tars of 1 game, generation 9
        hub.add(f"r3-box-g9-c{c}.tar", make_tar(one))
    hub.add("r3-box.explore-g9-c0.tar", make_tar([[(["A", "B"], ["A", "B"], [1, 1])]]))
    hub.add("r3-box.explore-g9-c1.tar", make_tar(one), sidecar="0" * 64)  # corrupt
    hub.add("r3-gate-x-s1-records.tar", make_tar(one))  # gate records: never counted
    hub["r3-box-g9-c9.tar"] = make_tar(one)  # no sidecar yet: not counted
    cache = tmp_path / "cache.json"
    doc = least_tried.refresh_counts(sorted(hub), hub.get, cache, window_games=3, log=lambda m: None)
    # newest first by generation and chunk; stops once new tars fill the window of 3 games
    assert doc["games"] == 3 and len(doc["tars"]) == 3
    assert "r3-gate-x-s1-records.tar" not in hub.gets and "r3-box-g9-c9.tar" not in hub.gets
    assert hub.gets["r3-box.explore-g9-c1.tar"] == 1  # tried; bad sha256: retried next time
    first = dict(hub.gets)
    doc2 = least_tried.refresh_counts(sorted(hub), hub.get, cache, window_games=3, log=lambda m: None)
    assert hub.gets["r3-box.explore-g9-c1.tar"] == 2
    assert all(hub.gets[n] == first[n] for n in first if n != "r3-box.explore-g9-c1.tar")  # nothing twice
    assert doc2["games"] == 3
    hub.add("r3-box-g10-c0.tar", make_tar([[(["A", "B"], ["A", "B"], [1, 0])]] * 2))  # a new generation, 2 games
    doc3 = least_tried.refresh_counts(sorted(hub), hub.get, cache, window_games=3, log=lambda m: None)
    assert doc3["tars"][0] == "r3-box-g10-c0.tar" and doc3["games"] >= 3 and hub.gets["r3-box-g10-c0.tar"] == 1
    assert doc3["counts"]["P"]["B"] >= 2 and "options" in doc3
    path = tmp_path / "c.json"
    sha = least_tried.write_counts({**doc3, "taper": {"P": 0.015}}, path)
    lt = LeastTried.load({"p": 0.03, "k": 2, "window_games": 3}, path)
    assert lt.counts_sha256 == sha and lt.taper == {"P": 0.015} and lt.counts == doc3["counts"]


def test_taper_from_coverage_takes_only_tapered_characters():
    doc = {"taper": {"P": {"tapered": True, "floor": 0.015}, "Q": {"tapered": False, "floor": 0.015},
                     "R": {"tapered": True, "floor": 0}}}
    assert least_tried.taper_from_coverage(doc) == {"P": 0.015}
    assert least_tried.taper_from_coverage(None) == {}


# --- coverage table ------------------------------------------------------------------------------------

def toy_record(decisions, winner=0, chars=("P", "Q")):
    """decisions: (actions0, actions1, picks, value, search, explore, explore_mode)."""
    lines = [{"format": 3, "characters": list(chars), "seed": 1}]
    for a0, a1, picks, value, search, explore, mode in decisions:
        d = {"turn": 1, "state": {"players": []}, "actionable": [1, 1], "search": search,
             "strategies": [[1.0], [1.0]], "value": value}
        if explore:
            d.update(explore=explore, explore_mode=mode)
        d.update(candidates=[[{"action": a} for a in a0], [{"action": a} for a in a1]], picks=picks, ms=1)
        lines.append(d)
    lines.append({"result": {"winner": winner, "outcome": "p1" if winner == 0 else "p2"}})
    return "".join(json.dumps(x) + "\n" for x in lines)


def test_coverage_numbers_on_a_toy_record_set():
    texts = [
        toy_record([(["A", "B", "B"], ["A", "C"], [1, 0], 0.5, "full", None, None),       # P picks B
                    (["A", "B"], ["A", "C"], [0, 1], 0.0, "cheap", [True, False], "least-tried"),  # after B (cheap)
                    (["A", "B"], ["A"], [1, 0], 1.0, "full", None, None)], winner=0),       # after A; z = +1
        toy_record([(["A", "C"], ["A", "C"], [1, 1], -0.5, "full", [False, True], "uniform-moves"),
                    (["A", "C"], ["A", "C"], [0, 0], -1.0, "full", None, None)], winner=1),   # z = -1
    ]
    stats, totals = {}, {}
    for t in texts:
        coverage.record_stats(t, stats, totals)
    raw = {"games": 2, "stats": stats, "totals": totals}
    P, Q = stats["P"], stats["Q"]
    assert P["B"]["legal"] == 2 and P["B"]["offered"] == 3 and P["B"]["picked"] == 2 and P["B"]["explored"] == 0
    assert P["A"]["picked"] == 1 and P["A"]["explored"] == 1 and P["A"]["least_tried"] == 1
    assert P["C"]["picked"] == 1 and P["C"]["legal"] == 2  # game 2: P picks C, then A
    assert P["B"]["after"] == 1 and P["B"]["mse_n"] == 0  # the position after B was a quick turn
    assert P["A"]["after"] == 1 and P["A"]["mse_n"] == 1 and P["A"]["mse_sum"] == pytest.approx(0.0)  # (1-1)^2
    assert P["C"]["after"] == 1 and P["C"]["mse_sum"] == pytest.approx(0.0)  # value -1, z -1
    assert Q["C"]["explored"] == 1 and Q["C"]["least_tried"] == 0 and Q["A"]["picked"] == 2
    assert "A" not in Q or Q["A"]["legal"] >= 1
    assert totals["P"]["decisions"] == 5 and totals["Q"]["decisions"] == 4  # Q had one forced turn
    doc = coverage.summarize(raw, min_positions=1, error_ratio=1.2, floor=0.015, min_error_positions=1)
    assert doc["characters"]["P"]["types"]["B"]["caught_up"] is False  # no valued position after it
    assert doc["taper"]["P"]["tapered"] is False and doc["taper"]["P"]["behind"] == ["B"]
    md = coverage.markdown(doc, 2)
    assert "| P |" in md and "| B |" in md


def test_taper_flag_from_the_coverage_rule():
    def rec(winner, value):
        return toy_record([(["A", "B"], ["A", "B"], [0, 1], value, "full", None, None),
                           (["A", "B"], ["A", "B"], [1, 0], value, "full", None, None),
                           (["A", "B"], ["A", "B"], [0, 0], value, "full", None, None)], winner=winner)
    stats, totals = {}, {}
    for _ in range(10):
        coverage.record_stats(rec(0, 0.9), stats, totals)
    raw = {"games": 10, "stats": stats, "totals": totals}
    doc = coverage.summarize(raw, min_positions=5, error_ratio=1.2, floor=0.015, min_error_positions=5)
    assert doc["taper"]["P"] == {"tapered": True, "floor": 0.015, "caught_up": 2, "legal_types": 2, "behind": [],
                                 "legal_option_values": 0, "option_values_caught_up": 0, "options_behind": [],
                                 "legal_di_values": 0, "di_values_caught_up": 0, "di_behind": []}
    assert least_tried.taper_from_coverage(doc) == {"P": 0.015, "Q": 0.015}
    strict = coverage.summarize(raw, min_positions=1000, error_ratio=1.2, floor=0.015, min_error_positions=5)
    assert strict["taper"]["P"]["tapered"] is False and least_tried.taper_from_coverage(strict) == {}
    # one type's error far above common moves': not caught up
    for _ in range(10):
        coverage.record_stats(toy_record([(["A", "B", "C"], ["A"], [2, 0], 0.9, "full", None, None),
                                          (["A", "B"], ["A"], [0, 0], -0.9, "full", None, None)], winner=0),
                              stats, totals)
    doc = coverage.summarize({"games": 20, "stats": stats, "totals": totals}, min_positions=5, error_ratio=1.2,
                             floor=0.015, min_error_positions=5)
    assert doc["characters"]["P"]["types"]["C"]["caught_up"] is False and doc["taper"]["P"]["tapered"] is False


def test_coverage_cli_writes_md_and_json_from_tars(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for n in range(3):
            data = gzip.compress(toy_record([(["A", "B"], ["A", "B"], [0, 1], 0.2, "full", None, None),
                                             (["A", "B"], ["A", "B"], [1, 0], 0.4, "full", None, None)]).encode())
            info = tarfile.TarInfo(f"g{n:012d}-abc.jsonl.gz")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    (tmp_path / "r3-box-g9-c0.tar").write_bytes(buf.getvalue())
    (tmp_path / "r3-box-g9-c1.tar").write_bytes(buf.getvalue())
    out = tmp_path / "res" / "move-coverage"
    import os
    import subprocess
    import sys
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])}
    subprocess.run([sys.executable, str(ROOT / "tools/move_type_coverage.py"), str(tmp_path / "r3-*.tar"),
                    "--out", str(out), "--procs", "2"], check=True, env=env, timeout=120)
    doc = json.loads((tmp_path / "res" / "move-coverage.json").read_text())
    assert doc["games"] == 6 and doc["characters"]["P"]["types"]["A"]["picked"] == 6
    assert doc["characters"]["P"]["types"]["A"]["after"] == 6  # P picked A first in every game
    assert (tmp_path / "res" / "move-coverage.md").read_text().startswith("# Move-type coverage")
    with pytest.raises(SystemExit):
        coverage.main([str(tmp_path / "r3-*.tar"), "--floor", "0"])


def opt_record(picks_speed, value, winner=0):
    """Records whose P candidates of type M carry an option (data.Speed slow/fast) and a toggle."""
    m = [{"action": "M", "data": {"Speed": "slow"}, "extra": {"t": False}},
         {"action": "M", "data": {"Speed": "fast"}, "extra": {"t": False}}, {"action": "W", "data": None}]
    lines = [{"format": 3, "characters": ["P", "Q"], "seed": 1}]
    for i, sp in enumerate(picks_speed):
        lines.append({"turn": i, "state": {}, "actionable": [1, 1], "search": "full", "strategies": [[1.0], [1.0]],
                      "value": value, "candidates": [m, [{"action": "W"}, {"action": "V"}]],
                      "picks": [sp, 0], "ms": 1})
    lines.append({"result": {"winner": winner, "outcome": "p1"}})
    return "".join(json.dumps(x) + "\n" for x in lines)


def test_coverage_and_taper_per_option_value():
    stats, totals, ostats = {}, {}, {}
    for _ in range(10):
        coverage.record_stats(opt_record([0, 2, 1, 0, 2], 0.9), stats, totals, ostats)  # slow, W, fast, slow, W
    slow = ostats["P"]["M"]["data.Speed"]['"slow"']
    fast = ostats["P"]["M"]["data.Speed"]['"fast"']
    assert slow["legal"] == fast["legal"] == 50 and slow["picked"] == 20 and fast["picked"] == 10
    assert slow["after"] == 20 and fast["after"] == 10 and slow["mse_n"] == 20
    assert ostats["P"]["M"]["extra.t"]["false"]["legal"] == 0  # one value: never a choice
    raw = {"games": 10, "stats": stats, "totals": totals, "ostats": ostats}
    doc = coverage.summarize(raw, min_positions=10, error_ratio=1.2, floor=0.015, min_error_positions=5)
    ch = doc["characters"]["P"]
    assert set(ch["options"]["M"]) == {"data.Speed"} and ch["legal_option_values"] == 2
    assert ch["options"]["M"]["data.Speed"]['"fast"']["caught_up"] is True
    assert "offered" not in ch["options"]["M"]["data.Speed"]['"fast"']
    assert doc["taper"]["P"]["tapered"] is True
    strict = coverage.summarize(raw, min_positions=15, error_ratio=1.2, floor=0.015, min_error_positions=5)
    # W (10 after) and the fast value (10 after) fall short of 15; M (30) and slow (20) don't
    assert strict["taper"]["P"]["tapered"] is False
    assert strict["taper"]["P"]["options_behind"] == ['M data.Speed="fast"'] and strict["taper"]["P"]["behind"] == ["W"]
    md = coverage.markdown(doc, 1)
    assert "| M | data.Speed |" in md


def test_di_values_taper_per_character_pooled_over_move_types():
    def row(legal, after, mse=0.25):
        return {"legal": legal, "offered": 0, "picked": after, "explored": 0, "least_tried": 0, "after": after,
                "mse_n": after, "mse_sum": mse * after}
    stats = {"P": {"M": row(50, 40), "N": row(50, 40)}}
    totals = {"P": {"decisions": 100, "full_decisions": 100, "positions": 200, "mse_n": 80, "mse_sum": 20.0}}
    # value a: 6 after under each type (12 pooled); value b: 2 under each (4 pooled); a non-DI option judged per type
    ostats = {"P": {"M": {"extra.DI": {"a": row(9, 6), "b": row(9, 2)}, "data.S": {"x": row(9, 12), "y": row(9, 12)}},
                    "N": {"extra.DI": {"a": row(9, 6), "b": row(9, 2)}}}}
    raw = {"games": 10, "stats": stats, "totals": totals, "ostats": ostats}
    doc = coverage.summarize(raw, min_positions=10, error_ratio=1.2, floor=0.015, min_error_positions=1)
    ch, t = doc["characters"]["P"], doc["taper"]["P"]
    assert "extra.DI" not in ch["options"].get("M", {}) and "N" not in ch["options"]  # not judged per type x value
    assert ch["di"]["a"]["after"] == 12 and ch["di"]["a"]["legal"] == 18 and ch["di"]["a"]["caught_up"] is True
    assert ch["di"]["b"]["after"] == 4 and ch["di"]["b"]["caught_up"] is False
    assert (t["legal_di_values"], t["di_values_caught_up"], t["di_behind"]) == (2, 1, ["b"])
    assert t["legal_option_values"] == 2 and t["option_values_caught_up"] == 2  # other options as before
    assert t["tapered"] is False  # b holds the taper back
    loose = coverage.summarize(raw, min_positions=4, error_ratio=1.2, floor=0.015, min_error_positions=1)
    assert loose["taper"]["P"]["tapered"] is True and loose["summary"]["P"]["di_values_caught_up"] == 2
    assert "DI values" in coverage.markdown(doc, 1)


# --- "expand": the chosen type's factor variants (set 6's expand-only option values) -------------------

def test_expand_setting_parses_and_is_written_only_when_on():
    assert least_tried.parse_settings({"p": 0.03, "expand": True}) == {"p": 0.03, "k": 2, "window_games": 20000,
                                                                        "expand": True}
    assert least_tried.parse_settings({"p": 0.03, "expand": False}) == {"p": 0.03, "k": 2, "window_games": 20000}
    with pytest.raises(ValueError):
        least_tried.parse_settings({"expand": "yes"})
    assert least_tried.settings_tag({"p": 0.03, "k": 2, "window_games": 1, "expand": True}).endswith(",expand:True")
    off = LeastTried(1.0, 2, 100, {}, {}, "0" * 64, 0)
    on = off._replace(expand=True)
    assert "expand" not in off.header() and on.header()["expand"] is True


def test_expand_lets_the_option_choice_reach_variants_the_base_list_lacks():
    counts = {"X": {"T": 0, "U": 9}}
    options = {"X": {"T": {"extra.s": {"0": 40, "1": 0}}}}  # value 1 never picked; only a variant has it
    lt = LeastTried(1.0, 2, 100, counts, {}, "0" * 64, 0, options, expand=True)
    calls = []

    def run(seed):
        cands = [[cand("T", extra={"s": 0}), cand("T", extra={"s": 0}, data={"a": 1}), cand("U")], [cand("U")]]

        def expand(seat, idx):
            calls.append((seat, list(idx)))
            cands[seat].append(cand("T", extra={"s": 1}))  # the engine appends in place
            return 1
        return cands, lt.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(seed), expand)

    for s in range(20):
        cands, (seat, idx, move, info) = run(s)
        assert seat == 0 and move == "T" and cands[0][idx].extra == {"s": 1}
        assert info["options"] == {"extra.s": "1"} and info["expanded"] == 1 and info["matched"] == 1
    assert calls[0] == (0, [0, 1])  # the chosen type's candidates only
    # the same seed gives the same draw
    assert run(7)[1] == run(7)[1]
    # without expand (setting off, or no callback), the variant is out of reach and nothing is called
    calls.clear()
    plain = lt._replace(expand=False)
    cands = [[cand("T", extra={"s": 0}), cand("U")], [cand("U")]]
    seat, idx, move, info = plain.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(1), lambda s, i: 1 / 0)
    assert "expanded" not in info and not calls
    assert lt.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(1)) == \
        plain.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(1))


def test_expand_caps_the_candidates_it_expands():
    lt = LeastTried(1.0, 2, 100, {"X": {"T": 0, "U": 9}}, {}, "0" * 64, 0, {}, expand=True)
    cands = [[cand("T", data={"i": i}) for i in range(20)] + [cand("U")], [cand("U")]]
    got = []
    lt.choose(cands, ("X", "X"), [0, 2], [True, True], random.Random(3), lambda s, idx: got.append(idx) or 0)
    assert len(got[0]) == least_tried.EXPAND_MAX and got[0] == sorted(got[0]) and set(got[0]) <= set(range(20))


def test_selfplay_with_expand_on_an_engine_without_factors_plays_as_without_it(tmp_path):
    counts = {"Toy": {"Wait": 900, "Walk": 800, "Jab": 700, "Heavy": 5, "Grab": 600, "Block": 400}}
    path, _ = write_counts(tmp_path, counts)
    plain = LeastTried.load({"p": 1.0, "k": 2, "window_games": 100}, path)
    wide = LeastTried.load({"p": 1.0, "k": 2, "window_games": 100, "expand": True}, path)
    a = play(tmp_path / "a", games=4, explore_least_tried=plain)
    b = play(tmp_path / "b", games=4, explore_least_tried=wide)
    for x, y in zip(a, b):
        assert y[0]["explore_least_tried"].pop("expand") is True
        assert strip_timing(x)[0] == strip_timing(y)[0]
        for tx, ty in zip(strip_timing(x)[1:-1], strip_timing(y)[1:-1]):
            if "explore_pick" in ty:
                assert ty["explore_pick"].pop("expanded") == 0  # the toy has no factors: nothing to add
            assert tx == ty
