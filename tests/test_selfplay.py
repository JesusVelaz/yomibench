import json

from yomibench.bots import MatrixBot
from yomibench.engine import connect, expand_specs
from yomibench.selfplay import read_record, run_selfplay


def test_expand_specs_ranges_and_pool_files(tmp_path):
    pool = tmp_path / "pool.txt"
    pool.write_text("# my-pc\ntcp:10.0.0.2:7777-7778\ntcp:10.0.0.3:7777, toy\n")
    assert expand_specs(f"toy,@{pool}") == [
        "toy", "tcp:10.0.0.2:7777", "tcp:10.0.0.2:7778", "tcp:10.0.0.3:7777", "toy"]


def test_selfplay_records_replay_to_the_same_result(tmp_path):
    written, dropped = run_selfplay([lambda: connect("toy")] * 2, lambda: MatrixBot(iterations=50),
                                    [("Toy", "Toy")], range(3), tmp_path, full_previews_fraction=1.0)
    assert (written, dropped) == (3, 0)
    files = sorted(tmp_path.glob("g*.jsonl.gz"))
    assert len(files) == 3
    lines = read_record(files[0])
    header, turns, result = lines[0], lines[1:-1], lines[-1]["result"]
    assert header["engine"]["engine"] == "toy" and header["seed"] == 0 and header["format"] == 3
    assert header["bots"][0]["bot"] == "matrix" and result["end"] in ("ko", "time")
    # The record alone replays the game: picks index the stored candidate lists.
    engine = connect("toy")
    obs = engine.new_match("Toy", "Toy", 0)
    for t in turns:
        assert obs.state == t["state"]
        assert [[c.action for c in s] for s in obs.candidates] == [[c["action"] for c in s] for s in t["candidates"]]
        if t["strategies"] is not None:
            for seat in (0, 1):
                s = t["strategies"][seat]
                assert len(s) == len(t["candidates"][seat]) and abs(sum(s) - 1) < 1e-3
            n0, n1 = (len(c) for c in t["candidates"])
            previewed = {(i, j): st for i, j, st in t["previewed"]}
            assert t["previews"] == len(previewed) == n0 * n1
            # The stored states are what previewing the turn gives.
            i, j = t["picks"]
            assert engine.preview([(i, j)], candidates=False)[0].state == previewed[i, j]
        obs = engine.step(*t["picks"])
    assert [p["hp"] for p in obs.state["players"]] == result["hp"]
    # A rerun skips finished seeds.
    assert run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50), [("Toy", "Toy")],
                        range(4), tmp_path) == (1, 0)


def test_a_crashed_game_is_dropped_not_written(tmp_path):
    from yomibench.engine import Engine, EngineCrashed, InProcess
    from yomibench.toy import ToyEngine

    class Dies(InProcess):
        def request(self, msg):
            if msg.get("cmd") == "step":
                raise EngineCrashed("boom")
            return super().request(msg)

    made = []

    def make():
        made.append(1)
        return Engine(Dies(ToyEngine()) if len(made) == 1 else InProcess(ToyEngine()))

    assert run_selfplay([make], lambda: MatrixBot(iterations=20), [("Toy", "Toy")], range(2), tmp_path) == (1, 1)
    assert len(list(tmp_path.glob("g*.jsonl.gz"))) == 1
    assert json.loads((tmp_path / "drops.jsonl").read_text())["seed"] == 0


def test_a_turn_limited_game_is_truncated_not_a_draw():
    from yomibench.selfplay import play_recorded

    engine = connect("toy")
    lines = play_recorded(engine, MatrixBot(iterations=20), ("Toy", "Toy"), 0, engine.hello(), max_turns=3)
    r = lines[-1]["result"]
    assert (r["outcome"], r["end"], r["done"], r["winner"], r["turns"]) == ("truncated", "max_turns", False, None, 3)


def test_a_server_restarting_at_startup_is_waited_for_and_a_busy_one_skipped(tmp_path, monkeypatch):
    from yomibench import selfplay
    from yomibench.engine import Engine, EngineCrashed, InProcess
    from yomibench.toy import ToyEngine

    monkeypatch.setattr(selfplay.time, "sleep", lambda s: None)
    attempts = []

    def restarting():
        attempts.append(1)
        if len(attempts) < 3:
            raise EngineCrashed("engine closed the connection")
        return connect("toy")

    class Busy(InProcess):
        def request(self, msg):
            return {"error": {"code": "busy", "message": "this server already has a client"}}

    written, dropped = run_selfplay([restarting, lambda: Engine(Busy(ToyEngine()))],
                                    lambda: MatrixBot(iterations=20), [("Toy", "Toy")], range(2), tmp_path)
    assert (written, dropped, len(attempts)) == (2, 0, 3)


def test_a_server_that_differs_from_the_pool_is_skipped(tmp_path):
    from yomibench.engine import Engine, InProcess
    from yomibench.toy import ToyEngine

    class OtherRules(InProcess):
        def request(self, msg):
            r = super().request(msg)
            if msg.get("cmd") == "hello":
                r = {**r, "match_settings": {"stage_width": 1100}}
            return r

    made = []

    def first():
        made.append("ok")
        return connect("toy")

    written, _ = run_selfplay([first, lambda: Engine(OtherRules(ToyEngine()))], lambda: MatrixBot(iterations=20),
                              [("Toy", "Toy")], range(3), tmp_path)
    assert written == 3  # all games went to the matching server


def test_terminal_positions_get_exact_values():
    from yomibench.search import evaluate, terminal_value

    p = lambda hp: {"hp": hp, "max_hp": 100}  # noqa: E731
    assert terminal_value({"players": [p(50), p(40)], "time_left": 10}, 0) is None
    assert terminal_value({"players": [p(50), p(40)], "time_left": -1}, 0) == 1.0  # ahead at time: a win
    assert terminal_value({"players": [p(0), p(40)]}, 0) == -1.0
    assert terminal_value({"players": [p(40), p(40)], "time_left": -1}, 1) == 0.0
    assert evaluate({"players": [p(51), p(50)], "time_left": -1}, 0) == 1.0


def test_stale_temp_records_are_removed_and_fresh_ones_kept(tmp_path):
    import os
    import time

    from yomibench.selfplay import _remove_stale_temp_files

    old, fresh = tmp_path / ".g000000000001.1.2.tmp", tmp_path / ".g000000000002.1.3.tmp"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(old, (time.time() - 7200, time.time() - 7200))
    _remove_stale_temp_files(tmp_path)
    assert not old.exists() and fresh.exists()


def test_records_are_slim_by_default_and_a_seeded_share_keeps_previews(tmp_path):
    from yomibench.selfplay import full_previews

    keep = [s for s in range(400) if full_previews(s, 0.05)]
    assert 5 <= len(keep) <= 40 and keep == [s for s in range(400) if full_previews(s, 0.05)]  # fixed per seed
    assert not any(full_previews(s, 0.0) for s in range(100)) and all(full_previews(s, 1.0) for s in range(100))
    seeds = range(keep[0] - 2, keep[0] + 2)  # three slim games around one that keeps previews
    run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=50, search="oracle"), [("Toy", "Toy")],
                 seeds, tmp_path, full_previews_fraction=0.05)
    for path in sorted(tmp_path.glob("g*.jsonl.gz")):
        lines = read_record(path)
        full = full_previews(lines[0]["seed"], 0.05)
        assert lines[0]["previews"] == ("full" if full else "slim") and lines[0]["full_previews_fraction"] == 0.05
        searched = [ln for ln in lines[1:-1] if ln["strategies"] is not None]
        assert searched and all(("previewed" in ln) == full for ln in searched)
        assert all(ln["previews"] > 0 and ln["candidate_values"] is not None for ln in searched)  # survive slim


def _old_shard_seeds(seeds, k, n, n_matchups):
    """The dealing before round 4: whole matchup cycles per child."""
    return [s for s in seeds if (s // max(1, n_matchups)) % n == k]


def test_shards_deal_every_seed_once_and_keep_every_child_busy():
    # A chunk shorter than a few long weighted cycles (336 games, 96 slots, 6 children) used to be
    # 3.5 whole cycles dealt to 4 children while 2 idled. Round-robin dealing balances any chunk.
    # Within 1 is impossible with every child playing every matchup: every 336- and 200-seed run
    # balanced exactly over 8 children would make the deal repeat every gcd(336, 200) = 8 seeds,
    # giving each child one of 4 matchups. So the bound is 2.
    import random

    from yomibench.cli import shard_seeds

    rng = random.Random(4)
    starts = [0, 1, 47, 95, 1000] + [rng.randrange(10 ** 7) for _ in range(6)]
    for m in (4, 14, 48, 96):
        for n in (1, 2, 4, 6, 8):
            for length in (100, 200, 300, 336):
                for first in starts:
                    seeds = range(first, first + length)
                    shards = [shard_seeds(seeds, k, n, m) for k in range(n)]
                    assert sorted(s for mine in shards for s in mine) == list(seeds)
                    loads = [len(mine) for mine in shards]
                    assert max(loads) - min(loads) <= 2, (m, n, length, first, loads)
    old = [len(_old_shard_seeds(range(336), k, 6, 96)) for k in range(6)]
    assert old == [96, 96, 96, 48, 0, 0]  # the problem: four children busy, two idle
    assert [len(shard_seeds(range(336), k, 6, 96)) for k in range(6)] == [56] * 6


def test_shards_give_every_child_every_matchup():
    # With 8 children and 4 matchups, `seed % 8` gives each child one matchup.
    from yomibench.cli import shard_seeds

    for m, n in ((4, 8), (48, 6), (14, 8), (96, 6)):
        for first in (0, 3, 1000, 123457):
            seeds = range(first, first + (n + 2) * m)  # a few cycles, from any seed
            for k in range(n):
                assert {s % m for s in shard_seeds(seeds, k, n, m)} == set(range(m)), (m, n, first, k)


def test_shards_play_the_same_games_whatever_the_dealing(tmp_path, monkeypatch):
    """Each `--shard K/N` child writes exactly its seeds, and the children's records together are
    the unsharded run's games, as they were with the old dealing: a game depends on its seed only."""
    from yomibench import cli

    def run(out, *extra):
        assert cli.main(["selfplay", "--engine", "toy", "--games", "22", "--first-seed", "5",
                         "--matchups", ",".join(["Toy:Toy"] * 6), "--out", str(out), "--no-ledger",
                         "--allow-sleep", *extra]) == 0

    def games(out):
        drop = {"ms", "started", "seconds", "perf"}  # timings

        def strip(x):
            if isinstance(x, dict):
                return {k: strip(v) for k, v in x.items() if k not in drop}
            return [strip(v) for v in x] if isinstance(x, list) else x
        return {int(f.name[1:13]): strip(read_record(f)) for f in out.glob("g*.jsonl.gz")}

    def sharded(out, n=4):
        for k in range(n):
            before = set(games(out))
            run(out, "--shard", f"{k}/{n}")
            assert set(games(out)) - before == set(cli.shard_seeds(range(5, 27), k, n, 6))
        return games(out)

    run(tmp_path / "whole")
    whole = games(tmp_path / "whole")
    assert sorted(whole) == list(range(5, 27))
    assert sharded(tmp_path / "new") == whole
    monkeypatch.setattr(cli, "shard_seeds", _old_shard_seeds)
    assert sharded(tmp_path / "old") == whole


def test_a_claimed_seed_is_played_by_one_process_only(tmp_path):
    """Processes sharing one seed list play only the seeds they claim; skipped counts only `owns`."""
    from yomibench.cli import _seed_claimer

    claims = tmp_path / "claims"
    claims.mkdir()
    taken = _seed_claimer(claims)
    assert taken(1) and taken(3) and not taken(1)  # another process already holds 1 and 3
    stats = {}
    written, _ = run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=20), [("Toy", "Toy")],
                              range(5), tmp_path / "out", claim=_seed_claimer(claims), stats=stats,
                              owns=lambda s: s % 2 == 0)
    assert written == 3 and sorted(int(f.name[1:13]) for f in (tmp_path / "out").glob("g*.jsonl.gz")) == [0, 2, 4]
    written, _ = run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=20), [("Toy", "Toy")],
                              range(5), tmp_path / "out", stats=stats, owns=lambda s: s % 2 == 0)
    assert written == 2 and stats["skipped"] == 3  # 0, 2 and 4 were recorded and this process owns them


def test_procs_children_share_the_seeds_and_play_each_once(tmp_path):
    """The merged ledger row counts every game a child wrote, so a seed played twice (whose second
    record would overwrite the first file) shows up as more games written than seeds."""
    from yomibench import cli

    def run(games):
        led = tmp_path / f"l{games}.jsonl"
        assert cli.main(["selfplay", "--engine", "toy", "--games", str(games), "--procs", "3", "--out", str(out),
                         "--ledger", str(led)]) == 0
        (row,) = [json.loads(x) for x in led.read_text().splitlines()]
        return row["metrics"]["games_written"], row["metrics"]["games_skipped"]

    out = tmp_path / "g"
    assert run(9) == (9, 0)
    assert sorted(int(f.name[1:13]) for f in out.glob("g*.jsonl.gz")) == list(range(9))
    assert run(12) == (3, 9)  # a rerun plays only the 3 new seeds and counts the 9 recorded ones once


def test_category_exploration_picks_only_category_moves_and_leaves_other_records_unchanged(tmp_path, monkeypatch):
    """The draw arm: with --explore-category NAME:E, a full-turn decision where the
    category's character has category candidates plays a uniform pick among them with probability E,
    marked explore_mode category:NAME. Off, records are byte-identical to before (own RNG stream)."""
    import gzip

    import pytest

    from yomibench import selfplay

    monkeypatch.setitem(selfplay.CATEGORIES, "toy-strike", ("Toy", lambda c: c.action in ("Jab", "Heavy")))
    with pytest.raises(ValueError):
        selfplay.parse_category("no-such:0.1")
    with pytest.raises(ValueError):
        selfplay.parse_category("toy-strike:0")
    assert selfplay.parse_category("toy-strike:0.5") == ("toy-strike", 0.5)

    def play(out, **kw):
        run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(4), out,
                     full_previews_fraction=0.0, explore_cfg=None, **kw)
        return sorted(out.glob("g*.jsonl.gz"))

    base = play(tmp_path / "base")
    again = play(tmp_path / "again", explore_category=None)
    def strip(f):
        with gzip.open(f, "rt") as g:
            return g.read().splitlines()

    for a, b in zip(base, again):  # off: same games (timings aside)
        ta = [json.loads(x) for x in strip(a)][1:-1]
        tb = [json.loads(x) for x in strip(b)][1:-1]
        assert [t["picks"] for t in ta] == [t["picks"] for t in tb]
        assert "explore_category" not in json.loads(strip(b)[0])

    files = play(tmp_path / "arm", explore_category=("toy-strike", 0.5))
    picked = eligible = 0
    for f in files:
        lines = [json.loads(x) for x in strip(f)]
        assert lines[0]["explore_category"] == {"name": "toy-strike", "epsilon": 0.5}
        for t in lines[1:-1]:
            if t.get("explore_mode") == "category:toy-strike":
                picked += 1
                seat = t["explore"].index(True)
                assert t["candidates"][seat][t["picks"][seat]]["action"] in ("Jab", "Heavy")
            if t.get("strategies") and any(c["action"] in ("Jab", "Heavy") for c in t["candidates"][0]):
                eligible += 1
    assert eligible > 20 and 0.25 < picked / eligible < 0.75  # about E on eligible decisions


def test_cli_explore_category_reaches_the_records(tmp_path):
    """--explore-category NAME:E goes through `yomibench.cli selfplay` (and its child processes) into
    every record's header; a character the category doesn't name is never explored."""
    import gzip

    from yomibench import cli

    out = tmp_path / "out"
    assert cli.main(["selfplay", "--engine", "toy", "--games", "2", "--out", str(out), "--bot", "matrix",
                     "--explore-epsilon", "0", "--explore-category", "cowboy-gun:0.5", "--no-ledger"]) in (0, None)
    files = sorted(out.glob("g*.jsonl.gz"))
    assert files
    for f in files:
        with gzip.open(f, "rt") as g:
            lines = [json.loads(x) for x in g.read().splitlines()]
        assert lines[0]["explore_category"] == {"name": "cowboy-gun", "epsilon": 0.5}
        assert not any(t.get("explore_mode", "").startswith("category:") for t in lines[1:-1])  # no Cowboy here


def test_uniform_moves_exploration_is_generic_and_uniform_over_moves(tmp_path):
    """The generic arm: --explore-uniform-moves E picks, on about E of full-turn decisions, a move
    uniformly among a seat's distinct actions and then one of its candidates, so a move with many
    variants (the toy's 4 Walks) is no likelier than any other. Off, games are unchanged."""
    import collections
    import gzip

    def play(out, **kw):
        run_selfplay([lambda: connect("toy")], lambda: MatrixBot(iterations=30), [("Toy", "Toy")], range(6), out,
                     full_previews_fraction=0.0, explore_cfg=None, **kw)
        files = sorted(out.glob("g*.jsonl.gz"))
        games = []
        for f in files:
            with gzip.open(f, "rt") as g:
                games.append([json.loads(x) for x in g.read().splitlines()])
        return games

    base, again = play(tmp_path / "base"), play(tmp_path / "again", explore_uniform_moves=None)
    for a, b in zip(base, again):
        assert [t["picks"] for t in a[1:-1]] == [t["picks"] for t in b[1:-1]]
        assert "explore_uniform_moves" not in b[0]

    arm = play(tmp_path / "arm", explore_uniform_moves=0.6)
    explored = eligible = 0
    moves = collections.Counter()
    for lines in arm:
        assert lines[0]["explore_uniform_moves"] == {"epsilon": 0.6}
        for t in lines[1:-1]:
            if t.get("strategies"):
                eligible += 1
            if t.get("explore_mode") == "uniform-moves":
                explored += 1
                assert sum(t["explore"]) == 1  # one seat
                seat = t["explore"].index(True)
                moves[t["candidates"][seat][t["picks"][seat]]["action"]] += 1
    assert eligible > 30 and 0.4 < explored / eligible < 0.8  # about E
    share_walk = moves["Walk"] / sum(moves.values())
    assert share_walk < 0.35  # uniform over ~6 moves (1/6), not over 9 candidates (4/9 for Walk)


def _selfplay_run(out, *extra, games=6, procs=2):
    """`yomibench selfplay` on the toy engine: (seed -> each turn's picks, its ledger row's metrics)."""
    import gzip

    from yomibench import cli

    led = out.parent / f"{out.name}.jsonl"
    assert cli.main(["selfplay", "--engine", "toy", "--games", str(games), "--procs", str(procs), "--out", str(out),
                     "--bot", "matrix", "--allow-sleep", "--ledger", str(led), *extra]) == 0
    games_by_seed = {}
    for f in out.glob("g*.jsonl.gz"):
        with gzip.open(f, "rt") as g:
            lines = [json.loads(x) for x in g.read().splitlines()]
        games_by_seed[lines[0]["seed"]] = [t["picks"] for t in lines[1:-1]]
    return games_by_seed, [json.loads(x) for x in led.read_text().splitlines()][-1]["metrics"]


def test_the_memory_cap_changes_no_game_and_no_counter_until_it_trips(tmp_path):
    """--max-proc-mem-gb off is the run as before; on with nothing over the cap (with --procs 1
    too, which then plays in a child) it plays the same games and counts them the same, plus a
    zero games_dropped_memory."""
    off, m_off = _selfplay_run(tmp_path / "off")
    on, m_on = _selfplay_run(tmp_path / "on", "--max-proc-mem-gb", "1000")
    one_off, m_one_off = _selfplay_run(tmp_path / "one-off", procs=1)
    one_on, m_one_on = _selfplay_run(tmp_path / "one-on", "--max-proc-mem-gb", "1000", procs=1)
    assert sorted(off) == list(range(6)) and off == on == one_off == one_on
    keys = ("games_written", "games_valid", "games_dropped", "games_truncated", "games_skipped", "turns",
            "searched_turns", "interrupted")
    assert {k: m_off[k] for k in keys} == {k: m_on[k] for k in keys} == {k: m_one_on[k] for k in keys}
    assert m_off["games_dropped"] == 0 and "games_dropped_memory" not in m_off and "games_dropped_memory" not in m_one_off
    assert m_on["games_dropped_memory"] == m_one_on["games_dropped_memory"] == 0
    assert not (tmp_path / "off" / "drops.jsonl").exists() and not (tmp_path / "on" / "drops.jsonl").exists()


def test_a_runaway_game_costs_that_game_not_the_run(tmp_path, runaway, capfd):
    """A child whose memory passes --max-proc-mem-gb drops the game it is playing and ends; the
    parent logs and counts the drop, a replacement child plays on, and the run ends normally with
    every other seed recorded. A rerun doesn't replay the dropped seed."""
    runaway(3)
    games, m = _selfplay_run(tmp_path / "out", "--max-proc-mem-gb", "0.5", games=8)
    assert sorted(games) == [0, 1, 2, 4, 5, 6, 7]
    assert "seed 3: dropped (memory cap 0.5 GB exceeded: 64 GB, matchup Toy:Toy)" in capfd.readouterr().out
    assert (m["games_written"], m["games_dropped"], m["games_dropped_memory"], m["interrupted"]) == (7, 1, 1, False)
    (drop,) = [json.loads(x) for x in (tmp_path / "out" / "drops.jsonl").read_text().splitlines()]
    assert (drop["seed"], drop["characters"], drop["memory_cap"], drop["memory_bytes"]) == (3, ["Toy", "Toy"], True,
                                                                                           64 << 30)
    assert not list((tmp_path / "out").glob(".g*.tmp"))  # the dropped game's unfinished record is removed

    again, m = _selfplay_run(tmp_path / "out", "--max-proc-mem-gb", "0.5", games=8)
    assert sorted(again) == sorted(games)
    assert (m["games_written"], m["games_skipped"], m["games_dropped_memory"]) == (0, 7, 1)


class _FakeChild:
    """A spawned child as _wait_replacing sees it: a pid and an exit code once finished."""

    def __init__(self, pid: int, rc: int):
        self.pid, self.rc = pid, rc

    def poll(self):
        return self.rc


def test_a_capped_child_is_replaced_when_a_launcher_sits_between_parent_and_interpreter(tmp_path):
    """On Windows a venv's python.exe is a launcher: the parent holds the launcher's pid while the
    interpreter that trips the cap writes its own pid into the drop file. The drop file's parent pid
    (the launcher) must still identify the child, so it is replaced and the run ends cleanly instead
    of reporting the cap's exit code as a crash."""
    import json as _json

    from yomibench.cli import _wait_replacing
    from yomibench.memcap import EXIT_CODE

    claim, out = tmp_path / "claims", tmp_path / "out"
    claim.mkdir(); out.mkdir()
    launcher_pid, interpreter_pid = 40024, 24924
    (claim / "dropped-7.json").write_text(_json.dumps(
        {"seed": 7, "characters": ["Toy", "Toy"], "memory_bytes": 64 << 30, "cap_gb": 0.5,
         "pid": interpreter_pid, "ppid": launcher_pid, "time": 0.0}), encoding="utf-8")
    spawned = []

    def spawn(k, gen):
        spawned.append((k, gen))
        return _FakeChild(50000 + gen, 0)

    children = [_FakeChild(launcher_pid, EXIT_CODE), _FakeChild(41000, 0)]
    drops: list[dict] = []
    assert _wait_replacing(children, spawn, claim, out, drops) == 0
    assert spawned == [(0, 1)]
    assert [d["seed"] for d in drops] == [7]


def test_a_child_exiting_with_the_cap_code_but_no_drop_file_is_still_a_failure(tmp_path):
    """The parent pid widens the match only for real drop files: an exit code without one is kept."""
    from yomibench.cli import _wait_replacing
    from yomibench.memcap import EXIT_CODE

    claim, out = tmp_path / "claims", tmp_path / "out"
    claim.mkdir(); out.mkdir()
    children = [_FakeChild(40024, EXIT_CODE)]
    assert _wait_replacing(children, lambda k, g: _FakeChild(1, 0), claim, out, []) == EXIT_CODE
