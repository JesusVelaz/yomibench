"""PoolEngine (option D): N engines holding one match answer exactly as one engine does."""
from yomibench.engine import Engine, InProcess
from yomibench.pool_engine import PoolEngine, auto_size
from yomibench.toy import ToyEngine


def toy():
    return Engine(InProcess(ToyEngine()))


def test_pool_answers_like_one_engine():
    one, pool = toy(), PoolEngine([toy(), toy(), toy()])
    a, b = one.new_match("A", "B", 7), pool.new_match("A", "B", 7)
    assert a.state == b.state and a.candidates == b.candidates
    for _ in range(4):
        n0, n1 = len(a.candidates[0]), len(a.candidates[1])
        pairs = [(i, j) for i in range(n0) for j in range(n1)]
        pa, pb = one.preview(pairs), pool.preview(pairs)
        assert [o.state for o in pa] == [o.state for o in pb]  # split over 3 servers, joined in order
        assert len({o.handle for o in pb}) == len(pb)             # pool handles are unique
        # continue_from reaches the server holding the handle (the last pairs live on server 2)
        ca, cb = one.continue_from(pa[-1].handle), pool.continue_from(pb[-1].handle)
        assert ca.state == cb.state
        if a.done:
            break
        a, b = one.step(0, n1 - 1), pool.step(0, n1 - 1)
        assert a.state == b.state
    assert pool.stats["preview_pairs"] == one.stats["preview_pairs"]


def test_pool_settings_reach_every_server():
    pool = PoolEngine([toy(), toy()])
    pool.candidate_set = 5
    pool.match_settings = {"stage_width": 600}
    assert all(e.candidate_set == 5 and e.match_settings == {"stage_width": 600} for e in pool.engines)
    assert pool.reusable and pool.hello()["engine_pool"] == 2


def test_auto_size_follows_cores_and_memory():
    assert auto_size(cores=28, free_mb=16000) == 4          # a big PC: capped at 4
    assert auto_size(cores=8, free_mb=16000) == 3           # (8 - 2) // 2
    assert auto_size(cores=4, free_mb=16000) == 1
    assert auto_size(cores=28, free_mb=1024 + 2 * 300) == 2  # memory-bound, 1 GB kept back
    assert auto_size(cores=2, free_mb=500) == 1              # never below one


def test_connect_server_retries_a_busy_refusal(monkeypatch):
    """A server refuses a second client with `busy` until it notices wait_ready's probe closed: retry, then connect."""
    import yomibench.engine as eng
    from yomibench import game

    calls = []

    class FakeTcp:
        def __init__(self, host, port):
            calls.append(port)

        def request(self, msg):
            if len(calls) < 3:
                return {"error": {"code": "busy", "message": "this server already has a client"}}
            return {"cmd": "hello", "candidate_set": "5"}

        def close(self):
            pass

    monkeypatch.setattr(eng, "Tcp", FakeTcp)
    monkeypatch.setattr(game.time, "sleep", lambda s: None)
    e = game.connect_server(7777)
    assert len(calls) == 3 and e.hello()["candidate_set"] == "5"
