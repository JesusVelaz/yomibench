"""The regret-matching solver: native extension (if installed) vs the pure-Python reference,
seat symmetry, certified exploitability on every solve, and run-to-run determinism."""

import random
import sys

import pytest

from yomibench import search
from yomibench.search import _alternating_py, double_oracle, exploitability, solve_both


def _random_matrix(rng, r, c, kind):
    if kind == 0:
        return [[rng.choice([0, 1, -1]) for _ in range(c)] for _ in range(r)]
    if kind == 1:
        return [[rng.choice([0.0, rng.uniform(-1, 1)]) for _ in range(c)] for _ in range(r)]
    return [[round(rng.uniform(-1, 1), 5) for _ in range(c)] for _ in range(r)]


@pytest.mark.skipif(search._native is None, reason="yomisolve (native/yomisolve) not installed")
@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python < 3.12 sums floats without compensation")
def test_native_matches_the_python_reference_bit_for_bit():
    rng = random.Random(1)
    for trial in range(200):
        m = _random_matrix(rng, rng.randint(1, 9), rng.randint(1, 9), trial % 3)
        its = rng.choice([1, 2, 50, 700])
        assert search._native(m, its) == _alternating_py(m, its), (m, its)


def test_solve_both_is_exactly_seat_symmetric_with_the_active_solver():
    rng = random.Random(2)
    for _ in range(30):
        m = _random_matrix(rng, rng.randint(1, 8), rng.randint(1, 8), 1)
        mt = [[-m[i][j] for i in range(len(m))] for j in range(len(m[0]))]
        r1, c1, _ = solve_both(m, 400)
        r2, c2, _ = solve_both(mt, 400)
        assert (r1, c1) == (c2, r2)


def test_every_double_oracle_solve_is_certified():
    rng = random.Random(3)
    for trial in range(40):
        n0, n1 = rng.randint(2, 30), rng.randint(2, 30)
        m = _random_matrix(rng, n0, n1, 1 + trial % 2)
        r = double_oracle(n0, n1, lambda ps: [m[i][j] for i, j in ps], start=([0, 1], [0]))
        assert r.converged and max(exploitability(m, r.row, r.col)) <= 1e-3 + 1e-9


def test_self_play_is_deterministic_run_to_run():
    from yomibench.bots import MatrixBot
    from yomibench.engine import connect
    from yomibench.selfplay import play_recorded

    def picks():
        e = connect("toy")
        lines = play_recorded(e, MatrixBot(iterations=500, search="oracle"), ("Toy", "Toy"), 9, e.hello(),
                              max_turns=25)
        assert lines[0]["solver"] == search.SOLVER and lines[0]["bots"][0]["solver"] == search.SOLVER
        return [(ln["picks"], ln["strategies"]) for ln in lines[1:-1]]

    assert picks() == picks()
