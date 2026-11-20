# yomisolve

Optional native regret matching+ for `yomibench.search` (`_alternating`, the inner loop of
`solve_both` and every double-oracle restricted solve). It is the same arithmetic as the
pure-Python `search._alternating_py` (f64, same operation order, CPython 3.12+'s compensated float
`sum()`), so solves are **bit-identical** to the pure-Python solver on Python 3.12+, deterministic,
and keep `solve_both`'s exact seat symmetry. On the small tables the double oracle solves
(3x3 to 7x7) it is 30-130x faster.

Install it into the environment that runs yomibench (needs a Rust toolchain):

```
uv pip install ./native/yomisolve        # or: cd native/yomisolve && maturin develop --release
python -c "import yomibench.search as s; print(s.SOLVER)"   # native
```

Without it, yomibench uses the pure-Python loop. `YOMIBENCH_SOLVER=python` forces the pure-Python
loop even when the extension is installed. Bot configs in records name the solver (`solver`).
