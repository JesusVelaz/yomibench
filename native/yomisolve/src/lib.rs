//! One alternating regret matching+ run (row player first, linear averaging), exactly as
//! `yomibench.search._alternating`: the same operations in the same order, f64 throughout, no FMA
//! contraction or reassociation, and every `sum()` done the way CPython 3.12+ sums floats
//! (Neumaier compensated summation, `Sum` below). So results are bit-identical to the pure-Python
//! solver on Python 3.12+, deterministic, and the seat symmetry of `solve_both` (which runs this
//! on the payoff and on its negated transpose) carries over unchanged.

use pyo3::prelude::*;

/// CPython 3.12+ `sum()` of floats: starts from int 0 (so the first term becomes 0 + x), then
/// Neumaier's improved Kahan-Babuska compensation, adding the compensation at the end if finite.
struct Sum {
    f: f64,
    c: f64,
    started: bool,
}

impl Sum {
    #[inline]
    fn new() -> Self {
        Sum { f: 0.0, c: 0.0, started: false }
    }
    #[inline]
    fn add(&mut self, x: f64) {
        if !self.started {
            self.f = 0.0 + x;
            self.started = true;
            return;
        }
        let t = self.f + x;
        if self.f.abs() >= x.abs() {
            self.c += (self.f - t) + x;
        } else {
            self.c += (x - t) + self.f;
        }
        self.f = t;
    }
    #[inline]
    fn result(&self) -> f64 {
        if self.c != 0.0 && self.c.is_finite() {
            self.f + self.c
        } else {
            self.f
        }
    }
}

fn regret_match(regret: &[f64], out: &mut [f64]) {
    let mut acc = Sum::new();
    for &r in regret {
        acc.add(r);
    }
    let total = acc.result();
    if total <= 0.0 {
        let u = 1.0 / (regret.len() as f64);
        for o in out.iter_mut() {
            *o = u;
        }
    } else {
        for (o, &r) in out.iter_mut().zip(regret) {
            *o = r / total;
        }
    }
}

#[inline]
fn plus(x: f64) -> f64 {
    // Python's max(0.0, x): 0.0 unless x > 0 (also for x == -0.0)
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// alternating(payoff, iterations) -> (row strategy, column strategy)
#[pyfunction]
fn alternating(payoff: Vec<Vec<f64>>, iterations: usize) -> PyResult<(Vec<f64>, Vec<f64>)> {
    let rows = payoff.len();
    if rows == 0 || payoff[0].is_empty() || payoff.iter().any(|r| r.len() != payoff[0].len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("payoff must be a non-empty rectangular matrix"));
    }
    let cols = payoff[0].len();
    let mut regret_r = vec![0.0f64; rows];
    let mut regret_c = vec![0.0f64; cols];
    let mut avg_r = vec![0.0f64; rows];
    let mut avg_c = vec![0.0f64; cols];
    let mut sr = vec![0.0f64; rows];
    let mut sc = vec![0.0f64; cols];
    let mut row_vals = vec![0.0f64; rows];
    let mut col_vals = vec![0.0f64; cols];
    regret_match(&regret_c, &mut sc);
    for t in 1..=iterations {
        for i in 0..rows {
            let mut acc = Sum::new();
            for j in 0..cols {
                acc.add(payoff[i][j] * sc[j]);
            }
            row_vals[i] = acc.result();
        }
        regret_match(&regret_r, &mut sr);
        let mut acc = Sum::new();
        for i in 0..rows {
            acc.add(sr[i] * row_vals[i]);
        }
        let ev = acc.result();
        for i in 0..rows {
            regret_r[i] = plus(regret_r[i] + row_vals[i] - ev);
        }
        regret_match(&regret_r, &mut sr);
        for j in 0..cols {
            let mut acc = Sum::new();
            for i in 0..rows {
                acc.add(payoff[i][j] * sr[i]);
            }
            col_vals[j] = -acc.result();
        }
        let mut acc = Sum::new();
        for j in 0..cols {
            acc.add(sc[j] * col_vals[j]);
        }
        let ev = acc.result();
        for j in 0..cols {
            regret_c[j] = plus(regret_c[j] + col_vals[j] - ev);
        }
        regret_match(&regret_c, &mut sc);
        let tf = t as f64;
        for i in 0..rows {
            avg_r[i] += tf * sr[i];
        }
        for j in 0..cols {
            avg_c[j] += tf * sc[j];
        }
    }
    let total = |v: &[f64]| {
        let mut acc = Sum::new();
        for &x in v {
            acc.add(x);
        }
        acc.result()
    };
    let (total_r, total_c) = (total(&avg_r), total(&avg_c));
    Ok((avg_r.iter().map(|a| a / total_r).collect(), avg_c.iter().map(|a| a / total_c).collect()))
}

#[pymodule]
fn yomisolve(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(alternating, m)?)?;
    Ok(())
}
