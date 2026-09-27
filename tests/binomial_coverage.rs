//! Exact coverage of the binomial confidence intervals.
//!
//! Coverage is `C(p) = P_p(interval(X) ∋ p) = Σ_k pmf(k; n, p) · 1[p ∈ CI(k)]`.
//! For moderate `n` this sum can be evaluated exactly, so these tests are
//! deterministic proofs about the interval procedures, not Monte Carlo
//! estimates.
//!
//! This is the property the CFAR validation relies on: if Clopper–Pearson
//! coverage is ≥ 1 - α for every p, a correct detector fails a Pfa
//! assertion with probability ≤ α.

use cfar_bench::stats::{BinomialCount, Confidence, Interval};

/// Binomial pmf for all k, via the recurrence
/// `pmf(k+1) = pmf(k) · (n-k)/(k+1) · p/(1-p)`. Independent of the special
/// functions used to build the intervals.
fn binomial_pmf(n: u64, p: f64) -> Vec<f64> {
    let mut pmf = Vec::with_capacity(n as usize + 1);
    pmf.push((1.0 - p).powf(n as f64));
    for k in 0..n {
        let next = pmf[k as usize] * (n - k) as f64 / (k + 1) as f64 * p / (1.0 - p);
        pmf.push(next);
    }
    pmf
}

fn coverage(intervals: &[Interval], p: f64) -> f64 {
    let n = intervals.len() as u64 - 1;
    binomial_pmf(n, p)
        .iter()
        .zip(intervals)
        .filter(|(_, ci)| ci.contains(p))
        .map(|(mass, _)| mass)
        .sum()
}

/// A dense grid of p over (0, 1), avoiding the endpoints.
fn p_grid() -> impl Iterator<Item = f64> {
    (1..1000).map(|i| f64::from(i) / 1000.0)
}

const N: u64 = 40;

fn all_intervals<F>(n: u64, make: F) -> Vec<Interval>
where
    F: Fn(BinomialCount) -> Interval,
{
    (0..=n)
        .map(|k| make(BinomialCount::new(k, n).unwrap()))
        .collect()
}

#[test]
fn clopper_pearson_coverage_is_never_below_nominal() {
    for level in [0.9, 0.95, 0.99] {
        let conf = Confidence::from_level(level).unwrap();
        let cis = all_intervals(N, |c| c.clopper_pearson(conf).unwrap());

        let worst = p_grid()
            .map(|p| coverage(&cis, p))
            .fold(f64::INFINITY, f64::min);
        // Tolerance only for floating-point summation of the pmf.
        assert!(
            worst >= level - 1e-12,
            "level {level}: minimum coverage {worst}"
        );
    }
}

#[test]
fn wilson_coverage_is_near_nominal_on_average_but_not_guaranteed() {
    // Documents *why* the pass/fail tests use Clopper–Pearson: Wilson's
    // average coverage is close to nominal, but it dips below for some p.
    let level = 0.95;
    let conf = Confidence::from_level(level).unwrap();
    let cis = all_intervals(N, |c| c.wilson(conf));

    let covs: Vec<f64> = p_grid().map(|p| coverage(&cis, p)).collect();
    let mean = covs.iter().sum::<f64>() / covs.len() as f64;
    let worst = covs.iter().copied().fold(f64::INFINITY, f64::min);

    assert!((mean - level).abs() < 0.01, "mean coverage {mean}");
    assert!(worst < level, "expected a dip below nominal, min {worst}");
}

#[test]
fn clopper_pearson_is_wider_than_wilson_everywhere() {
    let conf = Confidence::from_level(0.95).unwrap();
    for k in 0..=N {
        let c = BinomialCount::new(k, N).unwrap();
        assert!(c.clopper_pearson(conf).unwrap().width() >= c.wilson(conf).width());
    }
}
