//! Confidence intervals for a binomial proportion.
//!
//! This is what every Pfa and Pd measurement reduces to: `k` detections in
//! `n` independent trials, and the question "is the true probability
//! consistent with theory?".
//!
//! # Why not the normal ("Wald") interval
//!
//! `p̂ ± z·√(p̂(1-p̂)/n)` assumes `k` is approximately normal. At
//! `Pfa = 1e-4` with a modest `n`, `k` is a handful of counts from a very
//! skewed distribution: the Wald interval is symmetric when the truth is
//! not, can extend below zero, and collapses to zero width at `k = 0`.

use super::{Confidence, StatsError};
use crate::special::regularized_incomplete_beta;

/// Number of successes in a number of independent Bernoulli trials.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinomialCount {
    successes: u64,
    trials: u64,
}

/// A closed interval `[lower, upper]` for a probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

impl Interval {
    /// Whether `p` lies inside the interval (bounds included).
    #[must_use]
    pub fn contains(&self, p: f64) -> bool {
        self.lower <= p && p <= self.upper
    }

    /// `upper - lower`.
    #[must_use]
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}

impl BinomialCount {
    /// # Errors
    ///
    /// [`StatsError::InvalidCount`] if `trials == 0` or `successes > trials`.
    pub fn new(successes: u64, trials: u64) -> Result<Self, StatsError> {
        if trials == 0 || successes > trials {
            return Err(StatsError::InvalidCount { successes, trials });
        }
        Ok(Self { successes, trials })
    }

    /// Number of successes `k`.
    #[must_use]
    pub fn successes(&self) -> u64 {
        self.successes
    }

    /// Number of trials `n`.
    #[must_use]
    pub fn trials(&self) -> u64 {
        self.trials
    }

    /// Point estimate `k / n`.
    #[must_use]
    pub fn proportion(&self) -> f64 {
        self.successes as f64 / self.trials as f64
    }

    /// Exact (Clopper–Pearson) two-sided interval.
    ///
    /// The bounds are the values of `p` at which observing `k` would just be
    /// rejected by a one-sided test at level `α/2`:
    ///
    /// - lower: `P(X ≥ k | p) = α/2`, i.e. `I_p(k, n-k+1) = α/2`
    /// - upper: `P(X ≤ k | p) = α/2`, i.e. `I_p(k+1, n-k) = 1 - α/2`
    ///
    /// Coverage is **guaranteed** to be at least `1 - α` for every `p`
    /// (it is conservative). That makes it the right choice for a pass/fail
    /// validation test: the false-failure rate is bounded by `α`.
    ///
    /// # Errors
    ///
    /// [`StatsError::NoConvergence`] if the root finder fails (not expected
    /// for valid inputs).
    pub fn clopper_pearson(&self, confidence: Confidence) -> Result<Interval, StatsError> {
        let k = self.successes as f64;
        let n = self.trials as f64;
        let half_alpha = confidence.alpha() / 2.0;

        let lower = if self.successes == 0 {
            0.0
        } else {
            solve_increasing(
                |p| regularized_incomplete_beta(p, k, n - k + 1.0),
                half_alpha,
            )?
        };
        let upper = if self.successes == self.trials {
            1.0
        } else {
            solve_increasing(
                |p| regularized_incomplete_beta(p, k + 1.0, n - k),
                1.0 - half_alpha,
            )?
        };
        Ok(Interval { lower, upper })
    }

    /// Wilson score interval.
    ///
    /// Obtained by inverting the score test, which has a closed form. Its
    /// *average* coverage is close to nominal and it is narrower than
    /// Clopper–Pearson, but coverage dips below `1 - α` for some `p`. Useful
    /// for reporting; the pass/fail tests in this crate use Clopper–Pearson.
    #[must_use]
    pub fn wilson(&self, confidence: Confidence) -> Interval {
        let k = self.successes as f64;
        let n = self.trials as f64;
        let z = confidence.two_sided_z();
        let z2 = z * z;

        let center = (k + z2 / 2.0) / (n + z2);
        let half = z / (n + z2) * (k * (n - k) / n + z2 / 4.0).sqrt();
        Interval {
            lower: (center - half).max(0.0),
            upper: (center + half).min(1.0),
        }
    }
}

/// Finds `p ∈ (0, 1)` with `f(p) = target`, for `f` increasing in `p`.
///
/// Plain bisection: slower than Newton, but it cannot diverge, and each step
/// is cheap relative to the Monte Carlo runs that produce the counts. The
/// stopping rule is *relative* to `p`, because Pfa values span many decades
/// and an absolute tolerance of, say, 1e-12 would be useless at p = 1e-13.
fn solve_increasing<F>(f: F, target: f64) -> Result<f64, StatsError>
where
    F: Fn(f64) -> f64,
{
    const MAX_ITER: usize = 2_000;
    const REL_TOL: f64 = 1e-13;

    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..MAX_ITER {
        let mid = 0.5 * (lo + hi);
        let value = f(mid);
        if value.is_nan() {
            return Err(StatsError::NoConvergence);
        }
        if value < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= REL_TOL * hi {
            return Ok(0.5 * (lo + hi));
        }
    }
    Err(StatsError::NoConvergence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conf(level: f64) -> Confidence {
        Confidence::from_level(level).unwrap()
    }

    fn assert_rel(got: f64, expected: f64, tol: f64) {
        assert!(
            ((got - expected) / expected).abs() < tol,
            "{got} vs {expected}"
        );
    }

    #[test]
    fn clopper_pearson_matches_scipy() {
        // Reference: scipy.stats.beta.ppf, i.e. the textbook definition.
        let cases = [
            (
                5,
                10,
                0.95,
                0.187_086_028_447_398_55,
                0.812_913_971_552_601_5,
            ),
            (
                3,
                1_000,
                0.95,
                6.190_999_316_495_715e-4,
                8.742_023_238_478_303e-3,
            ),
            // Pfa-regime: 100 false alarms in 1e6 cells.
            (
                100,
                1_000_000,
                0.99,
                7.612_136_667_211_263e-5,
                1.287_587_284_781_443e-4,
            ),
        ];
        for (k, n, level, lo, hi) in cases {
            let ci = BinomialCount::new(k, n)
                .unwrap()
                .clopper_pearson(conf(level))
                .unwrap();
            assert_rel(ci.lower, lo, 1e-7);
            assert_rel(ci.upper, hi, 1e-7);
        }
    }

    #[test]
    fn clopper_pearson_zero_and_all_successes_have_closed_forms() {
        // k = 0: upper = 1 - (α/2)^(1/n). k = n: lower = (α/2)^(1/n).
        let (n, level) = (50_u64, 0.95);
        let half_alpha: f64 = (1.0 - level) / 2.0;
        let edge = half_alpha.powf(1.0 / n as f64);

        let zero = BinomialCount::new(0, n)
            .unwrap()
            .clopper_pearson(conf(level))
            .unwrap();
        assert!(zero.lower.abs() < f64::EPSILON);
        assert_rel(zero.upper, 1.0 - edge, 1e-10);

        let all = BinomialCount::new(n, n)
            .unwrap()
            .clopper_pearson(conf(level))
            .unwrap();
        assert_rel(all.lower, edge, 1e-10);
        assert!((all.upper - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn wilson_matches_closed_form_example() {
        // k = 5, n = 10, 95 %. Reference:
        // scipy.stats.binomtest(5, 10).proportion_ci(0.95, "wilson").
        let ci = BinomialCount::new(5, 10).unwrap().wilson(conf(0.95));
        assert!((ci.lower - 0.236_593_090_512_564).abs() < 1e-12);
        assert!((ci.upper - 0.763_406_909_487_436_1).abs() < 1e-12);
    }

    #[test]
    fn wilson_is_narrower_than_clopper_pearson() {
        let c = BinomialCount::new(100, 1_000_000).unwrap();
        let cp = c.clopper_pearson(conf(0.99)).unwrap();
        let w = c.wilson(conf(0.99));
        assert!(w.width() < cp.width());
    }

    #[test]
    fn rejects_invalid_counts() {
        assert!(BinomialCount::new(0, 0).is_err());
        assert!(BinomialCount::new(11, 10).is_err());
    }
}
