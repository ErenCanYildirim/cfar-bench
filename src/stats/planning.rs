//! Monte Carlo trial-count planning.
//!
//! The number of trials needed to measure a probability `p` scales like
//! `1/p`, which dominates the runtime of every Pfa experiment. These helpers
//! answer "how many trials do I need?" before running anything.
//!
//! They use the normal approximation to the binomial, which is fine for
//! *planning* (it is accurate once `n·p` is more than a few tens, which
//! any useful plan satisfies). The pass/fail assertions themselves use the
//! exact intervals in [`super::BinomialCount`].

use super::{Confidence, StatsError};

fn check_probability(p: f64) -> Result<(), StatsError> {
    if p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(StatsError::InvalidProbability(p))
    }
}

/// Relative standard error of the estimate `k/n` of a probability `p`:
/// `√((1-p) / (n·p))`.
///
/// # Errors
///
/// - [`StatsError::InvalidProbability`] unless `0 < p < 1`.
/// - [`StatsError::InvalidCount`] if `trials == 0`.
pub fn relative_standard_error(p: f64, trials: u64) -> Result<f64, StatsError> {
    check_probability(p)?;
    if trials == 0 {
        return Err(StatsError::InvalidCount {
            successes: 0,
            trials,
        });
    }
    Ok(((1.0 - p) / (trials as f64 * p)).sqrt())
}

/// Number of independent trials needed so that the two-sided confidence
/// interval on `p` has relative half-width `relative_half_width`:
///
/// `n = z² (1 - p) / (p r²)`, rounded up.
///
/// Note the confidence level matters a lot: at `p = 1e-4` and ±10 %,
/// one standard error (z = 1) needs 1.0e6 trials, 95 % (z = 1.96) needs
/// 3.8e6, and 99 % (z = 2.58) needs 6.6e6.
///
/// # Errors
///
/// - [`StatsError::InvalidProbability`] unless `0 < p < 1`.
/// - [`StatsError::InvalidPrecision`] unless `relative_half_width` is
///   positive and finite.
pub fn trials_for_relative_precision(
    p: f64,
    relative_half_width: f64,
    confidence: Confidence,
) -> Result<u64, StatsError> {
    check_probability(p)?;
    if !(relative_half_width > 0.0 && relative_half_width.is_finite()) {
        return Err(StatsError::InvalidPrecision(relative_half_width));
    }
    let z = confidence.two_sided_z();
    let n = z * z * (1.0 - p) / (p * relative_half_width * relative_half_width);
    // Float-to-int `as` casts saturate in Rust (no UB, no wrap-around), so an
    // absurd request yields u64::MAX rather than garbage.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    Ok(n.ceil() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_standard_error_at_pfa_1e4() {
        // √((1 - 1e-4) / 100) ≈ 0.099995.
        let rse = relative_standard_error(1e-4, 1_000_000).unwrap();
        assert!((rse - 0.099_995).abs() < 1e-6);
    }

    #[test]
    fn trial_counts_for_pfa_1e4_at_ten_percent() {
        let n_at = |level: f64| {
            trials_for_relative_precision(1e-4, 0.1, Confidence::from_level(level).unwrap())
                .unwrap()
        };
        // 1σ ≈ 68.27 %.
        let one_sigma = n_at(0.682_689_492_137_085_9);
        assert!((one_sigma as f64 - 999_900.0).abs() < 2.0, "{one_sigma}");
        // z² (1-p) / (p r²) with z = 1.959964: 3,841,074.
        let n95 = n_at(0.95);
        assert!((n95 as f64 - 3_841_074.0).abs() < 2.0, "{n95}");
    }

    #[test]
    fn plan_agrees_with_exact_interval() {
        // Cross-check the normal-approximation plan against the exact
        // Clopper–Pearson interval at the planned n: the relative half-width
        // should come out close to what was asked for.
        use super::super::BinomialCount;

        let (p, r) = (1e-4, 0.1);
        let conf = Confidence::from_level(0.95).unwrap();
        let n = trials_for_relative_precision(p, r, conf).unwrap();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = (p * n as f64).round() as u64;
        let ci = BinomialCount::new(k, n)
            .unwrap()
            .clopper_pearson(conf)
            .unwrap();
        let achieved = ci.width() / 2.0 / p;
        // CP is slightly wider (conservative, and skewed at k ≈ 384).
        assert!(achieved > r && achieved < 1.05 * r, "achieved {achieved}");
    }

    #[test]
    fn rejects_bad_arguments() {
        let c = Confidence::from_level(0.95).unwrap();
        assert!(trials_for_relative_precision(0.0, 0.1, c).is_err());
        assert!(trials_for_relative_precision(1e-4, 0.0, c).is_err());
        assert!(trials_for_relative_precision(1e-4, f64::INFINITY, c).is_err());
        assert!(relative_standard_error(1e-4, 0).is_err());
    }
}
