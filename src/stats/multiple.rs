//! Multiple-comparisons control. The project policy is documented on
//! [`TestFamily`] (this module is private, so its own docs are not rendered).

use super::{Confidence, StatsError};

/// A family of `size` statistical assertions sharing a family-wise error
/// rate, controlled with Bonferroni.
///
/// # Project policy
///
/// A *family* is the set of statistical assertions in one validation test
/// file. Each family declares its size `m` and a family-wise error rate
/// `α_FW`, and every assertion in it uses the Bonferroni per-test level
/// `α_FW / m`.
///
/// Default: `α_FW = 1e-3`, i.e. a correct implementation fails a given
/// validation file by chance at most once in a thousand seed choices.
///
/// Bonferroni is chosen over Šidák because it holds under **any**
/// dependence between the tests. Tests in one file often share structure
/// (same generator, overlapping parameters), so assuming independence
/// would be unjustified. The price is slight conservatism, which is
/// negligible at small α (see [`sidak_alpha`]).
///
/// Negative controls (tests that must *reject* deliberately broken input)
/// are not part of the family: their failure mode is a miss, not a false
/// alarm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TestFamily {
    family: Confidence,
    size: usize,
}

impl TestFamily {
    /// # Errors
    ///
    /// - [`StatsError::InvalidProbability`] unless `0 < family_alpha < 1`.
    /// - [`StatsError::EmptyFamily`] if `size == 0`.
    pub fn bonferroni(family_alpha: f64, size: usize) -> Result<Self, StatsError> {
        let family = Confidence::from_alpha(family_alpha)?;
        if size == 0 {
            return Err(StatsError::EmptyFamily);
        }
        Ok(Self { family, size })
    }

    /// Number of assertions in the family.
    #[must_use]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Family-wise error rate `α_FW`.
    #[must_use]
    pub fn family_alpha(&self) -> f64 {
        self.family.alpha()
    }

    /// Confidence to use for each individual assertion: `α = α_FW / m`.
    #[must_use]
    pub fn per_test(&self) -> Confidence {
        // α_FW is in (0,1) and m ≥ 1, so α_FW/m is in (0,1): cannot fail.
        Confidence::from_alpha(bonferroni_alpha(self.family.alpha(), self.size))
            .expect("α_FW / m lies in (0, 1) by construction")
    }
}

/// Bonferroni per-test level `α_FW / m`.
#[must_use]
pub fn bonferroni_alpha(family_alpha: f64, m: usize) -> f64 {
    family_alpha / m as f64
}

/// Šidák per-test level `1 - (1 - α_FW)^(1/m)`: exact for independent
/// tests. Computed via `expm1`/`ln_1p` so small α keeps full precision.
#[must_use]
pub fn sidak_alpha(family_alpha: f64, m: usize) -> f64 {
    -((-family_alpha).ln_1p() / m as f64).exp_m1()
}

/// Probability of at least one false rejection among `m` **independent**
/// tests at per-test level `α`: `1 - (1 - α)^m`.
///
/// For example, 20 independent tests at 99 % confidence fail at least once
/// with probability ≈ 18 %.
#[must_use]
pub fn family_wise_error_rate(per_test_alpha: f64, m: usize) -> f64 {
    -(m as f64 * (-per_test_alpha).ln_1p()).exp_m1()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twenty_tests_at_99_percent() {
        // 1 - 0.99^20 = 0.182093062402769...
        let fwer = family_wise_error_rate(0.01, 20);
        assert!((fwer - 0.182_093_062_402_769_23).abs() < 1e-12);
    }

    #[test]
    fn sidak_inverts_family_wise_error_rate() {
        for (a, m) in [(0.05, 20), (1e-3, 27), (1e-6, 1000)] {
            let per = sidak_alpha(a, m);
            assert!((family_wise_error_rate(per, m) - a).abs() < 1e-15 + 1e-12 * a);
        }
    }

    #[test]
    fn bonferroni_is_conservative_but_close_at_small_alpha() {
        let (a, m) = (1e-3, 27);
        let b = bonferroni_alpha(a, m);
        let s = sidak_alpha(a, m);
        assert!(b <= s);
        assert!((s - b) / s < 1e-3);
    }

    #[test]
    fn family_per_test_confidence() {
        let fam = TestFamily::bonferroni(1e-3, 20).unwrap();
        assert!((fam.per_test().alpha() - 5e-5).abs() < 1e-18);
        assert_eq!(fam.size(), 20);
    }

    #[test]
    fn rejects_invalid_family() {
        assert_eq!(
            TestFamily::bonferroni(1e-3, 0),
            Err(StatsError::EmptyFamily)
        );
        assert!(TestFamily::bonferroni(0.0, 5).is_err());
    }
}
