//! A validated confidence level.

use super::StatsError;
use crate::special::normal_quantile;

/// A confidence level `1 - α`, stored as the significance level `α`.
///
/// Why a dedicated type instead of a bare `f64`:
///
/// - **No α / 1-α mix-ups.** A function taking `Confidence` cannot be handed
///   `0.01` when it meant `0.99`; the caller must say which one they mean
///   via [`Confidence::from_level`] or [`Confidence::from_alpha`].
/// - **Precision.** Storing `α` rather than `1 - α` keeps full relative
///   precision for the small α values that multiple-comparison corrections
///   produce (e.g. `α = 1e-6`, where `1 - α` has already lost ~6 digits).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Confidence {
    alpha: f64,
}

impl Confidence {
    /// From a confidence level, e.g. `0.99` for 99 %.
    ///
    /// # Errors
    ///
    /// [`StatsError::InvalidProbability`] unless `0 < level < 1`.
    pub fn from_level(level: f64) -> Result<Self, StatsError> {
        if !(level > 0.0 && level < 1.0) {
            return Err(StatsError::InvalidProbability(level));
        }
        Ok(Self { alpha: 1.0 - level })
    }

    /// From a significance level, e.g. `0.01` for 99 % confidence.
    ///
    /// # Errors
    ///
    /// [`StatsError::InvalidProbability`] unless `0 < alpha < 1`.
    pub fn from_alpha(alpha: f64) -> Result<Self, StatsError> {
        if !(alpha > 0.0 && alpha < 1.0) {
            return Err(StatsError::InvalidProbability(alpha));
        }
        Ok(Self { alpha })
    }

    /// Significance level `α`.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Confidence level `1 - α`.
    #[must_use]
    pub fn level(&self) -> f64 {
        1.0 - self.alpha
    }

    /// Two-sided normal critical value `z` with `P(|Z| > z) = α`.
    ///
    /// Computed as `-Φ⁻¹(α/2)` rather than `Φ⁻¹(1 - α/2)`: forming
    /// `1 - α/2` in floating point would round away the digits of a small α.
    #[must_use]
    pub fn two_sided_z(&self) -> f64 {
        -normal_quantile(self.alpha / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_and_alpha_round_trip() {
        let c = Confidence::from_level(0.95).unwrap();
        assert!((c.alpha() - 0.05).abs() < 1e-15);
        assert!((c.level() - 0.95).abs() < 1e-15);
    }

    #[test]
    fn familiar_critical_values() {
        // References are -scipy.special.ndtri(α/2). Note: ndtri(1 - α/2)
        // gives 3.2905267314919255 for α = 0.001, which differs in the 14th
        // digit because 0.9995 is not exactly representable. That rounding is
        // exactly what `two_sided_z` avoids.
        let z = |a| Confidence::from_alpha(a).unwrap().two_sided_z();
        assert!((z(0.05) - 1.959_963_984_540_054_5).abs() < 1e-14);
        assert!((z(0.01) - 2.575_829_303_548_901).abs() < 1e-14);
        assert!((z(0.001) - 3.290_526_731_491_894_5).abs() < 1e-14);
    }

    #[test]
    fn small_alpha_keeps_precision() {
        // z for α = 1e-12 is about 7.13. With the naive 1 - α/2 route the
        // input would already be rounded to ~4 significant digits of α.
        let z = Confidence::from_alpha(1e-12).unwrap().two_sided_z();
        assert!(z > 7.0 && z < 7.2);
    }

    #[test]
    fn rejects_out_of_range() {
        for bad in [0.0, 1.0, -0.1, 1.5, f64::NAN] {
            assert!(Confidence::from_level(bad).is_err());
            assert!(Confidence::from_alpha(bad).is_err());
        }
    }
}
