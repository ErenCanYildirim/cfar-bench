//! Fixed-threshold detector (known noise power).

use super::CfarError;

/// Detector with a fixed threshold: declares a detection when `CUT > T`.
///
/// This is **not** a CFAR detector. It is the benchmark CFAR is measured
/// against: if the noise power were known exactly, a fixed threshold would
/// be optimal, and the SNR that CA-CFAR needs *on top* of it for the same
/// Pd is the CFAR loss.
///
/// Use [`crate::theory::known_noise::threshold_for_pfa`] to set `T` for a
/// design Pfa and a known noise power.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedThreshold {
    threshold: f64,
}

impl FixedThreshold {
    /// # Errors
    ///
    /// [`CfarError::InvalidThreshold`] unless `threshold` is positive and
    /// finite.
    pub fn new(threshold: f64) -> Result<Self, CfarError> {
        if !(threshold.is_finite() && threshold > 0.0) {
            return Err(CfarError::InvalidThreshold(threshold));
        }
        Ok(Self { threshold })
    }

    /// The threshold `T`.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Decision for a single square-law detected cell.
    #[must_use]
    pub fn detects(&self, cut: f64) -> bool {
        cut > self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_comparison() {
        let d = FixedThreshold::new(2.0).unwrap();
        assert!(d.detects(2.1));
        assert!(!d.detects(2.0));
    }

    #[test]
    fn rejects_invalid_threshold() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(FixedThreshold::new(bad).is_err());
        }
    }
}
