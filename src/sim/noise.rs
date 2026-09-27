//! Circularly-symmetric complex white Gaussian noise.

use std::error::Error;
use std::fmt;

use num_complex::Complex64;
use rand::Rng;
use rand::distr::Distribution;
use rand_distr::StandardNormal;

/// Circularly-symmetric complex Gaussian noise, `CN(0, P)`.
///
/// Each sample is `x = I + jQ` with `I, Q ~ N(0, P/2)` independent, so
/// `E[|x|²] = P`. After square-law detection `|x|²` is exponential with
/// mean `P`, which is the noise model every CFAR result in this crate
/// relies on.
///
/// Samples are drawn through the [`Distribution`] trait, so the caller
/// chooses the RNG.
///
/// # Example
///
/// ```
/// use cfar_bench::sim::ComplexAwgn;
/// use rand::SeedableRng;
/// use rand::distr::Distribution;
/// use rand::rngs::ChaCha8Rng;
///
/// let noise = ComplexAwgn::new(2.0).unwrap();
/// let mut rng = ChaCha8Rng::seed_from_u64(7);
/// let samples: Vec<_> = noise.sample_iter(&mut rng).take(1024).collect();
/// assert_eq!(samples.len(), 1024);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexAwgn {
    power: f64,
    component_std: f64,
}

impl ComplexAwgn {
    /// Creates a noise source with total power `power = E[|x|²]` (linear, not dB).
    ///
    /// Zero power is allowed (useful for noiseless reference runs).
    ///
    /// # Errors
    ///
    /// Returns [`NoiseError::InvalidPower`] if `power` is negative, NaN or infinite.
    pub fn new(power: f64) -> Result<Self, NoiseError> {
        // `!(power >= 0.0)` would also catch NaN, but spelling out
        // `is_finite` makes the intent obvious to the reader.
        if !power.is_finite() || power < 0.0 {
            return Err(NoiseError::InvalidPower(power));
        }
        Ok(Self {
            power,
            component_std: (power / 2.0).sqrt(),
        })
    }

    /// Total noise power `E[|x|²]`.
    #[must_use]
    pub fn power(&self) -> f64 {
        self.power
    }
}

impl Distribution<Complex64> for ComplexAwgn {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex64 {
        let i: f64 = StandardNormal.sample(rng);
        let q: f64 = StandardNormal.sample(rng);
        Complex64::new(i, q) * self.component_std
    }
}

/// Errors from constructing a noise source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseError {
    /// Power must be finite and non-negative.
    InvalidPower(f64),
}

impl fmt::Display for NoiseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPower(p) => {
                write!(f, "noise power must be finite and non-negative, got {p}")
            }
        }
    }
}

impl Error for NoiseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_negative_nan_and_infinite_power() {
        for p in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(ComplexAwgn::new(p).is_err(), "power {p} should be rejected");
        }
    }

    #[test]
    fn accepts_zero_and_positive_power() {
        assert!(ComplexAwgn::new(0.0).is_ok());
        assert!(ComplexAwgn::new(1e-12).is_ok());
        assert!(ComplexAwgn::new(1e6).is_ok());
    }

    #[test]
    fn splits_power_equally_between_components() {
        let noise = ComplexAwgn::new(8.0).unwrap();
        // 2 * sigma² must equal the total power.
        let total = 2.0 * noise.component_std.powi(2);
        assert!((total - 8.0).abs() < 1e-12);
    }
}