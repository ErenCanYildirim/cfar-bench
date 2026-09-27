//! Target echo models.

use num_complex::Complex64;
use rand::Rng;
use rand::distr::Distribution;

use super::{ComplexAwgn, NoiseError};

/// Swerling I/II target, single pulse: the echo's complex amplitude is
/// `CN(0, S)`, where `S` is the mean target power.
///
/// This models a target made of many comparable scatterers: the sum of
/// their returns is complex Gaussian (central limit theorem), so the
/// amplitude is Rayleigh and the power `|a|²` is exponential.
///
/// Swerling I and II differ only in *how often* the amplitude decorrelates
/// (scan-to-scan vs pulse-to-pulse). For a single pulse, which is all this
/// crate simulates so far, they are identical.
///
/// Statistically the echo is the same distribution as [`ComplexAwgn`] at
/// power `S`. It is a separate type anyway so that code says what it means:
/// a function taking a `SwerlingOne` cannot be handed a noise source by
/// mistake.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwerlingOne {
    amplitude: ComplexAwgn,
}

impl SwerlingOne {
    /// Target with mean echo power `mean_power` (linear).
    ///
    /// # Errors
    ///
    /// [`NoiseError::InvalidPower`] if `mean_power` is negative, NaN or
    /// infinite.
    pub fn new(mean_power: f64) -> Result<Self, NoiseError> {
        Ok(Self {
            amplitude: ComplexAwgn::new(mean_power)?,
        })
    }

    /// Target whose mean power is `snr` times `noise_power` (both linear).
    ///
    /// # Errors
    ///
    /// [`NoiseError::InvalidPower`] if the product is not a valid power.
    pub fn from_snr(snr: f64, noise_power: f64) -> Result<Self, NoiseError> {
        Self::new(snr * noise_power)
    }

    /// Mean echo power `S`.
    #[must_use]
    pub fn mean_power(&self) -> f64 {
        self.amplitude.power()
    }
}

impl Distribution<Complex64> for SwerlingOne {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Complex64 {
        self.amplitude.sample(rng)
    }
}
