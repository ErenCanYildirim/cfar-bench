use std::error::Error;
use std::fmt;

/// Errors from statistical estimators and tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsError {
    /// The estimator needs at least `needed` samples.
    InsufficientSamples {
        /// Minimum number of samples required.
        needed: usize,
        /// Number of samples provided.
        got: usize,
    },
    /// Two paired inputs have different lengths.
    LengthMismatch {
        /// Length of the first input.
        left: usize,
        /// Length of the second input.
        right: usize,
    },
    /// An input contained NaN or ±∞.
    NonFinite,
    /// A quantity that is divided by was exactly zero (e.g. zero variance in a
    /// correlation).
    ZeroVariance,
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientSamples { needed, got } => {
                write!(f, "need at least {needed} samples, got {got}")
            }
            Self::LengthMismatch { left, right } => {
                write!(f, "paired inputs differ in length ({left} vs {right})")
            }
            Self::NonFinite => write!(f, "input contains NaN or infinite values"),
            Self::ZeroVariance => write!(f, "input has zero variance"),
        }
    }
}

impl Error for StatsError {}
