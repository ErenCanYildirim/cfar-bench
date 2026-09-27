use std::error::Error;
use std::fmt;

/// Errors from statistical estimators and tests.
///
/// Only `PartialEq`, not `Eq`: some variants carry an `f64`, and `f64` is
/// not `Eq` because `NaN != NaN`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// A probability (or confidence level, or significance level) was outside
    /// its valid open interval `(0, 1)`.
    InvalidProbability(f64),
    /// A success count exceeded the number of trials, or there were no trials.
    InvalidCount {
        /// Number of successes.
        successes: u64,
        /// Number of trials.
        trials: u64,
    },
    /// A requested relative precision was not a positive finite number.
    InvalidPrecision(f64),
    /// A test family must contain at least one test.
    EmptyFamily,
    /// A numerical routine failed to converge.
    NoConvergence,
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
            Self::InvalidProbability(p) => {
                write!(f, "probability must lie strictly between 0 and 1, got {p}")
            }
            Self::InvalidCount { successes, trials } => write!(
                f,
                "invalid binomial count: {successes} successes in {trials} trials"
            ),
            Self::InvalidPrecision(r) => {
                write!(f, "relative precision must be positive and finite, got {r}")
            }
            Self::EmptyFamily => write!(f, "a test family must contain at least one test"),
            Self::NoConvergence => write!(f, "numerical routine did not converge"),
        }
    }
}

impl Error for StatsError {}
