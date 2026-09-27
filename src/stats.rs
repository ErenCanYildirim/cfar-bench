//! Estimators and hypothesis tests used to validate the signal processing
//! against theory.
//!
//! These functions are part of the verification layer: they measure what the
//! code under test produced. The expected values come from [`crate::theory`].

mod descriptive;
mod error;
mod ks;

pub use descriptive::{mean, pearson_correlation, standard_error_of_mean, variance};
pub use error::StatsError;
pub use ks::{KsResult, kolmogorov_survival, ks_one_sample};
