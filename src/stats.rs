//! Estimators and hypothesis tests used to validate the signal processing
//! against theory.
//!
//! These functions are part of the verification layer: they measure what the
//! code under test produced. The expected values come from [`crate::theory`].
//!
//! The multiple-comparisons policy every validation test follows is
//! documented on [`TestFamily`].

mod binomial;
mod confidence;
mod descriptive;
mod error;
mod ks;
mod multiple;
mod planning;

pub use binomial::{BinomialCount, Interval};
pub use confidence::Confidence;
pub use descriptive::{mean, pearson_correlation, standard_error_of_mean, variance};
pub use error::StatsError;
pub use ks::{KsResult, kolmogorov_survival, ks_one_sample};
pub use multiple::{TestFamily, bonferroni_alpha, family_wise_error_rate, sidak_alpha};
pub use planning::{relative_standard_error, trials_for_relative_precision};
