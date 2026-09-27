//! Special functions needed by the statistics and theory layers.
//!
//! Conventions follow `f64`'s own methods: out-of-domain inputs return NaN
//! rather than panicking or returning `Result`, so these compose like
//! `sqrt` or `ln`. Callers that expose a public API check for NaN and turn
//! it into a typed error.
//!
//! Each function is tested against independent reference values (SciPy)
//! and against identities that hold exactly.

mod beta;
mod gamma;
mod normal;

pub use beta::regularized_incomplete_beta;
pub use gamma::ln_gamma;
pub use normal::normal_quantile;
