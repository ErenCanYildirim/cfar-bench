//! Constant false alarm rate (CFAR) detectors.
//!
//! All detectors here operate on **square-law detected** data, i.e. power
//! samples `|x|²`, and are parameterised by a threshold multiplier `α`.
//! Converting a design Pfa into `α` is a theory question and lives in
//! [`crate::theory`]; keeping it out of the detector means the validation
//! tests compare the detector against an answer key that shares no code
//! with it.
//!
//! OS-CFAR is added in PR 5.

mod ca;
mod error;
mod window;

pub use ca::{CaCfar, CfarProfile};
pub use error::CfarError;
pub use window::CfarWindow;
