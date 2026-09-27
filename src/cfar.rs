//! Constant false alarm rate (CFAR) detectors.
//!
//! All detectors here operate on **square-law detected** data, i.e. power
//! samples `|x|²`, and are parameterised by a threshold multiplier `α`.
//! Converting a design Pfa into `α` is a theory question and lives in
//! [`crate::theory`]; keeping it out of the detector means the validation
//! tests compare the detector against an answer key that shares no code
//! with it.
//!
//! - [`CaCfar`]: cell averaging, optimal in homogeneous noise.
//! - [`OsCfar`]: ordered statistic, robust to outliers in the window.
//! - [`FixedThreshold`]: the known-noise benchmark CFAR loss is measured
//!   against.
//!
//! All windowed detectors share [`CfarWindow`] and its edge policy
//! (documented on [`CfarProfile`]).

mod ca;
mod error;
mod fixed;
mod os;
mod profile;
mod window;

pub use ca::CaCfar;
pub use error::CfarError;
pub use fixed::FixedThreshold;
pub use os::OsCfar;
pub use profile::CfarProfile;
pub use window::CfarWindow;
