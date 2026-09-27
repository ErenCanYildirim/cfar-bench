use std::error::Error;
use std::fmt;

/// Errors from constructing a detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CfarError {
    /// The window must have at least one reference cell per side.
    NoReferenceCells,
    /// The threshold multiplier must be positive and finite.
    InvalidMultiplier(f64),
    /// A fixed threshold must be positive and finite.
    InvalidThreshold(f64),
}

impl fmt::Display for CfarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoReferenceCells => {
                write!(f, "CFAR window needs at least one reference cell per side")
            }
            Self::InvalidMultiplier(a) => {
                write!(
                    f,
                    "threshold multiplier must be positive and finite, got {a}"
                )
            }
            Self::InvalidThreshold(t) => {
                write!(f, "threshold must be positive and finite, got {t}")
            }
        }
    }
}

impl Error for CfarError {}
