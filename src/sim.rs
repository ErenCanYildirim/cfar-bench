//! Synthetic data generation.
//!
//! Everything here is generic over the random number generator, so callers
//! control seeding and reproducibility.

mod noise;

pub use noise::{ComplexAwgn, NoiseError};
