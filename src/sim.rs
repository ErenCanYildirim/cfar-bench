//! Synthetic data generation.
//!
//! Everything here is generic over the random number generator, so callers
//! control seeding and reproducibility.

mod noise;
mod scene;
mod target;
mod trial;

pub use noise::{ComplexAwgn, NoiseError};
pub use scene::{Scene, SceneError};
pub use target::SwerlingOne;
pub use trial::draw_cell_trial;
