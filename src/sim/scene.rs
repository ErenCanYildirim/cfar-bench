//! Range profiles with a non-uniform background and point targets.

use std::error::Error;
use std::fmt;

use rand::Rng;
use rand::distr::Distribution;

use super::{ComplexAwgn, NoiseError, SwerlingOne};

/// A one-dimensional range profile: a per-cell background (noise or
/// clutter, circular complex Gaussian) plus Swerling I point targets.
///
/// Every cell is independent. Built with a consuming builder:
///
/// ```
/// use cfar_bench::sim::Scene;
///
/// let scene = Scene::homogeneous(64, 1.0)?
///     .with_clutter_edge(32, 100.0)?   // cells 32.. are clutter at 20 dB
///     .with_target(10, 50.0)?;         // Swerling I target in cell 10
/// assert_eq!(scene.len(), 64);
/// # Ok::<(), cfar_bench::sim::SceneError>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Scene {
    background: Vec<ComplexAwgn>,
    targets: Vec<(usize, SwerlingOne)>,
}

/// Errors from building a [`Scene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SceneError {
    /// A power was negative, NaN or infinite.
    InvalidPower(f64),
    /// A cell index was outside the profile.
    CellOutOfRange {
        /// The requested cell.
        cell: usize,
        /// Profile length.
        len: usize,
    },
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPower(p) => write!(f, "invalid power {p}"),
            Self::CellOutOfRange { cell, len } => {
                write!(f, "cell {cell} outside profile of length {len}")
            }
        }
    }
}

impl Error for SceneError {}

/// Lets `?` convert a [`NoiseError`] into a [`SceneError`] automatically.
impl From<NoiseError> for SceneError {
    fn from(e: NoiseError) -> Self {
        match e {
            NoiseError::InvalidPower(p) => Self::InvalidPower(p),
        }
    }
}

impl Scene {
    /// `len` cells of noise with power `noise_power`, no targets.
    ///
    /// # Errors
    ///
    /// [`SceneError::InvalidPower`] for an invalid power.
    pub fn homogeneous(len: usize, noise_power: f64) -> Result<Self, SceneError> {
        Ok(Self {
            background: vec![ComplexAwgn::new(noise_power)?; len],
            targets: Vec::new(),
        })
    }

    /// Sets the background of cells `start..` to `power` (a clutter edge,
    /// or any step in background level).
    ///
    /// # Errors
    ///
    /// [`SceneError::CellOutOfRange`] if `start ≥ len`, or
    /// [`SceneError::InvalidPower`].
    pub fn with_clutter_edge(mut self, start: usize, power: f64) -> Result<Self, SceneError> {
        self.check_cell(start)?;
        let source = ComplexAwgn::new(power)?;
        for cell in &mut self.background[start..] {
            *cell = source;
        }
        Ok(self)
    }

    /// Adds a Swerling I target of mean power `mean_power` in `cell`.
    ///
    /// # Errors
    ///
    /// [`SceneError::CellOutOfRange`] or [`SceneError::InvalidPower`].
    pub fn with_target(mut self, cell: usize, mean_power: f64) -> Result<Self, SceneError> {
        self.check_cell(cell)?;
        self.targets.push((cell, SwerlingOne::new(mean_power)?));
        Ok(self)
    }

    fn check_cell(&self, cell: usize) -> Result<(), SceneError> {
        if cell < self.len() {
            Ok(())
        } else {
            Err(SceneError::CellOutOfRange {
                cell,
                len: self.len(),
            })
        }
    }

    /// Number of cells.
    #[must_use]
    pub fn len(&self) -> usize {
        self.background.len()
    }

    /// Whether the profile has no cells.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.background.is_empty()
    }

    /// Mean power of each cell: background plus any targets in it. This is
    /// what closed-form theory for heterogeneous backgrounds needs.
    #[must_use]
    pub fn mean_powers(&self) -> Vec<f64> {
        let mut powers: Vec<f64> = self.background.iter().map(ComplexAwgn::power).collect();
        for (cell, target) in &self.targets {
            powers[*cell] += target.mean_power();
        }
        powers
    }

    /// Draws one realisation and writes the square-law detected powers
    /// `|x|²` into `out`, which the caller owns and reuses.
    ///
    /// # Panics
    ///
    /// If `out.len() != self.len()`.
    pub fn draw_power<R: Rng + ?Sized>(&self, rng: &mut R, out: &mut [f64]) {
        assert_eq!(
            out.len(),
            self.len(),
            "output length must equal scene length"
        );
        // Complex amplitudes must be summed *before* squaring: a target and
        // the background in the same cell interfere coherently. Done cell by
        // cell so no temporary buffer is allocated per draw; scenes have few
        // targets, so scanning the target list per cell is cheap.
        for (i, (o, background)) in out.iter_mut().zip(&self.background).enumerate() {
            let mut amplitude = background.sample(rng);
            for (_, target) in self.targets.iter().filter(|(cell, _)| *cell == i) {
                amplitude += target.sample(rng);
            }
            *o = amplitude.norm_sqr();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::ChaCha8Rng;

    #[test]
    fn mean_powers_reflect_edge_and_targets() {
        let s = Scene::homogeneous(6, 1.0)
            .unwrap()
            .with_clutter_edge(3, 10.0)
            .unwrap()
            .with_target(1, 5.0)
            .unwrap()
            .with_target(1, 2.0)
            .unwrap();
        let expected = [1.0, 8.0, 1.0, 10.0, 10.0, 10.0];
        for (got, want) in s.mean_powers().iter().zip(expected) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn drawn_powers_have_the_right_means() {
        // Loose statistical check (the full noise validation is PR 1's job):
        // 20k draws, per-cell relative SE ≈ 0.7 %, bound at 5 %.
        let s = Scene::homogeneous(4, 2.0)
            .unwrap()
            .with_clutter_edge(2, 50.0)
            .unwrap()
            .with_target(0, 6.0)
            .unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut out = vec![0.0; 4];
        let mut sums = [0.0; 4];
        let draws = 20_000;
        for _ in 0..draws {
            s.draw_power(&mut rng, &mut out);
            for (acc, v) in sums.iter_mut().zip(&out) {
                *acc += v;
            }
        }
        for (sum, want) in sums.iter().zip(s.mean_powers()) {
            let mean = sum / f64::from(draws);
            assert!(((mean - want) / want).abs() < 0.05, "{mean} vs {want}");
        }
    }

    #[test]
    fn rejects_bad_cells_and_powers() {
        let s = Scene::homogeneous(4, 1.0).unwrap();
        assert_eq!(
            s.clone().with_target(4, 1.0),
            Err(SceneError::CellOutOfRange { cell: 4, len: 4 })
        );
        assert!(s.clone().with_clutter_edge(1, -1.0).is_err());
        assert!(Scene::homogeneous(4, f64::NAN).is_err());
    }
}
