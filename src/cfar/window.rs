//! Geometry of a symmetric CFAR window.

use super::CfarError;

/// A symmetric sliding window around the cell under test (CUT):
///
/// ```text
/// [ R R R R | G G | CUT | G G | R R R R ]
///   reference guard        guard reference
/// ```
///
/// Guard cells keep a target's own energy (which spreads into neighbouring
/// cells after pulse compression) out of the noise estimate.
///
/// The total number of reference cells, `N` in the theory, is
/// `2 · reference_per_side`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CfarWindow {
    reference_per_side: usize,
    guard_per_side: usize,
}

impl CfarWindow {
    /// # Errors
    ///
    /// [`CfarError::NoReferenceCells`] if `reference_per_side == 0`.
    pub fn new(reference_per_side: usize, guard_per_side: usize) -> Result<Self, CfarError> {
        if reference_per_side == 0 {
            return Err(CfarError::NoReferenceCells);
        }
        Ok(Self {
            reference_per_side,
            guard_per_side,
        })
    }

    /// Reference cells on each side of the CUT.
    #[must_use]
    pub fn reference_per_side(&self) -> usize {
        self.reference_per_side
    }

    /// Guard cells on each side of the CUT.
    #[must_use]
    pub fn guard_per_side(&self) -> usize {
        self.guard_per_side
    }

    /// Total number of reference cells, `N`.
    #[must_use]
    pub fn reference_cells(&self) -> usize {
        2 * self.reference_per_side
    }

    /// Distance from the CUT to the outermost reference cell.
    #[must_use]
    pub fn half_width(&self) -> usize {
        self.guard_per_side + self.reference_per_side
    }
}
