//! Geometry of a symmetric CFAR window.

use std::ops::Range;

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

    /// Cells of a profile of length `len` that have a full window: the edge
    /// policy documented on [`super::CfarProfile`]. Empty (but starting at
    /// `half_width`) if the profile is shorter than one window.
    #[must_use]
    pub fn decided_cells(&self, len: usize) -> Range<usize> {
        let half = self.half_width();
        // `saturating_sub` clamps at 0 instead of underflowing, and `max`
        // turns a too-short profile into the empty range `half..half`.
        half..len.saturating_sub(half).max(half)
    }

    /// The lagging and leading reference cells of `cut`, as two sub-slices
    /// of `power` (guard cells and the CUT itself excluded).
    ///
    /// The returned slices borrow from `power`, which the lifetime `'a`
    /// makes explicit: they cannot outlive the profile they point into.
    ///
    /// # Panics
    ///
    /// If `cut` is not in [`Self::decided_cells`] for `power.len()`.
    #[must_use]
    pub fn reference_halves<'a>(&self, power: &'a [f64], cut: usize) -> (&'a [f64], &'a [f64]) {
        let half = self.half_width();
        let guard = self.guard_per_side;
        (
            &power[cut - half..cut - guard],
            &power[cut + guard + 1..=cut + half],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decided_cells_edge_cases() {
        let w = CfarWindow::new(4, 2).unwrap(); // half width 6
        assert_eq!(w.decided_cells(20), 6..14);
        assert_eq!(w.decided_cells(13), 6..7);
        assert!(w.decided_cells(12).is_empty());
        assert!(w.decided_cells(0).is_empty());
    }

    #[test]
    fn reference_halves_skip_guards_and_cut() {
        let w = CfarWindow::new(2, 1).unwrap();
        let power: Vec<f64> = (0..7).map(f64::from).collect();
        // Layout: R R G CUT G R R at indices 0..7, CUT = 3.
        let (lag, lead) = w.reference_halves(&power, 3);
        assert_eq!(lag, &[0.0, 1.0]);
        assert_eq!(lead, &[5.0, 6.0]);
    }
}
