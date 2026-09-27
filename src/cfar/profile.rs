//! Output of a CFAR detector run along a profile.

use std::ops::Range;

/// Result of running a CFAR detector along a power profile.
///
/// # Edge policy
///
/// A decision is produced only for cells whose **full** window lies inside
/// the profile: indices `half_width ..= len - 1 - half_width`. Cells closer
/// to either end get no decision at all.
///
/// Alternatives were rejected deliberately:
///
/// - *Truncated window* (use whatever reference cells exist): changes `N`
///   near the edges, so a fixed `α` would give a different, higher Pfa there
///   unless `α` were re-derived per cell.
/// - *Wrap-around* (circular indexing): treats the far end of the profile as
///   a neighbour, which is only valid for genuinely circular data (e.g. a
///   full FFT spectrum), not for range profiles.
///
/// Skipping is the only policy under which every reported decision has
/// exactly the design Pfa. All detectors in this module share it through
/// [`super::CfarWindow`].
#[derive(Debug, Clone, PartialEq)]
pub struct CfarProfile {
    /// Input index of `thresholds[0]`.
    pub first_cell: usize,
    /// Threshold for each decided cell, aligned with `first_cell`.
    pub thresholds: Vec<f64>,
    /// Input indices of the cells that exceeded their threshold.
    pub detections: Vec<usize>,
}

impl CfarProfile {
    /// Input indices that received a decision.
    #[must_use]
    pub fn decided_cells(&self) -> Range<usize> {
        self.first_cell..self.first_cell + self.thresholds.len()
    }
}
