//! Cell-averaging CFAR.

use std::ops::Range;

use super::{CfarError, CfarWindow};

/// Cell-averaging CFAR detector.
///
/// The noise power at the cell under test is estimated as the mean of the
/// `N` reference cells, and a detection is declared when
///
/// `CUT > α · (1/N) Σ reference`.
///
/// The comparison is strict. For continuous-valued data, equality has
/// probability zero, so this choice does not affect Pfa.
///
/// # Input contract
///
/// Inputs are square-law detected powers (`|x|² ≥ 0`). This is not checked
/// at runtime, since the detector sits in the inner loop of every
/// experiment; feeding it amplitudes `|x|` instead of powers is the classic
/// way to get a wrong Pfa, and the validation tests catch it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CaCfar {
    window: CfarWindow,
    multiplier: f64,
}

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
/// exactly the design Pfa.
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

impl CaCfar {
    /// Creates a detector with threshold multiplier `multiplier` (`α`).
    ///
    /// Use [`crate::theory::ca_cfar::multiplier_for_pfa`] to obtain `α` for
    /// a design false alarm probability.
    ///
    /// # Errors
    ///
    /// [`CfarError::InvalidMultiplier`] unless `multiplier` is positive and
    /// finite.
    pub fn new(window: CfarWindow, multiplier: f64) -> Result<Self, CfarError> {
        if !(multiplier.is_finite() && multiplier > 0.0) {
            return Err(CfarError::InvalidMultiplier(multiplier));
        }
        Ok(Self { window, multiplier })
    }

    /// The window geometry.
    #[must_use]
    pub fn window(&self) -> CfarWindow {
        self.window
    }

    /// The threshold multiplier `α`.
    #[must_use]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Threshold for a given sum of the `N` reference cells.
    fn threshold_from_sum(&self, reference_sum: f64) -> f64 {
        self.multiplier * reference_sum / self.window.reference_cells() as f64
    }

    /// Decision for a single cell under test, given its reference cells.
    ///
    /// This is the building block for the independent-trials Monte Carlo:
    /// the caller supplies a fresh CUT and fresh reference data per trial, so
    /// successive decisions are statistically independent.
    ///
    /// # Panics
    ///
    /// If `reference.len()` differs from the window's `N`. That is a
    /// programming error, not a runtime condition.
    #[must_use]
    pub fn detects(&self, cut: f64, reference: &[f64]) -> bool {
        assert_eq!(
            reference.len(),
            self.window.reference_cells(),
            "reference slice length must equal N"
        );
        cut > self.threshold_from_sum(reference.iter().sum())
    }

    /// Runs the detector along a power profile. See [`CfarProfile`] for the
    /// edge policy. A profile too short for one full window yields an empty
    /// result rather than an error.
    ///
    /// Cost is `O(len · N)`: each reference window is summed directly. A
    /// running-sum version would be `O(len)`, but direct summation is exact
    /// per cell and clearly correct, and `N ≤ 64` in practice.
    #[must_use]
    pub fn run(&self, power: &[f64]) -> CfarProfile {
        let half = self.window.half_width();
        let guard = self.window.guard_per_side();
        let first_cell = half;

        // `checked_sub` returns None instead of underflowing when the
        // profile is shorter than one full window.
        let Some(last_cell) = power.len().checked_sub(half + 1) else {
            return CfarProfile {
                first_cell,
                thresholds: Vec::new(),
                detections: Vec::new(),
            };
        };
        if last_cell < first_cell {
            return CfarProfile {
                first_cell,
                thresholds: Vec::new(),
                detections: Vec::new(),
            };
        }

        let mut thresholds = Vec::with_capacity(last_cell - first_cell + 1);
        let mut detections = Vec::new();

        for cut in first_cell..=last_cell {
            let lagging = &power[cut - half..cut - guard];
            let leading = &power[cut + guard + 1..=cut + half];
            let sum: f64 = lagging.iter().chain(leading).sum();
            let threshold = self.threshold_from_sum(sum);
            if power[cut] > threshold {
                detections.push(cut);
            }
            thresholds.push(threshold);
        }

        CfarProfile {
            first_cell,
            thresholds,
            detections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector(reference_per_side: usize, guard_per_side: usize, alpha: f64) -> CaCfar {
        CaCfar::new(
            CfarWindow::new(reference_per_side, guard_per_side).unwrap(),
            alpha,
        )
        .unwrap()
    }

    #[test]
    fn single_cell_threshold_is_alpha_times_reference_mean() {
        let d = detector(2, 0, 3.0);
        // mean = 2.5, threshold = 7.5
        let reference = [1.0, 2.0, 3.0, 4.0];
        assert!(d.detects(7.6, &reference));
        assert!(!d.detects(7.4, &reference));
        // Strict comparison exactly at the threshold.
        assert!(!d.detects(7.5, &reference));
    }

    #[test]
    #[should_panic(expected = "reference slice length must equal N")]
    fn single_cell_rejects_wrong_reference_length() {
        let _ = detector(2, 0, 3.0).detects(1.0, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn profile_uses_exactly_the_reference_cells() {
        // Window: 2 reference + 1 guard per side, half width 3.
        // Index:   0    1    2    3    4    5    6
        // Role:    R    R    G   CUT   G    R    R
        // Guards and CUT hold huge values that must NOT enter the estimate.
        let d = detector(2, 1, 2.0);
        let power = [1.0, 3.0, 1e9, 5.0, 1e9, 2.0, 6.0];
        let out = d.run(&power);

        assert_eq!(out.decided_cells(), 3..4);
        // mean(1, 3, 2, 6) = 3, threshold = 6
        assert!((out.thresholds[0] - 6.0).abs() < 1e-12);
        assert!(out.detections.is_empty()); // CUT = 5 < 6
    }

    #[test]
    fn profile_matches_single_cell_decisions() {
        // The sliding-window path and the single-cell path must agree
        // cell by cell.
        let d = detector(3, 2, 4.0);
        let power: Vec<f64> = (0..60)
            .map(|i| 1.0 + f64::from((i * 37) % 11) + if i == 30 { 40.0 } else { 0.0 })
            .collect();
        let out = d.run(&power);
        let half = d.window().half_width();
        let guard = d.window().guard_per_side();

        for cut in out.decided_cells() {
            let reference: Vec<f64> = power[cut - half..cut - guard]
                .iter()
                .chain(&power[cut + guard + 1..=cut + half])
                .copied()
                .collect();
            assert_eq!(
                d.detects(power[cut], &reference),
                out.detections.contains(&cut),
                "cell {cut}"
            );
        }
        assert!(out.detections.contains(&30));
    }

    #[test]
    fn edge_policy_skips_cells_without_a_full_window() {
        let d = detector(4, 2, 5.0); // half width 6, full window 13 cells
        assert_eq!(d.run(&[1.0; 20]).decided_cells(), 6..14);
        assert_eq!(d.run(&[1.0; 13]).decided_cells(), 6..7);
        assert!(d.run(&[1.0; 12]).thresholds.is_empty());
        assert!(d.run(&[]).thresholds.is_empty());
    }

    #[test]
    fn rejects_invalid_configuration() {
        assert_eq!(CfarWindow::new(0, 2), Err(CfarError::NoReferenceCells));
        let w = CfarWindow::new(8, 2).unwrap();
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(CaCfar::new(w, bad).is_err(), "alpha = {bad}");
        }
    }
}
