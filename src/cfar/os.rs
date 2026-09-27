//! Ordered-statistic CFAR.

use super::{CfarError, CfarProfile, CfarWindow};

/// Ordered-statistic CFAR detector (Rohling, 1983).
///
/// The noise level is estimated by the `k`-th smallest reference cell
/// `X₍ₖ₎` (rank `k`, 1-based: `k = 1` is the minimum, `k = N` the maximum),
/// and a detection is declared when
///
/// `CUT > α · X₍ₖ₎`.
///
/// # Why it is robust
///
/// Up to `N - k` reference cells can be arbitrarily large (interfering
/// targets, clutter spikes) without changing `X₍ₖ₎` by more than a shift
/// of rank; a CA-CFAR mean is pulled up by a single large cell. The price
/// is a slightly worse noise estimate in homogeneous noise, hence a
/// slightly larger CFAR loss.
///
/// # Input contract
///
/// Square-law detected powers, as for [`super::CaCfar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OsCfar {
    window: CfarWindow,
    rank: usize,
    multiplier: f64,
}

impl OsCfar {
    /// Creates a detector using the `rank`-th smallest reference cell and
    /// threshold multiplier `multiplier` (`α`).
    ///
    /// Use [`crate::theory::os_cfar::multiplier_for_pfa`] to obtain `α` for
    /// a design false alarm probability.
    ///
    /// # Errors
    ///
    /// - [`CfarError::InvalidRank`] unless `1 ≤ rank ≤ N`.
    /// - [`CfarError::InvalidMultiplier`] unless `multiplier` is positive
    ///   and finite.
    pub fn new(window: CfarWindow, rank: usize, multiplier: f64) -> Result<Self, CfarError> {
        let n = window.reference_cells();
        if !(1..=n).contains(&rank) {
            return Err(CfarError::InvalidRank {
                rank,
                reference_cells: n,
            });
        }
        if !(multiplier.is_finite() && multiplier > 0.0) {
            return Err(CfarError::InvalidMultiplier(multiplier));
        }
        Ok(Self {
            window,
            rank,
            multiplier,
        })
    }

    /// The window geometry.
    #[must_use]
    pub fn window(&self) -> CfarWindow {
        self.window
    }

    /// The rank `k` (1-based).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The threshold multiplier `α`.
    #[must_use]
    pub fn multiplier(&self) -> f64 {
        self.multiplier
    }

    /// Decision for a single cell under test, given its reference cells.
    ///
    /// Implemented **without sorting**: `CUT > α·X₍ₖ₎` holds exactly when at
    /// least `k` reference cells satisfy `α·xᵢ < CUT`. Counting is `O(N)`,
    /// needs no scratch buffer and leaves `reference` untouched. It is
    /// exactly equivalent in floating point too, because multiplication by a
    /// positive `α` is monotone, so `α·X₍ₖ₎` is the `k`-th smallest of the
    /// `α·xᵢ`.
    ///
    /// # Panics
    ///
    /// If `reference.len()` differs from the window's `N`.
    #[must_use]
    pub fn detects(&self, cut: f64, reference: &[f64]) -> bool {
        assert_eq!(
            reference.len(),
            self.window.reference_cells(),
            "reference slice length must equal N"
        );
        let below = reference
            .iter()
            .filter(|&&x| self.multiplier * x < cut)
            .count();
        below >= self.rank
    }

    /// Runs the detector along a power profile. Same edge policy as all
    /// detectors (see [`CfarProfile`]).
    ///
    /// Unlike [`Self::detects`], this must report the threshold itself, so it
    /// selects `X₍ₖ₎` with `select_nth_unstable` (average `O(N)`) in a
    /// scratch buffer allocated once per call.
    #[must_use]
    pub fn run(&self, power: &[f64]) -> CfarProfile {
        let cells = self.window.decided_cells(power.len());
        let mut thresholds = Vec::with_capacity(cells.len());
        let mut detections = Vec::new();
        let mut scratch = Vec::with_capacity(self.window.reference_cells());

        for cut in cells.clone() {
            let (lagging, leading) = self.window.reference_halves(power, cut);
            scratch.clear();
            scratch.extend_from_slice(lagging);
            scratch.extend_from_slice(leading);
            // Partially reorders `scratch` so that index k-1 holds the value
            // a full sort would put there. `total_cmp` because f64 is not Ord.
            let (_, kth, _) = scratch.select_nth_unstable_by(self.rank - 1, f64::total_cmp);
            let threshold = self.multiplier * *kth;
            if power[cut] > threshold {
                detections.push(cut);
            }
            thresholds.push(threshold);
        }

        CfarProfile {
            first_cell: cells.start,
            thresholds,
            detections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detector(per_side: usize, guard: usize, rank: usize, alpha: f64) -> OsCfar {
        OsCfar::new(CfarWindow::new(per_side, guard).unwrap(), rank, alpha).unwrap()
    }

    #[test]
    fn threshold_uses_kth_smallest_reference_cell() {
        // Reference {7, 1, 5, 3}: sorted 1, 3, 5, 7. Rank 3 → 5, α = 2 → 10.
        let d = detector(2, 0, 3, 2.0);
        let reference = [7.0, 1.0, 5.0, 3.0];
        assert!(d.detects(10.5, &reference));
        assert!(!d.detects(9.5, &reference));
        assert!(!d.detects(10.0, &reference)); // strict
        // `detects` must not reorder the caller's data. This is an identity
        // check, not a numerical one, so compare bit patterns.
        assert_eq!(
            reference.map(f64::to_bits),
            [7.0, 1.0, 5.0, 3.0].map(f64::to_bits)
        );
    }

    #[test]
    fn large_outliers_do_not_move_the_threshold_below_rank() {
        // Two huge cells among 8, rank 6 of 8: the 6th smallest is still a
        // "noise" cell, so the threshold is unchanged. CA would be ruined.
        let d = detector(4, 0, 6, 1.0);
        let clean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let dirty = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1e6, 1e7];
        assert!(d.detects(6.5, &clean));
        assert!(d.detects(6.5, &dirty));
    }

    #[test]
    fn profile_matches_single_cell_decisions() {
        let d = detector(3, 1, 4, 3.0);
        let power: Vec<f64> = (0..80)
            .map(|i| 0.3 + f64::from((i * 29) % 13) * 0.7 + if i == 40 { 30.0 } else { 0.0 })
            .collect();
        let out = d.run(&power);
        for cut in out.decided_cells() {
            let (lag, lead) = d.window().reference_halves(&power, cut);
            let reference: Vec<f64> = lag.iter().chain(lead).copied().collect();
            assert_eq!(
                d.detects(power[cut], &reference),
                out.detections.contains(&cut),
                "cell {cut}"
            );
        }
        assert!(out.detections.contains(&40));
    }

    #[test]
    fn rejects_invalid_configuration() {
        let w = CfarWindow::new(4, 0).unwrap(); // N = 8
        assert_eq!(
            OsCfar::new(w, 0, 1.0),
            Err(CfarError::InvalidRank {
                rank: 0,
                reference_cells: 8
            })
        );
        assert!(OsCfar::new(w, 9, 1.0).is_err());
        assert!(OsCfar::new(w, 8, 1.0).is_ok());
        assert!(OsCfar::new(w, 4, f64::NAN).is_err());
    }
}
