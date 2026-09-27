//! CA-CFAR in homogeneous exponential noise.
//!
//! # Derivation
//!
//! Square-law detected noise is exponential: the CUT is `Y ~ Exp(P)` and the
//! `N` reference cells `Xᵢ ~ Exp(P)`, all independent. Their sum
//! `Z = Σ Xᵢ ~ Gamma(N, P)`. With threshold `T = α Z / N`:
//!
//! ```text
//! P(Y > T | Z) = exp(-α Z / (N P))
//! Pfa = E_Z[exp(-α Z / (N P))]
//!     = M_Z(-α / (N P))              (Gamma MGF: (1 - sP)^(-N))
//!     = (1 + α / N)^(-N)
//! ```
//!
//! The noise power `P` cancels: that is the "constant false alarm rate"
//! property. As `N → ∞` the estimate becomes exact and
//! `Pfa → exp(-α)`, the known-noise detector.

/// False alarm probability of CA-CFAR with multiplier `alpha` and `n`
/// reference cells: `(1 + α/N)^(-N)`.
///
/// Evaluated as `exp(-N · ln(1 + α/N))` using `ln_1p`, which stays accurate
/// when `α/N` is small.
#[must_use]
pub fn pfa(alpha: f64, n: usize) -> f64 {
    let n = n as f64;
    (-n * (alpha / n).ln_1p()).exp()
}

/// Multiplier giving false alarm probability `pfa` with `n` reference cells:
/// `α = N (Pfa^(-1/N) - 1)`, the inverse of [`pfa`].
///
/// Evaluated as `N · expm1(-ln(Pfa) / N)`, which avoids cancellation in
/// `Pfa^(-1/N) - 1` when `N` is large.
#[must_use]
pub fn multiplier_for_pfa(pfa: f64, n: usize) -> f64 {
    let n = n as f64;
    n * (-pfa.ln() / n).exp_m1()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_round_trips() {
        for n in [1, 2, 8, 16, 32, 128] {
            for target in [1e-2, 1e-4, 1e-6, 1e-9] {
                let alpha = multiplier_for_pfa(target, n);
                let back = pfa(alpha, n);
                assert!(
                    ((back - target) / target).abs() < 1e-12,
                    "N={n}, Pfa={target}"
                );
            }
        }
    }

    #[test]
    fn hand_computed_values() {
        // N = 16, Pfa = 1e-6: α = 16 (10^(6/16) - 1).
        let expected = 16.0 * (10f64.powf(6.0 / 16.0) - 1.0);
        assert!((multiplier_for_pfa(1e-6, 16) - expected).abs() < 1e-12);
        // N = 1: Pfa = 1 / (1 + α).
        assert!((pfa(9.0, 1) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn approaches_known_noise_detector_as_n_grows() {
        // Pfa → exp(-α), so α → -ln Pfa. The finite-N multiplier is always
        // larger: estimating the noise costs threshold margin (CFAR loss).
        let target: f64 = 1e-4;
        let ideal = -target.ln();
        let mut previous = f64::INFINITY;
        for n in [4, 16, 64, 256, 4096, 1 << 20] {
            let alpha = multiplier_for_pfa(target, n);
            assert!(alpha > ideal && alpha < previous, "N = {n}");
            previous = alpha;
        }
        assert!((previous - ideal) / ideal < 1e-4);
    }

    #[test]
    fn pfa_decreases_with_multiplier() {
        let mut last = 1.0;
        for alpha in [0.1, 1.0, 5.0, 20.0, 100.0] {
            let p = pfa(alpha, 16);
            assert!(p < last);
            last = p;
        }
    }
}
