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
//!
//! # Swerling I target
//!
//! With a Swerling I target of SNR `S`, the CUT is target + noise, a sum of
//! independent complex Gaussians: `CN(0, P(1 + S))`. Its power is
//! `Exp(P(1 + S))`, and the reference cells are unchanged. Repeating the
//! derivation with `P → P(1 + S)` in the CUT only:
//!
//! ```text
//! Pd = (1 + α / (N (1 + S)))^(-N)  =  pfa(α / (1 + S), N)
//! ```
//!
//! Solving `Pd(S) = Pd*` for `S` reuses the Pfa inversion:
//! `α / (1 + S) = multiplier_for_pfa(Pd*, N)`, so
//!
//! ```text
//! 1 + S_required = α(Pfa, N) / α(Pd*, N).
//! ```
//!
//! # Heterogeneous background
//!
//! The derivation never used that the cells share a power. If the CUT is
//! `Exp(p_c)` and reference cell `i` is `Exp(pᵢ)`, all independent, then the
//! reference sum's MGF is the product of the individual MGFs, and
//!
//! ```text
//! P(CUT > α Z / N) = ∏ᵢ (1 + α pᵢ / (N p_c))^(-1)
//! ```
//!
//! This one formula covers homogeneous noise (all equal), Swerling I
//! targets (`p_c = P(1 + S)`), Swerling I interferers in the window
//! (`pᵢ = P(1 + INR)`) and clutter edges (a step in the `pᵢ` and/or `p_c`).
//! See [`exceedance_probability`].

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

/// Probability of detection of a single-pulse Swerling I target with
/// (linear) SNR `snr`: `(1 + α/(N(1 + S)))^(-N)`.
#[must_use]
pub fn pd_swerling1(alpha: f64, n: usize, snr: f64) -> f64 {
    pfa(alpha / (1.0 + snr), n)
}

/// Linear SNR at which a Swerling I target reaches detection probability
/// `pd`, with multiplier `alpha`: `α / α(Pd, N) - 1`. The inverse of
/// [`pd_swerling1`] in `snr`.
#[must_use]
pub fn required_snr_swerling1(alpha: f64, n: usize, pd: f64) -> f64 {
    alpha / multiplier_for_pfa(pd, n) - 1.0
}

/// Probability that the CUT exceeds the CA-CFAR threshold when every cell
/// is independently exponential with its own mean power:
/// `∏ᵢ (1 + α pᵢ / (N p_c))^(-1)`, with `N = reference_powers.len()`.
///
/// It is Pfa when the CUT holds only background and Pd when it holds a
/// Swerling I target. Mean powers are what [`crate::sim::Scene::mean_powers`]
/// reports, so a scene can be fed straight in.
///
/// # Panics
///
/// If `reference_powers` is empty.
#[must_use]
pub fn exceedance_probability(alpha: f64, cut_power: f64, reference_powers: &[f64]) -> f64 {
    assert!(
        !reference_powers.is_empty(),
        "need at least one reference cell"
    );
    let n = reference_powers.len() as f64;
    let log_p: f64 = reference_powers
        .iter()
        .map(|&p| -(alpha * p / (n * cut_power)).ln_1p())
        .sum();
    log_p.exp()
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
    fn heterogeneous_formula_reduces_to_homogeneous_cases() {
        let (n, alpha, snr) = (16, 8.6, 12.0);
        let flat = vec![2.5; n];
        // All cells equal: Pfa. Scale invariance: 2.5 cancels.
        assert!((exceedance_probability(alpha, 2.5, &flat) - pfa(alpha, n)).abs() < 1e-15);
        // Target in the CUT: Swerling I Pd.
        let pd = exceedance_probability(alpha, 2.5 * (1.0 + snr), &flat);
        assert!((pd - pd_swerling1(alpha, n, snr)).abs() < 1e-15);
    }

    #[test]
    fn interferers_mask_and_clutter_edges_spike() {
        let (n, alpha) = (16, multiplier_for_pfa(1e-3, 16));
        // Two 20 dB interferers in the window lower Pfa (masking).
        let mut refs = vec![1.0; n];
        refs[0] = 101.0;
        refs[1] = 101.0;
        assert!(exceedance_probability(alpha, 1.0, &refs) < 1e-3);
        // CUT in 20 dB clutter with half the window still in noise: the
        // threshold is set too low, and Pfa jumps.
        let refs: Vec<f64> = (0..n)
            .map(|i| if i < n / 2 { 1.0 } else { 100.0 })
            .collect();
        assert!(exceedance_probability(alpha, 100.0, &refs) > 1e-2);
    }

    #[test]
    fn pd_at_zero_snr_is_pfa() {
        let alpha = multiplier_for_pfa(1e-4, 16);
        assert!((pd_swerling1(alpha, 16, 0.0) - 1e-4).abs() < 1e-16);
    }

    #[test]
    fn required_snr_inverts_pd() {
        for n in [4, 16, 64] {
            let alpha = multiplier_for_pfa(1e-6, n);
            for target_pd in [0.1, 0.5, 0.9, 0.99] {
                let snr = required_snr_swerling1(alpha, n, target_pd);
                let back = pd_swerling1(alpha, n, snr);
                assert!((back - target_pd).abs() < 1e-12, "N={n}, Pd={target_pd}");
            }
        }
    }

    #[test]
    fn pd_increases_with_snr() {
        let alpha = multiplier_for_pfa(1e-3, 16);
        let mut last = 0.0;
        for snr in [0.0, 1.0, 10.0, 100.0, 1e4] {
            let pd = pd_swerling1(alpha, 16, snr);
            assert!(pd > last);
            last = pd;
        }
        assert!(last > 0.999);
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
