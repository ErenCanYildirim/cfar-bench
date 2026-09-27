//! Fixed-threshold detection with exactly known noise power: the benchmark
//! against which CFAR loss is measured.
//!
//! # Derivation
//!
//! Noise-only CUT `Y ~ Exp(P)`, so `Pfa = P(Y > T) = exp(-T/P)` and
//! `T = -P ln(Pfa)`.
//!
//! With a Swerling I target of SNR `S`, `Y ~ Exp(P(1 + S))`:
//!
//! ```text
//! Pd = exp(-T / (P(1 + S))) = Pfa^(1 / (1 + S))
//! ```
//!
//! and solving for the SNR that reaches a given Pd:
//! `1 + S = ln(Pfa) / ln(Pd)`.

/// Threshold giving false alarm probability `pfa` on exponential noise of
/// power `noise_power`: `T = -P ln(Pfa)`.
#[must_use]
pub fn threshold_for_pfa(pfa: f64, noise_power: f64) -> f64 {
    -noise_power * pfa.ln()
}

/// Detection probability of a single-pulse Swerling I target with linear
/// SNR `snr`: `Pfa^(1/(1+S))`.
#[must_use]
pub fn pd_swerling1(pfa: f64, snr: f64) -> f64 {
    (pfa.ln() / (1.0 + snr)).exp()
}

/// Linear SNR at which a Swerling I target reaches detection probability
/// `pd`: `ln(Pfa)/ln(Pd) - 1`.
#[must_use]
pub fn required_snr_swerling1(pfa: f64, pd: f64) -> f64 {
    pfa.ln() / pd.ln() - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theory::ca_cfar;

    #[test]
    fn threshold_gives_design_pfa() {
        let t = threshold_for_pfa(1e-6, 2.5);
        assert!(((-t / 2.5).exp() - 1e-6).abs() < 1e-18);
    }

    #[test]
    fn required_snr_inverts_pd() {
        for pd in [0.1, 0.5, 0.9, 0.99] {
            let s = required_snr_swerling1(1e-6, pd);
            assert!((pd_swerling1(1e-6, s) - pd).abs() < 1e-12);
        }
    }

    #[test]
    fn is_the_large_n_limit_of_ca_cfar() {
        // Cross-module consistency: CA-CFAR with a huge window estimates the
        // noise almost perfectly and must approach the known-noise result.
        let (pfa, snr, n) = (1e-4, 20.0, 1 << 22);
        let alpha = ca_cfar::multiplier_for_pfa(pfa, n);
        let ca = ca_cfar::pd_swerling1(alpha, n, snr);
        assert!((ca - pd_swerling1(pfa, snr)).abs() < 1e-5);
    }

    #[test]
    fn cfar_loss_matches_large_n_expansion() {
        // Expanding α(p, N) = -ln p + (ln p)²/(2N) + O(1/N²) in
        // 1 + S_ca = α(Pfa)/α(Pd) gives
        //   (1 + S_ca) / (1 + S_ideal) ≈ 1 + (ln Pd - ln Pfa) / (2N).
        // An independent analytic check on the exact formulas.
        let (pfa, pd): (f64, f64) = (1e-6, 0.9);
        let ideal = 1.0 + required_snr_swerling1(pfa, pd);
        for n in [256, 1024, 4096] {
            let alpha = ca_cfar::multiplier_for_pfa(pfa, n);
            let ca = 1.0 + ca_cfar::required_snr_swerling1(alpha, n, pd);
            let exact = ca / ideal - 1.0;
            let first_order = (pd.ln() - pfa.ln()) / (2.0 * n as f64);
            assert!(
                ((exact - first_order) / first_order).abs() < 20.0 / n as f64,
                "N = {n}: exact {exact:e}, first order {first_order:e}"
            );
        }
    }
}
