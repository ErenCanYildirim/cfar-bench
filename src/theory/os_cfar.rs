//! OS-CFAR in homogeneous exponential noise.
//!
//! # Derivation
//!
//! With `N` reference cells `Xᵢ ~ Exp(P)` and the CUT `Y ~ Exp(P)`, the
//! threshold is `α X₍ₖ₎`, where `X₍ₖ₎` is the `k`-th smallest reference cell.
//! As for CA-CFAR, `Pfa = E[exp(-α X₍ₖ₎ / P)]`, the Laplace transform of the
//! order statistic.
//!
//! By the Rényi representation, the spacings of exponential order
//! statistics are independent: `X₍ₖ₎ = P Σᵢ₌₀^{k-1} Eᵢ / (N - i)` with
//! `Eᵢ ~ Exp(1)` i.i.d. The transform of a sum of independent terms is the
//! product of their transforms, `E[exp(-s E/m)] = m / (m + s)`, so
//!
//! ```text
//! Pfa = ∏ᵢ₌₀^{k-1} (N - i) / (N - i + α)
//! ```
//!
//! Again `P` cancels (CFAR property). For a Swerling I target the CUT is
//! `Exp(P(1 + S))`, which replaces `α` by `α / (1 + S)`.
//!
//! Unlike CA-CFAR there is no closed-form inverse, so
//! [`multiplier_for_pfa`] solves for `α` by bisection. `Pfa` is strictly
//! decreasing in `α` (every factor is), so the root is unique.

/// False alarm probability of OS-CFAR with multiplier `alpha`, `n`
/// reference cells and rank `k`: `∏ (N - i)/(N - i + α)` for `i < k`.
///
/// Evaluated in log space as `exp(-Σ ln(1 + α/(N - i)))`, using `ln_1p`.
///
/// # Panics
///
/// Unless `1 ≤ k ≤ n`.
#[must_use]
pub fn pfa(alpha: f64, n: usize, k: usize) -> f64 {
    assert!((1..=n).contains(&k), "rank must satisfy 1 <= k <= N");
    let log_pfa: f64 = (0..k).map(|i| -(alpha / (n - i) as f64).ln_1p()).sum();
    log_pfa.exp()
}

/// Multiplier giving false alarm probability `target` with `n` reference
/// cells and rank `k`: the root of `pfa(α, n, k) = target`, by bisection.
///
/// # Panics
///
/// Unless `1 ≤ k ≤ n` and `0 < target < 1`.
#[must_use]
pub fn multiplier_for_pfa(target: f64, n: usize, k: usize) -> f64 {
    assert!(
        target > 0.0 && target < 1.0,
        "target Pfa must lie in (0, 1)"
    );
    // pfa(0) = 1 > target. Double `hi` until it brackets the root.
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    while pfa(hi, n, k) > target {
        lo = hi;
        hi *= 2.0;
    }
    // Relative tolerance: α spans orders of magnitude across (N, k, Pfa).
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if pfa(mid, n, k) > target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 1e-14 * hi {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// Probability of detection of a single-pulse Swerling I target with linear
/// SNR `snr`: `pfa(α / (1 + S), N, k)`.
#[must_use]
pub fn pd_swerling1(alpha: f64, n: usize, k: usize, snr: f64) -> f64 {
    pfa(alpha / (1.0 + snr), n, k)
}

/// Linear SNR at which a Swerling I target reaches detection probability
/// `pd`: `α / α(Pd) - 1`, where `α(Pd)` is the multiplier that would give
/// "Pfa" = `pd`. Same argument as for CA-CFAR.
#[must_use]
pub fn required_snr_swerling1(alpha: f64, n: usize, k: usize, pd: f64) -> f64 {
    alpha / multiplier_for_pfa(pd, n, k) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::ln_gamma;
    use crate::theory::ca_cfar;

    #[test]
    fn rank_one_has_closed_form() {
        // The minimum of N Exp(1) is Exp(N): Pfa = N / (N + α).
        for (n, alpha) in [(8, 3.0), (16, 100.0), (32, 0.5)] {
            let expected = n as f64 / (n as f64 + alpha);
            assert!((pfa(alpha, n, 1) - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn matches_gamma_function_form() {
        // Rohling's form: Pfa = Γ(N+1) Γ(α+N-k+1) / (Γ(N-k+1) Γ(α+N+1)).
        // Different algebra, computed with ln Γ: an independent check.
        for (n, k, alpha) in [(16, 12, 7.4), (32, 24, 11.0), (8, 6, 20.0), (24, 24, 3.3)] {
            let (nf, kf) = (n as f64, k as f64);
            let log_pfa = ln_gamma(nf + 1.0) + ln_gamma(alpha + nf - kf + 1.0)
                - ln_gamma(nf - kf + 1.0)
                - ln_gamma(alpha + nf + 1.0);
            let got = pfa(alpha, n, k);
            assert!(
                ((got - log_pfa.exp()) / got).abs() < 1e-11,
                "N={n}, k={k}, α={alpha}"
            );
        }
    }

    #[test]
    fn matches_numerical_integration_over_order_statistic() {
        // Pfa = ∫ exp(-α x) f₍ₖ₎(x) dx, with the k-th order statistic density
        // f₍ₖ₎(x) = k C(N,k) (1 - e^{-x})^{k-1} e^{-(N-k+1)x}. Composite
        // Simpson on [0, 40] (the integrand is < 1e-100 beyond).
        let (n, k, alpha) = (16_usize, 12_usize, 7.4_f64);
        let (nf, kf) = (n as f64, k as f64);
        let log_coef = kf.ln() + ln_gamma(nf + 1.0) - ln_gamma(kf + 1.0) - ln_gamma(nf - kf + 1.0);
        let integrand = |x: f64| {
            if x <= 0.0 {
                return 0.0;
            }
            (log_coef + (kf - 1.0) * (-(-x).exp_m1()).ln() - (nf - kf + 1.0) * x - alpha * x).exp()
        };
        let (a, b, m) = (0.0, 40.0, 40_000);
        let h = (b - a) / m as f64;
        let sum: f64 = (0..=m)
            .map(|i| {
                let w = if i == 0 || i == m {
                    1.0
                } else if i % 2 == 1 {
                    4.0
                } else {
                    2.0
                };
                w * integrand(a + i as f64 * h)
            })
            .sum();
        let numeric = sum * h / 3.0;
        let exact = pfa(alpha, n, k);
        assert!(
            ((numeric - exact) / exact).abs() < 1e-8,
            "{numeric} vs {exact}"
        );
    }

    #[test]
    fn multiplier_inverts_pfa() {
        for (n, k) in [(8, 6), (16, 12), (32, 24), (16, 1), (16, 16)] {
            for target in [1e-2, 1e-4, 1e-6] {
                let alpha = multiplier_for_pfa(target, n, k);
                assert!(
                    ((pfa(alpha, n, k) - target) / target).abs() < 1e-11,
                    "N={n}, k={k}, Pfa={target}"
                );
            }
        }
    }

    #[test]
    fn required_snr_inverts_pd() {
        let alpha = multiplier_for_pfa(1e-6, 16, 12);
        for pd in [0.1, 0.5, 0.9] {
            let s = required_snr_swerling1(alpha, 16, 12, pd);
            assert!((pd_swerling1(alpha, 16, 12, s) - pd).abs() < 1e-10);
        }
    }

    #[test]
    fn needs_more_snr_than_ca_in_homogeneous_noise() {
        // The price of robustness: in homogeneous noise the sample mean is
        // the better noise estimate, so OS-CFAR needs more SNR for the same
        // (Pfa, Pd). The penalty shrinks as N grows.
        let (pfa_design, pd) = (1e-6, 0.9);
        let mut last_gap = f64::INFINITY;
        for n in [8, 16, 32, 64] {
            let k = 3 * n / 4;
            let s_os = required_snr_swerling1(multiplier_for_pfa(pfa_design, n, k), n, k, pd);
            let s_ca =
                ca_cfar::required_snr_swerling1(ca_cfar::multiplier_for_pfa(pfa_design, n), n, pd);
            let gap = s_os / s_ca;
            assert!(gap > 1.0 && gap < last_gap, "N = {n}: gap {gap}");
            last_gap = gap;
        }
    }
}
