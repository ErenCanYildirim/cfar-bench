//! Statistical validation of `ComplexAwgn` against the `CN(0, P)` model.
//!
//! Every tolerance is a confidence bound derived from the sampling
//! distribution of the estimator, not a hand-picked percentage. Seeds are
//! fixed, so CI is deterministic; the bounds are what make passing
//! meaningful.
//!
//! Tolerance convention (see `cfar_bench::stats::TestFamily` for the
//! project policy): all statistical assertions in this file form one
//! family with family-wise error rate `FAMILY_ALPHA`. Each assertion runs at
//! the Bonferroni level `α = FAMILY_ALPHA / m`, i.e.
//! `|estimate - truth| <= z(α) * standard_error`, and the KS test requires
//! `p > α`. Negative controls are not part of the family.

use cfar_bench::sim::ComplexAwgn;
use cfar_bench::stats::{self, TestFamily, ks_one_sample};
use cfar_bench::theory::exponential;
use num_complex::Complex64;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;

/// Samples per test. Relative standard error of the power estimate is
/// `1/√N ≈ 0.1 %`.
const N: usize = 1_000_000;
/// Power levels spanning several decades, to catch scaling bugs that happen
/// to vanish at `P = 1`.
const POWERS: [f64; 3] = [0.1, 1.0, 42.0];
/// Highest raw moment checked.
const MAX_MOMENT: u32 = 4;
/// Statistical assertions per power level: mean power (1), I power, Q power
/// and I/Q correlation (3), KS (1), raw moments (`MAX_MOMENT`).
const ASSERTIONS_PER_POWER: usize = 1 + 3 + 1 + MAX_MOMENT as usize;
/// Chance that a *correct* generator fails this file for a fresh seed.
const FAMILY_ALPHA: f64 = 1e-3;

/// The family is derived from the constants above, so adding a power level
/// or a moment automatically tightens every per-test threshold.
fn family() -> TestFamily {
    TestFamily::bonferroni(FAMILY_ALPHA, POWERS.len() * ASSERTIONS_PER_POWER).expect("valid family")
}

/// Two-sided critical value for a single assertion (≈ 4.13 for m = 27).
fn z() -> f64 {
    family().per_test().two_sided_z()
}

fn draw(power: f64, seed: u64) -> Vec<Complex64> {
    let noise = ComplexAwgn::new(power).expect("valid power");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    noise.sample_iter(&mut rng).take(N).collect()
}

fn squared_magnitudes(x: &[Complex64]) -> Vec<f64> {
    x.iter().map(Complex64::norm_sqr).collect()
}

#[test]
fn measured_power_matches_target() {
    for (seed, &p) in POWERS.iter().enumerate() {
        let y = squared_magnitudes(&draw(p, seed as u64));

        let z = z();
        let p_hat = stats::mean(&y).unwrap();
        // Standard error estimated from the data itself (sample variance of
        // |x|²), not assumed from the model.
        let se = stats::standard_error_of_mean(&y).unwrap();

        assert!(
            (p_hat - p).abs() <= z * se,
            "P = {p}: measured {p_hat}, |error| = {:.3e} > {z:.2}·SE = {:.3e}",
            (p_hat - p).abs(),
            z * se
        );
    }
}

#[test]
fn i_and_q_have_equal_power_and_are_uncorrelated() {
    for (seed, &p) in POWERS.iter().enumerate() {
        let z = z();
        let x = draw(p, 100 + seed as u64);
        let i: Vec<f64> = x.iter().map(|c| c.re).collect();
        let q: Vec<f64> = x.iter().map(|c| c.im).collect();

        // Each component should carry P/2. Estimate E[I²] as the mean of I².
        for (name, comp) in [("I", &i), ("Q", &q)] {
            let sq: Vec<f64> = comp.iter().map(|v| v * v).collect();
            let est = stats::mean(&sq).unwrap();
            let se = stats::standard_error_of_mean(&sq).unwrap();
            assert!(
                (est - p / 2.0).abs() <= z * se,
                "P = {p}: {name} power {est}, expected {}",
                p / 2.0
            );
        }

        // Under independence, √N · r is asymptotically N(0, 1).
        let r = stats::pearson_correlation(&i, &q).unwrap();
        let bound = z / (N as f64).sqrt();
        assert!(
            r.abs() <= bound,
            "P = {p}: corr(I, Q) = {r:.3e}, bound {bound:.3e}"
        );
    }
}

#[test]
fn squared_magnitude_passes_ks_test_for_exponential() {
    for (seed, &p) in POWERS.iter().enumerate() {
        let alpha = family().per_test().alpha();
        let y = squared_magnitudes(&draw(p, 200 + seed as u64));
        // The reference CDF uses the *true* P, not an estimate from `y`;
        // estimating it would invalidate the KS p-value.
        let r = ks_one_sample(&y, |v| exponential::cdf(v, p)).unwrap();
        assert!(
            r.p_value > alpha,
            "P = {p}: KS D = {:.3e}, p = {:.3e}",
            r.statistic,
            r.p_value
        );
    }
}

#[test]
fn squared_magnitude_moments_match_exponential() {
    for (seed, &p) in POWERS.iter().enumerate() {
        let z = z();
        let y = squared_magnitudes(&draw(p, 300 + seed as u64));

        for k in 1..=MAX_MOMENT {
            let yk: Vec<f64> = y.iter().map(|v| v.powi(k as i32)).collect();
            let est = stats::mean(&yk).unwrap();
            let truth = exponential::raw_moment(k, p);
            // Exact standard error from theory: Var(y^k) / N.
            let se = (exponential::raw_moment_variance(k, p) / N as f64).sqrt();
            assert!(
                (est - truth).abs() <= z * se,
                "P = {p}, k = {k}: E[y^k] = {est}, expected {truth}, {:.2} SE off",
                (est - truth) / se
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Negative controls: a validation test is only worth something if it fails
// on plausibly wrong data. These feed the same tests deliberately broken
// noise and require that they reject it.
// ---------------------------------------------------------------------------

#[test]
fn ks_rejects_real_valued_noise() {
    // Classic bug: all power in I, none in Q. Mean power is still correct,
    // so a power check alone would pass. |x|² is then P·χ²₁, not exponential.
    let p = 1.0;
    let y: Vec<f64> = draw(p, 400)
        .iter()
        .map(|c| (c.re * std::f64::consts::SQRT_2).powi(2))
        .collect();

    let mean_power = stats::mean(&y).unwrap();
    assert!(
        (mean_power - p).abs() < 0.01,
        "control setup: power should still be ≈ P"
    );

    let r = ks_one_sample(&y, |v| exponential::cdf(v, p)).unwrap();
    assert!(
        r.p_value < 1e-12,
        "KS failed to reject χ²₁: p = {:.3e}",
        r.p_value
    );
}

#[test]
fn ks_rejects_one_percent_power_miscalibration() {
    // A 1 % power error shifts the CDF by up to ~e⁻¹ · 1 % ≈ 0.37 %, about
    // 3.7 / √N at N = 1e6, far beyond the KS critical value 1.36 / √N.
    let p = 1.0;
    let y = squared_magnitudes(&draw(1.01 * p, 500));
    let r = ks_one_sample(&y, |v| exponential::cdf(v, p)).unwrap();
    assert!(
        r.p_value < 1e-6,
        "KS missed 1 % miscalibration: p = {:.3e}",
        r.p_value
    );
}
