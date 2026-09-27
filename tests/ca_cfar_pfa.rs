//! CA-CFAR false alarm probability: Monte Carlo vs closed-form theory.
//!
//! # Method
//!
//! Each trial draws a fresh cell under test and `N` fresh reference cells
//! from `CN(0, P)` noise (PR 1 generator), square-law detects them, and
//! records the CA-CFAR decision. Trials are therefore **independent**, the
//! false alarm count is exactly `Binomial(trials, Pfa)`, and the exact
//! Clopper–Pearson interval (PR 2) is valid.
//!
//! This is deliberately *not* done by sliding the detector along one long
//! profile: neighbouring decisions there share reference cells and are
//! correlated, so a binomial interval over profile cells is miscalibrated.
//! That case is handled separately in `ca_cfar_profile.rs`.
//!
//! # Assertions (one family, see `TestFamily`)
//!
//! - Grid: `N ∈ {8, 16, 32}` × `Pfa ∈ {1e-2, 1e-3, 1e-4}` (9 assertions):
//!   the theoretical Pfa lies inside the CI of the measured one.
//! - CFAR property: at `N = 16`, `Pfa = 1e-3`, the same holds for noise
//!   powers `1e-3` and `1e3` (2 assertions). Six decades of noise power
//!   must not move the false alarm rate.
//!
//! Negative controls (not in the family) check that the test can fail.

use cfar_bench::cfar::{CaCfar, CfarWindow};
use cfar_bench::sim::ComplexAwgn;
use cfar_bench::stats::{BinomialCount, Interval, TestFamily, trials_for_relative_precision};
use cfar_bench::theory::ca_cfar;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;

const REFERENCE_CELLS: [usize; 3] = [8, 16, 32];
const DESIGN_PFAS: [f64; 3] = [1e-2, 1e-3, 1e-4];
const INVARIANCE_POWERS: [f64; 2] = [1e-3, 1e3];
const FAMILY_ALPHA: f64 = 1e-3;
/// Target relative half-width of each interval. Sets the trial count and
/// hence the test's power: a detector whose Pfa is off by more than about
/// this fraction is caught.
const RELATIVE_PRECISION: f64 = 0.15;
/// Floor so the high-Pfa points still get a meaningful sample.
const MIN_TRIALS: u64 = 100_000;

fn family() -> TestFamily {
    let size = REFERENCE_CELLS.len() * DESIGN_PFAS.len() + INVARIANCE_POWERS.len();
    TestFamily::bonferroni(FAMILY_ALPHA, size).expect("valid family")
}

fn trials_for(pfa: f64) -> u64 {
    trials_for_relative_precision(pfa, RELATIVE_PRECISION, family().per_test())
        .expect("valid plan")
        .max(MIN_TRIALS)
}

/// Runs `trials` independent single-cell trials and returns the number of
/// detections. `transform` maps each complex sample to the detector input;
/// the correct choice is `|x|²`.
fn count_detections(
    detector: &CaCfar,
    noise_power: f64,
    trials: u64,
    seed: u64,
    transform: fn(num_complex::Complex64) -> f64,
) -> u64 {
    let noise = ComplexAwgn::new(noise_power).expect("valid power");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // One buffer reused across trials: no allocation in the hot loop.
    let mut reference = vec![0.0; detector.window().reference_cells()];
    let mut hits = 0;
    for _ in 0..trials {
        for cell in &mut reference {
            *cell = transform(noise.sample(&mut rng));
        }
        let cut = transform(noise.sample(&mut rng));
        if detector.detects(cut, &reference) {
            hits += 1;
        }
    }
    hits
}

fn square_law(x: num_complex::Complex64) -> f64 {
    x.norm_sqr()
}

fn detector_for(n: usize, alpha: f64) -> CaCfar {
    // Guard cells do not matter for independent trials (there is no target
    // energy to keep out), so the window is built without them.
    CaCfar::new(CfarWindow::new(n / 2, 0).expect("valid window"), alpha).expect("valid alpha")
}

fn measured_interval(hits: u64, trials: u64) -> Interval {
    BinomialCount::new(hits, trials)
        .expect("valid count")
        .clopper_pearson(family().per_test())
        .expect("interval converges")
}

fn check_grid_row(pfa_index: usize) {
    let target = DESIGN_PFAS[pfa_index];
    let trials = trials_for(target);
    for &n in &REFERENCE_CELLS {
        let detector = detector_for(n, ca_cfar::multiplier_for_pfa(target, n));
        let seed = 10_000 + 100 * pfa_index as u64 + n as u64;
        let hits = count_detections(&detector, 1.0, trials, seed, square_law);
        let ci = measured_interval(hits, trials);
        assert!(
            ci.contains(target),
            "N = {n}, Pfa = {target:e}: measured {:.4e} ({hits}/{trials}), \
             CI [{:.4e}, {:.4e}]",
            hits as f64 / trials as f64,
            ci.lower,
            ci.upper
        );
    }
}

// One test per Pfa row so the rows run in parallel test threads.
#[test]
fn pfa_matches_theory_at_1e_2() {
    check_grid_row(0);
}

#[test]
fn pfa_matches_theory_at_1e_3() {
    check_grid_row(1);
}

#[test]
fn pfa_matches_theory_at_1e_4() {
    check_grid_row(2);
}

#[test]
fn pfa_is_independent_of_noise_power() {
    let (n, target) = (16, 1e-3);
    let detector = detector_for(n, ca_cfar::multiplier_for_pfa(target, n));
    let trials = trials_for(target);
    for (i, &power) in INVARIANCE_POWERS.iter().enumerate() {
        let hits = count_detections(&detector, power, trials, 20_000 + i as u64, square_law);
        let ci = measured_interval(hits, trials);
        assert!(
            ci.contains(target),
            "noise power {power:e}: measured {:.4e}, CI [{:.4e}, {:.4e}]",
            hits as f64 / trials as f64,
            ci.lower,
            ci.upper
        );
    }
}

// ---------------------------------------------------------------------------
// Negative controls: plausible bugs that must be detected.
// ---------------------------------------------------------------------------

#[test]
fn rejects_known_noise_multiplier() {
    // Bug: using the ideal (known-noise) multiplier α = -ln(Pfa), which
    // ignores that the noise level is only *estimated* from N cells.
    // At N = 8, Pfa = 1e-3 this gives a true Pfa of (1 + α/8)^-8 ≈ 6.9e-3.
    let (n, target): (usize, f64) = (8, 1e-3);
    let detector = detector_for(n, -target.ln());
    let trials = MIN_TRIALS;
    let hits = count_detections(&detector, 1.0, trials, 30_000, square_law);
    let ci = measured_interval(hits, trials);
    assert!(
        !ci.contains(target),
        "known-noise multiplier not caught: CI [{:.4e}, {:.4e}]",
        ci.lower,
        ci.upper
    );
    // And the measurement agrees with what theory predicts for that wrong α.
    assert!(ci.contains(ca_cfar::pfa(-target.ln(), n)));
}

#[test]
fn rejects_envelope_instead_of_square_law_input() {
    // Bug: feeding |x| (envelope) instead of |x|² (power). The multiplier is
    // designed for exponential data; on Rayleigh data it is far too large,
    // and the false alarm rate collapses.
    let (n, target) = (16, 1e-3);
    let detector = detector_for(n, ca_cfar::multiplier_for_pfa(target, n));
    let trials = trials_for(target);
    let hits = count_detections(&detector, 1.0, trials, 40_000, |x| x.norm());
    let ci = measured_interval(hits, trials);
    assert!(
        !ci.contains(target),
        "envelope input not caught: {hits}/{trials}, CI [{:.4e}, {:.4e}]",
        ci.lower,
        ci.upper
    );
}
