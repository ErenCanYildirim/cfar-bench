//! OS-CFAR: Monte Carlo vs closed-form theory, Pfa and Swerling I Pd.
//!
//! Same method as the CA-CFAR tests: independent single-cell trials (fresh
//! CUT and reference cells per trial), exact binomial counts, and
//! Clopper–Pearson intervals under one Bonferroni family.
//!
//! # Assertions (one family)
//!
//! - Pfa grid, `k = 3N/4`: `N ∈ {8, 16, 32}` × `Pfa ∈ {1e-2, 1e-3, 1e-4}`
//!   (9 points).
//! - Rank sweep at `N = 16`, `Pfa = 1e-3`: `k ∈ {1, 4, 8, 16}`, covering
//!   both extremes (minimum and maximum) of the order statistic (4 points).
//! - Pd vs SNR at `Pfa = 1e-3`, `k = 3N/4`: `N ∈ {8, 16, 32}` at 5–25 dB
//!   (15 points).
//! - ROC at 13 dB, `N = 16`, `k = 12`: `Pfa ∈ {1e-1, ..., 1e-6}` (6 points).
//!
//! Negative controls: using the CA-CFAR multiplier, and an off-by-one rank.

use cfar_bench::cfar::{CfarWindow, OsCfar};
use cfar_bench::sim::{ComplexAwgn, SwerlingOne, draw_cell_trial};
use cfar_bench::stats::{BinomialCount, Interval, TestFamily, trials_for_relative_precision};
use cfar_bench::theory::{ca_cfar, os_cfar};
use cfar_bench::units::db_to_linear;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

const GRID_N: [usize; 3] = [8, 16, 32];
const GRID_PFA: [f64; 3] = [1e-2, 1e-3, 1e-4];

const SWEEP_N: usize = 16;
const SWEEP_PFA: f64 = 1e-3;
const SWEEP_RANKS: [usize; 4] = [1, 4, 8, 16];

const CURVE_PFA: f64 = 1e-3;
const CURVE_SNR_DB: [f64; 5] = [5.0, 10.0, 15.0, 20.0, 25.0];

const ROC_N: usize = 16;
const ROC_SNR_DB: f64 = 13.0;
const ROC_PFA: [f64; 6] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];

const NOISE_POWER: f64 = 0.42;
const FAMILY_ALPHA: f64 = 1e-3;
const PFA_PRECISION: f64 = 0.15;
const PD_PRECISION: f64 = 0.1;
const MIN_TRIALS: u64 = 20_000;

fn family() -> TestFamily {
    let size = GRID_N.len() * GRID_PFA.len()
        + SWEEP_RANKS.len()
        + GRID_N.len() * CURVE_SNR_DB.len()
        + ROC_PFA.len();
    TestFamily::bonferroni(FAMILY_ALPHA, size).expect("valid family")
}

/// The usual design choice `k = 3N/4`.
fn default_rank(n: usize) -> usize {
    3 * n / 4
}

fn detector(n: usize, k: usize, alpha: f64) -> OsCfar {
    OsCfar::new(CfarWindow::new(n / 2, 0).expect("valid window"), k, alpha).expect("valid OS")
}

fn designed(n: usize, k: usize, pfa: f64) -> OsCfar {
    detector(n, k, os_cfar::multiplier_for_pfa(pfa, n, k))
}

/// Trials so that `min(p, 1-p)` is measured to `precision` (relative).
fn trials_for(p: f64, precision: f64) -> u64 {
    trials_for_relative_precision(p.min(1.0 - p), precision, family().per_test())
        .expect("valid plan")
        .max(MIN_TRIALS)
}

fn count_detections(d: &OsCfar, snr: f64, trials: u64, seed: u64) -> u64 {
    let noise = ComplexAwgn::new(NOISE_POWER).expect("valid power");
    let target = SwerlingOne::from_snr(snr, NOISE_POWER).expect("valid target");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reference = vec![0.0; d.window().reference_cells()];
    (0..trials)
        .filter(|_| {
            let cut = draw_cell_trial(&mut rng, &noise, &target, &mut reference);
            d.detects(cut, &reference)
        })
        .count() as u64
}

fn interval(hits: u64, trials: u64) -> Interval {
    BinomialCount::new(hits, trials)
        .expect("valid count")
        .clopper_pearson(family().per_test())
        .expect("interval converges")
}

/// Measures the detection rate at linear SNR `snr` (0 for Pfa) and asserts
/// that `expected` lies inside its confidence interval.
fn check(d: &OsCfar, snr: f64, expected: f64, precision: f64, seed: u64, label: &str) {
    let trials = trials_for(expected, precision);
    let hits = count_detections(d, snr, trials, seed);
    let ci = interval(hits, trials);
    assert!(
        ci.contains(expected),
        "{label}: measured {:.4e} ({hits}/{trials}), theory {expected:.4e}, \
         CI [{:.4e}, {:.4e}]",
        hits as f64 / trials as f64,
        ci.lower,
        ci.upper
    );
}

fn check_pfa_row(pfa_index: usize) {
    let pfa = GRID_PFA[pfa_index];
    for &n in &GRID_N {
        let k = default_rank(n);
        let seed = 50_000 + 100 * pfa_index as u64 + n as u64;
        check(
            &designed(n, k, pfa),
            0.0,
            pfa,
            PFA_PRECISION,
            seed,
            &format!("Pfa, N={n}, k={k}, Pfa={pfa:e}"),
        );
    }
}

#[test]
fn pfa_matches_theory_at_1e_2() {
    check_pfa_row(0);
}

#[test]
fn pfa_matches_theory_at_1e_3() {
    check_pfa_row(1);
}

#[test]
fn pfa_matches_theory_at_1e_4() {
    check_pfa_row(2);
}

#[test]
fn pfa_matches_theory_across_ranks() {
    for (i, &k) in SWEEP_RANKS.iter().enumerate() {
        check(
            &designed(SWEEP_N, k, SWEEP_PFA),
            0.0,
            SWEEP_PFA,
            PFA_PRECISION,
            51_000 + i as u64,
            &format!("rank sweep, N={SWEEP_N}, k={k}"),
        );
    }
}

#[test]
fn pd_curves_match_theory() {
    for &n in &GRID_N {
        let k = default_rank(n);
        let d = designed(n, k, CURVE_PFA);
        for (i, &snr_db) in CURVE_SNR_DB.iter().enumerate() {
            let snr = db_to_linear(snr_db);
            let expected = os_cfar::pd_swerling1(d.multiplier(), n, k, snr);
            check(
                &d,
                snr,
                expected,
                PD_PRECISION,
                52_000 + 10 * n as u64 + i as u64,
                &format!("Pd, N={n}, k={k}, SNR={snr_db} dB"),
            );
        }
    }
}

#[test]
fn roc_matches_theory() {
    let k = default_rank(ROC_N);
    let snr = db_to_linear(ROC_SNR_DB);
    for (i, &pfa) in ROC_PFA.iter().enumerate() {
        let d = designed(ROC_N, k, pfa);
        let expected = os_cfar::pd_swerling1(d.multiplier(), ROC_N, k, snr);
        check(
            &d,
            snr,
            expected,
            PD_PRECISION,
            53_000 + i as u64,
            &format!("ROC, Pfa={pfa:e}"),
        );
    }
}

// --- Negative controls -------------------------------------------------------

fn assert_pfa_rejected(d: &OsCfar, seed: u64, what: &str) {
    let trials = trials_for(SWEEP_PFA, PFA_PRECISION);
    let hits = count_detections(d, 0.0, trials, seed);
    let ci = interval(hits, trials);
    assert!(
        !ci.contains(SWEEP_PFA),
        "{what} not caught: measured {:.4e}, CI [{:.4e}, {:.4e}]",
        hits as f64 / trials as f64,
        ci.lower,
        ci.upper
    );
}

#[test]
fn rejects_ca_cfar_multiplier() {
    // Bug: reusing the CA-CFAR α for OS-CFAR. At N = 16, k = 12 this gives
    // Pfa ≈ 4.4e-4 instead of 1e-3.
    let (n, k) = (16, 12);
    let d = detector(n, k, ca_cfar::multiplier_for_pfa(SWEEP_PFA, n));
    assert_pfa_rejected(&d, 54_000, "CA multiplier on OS-CFAR");
}

#[test]
fn rejects_off_by_one_rank() {
    // Bug: α designed for rank 12, detector built with rank 13 (a 0- vs
    // 1-based index slip). Pfa ≈ 3.5e-4 instead of 1e-3.
    let (n, k) = (16, 12);
    let d = detector(n, k + 1, os_cfar::multiplier_for_pfa(SWEEP_PFA, n, k));
    assert_pfa_rejected(&d, 54_100, "off-by-one rank");
}
