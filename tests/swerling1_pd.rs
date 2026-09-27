//! Probability of detection for a Swerling I target: Monte Carlo vs theory.
//!
//! # Method
//!
//! Independent single-cell trials (`sim::draw_cell_trial`): each trial draws
//! a fresh Swerling I echo, fresh noise in the CUT and fresh reference
//! cells. Detection counts are exactly binomial, so each point is checked
//! with a Clopper–Pearson interval, as in the Pfa tests.
//!
//! Two detectors are measured on identical trial statistics:
//!
//! - CA-CFAR: `Pd = (1 + α/(N(1+S)))^(-N)`
//! - Fixed threshold with known noise power: `Pd = Pfa^(1/(1+S))`
//!
//! # Assertions (one family)
//!
//! - Pd vs SNR at `Pfa = 1e-3`: CA-CFAR at `N ∈ {8, 16, 32}` and the
//!   known-noise detector, at 5, 10, 15, 20 and 25 dB (20 points).
//! - ROC at 13 dB: CA-CFAR (`N = 16`) and known-noise detector at
//!   `Pfa ∈ {1e-1, ..., 1e-6}` (12 points).
//!
//! Negative controls check that a wrong target model or a wrong dB
//! convention is detected.

use cfar_bench::cfar::{CaCfar, CfarWindow, FixedThreshold};
use cfar_bench::sim::{ComplexAwgn, SwerlingOne, draw_cell_trial};
use cfar_bench::stats::{BinomialCount, Interval, TestFamily, trials_for_relative_precision};
use cfar_bench::theory::{ca_cfar, known_noise};
use cfar_bench::units::db_to_linear;
use num_complex::Complex64;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;
use rand::{Rng, SeedableRng};

const CURVE_PFA: f64 = 1e-3;
const CURVE_N: [usize; 3] = [8, 16, 32];
const CURVE_SNR_DB: [f64; 5] = [5.0, 10.0, 15.0, 20.0, 25.0];

const ROC_SNR_DB: f64 = 13.0;
const ROC_N: usize = 16;
const ROC_PFA: [f64; 6] = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6];

const FAMILY_ALPHA: f64 = 1e-3;
/// Relative precision on `min(Pd, 1 - Pd)`: sizes the trials so that both
/// the detection and the miss probability are measured well.
const RELATIVE_PRECISION: f64 = 0.1;
const MIN_TRIALS: u64 = 20_000;

/// Noise power. The results must not depend on it (CFAR property, and SNR
/// is defined relative to it), so a non-unit value is used deliberately.
const NOISE_POWER: f64 = 3.7;

fn family() -> TestFamily {
    let curve = CURVE_SNR_DB.len() * (CURVE_N.len() + 1);
    let roc = 2 * ROC_PFA.len();
    TestFamily::bonferroni(FAMILY_ALPHA, curve + roc).expect("valid family")
}

/// The two detectors under test. An enum rather than a trait object: the
/// set is closed and known, and `match` makes each case explicit.
#[derive(Debug, Clone, Copy)]
enum Detector {
    CaCfar(CaCfar),
    KnownNoise(FixedThreshold),
}

impl Detector {
    fn ca_cfar(n: usize, pfa: f64) -> Self {
        let window = CfarWindow::new(n / 2, 0).expect("valid window");
        let alpha = ca_cfar::multiplier_for_pfa(pfa, n);
        Self::CaCfar(CaCfar::new(window, alpha).expect("valid alpha"))
    }

    fn known_noise(pfa: f64) -> Self {
        let threshold = known_noise::threshold_for_pfa(pfa, NOISE_POWER);
        Self::KnownNoise(FixedThreshold::new(threshold).expect("valid threshold"))
    }

    fn reference_cells(&self) -> usize {
        match self {
            Self::CaCfar(d) => d.window().reference_cells(),
            Self::KnownNoise(_) => 0,
        }
    }

    fn detects(&self, cut: f64, reference: &[f64]) -> bool {
        match self {
            Self::CaCfar(d) => d.detects(cut, reference),
            Self::KnownNoise(d) => d.detects(cut),
        }
    }

    /// Theoretical Pd for a Swerling I target at linear SNR `snr`.
    fn theory_pd(&self, pfa: f64, snr: f64) -> f64 {
        match self {
            Self::CaCfar(d) => {
                ca_cfar::pd_swerling1(d.multiplier(), d.window().reference_cells(), snr)
            }
            Self::KnownNoise(_) => known_noise::pd_swerling1(pfa, snr),
        }
    }
}

fn trials_for(pd: f64) -> u64 {
    trials_for_relative_precision(pd.min(1.0 - pd), RELATIVE_PRECISION, family().per_test())
        .expect("valid plan")
        .max(MIN_TRIALS)
}

/// Counts detections over `trials` independent trials with echoes drawn
/// from `target`.
fn count_detections<T>(detector: &Detector, target: &T, trials: u64, seed: u64) -> u64
where
    T: Distribution<Complex64>,
{
    let noise = ComplexAwgn::new(NOISE_POWER).expect("valid power");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut reference = vec![0.0; detector.reference_cells()];
    (0..trials)
        .filter(|_| {
            let cut = draw_cell_trial(&mut rng, &noise, target, &mut reference);
            detector.detects(cut, &reference)
        })
        .count() as u64
}

fn interval(hits: u64, trials: u64) -> Interval {
    BinomialCount::new(hits, trials)
        .expect("valid count")
        .clopper_pearson(family().per_test())
        .expect("interval converges")
}

/// Measures Pd at `snr_db` and asserts the theory value lies in its CI.
fn check_point(detector: &Detector, pfa: f64, snr_db: f64, seed: u64) {
    let snr = db_to_linear(snr_db);
    let expected = detector.theory_pd(pfa, snr);
    let trials = trials_for(expected);
    let target = SwerlingOne::from_snr(snr, NOISE_POWER).expect("valid target");
    let hits = count_detections(detector, &target, trials, seed);
    let ci = interval(hits, trials);
    assert!(
        ci.contains(expected),
        "{detector:?}, Pfa {pfa:e}, SNR {snr_db} dB: measured Pd {:.4} ({hits}/{trials}), \
         theory {expected:.4}, CI [{:.4}, {:.4}]",
        hits as f64 / trials as f64,
        ci.lower,
        ci.upper
    );
}

fn check_curve(detector: &Detector, seed_base: u64) {
    for (i, &snr_db) in CURVE_SNR_DB.iter().enumerate() {
        check_point(detector, CURVE_PFA, snr_db, seed_base + i as u64);
    }
}

// --- Pd vs SNR ---------------------------------------------------------------

#[test]
fn pd_curve_ca_cfar_n8() {
    check_curve(&Detector::ca_cfar(8, CURVE_PFA), 100);
}

#[test]
fn pd_curve_ca_cfar_n16() {
    check_curve(&Detector::ca_cfar(16, CURVE_PFA), 200);
}

#[test]
fn pd_curve_ca_cfar_n32() {
    check_curve(&Detector::ca_cfar(32, CURVE_PFA), 300);
}

#[test]
fn pd_curve_known_noise() {
    check_curve(&Detector::known_noise(CURVE_PFA), 400);
}

// --- ROC at fixed SNR --------------------------------------------------------

#[test]
fn roc_ca_cfar() {
    for (i, &pfa) in ROC_PFA.iter().enumerate() {
        check_point(
            &Detector::ca_cfar(ROC_N, pfa),
            pfa,
            ROC_SNR_DB,
            500 + i as u64,
        );
    }
}

#[test]
fn roc_known_noise() {
    for (i, &pfa) in ROC_PFA.iter().enumerate() {
        check_point(&Detector::known_noise(pfa), pfa, ROC_SNR_DB, 600 + i as u64);
    }
}

// --- Negative controls -------------------------------------------------------

/// Non-fluctuating (Swerling 0) echo: constant amplitude. The phase is
/// fixed, which is harmless because the noise is circularly symmetric.
struct ConstantAmplitude(f64);

impl Distribution<Complex64> for ConstantAmplitude {
    fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> Complex64 {
        Complex64::new(self.0, 0.0)
    }
}

fn assert_rejected<T>(target: &T, snr_db: f64, seed: u64, what: &str)
where
    T: Distribution<Complex64>,
{
    let detector = Detector::ca_cfar(16, CURVE_PFA);
    let expected = detector.theory_pd(CURVE_PFA, db_to_linear(snr_db));
    let trials = trials_for(expected);
    let hits = count_detections(&detector, target, trials, seed);
    let ci = interval(hits, trials);
    assert!(
        !ci.contains(expected),
        "{what} not caught: measured {:.4}, Swerling I theory {expected:.4}",
        hits as f64 / trials as f64
    );
}

#[test]
fn rejects_non_fluctuating_target() {
    // Same mean SNR, different target model. At 15 dB a steady target is
    // detected far more often than a fluctuating one (Pd ≈ 1 vs ≈ 0.77):
    // fluctuation loss.
    let snr_db = 15.0;
    let amplitude = (db_to_linear(snr_db) * NOISE_POWER).sqrt();
    assert_rejected(
        &ConstantAmplitude(amplitude),
        snr_db,
        700,
        "Swerling 0 target",
    );
}

#[test]
fn rejects_amplitude_db_convention() {
    // Bug: converting SNR with 10^(dB/20) (an amplitude ratio) instead of
    // 10^(dB/10). A "15 dB" target then has only 7.5 dB of power SNR.
    let snr_db: f64 = 15.0;
    let wrong_snr = 10f64.powf(snr_db / 20.0);
    let target = SwerlingOne::from_snr(wrong_snr, NOISE_POWER).expect("valid target");
    assert_rejected(&target, snr_db, 800, "10^(dB/20) SNR conversion");
}
