//! CA-CFAR run along whole profiles: correlated decisions.
//!
//! # The issue
//!
//! Sliding the detector along a profile makes neighbouring decisions
//! dependent, through two opposing mechanisms:
//!
//! - **Shared reference cells** (positive correlation): neighbours estimate
//!   the noise from mostly the same samples, so a low estimate lowers both
//!   thresholds together.
//! - **CUT-in-reference suppression** (negative correlation): a false alarm
//!   is a large sample, and that sample sits in its neighbours' reference
//!   windows, raising their thresholds.
//!
//! The false alarm count over a profile is therefore **not** binomial, and a
//! binomial interval over profile cells is miscalibrated. The sign of the
//! error depends on window geometry (see `examples/correlated_profile.rs`,
//! which measures it).
//!
//! # What *is* still true, and tested here
//!
//! Every decided cell has a full window of independent noise, so each
//! decision's **marginal** false alarm probability is exactly the theory
//! value. The mean false alarm rate over a profile is therefore unbiased.
//!
//! To test it with a valid interval, we use **batch means**: many
//! independently generated profiles, one false alarm rate per profile. Those
//! rates are i.i.d., so their mean has an honest standard error regardless
//! of the correlation inside each profile.

use cfar_bench::cfar::{CaCfar, CfarWindow};
use cfar_bench::sim::ComplexAwgn;
use cfar_bench::stats::{self, TestFamily};
use cfar_bench::theory::ca_cfar;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;

/// (reference cells N, guard cells per side).
const WINDOWS: [(usize, usize); 3] = [(8, 0), (16, 2), (32, 0)];
const DESIGN_PFA: f64 = 1e-2;
const PROFILES: u64 = 2_000;
const PROFILE_LEN: usize = 2_000;
const FAMILY_ALPHA: f64 = 1e-3;

fn family() -> TestFamily {
    TestFamily::bonferroni(FAMILY_ALPHA, WINDOWS.len()).expect("valid family")
}

/// False alarm rate of each of `PROFILES` independent noise-only profiles.
fn per_profile_rates(detector: &CaCfar, seed_base: u64) -> Vec<f64> {
    let noise = ComplexAwgn::new(1.0).expect("valid power");
    (0..PROFILES)
        .map(|i| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed_base + i);
            let power: Vec<f64> = noise
                .sample_iter(&mut rng)
                .take(PROFILE_LEN)
                .map(|x| x.norm_sqr())
                .collect();
            let out = detector.run(&power);
            out.detections.len() as f64 / out.thresholds.len() as f64
        })
        .collect()
}

#[test]
fn mean_profile_false_alarm_rate_matches_theory() {
    // Normal-theory interval on the mean of 2000 i.i.d. per-profile rates.
    // Each rate averages ~2000 cells, so the rates are close to normal and
    // the CLT approximation is sound here.
    let z = family().per_test().two_sided_z();
    for (i, &(n, guard)) in WINDOWS.iter().enumerate() {
        let window = CfarWindow::new(n / 2, guard).expect("valid window");
        let detector =
            CaCfar::new(window, ca_cfar::multiplier_for_pfa(DESIGN_PFA, n)).expect("valid alpha");

        let rates = per_profile_rates(&detector, 1_000_000 * (i as u64 + 1));
        let mean = stats::mean(&rates).unwrap();
        let se = stats::standard_error_of_mean(&rates).unwrap();
        assert!(
            (mean - DESIGN_PFA).abs() <= z * se,
            "N = {n}, G = {guard}: mean rate {mean:.4e}, expected {DESIGN_PFA:e}, \
             {:.2} SE off",
            (mean - DESIGN_PFA) / se
        );
    }
}
