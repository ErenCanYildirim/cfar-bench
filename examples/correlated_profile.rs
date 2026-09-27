//! Measures how far false alarm counts along a CA-CFAR profile are from
//! binomial.
//!
//! Run with: `cargo run --release --example correlated_profile`
//!
//! For each window geometry it generates many independent noise-only
//! profiles, counts false alarms in each, and reports the **design effect**
//!
//! `DE = Var(count) / (n · p · (1 - p))`,
//!
//! the ratio of the observed count variance to the binomial variance for the
//! same number of decided cells. `DE = 1` means binomial behaviour. `DE > 1`
//! means a binomial interval would be too narrow (overconfident), and
//! `DE < 1` means too wide (conservative).
//!
//! The standard error printed for DE uses `Var(s²) ≈ 2σ⁴/(R-1)`, which
//! assumes near-normal counts; it is a guide, not an exact interval.

use cfar_bench::cfar::{CaCfar, CfarWindow};
use cfar_bench::sim::ComplexAwgn;
use cfar_bench::stats;
use cfar_bench::theory::ca_cfar;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;

const PROFILES: u64 = 4_000;
const PROFILE_LEN: usize = 2_000;

fn main() {
    let noise = ComplexAwgn::new(1.0).expect("valid power");
    let configs: [(usize, usize, f64); 7] = [
        (8, 0, 1e-2),
        (16, 0, 1e-2),
        (32, 0, 1e-2),
        (16, 1, 1e-2),
        (16, 2, 1e-2),
        (16, 4, 1e-2),
        (16, 0, 1e-3),
    ];

    println!(
        "{:>4} {:>4} {:>8} {:>12} {:>8} {:>8}",
        "N", "G", "Pfa", "measured", "DE", "±SE"
    );
    for (i, &(n, guard, pfa)) in configs.iter().enumerate() {
        let window = CfarWindow::new(n / 2, guard).expect("valid window");
        let detector =
            CaCfar::new(window, ca_cfar::multiplier_for_pfa(pfa, n)).expect("valid alpha");

        let mut decided = 0;
        let counts: Vec<f64> = (0..PROFILES)
            .map(|s| {
                let mut rng = ChaCha8Rng::seed_from_u64(5_000_000 * (i as u64 + 1) + s);
                let power: Vec<f64> = noise
                    .sample_iter(&mut rng)
                    .take(PROFILE_LEN)
                    .map(|x| x.norm_sqr())
                    .collect();
                let out = detector.run(&power);
                decided = out.thresholds.len();
                out.detections.len() as f64
            })
            .collect();

        let mean = stats::mean(&counts).expect("non-empty");
        let var = stats::variance(&counts).expect("at least two profiles");
        let p_hat = mean / decided as f64;
        let binomial_var = decided as f64 * p_hat * (1.0 - p_hat);
        let de = var / binomial_var;
        let de_se = de * (2.0 / (PROFILES as f64 - 1.0)).sqrt();

        println!("{n:>4} {guard:>4} {pfa:>8.0e} {p_hat:>12.4e} {de:>8.3} {de_se:>8.3}");
    }
}
