//! CA- vs OS-CFAR with interfering targets in the reference window.
//!
//! Run with: `cargo run --release --example interfering_targets`
//!
//! Setup: `N = 16` reference cells (8 per side), 1 guard cell per side,
//! OS rank `k = 12`, design `Pfa = 1e-3`. A 20 dB Swerling I target sits in
//! the CUT; `r` Swerling I interferers of 20 dB occupy the reference cells
//! nearest to it (alternating sides).
//!
//! Columns:
//!
//! - CA Pd measured, and the **exact** closed form from
//!   `theory::ca_cfar::exceedance_probability` (heterogeneous product).
//! - OS Pd measured, and the **large-INR approximation**: if every
//!   interferer is far above the noise, `X₍ₖ₎` is the `k`-th smallest of the
//!   `N - r` noise cells, so the homogeneous OS formula applies with
//!   `N → N - r`. Only meaningful for `r ≤ N - k`; beyond that the `k`-th
//!   smallest cell *is* an interferer and OS-CFAR breaks down too.
//! - Pfa (no target in the CUT): CA exact theory (it is far too small to
//!   measure), OS measured with its large-INR approximation.
//!
//! Intervals are 95 % Clopper–Pearson. This is a report experiment: it
//! quantifies degradation relative to the homogeneous case rather than
//! asserting a closed form for OS-CFAR.

use cfar_bench::cfar::{CaCfar, CfarWindow, OsCfar};
use cfar_bench::sim::Scene;
use cfar_bench::stats::{BinomialCount, Confidence};
use cfar_bench::theory::{ca_cfar, os_cfar};
use cfar_bench::units::db_to_linear;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

const PER_SIDE: usize = 8;
const GUARD: usize = 1;
const RANK: usize = 12;
const DESIGN_PFA: f64 = 1e-3;
const TARGET_SNR_DB: f64 = 20.0;
const INR_DB: f64 = 20.0;
const MAX_INTERFERERS: usize = 8;
const PD_TRIALS: u64 = 100_000;
const PFA_TRIALS: u64 = 1_000_000;

struct Counts {
    ca: u64,
    os: u64,
}

fn run(scene: &Scene, ca: &CaCfar, os: &OsCfar, trials: u64, seed: u64) -> Counts {
    let window = ca.window();
    let cut = window.half_width();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut power = vec![0.0; scene.len()];
    let mut reference = Vec::with_capacity(window.reference_cells());
    let mut counts = Counts { ca: 0, os: 0 };
    for _ in 0..trials {
        scene.draw_power(&mut rng, &mut power);
        let (lag, lead) = window.reference_halves(&power, cut);
        reference.clear();
        reference.extend_from_slice(lag);
        reference.extend_from_slice(lead);
        counts.ca += u64::from(ca.detects(power[cut], &reference));
        counts.os += u64::from(os.detects(power[cut], &reference));
    }
    counts
}

/// Point estimate with 95 % Clopper–Pearson interval, each number rendered
/// by `num` (fixed-point for Pd, scientific for Pfa).
fn fmt_ci(hits: u64, trials: u64, num: fn(f64) -> String) -> String {
    let conf = Confidence::from_level(0.95).expect("valid level");
    let ci = BinomialCount::new(hits, trials)
        .expect("valid count")
        .clopper_pearson(conf)
        .expect("converges");
    format!(
        "{} [{},{}]",
        num(hits as f64 / trials as f64),
        num(ci.lower),
        num(ci.upper)
    )
}

fn fixed(v: f64) -> String {
    format!("{v:.4}")
}

fn sci(v: f64) -> String {
    format!("{v:.2e}")
}

fn main() {
    let window = CfarWindow::new(PER_SIDE, GUARD).expect("valid window");
    let n = window.reference_cells();
    let ca_alpha = ca_cfar::multiplier_for_pfa(DESIGN_PFA, n);
    let os_alpha = os_cfar::multiplier_for_pfa(DESIGN_PFA, n, RANK);
    let ca = CaCfar::new(window, ca_alpha).expect("valid CA");
    let os = OsCfar::new(window, RANK, os_alpha).expect("valid OS");

    let len = 2 * window.half_width() + 1;
    let cut = window.half_width();
    let positions: Vec<usize> = (1..=PER_SIDE)
        .flat_map(|d| [cut + GUARD + d, cut - GUARD - d])
        .collect();
    let snr = db_to_linear(TARGET_SNR_DB);
    let inr = db_to_linear(INR_DB);

    println!(
        "N = {n}, G = {GUARD}, OS k = {RANK}, Pfa = {DESIGN_PFA:e}, target {TARGET_SNR_DB} dB, \
         interferers {INR_DB} dB each\n"
    );
    println!(
        "{:>2} | {:>24} {:>7} | {:>24} {:>7} | {:>9} | {:>30} {:>9}",
        "r",
        "CA Pd meas [95% CI]",
        "exact",
        "OS Pd meas [95% CI]",
        "approx",
        "CA Pfa",
        "OS Pfa meas [95% CI]",
        "approx"
    );

    for r in 0..=MAX_INTERFERERS {
        let mut background = Scene::homogeneous(len, 1.0).expect("valid scene");
        for &p in &positions[..r] {
            background = background.with_target(p, inr).expect("valid interferer");
        }
        let with_target = background
            .clone()
            .with_target(cut, snr)
            .expect("valid target");

        // Exact CA theory from per-cell mean powers.
        let ca_exact = |scene: &Scene| {
            let powers = scene.mean_powers();
            let (lag, lead) = window.reference_halves(&powers, cut);
            let refs: Vec<f64> = lag.iter().chain(lead).copied().collect();
            ca_cfar::exceedance_probability(ca_alpha, powers[cut], &refs)
        };
        // OS large-INR approximation: homogeneous formula on N - r cells.
        // "—" once the k-th smallest cell must be an interferer.
        let os_approx = |target_snr: f64, num: fn(f64) -> String| {
            if RANK <= n - r {
                num(os_cfar::pd_swerling1(os_alpha, n - r, RANK, target_snr))
            } else {
                "—".to_string()
            }
        };

        let pd = run(&with_target, &ca, &os, PD_TRIALS, 70_000 + r as u64);
        let fa = run(&background, &ca, &os, PFA_TRIALS, 71_000 + r as u64);

        println!(
            "{r:>2} | {:>24} {:>7.4} | {:>24} {:>7} | {:>9.2e} | {:>30} {:>9}",
            fmt_ci(pd.ca, PD_TRIALS, fixed),
            ca_exact(&with_target),
            fmt_ci(pd.os, PD_TRIALS, fixed),
            os_approx(snr, fixed),
            ca_exact(&background),
            fmt_ci(fa.os, PFA_TRIALS, sci),
            os_approx(0.0, sci),
        );
    }
    println!(
        "\nOS-CFAR tolerates up to N - k = {} interferers; homogeneous theory: \
         CA Pd {:.4}, OS Pd {:.4}.",
        n - RANK,
        ca_cfar::pd_swerling1(ca_alpha, n, snr),
        os_cfar::pd_swerling1(os_alpha, n, RANK, snr)
    );
}
