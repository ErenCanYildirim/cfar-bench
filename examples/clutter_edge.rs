//! CA- vs OS-CFAR false alarm rate across a clutter edge.
//!
//! Run with: `cargo run --release --example clutter_edge`
//!
//! Setup: a profile of noise (power 1) that steps up to clutter at
//! `CNR = 20 dB` at cell `EDGE`. No targets. Both detectors run along the
//! whole profile (`N = 16`, 1 guard cell per side, OS `k = 12`, design
//! `Pfa = 1e-3`), and the false alarm rate is measured per cell over many
//! independent profiles.
//!
//! Expected shape for CA-CFAR, given exactly by the heterogeneous product
//! formula (`theory::ca_cfar::exceedance_probability`):
//!
//! - **Just inside the clutter**: part of the window still sees noise, the
//!   mean underestimates the clutter level, the threshold is too low and
//!   Pfa spikes.
//! - **Just before the edge**: the window already contains clutter cells,
//!   the threshold is too high, and the noise cells are masked (Pfa ≪ design,
//!   and a weak target there would be missed).
//!
//! OS-CFAR has no simple closed form here, so it is measured. Its behaviour
//! is mixed, and the mechanism explains both halves:
//!
//! - **Before the edge** it masks far less: while at most `N - k` clutter
//!   cells are in the window, the `k`-th smallest cell is still a noise cell
//!   (the same effect as tolerating interfering targets).
//! - **At the edge** its spike is *larger* than CA's in this configuration:
//!   with half the window in clutter, the `k = 12`-th smallest of 16 cells is
//!   the 4th smallest of the 8 clutter cells, a low and high-variance
//!   estimate of the clutter level, so the threshold undershoots more often.
//!
//! OS-CFAR is designed for interfering targets, not clutter edges; variants
//! such as GO-CFAR (greatest-of the two half-windows) target the edge case.

use cfar_bench::cfar::{CaCfar, CfarWindow, OsCfar};
use cfar_bench::sim::Scene;
use cfar_bench::theory::{ca_cfar, os_cfar};
use cfar_bench::units::db_to_linear;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

const PER_SIDE: usize = 8;
const GUARD: usize = 1;
const RANK: usize = 12;
const DESIGN_PFA: f64 = 1e-3;
const CNR_DB: f64 = 20.0;
const LEN: usize = 80;
const EDGE: usize = 40;
const SHOW: i64 = 12;
const PROFILES: u64 = 300_000;

fn main() {
    let window = CfarWindow::new(PER_SIDE, GUARD).expect("valid window");
    let n = window.reference_cells();
    let ca_alpha = ca_cfar::multiplier_for_pfa(DESIGN_PFA, n);
    let ca = CaCfar::new(window, ca_alpha).expect("valid CA");
    let os = OsCfar::new(
        window,
        RANK,
        os_cfar::multiplier_for_pfa(DESIGN_PFA, n, RANK),
    )
    .expect("valid OS");
    let scene = Scene::homogeneous(LEN, 1.0)
        .and_then(|s| s.with_clutter_edge(EDGE, db_to_linear(CNR_DB)))
        .expect("valid scene");

    let mut ca_counts = vec![0_u64; LEN];
    let mut os_counts = vec![0_u64; LEN];
    let mut rng = ChaCha8Rng::seed_from_u64(80_000);
    let mut power = vec![0.0; LEN];
    for _ in 0..PROFILES {
        scene.draw_power(&mut rng, &mut power);
        for cell in ca.run(&power).detections {
            ca_counts[cell] += 1;
        }
        for cell in os.run(&power).detections {
            os_counts[cell] += 1;
        }
    }

    let powers = scene.mean_powers();
    let ca_theory = |cell: usize| {
        let (lag, lead) = window.reference_halves(&powers, cell);
        let refs: Vec<f64> = lag.iter().chain(lead).copied().collect();
        ca_cfar::exceedance_probability(ca_alpha, powers[cell], &refs)
    };

    println!(
        "Clutter edge at cell {EDGE}, CNR {CNR_DB} dB, N = {n}, G = {GUARD}, OS k = {RANK}, \
         design Pfa {DESIGN_PFA:e}, {PROFILES} profiles\n"
    );
    println!(
        "{:>6} {:>8} | {:>11} {:>11} | {:>11}",
        "offset", "region", "CA meas", "CA exact", "OS meas"
    );
    for offset in -SHOW..=SHOW {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let cell = (EDGE as i64 + offset) as usize;
        let rate = |c: &[u64]| c[cell] as f64 / PROFILES as f64;
        println!(
            "{offset:>6} {:>8} | {:>11.3e} {:>11.3e} | {:>11.3e}",
            if cell < EDGE { "noise" } else { "clutter" },
            rate(&ca_counts),
            ca_theory(cell),
            rate(&os_counts),
        );
    }

    let peak = |c: &[u64]| {
        let (cell, &max) = c
            .iter()
            .enumerate()
            .max_by_key(|&(_, v)| *v)
            .expect("non-empty");
        (cell as i64 - EDGE as i64, max as f64 / PROFILES as f64)
    };
    let (ca_at, ca_peak) = peak(&ca_counts);
    let (os_at, os_peak) = peak(&os_counts);
    println!(
        "\nPeak Pfa: CA {ca_peak:.3e} at offset {ca_at} ({:.0}x design), \
         OS {os_peak:.3e} at offset {os_at} ({:.0}x design)",
        ca_peak / DESIGN_PFA,
        os_peak / DESIGN_PFA
    );
    println!("offset = cell - edge; offset 0 is the first clutter cell.");
}
