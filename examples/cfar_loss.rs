//! Measured vs theoretical CA-CFAR loss for a Swerling I target.
//!
//! Run with: `cargo run --release --example cfar_loss`
//!
//! # Definition
//!
//! CFAR loss is the extra SNR CA-CFAR needs to reach a given Pd, compared
//! with a fixed threshold set from the exactly known noise power:
//!
//! `L = S_CA(Pd*) / S_known(Pd*)`, reported in dB.
//!
//! # Measurement
//!
//! 1. For each trial draw a unit target amplitude `g ~ CN(0, 1)`, CUT noise
//!    `n` and `N_max` reference cells, once.
//! 2. Sweep SNR over a fine dB grid, forming `CUT(S) = |√(S·P)·g + n|²` from
//!    the *same* draws at every grid point, and record each detector's
//!    decision. Using the first `N` reference cells for the `N`-cell
//!    detector means every detector also sees the same draws.
//! 3. Linearly interpolate the SNR (in dB) at which measured Pd first
//!    reaches `Pd*`.
//! 4. Repeat in `REPLICATIONS` independent replications; report the mean
//!    and standard error of the per-replication loss.
//!
//! Reusing draws across SNRs and detectors (common random numbers) makes
//! the measured curves smooth and makes the two detectors' errors largely
//! cancel in the loss, which is a difference of two nearby SNRs.

use cfar_bench::cfar::{CaCfar, CfarWindow, FixedThreshold};
use cfar_bench::sim::ComplexAwgn;
use cfar_bench::stats;
use cfar_bench::theory::{ca_cfar, known_noise};
use cfar_bench::units::{db_to_linear, linear_to_db};
use num_complex::Complex64;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::ChaCha8Rng;

const WINDOW_SIZES: [usize; 5] = [4, 8, 16, 32, 64];
const PFAS: [f64; 2] = [1e-4, 1e-6];
const TARGET_PD: f64 = 0.9;
const NOISE_POWER: f64 = 1.0;
const REPLICATIONS: u64 = 20;
const TRIALS_PER_REPLICATION: usize = 20_000;
const GRID_STEP_DB: f64 = 0.05;
/// Margin around the theoretical crossings covered by the SNR grid.
const GRID_MARGIN_DB: f64 = 1.5;

/// SNR (dB) at which `pd_curve` first reaches `target`, by linear
/// interpolation between the two bracketing grid points.
fn crossing_db(grid_db: &[f64], pd_curve: &[f64], target: f64) -> Option<f64> {
    let k = pd_curve.iter().position(|&pd| pd >= target)?;
    if k == 0 {
        return None; // grid starts above the crossing
    }
    let (x0, x1) = (grid_db[k - 1], grid_db[k]);
    let (y0, y1) = (pd_curve[k - 1], pd_curve[k]);
    Some(x0 + (target - y0) * (x1 - x0) / (y1 - y0))
}

struct Detectors {
    ca: Vec<CaCfar>,
    known: FixedThreshold,
}

impl Detectors {
    fn new(pfa: f64) -> Self {
        let ca = WINDOW_SIZES
            .iter()
            .map(|&n| {
                let window = CfarWindow::new(n / 2, 0).expect("valid window");
                CaCfar::new(window, ca_cfar::multiplier_for_pfa(pfa, n)).expect("valid alpha")
            })
            .collect();
        let known = FixedThreshold::new(known_noise::threshold_for_pfa(pfa, NOISE_POWER))
            .expect("valid threshold");
        Self { ca, known }
    }
}

/// One replication. Returns the measured required SNR in dB for the
/// known-noise detector and for each CA-CFAR window size.
fn replicate(detectors: &Detectors, grid_db: &[f64], seed: u64) -> (f64, Vec<f64>) {
    let n_max = *WINDOW_SIZES.iter().max().expect("non-empty");
    let noise = ComplexAwgn::new(NOISE_POWER).expect("valid power");
    let unit = ComplexAwgn::new(1.0).expect("valid power");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Target amplitude scale √(S·P) for each grid point.
    let scales: Vec<f64> = grid_db
        .iter()
        .map(|&db| (db_to_linear(db) * NOISE_POWER).sqrt())
        .collect();

    let mut known_hits = vec![0_u64; grid_db.len()];
    let mut ca_hits = vec![vec![0_u64; grid_db.len()]; WINDOW_SIZES.len()];
    let mut reference = vec![0.0; n_max];

    for _ in 0..TRIALS_PER_REPLICATION {
        let g: Complex64 = unit.sample(&mut rng);
        let n: Complex64 = noise.sample(&mut rng);
        for cell in &mut reference {
            *cell = noise.sample(&mut rng).norm_sqr();
        }
        for (k, &scale) in scales.iter().enumerate() {
            let cut = (g * scale + n).norm_sqr();
            if detectors.known.detects(cut) {
                known_hits[k] += 1;
            }
            for (d, hits) in detectors.ca.iter().zip(&mut ca_hits) {
                let n_ref = d.window().reference_cells();
                if d.detects(cut, &reference[..n_ref]) {
                    hits[k] += 1;
                }
            }
        }
    }

    let to_pd = |hits: &[u64]| -> Vec<f64> {
        hits.iter()
            .map(|&h| h as f64 / TRIALS_PER_REPLICATION as f64)
            .collect()
    };
    let known_db =
        crossing_db(grid_db, &to_pd(&known_hits), TARGET_PD).expect("grid covers crossing");
    let ca_db = ca_hits
        .iter()
        .map(|h| crossing_db(grid_db, &to_pd(h), TARGET_PD).expect("grid covers crossing"))
        .collect();
    (known_db, ca_db)
}

fn main() {
    println!("CA-CFAR loss, Swerling I, Pd = {TARGET_PD}");
    println!(
        "{REPLICATIONS} replications x {TRIALS_PER_REPLICATION} trials, \
         {GRID_STEP_DB} dB grid\n"
    );
    println!(
        "{:>6} {:>4} | {:>9} {:>9} | {:>9} {:>9} | {:>8} {:>15} {:>6}",
        "Pfa", "N", "S_known", "meas", "S_CA", "meas", "L theory", "L measured", "z"
    );

    for (p_idx, &pfa) in PFAS.iter().enumerate() {
        let detectors = Detectors::new(pfa);

        let known_theory_db = linear_to_db(known_noise::required_snr_swerling1(pfa, TARGET_PD));
        let ca_theory_db: Vec<f64> = detectors
            .ca
            .iter()
            .map(|d| {
                let n = d.window().reference_cells();
                linear_to_db(ca_cfar::required_snr_swerling1(
                    d.multiplier(),
                    n,
                    TARGET_PD,
                ))
            })
            .collect();

        // Grid spanning all theoretical crossings with margin.
        let lo = known_theory_db - GRID_MARGIN_DB;
        let hi = ca_theory_db.iter().copied().fold(f64::MIN, f64::max) + GRID_MARGIN_DB;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let points = ((hi - lo) / GRID_STEP_DB).ceil() as usize + 1;
        let grid_db: Vec<f64> = (0..points).map(|i| lo + i as f64 * GRID_STEP_DB).collect();

        let runs: Vec<(f64, Vec<f64>)> = (0..REPLICATIONS)
            .map(|r| replicate(&detectors, &grid_db, 1_000 * (p_idx as u64 + 1) + r))
            .collect();

        let known_meas: Vec<f64> = runs.iter().map(|(k, _)| *k).collect();
        let known_mean = stats::mean(&known_meas).expect("non-empty");

        for (i, d) in detectors.ca.iter().enumerate() {
            let n = d.window().reference_cells();
            let ca_meas: Vec<f64> = runs.iter().map(|(_, ca)| ca[i]).collect();
            let losses: Vec<f64> = runs.iter().map(|(k, ca)| ca[i] - k).collect();

            let loss_mean = stats::mean(&losses).expect("non-empty");
            let loss_se = stats::standard_error_of_mean(&losses).expect("two or more");
            let loss_theory = ca_theory_db[i] - known_theory_db;

            println!(
                "{pfa:>6.0e} {n:>4} | {known_theory_db:>9.3} {known_mean:>9.3} | \
                 {:>9.3} {:>9.3} | {loss_theory:>8.3} {loss_mean:>8.3} ± {loss_se:<5.3} {:>6.2}",
                ca_theory_db[i],
                stats::mean(&ca_meas).expect("non-empty"),
                (loss_mean - loss_theory) / loss_se,
            );
        }
        // The absolute required SNRs are far noisier than the loss: every row
        // shares the same draws, so they shift together between runs, and
        // that shared shift cancels in the loss. Print their SE to show it.
        let known_se = stats::standard_error_of_mean(&known_meas).expect("two or more");
        println!(
            "       S_known: measured {known_mean:.3} ± {known_se:.3} dB, theory \
             {known_theory_db:.3} dB, z = {:.2}\n",
            (known_mean - known_theory_db) / known_se
        );
    }
    println!("All SNRs in dB. z = (measured - theory) / SE of the measured loss.");
}
