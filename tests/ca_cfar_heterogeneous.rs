//! CA-CFAR in a heterogeneous background: Monte Carlo vs the closed-form
//! product formula `∏ᵢ (1 + α pᵢ / (N p_c))^(-1)`
//! (`theory::ca_cfar::exceedance_probability`).
//!
//! # Method
//!
//! Independent trials, each drawing a fresh window-sized [`Scene`] with the
//! CUT at its centre. Detection counts are binomial; Clopper–Pearson
//! intervals under one Bonferroni family.
//!
//! # Assertions (one family)
//!
//! Window `N = 16`, one guard cell per side, design `Pfa = 1e-3`:
//!
//! - Interfering targets: Pd of a 20 dB Swerling I target with 1, 2 and 4
//!   Swerling I interferers of 20 dB in the reference window (masking).
//! - Clutter edge (20 dB clutter-to-noise): Pfa for a CUT in clutter at 0,
//!   3 and 7 cells past the edge, where part of the window still sees only
//!   noise (false alarm spike).
//!
//! Very small probabilities (masked Pfa, down to 1e-16) are left to the
//! report experiments: they are correct in theory but too expensive to
//! measure in CI.

use cfar_bench::cfar::{CaCfar, CfarWindow};
use cfar_bench::sim::Scene;
use cfar_bench::stats::{BinomialCount, TestFamily, trials_for_relative_precision};
use cfar_bench::theory::ca_cfar;
use cfar_bench::units::db_to_linear;
use rand::SeedableRng;
use rand::rngs::ChaCha8Rng;

const PER_SIDE: usize = 8;
const GUARD: usize = 1;
const DESIGN_PFA: f64 = 1e-3;
const TARGET_SNR_DB: f64 = 20.0;
const INR_DB: f64 = 20.0;
const CNR_DB: f64 = 20.0;
const INTERFERER_COUNTS: [usize; 3] = [1, 2, 4];
/// CUT position relative to the first clutter cell.
const EDGE_OFFSETS: [usize; 3] = [0, 3, 7];
const FAMILY_ALPHA: f64 = 1e-3;
const PRECISION: f64 = 0.1;

fn family() -> TestFamily {
    TestFamily::bonferroni(FAMILY_ALPHA, INTERFERER_COUNTS.len() + EDGE_OFFSETS.len())
        .expect("valid family")
}

fn window() -> CfarWindow {
    CfarWindow::new(PER_SIDE, GUARD).expect("valid window")
}

fn detector() -> CaCfar {
    let n = window().reference_cells();
    CaCfar::new(window(), ca_cfar::multiplier_for_pfa(DESIGN_PFA, n)).expect("valid alpha")
}

/// Scene length: exactly one window, CUT in the middle.
fn scene_len() -> usize {
    2 * window().half_width() + 1
}

fn cut_index() -> usize {
    window().half_width()
}

/// Theory for this scene, straight from its per-cell mean powers.
fn theory(scene: &Scene) -> f64 {
    let powers = scene.mean_powers();
    let (lag, lead) = window().reference_halves(&powers, cut_index());
    let reference: Vec<f64> = lag.iter().chain(lead).copied().collect();
    ca_cfar::exceedance_probability(detector().multiplier(), powers[cut_index()], &reference)
}

fn check(scene: &Scene, seed: u64, label: &str) {
    let expected = theory(scene);
    let trials =
        trials_for_relative_precision(expected.min(1.0 - expected), PRECISION, family().per_test())
            .expect("valid plan");

    let d = detector();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut power = vec![0.0; scene.len()];
    let mut reference = Vec::with_capacity(window().reference_cells());
    let hits = (0..trials)
        .filter(|_| {
            scene.draw_power(&mut rng, &mut power);
            let (lag, lead) = window().reference_halves(&power, cut_index());
            reference.clear();
            reference.extend_from_slice(lag);
            reference.extend_from_slice(lead);
            d.detects(power[cut_index()], &reference)
        })
        .count() as u64;

    let ci = BinomialCount::new(hits, trials)
        .expect("valid count")
        .clopper_pearson(family().per_test())
        .expect("interval converges");
    assert!(
        ci.contains(expected),
        "{label}: measured {:.4e} ({hits}/{trials}), theory {expected:.4e}, \
         CI [{:.4e}, {:.4e}]",
        hits as f64 / trials as f64,
        ci.lower,
        ci.upper
    );
}

#[test]
fn interfering_targets_mask_as_predicted() {
    let cut = cut_index();
    // Interferers fill the reference cells nearest the CUT, alternating
    // sides: just past the guard cell on the leading side, then lagging, ...
    let positions: Vec<usize> = (1..=PER_SIDE)
        .flat_map(|d| [cut + GUARD + d, cut - GUARD - d])
        .collect();
    for (i, &r) in INTERFERER_COUNTS.iter().enumerate() {
        let mut scene = Scene::homogeneous(scene_len(), 1.0)
            .and_then(|s| s.with_target(cut, db_to_linear(TARGET_SNR_DB)))
            .expect("valid scene");
        for &p in &positions[..r] {
            scene = scene
                .with_target(p, db_to_linear(INR_DB))
                .expect("valid interferer");
        }
        check(
            &scene,
            60_000 + i as u64,
            &format!("Pd with {r} interferers"),
        );
    }
}

#[test]
fn clutter_edge_false_alarm_spike_as_predicted() {
    for (i, &offset) in EDGE_OFFSETS.iter().enumerate() {
        // CUT sits `offset` cells past the first clutter cell.
        let edge = cut_index() - offset;
        let scene = Scene::homogeneous(scene_len(), 1.0)
            .and_then(|s| s.with_clutter_edge(edge, db_to_linear(CNR_DB)))
            .expect("valid scene");
        check(
            &scene,
            61_000 + i as u64,
            &format!("Pfa, CUT {offset} cells into clutter"),
        );
    }
}
