//! Independent single-cell detection trials.

use num_complex::Complex64;
use rand::Rng;
use rand::distr::Distribution;

use super::ComplexAwgn;

/// Draws one independent detection trial and returns the square-law
/// detected cell under test.
///
/// - `reference` is overwritten with fresh noise powers `|n|²`. Its length
///   sets the number of reference cells; an empty slice gives a trial for a
///   detector with no reference window (e.g. a fixed threshold).
/// - The returned CUT is `|a + n|²`, with `a` drawn from `target` and `n`
///   from `noise`, independently of the reference cells.
///
/// Generic over the target model, so the same function serves Swerling I
/// here and other target models later. Pass a zero-power target (e.g.
/// `SwerlingOne::new(0.0)`) for noise-only trials.
///
/// The caller owns and reuses the `reference` buffer, so the hot loop of a
/// Monte Carlo run does not allocate.
pub fn draw_cell_trial<R, T>(
    rng: &mut R,
    noise: &ComplexAwgn,
    target: &T,
    reference: &mut [f64],
) -> f64
where
    R: Rng + ?Sized,
    T: Distribution<Complex64>,
{
    for cell in reference.iter_mut() {
        *cell = noise.sample(rng).norm_sqr();
    }
    (target.sample(rng) + noise.sample(rng)).norm_sqr()
}
