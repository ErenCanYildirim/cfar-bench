//! The exponential distribution: the law of square-law detected
//! complex Gaussian noise.
//!
//! If `x ~ CN(0, P)` then `y = |x|²` is exponential with mean `P`:
//! `f(y) = (1/P) exp(-y/P)` for `y ≥ 0`.

/// CDF of an exponential distribution with the given mean.
///
/// Uses `-expm1(-y/P)` rather than `1 - exp(-y/P)` to keep full relative
/// precision for small `y`, where `exp(-y/P)` is close to 1.
#[must_use]
pub fn cdf(y: f64, mean: f64) -> f64 {
    if y <= 0.0 { 0.0 } else { -(-y / mean).exp_m1() }
}

/// Raw moment `E[y^k] = k! · P^k`.
#[must_use]
pub fn raw_moment(k: u32, mean: f64) -> f64 {
    factorial(k) * mean.powf(f64::from(k))
}

/// Variance of `y^k`, i.e. `E[y^{2k}] - E[y^k]² = ((2k)! - (k!)²) · P^{2k}`.
///
/// Divided by the sample size this gives the exact variance of the sample
/// estimator of the `k`-th raw moment, which sets the test tolerance.
#[must_use]
pub fn raw_moment_variance(k: u32, mean: f64) -> f64 {
    raw_moment(2 * k, mean) - raw_moment(k, mean).powi(2)
}

fn factorial(k: u32) -> f64 {
    (1..=k).map(f64::from).product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdf_basic_values() {
        assert!(cdf(-1.0, 2.0).abs() < f64::EPSILON);
        assert!(cdf(0.0, 2.0).abs() < f64::EPSILON);
        // Median of Exp(mean P) is P ln 2.
        assert!((cdf(2.0 * std::f64::consts::LN_2, 2.0) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn moments_of_unit_exponential() {
        // E[y] = 1, E[y²] = 2, E[y³] = 6, Var(y) = 1.
        assert!((raw_moment(1, 1.0) - 1.0).abs() < 1e-15);
        assert!((raw_moment(2, 1.0) - 2.0).abs() < 1e-15);
        assert!((raw_moment(3, 1.0) - 6.0).abs() < 1e-15);
        assert!((raw_moment_variance(1, 1.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn moments_scale_with_mean() {
        // E[y^k] scales as P^k.
        assert!((raw_moment(3, 2.0) - 48.0).abs() < 1e-12);
    }
}
