//! One-sample Kolmogorov–Smirnov goodness-of-fit test.
 
use std::f64::consts::PI;
 
use super::StatsError;

/// Result of a one-sample KS test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsResult {
    /// `D_n = sup_x |F_n(x) - F(x)|`.
    pub statistic: f64,
    /// Asymptotic p-value under the null hypothesis that the samples come
    /// from `F`.
    pub p_value: f64,
    /// Number of samples.
    pub n: usize,
}

/// One-sample KS test of `samples` against a fully specified continuous CDF.
///
/// The CDF must be specified **a priori**. If its parameters were estimated
/// from the same samples, `D_n` is biased low and the p-value is invalid
/// (that case needs the Lilliefors correction instead).
///
/// The p-value uses the Kolmogorov limiting distribution with Stephens'
/// finite-sample correction, `λ = (√n + 0.12 + 0.11/√n) · D_n`, which is
/// accurate for `n` of a few tens and above.
///
/// # Errors
///
/// - [`StatsError::InsufficientSamples`] if `samples` is empty.
/// - [`StatsError::NonFinite`] if any sample is NaN or infinite.
pub fn ks_one_sample<F>(samples: &[f64], cdf: F) -> Result<KsResult, StatsError>
where
    F: Fn(f64) -> f64,
{
    if samples.is_empty() {
        return Err(StatsError::InsufficientSamples { needed: 1, got: 0 });
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::NonFinite);
    }
 
    // Sort a copy so the caller's data is left untouched. `f64` is not `Ord`
    // (because of NaN), so `sort()` does not compile; `total_cmp` supplies a
    // total order. We have already rejected NaN above.
    let mut sorted = samples.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);
 
    let n = sorted.len() as f64;
    // The empirical CDF jumps from (i)/n to (i+1)/n at the i-th order
    // statistic (0-based), so the supremum is attained at one of the two
    // sides of each jump.
    let statistic = sorted
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            let f = cdf(x);
            let below = f - i as f64 / n;
            let above = (i + 1) as f64 / n - f;
            below.max(above)
        })
        .fold(0.0_f64, f64::max);
 
    let sqrt_n = n.sqrt();
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * statistic;
 
    Ok(KsResult {
        statistic,
        p_value: kolmogorov_survival(lambda),
        n: sorted.len(),
    })
}

/// Survival function Kolmogorov distribution: Q(lambda) = P(K > lambda)
/// two series used as each converges quickly in a different range
#[must_use]
pub fn kolmogorov_survival(lambda: f64) -> f64 {
    const TERMS: i32 = 100;
    const TOL : f64 = 1e-16;

    if lambda <= 0.0 {
        return 1.0;
    }

    if lambda < 1.18 {
        let mut cdf = 0.0;
        for j in 1..=TERMS {
            let k = f64::from(2 * j - 1);
            let term = (-(k * k) * PI * PI / (8.0 * lambda * lambda)).exp();
            cdf += term;
            if term < TOL * cdf {
                break;
            }
        }
        cdf *= (2.0 * PI).sqrt() / lambda;
        return (1.0 - cdf).clamp(0.0, 1.0);
    }

    let mut q = 0.0;
    let mut sign = 1.0;
    for j in 1..=TERMS {
        let jf = f64::from(j);
        let term = (-2.0 * jf * jf * lambda * lambda).exp();
        q += sign * term;
        if term < TOL * q.abs() {
            break;
        }
        sign = -sign;
    }
    (2.0 * q).clamp(0.0, 1.0)
}


#[cfg(test)]
mod tests {
    use super::*;
 
    fn uniform_cdf(x: f64) -> f64 {
        x.clamp(0.0, 1.0)
    }
 
    #[test]
    fn statistic_of_hand_computed_example() {
        // Samples {0.1, 0.5, 0.9} against U(0,1).
        // Candidate gaps: |0.1-0|, |1/3-0.1|, |1/3-0.5|, |2/3-0.5|,
        // |2/3-0.9|, |1-0.9|. The largest is 1/3 - 0.1 = 0.2333...
        let r = ks_one_sample(&[0.9, 0.1, 0.5], uniform_cdf).unwrap();
        assert!((r.statistic - (1.0 / 3.0 - 0.1)).abs() < 1e-12);
        assert_eq!(r.n, 3);
    }
 
    #[test]
    fn survival_matches_tabulated_critical_values() {
        // Classical asymptotic critical values of the Kolmogorov distribution.
        assert!((kolmogorov_survival(1.2238) - 0.10).abs() < 5e-4);
        assert!((kolmogorov_survival(1.3581) - 0.05).abs() < 5e-4);
        assert!((kolmogorov_survival(1.6276) - 0.01).abs() < 5e-4);
    }
 
    #[test]
    fn survival_is_continuous_across_series_switch() {
        let lo = kolmogorov_survival(1.18 - 1e-9);
        let hi = kolmogorov_survival(1.18 + 1e-9);
        assert!((lo - hi).abs() < 1e-8, "{lo} vs {hi}");
    }
 
    #[test]
    fn survival_limits() {
        assert!((kolmogorov_survival(0.0) - 1.0).abs() < f64::EPSILON);
        assert!(kolmogorov_survival(0.1) > 0.999_999);
        assert!(kolmogorov_survival(5.0) < 1e-20);
    }
 
    #[test]
    fn rejects_empty_and_nan() {
        assert!(ks_one_sample(&[], uniform_cdf).is_err());
        assert_eq!(
            ks_one_sample(&[0.1, f64::NAN], uniform_cdf),
            Err(StatsError::NonFinite)
        );
    }
}