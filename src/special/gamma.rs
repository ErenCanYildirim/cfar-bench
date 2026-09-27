//! Log-gamma function.

use std::f64::consts::PI;

/// Lanczos approximation coefficients for `g = 7`, `n = 9`.
/// Relative accuracy is about 1e-15 for `ln Γ(x)`.
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEF: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];

/// Natural logarithm of the gamma function ln gamma(x), for x > 0
/// returns NaN for x <= 0, or NaN input
/// uses the Lanczos approximation, with the reflection formula
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    if x.is_nan() || x <= 0.0 {
        return f64::NAN;
    }

    if x < 0.5 {
        // ln Γ(x) = ln(π / sin(πx)) - ln Γ(1 - x). For 0 < x < 0.5,
        // sin(πx) > 0, so the log is defined.
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let t = x + LANCZOS_G + 0.5;
    let series = LANCZOS_COEF[1..]
        .iter()
        .enumerate()
        .fold(LANCZOS_COEF[0], |acc, (i, &c)| {
            acc + c / (x + (i + 1) as f64)
        });

    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel_err(a: f64, b: f64) -> f64 {
        ((a - b) / b).abs()
    }

    #[test]
    fn matches_scipy_reference_values() {
        // scipy.special.gammaln
        let cases = [
            (0.5, 0.572_364_942_924_700_1),
            (3.7, 1.428_072_326_665_388),
            (10.0, 12.801_827_480_081_469),
            (1e6, 12_815_504.569_147_611),
            (1e-3, 6.907_178_885_383_853),
        ];
        for (x, expected) in cases {
            let got = ln_gamma(x);
            assert!(
                rel_err(got, expected) < 1e-13,
                "x = {x}: {got} vs {expected}"
            );
        }
    }

    #[test]
    fn integer_arguments_are_log_factorials() {
        // Γ(n + 1) = n!
        let mut log_fact = 0.0_f64;
        for n in 1..=30_u32 {
            log_fact += f64::from(n).ln();
            let got = ln_gamma(f64::from(n) + 1.0);
            assert!(
                (got - log_fact).abs() < 1e-12 * log_fact.max(1.0),
                "n = {n}"
            );
        }
        assert!(ln_gamma(1.0).abs() < 1e-15);
        assert!(ln_gamma(2.0).abs() < 1e-15);
    }

    #[test]
    fn recurrence_holds_across_reflection_boundary() {
        // ln Γ(x + 1) = ln Γ(x) + ln x, checked on both sides of x = 0.5.
        for x in [0.1, 0.3, 0.49, 0.51, 0.7, 2.3] {
            let lhs = ln_gamma(x + 1.0);
            let rhs = ln_gamma(x) + f64::ln(x);
            assert!((lhs - rhs).abs() < 1e-13, "x = {x}");
        }
    }

    #[test]
    fn out_of_domain_is_nan() {
        assert!(ln_gamma(0.0).is_nan());
        assert!(ln_gamma(-1.5).is_nan());
        assert!(ln_gamma(f64::NAN).is_nan());
    }
}
