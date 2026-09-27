//! Regularized incomplete beta function.

use super::ln_gamma;

/// Regularized incomplete beta function `I_x(a, b)`, for `a, b > 0` and
/// `0 ≤ x ≤ 1`. Returns NaN outside that domain or if the continued
/// fraction fails to converge.
///
/// This is the CDF of a `Beta(a, b)` random variable, and it is what links
/// the binomial distribution to exact confidence intervals:
///
/// `P(X ≥ k) = I_p(k, n - k + 1)` for `X ~ Binomial(n, p)`.
///
/// # Method
///
/// The continued fraction of Numerical Recipes §6.4, evaluated with the
/// modified Lentz algorithm. The fraction converges quickly for
/// `x < (a + 1) / (a + b + 2)`; above that point the symmetry
/// `I_x(a, b) = 1 - I_{1-x}(b, a)` is used instead. The number of
/// iterations needed grows like `√max(a, b)`, so the iteration cap scales
/// with the arguments.
#[must_use]
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    // NaN fails every comparison, so it must be checked explicitly: with
    // only `a <= 0.0` a NaN `a` would slip through.
    let shape_invalid = |s: f64| s.is_nan() || s <= 0.0;
    if shape_invalid(a) || shape_invalid(b) || !(0.0..=1.0).contains(&x) {
        return f64::NAN;
    }
    // x is now in [0, 1], so `<=`/`>=` here are exact endpoint checks,
    // without an `==` on floats.
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    // Prefactor x^a (1-x)^b / B(a, b), assembled in log space to avoid
    // overflow/underflow at large a, b. `ln_1p(-x)` keeps precision when x
    // is small, which is exactly the Pfa regime.
    let ln_prefactor = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (-x).ln_1p();
    let prefactor = ln_prefactor.exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        prefactor * continued_fraction(x, a, b) / a
    } else {
        1.0 - prefactor * continued_fraction(1.0 - x, b, a) / b
    }
}

/// Continued fraction for the incomplete beta (modified Lentz). Returns NaN
/// if it does not converge within the iteration cap.
fn continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const EPS: f64 = 1e-15;
    // Guards against division by zero in Lentz's method.
    const TINY: f64 = 1e-300;

    // Truncation to usize is intended: this is only an iteration budget.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let max_iter = 200 + (10.0 * a.max(b).sqrt()) as usize;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let clamp_tiny = |v: f64| if v.abs() < TINY { TINY } else { v };

    let mut c = 1.0;
    let mut d = 1.0 / clamp_tiny(1.0 - qab * x / qap);
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        // Even step of the recurrence.
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 / clamp_tiny(1.0 + aa * d);
        c = clamp_tiny(1.0 + aa / c);
        h *= d * c;

        // Odd step.
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 / clamp_tiny(1.0 + aa * d);
        c = clamp_tiny(1.0 + aa / c);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            return h;
        }
    }
    f64::NAN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scipy_reference_values() {
        // scipy.special.betainc(a, b, x)
        let cases = [
            (0.3, 2.5, 7.1, 0.649_022_579_021_013_2),
            (0.9, 0.5, 0.5, 0.795_167_235_300_866_5),
            (0.5, 30.0, 30.0, 0.5),
            // Pfa-regime arguments: p ~ 1e-4, n ~ 1e6.
            (1e-4, 100.0, 999_901.0, 0.513_300_791_444_927),
            (1.2e-4, 100.0, 999_901.0, 0.972_143_404_456_198_6),
        ];
        for (x, a, b, expected) in cases {
            let got = regularized_incomplete_beta(x, a, b);
            // Large-argument cases lose some digits through the ln Γ
            // prefactor (see ln_gamma docs); 1e-7 relative is far tighter
            // than any Monte Carlo interval we will compare against.
            assert!(
                ((got - expected) / expected).abs() < 1e-7,
                "I_{x}({a}, {b}) = {got}, expected {expected}"
            );
        }
    }

    #[test]
    fn closed_form_special_cases() {
        // I_x(a, 1) = x^a and I_x(1, b) = 1 - (1-x)^b.
        for &x in &[0.01, 0.2, 0.5, 0.93] {
            for &s in &[0.5, 1.0, 3.0, 40.0] {
                let a1 = regularized_incomplete_beta(x, s, 1.0);
                let b1 = regularized_incomplete_beta(x, 1.0, s);
                assert!((a1 - x.powf(s)).abs() < 1e-13, "x={x}, a={s}");
                assert!(
                    (b1 - (1.0 - (1.0 - x).powf(s))).abs() < 1e-13,
                    "x={x}, b={s}"
                );
            }
        }
    }

    #[test]
    fn symmetry_identity() {
        for &(x, a, b) in &[(0.2, 2.0, 5.0), (0.7, 0.8, 3.3), (0.45, 12.0, 11.0)] {
            let lhs = regularized_incomplete_beta(x, a, b);
            let rhs = 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
            assert!((lhs - rhs).abs() < 1e-13);
        }
    }

    #[test]
    fn equals_binomial_upper_tail() {
        // P(X ≥ k) for X ~ Bin(n, p), summed directly from the pmf. The pmf
        // is built by the recurrence pmf(j+1) = pmf(j)·(n-j)/(j+1)·p/(1-p),
        // which shares no code with the continued fraction.
        let n = 20_u32;
        for &p in &[0.05_f64, 0.3, 0.5, 0.81] {
            let mut pmf = vec![(1.0 - p).powi(n as i32)];
            for j in 0..n {
                let next = pmf[j as usize] * f64::from(n - j) / f64::from(j + 1) * p / (1.0 - p);
                pmf.push(next);
            }
            for k in 1..=n {
                let tail: f64 = pmf[k as usize..].iter().sum();
                let beta = regularized_incomplete_beta(p, f64::from(k), f64::from(n - k + 1));
                assert!((tail - beta).abs() < 1e-13, "n={n}, k={k}, p={p}");
            }
        }
    }

    #[test]
    fn endpoints_and_domain() {
        assert!(regularized_incomplete_beta(0.0, 2.0, 3.0).abs() < f64::EPSILON);
        assert!((regularized_incomplete_beta(1.0, 2.0, 3.0) - 1.0).abs() < f64::EPSILON);
        assert!(regularized_incomplete_beta(-0.1, 2.0, 3.0).is_nan());
        assert!(regularized_incomplete_beta(0.5, 0.0, 3.0).is_nan());
        assert!(regularized_incomplete_beta(0.5, 2.0, f64::NAN).is_nan());
    }
}
