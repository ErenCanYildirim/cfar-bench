//! Sample moments and their standard errors.

use super::StatsError;

fn require_len(x: &[f64], needed: usize) -> Result<(), StatsError> {
    if x.len() < needed {
        return Err(StatsError::InsufficientSamples {
            needed,
            got: x.len(),
        });
    }
    Ok(())
}

/// Sample mean.
///
/// # Errors
///
/// [`StatsError::InsufficientSamples`] if `x` is empty.
pub fn mean(x: &[f64]) -> Result<f64, StatsError> {
    require_len(x, 1)?;
    Ok(x.iter().sum::<f64>() / x.len() as f64)
}

/// Unbiased sample variance (divides by `n - 1`).
///
/// Uses the two-pass algorithm: compute the mean first, then sum squared
/// deviations. This avoids the catastrophic cancellation of the one-pass
/// `E[x²] - E[x]²` formula.
///
/// # Errors
///
/// [`StatsError::InsufficientSamples`] if `x` has fewer than 2 elements.
pub fn variance(x: &[f64]) -> Result<f64, StatsError> {
    require_len(x, 2)?;
    let m = mean(x)?;
    let ss: f64 = x.iter().map(|&v| (v - m).powi(2)).sum();
    Ok(ss / (x.len() - 1) as f64)
}

/// Standard error of the sample mean, `s / √n`, estimated from the data.
///
/// # Errors
///
/// [`StatsError::InsufficientSamples`] if `x` has fewer than 2 elements.
pub fn standard_error_of_mean(x: &[f64]) -> Result<f64, StatsError> {
    Ok((variance(x)? / x.len() as f64).sqrt())
}

/// Pearson correlation coefficient between two paired samples.
///
/// # Errors
///
/// - [`StatsError::LengthMismatch`] if the inputs differ in length.
/// - [`StatsError::InsufficientSamples`] if there are fewer than 2 pairs.
/// - [`StatsError::ZeroVariance`] if either input is constant.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, StatsError> {
    if x.len() != y.len() {
        return Err(StatsError::LengthMismatch {
            left: x.len(),
            right: y.len(),
        });
    }
    require_len(x, 2)?;

    let mx = mean(x)?;
    let my = mean(y)?;

    // Fold the three sums into a single pass over the zipped pairs.
    let (sxy, sxx, syy) = x
        .iter()
        .zip(y)
        .fold((0.0, 0.0, 0.0), |(sxy, sxx, syy), (&a, &b)| {
            let (da, db) = (a - mx, b - my);
            (sxy + da * db, sxx + da * da, syy + db * db)
        });

    if sxx == 0.0 || syy == 0.0 {
        return Err(StatsError::ZeroVariance);
    }
    Ok(sxy / (sxx * syy).sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn mean_and_variance_of_known_data() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!(close(mean(&x).unwrap(), 5.0));
        // Sum of squared deviations is 32, n - 1 = 7.
        assert!(close(variance(&x).unwrap(), 32.0 / 7.0));
    }

    #[test]
    fn empty_and_short_inputs_are_errors() {
        assert_eq!(
            mean(&[]),
            Err(StatsError::InsufficientSamples { needed: 1, got: 0 })
        );
        assert_eq!(
            variance(&[1.0]),
            Err(StatsError::InsufficientSamples { needed: 2, got: 1 })
        );
    }

    #[test]
    fn correlation_of_linear_relations_is_plus_minus_one() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let up: Vec<f64> = x.iter().map(|v| 3.0 * v + 1.0).collect();
        let down: Vec<f64> = x.iter().map(|v| -0.5 * v).collect();
        assert!(close(pearson_correlation(&x, &up).unwrap(), 1.0));
        assert!(close(pearson_correlation(&x, &down).unwrap(), -1.0));
    }

    #[test]
    fn correlation_rejects_bad_inputs() {
        assert_eq!(
            pearson_correlation(&[1.0, 2.0], &[1.0]),
            Err(StatsError::LengthMismatch { left: 2, right: 1 })
        );
        assert_eq!(
            pearson_correlation(&[1.0, 1.0], &[1.0, 2.0]),
            Err(StatsError::ZeroVariance)
        );
    }
}