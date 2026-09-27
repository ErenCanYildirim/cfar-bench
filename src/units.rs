//! Decibel conversions for **power** quantities (SNR, noise power, loss).
//!
//! `dB = 10 log10(ratio)`. Amplitude ratios use `20 log10`; mixing the two
//! is a classic bug (a "15 dB" target generated with `10^(15/20)` has only
//! 7.5 dB of power SNR). All SNRs in this crate are power ratios.

/// Power ratio from decibels: `10^(dB/10)`.
#[must_use]
pub fn db_to_linear(db: f64) -> f64 {
    10f64.powf(db / 10.0)
}

/// Decibels from a power ratio: `10 log10(ratio)`.
#[must_use]
pub fn linear_to_db(ratio: f64) -> f64 {
    10.0 * ratio.log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 1e-15);
        assert!((db_to_linear(10.0) - 10.0).abs() < 1e-12);
        assert!((db_to_linear(3.0) - 1.995_262_314_968_879_5).abs() < 1e-12);
        assert!((linear_to_db(100.0) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn round_trip() {
        for db in [-30.0, -3.0, 0.0, 7.5, 42.0] {
            assert!((linear_to_db(db_to_linear(db)) - db).abs() < 1e-12);
        }
    }
}
