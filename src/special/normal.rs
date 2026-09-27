//! Standard normal quantile (inverse CDF).

/// Quantile of the standard normal distribution: returns `z` such that
/// `Φ(z) = p`.
///
/// Returns `-∞` at `p = 0`, `+∞` at `p = 1`, and NaN outside `[0, 1]`.
///
/// Wichura's algorithm AS 241 (`PPND16`, Applied Statistics 37(3), 1988):
/// three rational approximations over different ranges of `p`, with
/// relative accuracy about 1e-16.
///
/// Typical use: the two-sided critical value for confidence `1 - α` is
/// `normal_quantile(1 - α/2)`.
#[must_use]
pub fn normal_quantile(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    // p is now in [0, 1], so these are exact endpoint checks.
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    let q = p - 0.5;

    // Central region |q| ≤ 0.425.
    if q.abs() <= 0.425 {
        let r = 0.180_625 - q * q;
        return q * poly(&as241::CENTRAL_NUM, r) / poly(&as241::CENTRAL_DEN, r);
    }

    // Tails: work with the smaller of p and 1-p, then restore the sign.
    let tail = if q < 0.0 { p } else { 1.0 - p };
    let r = (-tail.ln()).sqrt();
    let z = if r <= 5.0 {
        let r = r - 1.6;
        poly(&as241::NEAR_TAIL_NUM, r) / poly(&as241::NEAR_TAIL_DEN, r)
    } else {
        let r = r - 5.0;
        poly(&as241::FAR_TAIL_NUM, r) / poly(&as241::FAR_TAIL_DEN, r)
    };
    if q < 0.0 { -z } else { z }
}

/// Evaluates `c[0] + c[1] x + ... + c[n] x^n` by Horner's rule.
fn poly(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Coefficients of AS 241, lowest order first.
///
/// They are kept exactly as printed in the paper (more digits than an `f64`
/// holds) so they can be audited line by line against the source; the
/// compiler rounds each to the nearest `f64`. The inner attribute below
/// silences `excessive_precision` for this module only.
mod as241 {
    #![allow(clippy::excessive_precision)]

    pub(super) const CENTRAL_NUM: [f64; 8] = [
        3.387_132_872_796_366_608,
        1.331_416_678_917_843_774_5e2,
        1.971_590_950_306_551_442_7e3,
        1.373_169_376_550_946_112_5e4,
        4.592_195_393_154_987_145_7e4,
        6.726_577_092_700_870_085_3e4,
        3.343_057_558_358_812_810_5e4,
        2.509_080_928_730_122_672_7e3,
    ];
    pub(super) const CENTRAL_DEN: [f64; 8] = [
        1.0,
        4.231_333_070_160_091_125_2e1,
        6.871_870_074_920_579_083e2,
        5.394_196_021_424_751_107_7e3,
        2.121_379_430_158_659_586_7e4,
        3.930_789_580_009_271_061e4,
        2.872_908_573_572_194_267_4e4,
        5.226_495_278_852_854_561e3,
    ];
    pub(super) const NEAR_TAIL_NUM: [f64; 8] = [
        1.423_437_110_749_683_577_34,
        4.630_337_846_156_545_295_9,
        5.769_497_221_460_691_405_5,
        3.647_848_324_763_204_605_04,
        1.270_458_252_452_368_382_58,
        2.417_807_251_774_506_117_7e-1,
        2.272_384_498_926_918_458_33e-2,
        7.745_450_142_783_414_076_4e-4,
    ];
    pub(super) const NEAR_TAIL_DEN: [f64; 8] = [
        1.0,
        2.053_191_626_637_758_821_87,
        1.676_384_830_183_803_849_4,
        6.897_673_349_851_000_045_5e-1,
        1.481_039_764_274_800_745_9e-1,
        1.519_866_656_361_645_719_66e-2,
        5.475_938_084_995_344_946e-4,
        1.050_750_071_644_416_843_24e-9,
    ];
    pub(super) const FAR_TAIL_NUM: [f64; 8] = [
        6.657_904_643_501_103_777_2,
        5.463_784_911_164_114_369_9,
        1.784_826_539_917_291_335_8,
        2.965_605_718_285_048_912_3e-1,
        2.653_218_952_657_612_309_3e-2,
        1.242_660_947_388_078_438_6e-3,
        2.711_555_568_743_487_578_15e-5,
        2.010_334_399_292_288_132_65e-7,
    ];
    pub(super) const FAR_TAIL_DEN: [f64; 8] = [
        1.0,
        5.998_322_065_558_879_376_9e-1,
        1.369_298_809_227_358_053_1e-1,
        1.487_536_129_085_061_485_25e-2,
        7.868_691_311_456_132_591e-4,
        1.846_318_317_510_054_681_8e-5,
        1.421_511_758_316_445_888_7e-7,
        2.044_263_103_389_939_785_64e-15,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_scipy_reference_values() {
        // scipy.special.ndtri; covers all three branches of AS 241.
        let cases = [
            (0.975, 1.959_963_984_540_054),
            (0.995, 2.575_829_303_548_900_4),
            (0.9995, 3.290_526_731_491_925_5),
            (0.3, -0.524_400_512_708_040_9),
            (0.024_25, -1.972_961_051_311_885),
            (1e-10, -6.361_340_902_404_056),
            (1e-300, -37.047_096_299_361_2),
        ];
        for (p, expected) in cases {
            let got = normal_quantile(p);
            assert!(
                ((got - expected) / expected).abs() < 1e-14,
                "p = {p}: {got} vs {expected}"
            );
        }
        assert!(normal_quantile(0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn is_antisymmetric() {
        for p in [1e-8, 0.01, 0.2, 0.4999] {
            assert!((normal_quantile(p) + normal_quantile(1.0 - p)).abs() < 1e-9);
        }
    }

    #[test]
    fn endpoints_and_domain() {
        let lo = normal_quantile(0.0);
        let hi = normal_quantile(1.0);
        assert!(lo.is_infinite() && lo.is_sign_negative());
        assert!(hi.is_infinite() && hi.is_sign_positive());
        assert!(normal_quantile(-0.1).is_nan());
        assert!(normal_quantile(1.1).is_nan());
        assert!(normal_quantile(f64::NAN).is_nan());
    }
}
