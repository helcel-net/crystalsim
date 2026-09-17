//! Physical environment of the growing crystal.
//!
//! Everything here is expressed in the quantities used on snowcrystals.com:
//! temperature in °C and *supersaturation* as the excess water-vapour density
//! above ice saturation, in g/m³ — the axes of the Nakaya morphology diagram.
//!
//! The growth automaton in [`crate::automaton`] needs the surface physics
//! of Libbrecht's *Snow Crystals* (2021): the kinetic velocity and diffusion
//! length of water vapour in air, and the attachment coefficients of the
//! basal and prism facets — [`kinetics_params`] provides them.  The habit
//! names of the morphology diagram are kept as labels ([`Morphology`]) for
//! reporting what the diagram predicts at given conditions.

use std::fmt;

/// Specific gas constant of water vapour, J/(kg·K).
const R_V: f64 = 461.52;

fn kelvin(t_c: f64) -> f64 {
    t_c + 273.15
}

/// Saturation vapour pressure over ice, Pa (Murphy & Koop 2005, eq. 7).
pub fn p_sat_ice(t_c: f64) -> f64 {
    let t = kelvin(t_c);
    (9.550_426 - 5723.265 / t + 3.530_68 * t.ln() - 0.007_283_32 * t).exp()
}

/// Saturation vapour pressure over (supercooled) liquid water, Pa
/// (Murphy & Koop 2005, eq. 10).
pub fn p_sat_water(t_c: f64) -> f64 {
    let t = kelvin(t_c);
    (54.842_763 - 6763.22 / t - 4.210 * t.ln()
        + 0.000_367 * t
        + (0.0415 * (t - 218.8)).tanh() * (53.878 - 1331.22 / t - 9.445_23 * t.ln() + 0.014_025 * t))
        .exp()
}

/// Vapour density in g/m³ for a partial pressure `p` (Pa) at `t_c` °C.
pub fn vapor_density(p: f64, t_c: f64) -> f64 {
    p / (R_V * kelvin(t_c)) * 1000.0
}

/// Ice-saturation vapour density, g/m³.
pub fn rho_sat_ice(t_c: f64) -> f64 {
    vapor_density(p_sat_ice(t_c), t_c)
}

/// Water-saturation vapour density, g/m³.
pub fn rho_sat_water(t_c: f64) -> f64 {
    vapor_density(p_sat_water(t_c), t_c)
}

/// Excess vapour density of water-saturated air over ice saturation, g/m³.
/// This is the "water saturation" curve drawn on the morphology diagram:
/// the supersaturation inside a cloud of supercooled droplets.
#[cfg(test)]
pub fn water_saturation_excess(t_c: f64) -> f64 {
    rho_sat_water(t_c) - rho_sat_ice(t_c)
}

/// Convert relative humidity (% w.r.t. liquid water) to supersaturation over
/// ice in g/m³.  Negative values mean the air is sub-saturated w.r.t. ice.
pub fn supersaturation_from_rh(rh_percent: f64, t_c: f64) -> f64 {
    rh_percent / 100.0 * rho_sat_water(t_c) - rho_sat_ice(t_c)
}

/// Environmental conditions at one instant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditions {
    /// Air temperature, °C.
    pub temperature: f64,
    /// Excess vapour density over ice saturation, g/m³ (≥ 0).
    pub supersaturation: f64,
    /// Air pressure, hPa.  Vapour diffusion scales as 1/p, so the diffusion
    /// length X₀ ∝ 1/p: low pressure makes growth less diffusion-limited.
    pub pressure: f64,
}

/// Standard atmosphere, hPa.
pub const P_STANDARD: f64 = 1013.25;

impl Conditions {
    /// Dimensionless supersaturation σ = (ρ_v − ρ_sat,ice) / ρ_sat,ice.
    pub fn relative_supersaturation(&self) -> f64 {
        self.supersaturation / rho_sat_ice(self.temperature)
    }

    /// Relative humidity w.r.t. liquid water, %.
    pub fn humidity_water(&self) -> f64 {
        (self.supersaturation + rho_sat_ice(self.temperature)) / rho_sat_water(self.temperature) * 100.0
    }

    /// Where the conditions sit on the Nakaya morphology diagram.
    pub fn morphology(&self) -> Morphology {
        Morphology::classify(self.temperature, self.supersaturation)
    }

    pub fn lerp(a: &Conditions, b: &Conditions, t: f64) -> Conditions {
        Conditions {
            temperature: a.temperature + (b.temperature - a.temperature) * t,
            supersaturation: a.supersaturation + (b.supersaturation - a.supersaturation) * t,
            pressure: a.pressure + (b.pressure - a.pressure) * t,
        }
    }
}

/// Crystal habit expected for given conditions, following the morphology
/// diagram on snowcrystals.com (Libbrecht's rendering of the Nakaya diagram).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Morphology {
    SolidPrism,
    ThinPlate,
    SolidColumn,
    HollowColumn,
    Needle,
    SectoredPlate,
    StellarPlate,
    StellarDendrite,
    FernlikeDendrite,
    ColdPlate,
    ColdColumn,
}

impl Morphology {
    pub fn classify(t: f64, s: f64) -> Morphology {
        use Morphology::*;
        // Growth is barely diffusion-limited below ~0.04 g/m³: simple prisms
        // ("solid plates" along the bottom of the diagram).
        if s < 0.04 {
            return SolidPrism;
        }
        if t > -3.5 || (t <= -10.0 && t > -22.0) {
            // Plate regimes: the branching ladder, on the supersaturation
            // scale of the local branching threshold.
            let e = effective_supersaturation(t, s);
            if e < 0.11 {
                ThinPlate
            } else if e < 0.17 {
                SectoredPlate
            } else if e < 0.22 {
                StellarPlate
            } else if e < 0.32 {
                StellarDendrite
            } else {
                FernlikeDendrite
            }
        } else if t > -10.0 {
            // Column regime around −5 °C.
            if s < 0.10 {
                SolidColumn
            } else if s >= 0.17 && t < -4.0 && t > -8.0 {
                Needle
            } else {
                HollowColumn
            }
        } else if s < 0.10 {
            // Cold regime: plates below water saturation ...
            ColdPlate
        } else {
            // ... columns (and bullet rosettes) above it.
            ColdColumn
        }
    }
}

impl fmt::Display for Morphology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Morphology::*;
        f.write_str(match self {
            SolidPrism => "solid prism",
            ThinPlate => "thin plate",
            SolidColumn => "solid column",
            HollowColumn => "hollow column",
            Needle => "needle",
            SectoredPlate => "sectored plate",
            StellarPlate => "stellar plate",
            StellarDendrite => "stellar dendrite",
            FernlikeDendrite => "fernlike stellar dendrite",
            ColdPlate => "plate (cold regime)",
            ColdColumn => "column (cold regime)",
        })
    }
}

/// Supersaturation (g/m³) at which the morphology diagram shows branching
/// taking over from faceting: ≈0.15 g/m³ near −2 °C but ≈0.25 g/m³ near
/// −15 °C.  Used only to name habits.
pub fn branching_threshold(t_c: f64) -> f64 {
    const KNOTS: &[(f64, f64)] = &[
        (2.0, 0.14),
        (-2.0, 0.15),
        (-5.0, 0.20),
        (-10.0, 0.24),
        (-15.0, 0.25),
        (-60.0, 0.25),
    ];
    piecewise_linear(KNOTS, t_c)
}

/// Supersaturation rescaled so that the diagram's branching ladder (plate →
/// sectored plate → stellar plate → dendrite → fern) sits at the same values
/// as at −15 °C.  Used only to name habits.
pub fn effective_supersaturation(t_c: f64, s: f64) -> f64 {
    s * 0.25 / branching_threshold(t_c)
}

/// Linear interpolation through knots sorted by *decreasing* x.
fn piecewise_linear(knots: &[(f64, f64)], x: f64) -> f64 {
    if x >= knots[0].0 {
        return knots[0].1;
    }
    for w in knots.windows(2) {
        let (x0, y0) = w[0];
        let (x1, y1) = w[1];
        if x <= x0 && x >= x1 {
            let t = if x0 == x1 { 0.0 } else { (x0 - x) / (x0 - x1) };
            return y0 + (y1 - y0) * t;
        }
    }
    knots[knots.len() - 1].1
}

/// Parameters of the attachment-kinetics growth model, after Libbrecht's
/// comprehensive attachment kinetics (CAK) model and his cellular-automaton
/// scheme (*Snow Crystals*, 2021, chapters 3–5).
///
/// Growth follows the Hertz–Knudsen law `v = α · v_kin · σ_surf` (eq. 4.1).
/// On a facet the attachment coefficient — the probability that an arriving
/// molecule sticks — is nucleation-limited, `α = A · exp(−σ₀,eff / σ_surf)`
/// (eq. 4.4), and on rough or kinked surfaces `α ≈ 1`.  Structure-dependent
/// attachment kinetics lower the barrier on a narrow facet of width `w`:
/// `σ₀,eff = σ₀ · (1 − exp(−w / w₀))`, which is the edge-sharpening
/// instability; the measured dips sit near −14 °C on the prism facets and
/// −4 °C on the basal facets (figs. 4.26, 4.27) and decide plates versus
/// columns.  The surface supersaturation comes from the diffusion field with
/// the mixed boundary condition `X₀ ∂σ/∂n = α σ_surf` (eq. 3.9), where
/// `X₀ = D / √(kT/2πm) ≈ 0.145 µm` in air (eq. 3.10).  σ₀(T) and A(T) are a
/// reconstruction of the book's figures 4.5–4.6.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KineticsParams {
    /// Supersaturation over ice far from the crystal, as a fraction.
    pub sigma_inf: f64,
    /// Kinetic velocity v_kin, µm/s (eq. 4.2).
    pub v_kin: f64,
    /// Diffusion length X₀, µm (eq. 3.10).
    pub x0: f64,
    /// Gibbs–Thomson length d₀, µm (v = α v_kin (σ − d₀κ), eq. 5.37).
    pub d0: f64,
    /// Prism (in-plane) facets: prefactor, large-facet barrier, SDAK width (µm).
    pub a_prism: f64,
    pub sigma0_prism: f64,
    pub w0_prism: f64,
    /// Basal (out-of-plane) facets.
    pub a_basal: f64,
    pub sigma0_basal: f64,
    pub w0_basal: f64,
}

/// Attachment coefficient of a facet, CAK form.
pub fn attachment_coefficient(a: f64, sigma0: f64, sigma_surf: f64) -> f64 {
    if sigma_surf <= 0.0 {
        return 0.0;
    }
    a * (-sigma0 / sigma_surf).exp()
}

/// Effective nucleation barrier of a facet only `width` (µm) wide.
pub fn sdak_barrier(sigma0: f64, width: f64, w0: f64) -> f64 {
    if w0 <= 0.0 {
        return sigma0;
    }
    sigma0 * (1.0 - (-width / w0).exp())
}

/// Boltzmann constant, J/K.
const K_B: f64 = 1.380_649e-23;
/// Mass of a water molecule, kg.
const M_H2O: f64 = 2.991e-26;
/// Density of ice, kg/m³.
const RHO_ICE: f64 = 917.0;

/// Mean thermal speed factor √(kT/2πm), m/s.
fn thermal_speed(t_c: f64) -> f64 {
    (K_B * kelvin(t_c) / (2.0 * std::f64::consts::PI * M_H2O)).sqrt()
}

/// Kinetic velocity v_kin = (c_sat/c_ice) √(kT/2πm), µm/s: the growth rate
/// a surface would have at α = 1 and σ_surf = 1.
pub fn kinetic_velocity(t_c: f64) -> f64 {
    let c_ratio = rho_sat_ice(t_c) * 1e-3 / RHO_ICE;
    c_ratio * thermal_speed(t_c) * 1e6
}

/// Diffusion constant of water vapour in air, m²/s (≈2.1e-5 at 0 °C and
/// one atmosphere, ∝ T^1.75 / p).
pub fn vapour_diffusivity(t_c: f64, p_hpa: f64) -> f64 {
    2.1e-5 * (kelvin(t_c) / 273.15).powf(1.75) * (P_STANDARD / p_hpa)
}

/// Diffusion length X₀ = D / √(kT/2πm), µm — the scale that separates
/// kinetics-limited (α ≪ X₀/R) from diffusion-limited growth.
pub fn diffusion_length(t_c: f64, p_hpa: f64) -> f64 {
    vapour_diffusivity(t_c, p_hpa) / thermal_speed(t_c) * 1e6
}

pub fn kinetics_params(c: &Conditions) -> KineticsParams {
    let t = c.temperature;
    // Large-facet nucleation barriers (fractions), a few percent.  Towards
    // 0 °C the basal step energy tends to the (larger) ice/water value while
    // the prism one falls well below it; at low temperature both approach
    // the rigid-lattice value and rise as kT drops.
    let sigma0_basal = piecewise_linear(
        &[
            (2.0, 0.036),
            (-5.0, 0.030),
            (-10.0, 0.025),
            (-15.0, 0.022),
            (-20.0, 0.024),
            (-30.0, 0.034),
            (-60.0, 0.05),
        ],
        t,
    );
    let sigma0_prism = piecewise_linear(
        &[
            (2.0, 0.008),
            (-2.0, 0.012),
            (-5.0, 0.018),
            (-10.0, 0.024),
            (-15.0, 0.024),
            (-20.0, 0.026),
            (-30.0, 0.034),
            (-60.0, 0.05),
        ],
        t,
    );
    // Prefactors in air at one atmosphere.  In near-vacuum both facets have
    // A ≈ 1 (Libbrecht & Rickerby 2013), but background air lowers the
    // attachment coefficient of a broad facet by up to two orders of
    // magnitude (Libbrecht 2016: the −5 °C prism coefficient falls ~100×
    // from 0.01 to 1 bar; §4.6), which is what keeps needles and columns
    // slender.  The basal one must be very small
    // around −15 °C (and −2 °C): a plate's basal faces are fed by diffusion
    // at ~0.1 µm/s for a 30 µm crystal at σ_∞ = 0.15, and with the vacuum
    // value they would take all of it (α_basal(1 %) ≈ 0.1) and make a
    // thick plate within a minute, whereas in air a frozen droplet grows
    // into a plate a few µm thin; the basal faces must therefore be
    // kinetics-limited at α ≲ 10⁻³.  Near −5 °C basal growth is fast
    // (needles, hollow columns) and below −22 °C columns need it fast
    // again.  The reduction scales with pressure, so it vanishes as the
    // air is pumped away.
    let a_prism_air = piecewise_linear(
        &[
            (2.0, 0.05),
            (0.0, 0.05),
            (-2.0, 0.1),
            (-4.0, 0.01),
            (-6.0, 0.01),
            (-8.0, 0.05),
            (-10.0, 0.5),
            (-12.0, 1.0),
            (-60.0, 1.0),
        ],
        t,
    );
    let a_basal_air = piecewise_linear(
        &[
            (2.0, 0.0005),
            (-2.0, 0.0005),
            (-3.5, 1.0),
            (-8.0, 1.0),
            (-10.0, 0.03),
            (-12.0, 0.002),
            (-18.0, 0.002),
            (-21.0, 0.03),
            (-24.0, 1.0),
            (-60.0, 1.0),
        ],
        t,
    );
    let in_air = |a_air: f64| 1.0 / (1.0 + (1.0 / a_air - 1.0) * (c.pressure / P_STANDARD).max(0.0));
    let a_prism = in_air(a_prism_air);
    let a_basal = in_air(a_basal_air);
    // SDAK widths in µm: large where the edge-sharpening instability is
    // strong.  The measured prism dip is centred near −14 °C (thin plates,
    // with a second one near −2 °C), the basal dip near −4 °C (needles,
    // thin-walled hollow columns).  The ESI settles edges at 1–2 µm, so a
    // width of a few µm halves the barrier on such an edge.
    let w0_prism = piecewise_linear(
        &[
            (2.0, 2.0),
            (-2.0, 5.0),
            (-4.0, 1.5),
            (-7.0, 0.3),
            (-10.0, 1.5),
            (-12.0, 4.0),
            (-14.0, 8.0),
            (-16.0, 5.0),
            (-18.0, 3.0),
            (-21.0, 1.0),
            (-30.0, 0.3),
            (-60.0, 0.3),
        ],
        t,
    );
    let w0_basal = piecewise_linear(
        &[
            (2.0, 0.3),
            (-2.0, 0.4),
            (-3.0, 2.0),
            (-4.0, 8.0),
            (-6.0, 5.0),
            (-8.0, 1.5),
            (-12.0, 0.3),
            (-22.0, 0.5),
            (-28.0, 3.0),
            (-35.0, 4.0),
            (-60.0, 4.0),
        ],
        t,
    );
    KineticsParams {
        sigma_inf: c.relative_supersaturation().max(0.0),
        v_kin: kinetic_velocity(t),
        x0: diffusion_length(t, c.pressure),
        d0: 0.001,
        a_prism,
        sigma0_prism,
        w0_prism,
        a_basal,
        sigma0_basal,
        w0_basal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cak_attachment_coefficient_behaves() {
        // Nucleation-limited: tiny at low supersaturation, → A at high.
        assert!(attachment_coefficient(1.0, 0.02, 0.005) < 0.02);
        assert!(attachment_coefficient(1.0, 0.02, 0.5) > 0.95);
        assert_eq!(attachment_coefficient(1.0, 0.02, 0.0), 0.0);
        // A narrow facet has a lower barrier; a wide one the full barrier.
        assert!(sdak_barrier(0.02, 0.5, 4.0) < 0.004);
        assert!((sdak_barrier(0.02, 100.0, 4.0) - 0.02).abs() < 1e-9);
        assert_eq!(sdak_barrier(0.02, 1.0, 0.0), 0.02);
    }

    #[test]
    fn kinetic_velocity_and_diffusion_length_match_the_book() {
        // Libbrecht: X₀ ≈ 0.145 µm in air at −15 °C; v_kin of order 100 µm/s.
        let x0 = diffusion_length(-15.0, P_STANDARD);
        assert!((x0 - 0.145).abs() < 0.02, "{x0}");
        // Half the pressure, twice the diffusion length.
        assert!((diffusion_length(-15.0, P_STANDARD / 2.0) / x0 - 2.0).abs() < 1e-9);
        let v = kinetic_velocity(-15.0);
        assert!(v > 150.0 && v < 300.0, "{v}");
        assert!(kinetic_velocity(-5.0) > 2.0 * v);
    }

    #[test]
    fn kinetics_dips_favour_plates_at_minus_15_and_columns_at_minus_5() {
        let k15 = kinetics_params(&Conditions {
            temperature: -15.0,
            supersaturation: 0.2,
            pressure: P_STANDARD,
        });
        let k5 = kinetics_params(&Conditions {
            temperature: -5.0,
            supersaturation: 0.2,
            pressure: P_STANDARD,
        });
        assert!(k15.w0_prism > k15.w0_basal * 5.0);
        assert!(k5.w0_basal > k5.w0_prism * 5.0);
        assert!(k15.sigma_inf > 0.1 && k15.sigma_inf < 0.2);
        // −5 °C, large facets: prism grows more readily than basal (thick plates at low σ).
        assert!(k5.sigma0_prism < k5.sigma0_basal);
    }

    #[test]
    fn saturation_pressures_match_reference_values() {
        // Murphy & Koop table values (Pa).
        assert!((p_sat_ice(0.0) - 611.15).abs() < 1.0);
        assert!((p_sat_water(0.0) - 611.21).abs() < 1.0);
        assert!((p_sat_ice(-15.0) - 165.3).abs() < 1.0);
        assert!((p_sat_water(-15.0) - 191.4).abs() < 1.0);
        assert!((p_sat_ice(-40.0) - 12.84).abs() < 0.2);
    }

    #[test]
    fn water_saturation_curve_peaks_near_minus_twelve() {
        let peak = (-30..0)
            .map(|t| (t as f64, water_saturation_excess(t as f64)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert!(peak.0 <= -10.0 && peak.0 >= -14.0, "peak at {:?}", peak);
        assert!((peak.1 - 0.21).abs() < 0.03);
    }

    #[test]
    fn rh_roundtrip() {
        let c = Conditions {
            temperature: -15.0,
            supersaturation: supersaturation_from_rh(100.0, -15.0),
            pressure: P_STANDARD,
        };
        assert!((c.humidity_water() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn branching_starts_earlier_near_minus_two() {
        assert!(branching_threshold(-2.0) < branching_threshold(-15.0));
        assert!((effective_supersaturation(-15.0, 0.2) - 0.2).abs() < 1e-12);
        assert!(effective_supersaturation(-2.0, 0.15) > 0.24);
    }

    #[test]
    fn morphology_follows_the_diagram() {
        use Morphology::*;
        assert_eq!(Morphology::classify(-15.0, 0.03), SolidPrism);
        assert_eq!(Morphology::classify(-15.0, 0.10), ThinPlate);
        assert_eq!(Morphology::classify(-15.0, 0.14), SectoredPlate);
        assert_eq!(Morphology::classify(-15.0, 0.19), StellarPlate);
        assert_eq!(Morphology::classify(-15.0, 0.28), StellarDendrite);
        assert_eq!(Morphology::classify(-15.0, 0.40), FernlikeDendrite);
        assert_eq!(Morphology::classify(-2.0, 0.05), ThinPlate);
        assert_eq!(Morphology::classify(-2.0, 0.17), StellarDendrite);
        assert_eq!(Morphology::classify(-5.0, 0.20), Needle);
        assert_eq!(Morphology::classify(-6.0, 0.12), HollowColumn);
        assert_eq!(Morphology::classify(-27.0, 0.06), ColdPlate);
        assert_eq!(Morphology::classify(-30.0, 0.15), ColdColumn);
    }
}
