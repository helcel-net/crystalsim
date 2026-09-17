//! Rendering of crystal snapshots.
//!
//! Cells are drawn as flat-topped hexagons so that the six lattice directions
//! (along which the primary arms grow) point at 30°, 90°, 150°, … — one arm
//! straight up, as in most photographs.

pub mod iso;
pub mod raster;
pub mod svg;

use crate::automaton::Snapshot;

pub const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Centre of cell (q, r) in "hex units" (circumradius 1), y pointing down.
#[inline]
pub fn cell_centre(q: i32, r: i32) -> (f64, f64) {
    (1.5 * q as f64, SQRT3 * (r as f64 + q as f64 / 2.0))
}

/// Cell containing the point (x, y) in hex units.
#[inline]
pub fn cell_at(x: f64, y: f64) -> (i32, i32) {
    let qf = x / 1.5;
    let rf = y / SQRT3 - qf / 2.0;
    cube_round(qf, rf)
}

fn cube_round(qf: f64, rf: f64) -> (i32, i32) {
    let sf = -qf - rf;
    let (mut q, mut r, s) = (qf.round(), rf.round(), sf.round());
    let (dq, dr, ds) = ((q - qf).abs(), (r - rf).abs(), (s - sf).abs());
    if dq > dr && dq > ds {
        q = -r - s;
    } else if dr > ds {
        r = -q - s;
    }
    (q as i32, r as i32)
}

/// Euclidean radius (hex units) that encloses every cell within hex distance
/// `n` of the centre, with a little margin.
pub fn enclosing_radius(n: i32) -> f64 {
    SQRT3 * (n as f64 + 1.5)
}

/// Colour scheme: dark sky, ice going from deep blue (thin) to white (thick).
#[derive(Clone, Copy, Debug)]
pub struct Palette;

impl Palette {
    pub const BACKGROUND: [f32; 3] = [6.0, 12.0, 34.0];

    /// Ice colour for normalised thickness `t` in [0, 1].
    pub fn ice(t: f32) -> [f32; 3] {
        const STOPS: [([f32; 3], f32); 4] = [
            ([58.0, 108.0, 196.0], 0.0),
            ([120.0, 170.0, 232.0], 0.35),
            ([190.0, 218.0, 250.0], 0.7),
            ([255.0, 255.0, 255.0], 1.0),
        ];
        let t = t.clamp(0.0, 1.0);
        for w in STOPS.windows(2) {
            let (c0, t0) = w[0];
            let (c1, t1) = w[1];
            if t <= t1 {
                let f = (t - t0) / (t1 - t0);
                return [
                    c0[0] + (c1[0] - c0[0]) * f,
                    c0[1] + (c1[1] - c0[1]) * f,
                    c0[2] + (c1[2] - c0[2]) * f,
                ];
            }
        }
        STOPS[3].0
    }

    pub fn blend(t: f32, coverage: f32) -> [u8; 3] {
        Self::blend_shaded(t, 1.0, coverage)
    }

    /// Ice of thickness `t` lit by `shade` (0..1), mixed over the background.
    pub fn blend_shaded(t: f32, shade: f32, coverage: f32) -> [u8; 3] {
        let ice = Self::ice(t);
        let bg = Self::BACKGROUND;
        let mut out = [0u8; 3];
        for k in 0..3 {
            out[k] = (bg[k] + (ice[k] * shade - bg[k]) * coverage)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        out
    }

    pub fn hex(rgb: [u8; 3]) -> String {
        format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
    }
}

/// Maps crystal mass to a normalised thickness in [0, 1].
///
/// Thickness is normalised against a high percentile of the *final* crystal so
/// that the same map can be used for every animation frame.
#[derive(Clone, Copy, Debug)]
pub struct MassScale {
    pub reference: f32,
}

impl MassScale {
    pub fn from_snapshot(snap: &Snapshot) -> MassScale {
        let mut masses: Vec<f32> = snap.cells().map(|(_, _, m)| m).collect();
        if masses.is_empty() {
            return MassScale { reference: 1.0 };
        }
        masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((masses.len() - 1) as f64 * 0.97) as usize;
        MassScale {
            reference: masses[idx].max(1e-6),
        }
    }

    /// Normalised thickness with a gamma lift so that thin, fast-grown
    /// dendrite tips still read as ice rather than vanishing into the sky.
    #[inline]
    pub fn thickness(&self, mass: f32) -> f32 {
        (mass / self.reference).clamp(0.0, 1.0).powf(0.6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_lookup_inverts_centre() {
        for q in -30..30 {
            for r in -30..30 {
                let (x, y) = cell_centre(q, r);
                assert_eq!(cell_at(x, y), (q, r));
                // Slightly off-centre still lands in the same cell.
                assert_eq!(cell_at(x + 0.4, y - 0.3), (q, r));
            }
        }
    }

    #[test]
    fn neighbours_are_at_unit_lattice_distance() {
        let (x0, y0) = cell_centre(0, 0);
        for (q, r) in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)] {
            let (x, y) = cell_centre(q, r);
            let dist = ((x - x0).powi(2) + (y - y0).powi(2)).sqrt();
            assert!((dist - SQRT3).abs() < 1e-9);
        }
    }
}
