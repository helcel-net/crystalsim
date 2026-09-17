//! Elevated ("isometric") view of the crystal as a solid.
//!
//! Every crystal cell is extruded into a hexagonal prism of its c-axis height,
//! centred on the basal plane, so plates come out thin and columns tall.  The
//! prisms are drawn back to front with an orthographic camera raised
//! `elevation` degrees above the plane; top faces are shaded by the local
//! slope of the height map and the visible side faces by their orientation.

use super::raster::Raster;
use super::{MassScale, SQRT3, cell_centre, enclosing_radius};
use crate::automaton::Snapshot;

#[derive(Clone, Copy, Debug)]
pub struct IsoView {
    /// Camera elevation above the basal plane, degrees (90 = straight down).
    pub elevation_deg: f64,
    /// In-plane rotation of the crystal, degrees.
    pub azimuth_deg: f64,
    /// Multiplier on the physical heights (1 = as grown).
    pub height_scale: f64,
}

impl Default for IsoView {
    fn default() -> Self {
        IsoView {
            elevation_deg: 35.0,
            azimuth_deg: 20.0,
            height_scale: 1.0,
        }
    }
}

/// Extent of a crystal, in hex units, that a frame has to hold.
#[derive(Clone, Copy, Debug)]
pub struct IsoFit {
    /// Radius of the footprint.
    pub radius: f64,
    /// Tallest prism.
    pub height: f64,
}

impl IsoFit {
    pub fn of(snap: &Snapshot, view: IsoView) -> IsoFit {
        IsoFit {
            radius: enclosing_radius(snap.crystal_radius.max(3)),
            height: cell_height(snap.max_height(), 1.0, view),
        }
    }

    /// Height of the projected extent relative to its width.
    fn aspect(&self, view: IsoView) -> f64 {
        let (sin_e, cos_e) = view.elevation_deg.to_radians().sin_cos();
        (self.radius * sin_e + self.height / 2.0 * cos_e) / self.radius
    }

    /// A natural frame `width` pixels wide: as tall as the projection needs,
    /// but never more than 1.3× the width (a very long needle is scaled down
    /// instead).
    pub fn frame(&self, width: u32, view: IsoView) -> (u32, u32) {
        let h = (width as f64 * self.aspect(view)).ceil().min(width as f64 * 1.3);
        (width, (h as u32).max(8))
    }
}

/// Physical height of a cell in hex units: `cells` of c-axis growth (one cell
/// is √3 hex units tall), plus a couple of cells where the ice mass says the
/// surface is thick (ridges).
fn cell_height(cells: f32, thickness: f32, view: IsoView) -> f64 {
    (cells as f64 + 3.0 * thickness as f64) * SQRT3 * view.height_scale
}

/// Light direction in screen-aligned world coordinates: x right, y towards
/// the viewer, z up.  From the upper left, slightly in front.
const LIGHT: [f64; 3] = [-0.45, 0.35, 0.82];

struct Canvas {
    w: usize,
    h: usize,
    coverage: Vec<f32>,
    thickness: Vec<f32>,
    shade: Vec<f32>,
}

impl Canvas {
    /// Fill a convex polygon (screen coordinates); a pixel is covered when
    /// its centre lies inside, half-open on the right and bottom so that
    /// polygons sharing an edge tile without gaps or double coverage.
    fn fill(&mut self, pts: &[(f64, f64)], t: f32, shade: f32) {
        let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
        for p in pts {
            ymin = ymin.min(p.1);
            ymax = ymax.max(p.1);
        }
        let y0 = ((ymin - 0.5).ceil().max(0.0)) as usize;
        let y1 = ((ymax - 0.5).ceil().clamp(0.0, self.h as f64)) as usize;
        for py in y0..y1 {
            let yc = py as f64 + 0.5;
            let (mut xl, mut xr) = (f64::MAX, f64::MIN);
            for i in 0..pts.len() {
                let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
                if a.1 == b.1 {
                    continue;
                }
                let (lo, hi) = if a.1 < b.1 { (a, b) } else { (b, a) };
                if yc >= lo.1 && yc < hi.1 {
                    let x = lo.0 + (hi.0 - lo.0) * (yc - lo.1) / (hi.1 - lo.1);
                    xl = xl.min(x);
                    xr = xr.max(x);
                }
            }
            if xl > xr {
                continue;
            }
            let x0 = ((xl - 0.5).ceil().max(0.0)) as usize;
            let x1 = ((xr - 0.5).ceil().clamp(0.0, self.w as f64)) as usize;
            let row = py * self.w;
            for px in x0..x1 {
                self.coverage[row + px] = 1.0;
                self.thickness[row + px] = t;
                self.shade[row + px] = shade;
            }
        }
    }
}

/// Render the crystal as a solid into a `width`×`height` frame, scaled so
/// that `fit` (normally the final crystal, so animation frames share one
/// scale) fills it.
pub fn render(
    snap: &Snapshot,
    mass: MassScale,
    (width, height): (u32, u32),
    fit: IsoFit,
    view: IsoView,
    supersample: u32,
) -> Raster {
    let ss = supersample.max(1) as usize;
    let (sin_e, cos_e) = view.elevation_deg.to_radians().sin_cos();
    let (sin_a, cos_a) = view.azimuth_deg.to_radians().sin_cos();
    let rot = |x: f64, y: f64| (x * cos_a - y * sin_a, x * sin_a + y * cos_a);
    let out_w = width as usize;
    let out_h = height as usize;
    let w = out_w * ss;
    let h = out_h * ss;
    let margin = 1.06;
    let half_extent_y = fit.radius * sin_e + fit.height / 2.0 * cos_e;
    let scale = (w as f64 / 2.0 / (fit.radius * margin)).min(h as f64 / 2.0 / (half_extent_y * margin));
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    // Orthographic projection: y (towards the viewer) drops down the screen
    // foreshortened, z rises.
    let project = |x: f64, y: f64, z: f64| (cx + x * scale, cy + (y * sin_e - z * cos_e) * scale);
    let light = {
        let n = (LIGHT[0].powi(2) + LIGHT[1].powi(2) + LIGHT[2].powi(2)).sqrt();
        [LIGHT[0] / n, LIGHT[1] / n, LIGHT[2] / n]
    };

    // Hexagon geometry (flat-topped, circumradius 1), rotated with the view.
    let verts: Vec<(f64, f64)> = (0..6)
        .map(|k| {
            let a = (60.0 * k as f64).to_radians();
            rot(a.cos(), a.sin())
        })
        .collect();
    // Side k runs from vertex k to k+1; its outward normal is at 30° + 60k.
    let sides: Vec<(bool, f32)> = (0..6)
        .map(|k| {
            let a = (30.0 + 60.0 * k as f64).to_radians();
            let (nx, ny) = rot(a.cos(), a.sin());
            let facing_viewer = ny > 1e-9;
            let lit = (nx * light[0] + ny * light[1]).max(0.0);
            (facing_viewer, (0.3 + 0.5 * lit) as f32)
        })
        .collect();
    // Neighbour directions for the slope estimate.
    let dirs: Vec<((i32, i32), (f64, f64))> = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        .iter()
        .map(|&(dq, dr)| {
            let (x, y) = cell_centre(dq, dr);
            ((dq, dr), (x / SQRT3, y / SQRT3))
        })
        .collect();

    let height_of = |q: i32, r: i32| {
        snap.mass_at(q, r)
            .map(|m| cell_height(snap.height_at(q, r), mass.thickness(m), view))
    };

    // Painter's order: farthest (smallest rotated y) first.
    let mut cells: Vec<(f64, f64, i32, i32, f32)> = snap
        .cells()
        .map(|(q, r, m)| {
            let (x, y) = cell_centre(q, r);
            let (xr, yr) = rot(x, y);
            (xr, yr, q, r, m)
        })
        .collect();
    cells.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut canvas = Canvas {
        w,
        h,
        coverage: vec![0.0; w * h],
        thickness: vec![0.0; w * h],
        shade: vec![0.0; w * h],
    };

    for &(xr, yr, q, r, m) in &cells {
        let hc = height_of(q, r).unwrap_or(0.0);
        let t = mass.thickness(m);

        // Slope of the height map for the top-face shading.
        let (mut gx, mut gy) = (0.0, 0.0);
        for &((dq, dr), (dx, dy)) in &dirs {
            if let Some(hn) = height_of(q + dq, r + dr) {
                let dh = hn - hc;
                gx += dh * dx;
                gy += dh * dy;
            }
        }
        let (gx, gy) = rot(gx / (3.0 * SQRT3), gy / (3.0 * SQRT3));
        let n = (-gx, -gy, 1.0);
        let nn = (n.0 * n.0 + n.1 * n.1 + 1.0).sqrt();
        let lit = (n.0 * light[0] + n.1 * light[1] + n.2 * light[2]) / nn;
        let top_shade = (0.55 + 0.45 * lit.max(0.0)) as f32;

        let top: Vec<(f64, f64)> = verts
            .iter()
            .map(|&(vx, vy)| project(xr + vx, yr + vy, hc / 2.0))
            .collect();
        let bottom: Vec<(f64, f64)> = verts
            .iter()
            .map(|&(vx, vy)| project(xr + vx, yr + vy, -hc / 2.0))
            .collect();
        for k in 0..6 {
            let (visible, side_shade) = sides[k];
            if !visible {
                continue;
            }
            let k1 = (k + 1) % 6;
            canvas.fill(&[top[k], top[k1], bottom[k1], bottom[k]], t * 0.85, side_shade);
        }
        canvas.fill(&top, t, top_shade);
    }

    // Downsample.
    let mut coverage = vec![0.0f32; out_w * out_h];
    let mut thickness = vec![0.0f32; out_w * out_h];
    let mut shade = vec![0.0f32; out_w * out_h];
    for py in 0..out_h {
        for px in 0..out_w {
            let (mut c, mut t, mut s) = (0.0, 0.0, 0.0);
            for sy in 0..ss {
                for sx in 0..ss {
                    let i = (py * ss + sy) * w + px * ss + sx;
                    c += canvas.coverage[i];
                    t += canvas.thickness[i] * canvas.coverage[i];
                    s += canvas.shade[i] * canvas.coverage[i];
                }
            }
            let o = py * out_w + px;
            if c > 0.0 {
                thickness[o] = t / c;
                shade[o] = s / c;
                coverage[o] = c / (ss * ss) as f32;
            }
        }
    }
    Raster {
        width,
        height,
        coverage,
        thickness,
        shade: Some(shade),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hexagon_snapshot(height: f32, mass_value: f32) -> Snapshot {
        let radius = 6;
        let width = 2 * radius as usize + 1;
        let mut mass = vec![-1.0; width * width];
        let mut heights = vec![0.0; width * width];
        for (q, r) in [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)] {
            let i = (r + radius) as usize * width + (q + radius) as usize;
            mass[i] = mass_value;
            heights[i] = height;
        }
        Snapshot {
            radius,
            width,
            mass,
            height: heights,
            crystal_radius: 1,
            crystal_cells: 7,
            step: 0,
            molecules: 0.0,
        }
    }

    #[test]
    fn fill_tiles_shared_edges_exactly_once() {
        let mut c = Canvas {
            w: 8,
            h: 8,
            coverage: vec![0.0; 64],
            thickness: vec![0.0; 64],
            shade: vec![0.0; 64],
        };
        // Two squares sharing the edge x = 4.
        c.fill(&[(0.0, 0.0), (4.0, 0.0), (4.0, 8.0), (0.0, 8.0)], 0.2, 1.0);
        c.fill(&[(4.0, 0.0), (8.0, 0.0), (8.0, 8.0), (4.0, 8.0)], 0.8, 1.0);
        let left = c.thickness.iter().filter(|&&t| t == 0.2).count();
        let right = c.thickness.iter().filter(|&&t| t == 0.8).count();
        assert_eq!((left, right), (32, 32));
    }

    #[test]
    fn taller_crystals_get_taller_frames_and_more_side_area() {
        let view = IsoView::default();
        let mass = MassScale { reference: 1.0 };
        // A thin plate (little c-axis growth, little ridge mass) and a column.
        let plate = hexagon_snapshot(0.2, 0.01);
        let column = hexagon_snapshot(60.0, 1.0);
        let (pf, cf) = (IsoFit::of(&plate, view), IsoFit::of(&column, view));
        assert!(cf.frame(64, view).1 > pf.frame(64, view).1);
        // Flat tops shade ≈0.92, side faces ≤0.6: a column shows mostly
        // sides, a plate mostly top.
        let side_fraction = |snap: &Snapshot, fit: IsoFit| {
            let r = render(snap, mass, (64, 64), fit, view, 2);
            let shade = r.shade.unwrap();
            let sides = shade.iter().filter(|&&s| s > 0.0 && s < 0.75).count() as f64;
            let tops = shade.iter().filter(|&&s| s >= 0.75).count() as f64;
            sides / (sides + tops)
        };
        assert!(side_fraction(&column, cf) > 0.6);
        assert!(side_fraction(&plate, pf) < 0.4);
    }
}
