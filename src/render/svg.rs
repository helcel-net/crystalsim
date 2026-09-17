//! Vector output.
//!
//! Rather than one polygon per cell, the crystal is drawn as a handful of
//! nested thickness bands, each traced as the outline of the set of cells
//! above a mass threshold.  Outlines are emitted in integer lattice units
//! (x doubled, y stretched by 2/√3) so every edge is one of six short
//! relative moves; a group transform restores the true hexagonal geometry.

use super::{MassScale, Palette, SQRT3, enclosing_radius};
use crate::automaton::Snapshot;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

/// Vertex of the honeycomb: every vertex is the right (`side = 0`) or left
/// (`side = 1`) vertex of exactly one flat-topped cell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct Vertex {
    q: i32,
    r: i32,
    side: u8,
}

impl Vertex {
    /// Integer coordinates: cell centre is (3q, 4r + 2q), vertices are ±2 in x.
    fn coords(self) -> (i32, i32) {
        let x = 3 * self.q + if self.side == 0 { 2 } else { -2 };
        (x, 4 * self.r + 2 * self.q)
    }
}

/// The six vertices of cell (q, r), clockwise on screen, and the neighbour
/// across the edge that starts at each vertex.
fn cell_vertices(q: i32, r: i32) -> [Vertex; 6] {
    [
        Vertex { q, r, side: 0 },
        Vertex { q: q + 1, r, side: 1 },
        Vertex {
            q: q - 1,
            r: r + 1,
            side: 0,
        },
        Vertex { q, r, side: 1 },
        Vertex { q: q - 1, r, side: 0 },
        Vertex {
            q: q + 1,
            r: r - 1,
            side: 1,
        },
    ]
}

const EDGE_NEIGHBOURS: [(i32, i32); 6] = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)];

/// Closed outlines (as vertex lists) of the set of crystal cells with mass
/// at least `threshold`.
fn trace_region(snap: &Snapshot, threshold: f32) -> Vec<Vec<Vertex>> {
    let inside = |q: i32, r: i32| snap.mass_at(q, r).is_some_and(|m| m >= threshold);
    let mut next: HashMap<Vertex, Vertex> = HashMap::new();
    for (q, r, m) in snap.cells() {
        if m < threshold {
            continue;
        }
        let verts = cell_vertices(q, r);
        for k in 0..6 {
            let (dq, dr) = EDGE_NEIGHBOURS[k];
            if !inside(q + dq, r + dr) {
                next.insert(verts[k], verts[(k + 1) % 6]);
            }
        }
    }

    let mut loops = Vec::new();
    while let Some(&start) = next.keys().next() {
        let mut ring = vec![start];
        let mut cur = start;
        while let Some(nxt) = next.remove(&cur) {
            if nxt == start {
                break;
            }
            ring.push(nxt);
            cur = nxt;
        }
        loops.push(ring);
    }
    loops
}

fn path_data(loops: &[Vec<Vertex>]) -> String {
    let mut d = String::new();
    for ring in loops {
        let (x0, y0) = ring[0].coords();
        let _ = write!(d, "M{x0} {y0}");
        let mut prev = (x0, y0);
        for v in &ring[1..] {
            let (x, y) = v.coords();
            let _ = write!(d, "l{} {}", x - prev.0, y - prev.1);
            prev = (x, y);
        }
        d.push('z');
    }
    d
}

/// Thickness bands as (mass threshold, colour) from dark to bright.
fn bands(snap: &Snapshot, scale: MassScale) -> Vec<(f32, [u8; 3])> {
    let mut masses: Vec<f32> = snap.cells().map(|(_, _, m)| m).collect();
    masses.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quantile = |p: f64| masses[((masses.len() - 1) as f64 * p) as usize];
    let mut out = Vec::new();
    let mut last = f32::NEG_INFINITY;
    for p in [0.0, 0.15, 0.3, 0.45, 0.6, 0.73, 0.84, 0.92, 0.97] {
        let th = quantile(p);
        if th <= last {
            continue; // collapsed quantiles (e.g. tiny crystals)
        }
        last = th;
        out.push((th, Palette::blend(scale.thickness(th), 1.0)));
    }
    out
}

pub fn write_svg(path: &Path, snap: &Snapshot, size: u32, scale: MassScale, description: &str) -> Result<()> {
    let extent = enclosing_radius(snap.crystal_radius.max(3));
    let px_per_unit = size as f64 / 2.0 / extent;
    // Integer lattice units: x is doubled, y is stretched by 2/√3.
    let sx = px_per_unit / 2.0;
    let sy = sx * SQRT3 / 2.0;
    let half = size as f64 / 2.0;

    let mut svg = String::new();
    let _ = writeln!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">"#
    );
    let _ = writeln!(svg, "<title>flakesim snow crystal</title>");
    let _ = writeln!(svg, "<desc>{}</desc>", escape(description));
    let _ = writeln!(
        svg,
        r#"<rect width="100%" height="100%" fill="{}"/>"#,
        Palette::hex(Palette::blend(0.0, 0.0))
    );
    let _ = writeln!(
        svg,
        r#"<g transform="translate({half:.3} {half:.3}) scale({sx:.6} {sy:.6})">"#
    );
    for (threshold, colour) in bands(snap, scale) {
        let loops = trace_region(snap, threshold);
        if loops.is_empty() {
            continue;
        }
        let _ = writeln!(
            svg,
            r#"<path fill="{}" fill-rule="evenodd" d="{}"/>"#,
            Palette::hex(colour),
            path_data(&loops)
        );
    }
    svg.push_str("</g>\n</svg>\n");
    std::fs::write(path, svg).with_context(|| format!("cannot write {}", path.display()))
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot_with(cells: &[(i32, i32)]) -> Snapshot {
        let radius = 5;
        let width = 2 * radius as usize + 1;
        let mut mass = vec![-1.0; width * width];
        for &(q, r) in cells {
            mass[(r + radius) as usize * width + (q + radius) as usize] = 1.0;
        }
        Snapshot {
            radius,
            width,
            mass,
            height: vec![0.0; width * width],
            molecules: 0.0,
            crystal_radius: 2,
            crystal_cells: cells.len(),
            step: 0,
        }
    }

    #[test]
    fn single_cell_traces_one_hexagon() {
        let snap = snapshot_with(&[(0, 0)]);
        let loops = trace_region(&snap, 0.0);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].len(), 6);
        let d = path_data(&loops);
        assert!(d.starts_with('M') && d.ends_with('z'));
    }

    #[test]
    fn ring_of_cells_has_outer_and_inner_outline() {
        // The six neighbours of the origin, without the origin: an annulus.
        let ring: Vec<(i32, i32)> = EDGE_NEIGHBOURS.to_vec();
        let snap = snapshot_with(&ring);
        let loops = trace_region(&snap, 0.0);
        assert_eq!(loops.len(), 2);
        let mut lens: Vec<usize> = loops.iter().map(Vec::len).collect();
        lens.sort();
        assert_eq!(lens, vec![6, 18]);
    }

    #[test]
    fn shared_vertices_agree_between_cells() {
        // The vertex at 60° of (0,0) is the same as the vertex at 180° of (1,0).
        let a = cell_vertices(0, 0);
        let b = cell_vertices(1, 0);
        assert_eq!(a[1], b[3]);
        assert_eq!(a[1].coords(), b[3].coords());
        // Every edge vector is one of the six unit moves.
        for k in 0..6 {
            let (x0, y0) = a[k].coords();
            let (x1, y1) = a[(k + 1) % 6].coords();
            let dx = x1 - x0;
            let dy = y1 - y0;
            assert!(
                matches!((dx, dy), (-1, 2) | (-2, 0) | (-1, -2) | (1, -2) | (2, 0) | (1, 2)),
                "{:?}",
                (dx, dy)
            );
        }
    }
}
