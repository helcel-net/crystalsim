//! Reproduce the morphology diagram: grow one crystal per (temperature,
//! supersaturation) cell under constant conditions and tile the results into
//! a chart laid out like the diagram on snowcrystals.com — warm on the left,
//! humid at the top.  Each cell shows the crystal from above and, below it,
//! in the elevated relief view.

use crate::automaton::{Automaton, Nucleation, Snapshot};
use crate::par::*;
use crate::physics::{self, Conditions, kinetics_params};
use crate::render::iso::{self, IsoFit, IsoView};
use crate::render::raster::{self, View};
use crate::render::{MassScale, Palette, enclosing_radius};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

pub struct DiagramSpec {
    pub temperatures: Vec<f64>,
    pub supersaturations: Vec<f64>,
    pub duration: f64,
    /// Stop each crystal at this many molecules, if set.
    pub molecules: Option<f64>,
    pub radius: usize,
    pub nucleation: f64,
    /// Rate quantum of the quantised growth model (0 = Poisson model).
    pub quantum: f64,
    pub smoothing: usize,
    pub tile: u32,
    /// Physical cell size, µm.
    pub cell_size_um: f64,
    /// Relief view drawn under each top view; `None` for top views only.
    pub iso: Option<IsoView>,
}

struct Cell {
    conditions: Conditions,
    snapshot: Snapshot,
}

fn grow(spec: &DiagramSpec, conditions: Conditions) -> Snapshot {
    let kp = kinetics_params(&conditions);
    let kinetics = kp.into();
    let mut auto = Automaton::new(spec.radius, kp.sigma_inf, 1).with_cell_size(spec.cell_size_um as f32);
    auto.seed_hexagon();
    auto.converge_field(&kinetics);
    let stop_radius = spec.radius as i32 - (spec.radius as i32 / 16).max(6);
    let nucleation = Nucleation {
        rate: spec.nucleation as f32,
        symmetric: true,
        quantum: spec.quantum as f32,
    };
    let mut t = 0.0f64;
    while t < spec.duration && auto.step_count() < 2_000_000 {
        let remaining = (spec.duration - t) as f32;
        t += auto.step(&kinetics, nucleation, remaining.min(2.0)) as f64;
        if auto.crystal_radius() >= stop_radius {
            break;
        }
        if spec.molecules.is_some_and(|m| auto.molecules() >= m) {
            break;
        }
    }
    auto.snapshot().smoothed(spec.smoothing)
}

pub fn run(spec: &DiagramSpec, out: &Path) -> Result<()> {
    let t0 = Instant::now();
    let mut jobs = Vec::new();
    for &s in &spec.supersaturations {
        for &t in &spec.temperatures {
            jobs.push(Conditions {
                temperature: t,
                supersaturation: s,
                pressure: physics::P_STANDARD,
            });
        }
    }
    eprintln!(
        "growing {} crystals for {:.0} s each on radius-{} lattices of {} µm cells…",
        jobs.len(),
        spec.duration,
        spec.radius,
        spec.cell_size_um
    );
    let cells: Vec<Cell> = jobs
        .into_par_iter()
        .map(|conditions| {
            let snapshot = grow(spec, conditions);
            Cell { conditions, snapshot }
        })
        .collect();
    eprintln!("grown in {:.1} s", t0.elapsed().as_secs_f64());

    // Text summary: what the diagram predicts vs. what grew.
    println!(
        "{:>8} {:>9}  {:<28} {:>7} {:>6} {:>6} {:>10}",
        "T[°C]", "σ[g/m³]", "habit (diagram)", "cells", "r", "steps", "molecules"
    );
    for c in &cells {
        println!(
            "{:>8.1} {:>9.3}  {:<28} {:>7} {:>6} {:>6} {:>10.2e}",
            c.conditions.temperature,
            c.conditions.supersaturation,
            c.conditions.morphology().to_string(),
            c.snapshot.crystal_cells,
            c.snapshot.crystal_radius,
            c.snapshot.step,
            c.snapshot.molecules
        );
    }

    // Layout: rows are supersaturations, highest on top; columns are
    // temperatures, warmest on the left.  Each cell is the top view with the
    // relief view underneath.
    let cols = spec.temperatures.len();
    let rows = spec.supersaturations.len();
    let tile = spec.tile as usize;
    // The relief view gets a square tile of its own: a needle is as tall as
    // a plate is wide.
    let render_iso = |snap: &Snapshot, scale: MassScale| {
        spec.iso
            .map(|v| iso::render(snap, scale, (spec.tile, spec.tile), IsoFit::of(snap, v), v, 2))
    };
    let cell_h = if spec.iso.is_some() { 2 * tile } else { tile };
    let label_w = 7 * FONT_W * FONT_SCALE + 12;
    let header_h = FONT_H * FONT_SCALE + 12;
    let width = label_w + cols * tile;
    let height = header_h + rows * cell_h;
    let mut canvas = Canvas::new(width, height);

    let ink = [200u8, 215, 240];
    for (ci, t) in spec.temperatures.iter().enumerate() {
        canvas.text(label_w + ci * tile + 6, 6, &format!("{t:.0}C"), ink);
    }
    for (ri, s) in spec.supersaturations.iter().rev().enumerate() {
        canvas.text(6, header_h + ri * cell_h + 6, &format!("{s:.2}"), ink);
    }

    for cell in &cells {
        let ci = spec
            .temperatures
            .iter()
            .position(|&t| t == cell.conditions.temperature)
            .unwrap();
        let ri = rows
            - 1
            - spec
                .supersaturations
                .iter()
                .position(|&s| s == cell.conditions.supersaturation)
                .unwrap();
        let snap = &cell.snapshot;
        let scale = MassScale::from_snapshot(snap);
        let fit = enclosing_radius(snap.crystal_radius.max(3));
        let rgb = raster::render(snap, View::fit(fit, spec.tile), scale, 2).rgb();
        let (x, y) = (label_w + ci * tile, header_h + ri * cell_h);
        canvas.blit(x, y, tile, tile, &rgb);
        canvas.text(
            x + 4,
            y + tile - FONT_H * FONT_SCALE - 4,
            &format!("r={}", snap.crystal_radius),
            [150, 165, 200],
        );
        if let Some(relief) = render_iso(snap, scale) {
            canvas.blit(
                x,
                y + tile,
                relief.width as usize,
                relief.height as usize,
                &relief.rgb(),
            );
        }
    }
    // Thin grid lines between cells.
    for ci in 0..=cols {
        canvas.vline(label_w + ci * tile, header_h, height, [40, 55, 90]);
    }
    for ri in 0..=rows {
        canvas.hline(label_w, width, header_h + ri * cell_h, [40, 55, 90]);
    }

    canvas.write_png(out)?;
    eprintln!("wrote {} ({}×{})", out.display(), width, height);
    Ok(())
}

/// Simple RGB canvas with a tiny bitmap font for labels.
struct Canvas {
    width: usize,
    height: usize,
    rgb: Vec<u8>,
}

const FONT_W: usize = 4; // 3 px glyph + 1 px gap
const FONT_H: usize = 5;
const FONT_SCALE: usize = 2;

/// 3×5 glyphs, one byte per row, bit 2 = left pixel.
fn glyph(c: char) -> Option<[u8; 5]> {
    Some(match c {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '=' => [0b000, 0b111, 0b000, 0b111, 0b000],
        'r' => [0b000, 0b000, 0b110, 0b100, 0b100],
        'C' => [0b111, 0b100, 0b100, 0b100, 0b111],
        ' ' => [0; 5],
        _ => return None,
    })
}

impl Canvas {
    fn new(width: usize, height: usize) -> Canvas {
        let bg = Palette::blend(0.0, 0.0);
        let mut rgb = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            rgb.extend_from_slice(&bg);
        }
        Canvas { width, height, rgb }
    }

    fn put(&mut self, x: usize, y: usize, c: [u8; 3]) {
        if x < self.width && y < self.height {
            let i = (y * self.width + x) * 3;
            self.rgb[i..i + 3].copy_from_slice(&c);
        }
    }

    fn blit(&mut self, x0: usize, y0: usize, w: usize, h: usize, rgb: &[u8]) {
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                self.put(x0 + x, y0 + y, [rgb[i], rgb[i + 1], rgb[i + 2]]);
            }
        }
    }

    fn hline(&mut self, x0: usize, x1: usize, y: usize, c: [u8; 3]) {
        for x in x0..x1 {
            self.put(x, y, c);
        }
    }

    fn vline(&mut self, x: usize, y0: usize, y1: usize, c: [u8; 3]) {
        for y in y0..y1 {
            self.put(x, y, c);
        }
    }

    fn text(&mut self, x0: usize, y0: usize, s: &str, c: [u8; 3]) {
        let mut x = x0;
        for ch in s.chars() {
            let Some(g) = glyph(ch) else { continue };
            for (row, bits) in g.iter().enumerate() {
                for col in 0..3 {
                    if bits & (0b100 >> col) != 0 {
                        for dy in 0..FONT_SCALE {
                            for dx in 0..FONT_SCALE {
                                self.put(x + col * FONT_SCALE + dx, y0 + row * FONT_SCALE + dy, c);
                            }
                        }
                    }
                }
            }
            x += FONT_W * FONT_SCALE;
        }
    }

    fn write_png(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(file), self.width as u32, self.height as u32);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&self.rgb)?;
        writer.finish()?;
        Ok(())
    }
}
