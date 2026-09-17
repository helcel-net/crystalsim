//! Pixel rendering, PNG stills and animated GIFs.

use super::{MassScale, Palette, cell_at};
use crate::automaton::Snapshot;
use crate::par::*;
use anyhow::{Context, Result};
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

/// A rendered image: per pixel the ice coverage (anti-aliasing), the mean
/// normalised thickness of the covered part and, for lit renders, a shading
/// factor.
pub struct Raster {
    pub width: u32,
    pub height: u32,
    pub coverage: Vec<f32>,
    pub thickness: Vec<f32>,
    pub shade: Option<Vec<f32>>,
}

/// Viewport: which part of the lattice fills the image.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// Pixels per hex unit.
    pub scale: f64,
    pub size: u32,
}

impl View {
    /// Fit a circle of `radius` hex units into a `size`-pixel square.
    pub fn fit(radius: f64, size: u32) -> View {
        View {
            scale: size as f64 / 2.0 / radius,
            size,
        }
    }
}

pub fn render(snap: &Snapshot, view: View, mass: MassScale, supersample: u32) -> Raster {
    let size = view.size as usize;
    let ss = supersample.max(1) as usize;
    let half = view.size as f64 / 2.0;
    let mut coverage = vec![0.0f32; size * size];
    let mut thickness = vec![0.0f32; size * size];

    coverage
        .par_chunks_mut(size)
        .zip(thickness.par_chunks_mut(size))
        .enumerate()
        .for_each(|(py, (cov_row, th_row))| {
            for px in 0..size {
                let mut hits = 0u32;
                let mut sum = 0.0f32;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let x = (px as f64 + (sx as f64 + 0.5) / ss as f64 - half) / view.scale;
                        let y = (py as f64 + (sy as f64 + 0.5) / ss as f64 - half) / view.scale;
                        let (q, r) = cell_at(x, y);
                        if let Some(m) = snap.mass_at(q, r) {
                            hits += 1;
                            sum += mass.thickness(m);
                        }
                    }
                }
                if hits > 0 {
                    cov_row[px] = hits as f32 / (ss * ss) as f32;
                    th_row[px] = sum / hits as f32;
                }
            }
        });

    Raster {
        width: view.size,
        height: view.size,
        coverage,
        thickness,
        shade: None,
    }
}

impl Raster {
    fn shade_at(&self, i: usize) -> f32 {
        self.shade.as_ref().map_or(1.0, |s| s[i])
    }

    pub fn rgb(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.coverage.len() * 3);
        for (i, (c, t)) in self.coverage.iter().zip(&self.thickness).enumerate() {
            out.extend_from_slice(&Palette::blend_shaded(*t, self.shade_at(i), *c));
        }
        out
    }

    pub fn write_png(&self, path: &Path) -> Result<()> {
        let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
        let mut enc = png::Encoder::new(BufWriter::new(file), self.width, self.height);
        enc.set_color(png::ColorType::Rgb);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header()?;
        writer.write_image_data(&self.rgb())?;
        writer.finish()?;
        Ok(())
    }
}

/// GIF palettes: 16 coverage × 16 thickness levels for flat renders, or
/// 4 coverage × 8 thickness × 8 shade levels for lit ones.
#[derive(Clone, Copy)]
struct GifPalette {
    coverage: usize,
    thickness: usize,
    shade: usize,
}

impl GifPalette {
    fn for_raster(r: &Raster) -> GifPalette {
        if r.shade.is_some() {
            GifPalette {
                coverage: 4,
                thickness: 8,
                shade: 8,
            }
        } else {
            GifPalette {
                coverage: 16,
                thickness: 16,
                shade: 1,
            }
        }
    }

    fn colours(&self) -> Vec<u8> {
        let mut pal = Vec::with_capacity(self.coverage * self.thickness * self.shade * 3);
        for ci in 0..self.coverage {
            for ti in 0..self.thickness {
                for si in 0..self.shade {
                    let c = ci as f32 / (self.coverage - 1) as f32;
                    let t = ti as f32 / (self.thickness - 1) as f32;
                    let s = if self.shade > 1 {
                        si as f32 / (self.shade - 1) as f32
                    } else {
                        1.0
                    };
                    pal.extend_from_slice(&Palette::blend_shaded(t, s, c));
                }
            }
        }
        pal
    }

    fn indices(&self, raster: &Raster) -> Vec<u8> {
        let level = |v: f32, n: usize| (v.clamp(0.0, 1.0) * (n - 1) as f32).round() as usize;
        (0..raster.coverage.len())
            .map(|i| {
                let ci = level(raster.coverage[i], self.coverage);
                let ti = level(raster.thickness[i], self.thickness);
                let si = if self.shade > 1 {
                    level(raster.shade_at(i), self.shade)
                } else {
                    0
                };
                ((ci * self.thickness + ti) * self.shade + si) as u8
            })
            .collect()
    }
}

/// Write an animated GIF; frames are given in order, `delay_ms` apart, and
/// the last frame is held for `hold_ms`.
pub fn write_gif(path: &Path, frames: &[Raster], delay_ms: u32, hold_ms: u32) -> Result<()> {
    let Some(first) = frames.first() else {
        anyhow::bail!("no frames to write")
    };
    let (width, height) = (first.width as u16, first.height as u16);
    let palette = GifPalette::for_raster(first);
    let file = File::create(path).with_context(|| format!("cannot create {}", path.display()))?;
    let mut enc = gif::Encoder::new(BufWriter::new(file), width, height, &palette.colours())?;
    enc.set_repeat(gif::Repeat::Infinite)?;
    for (i, raster) in frames.iter().enumerate() {
        let mut frame = gif::Frame {
            width,
            height,
            buffer: std::borrow::Cow::Owned(palette.indices(raster)),
            ..gif::Frame::default()
        };
        let ms = if i + 1 == frames.len() { hold_ms } else { delay_ms };
        frame.delay = (ms / 10).max(2) as u16;
        enc.write_frame(&frame)?;
    }
    Ok(())
}
