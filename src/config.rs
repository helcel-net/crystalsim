//! The environmental-history file: a TOML document with simulation settings,
//! output settings and a list of keyframes that is interpolated linearly in
//! time.

use crate::physics::{self, Conditions};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryFile {
    #[serde(default)]
    pub simulation: SimulationCfg,
    #[serde(default)]
    pub output: OutputCfg,
    #[serde(rename = "history")]
    pub keyframes: Vec<Keyframe>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimulationCfg {
    /// Hex-cell radius of the simulation domain.
    #[serde(default = "d_radius")]
    pub radius: usize,
    /// Physical size of a lattice cell, µm (default 0.5).  Growth rates are
    /// validated at 0.5–1 µm; sidebranching needs 0.5 µm, 1 µm gives arms
    /// only, 2 µm is too coarse for the boundary condition.
    #[serde(default = "d_cell_size")]
    pub cell_size_um: f64,
    /// Stop once the crystal holds this many water molecules (optional; a
    /// 1 mm plate 20 µm thick is ~5e17).  Useful for comparing habits at
    /// equal mass.
    pub stop_molecules: Option<f64>,
    /// Seed of the vapour-noise generator.
    #[serde(default)]
    pub seed: u64,
    /// Once a facet site holds a full cell of mass it nucleates as a Poisson
    /// process with this rate, in units of its own fill rate (default 2:
    /// the mean extra wait is half a fill time).  Independent of the time
    /// step; every site draws independently.  `inf` is deterministic:
    /// perfectly regular sidebranches and featureless facets; lower values
    /// give ridges, sectors and irregular branching.
    #[serde(default = "d_nucleation")]
    pub nucleation_rate: f64,
    /// Growth model: "poisson" (default; stochastic layer nucleation, see
    /// `nucleation_rate`) or "quantised" (deterministic: every site's rate
    /// is snapped to a geometric grid with steps of 1 + `rate_quantum`, and
    /// equal-rate sites attach in lockstep — homogeneous, exactly symmetric
    /// growth, no randomness).
    #[serde(default = "d_growth")]
    pub growth: String,
    /// Rate quantum of the quantised model (default 0.05: rates within
    /// 2.5 % of each other are made equal; smaller is closer to the
    /// deterministic Poisson model, larger more regular).
    #[serde(default = "d_quantum")]
    pub rate_quantum: f64,
    /// Use the same random draw for all twelve symmetric images of a cell,
    /// so the crystal stays perfectly symmetric (default true).  Set to false
    /// for independent per-cell randomness, which makes the six arms differ.
    #[serde(default = "d_true")]
    pub symmetric: bool,
}

impl SimulationCfg {
    /// Rate quantum in force: the configured one for the quantised model,
    /// zero for the Poisson model.
    pub fn quantum(&self) -> f64 {
        if self.growth == "quantised" {
            self.rate_quantum
        } else {
            0.0
        }
    }
}

impl Default for SimulationCfg {
    fn default() -> Self {
        Self {
            radius: d_radius(),
            cell_size_um: d_cell_size(),
            stop_molecules: None,
            seed: 0,
            nucleation_rate: d_nucleation(),
            growth: d_growth(),
            rate_quantum: d_quantum(),
            symmetric: true,
        }
    }
}

fn d_true() -> bool {
    true
}

fn d_nucleation() -> f64 {
    2.0
}

fn d_growth() -> String {
    "poisson".to_string()
}

fn d_quantum() -> f64 {
    0.05
}

fn d_radius() -> usize {
    600
}
fn d_cell_size() -> f64 {
    0.5
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputCfg {
    pub svg: Option<PathBuf>,
    pub gif: Option<PathBuf>,
    pub png: Option<PathBuf>,
    /// Elevated relief view of the final crystal (PNG).
    pub iso: Option<PathBuf>,
    /// Which view the animation uses: "top" (default) or "iso".
    #[serde(default)]
    pub gif_view: GifView,
    /// Camera elevation of the relief view, degrees above the basal plane.
    #[serde(default = "d_iso_elevation")]
    pub iso_elevation: f64,
    /// In-plane rotation of the crystal in the relief view, degrees.
    #[serde(default = "d_iso_azimuth")]
    pub iso_azimuth: f64,
    /// Multiplier on the physical c-axis heights in the relief view
    /// (1 = as grown).
    #[serde(default = "d_iso_height_scale")]
    pub iso_height_scale: f64,
    /// Pixel size of raster outputs (square).
    #[serde(default = "d_image_size")]
    pub image_size: u32,
    /// Number of animation frames spread over the run.
    #[serde(default = "d_frames")]
    pub frames: usize,
    /// Delay between animation frames, milliseconds.
    #[serde(default = "d_frame_delay")]
    pub frame_delay_ms: u32,
    /// Smoothing passes applied to the thickness map before rendering
    /// (0 shows the raw per-cell mass with its layer-by-layer stripes).
    #[serde(default = "d_smoothing")]
    pub smoothing: usize,
}

impl Default for OutputCfg {
    fn default() -> Self {
        Self {
            svg: None,
            gif: None,
            png: None,
            iso: None,
            gif_view: GifView::Top,
            iso_elevation: d_iso_elevation(),
            iso_azimuth: d_iso_azimuth(),
            iso_height_scale: d_iso_height_scale(),
            image_size: d_image_size(),
            frames: d_frames(),
            frame_delay_ms: d_frame_delay(),
            smoothing: d_smoothing(),
        }
    }
}

fn d_smoothing() -> usize {
    2
}
fn d_iso_elevation() -> f64 {
    35.0
}
fn d_iso_azimuth() -> f64 {
    20.0
}
fn d_iso_height_scale() -> f64 {
    1.0
}

#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GifView {
    #[default]
    Top,
    Iso,
}

fn d_image_size() -> u32 {
    800
}
fn d_frames() -> usize {
    80
}
fn d_frame_delay() -> u32 {
    60
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Keyframe {
    /// Seconds since the start of growth.
    pub time: f64,
    /// Air temperature, °C.
    pub temperature: f64,
    /// Excess vapour density over ice saturation, g/m³.
    pub supersaturation: Option<f64>,
    /// Relative humidity w.r.t. liquid water, %.  Alternative to
    /// `supersaturation`; 100 % is the "water saturation" line of the
    /// morphology diagram.
    pub humidity: Option<f64>,
    /// Air pressure, hPa (default 1013.25).  Diffusion is faster in thinner
    /// air, so growth is less diffusion-limited at altitude.
    pub pressure: Option<f64>,
}

impl Keyframe {
    fn conditions(&self) -> Result<Conditions> {
        let supersaturation = match (self.supersaturation, self.humidity) {
            (Some(s), None) => s,
            (None, Some(rh)) => physics::supersaturation_from_rh(rh, self.temperature),
            (Some(_), Some(_)) => bail!(
                "keyframe at t={}: give either `supersaturation` or `humidity`, not both",
                self.time
            ),
            (None, None) => bail!(
                "keyframe at t={}: needs `supersaturation` (g/m³) or `humidity` (%)",
                self.time
            ),
        };
        ensure!(
            self.temperature > -80.0 && self.temperature < 5.0,
            "keyframe at t={}: temperature {} °C is outside the range of ice growth",
            self.time,
            self.temperature
        );
        let pressure = self.pressure.unwrap_or(physics::P_STANDARD);
        ensure!(
            pressure > 10.0 && pressure < 2000.0,
            "keyframe at t={}: pressure {} hPa is out of range",
            self.time,
            pressure
        );
        Ok(Conditions {
            temperature: self.temperature,
            supersaturation,
            pressure,
        })
    }
}

/// A validated, interpolable environmental history.
#[derive(Debug, Clone)]
pub struct History {
    times: Vec<f64>,
    conditions: Vec<Conditions>,
}

impl History {
    pub fn from_keyframes(keyframes: &[Keyframe]) -> Result<History> {
        ensure!(
            keyframes.len() >= 2,
            "the history needs at least two [[history]] keyframes"
        );
        let mut times = Vec::with_capacity(keyframes.len());
        let mut conditions = Vec::with_capacity(keyframes.len());
        for (i, k) in keyframes.iter().enumerate() {
            ensure!(
                k.time.is_finite() && k.time >= 0.0,
                "keyframe {} has an invalid time",
                i
            );
            if let Some(prev) = times.last() {
                ensure!(
                    k.time >= *prev,
                    "keyframes must be sorted by time (keyframe {})",
                    i
                );
            }
            let mut c = k.conditions()?;
            if c.supersaturation < 0.0 {
                eprintln!(
                    "warning: keyframe at t={} is sub-saturated w.r.t. ice ({:.3} g/m³); \
                     the crystal will not grow there (sublimation is not modelled)",
                    k.time, c.supersaturation
                );
                c.supersaturation = 0.0;
            }
            times.push(k.time);
            conditions.push(c);
        }
        ensure!(times[times.len() - 1] > times[0], "the history has zero duration");
        Ok(History { times, conditions })
    }

    pub fn start(&self) -> f64 {
        self.times[0]
    }

    pub fn end(&self) -> f64 {
        self.times[self.times.len() - 1]
    }

    pub fn duration(&self) -> f64 {
        self.end() - self.start()
    }

    pub fn keyframes(&self) -> impl Iterator<Item = (f64, &Conditions)> {
        self.times.iter().copied().zip(self.conditions.iter())
    }

    /// Conditions at time `t` (clamped to the history's span).
    pub fn at(&self, t: f64) -> Conditions {
        if t <= self.start() {
            return self.conditions[0];
        }
        if t >= self.end() {
            return self.conditions[self.conditions.len() - 1];
        }
        // First keyframe strictly after t; the segment is [i-1, i].
        let i = self.times.partition_point(|&kt| kt <= t);
        let (t0, t1) = (self.times[i - 1], self.times[i]);
        let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 1.0 };
        Conditions::lerp(&self.conditions[i - 1], &self.conditions[i], f)
    }
}

pub fn load(path: &Path) -> Result<HistoryFile> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read history file {}", path.display()))?;
    parse(&text).with_context(|| format!("in {}", path.display()))
}

/// Parse and validate a history file's text.
pub fn parse(text: &str) -> Result<HistoryFile> {
    let file: HistoryFile = toml::from_str(text).context("cannot parse the history")?;
    ensure!(
        file.simulation.radius >= 20,
        "simulation.radius must be at least 20"
    );
    ensure!(
        file.simulation.radius <= 3000,
        "simulation.radius above 3000 is impractical"
    );
    ensure!(
        file.simulation.nucleation_rate > 0.0,
        "simulation.nucleation_rate must be positive (use inf for deterministic)"
    );
    ensure!(
        matches!(file.simulation.growth.as_str(), "poisson" | "quantised"),
        "simulation.growth must be \"poisson\" or \"quantised\""
    );
    ensure!(
        file.simulation.rate_quantum > 0.0 && file.simulation.rate_quantum <= 1.0,
        "simulation.rate_quantum must be in (0, 1]"
    );
    ensure!(
        file.output.image_size >= 64,
        "output.image_size must be at least 64"
    );
    ensure!(file.output.frames >= 2, "output.frames must be at least 2");
    ensure!(
        file.output.iso_elevation > 5.0 && file.output.iso_elevation <= 90.0,
        "output.iso_elevation must be between 5 and 90 degrees"
    );
    ensure!(
        file.output.iso_height_scale > 0.0,
        "output.iso_height_scale must be positive"
    );
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kf(time: f64, temperature: f64, s: f64) -> Keyframe {
        Keyframe {
            time,
            temperature,
            supersaturation: Some(s),
            humidity: None,
            pressure: None,
        }
    }

    #[test]
    fn interpolates_between_keyframes() {
        let h = History::from_keyframes(&[kf(0.0, -10.0, 0.1), kf(100.0, -20.0, 0.3)]).unwrap();
        let c = h.at(50.0);
        assert!((c.temperature + 15.0).abs() < 1e-12);
        assert!((c.supersaturation - 0.2).abs() < 1e-12);
        assert_eq!(h.at(-5.0), h.at(0.0));
        assert_eq!(h.at(500.0), h.at(100.0));
    }

    #[test]
    fn step_changes_via_duplicate_times() {
        let h = History::from_keyframes(&[
            kf(0.0, -15.0, 0.3),
            kf(60.0, -15.0, 0.3),
            kf(60.0, -2.0, 0.05),
            kf(120.0, -2.0, 0.05),
        ])
        .unwrap();
        assert!((h.at(59.999).temperature + 15.0).abs() < 1e-9);
        assert!((h.at(60.0).temperature + 2.0).abs() < 1e-9);
    }

    #[test]
    fn missing_sections_get_defaults() {
        let file: HistoryFile = toml::from_str(
            "[[history]]\ntime=0\ntemperature=-15\nsupersaturation=0.2\n\
             [[history]]\ntime=10\ntemperature=-15\nsupersaturation=0.2\n",
        )
        .unwrap();
        assert_eq!(file.output.image_size, d_image_size());
        assert_eq!(file.output.frames, d_frames());
        assert_eq!(file.simulation.radius, d_radius());
    }

    #[test]
    fn rejects_unsorted_and_short_histories() {
        assert!(History::from_keyframes(&[kf(0.0, -15.0, 0.3)]).is_err());
        assert!(History::from_keyframes(&[kf(10.0, -15.0, 0.3), kf(0.0, -15.0, 0.3)]).is_err());
    }
}
