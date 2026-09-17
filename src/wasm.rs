//! Browser API: a growth session driven from JavaScript, built with
//! wasm-bindgen.  The physics is the library's; the session runs single-
//! threaded (see [`crate::par`]) and renders RGBA frames for a canvas.

use crate::automaton::{Automaton, Nucleation, Snapshot};
use crate::config::{self, History};
use crate::physics::kinetics_params;
use crate::render::iso::{self, IsoFit, IsoView};
use crate::render::raster::{self, View};
use crate::render::{MassScale, enclosing_radius};
use wasm_bindgen::prelude::*;

/// Largest time step, s (the history is sampled at least this finely).
const DT_MAX: f64 = 2.0;

/// A crystal growing through a history.
#[wasm_bindgen]
pub struct Sim {
    auto: Automaton,
    history: History,
    nucleation: Nucleation,
    smoothing: usize,
    iso_view: IsoView,
    t: f64,
    dt_max: f64,
    stop_radius: i32,
    stop_molecules: Option<f64>,
    stopped: bool,
    snapshot: Option<Snapshot>,
}

#[wasm_bindgen]
impl Sim {
    /// Start a session from the text of a history file (the same TOML the
    /// command line reads).  The vapour field is converged around the seed
    /// before the first step.
    #[wasm_bindgen(constructor)]
    pub fn new(history_toml: &str) -> Result<Sim, JsError> {
        let file = config::parse(history_toml).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let history =
            History::from_keyframes(&file.keyframes).map_err(|e| JsError::new(&format!("{e:#}")))?;
        let sim = &file.simulation;
        let start = history.at(history.start());
        let nucleation = Nucleation {
            rate: sim.nucleation_rate as f32,
            symmetric: sim.symmetric,
            quantum: sim.quantum() as f32,
        };
        let mut auto = Automaton::new(sim.radius, kinetics_params(&start).sigma_inf, sim.seed)
            .with_cell_size(sim.cell_size_um as f32);
        auto.seed_hexagon();
        auto.converge_field(&kinetics_params(&start).into());
        Ok(Sim {
            auto,
            t: history.start(),
            dt_max: DT_MAX.min(history.duration() / 200.0),
            stop_radius: sim.radius as i32 - (sim.radius as i32 / 16).max(6),
            stop_molecules: sim.stop_molecules,
            history,
            nucleation,
            smoothing: file.output.smoothing,
            iso_view: IsoView {
                elevation_deg: file.output.iso_elevation,
                azimuth_deg: file.output.iso_azimuth,
                height_scale: file.output.iso_height_scale,
            },
            stopped: false,
            snapshot: None,
        })
    }

    /// Grow for up to `seconds` of simulated time (or until the history
    /// ends, the crystal nears the lattice edge, or the molecule limit is
    /// reached).  Returns the simulated time reached.
    pub fn advance(&mut self, seconds: f64) -> f64 {
        let end = self.history.end();
        let until = (self.t + seconds).min(end);
        while !self.stopped && self.t < until - 1e-9 {
            let cond = self.history.at(self.t);
            let remaining = (end - self.t) as f32;
            let dt = self.auto.step(
                &kinetics_params(&cond).into(),
                self.nucleation,
                (self.dt_max as f32).min(remaining),
            ) as f64;
            self.t = (self.t + dt).min(end);
            if self.auto.crystal_radius() >= self.stop_radius
                || self
                    .stop_molecules
                    .is_some_and(|limit| self.auto.molecules() >= limit)
            {
                self.stopped = true;
            }
        }
        if self.t >= end - 1e-9 {
            self.stopped = true;
        }
        self.snapshot = None;
        self.t
    }

    /// Simulated time reached, s.
    pub fn time(&self) -> f64 {
        self.t
    }

    /// End of the history, s.
    pub fn end(&self) -> f64 {
        self.history.end()
    }

    /// Whether growth has finished (history over, lattice edge reached, or
    /// molecule limit hit).
    pub fn done(&self) -> bool {
        self.stopped
    }

    pub fn cells(&self) -> u32 {
        self.auto.crystal_cells() as u32
    }

    /// Crystal radius, cells.
    pub fn radius(&self) -> i32 {
        self.auto.crystal_radius()
    }

    pub fn steps(&self) -> u32 {
        self.auto.step_count() as u32
    }

    pub fn molecules(&self) -> f64 {
        self.auto.molecules()
    }

    /// Temperature (°C) of the history at the current time.
    pub fn temperature(&self) -> f64 {
        self.history.at(self.t).temperature
    }

    /// Supersaturation (g/m³ over ice) of the history at the current time.
    pub fn supersaturation(&self) -> f64 {
        self.history.at(self.t).supersaturation
    }

    /// Habit the morphology diagram predicts for the current conditions.
    pub fn habit(&self) -> String {
        self.history.at(self.t).morphology().to_string()
    }

    fn snap(&mut self) -> &Snapshot {
        if self.snapshot.is_none() {
            self.snapshot = Some(self.auto.snapshot().smoothed(self.smoothing));
        }
        self.snapshot.as_ref().unwrap()
    }

    /// The crystal from above, `size`×`size` RGBA pixels.
    pub fn render_top(&mut self, size: u32) -> Vec<u8> {
        let snap = self.snap();
        let view = View::fit(enclosing_radius(snap.crystal_radius.max(3)), size);
        let scale = MassScale::from_snapshot(snap);
        rgba(&raster::render(snap, view, scale, 2).rgb())
    }

    /// Height of the relief frame (pixels) for a frame `width` wide.
    pub fn iso_height(&mut self, width: u32) -> u32 {
        let view = self.iso_view;
        let snap = self.snap();
        IsoFit::of(snap, view).frame(width, view).1
    }

    /// The crystal in the relief view, `width`×`iso_height(width)` RGBA.
    pub fn render_iso(&mut self, width: u32) -> Vec<u8> {
        let view = self.iso_view;
        let snap = self.snap();
        let fit = IsoFit::of(snap, view);
        let frame = fit.frame(width, view);
        let scale = MassScale::from_snapshot(snap);
        rgba(&iso::render(snap, scale, frame, fit, view, 2).rgb())
    }
}

fn rgba(rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rgb.len() / 3 * 4);
    for px in rgb.chunks(3) {
        out.extend_from_slice(px);
        out.push(255);
    }
    out
}
