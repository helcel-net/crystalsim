//! flakesim — grow a snow crystal from a time series of environmental
//! conditions and render the result.

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use flakesim::automaton::{Automaton, Nucleation, Snapshot};
use flakesim::config::{self, GifView, History, HistoryFile};
use flakesim::diagram;
use flakesim::physics::{Conditions, kinetics_params};
use flakesim::render::iso::{self, IsoFit, IsoView};
use flakesim::render::raster::{self, Raster, View};
use flakesim::render::{MassScale, enclosing_radius, svg};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "flakesim",
    version,
    about = "Grow a snow crystal from an environmental history"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the growth simulation described by a history file and render it.
    Run(RunArgs),
    /// Show how a history file is interpreted: conditions over time, the
    /// habit the morphology diagram predicts, and the kinetics in force.
    Info {
        /// TOML history file.
        history: PathBuf,
    },
    /// Grow one crystal per (temperature, supersaturation) cell and tile
    /// them into a chart laid out like the morphology diagram.
    Diagram(DiagramArgs),
}

#[derive(Args)]
struct DiagramArgs {
    /// Output PNG.
    #[arg(long, default_value = "diagram.png")]
    out: PathBuf,
    /// Temperatures in °C, warmest first (comma separated).
    #[arg(long, value_delimiter = ',', default_value = "-2,-5,-8,-12,-15,-18,-22,-28")]
    temperatures: Vec<f64>,
    /// Supersaturations in g/m³ (comma separated).
    #[arg(long, value_delimiter = ',', default_value = "0.05,0.10,0.15,0.20,0.26,0.34")]
    supersaturations: Vec<f64>,
    /// Growth time per crystal, seconds.
    #[arg(long, default_value_t = 600.0)]
    duration: f64,
    /// Stop each crystal once it holds this many water molecules (equal-mass
    /// comparison; a 1 mm plate 20 µm thick is ~5e17).
    #[arg(long)]
    molecules: Option<f64>,
    /// Lattice radius per crystal, cells.
    #[arg(long, default_value_t = 300)]
    radius: usize,
    /// Nucleation rate of full facet sites, in fill rates (inf = deterministic).
    #[arg(long, default_value_t = 2.0)]
    nucleation: f64,
    /// Growth model: poisson (default) or quantised (deterministic, rates
    /// snapped to a geometric grid of 1 + --rate-quantum, equal rates in
    /// lockstep).
    #[arg(long, default_value = "poisson")]
    growth: String,
    /// Rate quantum of the quantised model.
    #[arg(long, default_value_t = 0.05)]
    rate_quantum: f64,
    /// Thickness-map smoothing passes.
    #[arg(long, default_value_t = 2)]
    smoothing: usize,
    /// Pixel size of each tile.
    #[arg(long, default_value_t = 220)]
    tile: u32,
    /// Top views only, without the relief view under each.
    #[arg(long)]
    no_iso: bool,
    /// Physical cell size, µm (1 µm is 4× cheaper but loses sidebranches).
    #[arg(long, default_value_t = 0.5)]
    cell_size_um: f64,
    /// Multiplier on the physical heights in the relief views (1 = as grown).
    #[arg(long, default_value_t = 1.0)]
    iso_height_scale: f64,
}

#[derive(Args)]
struct RunArgs {
    /// TOML history file.
    history: PathBuf,
    /// Final crystal as SVG (default: <history>.svg; "-" to skip).
    #[arg(long)]
    svg: Option<PathBuf>,
    /// Growth animation as GIF (default: <history>.gif; "-" to skip).
    #[arg(long)]
    gif: Option<PathBuf>,
    /// Final crystal as PNG (off unless given here or in the file).
    #[arg(long)]
    png: Option<PathBuf>,
    /// Final crystal in an elevated relief view, PNG (off unless given here or in the file).
    #[arg(long)]
    iso: Option<PathBuf>,
    /// Render the animation in the relief view instead of top-down.
    #[arg(long)]
    iso_gif: bool,
    /// Do not print progress.
    #[arg(long, short)]
    quiet: bool,
}

fn main() -> Result<()> {
    // The relaxation parallelises over (layer, row block) tasks; beyond a
    // few dozen threads memory bandwidth, not compute, is the limit.
    // RAYON_NUM_THREADS still overrides this.
    let cli = Cli::parse();
    if std::env::var_os("RAYON_NUM_THREADS").is_none() && !matches!(cli.cmd, Cmd::Diagram(_)) {
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(32);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .ok();
    }
    match cli.cmd {
        Cmd::Run(args) => run(args),
        Cmd::Info { history } => info(&history),
        Cmd::Diagram(args) => diagram::run(
            &diagram::DiagramSpec {
                temperatures: args.temperatures,
                supersaturations: args.supersaturations,
                duration: args.duration,
                molecules: args.molecules,
                radius: args.radius,
                nucleation: args.nucleation,
                quantum: if args.growth == "quantised" {
                    args.rate_quantum
                } else {
                    0.0
                },
                smoothing: args.smoothing,
                tile: args.tile,
                cell_size_um: args.cell_size_um,
                iso: (!args.no_iso).then_some(IsoView {
                    height_scale: args.iso_height_scale,
                    ..IsoView::default()
                }),
            },
            &args.out,
        ),
    }
}

/// Resolve an output path: CLI flag beats file setting beats default; "-"
/// disables the output.
fn output_path(flag: Option<PathBuf>, cfg: Option<PathBuf>, default: Option<PathBuf>) -> Option<PathBuf> {
    let p = flag.or(cfg).or(default)?;
    (p.as_os_str() != "-").then_some(p)
}

fn run(args: RunArgs) -> Result<()> {
    let file = config::load(&args.history)?;
    let history = History::from_keyframes(&file.keyframes)?;
    let stem = args
        .history
        .file_stem()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("flake"));
    let svg_path = output_path(
        args.svg,
        file.output.svg.clone(),
        Some(stem.with_extension("svg")),
    );
    let gif_path = output_path(
        args.gif,
        file.output.gif.clone(),
        Some(stem.with_extension("gif")),
    );
    let png_path = output_path(args.png, file.output.png.clone(), None);
    let iso_path = output_path(args.iso, file.output.iso.clone(), None);
    let iso_view = IsoView {
        elevation_deg: file.output.iso_elevation,
        azimuth_deg: file.output.iso_azimuth,
        height_scale: file.output.iso_height_scale,
    };
    let gif_view = if args.iso_gif {
        GifView::Iso
    } else {
        file.output.gif_view
    };

    let mut outcome = simulate(&file, &history, !args.quiet)?;
    for frame in &mut outcome.frames {
        *frame = frame.smoothed(file.output.smoothing);
    }
    let final_snap = outcome.frames.last().context("no frames")?;
    let scale = MassScale::from_snapshot(final_snap);
    let description = format!(
        "{} cells, radius {} cells, {} steps, {:.0} s of growth; final conditions {:.1} °C, {:.3} g/m³ ({})",
        final_snap.crystal_cells,
        final_snap.crystal_radius,
        final_snap.step,
        outcome.simulated_seconds,
        outcome.final_conditions.temperature,
        outcome.final_conditions.supersaturation,
        outcome.final_conditions.morphology()
    );

    let size = file.output.image_size;
    if let Some(p) = &svg_path {
        svg::write_svg(p, final_snap, size, scale, &description)?;
        log(!args.quiet, &format!("wrote {}", p.display()));
    }
    if let Some(p) = &png_path {
        let view = View::fit(enclosing_radius(final_snap.crystal_radius.max(3)), size);
        raster::render(final_snap, view, scale, 3).write_png(p)?;
        log(!args.quiet, &format!("wrote {}", p.display()));
    }
    let iso_fit = IsoFit::of(final_snap, iso_view);
    let iso_frame = iso_fit.frame(size, iso_view);
    if let Some(p) = &iso_path {
        iso::render(final_snap, scale, iso_frame, iso_fit, iso_view, 3).write_png(p)?;
        log(!args.quiet, &format!("wrote {}", p.display()));
    }
    if let Some(p) = &gif_path {
        let fit = enclosing_radius(final_snap.crystal_radius.max(3));
        let frames: Vec<Raster> = match gif_view {
            GifView::Top => {
                let view = View::fit(fit, size);
                outcome
                    .frames
                    .iter()
                    .map(|s| raster::render(s, view, scale, 2))
                    .collect()
            }
            GifView::Iso => outcome
                .frames
                .iter()
                .map(|s| iso::render(s, scale, iso_frame, iso_fit, iso_view, 2))
                .collect(),
        };
        raster::write_gif(p, &frames, file.output.frame_delay_ms, 2500)?;
        log(
            !args.quiet,
            &format!("wrote {} ({} frames)", p.display(), frames.len()),
        );
    }
    Ok(())
}

struct Outcome {
    frames: Vec<Snapshot>,
    simulated_seconds: f64,
    final_conditions: Conditions,
}

fn simulate(file: &HistoryFile, history: &History, verbose: bool) -> Result<Outcome> {
    let sim = &file.simulation;
    // Each step is sized from the fastest-growing site, so time is real
    // seconds; cap the step so the history is sampled finely enough.
    let dt_max = DT_MAX.min(history.duration() / 200.0);
    let n_frames = file.output.frames.max(2);
    let frame_time = |k: usize| history.start() + history.duration() * k as f64 / (n_frames - 1) as f64;
    // Stop before the crystal feels the fixed-vapour boundary.
    let stop_radius = sim.radius as i32 - (sim.radius as i32 / 16).max(6);

    let start = history.at(history.start());
    let nucleation = Nucleation {
        rate: sim.nucleation_rate as f32,
        symmetric: sim.symmetric,
        quantum: sim.quantum() as f32,
    };
    let mut auto = Automaton::new(sim.radius, kinetics_params(&start).sigma_inf, sim.seed)
        .with_cell_size(sim.cell_size_um as f32);
    auto.seed_hexagon();
    // Establish the quasi-static field around the seed before growing.
    auto.converge_field(&kinetics_params(&start).into());
    let mut frames = Vec::with_capacity(n_frames);
    let mut next_frame = 0usize;
    let t0 = Instant::now();
    let mut last_report = Instant::now();

    log(
        verbose,
        &format!(
            "growing for {:.0} s on a radius-{} lattice ({} µm cells, {:.0} µm across), {} keyframes",
            history.duration(),
            sim.radius,
            sim.cell_size_um,
            2.0 * sim.radius as f64 * sim.cell_size_um,
            file.keyframes.len()
        ),
    );

    let mut t = history.start();
    let end = history.end();
    loop {
        let cond = history.at(t);
        while next_frame < n_frames && frame_time(next_frame) <= t + 1e-9 {
            frames.push(auto.snapshot());
            next_frame += 1;
        }
        if t >= end - 1e-9 || auto.step_count() >= MAX_STEPS {
            break;
        }
        let remaining = (end - t) as f32;
        let dt = auto.step(
            &kinetics_params(&cond).into(),
            nucleation,
            (dt_max as f32).min(remaining),
        ) as f64;
        t = (t + dt).min(end);

        if auto.crystal_radius() >= stop_radius {
            eprintln!(
                "warning: crystal reached the edge of the lattice at t={:.0} s (step {}); \
                 stopping early — raise simulation.radius for a longer history",
                t,
                auto.step_count()
            );
            frames.push(auto.snapshot());
            break;
        }
        if sim.stop_molecules.is_some_and(|limit| auto.molecules() >= limit) {
            log(
                verbose,
                &format!("reached {:.2e} molecules at t={:.0} s", auto.molecules(), t),
            );
            frames.push(auto.snapshot());
            break;
        }
        if verbose && last_report.elapsed().as_secs_f64() > 1.0 {
            last_report = Instant::now();
            eprintln!(
                "  t={:>6.0} s  {:>5.1} °C  {:.3} g/m³  {:<26} cells={:<7} r={:<4} ({:.0} steps/s)",
                t,
                cond.temperature,
                cond.supersaturation,
                cond.morphology().to_string(),
                auto.crystal_cells(),
                auto.crystal_radius(),
                auto.step_count() as f64 / t0.elapsed().as_secs_f64()
            );
        }
    }
    if frames.last().is_none_or(|f| f.step != auto.step_count()) {
        frames.push(auto.snapshot());
    }
    log(
        verbose,
        &format!(
            "done: {} cells, radius {} cells, {:.2e} molecules after {} steps ({:.1} sweeps/step, {:.0} s simulated) in {:.1} s",
            auto.crystal_cells(),
            auto.crystal_radius(),
            auto.molecules(),
            auto.step_count(),
            auto.sweep_count() as f64 / auto.step_count().max(1) as f64,
            t - history.start(),
            t0.elapsed().as_secs_f64()
        ),
    );
    let simulated_seconds = t - history.start();
    let final_conditions = history.at(t);
    Ok(Outcome {
        frames,
        simulated_seconds,
        final_conditions,
    })
}

/// Longest time step, s.
const DT_MAX: f64 = 2.0;
/// Safety cap on the number of growth steps of one run.
const MAX_STEPS: u64 = 2_000_000;

fn info(path: &Path) -> Result<()> {
    let file = config::load(path)?;
    let history = History::from_keyframes(&file.keyframes)?;
    let sim = &file.simulation;
    println!(
        "duration {:.0} s, lattice radius {} cells of {} µm ({:.0} µm across), {}",
        history.duration(),
        sim.radius,
        sim.cell_size_um,
        2.0 * sim.radius as f64 * sim.cell_size_um,
        if sim.growth == "quantised" {
            format!("quantised growth (rate quantum {})", sim.rate_quantum)
        } else {
            format!(
                "nucleation rate {}{}",
                sim.nucleation_rate,
                if sim.nucleation_rate.is_finite() && sim.symmetric {
                    " (symmetric)"
                } else {
                    ""
                }
            )
        }
    );
    println!();
    println!(
        "{:>8}  {:>7}  {:>9}  {:>7}  {:>6}  {:>6}  {:<26}  {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "time[s]",
        "T[°C]",
        "σ[g/m³]",
        "σ_ice",
        "RH_w%",
        "p[hPa]",
        "habit (morphology diagram)",
        "v_kin",
        "X0",
        "σ0_pr",
        "σ0_ba",
        "w0_pr",
        "w0_ba"
    );
    println!(
        "{:>8}  {:>7}  {:>9}  {:>7}  {:>6}  {:>6}  {:<26}  {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "", "", "", "", "", "", "", "µm/s", "µm", "%", "%", "µm", "µm"
    );
    for (t, c) in history.keyframes() {
        print_row(t, c);
    }
    println!();
    println!("sampled every {:.0} s:", history.duration() / 10.0);
    for k in 0..=10 {
        let t = history.start() + history.duration() * k as f64 / 10.0;
        print_row(t, &history.at(t));
    }
    Ok(())
}

fn print_row(t: f64, c: &Conditions) {
    let k = kinetics_params(c);
    println!(
        "{:>8.1}  {:>7.2}  {:>9.3}  {:>6.1}%  {:>5.1}  {:>6.0}  {:<26}  {:>6.0} {:>6.3} {:>6.2} {:>6.2} {:>6.1} {:>6.1}",
        t,
        c.temperature,
        c.supersaturation,
        c.relative_supersaturation() * 100.0,
        c.humidity_water(),
        c.pressure,
        c.morphology().to_string(),
        k.v_kin,
        k.x0,
        k.sigma0_prism * 100.0,
        k.sigma0_basal * 100.0,
        k.w0_prism,
        k.w0_basal
    );
}

fn log(verbose: bool, msg: &str) {
    if verbose {
        eprintln!("{msg}");
    }
}
