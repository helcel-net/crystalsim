//! Snow-crystal growth on a hexagonal lattice, following the cellular-
//! automaton scheme of Libbrecht's *Snow Crystals* (2021, chapters 3–5).
//!
//! The lattice is the basal plane, one cell `dx_um` micrometres across, and
//! the crystal is a height map on it: each crystal cell carries its extent
//! `h` along the c-axis, centred on the plane.  The supersaturation field is
//! three-dimensional: the same lattice stacked in layers along the c-axis,
//! one cell thick next to the plane and doubling in thickness outward, with
//! the plane itself a mirror.  A thin plate edge is thus fed from above and
//! below as it is in reality, and a tall prism face is the slab it is.  One
//! growth step:
//!
//! 1. the field is relaxed to its quasi-static solution with the mixed
//!    boundary condition `X₀ ∂σ/∂n = α σ_surf` on every ice face — prism
//!    faces at the edges of the height map, basal faces on top of it —
//!    where `α` is the attachment coefficient of that face;
//! 2. every boundary pixel fills at `v = α v_kin (σ_surf − d₀κ)`, averaged
//!    over the height of the edge it extends, for a time step chosen so the
//!    fastest pixel fills in two steps; full pixels become ice (facet sites
//!    subject to a nucleation draw);
//! 3. new ice inherits the height of its neighbours, thinned where the
//!    edge-sharpening instability acts, and every basal face grows with the
//!    same kinetics from the vapour above it.
//!
//! The random draws can be shared between the twelve symmetric images of a
//! cell so the crystal stays symmetric while its sidebranching is irregular.

use crate::par::*;
use crate::physics::{KineticsParams, attachment_coefficient, sdak_barrier};

const SQRT3: f32 = 1.732_050_8;

/// Raw pointer that may be shared between threads; the relaxation passes
/// guarantee disjoint writes (see `relax_field_with`).
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// Per-layer data for the cell update (see `Automaton::cell_terms`).
struct LayerCtx<'a> {
    l: &'a Layer,
    occ: &'a [f32],
    hmean: &'a [f32],
    edges: &'a [f32],
    bas_narrow: &'a [f32],
    bas_wide: &'a [f32],
    /// Offsets of the six in-plane neighbours in the layer.
    offs: [isize; 6],
}

/// Flux balance of one field cell (see `Automaton::cell_terms`).
struct CellTerms {
    sum: f32,
    g: f32,
    uptake: f32,
    sigma: f32,
    idx: usize,
}

/// Attachment-kinetics model parameters in single precision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Kinetics {
    pub sigma_inf: f32,
    pub v_kin: f32,
    pub x0: f32,
    pub d0: f32,
    pub a_prism: f32,
    pub sigma0_prism: f32,
    pub w0_prism: f32,
    pub a_basal: f32,
    pub sigma0_basal: f32,
    pub w0_basal: f32,
}

impl From<KineticsParams> for Kinetics {
    fn from(k: KineticsParams) -> Self {
        Kinetics {
            sigma_inf: k.sigma_inf as f32,
            v_kin: k.v_kin as f32,
            x0: k.x0 as f32,
            d0: k.d0 as f32,
            a_prism: k.a_prism as f32,
            sigma0_prism: k.sigma0_prism as f32,
            w0_prism: k.w0_prism as f32,
            a_basal: k.a_basal as f32,
            sigma0_basal: k.sigma0_basal as f32,
            w0_basal: k.w0_basal as f32,
        }
    }
}

/// How boundary sites turn their growth rate into attachments: the Poisson
/// model (`quantum == 0`) or the quantised-rate model (`quantum > 0`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nucleation {
    /// Poisson model: once a facet site holds a full cell of mass it
    /// nucleates as a Poisson process with this rate, in units of the site's
    /// own fill rate (so the mean extra wait is 1/rate of a fill time,
    /// whatever the time step).  Every site draws independently, so any
    /// number of sites can nucleate in the same step; unattached mass is
    /// carried over.  `f32::INFINITY` is deterministic.
    pub rate: f32,
    /// Poisson model: draw one random number per symmetry class of the
    /// lattice instead of per cell, so the crystal stays perfectly six-fold
    /// (and mirror) symmetric while its sidebranching is still irregular —
    /// the way real crystals look, since all six arms grow through the same
    /// history.
    pub symmetric: bool,
    /// Quantised model (when positive): no random draws.  Every site's rate
    /// is expressed as a fraction of the fastest site's and snapped to a
    /// geometric grid with steps of (1 + quantum) — so with 0.1 every site
    /// is within 5 % of its own rate, and all sites with the same quantised
    /// rate attach in lockstep: the six vertices of a hexagon, a whole
    /// facet row, the whole rim of a column.  Basal growth is quantised the
    /// same way.  Growth is homogeneous and exactly symmetric, and as the
    /// quantum goes to zero the model becomes the deterministic Poisson
    /// model (`rate = ∞`).
    pub quantum: f32,
}

impl Nucleation {
    #[cfg(test)]
    pub const DETERMINISTIC: Nucleation = Nucleation {
        rate: f32::INFINITY,
        symmetric: true,
        quantum: 0.0,
    };

    /// Whether the quantised-rate model is in force.
    pub fn quantised(&self) -> bool {
        self.quantum > 0.0
    }

    /// Quantise a rate: `value` relative to `max` is snapped to a geometric
    /// grid with steps of (1 + quantum) — the fastest site, then 1/1.1,
    /// 1/1.1², … — so every site stays within half a quantum of its own
    /// rate whatever its speed, and sites within that of each other attach
    /// in lockstep.
    #[inline]
    fn quantise(&self, value: f32, max: f32) -> f32 {
        if value <= 0.0 || max <= 0.0 {
            return 0.0;
        }
        let p = (value / max).min(1.0);
        let step = (1.0 + self.quantum).ln();
        let k = (p.ln() / step).round();
        (k * step).exp() * max
    }
}

/// Frozen copy of the crystal for rendering.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Domain radius in cells.
    pub radius: i32,
    /// Row stride (2·radius + 1).
    pub width: usize,
    /// Crystal mass per cell; negative for cells that are not crystal.
    pub mass: Vec<f32>,
    /// Extent of each crystal cell along the c-axis, in cells (0 elsewhere).
    pub height: Vec<f32>,
    /// Largest hex distance of any crystal cell from the centre.
    pub crystal_radius: i32,
    /// Number of crystal cells.
    pub crystal_cells: usize,
    /// Automaton step at which the snapshot was taken.
    pub step: u64,
    /// Water molecules in the crystal.
    pub molecules: f64,
}

impl Snapshot {
    /// Crystal mass at axial coordinates (q, r), or `None` outside the crystal.
    pub fn mass_at(&self, q: i32, r: i32) -> Option<f32> {
        let big_r = self.radius;
        if q < -big_r || q > big_r || r < -big_r || r > big_r {
            return None;
        }
        let m = self.mass[(r + big_r) as usize * self.width + (q + big_r) as usize];
        (m >= 0.0).then_some(m)
    }

    /// Copy with the crystal mass averaged over each cell and its crystal
    /// neighbours, `passes` times.  Facets grow layer by layer, and the
    /// waiting time of each layer leaves a one-cell-wide stripe in the mass;
    /// real thickness varies smoothly, so rendering uses a smoothed map in
    /// which the radial ridges survive and the stripes do not.
    pub fn smoothed(&self, passes: usize) -> Snapshot {
        let mut out = self.clone();
        let w = self.width as isize;
        let offs = [1isize, -1, w, -w, 1 - w, w - 1];
        for _ in 0..passes {
            let src = out.mass.clone();
            let src_h = out.height.clone();
            for i in 0..src.len() {
                if src[i] < 0.0 {
                    continue;
                }
                let (mut sum, mut sum_h, mut n) = (src[i], src_h[i], 1.0);
                for off in offs {
                    let j = i as isize + off;
                    if j >= 0 && (j as usize) < src.len() && src[j as usize] >= 0.0 {
                        sum += src[j as usize];
                        sum_h += src_h[j as usize];
                        n += 1.0;
                    }
                }
                out.mass[i] = sum / n;
                out.height[i] = sum_h / n;
            }
        }
        out
    }

    /// Height of cell (q, r) along the c-axis, in cells (0 outside the crystal).
    pub fn height_at(&self, q: i32, r: i32) -> f32 {
        let big_r = self.radius;
        if q < -big_r || q > big_r || r < -big_r || r > big_r {
            return 0.0;
        }
        self.height[(r + big_r) as usize * self.width + (q + big_r) as usize]
    }

    /// Tallest cell, in cells.
    pub fn max_height(&self) -> f32 {
        self.height.iter().cloned().fold(0.0, f32::max)
    }

    /// Crystal cells as (q, r, mass).
    pub fn cells(&self) -> impl Iterator<Item = (i32, i32, f32)> + '_ {
        let big_r = self.radius;
        let w = self.width;
        self.mass
            .iter()
            .enumerate()
            .filter(|(_, m)| **m >= 0.0)
            .map(move |(i, m)| {
                let r = (i / w) as i32 - big_r;
                let q = (i % w) as i32 - big_r;
                (q, r, *m)
            })
    }
}

/// One layer of the field: the hex lattice at an in-plane spacing of `s`
/// cells (a sublattice of the plane's; layers thicker than four cells are
/// coarsened in step with their thickness, so the field far above the
/// plane costs little).
#[derive(Clone, Debug)]
struct Layer {
    /// In-plane spacing, cells (a power of two).
    s: i32,
    /// Lattice radius at that spacing.
    radius: i32,
    /// Row stride.
    w: usize,
    /// First index of the layer in the field array.
    off: usize,
    /// Cells in the layer.
    n: usize,
    /// Thickness, cells.
    dz: f32,
    /// Height of the layer's centre above the plane, cells.
    zc: f32,
    /// Conductance of an in-plane link: face s/√3 by dz over distance s.
    g_h: f32,
    /// Conductance from one cell of this layer to the layer above (or to
    /// the top boundary from the last layer; zero when that reflects).  A
    /// coarser layer above gathers four of these per coarse cell.
    g_up: f32,
}

impl Layer {
    /// Index of the cell at this layer's coordinates (q, r).
    #[inline]
    fn index(&self, q: i32, r: i32) -> usize {
        self.off + (r + self.radius) as usize * self.w + (q + self.radius) as usize
    }

    /// Nearest cell of this layer to fine-lattice coordinates (q, r).
    #[inline]
    fn nearest(&self, q: i32, r: i32) -> (i32, i32) {
        if self.s == 1 {
            return (q, r);
        }
        hex_round(q as f32 / self.s as f32, r as f32 / self.s as f32)
    }
}

/// Round fractional axial coordinates to the nearest hex cell.
#[inline]
fn hex_round(q: f32, r: f32) -> (i32, i32) {
    let y = -q - r;
    let (mut rq, mut rr, rs) = (q.round(), r.round(), y.round());
    let (dq, dr, ds) = ((rq - q).abs(), (rr - r).abs(), (rs - y).abs());
    if dq > dr && dq > ds {
        rq = -rr - rs;
    } else if dr > ds {
        rr = -rq - rs;
    }
    (rq as i32, rr as i32)
}

pub struct Automaton {
    radius: i32,
    w: usize,
    /// Cells per field layer (w²).
    n: usize,
    a: Vec<u8>,
    b: Vec<f32>,
    c: Vec<f32>,
    /// Supersaturation field, layer-major: layer `j` of cell `i` is
    /// `d[j * n + i]`.
    d: Vec<f32>,
    attach: Vec<u8>,
    /// c-axis extent of each crystal cell, in cells.
    h: Vec<f32>,
    /// Field layers each crystal cell fills (0 for vapour cells).
    top: Vec<u8>,
    /// Distance of each crystal cell from the crystal's rim, in cells
    /// (saturating), refreshed every few steps.
    rim: Vec<u8>,
    /// Tallest cell, in cells.
    h_max: f32,
    /// Kinetics model: fill rate of each boundary pixel, µm/s.
    rate: Vec<f32>,
    /// In-plane curvature of the interface at each boundary pixel, 1/µm
    /// (positive where the ice is convex), from the ice fraction of a disc
    /// [`Automaton::CURVATURE_UM`] in radius around it.
    curv: Vec<f32>,
    /// Calibrated gain turning that ice fraction into a curvature.
    curv_scale: f32,

    /// Physical cell size, µm.
    dx_um: f32,
    /// Thickness of each field layer, in cells.
    dz: Vec<f32>,
    /// Height of each layer's centre above the plane, in cells.
    zc: Vec<f32>,
    /// Geometry of each field layer.
    layers: Vec<Layer>,
    /// Coarse layers only, per coarse cell, restricted from the height map
    /// each step: ice fraction, mean ice height (cells), number of prism
    /// faces (boundary pixels) inside the cell, and basal face area (cells²)
    /// on narrow and on wide terraces.
    occ: Vec<Vec<f32>>,
    hmean: Vec<Vec<f32>>,
    edges: Vec<Vec<f32>>,
    bas_narrow: Vec<Vec<f32>>,
    bas_wide: Vec<Vec<f32>>,
    /// Far-field boundary: the outer ring and the top carry the monopole
    /// field σ_∞ − Q/(4πr) of the crystal's total uptake Q instead of σ_∞,
    /// so the box behaves like an infinite domain to first order.
    far_field: bool,
    /// Q/(4π) of the last field, in cells (the crystal's capacitance
    /// radius when σ_surf ≪ σ_∞).
    q_far: f32,
    /// Simulated time, s.
    #[allow(dead_code)]
    time: f64,
    /// Relaxation sweeps performed so far (cost diagnostics).
    sweeps: u64,
    /// Most relaxation sweeps per growth step.
    relax_budget: usize,
    /// Rows handled by one parallel task.
    rows_per_chunk: usize,
    rho: f32,
    seed: u64,
    step: u64,
    crystal_cells: usize,
    crystal_radius: i32,
}

/// Water molecules in `height_sum` cells of ice stacked on cells `dx_um` µm
/// apart (hexagonal cells of area (√3/2)·dx², ice at 917 kg/m³, 2.991e-26 kg
/// per molecule).
pub fn molecules_from_height_sum(height_sum: f64, dx_um: f64) -> f64 {
    let cell_volume_um3 = 0.866_025_4 * dx_um * dx_um * dx_um;
    height_sum * cell_volume_um3 * 1e-18 * 917.0 / 2.991e-26
}

/// Hex distance from the centre in axial coordinates.
#[inline]
pub fn hex_distance(q: i32, r: i32) -> i32 {
    q.abs().max(r.abs()).max((q + r).abs())
}

/// Range of q for row r inside the updated region (hex distance < radius).
#[inline]
fn inner_range(radius: i32, r: i32) -> Option<(i32, i32)> {
    let m = radius - 1;
    if r.abs() > m {
        return None;
    }
    Some(((-m).max(-m - r), m.min(m - r)))
}

impl Automaton {
    /// Area of a hexagonal cell of unit spacing, √3/2.
    const CELL_AREA: f32 = 0.866_025_4;

    /// Domain of the given cell radius, seeded with a single crystal cell at
    /// the centre and supersaturation `sigma_inf` everywhere else.
    pub fn new(radius: usize, sigma_inf: f64, seed: u64) -> Automaton {
        let big_r = radius as i32;
        let w = 2 * radius + 1;
        let n = w * w;
        let rho = sigma_inf as f32;
        let dz = Self::layers_for(radius);
        let layers = dz.len();
        let mut auto = Automaton {
            radius: big_r,
            w,
            n,
            a: vec![0; n],
            b: vec![0.0; n],
            c: vec![0.0; n],
            d: vec![rho; n * layers],
            attach: vec![0; n],
            h: vec![0.0; n],
            top: vec![0; n],
            rim: vec![0; n],
            h_max: 1.0,
            rate: vec![0.0; n],
            curv: vec![0.0; n],
            curv_scale: 1.0,
            dx_um: 10.0,
            dz: Vec::new(),
            zc: Vec::new(),
            layers: Vec::new(),
            occ: Vec::new(),
            hmean: Vec::new(),
            edges: Vec::new(),
            bas_narrow: Vec::new(),
            bas_wide: Vec::new(),
            far_field: true,
            q_far: 0.0,
            time: 0.0,
            sweeps: 0,
            relax_budget: Self::RELAX_MAX_SWEEPS,
            rows_per_chunk: (w / 64).max(1),
            rho,
            seed,
            step: 0,
            crystal_cells: 0,
            crystal_radius: 0,
        };
        auto.set_layers(dz, true, true);
        auto.calibrate_curvature();
        let centre = auto.index(0, 0);
        auto.a[centre] = 1;
        auto.c[centre] = 1.0;
        auto.h[centre] = 1.0;
        auto.top[centre] = 1;
        auto.d[centre] = 0.0;
        auto.crystal_cells = 1;
        auto
    }

    /// Layer thicknesses (cells) of the field along the c-axis for a
    /// lattice of the given radius: two layers one cell thick next to the
    /// plane, so that a two-cell-thick plate edge is resolved, then doubling
    /// up to [`Automaton::DZ_MAX`] until the stack is about as tall as the
    /// lattice is wide, so the far boundary is roughly equidistant in every
    /// direction.
    fn layers_for(radius: usize) -> Vec<f32> {
        let mut dz = vec![1.0f32, 1.0];
        let (mut z, mut next) = (2.0f32, 2.0f32);
        while z < 0.7 * radius as f32 {
            dz.push(next);
            z += next;
            next = (next * 2.0).min(Self::DZ_MAX);
        }
        dz
    }

    /// Thickest field layer, in cells: layers coarsened in the plane are
    /// cheap, and a tall column's basal face must not sit in a layer much
    /// thicker than the column is wide.
    const DZ_MAX: f32 = 32.0;

    /// Install the layer stack and its finite-volume conductances (unit
    /// diffusivity, lengths in cells).  Layers thicker than four cells use a
    /// hex sublattice of spacing dz/4, so consecutive layers differ by at
    /// most a factor of two in spacing and the field within four cells of
    /// the plane is fully resolved.  An in-plane link carries flux
    /// through a face s/√3 wide and dz tall over a distance s; a vertical
    /// link through the cell's area over the distance between layer
    /// centres.  The top boundary is held at the far field (`top_far`) or
    /// reflects (used to reduce the field to plain 2-D in tests).
    fn set_layers(&mut self, dz: Vec<f32>, top_far: bool, coarsen: bool) {
        let count = dz.len();
        let big_r = self.radius;
        let mut layers = Vec::with_capacity(count);
        let mut zc = Vec::with_capacity(count);
        let (mut z, mut off) = (0.0f32, 0usize);
        for j in 0..count {
            let t = dz[j];
            let s = if coarsen { ((t / 4.0) as i32).max(1) } else { 1 };
            let radius = if s == 1 { big_r } else { (big_r + s - 1) / s + 1 };
            let w = (2 * radius + 1) as usize;
            let n = w * w;
            let above = if j + 1 < count {
                dz[j + 1]
            } else if top_far {
                0.0
            } else {
                f32::INFINITY
            };
            let g_up = SQRT3 * (s * s) as f32 / (t + above);
            layers.push(Layer {
                s,
                radius,
                w,
                off,
                n,
                dz: t,
                zc: z + 0.5 * t,
                g_h: t / SQRT3,
                g_up,
            });
            zc.push(z + 0.5 * t);
            z += t;
            off += n;
        }
        self.d = vec![self.rho; off];
        self.occ = layers
            .iter()
            .map(|l| vec![0.0; if l.s > 1 { l.n } else { 0 }])
            .collect();
        self.hmean = self.occ.clone();
        self.edges = self.occ.clone();
        self.bas_narrow = self.occ.clone();
        self.bas_wide = self.occ.clone();
        self.dz = dz;
        self.zc = zc;
        self.layers = layers;
    }

    /// Field value seen by the fine pixel (q, r) at layer `j`: its own
    /// cell on fine layers; on coarse layers the nearest coarse cell, or,
    /// when that one is ice, the mean of the vapour cells around it (the
    /// boundary cells whose uptake the pixel shares).
    #[inline]
    fn value_at(&self, j: usize, q: i32, r: i32) -> f32 {
        Self::value_in(&self.layers, &self.d, &self.occ, j, q, r)
    }

    fn value_in(layers: &[Layer], d: &[f32], occ: &[Vec<f32>], j: usize, q: i32, r: i32) -> f32 {
        let l = &layers[j];
        let (cq, cr) = l.nearest(q, r);
        let idx = l.index(cq, cr);
        if l.s == 1 || occ[j][idx - l.off] < Automaton::COARSE_ICE {
            return d[idx];
        }
        let (mut sum, mut count) = (0.0f32, 0u32);
        for (dq, dr) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)] {
            let (nq, nr) = (cq + dq, cr + dr);
            if hex_distance(nq, nr) <= l.radius {
                let ni = l.index(nq, nr);
                if occ[j][ni - l.off] < Automaton::COARSE_ICE {
                    sum += d[ni];
                    count += 1;
                }
            }
        }
        if count > 0 { sum / count as f32 } else { d[idx] }
    }

    /// Restrict the height map to the coarse layers: per coarse cell, the
    /// fraction of its fine pixels that are ice at that layer, their mean
    /// height, the prism faces inside it (fine vapour pixels next to ice at
    /// that layer — the cell's sub-grid sink), and the basal face area of
    /// the pixels whose ice ends there (on narrow terraces next to the rim,
    /// and on wide ones).
    fn refresh_coarse(&mut self) {
        let w = self.w;
        let big_r = self.radius;
        let offs = Self::neighbour_offsets(w);
        for j in 0..self.layers.len() {
            let l = self.layers[j].clone();
            if l.s == 1 {
                continue;
            }
            let mut ice = vec![0.0f32; l.n];
            let mut hsum = vec![0.0f32; l.n];
            let mut edges = vec![0.0f32; l.n];
            let mut narrow = vec![0.0f32; l.n];
            let mut wide = vec![0.0f32; l.n];
            let mut cover = vec![0.0f32; l.n];
            for i in 0..self.a.len() {
                let (q, r) = ((i % w) as i32 - big_r, (i / w) as i32 - big_r);
                if hex_distance(q, r) >= big_r {
                    continue;
                }
                let (cq, cr) = l.nearest(q, r);
                let c = l.index(cq, cr) - l.off;
                cover[c] += 1.0;
                let top = if self.a[i] != 0 { self.top[i] as usize } else { 0 };
                if top > j {
                    ice[c] += 1.0;
                    hsum[c] += self.h[i];
                    continue;
                }
                if self.a[i] != 0 && top == j {
                    if self.rim[i] <= 1 {
                        narrow[c] += Self::CELL_AREA;
                    } else {
                        wide[c] += Self::CELL_AREA;
                    }
                }
                if offs.iter().any(|&off| {
                    let nb = (i as isize + off) as usize;
                    self.a[nb] != 0 && self.top[nb] as usize > j
                }) {
                    edges[c] += 1.0;
                }
            }
            for c in 0..l.n {
                self.occ[j][c] = if cover[c] > 0.0 { ice[c] / cover[c] } else { 0.0 };
                self.hmean[j][c] = if ice[c] > 0.0 { hsum[c] / ice[c] } else { 0.0 };
                self.edges[j][c] = edges[c];
                self.bas_narrow[j][c] = narrow[c];
                self.bas_wide[j][c] = wide[c];
            }
        }
    }

    #[inline]
    fn index(&self, q: i32, r: i32) -> usize {
        (r + self.radius) as usize * self.w + (q + self.radius) as usize
    }

    /// Offsets of the six neighbours in the flat array.
    #[inline]
    fn neighbour_offsets(w: usize) -> [isize; 6] {
        let w = w as isize;
        [1, -1, w, -w, 1 - w, w - 1]
    }

    /// Set the physical cell size (µm).
    pub fn with_cell_size(mut self, dx_um: f32) -> Automaton {
        self.dx_um = dx_um;
        self.calibrate_curvature();
        self
    }

    /// Thickness of the seed crystal (a frozen droplet), µm.
    pub const SEED_UM: f32 = 10.0;

    /// Seed a frozen droplet: a hexagonal disc [`Automaton::SEED_UM`]
    /// across and thick (a single cell has only tip sites, which nucleate
    /// too rarely in the kinetics model).
    pub fn seed_hexagon(&mut self) {
        let h0 = (Self::SEED_UM / self.dx_um).max(1.0);
        let rs = ((0.5 * Self::SEED_UM / self.dx_um).round() as i32).max(1);
        for dr in -rs..=rs {
            for dq in -rs..=rs {
                if hex_distance(dq, dr) > rs {
                    continue;
                }
                let i = self.index(dq, dr);
                if self.a[i] == 0 {
                    self.a[i] = 1;
                    self.c[i] = 1.0;
                    self.d[i] = 0.0;
                    self.crystal_cells += 1;
                }
                self.h[i] = h0;
                self.top[i] = self.layers_of(h0);
            }
        }
        self.h_max = self.h_max.max(h0);
        self.crystal_radius = self.crystal_radius.max(rs);
    }

    /// Number of field layers a cell of height `h` (cells) fills: those
    /// whose centre lies below its half-height, and always the first.
    #[inline]
    fn layers_of(&self, h: f32) -> u8 {
        let half = 0.5 * h;
        1 + self.zc[1..].iter().take_while(|&&z| z < half).count() as u8
    }

    /// Recompute the layer occupancy of every crystal cell from its height.
    fn refresh_top(&mut self) {
        let zc = &self.zc;
        let a = &self.a;
        let h = &self.h;
        self.top.par_iter_mut().enumerate().for_each(|(i, t)| {
            *t = if a[i] == 0 {
                0
            } else {
                let half = 0.5 * h[i];
                1 + zc[1..].iter().take_while(|&&z| z < half).count() as u8
            };
        });
    }

    /// Number of water molecules in the crystal: the ice volume (hexagonal
    /// cell area × c-axis height) at the density of ice.
    pub fn molecules(&self) -> f64 {
        let height_sum: f64 = self
            .h
            .iter()
            .zip(&self.a)
            .filter(|(_, a)| **a != 0)
            .map(|(h, _)| *h as f64)
            .sum();
        molecules_from_height_sum(height_sum, self.dx_um as f64)
    }

    pub fn crystal_cells(&self) -> usize {
        self.crystal_cells
    }

    pub fn crystal_radius(&self) -> i32 {
        self.crystal_radius
    }

    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Relaxation sweeps performed so far.
    pub fn sweep_count(&self) -> u64 {
        self.sweeps
    }

    /// Advance one growth step of the attachment-kinetics model and return
    /// the simulated time it covered, in seconds.
    ///
    /// Following Libbrecht's cellular-automaton scheme (*Snow Crystals*,
    /// §5.2–5.4): the supersaturation field is relaxed to its quasi-static
    /// solution with the mixed boundary condition `X₀ ∂σ/∂n = α σ_surf` at
    /// every boundary pixel; each boundary pixel then fills at
    /// `v = α v_kin (σ_surf − d₀κ)` for a time step chosen so that the
    /// fastest pixel fills in a couple of steps, and full pixels become ice.
    /// Basal faces grow the same way from the vapour above them.
    pub fn step(&mut self, k: &Kinetics, nucleation: Nucleation, dt_max: f32) -> f32 {
        self.set_ambient(k.sigma_inf);
        self.refresh_curvature();
        self.refresh_coarse();
        self.update_far_field(k);
        self.relax_field(k);
        let dt = self.fill_and_attach(k, nucleation, dt_max);
        // The edge-sharpening instability: where the prism SDAK dip is deep,
        // an edge that grows fast outruns the basal thickening behind it and
        // stays thin — the thinner, the faster it grows (a lower barrier and
        // a smaller sink), which is the feedback that sharpens dendrite
        // tips more than their flanks.  Slow growth keeps thick edges.
        let edge_thinning = 1.0 - (-k.w0_prism / 3.0).exp();
        self.apply_attachment(edge_thinning);
        self.grow_basal(k, dt, nucleation);
        self.time += dt as f64;
        self.step += 1;
        dt
    }

    /// Row spacing of the hex lattice in cells (√3/2): the distance a
    /// filled pixel advances the front.
    const ROW_SPACING: f32 = 0.866_025_4;
    /// Distance from a boundary pixel's centre to the ice interface, in
    /// cells: half the row spacing.
    const HALF_ROW: f32 = 0.433;
    /// Most relaxation sweeps per growth step.
    const RELAX_MAX_SWEEPS: usize = 100;
    /// Relative residual below which the field counts as converged.
    const RELAX_TOLERANCE: f32 = 1e-3;
    /// Under-relaxation of the boundary pixels, whose α(σ) feedback is steep
    /// (plain Gauss–Seidel on them fails to settle within the budget).
    const RELAX_DAMPING: f32 = 0.8;

    /// Attachment coefficient of a prism-face boundary pixel with `n_ice`
    /// crystal neighbours of mean height `h_nb` (cells), where the
    /// interface has in-plane curvature `curv` (1/µm), at supersaturation
    /// `sigma`, for cells `dx_um` µm wide.
    #[inline]
    fn prism_alpha(k: &Kinetics, dx_um: f32, n_ice: u32, h_nb: f32, curv: f32, sigma: f32) -> f32 {
        // Kink sites (concave pixel configurations) are the sides of steps
        // and the inside corners between facets: attachment there needs no
        // nucleation (§4.2).
        if n_ice >= 3 {
            return 1.0;
        }
        // The terrace a molecule lands on: its height is the edge height,
        // and around a convex tip it is also only about as wide as the
        // tip's radius of curvature; at a corner site (one ice neighbour —
        // the line where two faces meet) only a fraction of a micrometre.
        // The narrower dimension sets the SDAK barrier — this is the
        // edge-sharpening instability acting on tips and corners, which is
        // what makes corners sprout and keeps dendrite tips sharp.
        let mut width = h_nb * dx_um;
        if curv > 0.0 {
            width = width.min(1.0 / curv);
        }
        if n_ice == 1 {
            width = width.min(Self::CORNER_UM);
        }
        let sigma0 = sdak_barrier(k.sigma0_prism as f64, width as f64, k.w0_prism as f64);
        attachment_coefficient(k.a_prism as f64, sigma0, sigma as f64) as f32
    }

    /// Ice fraction above which a coarse cell counts as interior ice.
    const COARSE_ICE: f32 = 0.999;

    /// Fraction of a layer's height (bottom `zb`, thickness `dz`, cells)
    /// that an edge of total height `h` (cells, centred on the plane)
    /// covers, so that a thin edge is the sink its actual height makes it
    /// whatever the cell size.
    #[inline]
    fn edge_cover(h: f32, zb: f32, dz: f32) -> f32 {
        ((0.5 * h - zb) / dz).clamp(0.0, 1.0)
    }
    /// Least curvature of a tip site, in inverse cells (a tip one cell
    /// wide has a radius of about a cell).
    const TIP_CELLS_INV: f32 = 0.5;
    /// Terrace width (µm) at a corner site, where two faces meet.
    const CORNER_UM: f32 = 0.5;

    /// Radius (µm) of the disc over which the interface curvature is
    /// measured: about the radius of a dendrite tip, so that a tip is
    /// resolved as one while a single-cell bump is not mistaken for one.
    pub const CURVATURE_UM: f32 = 2.0;

    /// Measure the in-plane curvature at every boundary pixel (see
    /// [`disc_curvature`]).
    fn refresh_curvature(&mut self) {
        let rc = self.curvature_cells();
        disc_curvature(&self.a, self.w, self.radius, rc, self.curv_scale, &mut self.curv);
    }

    /// Radius of the curvature disc, in cells (at least three, below which
    /// the estimate is too coarse).
    fn curvature_cells(&self) -> i32 {
        ((Self::CURVATURE_UM / self.dx_um).round() as i32).max(3)
    }

    /// Calibrate the curvature estimate for this cell size against a disc
    /// of known radius on the lattice: the hexagonal "disc" and the
    /// staircase interface give the continuum formula a gain that depends
    /// on the disc's radius in cells.
    fn calibrate_curvature(&mut self) {
        let rc = self.curvature_cells();
        let rho = 12 * rc;
        // Hex distance exceeds Euclidean distance by up to 2/√3.
        let big_r = (rho as f32 * 1.16) as i32 + 2 * rc + 2;
        let w = (2 * big_r + 1) as usize;
        let mut a = vec![0u8; w * w];
        for i in 0..a.len() {
            let (q, r) = ((i % w) as i32 - big_r, (i / w) as i32 - big_r);
            let (x, y) = (q as f32 + 0.5 * r as f32, r as f32 * Self::CELL_AREA);
            a[i] = ((x * x + y * y).sqrt() <= rho as f32) as u8;
        }
        let mut curv = vec![0.0f32; w * w];
        disc_curvature(&a, w, big_r, rc, 1.0, &mut curv);
        let (sum, count) = (0..curv.len())
            .filter(|&i| a[i] == 0 && curv_is_boundary(&a, w, i))
            .fold((0.0f64, 0u32), |(s, n), i| (s + curv[i] as f64, n + 1));
        let mean = sum / count.max(1) as f64;
        let expected = 1.0 / (rho as f64 * self.dx_um as f64);
        self.curv_scale = (expected / mean) as f32;
    }

    /// Four one-cell layers, reflecting top and plain σ_∞ on the ring
    /// (tests: with a crystal that spans every layer this reduces the field
    /// to plain 2-D with the far value on the ring).
    #[cfg(test)]
    fn with_reflecting_top(mut self) -> Automaton {
        self.set_layers(vec![1.0; 4], false, false);
        self.far_field = false;
        self
    }

    /// Per-layer slices and offsets used by the cell update.
    fn layer_contexts(&self) -> Vec<LayerCtx<'_>> {
        self.layers
            .iter()
            .enumerate()
            .map(|(j, l)| {
                let w = l.w as isize;
                LayerCtx {
                    l,
                    occ: &self.occ[j],
                    hmean: &self.hmean[j],
                    edges: &self.edges[j],
                    bas_narrow: &self.bas_narrow[j],
                    bas_wide: &self.bas_wide[j],
                    offs: [1, -1, w, -w, 1 - w, w - 1],
                }
            })
            .collect()
    }

    /// Total vapour uptake of the crystal, Q, from the current field: the
    /// sum over every ice face of its boundary-condition flux (unit
    /// diffusivity, lengths in cells), both halves of the mirror.
    fn total_uptake(&self, k: &Kinetics) -> f32 {
        let ptr = SyncPtr(self.d.as_ptr() as *mut f32);
        let ctx = self.layer_contexts();
        let ctx = &ctx;
        let half: f32 = (0..self.layers.len())
            .into_par_iter()
            .map(|j| {
                let l = &self.layers[j];
                let mut sum = 0.0f32;
                for r in -l.radius..=l.radius {
                    let Some((q_lo, q_hi)) = inner_range(l.radius, r) else {
                        continue;
                    };
                    for q in q_lo..=q_hi {
                        // SAFETY: read-only; no concurrent writer.
                        if let Some(t) = unsafe { self.cell_terms(k, ctx, j, q, r, ptr) } {
                            sum += t.uptake * t.sigma;
                        }
                    }
                }
                sum
            })
            .sum();
        2.0 * half
    }

    /// Whether the cell at layer-local index `c` of layer `j` is ice
    /// (`fine`: the layer is at the plane's spacing).
    #[inline(always)]
    unsafe fn ice_at(&self, ctx: &LayerCtx, fine: bool, j: usize, c: usize) -> bool {
        unsafe {
            if fine {
                *self.a.get_unchecked(c) != 0 && *self.top.get_unchecked(c) as usize > j
            } else {
                *ctx.occ.get_unchecked(c) >= Self::COARSE_ICE
            }
        }
    }

    /// Flux balance of one vapour cell: the conductance-weighted sum of its
    /// neighbours (`sum`), the total conductance (`g`), the uptake
    /// coefficient of the ice faces it feeds (`uptake`, zero for a cell
    /// with none), and its current value.
    ///
    /// # Safety
    /// `ptr` must point at the field; concurrent writers may only touch
    /// cells of other colours/parities than (j, q, r) and its neighbours;
    /// (q, r) must lie inside layer `j`'s lattice with a ring of neighbours.
    #[inline(always)]
    unsafe fn cell_terms(
        &self,
        k: &Kinetics,
        ctx: &[LayerCtx],
        j: usize,
        q: i32,
        r: i32,
        ptr: SyncPtr<f32>,
    ) -> Option<CellTerms> {
        unsafe {
            let cx = ctx.get_unchecked(j);
            let l = cx.l;
            let d = ptr.get();
            let dx_um = self.dx_um;
            let dx_over_x0 = dx_um / k.x0;
            let (s, off, g_h, g_up) = (l.s, l.off, l.g_h, l.g_up);
            let fine = s == 1;
            let c = (r + l.radius) as usize * l.w + (q + l.radius) as usize;
            let idx = off + c;
            if self.ice_at(cx, fine, j, c) {
                return None;
            }
            let (mut sum, mut g) = (0.0f32, 0.0f32);
            let (mut n_ice, mut h_sum, mut cover) = (0u32, 0.0f32, 0.0f32);
            // In-plane neighbours.  On fine layers a crystal neighbour whose
            // edge reaches into this layer presents a face covering that
            // fraction of the layer's height (the cell itself is ice only
            // when the edge passes its centre).  On coarse layers a
            // wholly-ice cell is interior to the crystal: the link is cut,
            // and the ice faces are the sub-grid ones inside each cell
            // (`edges`).
            let dl = d.add(off);
            let mut links = 0u32;
            // The faces this cell sees start above its own ice, if any.
            let zb = (l.zc - 0.5 * l.dz).max(if fine && *self.a.get_unchecked(c) != 0 {
                0.5 * *self.h.get_unchecked(c)
            } else {
                0.0
            });
            let dz_seen = (l.zc + 0.5 * l.dz - zb).max(1e-6);
            for o in cx.offs {
                let nc = (c as isize + o) as usize;
                if fine && *self.a.get_unchecked(nc) != 0 {
                    let hn = *self.h.get_unchecked(nc);
                    let f = Self::edge_cover(hn, zb, dz_seen);
                    if f > 0.0 {
                        n_ice += 1;
                        h_sum += hn;
                        cover += f;
                    }
                    if *self.top.get_unchecked(nc) as usize > j {
                        continue;
                    }
                } else if self.ice_at(cx, fine, j, nc) {
                    continue;
                }
                sum += *dl.add(nc);
                links += 1;
            }
            sum *= g_h;
            g += g_h * links as f32;
            // Above: the next layer (same spacing, or twice as coarse —
            // then the two cells this one straddles, or the one it
            // centres), or the far boundary above the last layer.
            if j + 1 < ctx.len() {
                let la = ctx.get_unchecked(j + 1).l;
                if la.s == s {
                    sum += g_up * *d.add(la.off + c);
                } else {
                    let (a1, a2) = match (q.rem_euclid(2), r.rem_euclid(2)) {
                        (0, 0) => ((q / 2, r / 2), (q / 2, r / 2)),
                        (1, 0) => (((q - 1) / 2, r / 2), ((q + 1) / 2, r / 2)),
                        (0, 1) => ((q / 2, (r - 1) / 2), (q / 2, (r + 1) / 2)),
                        _ => (((q + 1) / 2, (r - 1) / 2), ((q - 1) / 2, (r + 1) / 2)),
                    };
                    // Cells past the coarser lattice's edge are at the far
                    // field.
                    let read = |cq: i32, cr: i32| -> f32 {
                        if hex_distance(cq, cr) <= la.radius {
                            *d.add(la.index(cq, cr))
                        } else {
                            let (fq, fr) = (cq * la.s, cr * la.s);
                            let (x, y) = (fq as f32 + 0.5 * fr as f32, fr as f32 * Self::CELL_AREA);
                            Self::far_value(self.rho, self.q_far, x, y, la.zc)
                        }
                    };
                    sum += g_up * 0.5 * (read(a1.0, a1.1) + read(a2.0, a2.1));
                }
                g += g_up;
            } else if g_up > 0.0 {
                let (fq, fr) = (q * s, r * s);
                let (x, y) = (fq as f32 + 0.5 * fr as f32, fr as f32 * Self::CELL_AREA);
                sum += g_up * Self::far_value(self.rho, self.q_far, x, y, l.zc + 0.5 * l.dz);
                g += g_up;
            }
            // Below: the previous layer (same spacing, or half as coarse —
            // then the fine cell this one centres and half of each of its
            // six neighbours), except where ice ends under this cell and it
            // feeds a basal face; layer 0 sits on the mirror plane.
            if j > 0 {
                let cb = ctx.get_unchecked(j - 1);
                let lb = cb.l;
                if lb.s == s {
                    if !self.ice_at(cb, fine, j - 1, c) {
                        sum += lb.g_up * *d.add(lb.off + c);
                        g += lb.g_up;
                    }
                } else {
                    let (cq, cr) = (2 * q, 2 * r);
                    for (n, (dq, dr)) in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
                        .into_iter()
                        .enumerate()
                    {
                        let (nq, nr) = (cq + dq, cr + dr);
                        let weight = if n == 0 { 1.0 } else { 0.5 };
                        // Finer cells past their lattice's edge are at the
                        // far field.
                        if hex_distance(nq, nr) <= lb.radius {
                            let nc = (nr + lb.radius) as usize * lb.w + (nq + lb.radius) as usize;
                            if !self.ice_at(cb, lb.s == 1, j - 1, nc) {
                                sum += weight * lb.g_up * *d.add(lb.off + nc);
                                g += weight * lb.g_up;
                            }
                        } else {
                            let (fq, fr) = (nq * lb.s, nr * lb.s);
                            let (x, y) = (fq as f32 + 0.5 * fr as f32, fr as f32 * Self::CELL_AREA);
                            sum += weight * lb.g_up * Self::far_value(self.rho, self.q_far, x, y, lb.zc);
                            g += weight * lb.g_up;
                        }
                    }
                }
            }
            let sigma = *d.add(idx);
            // Ice faces: the mixed boundary condition (eqs. 5.6/5.24) as a
            // flux balance — each face takes up A·κ·σ_s, κ = α·Δx/X₀, with
            // the surface value extrapolated from the cell centre to the
            // face, σ_s = σ_b/(1 + κ·δ), δ the distance in cells.  A prism
            // edge presents one face s wide per boundary cell (the pixel
            // advances the front by a row when it fills), half a row from
            // the centre.
            let mut uptake = 0.0f32;
            if fine {
                if n_ice > 0 {
                    let h_nb = h_sum / n_ice as f32;
                    let alpha = Self::prism_alpha(k, dx_um, n_ice, h_nb, *self.curv.get_unchecked(c), sigma);
                    let kappa = alpha * dx_over_x0;
                    uptake += (cover / n_ice as f32) * dz_seen * kappa / (1.0 + Self::HALF_ROW * kappa);
                }
            } else {
                let faces = *cx.edges.get_unchecked(c);
                if faces > 0.0 {
                    let alpha = Self::prism_alpha(k, dx_um, 2, *cx.hmean.get_unchecked(c), 0.0, sigma);
                    let kappa = alpha * dx_over_x0;
                    uptake += faces * l.dz * kappa / (1.0 + Self::HALF_ROW * kappa);
                }
            }
            if j > 0 {
                if fine {
                    if *self.a.get_unchecked(c) != 0 && *self.top.get_unchecked(c) as usize == j {
                        let alpha = Self::basal_alpha(k, dx_um, *self.rim.get_unchecked(c), sigma);
                        let kappa = alpha * dx_over_x0;
                        uptake += Self::CELL_AREA * kappa / (1.0 + 0.5 * l.dz * kappa);
                    }
                } else {
                    for (area, rim) in [
                        (*cx.bas_narrow.get_unchecked(c), 1u8),
                        (*cx.bas_wide.get_unchecked(c), u8::MAX),
                    ] {
                        if area > 0.0 {
                            let alpha = Self::basal_alpha(k, dx_um, rim, sigma);
                            let kappa = alpha * dx_over_x0;
                            uptake += area * kappa / (1.0 + 0.5 * l.dz * kappa);
                        }
                    }
                }
            }
            Some(CellTerms {
                sum,
                g,
                uptake,
                sigma,
                idx,
            })
        }
    }

    /// Far-field value at a point (x, y, z) in cells from the crystal's
    /// centre.
    #[inline]
    fn far_value(sigma_inf: f32, q_far: f32, x: f32, y: f32, z: f32) -> f32 {
        let r = (x * x + y * y + z * z).sqrt().max(1.0);
        (sigma_inf - q_far / r).max(0.0)
    }

    /// Relax the supersaturation field to its quasi-static solution
    /// (Laplace's equation with the mixed boundary condition), warm-started
    /// from the previous step.
    fn relax_field(&mut self, k: &Kinetics) {
        self.relax_field_with(k, self.relax_budget, Self::RELAX_TOLERANCE);
    }

    /// Converge the field from scratch (many sweeps, tight tolerance); used
    /// when a run starts or the ambient conditions jump.
    pub fn converge_field(&mut self, k: &Kinetics) -> usize {
        self.refresh_curvature();
        self.refresh_coarse();
        let mut sweeps = 0;
        for _ in 0..12 {
            sweeps += self.relax_field_with(k, 5_000, Self::RELAX_TOLERANCE * 0.1);
            if !self.far_field {
                break;
            }
            // The boundary carries the far field of the crystal's uptake;
            // relax and update it (damped) until the two agree.
            let q_new = self.total_uptake(k) / (4.0 * std::f32::consts::PI);
            let converged = (q_new - self.q_far).abs() <= 0.01 * q_new.abs() + 1e-6;
            self.q_far = 0.5 * (self.q_far + q_new);
            if converged {
                break;
            }
        }
        sweeps
    }

    /// Set the far-field boundary from the crystal's uptake in the current
    /// (converged) field, for the next relaxation.
    fn update_far_field(&mut self, k: &Kinetics) {
        if self.far_field {
            self.q_far = self.total_uptake(k) / (4.0 * std::f32::consts::PI);
        }
    }

    /// Over-relaxation factor for vapour pixels: the SOR optimum for a
    /// lattice `n` cells wide is 2/(1 + π/n), which turns the O(n²) sweeps
    /// of Gauss–Seidel into O(n); values too close to 2 oscillate against
    /// the damped nonlinear boundary pixels, so it is capped at 1.9.
    fn sor_omega(&self) -> f32 {
        let n = self.w as f32;
        (2.0 / (1.0 + std::f32::consts::PI / n)).min(1.9)
    }

    fn relax_field_with(&mut self, k: &Kinetics, max_sweeps: usize, rel_tolerance: f32) -> usize {
        let floor = 1e-4 * k.sigma_inf.max(1e-6);
        let omega = self.sor_omega();
        let damping = Self::RELAX_DAMPING;
        let mut used = 0;

        let (sigma_inf, q_far) = (self.rho, self.q_far);

        // The outer ring of every layer carries the far field; ice cells
        // carry 0.
        for j in 0..self.layers.len() {
            let l = self.layers[j].clone();
            let (a, top, occ) = (&self.a, &self.top, &self.occ[j]);
            let base = self.w;
            let big_r = self.radius;
            self.d[l.off..l.off + l.n]
                .par_chunks_mut(l.w)
                .enumerate()
                .for_each(|(ri, row)| {
                    let r = ri as i32 - l.radius;
                    let inner = inner_range(l.radius, r);
                    for (qi, v) in row.iter_mut().enumerate() {
                        let q = qi as i32 - l.radius;
                        let ice = if l.s == 1 {
                            let i = ri * base + qi;
                            a[i] != 0 && top[i] as usize > j
                        } else {
                            occ[ri * l.w + qi] >= Self::COARSE_ICE
                        };
                        if ice {
                            *v = 0.0;
                        } else if !inner.is_some_and(|(lo, hi)| q >= lo && q <= hi) {
                            let (fq, fr) = (q * l.s, r * l.s);
                            let _ = big_r;
                            let (x, y) = (fq as f32 + 0.5 * fr as f32, fr as f32 * Self::CELL_AREA);
                            *v = Self::far_value(sigma_inf, q_far, x, y, l.zc);
                        }
                    }
                });
        }

        // Gauss–Seidel with over-relaxation in six passes: layers of one
        // parity at a time (layers only couple to their neighbours above and
        // below), and within them a 3-colouring of the hex lattice, colour
        // (q − r) mod 3, which differs between every pair of in-plane
        // neighbours.  Each pass reads only cells of other passes and can
        // run in parallel, in place, deterministically.
        let rows_per_chunk = self.rows_per_chunk;
        let ptr = SyncPtr(self.d.as_mut_ptr());
        let ctx = self.layer_contexts();
        let ctx = &ctx;
        let mut tasks: Vec<(usize, i32)> = Vec::new(); // (layer, first row)
        for (j, l) in self.layers.iter().enumerate() {
            let mut r = -l.radius;
            while r <= l.radius {
                tasks.push((j, r));
                r += rows_per_chunk as i32;
            }
        }
        let mut deltas = vec![0.0f32; tasks.len()];
        for sweep in 0..max_sweeps {
            let mut max_delta = 0.0f32;
            for parity in 0..2usize {
                for colour in 0..3i32 {
                    tasks
                        .par_iter()
                        .map(|&(j, r0)| {
                            if j % 2 != parity {
                                return 0.0f32;
                            }
                            let l = &self.layers[j];
                            let mut worst = 0.0f32;
                            // Fast path: a layer at the plane's spacing
                            // between two more of the same (the bulk of the
                            // cells), with everything inlined.
                            let simple = l.s == 1
                                && j + 1 < self.layers.len()
                                && self.layers[j + 1].s == 1
                                && (j == 0 || self.layers[j - 1].s == 1);
                            if simple {
                                let (a, top, h, curv, rim) =
                                    (&self.a, &self.top, &self.h, &self.curv, &self.rim);
                                let w = l.w;
                                let big_r = l.radius;
                                let offs = Self::neighbour_offsets(w);
                                let (g_h, g_up, dzj) = (l.g_h, l.g_up, l.dz);
                                let zb = l.zc - 0.5 * dzj;
                                let g_dn = if j > 0 { self.layers[j - 1].g_up } else { 0.0 };
                                let (base, base_up) = (l.off, self.layers[j + 1].off);
                                let base_dn = if j > 0 { self.layers[j - 1].off } else { 0 };
                                let dx_over_x0 = self.dx_um / k.x0;
                                let d = ptr.get();
                                for r in r0..(r0 + rows_per_chunk as i32).min(big_r + 1) {
                                    let Some((q_lo, q_hi)) = inner_range(big_r, r) else {
                                        continue;
                                    };
                                    let mut q = q_lo;
                                    while (q - r).rem_euclid(3) != colour {
                                        q += 1;
                                    }
                                    while q <= q_hi {
                                        let i = (r + big_r) as usize * w + (q + big_r) as usize;
                                        // SAFETY: as below; indices are inside
                                        // the arrays.
                                        unsafe {
                                            let ice_here = *a.get_unchecked(i) != 0;
                                            let top_i = *top.get_unchecked(i) as usize;
                                            if ice_here && top_i > j {
                                                q += 3;
                                                continue;
                                            }
                                            let (mut n_ice, mut h_sum, mut cover, mut sum) =
                                                (0u32, 0.0f32, 0.0f32, 0.0f32);
                                            let mut links = 0u32;
                                            // The faces this cell sees start
                                            // above its own ice, if any.
                                            let zb_seen = if ice_here {
                                                zb.max(0.5 * *h.get_unchecked(i))
                                            } else {
                                                zb
                                            };
                                            let dz_seen = (zb + dzj - zb_seen).max(1e-6);
                                            for off in offs {
                                                let nb = (i as isize + off) as usize;
                                                if *a.get_unchecked(nb) != 0 {
                                                    let hn = *h.get_unchecked(nb);
                                                    let f = Self::edge_cover(hn, zb_seen, dz_seen);
                                                    if f > 0.0 {
                                                        n_ice += 1;
                                                        h_sum += hn;
                                                        cover += f;
                                                    }
                                                    if *top.get_unchecked(nb) as usize > j {
                                                        continue;
                                                    }
                                                }
                                                sum += *d.add(base + nb);
                                                links += 1;
                                            }
                                            sum *= g_h;
                                            let mut g = g_h * links as f32;
                                            sum += g_up * *d.add(base_up + i);
                                            g += g_up;
                                            let basal = j > 0 && ice_here && top_i == j;
                                            if j > 0 && !basal {
                                                sum += g_dn * *d.add(base_dn + i);
                                                g += g_dn;
                                            }
                                            let old = *d.add(base + i);
                                            let new = if n_ice == 0 && !basal {
                                                old + (sum / g - old) * omega
                                            } else {
                                                if n_ice > 0 {
                                                    let alpha = Self::prism_alpha(
                                                        k,
                                                        self.dx_um,
                                                        n_ice,
                                                        h_sum / n_ice as f32,
                                                        *curv.get_unchecked(i),
                                                        old,
                                                    );
                                                    let kappa = alpha * dx_over_x0;
                                                    g += (cover / n_ice as f32) * dz_seen * kappa
                                                        / (1.0 + Self::HALF_ROW * kappa);
                                                }
                                                if basal {
                                                    let alpha = Self::basal_alpha(
                                                        k,
                                                        self.dx_um,
                                                        *rim.get_unchecked(i),
                                                        old,
                                                    );
                                                    let kappa = alpha * dx_over_x0;
                                                    g += Self::CELL_AREA * kappa / (1.0 + 0.5 * dzj * kappa);
                                                }
                                                old + (sum / g - old) * damping
                                            };
                                            *d.add(base + i) = new;
                                            worst = worst.max((new - old).abs() / (old.abs() + floor));
                                        }
                                        q += 3;
                                    }
                                }
                                return worst;
                            }
                            for r in r0..(r0 + rows_per_chunk as i32).min(l.radius + 1) {
                                let Some((q_lo, q_hi)) = inner_range(l.radius, r) else {
                                    continue;
                                };
                                let mut q = q_lo;
                                while (q - r).rem_euclid(3) != colour {
                                    q += 1;
                                }
                                while q <= q_hi {
                                    // SAFETY: this pass writes only cells of
                                    // the current layer parity and colour and
                                    // reads only their neighbours, which are
                                    // of other passes; (layer, rows) are
                                    // partitioned between threads.
                                    unsafe {
                                        if let Some(t) = self.cell_terms(k, ctx, j, q, r, ptr) {
                                            let old = t.sigma;
                                            let new = if t.uptake == 0.0 {
                                                old + (t.sum / t.g - old) * omega
                                            } else {
                                                old + (t.sum / (t.g + t.uptake) - old) * damping
                                            };
                                            *ptr.get().add(t.idx) = new;
                                            // Relative change: surface values
                                            // can be orders of magnitude
                                            // below σ_∞.
                                            worst = worst.max((new - old).abs() / (old.abs() + floor));
                                        }
                                    }
                                    q += 3;
                                }
                            }
                            worst
                        })
                        .collect_into_vec(&mut deltas);
                    max_delta = max_delta.max(deltas.iter().cloned().fold(0.0, f32::max));
                }
            }
            used = sweep + 1;
            // The change per sweep is a poor measure of convergence (it
            // depends on the update order and misses smooth error); once it
            // is small, check the actual residual of the equations, every
            // few sweeps.
            if max_delta < rel_tolerance && (sweep % 4 == 1 || sweep + 1 == max_sweeps) {
                let residual = self.max_residual(k, ctx, ptr, floor);
                if residual < rel_tolerance {
                    break;
                }
            }
        }
        self.sweeps += used as u64;
        used
    }

    /// Largest relative residual of the flux balance over all vapour cells,
    /// evaluated on the current field (independent of the sweep order).
    fn max_residual(&self, k: &Kinetics, ctx: &[LayerCtx], ptr: SyncPtr<f32>, floor: f32) -> f32 {
        (0..self.layers.len())
            .into_par_iter()
            .map(|j| {
                let l = &self.layers[j];
                let mut worst = 0.0f32;
                for r in -l.radius..=l.radius {
                    let Some((q_lo, q_hi)) = inner_range(l.radius, r) else {
                        continue;
                    };
                    for q in q_lo..=q_hi {
                        // SAFETY: read-only; no concurrent writer.
                        if let Some(t) = unsafe { self.cell_terms(k, ctx, j, q, r, ptr) } {
                            let target = t.sum / (t.g + t.uptake);
                            worst = worst.max((target - t.sigma).abs() / (t.sigma.abs() + floor));
                        }
                    }
                }
                worst
            })
            .reduce(|| 0.0f32, f32::max)
    }

    /// Attachment coefficient of a basal face `rim` cells in from the
    /// crystal's edge at supersaturation `sigma`: the SDAK barrier of a
    /// terrace that wide (narrow basal terraces next to the rim have a
    /// lower barrier, which makes the thin-walled hollow columns and needles
    /// near −5 °C).
    #[inline]
    fn basal_alpha(k: &Kinetics, dx_um: f32, rim: u8, sigma: f32) -> f32 {
        let width = ((rim as f32 + 0.5) * dx_um) as f64;
        let sigma0 = sdak_barrier(k.sigma0_basal as f64, width, k.w0_basal as f64);
        attachment_coefficient(k.a_basal as f64, sigma0, sigma as f64) as f32
    }

    /// Fill boundary pixels for one time step and mark the full ones.
    /// Returns the time step used, s.
    fn fill_and_attach(&mut self, k: &Kinetics, nucleation: Nucleation, dt_max: f32) -> f32 {
        let w = self.w;
        let big_r = self.radius;
        let offs = Self::neighbour_offsets(w);
        let dx = self.dx_um;
        let dx_over_x0 = dx / k.x0;
        let chunk = self.rows_per_chunk * w;
        let a = &self.a;
        let d = &self.d;
        let h = &self.h;
        let curv = &self.curv;
        let (dz, zc) = (&self.dz, &self.zc);
        let layers = &self.layers;
        let occ = &self.occ;
        let value_at = |j: usize, q: i32, r: i32| -> f32 { Self::value_in(layers, d, occ, j, q, r) };
        let rows_per_chunk = self.rows_per_chunk;

        // Growth rate of every boundary pixel (µm/s); `attach` holds its
        // kind meanwhile: 1 facet/tip, 2 kink.
        let v_max = self
            .rate
            .par_chunks_mut(chunk)
            .zip(self.attach.par_chunks_mut(chunk))
            .enumerate()
            .map(|(ci, (rate_c, att_c))| {
                rate_c.fill(0.0);
                att_c.fill(0);
                let mut fastest = 0.0f32;
                for (kr, rate_r) in rate_c.chunks_mut(w).enumerate() {
                    let ri = ci * rows_per_chunk + kr;
                    let r = ri as i32 - big_r;
                    let Some((q_lo, q_hi)) = inner_range(big_r, r) else {
                        continue;
                    };
                    let att_r = &mut att_c[kr * w..(kr + 1) * w];
                    for q in q_lo..=q_hi {
                        let qi = (q + big_r) as usize;
                        let i = ri * w + qi;
                        if a[i] != 0 {
                            continue;
                        }
                        let mut n_ice = 0u32;
                        let mut h_sum = 0.0f32;
                        for off in offs {
                            let j = (i as isize + off) as usize;
                            if a[j] != 0 {
                                n_ice += 1;
                                h_sum += h[j];
                            }
                        }
                        if n_ice == 0 {
                            continue;
                        }
                        // The pixel extends an edge as tall as its
                        // neighbours; its rate is the mean over the layers
                        // that edge spans, each with its own surface value
                        // (the pixel value extrapolated half a row inward,
                        // see relax_field), less the Gibbs–Thomson
                        // correction for the measured in-plane curvature.
                        // A tip site (one ice neighbour) is a feature at
                        // most a couple of cells across, sharper than the
                        // curvature disc can see: the capillary term feels
                        // at least the curvature of a tip one cell in
                        // radius.  This is what keeps facets from roughening
                        // at the cell scale.
                        let h_nb = h_sum / n_ice as f32;
                        let kappa_gt = if n_ice == 1 {
                            curv[i].max(Self::TIP_CELLS_INV / dx)
                        } else {
                            curv[i]
                        };
                        let (mut v_sum, mut z_sum) = (0.0f32, 0.0f32);
                        for (j, &dzj) in dz.iter().enumerate() {
                            let zb = zc[j] - 0.5 * dzj;
                            let mut n_ice_j = 0u32;
                            for off in offs {
                                let nb = (i as isize + off) as usize;
                                if a[nb] != 0 && Self::edge_cover(h[nb], zb, dzj) > 0.0 {
                                    n_ice_j += 1;
                                }
                            }
                            if n_ice_j == 0 {
                                break;
                            }
                            let weight = Self::edge_cover(h_nb, zb, dzj) * dzj;
                            if weight <= 0.0 {
                                break;
                            }
                            let sigma_b = value_at(j, q, r);
                            let alpha = Self::prism_alpha(k, dx, n_ice_j, h_nb, kappa_gt, sigma_b);
                            let kappa = alpha * dx_over_x0;
                            let sigma_s = sigma_b / (1.0 + Self::HALF_ROW * kappa);
                            let sigma = (sigma_s - k.d0 * kappa_gt).max(0.0);
                            v_sum += weight * alpha * k.v_kin * sigma;
                            z_sum += weight;
                        }
                        // Kink sites have α = 1 (§4.2); with the mixed
                        // boundary condition that makes them
                        // diffusion-limited, a few times faster than the
                        // facet sites beside them — not instantaneous.
                        let v = if z_sum > 0.0 { v_sum / z_sum } else { 0.0 };
                        rate_r[qi] = v;
                        att_r[qi] = if n_ice >= 3 { 2 } else { 1 };
                        fastest = fastest.max(v);
                    }
                }
                fastest
            })
            .reduce(|| 0.0f32, f32::max);

        // With shared random draws the crystal is meant to be exactly
        // symmetric, but the relaxation's update order is not, so the
        // partially converged field can differ slightly between symmetric
        // sites.  Average each surface site's rate over its twelve images.
        if nucleation.symmetric || nucleation.quantised() {
            self.symmetrise_rates();
        }
        if nucleation.quantised() {
            let attach = &self.attach;
            self.rate.par_iter_mut().enumerate().for_each(|(i, v)| {
                if attach[i] != 0 {
                    *v = nucleation.quantise(*v, v_max);
                }
            });
        }

        // The time step: the fastest site fills in two steps.
        let dt = if v_max > 0.0 {
            (dx * Self::ROW_SPACING / (2.0 * v_max)).min(dt_max)
        } else {
            dt_max
        };

        // Fill, and decide who joins.
        let step_seed = self.seed ^ self.step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let stochastic = nucleation.rate.is_finite() && !nucleation.quantised();
        for i in 0..self.attach.len() {
            let kind = self.attach[i];
            if kind == 0 {
                continue;
            }
            // A filled pixel advances the front by one row of the hex
            // lattice, √3/2 cells.
            let fill = self.rate[i] * dt / (dx * Self::ROW_SPACING);
            self.b[i] += fill;
            // Waiting time, as an ice-thickness proxy for the renderer.
            self.c[i] += 0.02 * dt;
            // The time step is sized so the fastest pixels land exactly on
            // 1.0; a tolerance keeps rounding from splitting symmetric cells.
            if self.b[i] < 1.0 - 1e-4 {
                self.attach[i] = 0;
                continue;
            }
            let joins = if kind == 2 || !stochastic {
                true
            } else {
                let q = (i % w) as i32 - big_r;
                let r = (i / w) as i32 - big_r;
                let (cq, cr) = if nucleation.symmetric {
                    canonical(q, r)
                } else {
                    (q, r)
                };
                let key = ((cq as u32 as u64) << 32) | (cr as u32 as u64);
                // Poisson nucleation over this step: P = 1 − exp(−λ·dt),
                // with λ·dt expressed through the fill gained in the step.
                let p = 1.0 - (-nucleation.rate * fill).exp();
                uniform(hash64(step_seed ^ key)) < p
            };
            self.attach[i] = joins as u8;
        }
        dt
    }

    /// Replace every boundary pixel's growth rate by the mean over its
    /// twelve lattice-symmetric images.
    fn symmetrise_rates(&mut self) {
        let mut rate = std::mem::take(&mut self.rate);
        let attach = &self.attach;
        self.symmetrise(&mut rate, |i| attach[i] != 0);
        self.rate = rate;
    }

    /// Kinetics model: every basal face grows with the CAK coefficient from
    /// the vapour in the field cell above it, through the same mixed
    /// boundary condition as the prism faces.  Both basal faces of the
    /// crystal grow, so the c-axis extent advances at twice the face rate.
    /// Where a taller neighbour rises above a cell, the exposed prism face
    /// of that step also grows — sideways, over the lower cell, which is
    /// how a stepped basal surface advances — so the lower cell gains the
    /// height that face takes up.
    fn grow_basal(&mut self, k: &Kinetics, dt: f32, nucleation: Nucleation) {
        if self.step.is_multiple_of(Self::RIM_REFRESH) {
            self.refresh_rim_distance();
        }
        let w = self.w;
        let n = self.n;
        let layers = self.dz.len();
        let offs = Self::neighbour_offsets(w);
        let a = &self.a;
        let rim = &self.rim;
        let top = &self.top;
        let h_old = &self.h;
        let (dz, zc) = (&self.dz, &self.zc);
        let dx = self.dx_um;
        let dx_over_x0 = dx / k.x0;
        let big_r = self.radius;
        let growth: Vec<(usize, f32)> = (0..n)
            .into_par_iter()
            .filter(|&i| a[i] != 0)
            .map(|i| {
                let (q, r) = ((i % w) as i32 - big_r, (i / w) as i32 - big_r);
                let mut dh = 0.0f32;
                let j0 = top[i] as usize;
                if j0 < layers {
                    let sigma_b = self.value_at(j0, q, r);
                    let alpha = Self::basal_alpha(k, dx, rim[i], sigma_b);
                    let kappa = alpha * dx_over_x0;
                    let sigma_s = sigma_b / (1.0 + 0.5 * dz[j0] * kappa);
                    dh += 2.0 * alpha * k.v_kin * sigma_s * dt / dx;
                }
                // Steps: the exposed prism faces of taller neighbours, layer
                // by layer, over the part of the layer between this cell's
                // own top and theirs; what they take up fills this cell,
                // but never beyond the step's height.
                let h_here = h_old[i];
                let mut tallest = h_here;
                let mut step = 0.0f32;
                for j in j0..layers {
                    let zb = zc[j] - 0.5 * dz[j];
                    let (mut n_ice, mut h_sum, mut face) = (0u32, 0.0f32, 0.0f32);
                    for off in offs {
                        let nb = (i as isize + off) as usize;
                        if a[nb] != 0 {
                            let hn = h_old[nb];
                            let lo = zb.max(0.5 * h_here);
                            let hi = (zb + dz[j]).min(0.5 * hn);
                            if hi > lo {
                                n_ice += 1;
                                h_sum += hn;
                                face += hi - lo;
                                tallest = tallest.max(hn);
                            }
                        }
                    }
                    if n_ice == 0 {
                        break;
                    }
                    let sigma_b = self.value_at(j, q, r);
                    let alpha = Self::prism_alpha(k, dx, n_ice, h_sum / n_ice as f32, 0.0, sigma_b);
                    let kappa = alpha * dx_over_x0;
                    let sigma_s = sigma_b / (1.0 + Self::HALF_ROW * kappa);
                    step +=
                        2.0 * alpha * k.v_kin * sigma_s * (face / n_ice as f32) * dt / (Self::CELL_AREA * dx);
                }
                dh += step.min((tallest - h_here).max(0.0));
                (i, dh)
            })
            .collect();
        let mut dh_all = vec![0.0f32; n];
        for (i, dh) in growth {
            dh_all[i] = dh;
        }
        // Layer occupancy is discrete, so the smallest asymmetry of the
        // relaxed field would otherwise grow into different heights.
        if nucleation.symmetric || nucleation.quantised() {
            self.symmetrise(&mut dh_all, |i| a[i] != 0);
        }
        if nucleation.quantised() {
            let dh_max = dh_all.iter().cloned().fold(0.0f32, f32::max);
            for dh in dh_all.iter_mut() {
                *dh = nucleation.quantise(*dh, dh_max);
            }
        }
        let mut h_max = self.h_max;
        for (i, dh) in dh_all.into_iter().enumerate() {
            if a[i] != 0 {
                self.h[i] += dh;
                h_max = h_max.max(self.h[i]);
            }
        }
        self.h_max = h_max;
        self.refresh_top();
    }

    /// Replace every selected cell's value by the mean over its twelve
    /// lattice-symmetric images (those that are selected too).
    fn symmetrise(&self, values: &mut [f32], selected: impl Fn(usize) -> bool + Sync) {
        let w = self.w;
        let big_r = self.radius;
        let src: &[f32] = values;
        let symmetric: Vec<(usize, f32)> = (0..src.len())
            .into_par_iter()
            .filter(|&i| selected(i))
            .map(|i| {
                let q = (i % w) as i32 - big_r;
                let r = (i / w) as i32 - big_r;
                let (mut sum, mut count) = (0.0f32, 0u32);
                let (mut x, mut y) = (q, r);
                for _ in 0..6 {
                    for (iq, ir) in [(x, y), (y, x)] {
                        if hex_distance(iq, ir) < big_r {
                            let j = (ir + big_r) as usize * w + (iq + big_r) as usize;
                            if selected(j) {
                                sum += src[j];
                                count += 1;
                            }
                        }
                    }
                    (x, y) = (-y, x + y);
                }
                (i, if count > 0 { sum / count as f32 } else { src[i] })
            })
            .collect();
        for (i, v) in symmetric {
            values[i] = v;
        }
    }

    /// Change the ambient vapour density.  The whole vapour field is rescaled
    /// so the quasi-static depletion profile around the crystal is kept.
    fn set_ambient(&mut self, rho: f32) {
        if rho == self.rho {
            return;
        }
        if self.rho > 0.0 {
            let f = rho / self.rho;
            self.d.par_iter_mut().for_each(|v| *v *= f);
        } else {
            // Starting from a completely dry field: refill it uniformly.
            self.d.par_iter_mut().for_each(|v| {
                if *v > 0.0 {
                    *v = rho;
                }
            });
        }
        self.rho = rho;
    }

    /// Height (µm) of a sharpened edge advancing slowly; an edge advancing
    /// at `v` is thinner, EDGE_UM / (1 + v/ESI_UM_PER_S).
    pub const EDGE_UM: f32 = 2.0;
    /// Edge speed (µm/s) at which the edge-sharpening instability has
    /// halved the edge height (thin plates at −15 °C grow their edges at
    /// ~1 µm/s and are 1–2 µm thick; fern arms grow faster and are
    /// thinner).
    pub const ESI_UM_PER_S: f32 = 1.0;

    /// Height of a sharpened edge advancing at `v_um_s`, in cells (at least
    /// one).
    pub fn edge_height(&self, v_um_s: f32) -> f32 {
        (Self::EDGE_UM / (1.0 + v_um_s.max(0.0) / Self::ESI_UM_PER_S) / self.dx_um).max(1.0)
    }

    fn apply_attachment(&mut self, edge_thinning: f32) {
        let w = self.w;
        let big_r = self.radius;
        let offs = Self::neighbour_offsets(w);
        // Heights are inherited from the crystal as it was before this
        // step's attachments, so symmetric cells attaching together see the
        // same neighbours whatever the order they are processed in.  Where
        // the edge-sharpening instability acts, the new rim is thinner than
        // the ice behind it, towards the sharpened-edge height for its own
        // growth rate.
        let joining: Vec<(usize, f32)> = (0..self.attach.len())
            .filter(|&i| self.attach[i] != 0)
            .map(|i| {
                let edge = self.edge_height(self.rate[i]);
                let (mut sum, mut n) = (0.0, 0.0);
                for off in offs {
                    let j = (i as isize + off) as usize;
                    if self.a[j] != 0 {
                        sum += self.h[j];
                        n += 1.0;
                    }
                }
                let inherited = if n > 0.0 { sum / n } else { 1.0 };
                let thinned = if inherited > edge {
                    edge + (inherited - edge) * (1.0 - edge_thinning)
                } else {
                    inherited
                };
                (i, thinned)
            })
            .collect();
        for (i, h) in joining {
            self.attach[i] = 0;
            self.h[i] = h;
            self.a[i] = 1;
            let top = self.layers_of(h);
            self.top[i] = top;
            // Mass beyond one cell was delivered while the site waited to
            // nucleate; it stays with the cell as crystal mass.
            self.c[i] += self.b[i];
            self.b[i] = 0.0;
            for l in self.layers.iter().take(top as usize) {
                if l.s == 1 {
                    self.d[l.off + i] = 0.0;
                }
            }
            self.crystal_cells += 1;
            let q = (i % w) as i32 - big_r;
            let r = (i / w) as i32 - big_r;
            self.crystal_radius = self.crystal_radius.max(hex_distance(q, r));
        }
    }

    /// Steps between refreshes of the rim-distance map.
    const RIM_REFRESH: u64 = 16;

    /// Multi-source breadth-first distance from the rim over crystal cells.
    fn refresh_rim_distance(&mut self) {
        let w = self.w;
        let offs = Self::neighbour_offsets(w);
        let n = self.a.len();
        let mut queue = Vec::new();
        for i in 0..n {
            if self.a[i] == 0 {
                continue;
            }
            let on_rim = offs.iter().any(|&off| {
                let j = i as isize + off;
                j < 0 || j >= n as isize || self.a[j as usize] == 0
            });
            self.rim[i] = if on_rim { 0 } else { u8::MAX };
            if on_rim {
                queue.push(i);
            }
        }
        let mut head = 0;
        while head < queue.len() {
            let i = queue[head];
            head += 1;
            let d = self.rim[i].saturating_add(1);
            for off in offs {
                let j = i as isize + off;
                if j < 0 || j >= n as isize {
                    continue;
                }
                let j = j as usize;
                if self.a[j] != 0 && self.rim[j] > d {
                    self.rim[j] = d;
                    queue.push(j);
                }
            }
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        let mass = self
            .a
            .iter()
            .zip(&self.c)
            .map(|(&a, &c)| if a != 0 { c } else { -1.0 })
            .collect();
        Snapshot {
            radius: self.radius,
            width: self.w,
            mass,
            height: self.h.clone(),
            crystal_radius: self.crystal_radius,
            crystal_cells: self.crystal_cells,
            step: self.step,
            molecules: self.molecules(),
        }
    }
}

/// In-plane curvature (1/µm, positive where the ice is convex) at every
/// boundary pixel of the lattice `a` (row stride `w`, radius `big_r`),
/// into `out` (zero elsewhere).  The ice fraction of a disc of radius R
/// around a point on the interface is 1/2 − 2κR/(3π); the interface runs
/// between the boundary pixel and its ice neighbours, so the fraction is
/// averaged over discs centred on the pixel and on those neighbours, which
/// cancels the offset from the interface to first order whatever its
/// orientation, and the estimate is then averaged over the boundary
/// pixels within the disc, which smooths the staircase.  `scale` is the
/// lattice calibration of the factor 3π/(2R): κ = scale·(1/2 − f).
fn disc_curvature(a: &[u8], w: usize, big_r: i32, rc: i32, scale: f32, out: &mut [f32]) {
    let mut disc: Vec<isize> = Vec::new();
    for dr in -rc..=rc {
        for dq in -rc..=rc {
            if hex_distance(dq, dr) <= rc {
                disc.push(dr as isize * w as isize + dq as isize);
            }
        }
    }
    let disc = &disc;
    let offs = Automaton::neighbour_offsets(w);
    let fraction = move |i: usize| {
        disc.iter()
            .filter(|&&off| a[(i as isize + off) as usize] != 0)
            .count() as f32
            / disc.len() as f32
    };
    // Raw estimate per boundary pixel (NaN elsewhere).
    let mut raw = vec![f32::NAN; a.len()];
    raw.par_chunks_mut(w).enumerate().for_each(|(ri, row)| {
        let r = ri as i32 - big_r;
        let Some((q_lo, q_hi)) = inner_range(big_r - rc, r) else {
            return;
        };
        for (qi, c) in row.iter_mut().enumerate() {
            let q = qi as i32 - big_r;
            let i = ri * w + qi;
            if q < q_lo || q > q_hi || a[i] != 0 {
                continue;
            }
            let (mut sum, mut count) = (0.0f32, 0u32);
            for off in offs {
                let j = (i as isize + off) as usize;
                if a[j] != 0 {
                    sum += fraction(j);
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let f = 0.5 * (fraction(i) + sum / count as f32);
            *c = (0.5 - f) * scale;
        }
    });
    let raw = &raw;
    out.par_chunks_mut(w).enumerate().for_each(|(ri, row)| {
        let r = ri as i32 - big_r;
        let Some((q_lo, q_hi)) = inner_range(big_r - 2 * rc, r) else {
            row.fill(0.0);
            return;
        };
        for (qi, c) in row.iter_mut().enumerate() {
            let q = qi as i32 - big_r;
            let i = ri * w + qi;
            *c = 0.0;
            if q < q_lo || q > q_hi || raw[i].is_nan() {
                continue;
            }
            let (mut sum, mut count) = (0.0f32, 0u32);
            for &off in disc.iter() {
                let v = raw[(i as isize + off) as usize];
                if !v.is_nan() {
                    sum += v;
                    count += 1;
                }
            }
            *c = sum / count as f32;
        }
    });
}

/// Whether vapour cell `i` has an ice neighbour.
fn curv_is_boundary(a: &[u8], w: usize, i: usize) -> bool {
    Automaton::neighbour_offsets(w).iter().any(|&off| {
        let j = i as isize + off;
        j >= 0 && (j as usize) < a.len() && a[j as usize] != 0
    })
}

/// Representative of the symmetry class of cell (q, r) under the twelve
/// symmetries of the hexagonal lattice (six rotations, each with a mirror).
#[inline]
fn canonical(q: i32, r: i32) -> (i32, i32) {
    let mut best = (i32::MAX, i32::MAX);
    let (mut x, mut y) = (q, r);
    for _ in 0..6 {
        best = best.min((x, y)).min((y, x)); // (r, q) is the mirror image
        (x, y) = (-y, x + y); // rotate by 60°
    }
    best
}

/// splitmix64 finaliser: a cheap, well-mixed hash for per-cell randomness.
#[inline]
fn hash64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform number in [0, 1) from a hash.
#[inline]
fn uniform(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::{Conditions, kinetics_params};

    fn kinetics(t: f64, s: f64) -> Kinetics {
        kinetics_params(&Conditions {
            temperature: t,
            supersaturation: s,
            pressure: crate::physics::P_STANDARD,
        })
        .into()
    }

    fn grow(t: f64, s: f64, seconds: f32, nucleation: Nucleation, dx: f32) -> Automaton {
        let k = kinetics(t, s);
        let mut auto = Automaton::new(40, k.sigma_inf as f64, 1).with_cell_size(dx);
        auto.seed_hexagon();
        let mut elapsed = 0.0;
        while elapsed < seconds && auto.crystal_radius() < 34 {
            elapsed += auto.step(&k, nucleation, 2.0);
        }
        auto
    }

    /// Six-fold rotational symmetry (q, r) -> (-r, q + r) and mirror (q, r) -> (r, q)
    /// of the crystal's shape and heights.  (The waiting-time mass proxy can
    /// differ in the last rounding bits between symmetric cells.)
    fn assert_symmetric(snap: &Snapshot) {
        for (q, r, _) in snap.cells() {
            assert!(
                snap.mass_at(-r, q + r).is_some(),
                "rotated cell missing at ({q},{r})"
            );
            assert!(snap.mass_at(r, q).is_some(), "mirrored cell missing at ({q},{r})");
            let h = snap.height_at(q, r);
            assert!(
                (snap.height_at(-r, q + r) - h).abs() < 0.02 * h.max(1.0),
                "height asymmetric at ({q},{r})"
            );
        }
        // And nothing exists that its images do not: same cell count both ways.
        let n = snap.cells().count();
        let rotated = snap
            .cells()
            .filter(|&(q, r, _)| snap.mass_at(-r, q + r).is_some())
            .count();
        assert_eq!(n, rotated);
    }

    #[test]
    fn crystal_grows_and_stays_symmetric() {
        let auto = grow(
            -15.0,
            0.2,
            120.0,
            Nucleation {
                rate: 2.0,
                symmetric: true,
                quantum: 0.0,
            },
            2.0,
        );
        assert!(
            auto.crystal_cells() > 50,
            "grew only {} cells",
            auto.crystal_cells()
        );
        assert_symmetric(&auto.snapshot());
    }

    #[test]
    fn nothing_grows_in_saturated_air() {
        let auto = grow(-15.0, 0.0, 120.0, Nucleation::DETERMINISTIC, 2.0);
        let mut seed = Automaton::new(40, 0.0, 1).with_cell_size(2.0);
        seed.seed_hexagon();
        assert_eq!(auto.crystal_cells(), seed.crystal_cells());
    }

    #[test]
    fn per_cell_nucleation_is_deterministic_and_breaks_symmetry() {
        let run = |seed: u64| {
            let k = kinetics(-15.0, 0.25);
            let mut auto = Automaton::new(40, k.sigma_inf as f64, seed).with_cell_size(2.0);
            auto.seed_hexagon();
            let mut elapsed = 0.0;
            while elapsed < 120.0 && auto.crystal_radius() < 34 {
                elapsed += auto.step(
                    &k,
                    Nucleation {
                        rate: 1.0,
                        symmetric: false,
                        quantum: 0.0,
                    },
                    2.0,
                );
            }
            auto.snapshot()
        };
        assert_eq!(run(7).mass, run(7).mass);
        assert_ne!(run(7).mass, run(8).mass);
        let snap = run(7);
        let asymmetric = snap
            .cells()
            .any(|(q, r, m)| snap.mass_at(-r, q + r).is_none_or(|x| (x - m).abs() > 1e-3));
        assert!(asymmetric);
    }

    #[test]
    fn canonical_is_invariant_under_the_symmetry_group() {
        for q in -5..=5 {
            for r in -5..=5 {
                let c = canonical(q, r);
                assert_eq!(canonical(-r, q + r), c);
                assert_eq!(canonical(r, q), c);
            }
        }
        assert_eq!(canonical(0, 0), (0, 0));
    }

    #[test]
    fn needles_grow_along_c_and_plates_do_not() {
        let needle = grow(-5.0, 0.15, 200.0, Nucleation::DETERMINISTIC, 1.0);
        let plate = grow(-15.0, 0.15, 200.0, Nucleation::DETERMINISTIC, 1.0);
        let aspect = |a: &Automaton| a.snapshot().max_height() / a.crystal_radius().max(1) as f32;
        assert!(aspect(&needle) > 1.5, "needle aspect {}", aspect(&needle));
        assert!(aspect(&plate) < 0.5, "plate aspect {}", aspect(&plate));
        assert!(aspect(&needle) > 4.0 * aspect(&plate));
    }

    #[test]
    fn field_relaxes_towards_the_far_value() {
        let k = kinetics(-15.0, 0.1);
        let mut auto = Automaton::new(30, k.sigma_inf as f64, 1).with_cell_size(2.0);
        auto.seed_hexagon();
        auto.converge_field(&k);
        auto.step(&k, Nucleation::DETERMINISTIC, 1.0);
        // With the monopole far field the ring itself sits a little below
        // σ_∞ (the crystal's uptake), and the field rises towards it.
        let far = auto.d[auto.index(25, 0)];
        assert!(
            far > 0.5 * k.sigma_inf && far <= k.sigma_inf,
            "far field {far} vs {}",
            k.sigma_inf
        );
        let near = auto.d[auto.index(2, 0)];
        assert!(near < far, "surface {near} should be depleted below {far}");
    }

    /// Growth of a rough (α = 1) crystal against the exact 2-D cylindrical
    /// solution of Laplace's equation with the mixed boundary condition:
    /// σ_surf/σ_∞ = (X₀/R) / (X₀/R + α ln(R_out/R)), v = α v_kin σ_surf.
    /// Checks the in-plane discretisation, the boundary condition and the
    /// time stepping together.
    fn cylinder_growth_ratio(dx: f32) -> f64 {
        let radius = (120.0 / dx) as usize;
        let k = Kinetics {
            sigma_inf: 0.1,
            v_kin: 200.0,
            x0: 0.145,
            d0: 0.0,
            a_prism: 1.0,
            sigma0_prism: 0.0, // α = 1 everywhere: no facets
            w0_prism: 0.0,
            a_basal: 0.0,
            sigma0_basal: 1.0,
            w0_basal: 0.0,
        };
        let mut auto = Automaton::new(radius, k.sigma_inf as f64, 1)
            .with_cell_size(dx)
            .with_reflecting_top();
        // Seed a disc of radius 20 µm spanning every layer: with the
        // reflecting top the field is then exactly two-dimensional.
        let w = auto.w;
        for i in 0..auto.a.len() {
            let (q, r) = ((i % w) as i32 - radius as i32, (i / w) as i32 - radius as i32);
            if hex_distance(q, r) as f32 * dx <= 20.0 {
                auto.a[i] = 1;
                auto.h[i] = 1e6;
            }
        }
        auto.refresh_top();
        auto.crystal_cells = auto.a.iter().filter(|&&a| a != 0).count();
        auto.crystal_radius = (20.0 / dx) as i32;
        let sweeps = auto.converge_field(&k);
        let before = auto.sweep_count();
        for _ in 0..30 {
            auto.step(&k, Nucleation::DETERMINISTIC, 0.05);
        }
        eprintln!(
            "dx={dx}: initial convergence {sweeps} sweeps (ω={:.4}), then {:.1} sweeps/step; surface σ/σ∞ ≈ {:.4}",
            auto.sor_omega(),
            (auto.sweep_count() - before) as f64 / 30.0,
            auto.d[auto.index((20.0 / dx) as i32 + 1, 0)] / k.sigma_inf
        );
        let area = |a: &Automaton| a.crystal_cells as f64 * 0.866 * (dx as f64).powi(2);
        let r_eq = |a: &Automaton| (area(a) / std::f64::consts::PI).sqrt();
        let (r0, t0) = (r_eq(&auto), auto.time);
        while r_eq(&auto) < r0 + 8.0 {
            auto.step(&k, Nucleation::DETERMINISTIC, 2.0);
        }
        let (r1, t1) = (r_eq(&auto), auto.time);
        let measured = (r1 - r0) / (t1 - t0);
        // Analytic rate at the mid radius; the far boundary sits at the
        // lattice ring, whose inradius is radius·√3/2 cells.
        let r_mid = 0.5 * (r0 + r1);
        let r_out = radius as f64 * 0.866 * dx as f64;
        let x0_over_r = k.x0 as f64 / r_mid;
        let sigma_surf = k.sigma_inf as f64 * x0_over_r / (x0_over_r + (r_out / r_mid).ln());
        let analytic = k.v_kin as f64 * sigma_surf;
        eprintln!(
            "cylinder dx={dx}: measured {measured:.4} µm/s, analytic {analytic:.4} µm/s (ratio {:.3})",
            measured / analytic
        );
        measured / analytic
    }

    #[test]
    fn rough_cylinder_growth_matches_the_analytic_rate() {
        let r1 = cylinder_growth_ratio(1.0);
        let r05 = cylinder_growth_ratio(0.5);
        assert!((r1 - 1.0).abs() < 0.15, "1 µm cells: ratio {r1}");
        assert!((r05 - 1.0).abs() < 0.15, "0.5 µm cells: ratio {r05}");
    }

    /// Vapour uptake of a rough (α = 1) sphere in an unbounded field
    /// against the analytic solution of Laplace's equation with the mixed
    /// boundary condition: Q = 4πR σ_∞ / (1 + X₀/R) (unit diffusivity,
    /// lengths in cells).  The total uptake of a strongly absorbing body is
    /// set by the far field, so this checks the three-dimensional field —
    /// the layer stack, the vertical coupling, the far-field boundary and
    /// the basal faces together with the prism faces — without depending
    /// on how the staircase of the height map is counted as surface.
    fn sphere_uptake_ratio(dx: f32) -> f64 {
        let radius = (80.0 / dx) as usize;
        let k = Kinetics {
            sigma_inf: 0.1,
            v_kin: 200.0,
            x0: 0.145,
            d0: 0.0,
            a_prism: 1.0,
            sigma0_prism: 0.0,
            w0_prism: 0.0,
            a_basal: 1.0,
            sigma0_basal: 0.0,
            w0_basal: 0.0,
        };
        let mut auto = Automaton::new(radius, k.sigma_inf as f64, 1).with_cell_size(dx);
        let w = auto.w;
        let r_s = 10.0 / dx;
        for i in 0..auto.a.len() {
            let (q, r) = ((i % w) as i32 - radius as i32, (i / w) as i32 - radius as i32);
            let (x, y) = (q as f32 + 0.5 * r as f32, r as f32 * 0.866_025_4);
            let rho2 = x * x + y * y;
            if rho2 <= r_s * r_s {
                auto.a[i] = 1;
                auto.h[i] = (2.0 * (r_s * r_s - rho2).sqrt()).max(1.0);
            }
        }
        auto.refresh_top();
        auto.refresh_rim_distance();
        auto.crystal_cells = auto.a.iter().filter(|&&a| a != 0).count();
        auto.crystal_radius = r_s as i32;
        let sweeps = auto.converge_field(&k);
        let measured = auto.total_uptake(&k) as f64;
        let x0_cells = k.x0 as f64 / dx as f64;
        let analytic =
            4.0 * std::f64::consts::PI * r_s as f64 * k.sigma_inf as f64 / (1.0 + x0_cells / r_s as f64);
        eprintln!(
            "sphere dx={dx}: {} layers, converged in {sweeps} sweeps; uptake {measured:.3}, analytic {analytic:.3} (ratio {:.3})",
            auto.dz.len(),
            measured / analytic
        );
        measured / analytic
    }

    #[test]
    fn rough_sphere_uptake_matches_the_analytic_flux() {
        let r = sphere_uptake_ratio(1.0);
        assert!((r - 1.0).abs() < 0.1, "1 µm cells: ratio {r}");
        let r = sphere_uptake_ratio(0.5);
        assert!((r - 1.0).abs() < 0.1, "0.5 µm cells: ratio {r}");
    }

    /// The coarsened upper layers must not change the field near the
    /// crystal appreciably: compare against the same stack at full
    /// resolution around a thin plate.
    #[test]
    fn coarse_layers_match_full_resolution_near_the_plane() {
        let k = kinetics(-15.0, 0.3);
        let dx = 0.5f32;
        let radius = 200usize;
        let build = |coarsen: bool| {
            let mut auto = Automaton::new(radius, k.sigma_inf as f64, 1).with_cell_size(dx);
            let dz = auto.dz.clone();
            auto.set_layers(dz, true, coarsen);
            let w = auto.w;
            for i in 0..auto.a.len() {
                let (q, r) = ((i % w) as i32 - radius as i32, (i / w) as i32 - radius as i32);
                if hex_distance(q, r) <= 60 {
                    auto.a[i] = 1;
                    auto.h[i] = 4.0;
                }
            }
            auto.refresh_top();
            auto.refresh_rim_distance();
            auto.crystal_cells = auto.a.iter().filter(|&&a| a != 0).count();
            auto.crystal_radius = 60;
            auto.converge_field(&k);
            auto
        };
        let (fine, coarse) = (build(false), build(true));
        for j in 0..4 {
            let (mut worst, mut worst_at) = (0.0f32, (0, 0));
            let (mut sum_f, mut sum_c) = (0.0f64, 0.0f64);
            for q in -150..=150 {
                for r in -150..=150 {
                    if hex_distance(q, r) > 150 {
                        continue;
                    }
                    let (vf, vc) = (fine.value_at(j, q, r), coarse.value_at(j, q, r));
                    sum_f += vf as f64;
                    sum_c += vc as f64;
                    if vf > 1e-3 {
                        let rel = (vc - vf).abs() / vf;
                        if rel > worst {
                            worst = rel;
                            worst_at = (q, r);
                        }
                    }
                }
            }
            eprintln!(
                "layer {j}: worst relative difference {worst:.4} at {worst_at:?}; mean fine {:.5} coarse {:.5} (fine {} layers, coarse {} layers, q_far {:.2} vs {:.2})",
                sum_f / 70000.0,
                sum_c / 70000.0,
                fine.layers.len(),
                coarse.layers.len(),
                fine.q_far,
                coarse.q_far
            );
            assert!(worst < 0.05, "layer {j}: {worst}");
        }
    }

    #[test]
    #[ignore]
    fn corner_vs_facet_probe() {
        let k = kinetics(-15.0, 0.3);
        for dx in [1.0f32, 0.5] {
            let radius = (100.0 / dx) as usize;
            let rho = (30.0 / dx) as i32;
            let mut auto = Automaton::new(radius, k.sigma_inf as f64, 1).with_cell_size(dx);
            let w = auto.w;
            for i in 0..auto.a.len() {
                let (q, r) = ((i % w) as i32 - radius as i32, (i / w) as i32 - radius as i32);
                if hex_distance(q, r) <= rho {
                    auto.a[i] = 1;
                    auto.h[i] = 2.0 / dx;
                }
            }
            auto.refresh_top();
            auto.refresh_rim_distance();
            auto.crystal_cells = auto.a.iter().filter(|&&a| a != 0).count();
            auto.crystal_radius = rho;
            auto.converge_field(&k);
            auto.fill_and_attach(&k, Nucleation::DETERMINISTIC, 1.0);
            let show = |auto: &Automaton, q: i32, r: i32, label: &str| {
                let i = auto.index(q, r);
                eprintln!(
                    "dx {dx} {label:>14} ({q:4},{r:4}): rate {:.3} µm/s, curv {:.3}, σ0 {:.5} σ1 {:.5} σ2 {:.5} σ3 {:.5}",
                    auto.rate[i],
                    auto.curv[i],
                    auto.value_at(0, q, r),
                    auto.value_at(1, q, r),
                    auto.value_at(2, q, r),
                    auto.value_at(3, q, r)
                );
            };
            show(&auto, rho + 1, 0, "corner");
            show(&auto, rho + 1, -1, "corner side");
            show(&auto, rho / 2 + 1, -(rho + 1), "facet centre");
            show(&auto, 1, -(rho + 1), "facet end");
            let far = auto.value_at(0, rho + (10.0 / dx) as i32, 0);
            eprintln!(
                "dx {dx}: σ 10 µm beyond corner {far:.4}, q_far {:.3}, layers {:?}",
                auto.q_far, auto.dz
            );
        }
    }

    /// With a vanishing quantum the quantised model is the deterministic
    /// baseline: same crystal, cell for cell.
    #[test]
    fn quantised_model_with_a_tiny_quantum_is_the_deterministic_baseline() {
        let quantised = Nucleation {
            rate: f32::INFINITY,
            symmetric: true,
            quantum: 1e-6,
        };
        let a = grow(-15.0, 0.2, 60.0, Nucleation::DETERMINISTIC, 1.0).snapshot();
        let b = grow(-15.0, 0.2, 60.0, quantised, 1.0).snapshot();
        assert_eq!(a.crystal_cells, b.crystal_cells);
        assert_eq!(a.crystal_radius, b.crystal_radius);
        assert!(a.crystal_cells > 50);
        // Heights: the quantisation error is ~1e-6, but a layer-occupancy
        // flip can amplify it, so allow the symmetry test's tolerance.
        let heights_agree = a
            .height
            .iter()
            .zip(&b.height)
            .all(|(x, y)| (x - y).abs() <= 0.02 * x.abs().max(1.0));
        assert!(heights_agree, "heights differ between the models");
        // And a coarse quantum still grows a similar crystal.
        let coarse = Nucleation {
            rate: f32::INFINITY,
            symmetric: true,
            quantum: 0.1,
        };
        let c = grow(-15.0, 0.2, 60.0, coarse, 1.0).snapshot();
        let ratio = c.crystal_cells as f64 / a.crystal_cells as f64;
        assert!(
            (0.7..1.3).contains(&ratio),
            "coarse quantum: {} vs {} cells",
            c.crystal_cells,
            a.crystal_cells
        );
    }

    #[test]
    fn curvature_of_a_disc_is_its_inverse_radius() {
        for (dx, rho) in [(0.5f32, 40.0f32), (1.0, 40.0), (0.25, 60.0)] {
            let radius = 100usize;
            let mut auto = Automaton::new(radius, 0.1, 1).with_cell_size(dx);
            let w = auto.w;
            for i in 0..auto.a.len() {
                let (q, r) = ((i % w) as i32 - radius as i32, (i / w) as i32 - radius as i32);
                let (x, y) = (q as f32 + 0.5 * r as f32, r as f32 * 0.866_025_4);
                auto.a[i] = ((x * x + y * y).sqrt() <= rho) as u8;
            }
            auto.refresh_curvature();
            let boundary: Vec<f32> = (0..auto.a.len())
                .filter(|&i| auto.a[i] == 0 && curv_is_boundary(&auto.a, w, i))
                .map(|i| auto.curv[i])
                .collect();
            let mean = boundary.iter().sum::<f32>() / boundary.len() as f32;
            let expected = 1.0 / (rho * dx);
            eprintln!(
                "dx={dx}: mean curvature {mean:.4} /µm over {} boundary pixels, expected {expected:.4}",
                boundary.len()
            );
            let sd =
                (boundary.iter().map(|c| (c - mean).powi(2)).sum::<f32>() / boundary.len() as f32).sqrt();
            eprintln!("   ratio {:.3}, spread {:.4}", mean / expected, sd);
            assert!(
                (mean / expected - 1.0).abs() < 0.25,
                "dx={dx}: {mean} vs {expected}"
            );
        }
        // A flat facet reads as (nearly) flat.
        let radius = 60usize;
        let mut auto = Automaton::new(radius, 0.1, 1).with_cell_size(0.5);
        let w = auto.w;
        for i in 0..auto.a.len() {
            let r = (i / w) as i32 - radius as i32;
            auto.a[i] = (r <= 0) as u8;
        }
        auto.refresh_curvature();
        let c = auto.curv[auto.index(0, 1)];
        assert!(c.abs() < 0.05, "flat facet curvature {c}");
    }

    #[test]
    fn molecule_count_matches_a_hand_calculation() {
        // A 1 mm-wide, 20 µm-thick plate: 1.57e7 µm³ of ice, ~4.8e17 molecules.
        let cells = std::f64::consts::PI * 500.0f64.powi(2) / (0.866 * 100.0); // 10 µm cells
        let n = molecules_from_height_sum(cells * 2.0, 10.0);
        assert!(n > 4e17 && n < 6e17, "{n:e}");
        let auto = Automaton::new(10, 0.05, 1);
        assert!(auto.molecules() > 0.0);
    }
}
