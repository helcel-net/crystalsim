// Page logic: presets, the history editor, and a worker that grows the crystal.

const PRESETS = {
  'Stellar dendrite (−15 °C, 0.26 g/m³)': `# A stellar dendrite: −15 °C, well above water saturation.
[simulation]
radius = 150
cell_size_um = 1.0

[[history]]
time = 0
temperature = -15
supersaturation = 0.26

[[history]]
time = 200
temperature = -15
supersaturation = 0.26
`,
  'Hollow column (−5 °C, 0.15 g/m³)': `# A hollow column: −5 °C, where the basal faces run.
[simulation]
radius = 100
cell_size_um = 1.0

[[history]]
time = 0
temperature = -5
supersaturation = 0.15

[[history]]
time = 300
temperature = -5
supersaturation = 0.15
`,
  'Sectored plate (−15 °C, 0.10 g/m³)': `# A plate with sectors: −15 °C, below water saturation.
[simulation]
radius = 150
cell_size_um = 1.0

[[history]]
time = 0
temperature = -15
supersaturation = 0.10

[[history]]
time = 300
temperature = -15
supersaturation = 0.10
`,
  'Story flake (plate → dendrite → plate)': `# The classic life story: born as a plate, falls through the humid
# heart of the cloud and sprouts arms, then dries out and broadens again.
[simulation]
radius = 200
cell_size_um = 1.0

[[history]]
time = 0
temperature = -14
supersaturation = 0.08

[[history]]
time = 150
temperature = -15
supersaturation = 0.1

[[history]]
time = 165
temperature = -15
supersaturation = 0.3

[[history]]
time = 300
temperature = -15
supersaturation = 0.3

[[history]]
time = 320
temperature = -13
supersaturation = 0.12

[[history]]
time = 400
temperature = -12
supersaturation = 0.12
`,
  'Column then plate (−5 °C → −15 °C)': `# A column that drops into plate-growing air.
[simulation]
radius = 150
cell_size_um = 1.0

[[history]]
time = 0
temperature = -5.5
supersaturation = 0.15

[[history]]
time = 150
temperature = -5.5
supersaturation = 0.15

[[history]]
time = 180
temperature = -15
supersaturation = 0.25

[[history]]
time = 330
temperature = -15
supersaturation = 0.25
`,
};

const $ = (id) => document.getElementById(id);
const preset = $('preset');
const editor = $('toml');
const status = $('status');
const topCanvas = $('top');
const isoCanvas = $('iso');
const runBtn = $('run');
const stopBtn = $('stop');

for (const name of Object.keys(PRESETS)) {
  const o = document.createElement('option');
  o.value = name;
  o.textContent = name;
  preset.appendChild(o);
}
preset.onchange = () => { editor.value = PRESETS[preset.value]; };
editor.value = PRESETS[preset.value];

// Replace or insert the [simulation] keys the controls set.
function withControls(toml) {
  const radius = $('radius').value;
  const cell = $('cell').value;
  const growth = $('growth').value;
  let text = toml.replace(/^\s*(radius|cell_size_um|growth)\s*=.*$/gm, '');
  if (!/^\s*\[simulation\]/m.test(text)) text = '[simulation]\n' + text;
  return text.replace(/^\s*\[simulation\]\s*$/m,
    `[simulation]\nradius = ${radius}\ncell_size_um = ${cell}\ngrowth = "${growth}"`);
}

let worker = null;

function paint(canvas, buffer, width, height) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const img = new ImageData(new Uint8ClampedArray(buffer), width, height);
  ctx.putImageData(img, 0, 0);
}

function show(f) {
  const pct = f.end > 0 ? Math.round(100 * f.t / f.end) : 0;
  status.textContent =
    `t = ${f.t.toFixed(0)} s of ${f.end.toFixed(0)} s (${pct} %)   ` +
    `${f.temperature.toFixed(1)} °C, ${f.supersaturation.toFixed(2)} g/m³ → ${f.habit}\n` +
    `${f.cells} cells, radius ${f.radius} cells, ${f.steps} steps, ${f.molecules.toExponential(2)} molecules, ` +
    `${f.wall.toFixed(0)} s of your time${f.done ? '   — finished' : ''}`;
}

function stop() {
  if (worker) { worker.terminate(); worker = null; }
  runBtn.disabled = false;
  stopBtn.disabled = true;
}

runBtn.onclick = () => {
  stop();
  const toml = withControls(editor.value);
  const cell = parseFloat($('cell').value);
  // Frame cadence in simulated seconds: finer for slow (fine-celled) runs.
  const frameEvery = cell >= 2 ? 10 : cell >= 1 ? 5 : 2;
  worker = new Worker('worker.js', { type: 'module' });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'frame') {
      paint(topCanvas, m.top, m.size, m.size);
      paint(isoCanvas, m.iso, m.size, m.isoH);
      show(m);
    } else if (m.type === 'done') {
      stop();
    } else if (m.type === 'error') {
      status.textContent = 'Error: ' + m.message;
      stop();
    }
  };
  worker.onerror = (e) => { status.textContent = 'Worker error: ' + e.message; stop(); };
  status.textContent = 'Converging the vapour field around the seed…';
  runBtn.disabled = true;
  stopBtn.disabled = false;
  worker.postMessage({ toml, frameEvery, size: 512 });
};
stopBtn.onclick = stop;
