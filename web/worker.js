// Runs one growth session off the main thread and posts frames back.
import init, { Sim } from './pkg/flakesim.js';

const ready = init();

self.onmessage = async (e) => {
  const { toml, frameEvery, size } = e.data;
  try {
    await ready;
    const t0 = performance.now();
    const sim = new Sim(toml);
    const frame = () => {
      const isoH = sim.iso_height(size);
      const top = sim.render_top(size);
      const iso = sim.render_iso(size);
      self.postMessage(
        {
          type: 'frame',
          size,
          isoH,
          top: top.buffer,
          iso: iso.buffer,
          t: sim.time(),
          end: sim.end(),
          temperature: sim.temperature(),
          supersaturation: sim.supersaturation(),
          habit: sim.habit(),
          cells: sim.cells(),
          radius: sim.radius(),
          steps: sim.steps(),
          molecules: sim.molecules(),
          wall: (performance.now() - t0) / 1000,
          done: sim.done(),
        },
        [top.buffer, iso.buffer],
      );
    };
    frame();
    while (!sim.done()) {
      sim.advance(frameEvery);
      frame();
    }
    sim.free();
    self.postMessage({ type: 'done' });
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
