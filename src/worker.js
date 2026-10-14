// ---- runs in a Web Worker: steps the world and posts compact frames to the page ----
let skipLeft = 0, w = null, ch = null, paused = false, sps = 30, lastT = 0, due = 0, lastSend = 0, lastStats = 0, lastFood = 0, lastHist = 0, sel = -1, stepCount = 0, rateT = 0, rate = 0;
const STEP_BUDGET = 14;
function frame(now) {
  const n = w.pop, MAX = w.MAX, data = new Float32Array(n * 8), idx = new Int32Array(MAX).fill(-1);
  let k = 0;
  for (let i = 0; i < MAX && k < n; i++) {
    if (!w.alive[i]) continue;
    idx[i] = k; const o = k * 8; data[o + 7] = w.kind[i];
    data[o] = i; data[o + 1] = w.px[i]; data[o + 2] = w.py[i]; data[o + 3] = w.hue[i]; data[o + 4] = w.glue[i]; data[o + 5] = Math.min(1, w.energy[i] / w.C.REPRO); data[o + 6] = w.speak[i];
    k++;
  }
  const nb = w.nBonds, bonds = new Int32Array(nb * 2); let m = 0;
  for (let b = 0; b < nb; b++) {
    const a = idx[w.bondA[b]], c = idx[w.bondB[b]]; if (a < 0 || c < 0) continue;
    const dx = data[a * 8 + 1] - data[c * 8 + 1], dy = data[a * 8 + 2] - data[c * 8 + 2]; if (dx * dx + dy * dy > 6.25) continue;
    bonds[m++] = a; bonds[m++] = c;
  }
  const msg = { type: 'frame', n: k, data, bonds: bonds.subarray(0, m), tick: w.tick, skip: skipLeft };
  const tr = [data.buffer, bonds.buffer];
  if (now - lastStats > 500) {
    lastStats = now;
    const st = w.stats(), o = w.organisms();
    msg.stats = Object.assign(st, o, { rate, seed: w.seed, seasonLen: w.C.SEASON, seasonsOn: w.exp.seasons });
    if (sel >= 0) {
      if (w.alive[sel]) { const r = w.orgOf[sel]; msg.sel = { slot: sel, alive: true, kind: w.kind[sel], gen: w.gen[sel], age: w.age[sel], maxAge: w.C.MAX_AGE, energy: w.energy[sel], glue: w.glue[sel], hue: w.hue[sel], mem: w.mem[sel], nh: w.nh[sel], held: w.held[sel], msg: w.msg[sel], wasteHere: Math.min(1, w.waste[((w.py[sel]) | 0) * w.G + (w.px[sel] | 0)] / w.C.WASTE_CAP), org: r >= 0 ? w.orgSize[r] : 1, members: r >= 0 ? membersOf(r) : [sel] }; }
      else { msg.sel = { slot: sel, alive: false }; sel = -1; }
    }
  }
  if (ch) {
    const ev = ch.takeFresh(); if (ev.length) msg.events = ev;
    if (now - lastHist > 1000) { lastHist = now; const h = ch.history(); msg.hist = h; tr.push(h.data.buffer); msg.tree = ch.tree(); }
  }
  if (now - lastFood > 600) {
    lastFood = now;
    const G = w.G, f = new Uint8Array(G * G), wa = new Uint8Array(G * G), cap = w.C.FOOD_CAP, wc = w.C.WASTE_CAP, W_ = w.waste;
    for (let i = 0; i < G * G; i++) { f[i] = Math.min(255, w.food[i] / cap * 255); wa[i] = Math.min(255, W_[i] / wc * 255); }
    msg.food = f; msg.waste = wa; tr.push(f.buffer, wa.buffer);
  }
  postMessage(msg, tr);
}
function membersOf(root) { const out = []; for (let i = 0; i < w.MAX && out.length < 400; i++) if (w.alive[i] && w.orgOf[i] === root) out.push(i); return out; }
function stepOne() { w.step(); stepCount++; ch.sample(); }
function loop() {
  const now = performance.now();
  if (skipLeft > 0) {
    // jumping ahead: as many steps as fit in 30ms, then a frame so the page stays alive
    const t0 = performance.now();
    while (skipLeft > 0 && performance.now() - t0 < 30) { stepOne(); skipLeft--; }
    due = 0;
  } else if (!paused) {
    due += (now - lastT) * sps / 1000; if (due > 6) due = 6;
    const t0 = performance.now();
    while (due >= 1 && performance.now() - t0 < STEP_BUDGET) { stepOne(); due--; }
  }
  lastT = now;
  if (now - rateT > 1000) { rate = stepCount * 1000 / (now - rateT); stepCount = 0; rateT = now; }
  if (now - lastSend > 33) { lastSend = now; frame(now); }
  setTimeout(loop, 2);
}
// the graphics-card engine: run a batch of steps, copy the creatures back, send a frame, repeat
let gpuHold = null;
async function gpuLoop() {
  while (!w.lost) {
    if (gpuHold) { await gpuHold; continue; }
    const now = performance.now();
    if (skipLeft > 0) { const n = Math.min(skipLeft, 150); w.stepN(n); skipLeft -= n; stepCount += n; due = 0; }
    else if (!paused) {
      due += (now - lastT) * sps / 1000; if (due > 40) due = 40;
      const n = Math.floor(due); if (n > 0) { w.stepN(n); due -= n; stepCount += n; }
    }
    lastT = now;
    if (now - rateT > 1000) { rate = stepCount * 1000 / (now - rateT); stepCount = 0; rateT = now; }
    await w.readback();
    ch.sample();
    frame(performance.now());
    await new Promise(r => setTimeout(r, 4));
  }
  postMessage({ type: 'error', msg: 'The graphics card stopped responding.' });
}

// ---- saving and loading a world: typed arrays packed into one gzip, carried as base64 inside a JSON file ----
const TA = { Float32Array, Int32Array, Uint32Array, Uint8Array, Uint16Array };
function b64(u8) { let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); }
function unb64(s) { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }
async function saveWorld(label) {
  const snap = await w.snapshot(), head = { meta: snap.meta, arrays: [] }, parts = []; let off = 0;
  for (const [k, a] of Object.entries(snap.arrays)) {
    const u = new Uint8Array(a.buffer, a.byteOffset, a.byteLength); head.arrays.push([k, a.constructor.name, off, a.length]); parts.push(u); off += u.byteLength;
    const pad = (4 - off % 4) % 4; if (pad) { parts.push(new Uint8Array(pad)); off += pad; }
  }
  const gz = new Uint8Array(await new Response(new Blob(parts).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
  return JSON.stringify({ format: 'primordial-dish-world', version: 1, world: label, saved: new Date().toISOString(), tick: w.tick, seed: w.seed, chronicle: ch.save(), head, data: b64(gz) });
}
async function loadWorld(text) {
  const f = JSON.parse(text);
  if (f.format !== 'primordial-dish-world') throw new Error("That file isn't a saved dish.");
  const raw = new Uint8Array(await new Response(new Blob([unb64(f.data)]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
  const arrays = {};
  for (const [k, type, off, len] of f.head.arrays) { const T = TA[type]; arrays[k] = new T(raw.buffer.slice(off, off + len * T.BYTES_PER_ELEMENT)); }
  await w.restore({ meta: f.head.meta, arrays });
  ch.load(f.chronicle); sel = -1; skipLeft = 0; lastStats = lastFood = lastHist = 0;
}

onmessage = async e => {
  const m = e.data;
  if (m.type === 'init') {
    if (m.gpu) {
      try { w = await createGpuWorld({ G: m.G, MAX: m.MAX, C: m.C, exp: m.exp }); }
      catch (err) { postMessage({ type: 'error', msg: String(err && err.message || err), fallback: true }); return; }
      ch = createChronicle(w);
      w.reset(m.seed); lastT = rateT = performance.now(); postMessage({ type: 'ready', gpu: true, name: w.adapterName }); gpuLoop();
    } else { w = createWorld({ G: m.G, C: m.C }); if (m.exp) Object.assign(w.exp, m.exp); ch = createChronicle(w); w.reset(m.seed); lastT = rateT = performance.now(); postMessage({ type: 'ready', gpu: false }); loop(); }
  }
  else if (!w) return;
  else if (m.type === 'pattern') { if (w.setPattern) w.setPattern(m.pattern); }
  else if (m.type === 'set' && 'cellSize' in m) { if (w.setCellSize) w.setCellSize(m.cellSize); }
  else if (m.type === 'set') { if ('paused' in m) paused = m.paused; if ('sps' in m) sps = m.sps; for (const k of ['glue', 'memory', 'waste', 'seasons', 'hunters', 'fightBack']) if (k in m) w.exp[k] = m[k]; }
  else if (m.type === 'new') { skipLeft = 0; w.reset(m.seed); ch.reset(); sel = -1; lastStats = 0; lastFood = 0; lastHist = 0; }
  else if (m.type === 'feed') w.feed(m.x, m.y, m.r);
  else if (m.type === 'skip') { skipLeft += m.n; lastStats = 0; }
  else if (m.type === 'select') { sel = m.slot; lastStats = 0; }
  else if (m.type === 'save') {
    let release; gpuHold = w.gpu ? new Promise(r => release = r) : null;
    try { const text = await saveWorld(m.label); postMessage({ type: 'saved', text, tick: w.tick, seed: w.seed }); }
    catch (err) { postMessage({ type: 'saveError', msg: String(err && err.message || err) }); }
    finally { if (release) { gpuHold = null; release(); } }
  }
  else if (m.type === 'load') {
    let release; gpuHold = w.gpu ? new Promise(r => release = r) : null;
    try { await loadWorld(m.text); postMessage({ type: 'loaded', tick: w.tick, seed: w.seed }); }
    catch (err) { postMessage({ type: 'loadError', msg: String(err && err.message || err) }); }
    finally { if (release) { gpuHold = null; release(); } }
  }
};
