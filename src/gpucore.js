// Free Dish on the graphics card. Same rules as core.js, run as WebGPU compute passes.
// Lives in the worker; after each batch of steps the agents are copied back into plain arrays with the same names
// core.js uses, so the page and the stats code don't care which engine is running.
async function createGpuWorld(opt = {}) {
  if (!self.navigator || !navigator.gpu) throw new Error('no WebGPU');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no GPU adapter');
  const G = opt.G || 384, MAX = opt.MAX || 32768;
  const NI = 23, NH = 16, NO = 9, NW = NI * NH + NH * NO; // 512
  const need = MAX * NW * 4;
  const dev = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, Math.max(need, 134217728)), maxBufferSize: Math.min(adapter.limits.maxBufferSize, Math.max(need, 268435456)), maxStorageBuffersPerShaderStage: Math.min(8, adapter.limits.maxStorageBuffersPerShaderStage) } });
  let lost = false; dev.lost.then(() => { lost = true; });
  const Q = dev.queue;
  const C = Object.assign({
    BASE_COST: 0.3, MOVE_COST: 0.14, TURN_COST: 0.04, EAT: 1.4, REPRO: 30, MAX_AGE: 900, FOOD_CAP: 6, KIN_BONUS: 0.08,
    ROOM: 0.13, HUNTERS0: 60, H_MIN: 6, H_REPRO: 45, H_GAIN: 0.7, H_MEAT: 5, H_TRY: 0.15, SHIELD: 1, STING: 1, STING_COST: 0.1, NEURON_COST: 0.002, BRAIN_MUT: 0.03, START_NH: 6, SEASON: 4000, SEASON_AMP: 0.5, ARMOUR: 0.5, PATCHES: 1, PATCH_R: 10, WASTE_MAKE: 0.05, WASTE_EAT: 0.06, WASTE_DECAY: 0.98, WASTE_HARM: 0.06, WASTE_CAP: 4, BOND_K: 0.5, BOND_REST: 1, BOND_BREAK: 2.2, GAP: 0.05, LETGO: 0, GLUE_RATE: 0.1, SIZE_SHIELD: 0.5, KIN_CAP: 6, THRUST: 0.09, TURN: 0.32, DRAG: 0.82, BRAKE_DRAG: 0.55, VMAX: 0.6, PUSH: 0.45, GLUE: 0.09, MUT: 0.08, MUT_SD: 0.4,
  }, opt.C || {});
  const params = { food: 1, mut: 1 };
  const exp = Object.assign({ perks: true, hearing: true, glue: true, memory: true, waste: true, clonal: true, seasons: true, talkInside: true, hunters: true, fightBack: true, adhesion: true, contactBond: true, gap: false, sizeRefuge: false }, opt.exp || {});
  const CX = G / 2, CY = G / 2, DR = G / 2 * 0.955, CELLS = G * G;
  const AREA = CELLS / 36864;
  const BS = 2, HB = Math.ceil(G / BS), BK = 28;  // spatial buckets: 2x2 units, up to 28 creatures each (bodies pack tightly)
  const ST = 28;                                   // agent record: 28 words (20 + family id, attachment count, 6 partner slots)
  const SL = 6, LMAX = 40000;

  // ---------- parameters (one uniform block, generated from this list so JS and WGSL agree) ----------
  const PF = ['G:u', 'MAX:u', 'tick:u', 'seed:u', 'HB:u', 'BK:u', 'nSpawn:u', 'spawnKind:u', 'flags:u', 'popPrev:u', 'mode:u', 'pad0:u',
    'fx:f', 'fy:f', 'fr:f', 'dr:f', 'baseCost:f', 'moveCost:f', 'turnCost:f', 'eat:f', 'repro:f', 'hRepro:f', 'maxAge:f', 'foodCap:f',
    'kinBonus:f', 'thrust:f', 'turn:f', 'drag:f', 'brakeDrag:f', 'vmax:f', 'push:f', 'glueK:f', 'mutP:f', 'mutSd:f', 'wasteMake:f', 'wasteEat:f',
    'wasteDecay:f', 'wasteHarm:f', 'wasteCap:f', 'room:f', 'neuronCost:f', 'brainMut:f', 'seasonLen:f', 'seasonAmp:f', 'armour:f', 'hGain:f', 'hMeat:f', 'hTry:f',
    'shield:f', 'sting:f', 'stingCost:f', 'regen:f', 'startNh:f', 'cells:f', 'bondK:f', 'bondRest:f', 'bondBreak:f', 'gap:f', 'letGo:f', 'glueRate:f', 'sizeShield:f', 'kinCap:f', 'shN:f', 'shM:f', 'shK:f', 'csz:f', 'reach:u'];
  const PIDX = {}; PF.forEach((p, i) => PIDX[p.split(':')[0]] = i);
  const PU = new ArrayBuffer(Math.ceil(PF.length / 4) * 16), PUu = new Uint32Array(PU), PUf = new Float32Array(PU);
  const struct = 'struct P {\n' + PF.map(p => { const [n, t] = p.split(':'); return `  ${n}: ${t === 'u' ? 'u32' : 'f32'},`; }).join('\n') + '\n' +
    Array.from({ length: Math.ceil(PF.length / 4) * 4 - PF.length }, (_, i) => `  padx${i}: u32,`).join('\n') + '\n};';
  const setU = (n, v) => { PUu[PIDX[n]] = v >>> 0; }, setF = (n, v) => { PUf[PIDX[n]] = v; };
  const FLAG = { perks: 1, hearing: 2, glue: 4, memory: 8, waste: 16, clonal: 32, seasons: 64, talkInside: 128, hunters: 256, fightBack: 512, adhesion: 1024, contactBond: 2048, gap: 4096, sizeRefuge: 8192 };

  // ---------- buffers ----------
  const SB = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const buf = (size, usage = SB) => dev.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
  const B = {
    P: buf(PU.byteLength, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    ag: buf(MAX * ST * 4), prv: buf(MAX * ST * 4), W: buf(MAX * NW * 4),
    fd: buf(CELLS * 12),            // food (fixed point) then this step's deposits [signal, waste] per cell
    cap: buf(CELLS * 4), fld: buf(CELLS * 8), tmp: buf(CELLS * 8),   // fld: [signal, waste] per cell
    dek: buf(MAX * 8),              // per creature: [energy change from others, eaten flag]
    bk: buf((HB * HB + HB * HB * BK) * 4),   // bucket counts, then bucket slots
    free: buf(MAX * 4), ctr: buf(64),
    lt: buf(LMAX * 16),            // family table: [parent, founding hue, born step, kind] per family
  };
  const RB = { ag: dev.createBuffer({ size: MAX * ST * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    food: dev.createBuffer({ size: CELLS * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    waste: dev.createBuffer({ size: CELLS * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    ctr: dev.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    W: dev.createBuffer({ size: NW * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
    lt: dev.createBuffer({ size: LMAX * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) };

  const common = /* wgsl */`
${struct}
struct Agent {
  pos: vec2<f32>, vel: vec2<f32>,
  th: f32, energy: f32, hue: f32, glue: f32,
  mem: f32, msg: f32, speak: f32, lastCell: f32,
  age: u32, gen: u32, kind: u32, nh: u32,
  held: u32, alive: u32, kin: u32, lin: i32,
  nb: u32, bsize: u32, bp: array<u32, ${SL}>,
};
const NI = ${NI}u; const NH = ${NH}u; const NO = ${NO}u; const NW = ${NW}u;
@group(0) @binding(0) var<uniform> p: P;
fn hsh(x: u32) -> u32 { var v = x * 747796405u + 2891336453u; v = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u; return (v >> 22u) ^ v; }
fn rnd(a: u32, k: u32) -> f32 { return f32(hsh((a * 9781u) ^ hsh(p.tick * 6271u + k * 31337u + p.seed))) / 4294967296.0; }
fn gauss(a: u32, k: u32) -> f32 { let u = max(rnd(a, k), 1e-7); let v = rnd(a, k + 977u); return sqrt(-2.0 * log(u)) * cos(6.2831853 * v); }
fn on(f: u32) -> bool { return (p.flags & f) != 0u; }
fn sth(x: f32) -> f32 { return tanh(clamp(x, -12.0, 12.0)); }   // some drivers return NaN from tanh of large inputs
fn bad(x: f32) -> bool { return (bitcast<u32>(x) & 0x7f800000u) == 0x7f800000u; }
fn cellOf(x: f32, y: f32) -> u32 { let ix = clamp(i32(floor(x)), 0, i32(p.G) - 1); let iy = clamp(i32(floor(y)), 0, i32(p.G) - 1); return u32(iy) * p.G + u32(ix); }
fn hueDist(a: f32, b: f32) -> f32 { let d = abs(a - b) % 360.0; return select(d, 360.0 - d, d > 180.0); }
`;
  // field pass 1: regrow food (with seasons), spread calls and waste into tmp
  const fieldA = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> fd: array<atomic<i32>>;
@group(0) @binding(2) var<storage, read> cap: array<f32>;
@group(0) @binding(3) var<storage, read> fld: array<f32>;
@group(0) @binding(4) var<storage, read_write> tmp: array<f32>;
fn sg(i: u32) -> f32 { return fld[i * 2u]; }
fn ws(i: u32) -> f32 { return fld[i * 2u + 1u]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.G * p.G) { return; }
  var season = 1.0; if (on(64u)) { season = 1.0 + p.seasonAmp * sin(f32(p.tick) / p.seasonLen * 6.2831853); }
  let f = max(0.0, f32(atomicLoad(&fd[i])) / 1000.0); let t = cap[i] * season;
  atomicStore(&fd[i], i32((f + (t - f) * p.regen) * 1000.0));
  let x = i % p.G; let y = i / p.G;
  if (x == 0u || y == 0u || x == p.G - 1u || y == p.G - 1u || cap[i] <= 0.0) { tmp[i * 2u] = 0.0; tmp[i * 2u + 1u] = 0.0; return; }
  tmp[i * 2u] = (sg(i) * 0.52 + (sg(i - p.G) + sg(i + p.G) + sg(i - 1u) + sg(i + 1u)) * 0.115) * 0.93;
  if (on(16u)) { tmp[i * 2u + 1u] = (ws(i) * 0.8 + (ws(i - p.G) + ws(i + p.G) + ws(i - 1u) + ws(i + 1u)) * 0.05) * p.wasteDecay; }
  else { tmp[i * 2u + 1u] = 0.0; }
}`;
  // field pass 2: write the spread values back and add this step's deposits; also the feed brush
  const fieldB = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> fld: array<f32>;
@group(0) @binding(2) var<storage, read> tmp: array<f32>;
@group(0) @binding(3) var<storage, read_write> fd: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read> cap: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.G * p.G) { return; }
  let D = p.G * p.G;
  fld[i * 2u] = tmp[i * 2u] + f32(atomicExchange(&fd[D + i * 2u], 0)) / 1000.0;
  fld[i * 2u + 1u] = tmp[i * 2u + 1u] + f32(atomicExchange(&fd[D + i * 2u + 1u], 0)) / 1000.0;
  if (p.mode == 1u && cap[i] > 0.0) {
    let d = distance(vec2<f32>(f32(i % p.G) + 0.5, f32(i / p.G) + 0.5), vec2<f32>(p.fx, p.fy));
    if (d < p.fr) { let f = f32(atomicLoad(&fd[i])) / 1000.0; atomicStore(&fd[i], i32(min(p.foodCap * 1.5, f + p.foodCap * (1.0 - d / p.fr) * 1.2) * 1000.0)); }
  }
}`;
  const clearB = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> bk: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < p.HB * p.HB) { atomicStore(&bk[g.x], 0u); } }`;
  const insertB = common + /* wgsl */`
@group(0) @binding(1) var<storage, read> ag: array<Agent>;
@group(0) @binding(2) var<storage, read_write> bk: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> ctr: array<atomic<i32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.MAX || ag[i].alive == 0u) { return; }
  let bx = min(u32(max(ag[i].pos.x, 0.0) / ${BS}.0), p.HB - 1u); let by = min(u32(max(ag[i].pos.y, 0.0) / ${BS}.0), p.HB - 1u);
  let b = by * p.HB + bx; let k = atomicAdd(&bk[b], 1u);
  if (k < p.BK) { atomicStore(&bk[p.HB * p.HB + b * p.BK + k], i); } else { atomicAdd(&ctr[11], 1); }
}`;
  // the creature's turn: sense, think, act
  const think = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> ag: array<Agent>;
@group(0) @binding(2) var<storage, read> prv: array<Agent>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> fd: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read> fld: array<f32>;
@group(0) @binding(6) var<storage, read_write> dek: array<atomic<i32>>;
@group(0) @binding(7) var<storage, read_write> bk: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> ctr: array<atomic<i32>>;
fn sg(i: u32) -> f32 { return fld[i * 2u]; }
fn ws(i: u32) -> f32 { return fld[i * 2u + 1u]; }
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.MAX) { return; }
  var a = prv[i]; if (a.alive == 0u) { return; }
  let x = a.pos.x; let y = a.pos.y; let h = a.th; let kd = a.kind; let c = cellOf(x, y);
  var inp: array<f32, ${NI}>;
  let fNow = f32(atomicLoad(&fd[c])) / 1000.0 / p.foodCap;
  inp[0] = fNow;
  if (on(8u) && a.lastCell >= 0.0) { inp[1] = clamp((f32(atomicLoad(&fd[c])) - f32(atomicLoad(&fd[u32(a.lastCell)]))) / 1000.0 / p.foodCap * 6.0, -1.0, 1.0); }
  if (p.tick % 4u == 0u) { a.lastCell = f32(c); }
  for (var d = 0u; d < 4u; d++) {
    let an = h + f32(d) * 1.5708; let sc = cellOf(x + cos(an) * 1.6 * p.csz, y + sin(an) * 1.6 * p.csz);
    if (on(2u)) { inp[13u + d] = min(1.0, sg(sc) * 2.0); }
  }
  var nb4 = vec4<f32>(0.0); var oth4 = vec4<f32>(0.0);
  var fx = 0.0; var fy = 0.0; var kinN = 0u; var nHeld = 0u; var msgSum = 0.0;
  var prey = -1; var preyD = 9.0; var foe = -1; var foeD = 9.0;
  var kinList: array<i32, 8>; var nKin = 0u;
  let ch = cos(h); let sh = sin(h);
  var de = 0.0;
  if (on(1024u)) {
    // keep only attachments both cells still hold, both still want, and that aren't stretched past breaking
    var keep: array<u32, ${SL}>; var nk = 0u;
    for (var s = 0u; s < min(a.nb, ${SL}u); s++) {
      let j = a.bp[s]; if (j >= p.MAX) { continue; }
      let o = prv[j]; if (o.alive == 0u) { continue; }
      var mutual = false; for (var t = 0u; t < min(o.nb, ${SL}u); t++) { if (o.bp[t] == i) { mutual = true; } }
      if (!mutual) { continue; }
      let dx = o.pos.x - x; let dy = o.pos.y - y; let dd = sqrt(dx * dx + dy * dy) + 1e-6;
      if (a.glue < p.letGo || o.glue < p.letGo || !on(4u) || dd > p.bondBreak) { continue; }
      let g = p.bondK * (dd - p.bondRest); fx += dx / dd * g; fy += dy / dd * g;
      nHeld++; msgSum += o.msg;
      if (on(4096u)) { de -= (a.energy - o.energy) * p.gap; }
      keep[nk] = j; nk++;
    }
    a.nb = nk; for (var s = 0u; s < ${SL}u; s++) { a.bp[s] = select(0xffffffffu, keep[s], s < nk); }
  }
  let rch = i32(p.reach); let bx0 = i32(x / ${BS}.0) - rch; let by0 = i32(y / ${BS}.0) - rch;
  for (var by = by0; by <= by0 + 2 * rch; by++) { for (var bx = bx0; bx <= bx0 + 2 * rch; bx++) {
    if (bx < 0 || by < 0 || bx >= i32(p.HB) || by >= i32(p.HB)) { continue; }
    let b = u32(by) * p.HB + u32(bx); let n = min(atomicLoad(&bk[b]), p.BK);
    for (var s = 0u; s < n; s++) {
      let j = atomicLoad(&bk[p.HB * p.HB + b * p.BK + s]); if (j == i) { continue; }
      let o = prv[j]; let dx = o.pos.x - x; let dy = o.pos.y - y; let d2 = dx * dx + dy * dy; if (d2 >= 4.0 * p.csz * p.csz) { continue; }
      let dr = sqrt(d2) + 1e-6; let dd = dr / p.csz; let same = o.kind == kd; let kin = same && hueDist(o.hue, a.hue) < 25.0;
      let fwd = dx * ch + dy * sh; let side = -dx * sh + dy * ch;
      var q = 0u; if (abs(fwd) >= abs(side)) { q = select(2u, 0u, fwd >= 0.0); } else { q = select(3u, 1u, side >= 0.0); }
      let w = 1.0 - dd / 2.0; if (same) { nb4[q] += select(-w, w, kin); } else { oth4[q] += w; }
      if (dd < 1.0) { let pp = (1.0 - dd) * p.push; fx -= dx / dr * pp; fy -= dy / dr * pp; }
      if (kin && dd < 1.45) { kinN++; if (nKin < 8u) { kinList[nKin] = i32(j); nKin++; } }
      if (kin && on(4u) && dd < 1.95 && a.glue > 0.0 && o.glue > 0.0) {
        let gg = min(a.glue, o.glue) * p.glueK * (dd - 1.0); fx += dx / dr * gg; fy += dy / dr * gg;
        if (on(1024u)) {
          // touching family who both want to stick form a lasting attachment (both sides decide the same way)
          if (on(2048u) && dd < 1.1 && a.nb < ${SL}u && o.nb < ${SL}u) { var have = false; for (var t = 0u; t < a.nb; t++) { if (a.bp[t] == j) { have = true; } } if (!have) { a.bp[a.nb] = j; a.nb++; } }
        } else if (dd < 1.5) { nHeld++; msgSum += o.msg; }
      }
      if (q == 0u && dd < 1.3 && dd < preyD && select(same && !kin, !same, kd == 1u)) { prey = i32(j); preyD = dd; }
      if (kd == 0u && !same && q == 0u && dd < 1.3 && dd < foeD) { foe = i32(j); foeD = dd; }
    }
  } }
  for (var q = 0u; q < 4u; q++) { inp[5u + q] = clamp(nb4[q], -1.0, 1.0); inp[19u + q] = min(1.0, oth4[q]); }
  inp[9] = a.energy / 60.0; inp[10] = rnd(i, 1u) * 2.0 - 1.0; inp[11] = 1.0;
  if (on(2u)) { inp[12] = min(1.0, sg(c) * 2.0); }
  if (on(16u)) { inp[2] = min(1.0, ws(c) / p.wasteCap); inp[3] = min(1.0, ws(cellOf(x + ch * 1.6 * p.csz, y + sh * 1.6 * p.csz)) / p.wasteCap); }
  if (on(8u)) { inp[4] = a.mem; }
  inp[17] = min(1.0, f32(nHeld) / 6.0); if (on(128u) && nHeld > 0u) { inp[18] = msgSum / f32(nHeld); }
  // think
  let wb = i * NW; let nn = min(a.nh, NH);
  var hid: array<f32, ${NH}>; var outp: array<f32, ${NO}>;
  for (var n = 0u; n < nn; n++) { var s = 0.0; let r = wb + n * NI; for (var m = 0u; m < NI; m++) { s += inp[m] * W[r + m]; } hid[n] = sth(s); }
  let ob = wb + NI * NH;
  for (var o = 0u; o < NO; o++) { var s = 0.0; let r = ob + o * NH; for (var n = 0u; n < nn; n++) { s += hid[n] * W[r + n]; } outp[o] = s; }
  // act: 0 swim, 1 turn, 2 stickiness, 3 brake, 4 bite, 5 share, 6 call, 7 remember, 8 signal
  let thrust = max(0.0, sth(outp[0])); let turn = sth(outp[1]) * p.turn; let braking = outp[3] > 0.5;
  a.glue = a.glue + (sth(outp[2]) - a.glue) * p.glueRate; a.mem = sth(outp[7]); a.msg = sth(outp[8]); a.held = nHeld; a.kin = kinN;
  a.th = h + turn;
  var v = a.vel + vec2<f32>(fx, fy);
  // sound mode: the vibration nudges every cell towards the nearest still line, like sand on a plate
  if (p.shK > 0.0) {
    let q = (a.pos - vec2<f32>(f32(p.G) * 0.5)) / p.dr; let n = p.shN * 3.14159265; let m = p.shM * 3.14159265;
    let cnu = cos(n * q.x); let cmw = cos(m * q.y); let cmu = cos(m * q.x); let cnw = cos(n * q.y);
    let aa = cnu * cmw - cmu * cnw;
    let gr = vec2<f32>(-n * sin(n * q.x) * cmw + m * sin(m * q.x) * cnw, -m * cnu * sin(m * q.y) + n * cmu * sin(n * q.y));
    v -= p.shK * clamp(aa, -1.0, 1.0) * gr / (length(gr) + 1e-6);
  }
  if (!braking) { v += vec2<f32>(cos(a.th), sin(a.th)) * thrust * p.thrust; }
  v *= select(p.drag, p.brakeDrag, braking);
  let sp = length(v); if (sp > p.vmax) { v *= p.vmax / sp; }
  var np = a.pos + v;
  let rv = np - vec2<f32>(f32(p.G) * 0.5); let rr = length(rv);
  if (rr > p.dr - 0.5) { np = vec2<f32>(f32(p.G) * 0.5) + rv * ((p.dr - 0.5) / rr); let vr = dot(v, rv) / rr; if (vr > 0.0) { v -= vr * rv / rr * 1.5; } }
  a.pos = np; a.vel = v;
  // metabolism + eating
  let crowd = 0.6 + f32(atomicLoad(&ctr[4])) / (p.cells * p.room);
  de += -(p.baseCost * crowd - select(0.0, min(f32(kinN), p.kinCap) * p.kinBonus, on(1u)) + select(thrust * p.moveCost, 0.0, braking) + abs(turn) * p.turnCost + f32(nn) * p.neuronCost);
  a.age++;
  let ec = cellOf(np.x, np.y);
  var take = 0; if (kd == 0u) { take = min(atomicLoad(&fd[ec]), i32(p.eat * 1000.0)); if (take > 0) { atomicSub(&fd[ec], take); } }
  de += f32(take) / 1000.0;
  let D = p.G * p.G;
  if (on(16u)) { de -= min(ws(ec), p.wasteCap * 2.0) * p.wasteHarm; atomicAdd(&fd[D + ec * 2u + 1u], i32((p.wasteMake + f32(take) / 1000.0 * p.wasteEat) * 1000.0)); }
  // hunting, fighting back, biting strangers
  if (kd == 1u) {
    if (outp[4] > 0.5 && prey >= 0) {
      de -= p.hTry;
      let t = prv[u32(prey)];
      var refuge = 1.0; if (on(8192u)) { refuge = 1.0 + f32(max(t.bsize, 1u) - 1u) * p.sizeShield; }
      if (rnd(i, 7u) < 1.0 / (1.0 + p.shield * f32(t.held)) / refuge) {
        if (atomicCompareExchangeWeak(&dek[u32(prey) * 2u + 1u], 0, 1).exchanged) { de += t.energy * p.hGain + p.hMeat; atomicAdd(&ctr[2], 1); }
      }
    }
  } else if (on(512u) && outp[4] > 0.5 && foe >= 0) {
    de -= p.stingCost; let dmg = p.sting * (1.0 + f32(nHeld));
    atomicAdd(&dek[u32(foe) * 2u], -i32(dmg * 1000.0));
  } else if (outp[4] > 0.5 && prey >= 0) {
    de -= 0.1; let t = prv[u32(prey)];
    var myK = 0.0; var thK = 0.0; if (on(1u)) { myK = f32(min(kinN, 3u)); thK = f32(min(t.kin, 3u)); }
    if (a.energy * (1.0 + myK) > t.energy * (1.0 + thK)) {
      let steal = min(10.0, t.energy * 0.4) / (1.0 + p.armour * f32(t.held));
      // several cells may bite the same victim at once: together they can't take more than it has
      let before = atomicAdd(&dek[u32(prey) * 2u], -i32(steal * 1000.0));
      let got = clamp(t.energy + f32(before) / 1000.0, 0.0, steal); de += got * 0.7;
    }
  }
  // call
  if (outp[6] > 0.3) { let vv = min(1.0, outp[6] - 0.3); atomicAdd(&fd[D + ec * 2u], i32(vv * 600.0)); de -= 0.025 * vv; a.speak = vv; } else { a.speak = 0.0; }
  // share with hungrier family touching it
  if (outp[5] > 0.5 && a.energy > 18.0) { for (var m = 0u; m < nKin; m++) { let j = u32(kinList[m]); if (prv[j].energy < a.energy) { de -= 1.5; atomicAdd(&dek[j * 2u], 1500); } } }
  a.energy += de;
  ag[i] = a;
}`;
  // apply what others did to you, and die
  const death = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> ag: array<Agent>;
@group(0) @binding(2) var<storage, read_write> dek: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> fd: array<atomic<i32>>;
@group(0) @binding(4) var<storage, read_write> freeS: array<u32>;
@group(0) @binding(5) var<storage, read_write> ctr: array<atomic<i32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.MAX) { return; }
  let k = atomicExchange(&dek[i * 2u + 1u], 0); let d = atomicExchange(&dek[i * 2u], 0);
  if (ag[i].alive == 0u) { return; }
  var e = ag[i].energy + f32(d) / 1000.0; ag[i].energy = e;
  let hunterOff = ag[i].kind == 1u && !on(256u);
  let broken = bad(e) || bad(ag[i].pos.x) || bad(ag[i].pos.y) || bad(ag[i].vel.x) || bad(ag[i].vel.y) || bad(ag[i].th);
  if (broken || k != 0 || !(e > 0.0) || f32(ag[i].age) > p.maxAge || hunterOff) {
    if (d < 0 && ag[i].kind == 1u && e <= 0.0) { atomicAdd(&ctr[3], 1); }
    ag[i].alive = 0u;
    let c = cellOf(ag[i].pos.x, ag[i].pos.y);
    atomicAdd(&fd[c], 2500);
    let s = atomicAdd(&ctr[0], 1); freeS[u32(s)] = i;
  } else { atomicAdd(&ctr[5], 1); if (ag[i].kind == 1u) { atomicAdd(&ctr[7], 1); } }
}`;
  // dividing: the daughter takes a free slot and a mutated copy of the brain
  const birth = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> ag: array<Agent>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> freeS: array<u32>;
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> lt: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x; if (i >= p.MAX) { return; }
  var a = ag[i]; if (a.alive == 0u) { return; }
  if (!(a.energy > select(p.repro, p.hRepro, a.kind == 1u)) || bad(a.pos.x) || bad(a.pos.y)) { return; }
  let s = atomicSub(&ctr[0], 1) - 1; if (s < 0) { atomicAdd(&ctr[0], 1); return; }
  let ch = freeS[u32(s)];
  let half = a.energy * 0.45; a.energy = half;
  var c = a;
  let stay = on(32u) && on(4u) && a.glue > 0.0;
  c.nb = 0u; for (var q = 0u; q < ${SL}u; q++) { c.bp[q] = 0xffffffffu; }
  if (stay && on(1024u) && a.nb < ${SL}u) { a.bp[a.nb] = ch; a.nb++; c.nb = 1u; c.bp[0] = i; }
  ag[i] = a;
  let ba = a.th + 3.14159 + (rnd(i, 21u) - 0.5) * 1.6;
  let gap = select(0.55, 0.9, stay) * p.csz; let kick = select(0.15, 0.0, stay);
  c.pos = a.pos + vec2<f32>(cos(ba), sin(ba)) * gap; c.vel = vec2<f32>(cos(ba), sin(ba)) * kick;
  c.th = a.th + (rnd(i, 22u) - 0.5) * 2.0; c.age = 0u; c.gen = a.gen + 1u; c.energy = half; c.speak = 0.0; c.mem = 0.0; c.msg = 0.0; c.held = 0u; c.lastCell = -1.0;
  if (rnd(i, 23u) < p.brainMut * p.mutP) { c.nh = u32(clamp(i32(a.nh) + select(-1, 1, rnd(i, 24u) < 0.5), 1, i32(NH))); }
  var hh = a.hue + gauss(i, 25u) * 2.5 * p.mutP; if (rnd(i, 26u) < 0.003 * p.mutP) { hh = rnd(i, 27u) * 360.0; }
  c.hue = ((hh % 360.0) + 360.0) % 360.0; c.bsize = 1u;
  if (a.lin >= 0 && hueDist(c.hue, lt[u32(a.lin) * 4u + 1u]) >= 30.0) {
    let id = atomicAdd(&ctr[10], 1);
    if (id < ${LMAX}) { let o = u32(id) * 4u; lt[o] = f32(a.lin); lt[o + 1u] = c.hue; lt[o + 2u] = f32(p.tick); lt[o + 3u] = f32(a.kind); c.lin = id; } else { atomicSub(&ctr[10], 1); }
  }
  ag[ch] = c;
  let pb = i * NW; let cb = ch * NW;
  for (var w = 0u; w < NW; w++) { var v = W[pb + w]; if (rnd(i, 100u + w) < p.mutP * ${C.MUT}) { v += gauss(i, 2000u + w) * ${C.MUT_SD}; } W[cb + w] = v; }
  atomicAdd(&ctr[1], 1); atomicAdd(&ctr[6], 1);
  if (a.kind == 1u) { atomicAdd(&ctr[8], 1); }
}`;
  // new random creatures (start of a world, or when a kind dies out)
  const spawn = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> ag: array<Agent>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> freeS: array<u32>;
@group(0) @binding(4) var<storage, read_write> ctr: array<atomic<i32>>;
@group(0) @binding(5) var<storage, read_write> lt: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let j = g.x; if (j >= p.nSpawn) { return; }
  let s = atomicSub(&ctr[0], 1) - 1; if (s < 0) { atomicAdd(&ctr[0], 1); return; }
  let i = freeS[u32(s)]; let key = j + 7919u + p.pad0 * 1000003u;
  var a: Agent;
  let r = sqrt(rnd(key, 1u)) * (p.dr - 1.0); let an = rnd(key, 2u) * 6.2831853;
  a.pos = vec2<f32>(f32(p.G) * 0.5) + vec2<f32>(cos(an), sin(an)) * r; a.vel = vec2<f32>(0.0); a.th = rnd(key, 3u) * 6.2831853;
  a.energy = 30.0; a.hue = rnd(key, 4u) * 360.0; a.glue = 0.0; a.mem = 0.0; a.msg = 0.0; a.speak = 0.0; a.lastCell = -1.0;
  a.age = 0u; a.gen = 0u; a.kind = p.spawnKind; a.nh = u32(p.startNh); a.held = 0u; a.alive = 1u; a.kin = 0u; a.bsize = 1u;
  a.nb = 0u; for (var q = 0u; q < ${SL}u; q++) { a.bp[q] = 0xffffffffu; }
  a.lin = -1; let id = atomicAdd(&ctr[10], 1);
  if (id < ${LMAX}) { let o = u32(id) * 4u; lt[o] = -1.0; lt[o + 1u] = a.hue; lt[o + 2u] = f32(p.tick); lt[o + 3u] = f32(p.spawnKind); a.lin = id; } else { atomicSub(&ctr[10], 1); }
  ag[i] = a;
  let b = i * NW; for (var w = 0u; w < NW; w++) { W[b + w] = gauss(key, 10u + w) * 0.8; }
}`;
  // end of step: remember the population for crowding, reset per-step counters
  const finish = common + /* wgsl */`
@group(0) @binding(1) var<storage, read_write> ctr: array<atomic<i32>>;
@compute @workgroup_size(1) fn main() {
  if (p.MAX == 0u) { return; }
  atomicStore(&ctr[4], atomicLoad(&ctr[5]) + atomicLoad(&ctr[6]));
  atomicStore(&ctr[9], atomicLoad(&ctr[7]) + atomicLoad(&ctr[8]));
  atomicStore(&ctr[5], 0); atomicStore(&ctr[6], 0); atomicStore(&ctr[7], 0); atomicStore(&ctr[8], 0);
}`;
  async function pipe(code, label) {
    const m = dev.createShaderModule({ code, label });
    const info = await m.getCompilationInfo?.();
    if (info) { const errs = info.messages.filter(x => x.type === 'error'); if (errs.length) throw new Error(label + ': ' + errs.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('; ')); }
    return dev.createComputePipeline({ layout: 'auto', compute: { module: m, entryPoint: 'main' }, label });
  }
  const PP = {};
  for (const [k, code] of Object.entries({ fieldA, fieldB, clearB, insertB, think, death, birth, spawn, finish })) PP[k] = await pipe(code, k);
  const bg = (pl, group, list) => dev.createBindGroup({ layout: PP[pl].getBindGroupLayout(group), entries: list.map((b, i) => ({ binding: i, resource: { buffer: b } })) });
  const BG = {
    fieldA: [bg('fieldA', 0, [B.P, B.fd, B.cap, B.fld, B.tmp])],
    fieldB: [bg('fieldB', 0, [B.P, B.fld, B.tmp, B.fd, B.cap])],
    clearB: [bg('clearB', 0, [B.P, B.bk])],
    insertB: [bg('insertB', 0, [B.P, B.ag, B.bk, B.ctr])],
    think: [bg('think', 0, [B.P, B.ag, B.prv, B.W, B.fd, B.fld, B.dek, B.bk, B.ctr])],
    death: [bg('death', 0, [B.P, B.ag, B.dek, B.fd, B.free, B.ctr])],
    birth: [bg('birth', 0, [B.P, B.ag, B.W, B.free, B.ctr, B.lt])],
    spawn: [bg('spawn', 0, [B.P, B.ag, B.W, B.free, B.ctr, B.lt])],
    finish: [bg('finish', 0, [B.P, B.ctr])],
  };
  const run = (e, name, n) => { const ps = e.beginComputePass(); ps.setPipeline(PP[name]); BG[name].forEach((g, k) => ps.setBindGroup(k, g)); ps.dispatchWorkgroups(Math.ceil(n / 64)); ps.end(); };

  // ---------- CPU mirrors (same names as core.js) ----------
  const alive = new Uint8Array(MAX), px = new Float32Array(MAX), py = new Float32Array(MAX), hue = new Float32Array(MAX), glue = new Float32Array(MAX), energy = new Float32Array(MAX);
  const speak = new Float32Array(MAX), kind = new Uint8Array(MAX), gen = new Uint32Array(MAX), age = new Uint32Array(MAX), nh = new Uint8Array(MAX), held = new Uint8Array(MAX), msg = new Float32Array(MAX), mem = new Float32Array(MAX), kinN = new Uint8Array(MAX);
  const lin = new Int32Array(MAX), nbp = new Uint8Array(MAX), bp = new Int32Array(MAX * SL).fill(-1), bsize = new Uint16Array(MAX).fill(1);
  const linPar = new Int32Array(LMAX), linHue = new Float32Array(LMAX), linBorn = new Int32Array(LMAX), linKind = new Uint8Array(LMAX); let nLin = 0, nLinRead = 0;
  const food = new Float32Array(CELLS), waste = new Float32Array(CELLS), cap = new Float32Array(CELLS), wall = new Uint8Array(CELLS);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) if (Math.hypot(x + 0.5 - CX, y + 0.5 - CY) > DR) wall[y * G + x] = 1;
  const BMAX = 120000, bondA = new Int32Array(BMAX), bondB = new Int32Array(BMAX); let nBonds = 0;
  let overflow = 0, pop = 0, tick = 0, births = 0, kills = 0, stung = 0, reintroduced = 0, worldSeed = 0, blobs = [], popPrev = 0;
  const nKind = [0, 0];
  let seed = 1;
  function rand() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
  // sound mode: food grows only along the still lines of a vibrating plate (a Chladni pattern), one pattern per note
  let pattern = null;
  // cell size: every contact distance (push, sense, hold, bite, bonds, birth gap) scales with it; speeds do not
  let cs = 1;
  function buildCap() {
    if (pattern) {
      const n = pattern.n, m = pattern.m, w = 0.5;
      for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
        const u = (x + 0.5 - CX) / DR, v = (y + 0.5 - CY) / DR;
        const a = Math.cos(n * Math.PI * u) * Math.cos(m * Math.PI * v) - Math.cos(m * Math.PI * u) * Math.cos(n * Math.PI * v);
        cap[y * G + x] = wall[y * G + x] ? 0 : Math.min(1, 0.03 + Math.exp(-(a / w) * (a / w))) * C.FOOD_CAP;
      }
      Q.writeBuffer(B.cap, 0, cap);
      return;
    }
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
      let s = 0.04;
      for (const b of blobs) { const dx = x - b.x, dy = y - b.y; s += Math.exp(-(dx * dx + dy * dy) / (2 * b.r * b.r)); }
      cap[y * G + x] = wall[y * G + x] ? 0 : Math.min(1, s) * C.FOOD_CAP;
    }
    Q.writeBuffer(B.cap, 0, cap);
  }
  function driftBlobs() {
    for (const b of blobs) {
      b.vx += (rand() - .5) * 0.004; b.vy += (rand() - .5) * 0.004;
      b.vx = Math.max(-0.05, Math.min(0.05, b.vx)); b.vy = Math.max(-0.05, Math.min(0.05, b.vy));
      b.x += b.vx * 200; b.y += b.vy * 200;
      const r = Math.hypot(b.x - CX, b.y - CY); if (r > DR * 0.85) { b.x = CX + (b.x - CX) / r * DR * 0.85; b.y = CY + (b.y - CY) / r * DR * 0.85; b.vx = -b.vx; b.vy = -b.vy; }
    }
    buildCap();
  }
  function writeParams(mode = 0, extra = {}) {
    let flags = 0; for (const k in FLAG) if (exp[k]) flags |= FLAG[k];
    setU('G', G); setU('MAX', MAX); setU('tick', tick); setU('seed', worldSeed); setU('HB', HB); setU('BK', BK); setU('flags', flags); setU('popPrev', popPrev); setU('mode', mode);
    setU('nSpawn', extra.n || 0); setU('spawnKind', extra.kind || 0);
    setF('fx', extra.x || 0); setF('fy', extra.y || 0); setF('fr', extra.r || 0); setF('dr', DR);
    setF('baseCost', C.BASE_COST); setF('moveCost', C.MOVE_COST); setF('turnCost', C.TURN_COST); setF('eat', C.EAT); setF('repro', C.REPRO); setF('hRepro', C.H_REPRO); setF('maxAge', C.MAX_AGE); setF('foodCap', C.FOOD_CAP);
    setF('kinBonus', C.KIN_BONUS); setF('thrust', C.THRUST); setF('turn', C.TURN); setF('drag', C.DRAG); setF('brakeDrag', C.BRAKE_DRAG); setF('vmax', C.VMAX); setF('push', C.PUSH); setF('glueK', C.GLUE);
    setF('mutP', params.mut); setF('mutSd', C.MUT_SD); setF('wasteMake', C.WASTE_MAKE); setF('wasteEat', C.WASTE_EAT); setF('wasteDecay', C.WASTE_DECAY); setF('wasteHarm', C.WASTE_HARM); setF('wasteCap', C.WASTE_CAP); setF('room', C.ROOM);
    setF('neuronCost', C.NEURON_COST); setF('brainMut', C.BRAIN_MUT); setF('seasonLen', C.SEASON); setF('seasonAmp', C.SEASON_AMP); setF('armour', C.ARMOUR); setF('hGain', C.H_GAIN); setF('hMeat', C.H_MEAT); setF('hTry', C.H_TRY);
    setF('shield', C.SHIELD); setF('sting', C.STING); setF('stingCost', C.STING_COST); setF('regen', 0.012 * params.food); setF('startNh', C.START_NH); setF('cells', CELLS);
    setF('bondK', C.BOND_K); setF('bondRest', C.BOND_REST * cs); setF('bondBreak', C.BOND_BREAK * cs); setF('gap', C.GAP); setF('letGo', C.LETGO); setF('glueRate', C.GLUE_RATE); setF('sizeShield', C.SIZE_SHIELD); setF('csz', cs); setU('reach', Math.ceil(cs)); setF('shN', pattern ? pattern.n : 0); setF('shM', pattern ? pattern.m : 0); setF('shK', pattern ? 0.12 : 0); setF('kinCap', C.KIN_CAP);
    Q.writeBuffer(B.P, 0, PU);
  }
  let spawnCount = 0;
  function spawnN(n, k) { if (n <= 0) return; writeParams(0, { n, kind: k }); setU('pad0', ++spawnCount); Q.writeBuffer(B.P, 0, PU); const e = dev.createCommandEncoder(); run(e, 'spawn', n); Q.submit([e.finish()]); }
  function reset(s) {
    worldSeed = s != null ? s >>> 0 : (Math.random() * 1e9) >>> 0; seed = worldSeed;
    tick = 0; births = 0; kills = 0; stung = 0; reintroduced = 0; nBonds = 0; popPrev = 0;
    blobs = [];
    const nb = Math.max(9, Math.round(9 * C.PATCHES * AREA));
    for (let i = 0; i < nb; i++) { const r = Math.sqrt(rand()) * DR * 0.8, a = rand() * 6.2832; blobs.push({ x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, r: C.PATCH_R + rand() * C.PATCH_R * 1.8, vx: (rand() - .5) * 0.02, vy: (rand() - .5) * 0.02 }); }
    buildCap();
    const f = new Int32Array(CELLS * 3); for (let i = 0; i < CELLS; i++) f[i] = Math.round(cap[i] * 1000); Q.writeBuffer(B.fd, 0, f);
    for (const b of ['ag', 'prv', 'fld', 'tmp', 'dek']) Q.writeBuffer(B[b], 0, new Uint8Array(B[b].size));
    const fr = new Uint32Array(MAX); for (let i = 0; i < MAX; i++) fr[i] = MAX - 1 - i; Q.writeBuffer(B.free, 0, fr);
    const ct = new Int32Array(16); ct[0] = MAX; Q.writeBuffer(B.ctr, 0, ct); nLin = nLinRead = 0;
    spawnN(Math.round(900 * AREA), 0);
    if (exp.hunters) spawnN(Math.round(C.HUNTERS0 * AREA), 1);
    alive.fill(0); pop = 0;
  }
  // one step = a handful of passes; several steps go into one submit
  function encodeStep(e) {
    run(e, 'fieldA', CELLS); run(e, 'fieldB', CELLS);
    run(e, 'clearB', HB * HB); run(e, 'insertB', MAX);
    e.copyBufferToBuffer(B.ag, 0, B.prv, 0, MAX * ST * 4);
    run(e, 'think', MAX); run(e, 'death', MAX); run(e, 'birth', MAX); run(e, 'finish', 1);
  }
  let pendingFeed = null;
  function stepN(n) {
    for (let k = 0; k < n; k++) {
      tick++;
      if (tick % 200 === 0) driftBlobs();
      const feed = pendingFeed; pendingFeed = null;
      writeParams(feed ? 1 : 0, feed || {});
      const e = dev.createCommandEncoder(); encodeStep(e); Q.submit([e.finish()]);
    }
  }
  // copy the creatures (and now and then the jelly) back, then work out bonds for drawing and stats
  let rbCount = 0, busy = false;
  async function readback() {
    if (busy) return false; busy = true;
    try {
      const withFields = rbCount++ % 12 === 0, withLin = withFields && nLin > nLinRead;
      const e = dev.createCommandEncoder();
      e.copyBufferToBuffer(B.ag, 0, RB.ag, 0, MAX * ST * 4); e.copyBufferToBuffer(B.ctr, 0, RB.ctr, 0, 64);
      if (withFields) { e.copyBufferToBuffer(B.fd, 0, RB.food, 0, CELLS * 4); e.copyBufferToBuffer(B.fld, 0, RB.waste, 0, CELLS * 8); }
      const linTo = Math.min(LMAX, nLin);
      if (withLin) e.copyBufferToBuffer(B.lt, 0, RB.lt, 0, linTo * 16);
      Q.submit([e.finish()]);
      const maps = [RB.ag.mapAsync(GPUMapMode.READ), RB.ctr.mapAsync(GPUMapMode.READ)];
      if (withFields) maps.push(RB.food.mapAsync(GPUMapMode.READ), RB.waste.mapAsync(GPUMapMode.READ));
      if (withLin) maps.push(RB.lt.mapAsync(GPUMapMode.READ, 0, linTo * 16));
      await Promise.all(maps);
      if (withLin) { const L = new Float32Array(RB.lt.getMappedRange(0, linTo * 16)); for (let k = nLinRead; k < linTo; k++) { linPar[k] = L[k * 4]; linHue[k] = L[k * 4 + 1]; linBorn[k] = L[k * 4 + 2]; linKind[k] = L[k * 4 + 3]; } RB.lt.unmap(); nLinRead = linTo; }
      const u = new Uint32Array(RB.ag.getMappedRange()), f = new Float32Array(u.buffer), si = new Int32Array(u.buffer);
      let n = 0; nKind[0] = nKind[1] = 0;
      for (let i = 0; i < MAX; i++) {
        const o = i * ST, al = u[o + 17];
        alive[i] = al; if (!al) continue;
        px[i] = f[o]; py[i] = f[o + 1]; energy[i] = f[o + 5]; hue[i] = f[o + 6]; glue[i] = f[o + 7]; mem[i] = f[o + 8]; msg[i] = f[o + 9]; speak[i] = f[o + 10];
        age[i] = u[o + 12]; gen[i] = u[o + 13]; kind[i] = u[o + 14]; nh[i] = u[o + 15]; held[i] = u[o + 16]; kinN[i] = u[o + 18]; lin[i] = si[o + 19];
        const nb = Math.min(SL, u[o + 20]); nbp[i] = nb; for (let q = 0; q < SL; q++) bp[i * SL + q] = q < nb ? u[o + 22 + q] : -1;
        n++; nKind[kind[i]]++;
      }
      RB.ag.unmap();
      const ct = new Int32Array(RB.ctr.getMappedRange()); births = ct[1]; kills = ct[2]; stung = ct[3]; popPrev = Math.max(0, ct[4]); nLin = Math.min(LMAX, ct[10]); overflow = ct[11]; RB.ctr.unmap();
      if (withFields) {
        const fi = new Int32Array(RB.food.getMappedRange()); for (let i = 0; i < CELLS; i++) food[i] = fi[i] / 1000; RB.food.unmap();
        const wf = new Float32Array(RB.waste.getMappedRange()); for (let i = 0; i < CELLS; i++) waste[i] = wf[i * 2 + 1]; RB.waste.unmap();
      }
      pop = n;
      // keep both kinds alive (counted, like the CPU version)
      if (nKind[0] < 40) spawnN(60, 0);
      if (exp.hunters && nKind[1] < C.H_MIN) { spawnN(C.H_MIN * 2, 1); reintroduced++; }
      findBonds();
      return true;
    } finally { busy = false; }
  }
  // bonds: family members within 1.5 who both choose to stick (same rule as the simulation)
  const HS = 2, HG2 = Math.ceil(G / HS), head = new Int32Array(HG2 * HG2), nxt = new Int32Array(MAX);
  const hd = (a, b) => { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; };
  function findBonds() {
    nBonds = 0;
    if (exp.adhesion) {
      for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; for (let s = 0; s < nbp[i]; s++) { const j = bp[i * SL + s]; if (j > i && j < MAX && alive[j] && nBonds < BMAX) { let m = false; for (let t = 0; t < nbp[j]; t++) if (bp[j * SL + t] === i) m = true; if (m) { bondA[nBonds] = i; bondB[nBonds] = j; nBonds++; } } } }
      return;
    }
    head.fill(-1);
    for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; const h = Math.min(HG2 - 1, Math.max(0, (py[i] / HS) | 0)) * HG2 + Math.min(HG2 - 1, Math.max(0, (px[i] / HS) | 0)); nxt[i] = head[h]; head[h] = i; }
    for (let i = 0; i < MAX; i++) {
      if (!alive[i] || glue[i] <= 0 || !exp.glue) continue;
      const hx = Math.min(HG2 - 1, Math.max(0, (px[i] / HS) | 0)), hy = Math.min(HG2 - 1, Math.max(0, (py[i] / HS) | 0));
      for (let y = Math.max(0, hy - 1); y <= Math.min(HG2 - 1, hy + 1); y++) for (let x = Math.max(0, hx - 1); x <= Math.min(HG2 - 1, hx + 1); x++)
        for (let j = head[y * HG2 + x]; j >= 0; j = nxt[j]) {
          if (j <= i || glue[j] <= 0 || kind[j] !== kind[i] || hd(hue[i], hue[j]) >= 25) continue;
          const dx = px[j] - px[i], dy = py[j] - py[i]; if (dx * dx + dy * dy < 2.25 * cs * cs && nBonds < BMAX) { bondA[nBonds] = i; bondB[nBonds] = j; nBonds++; }
        }
    }
  }
  const uf = new Int32Array(MAX), orgSize = new Int32Array(MAX), orgOf = new Int32Array(MAX);
  const find = a => { while (uf[a] !== a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
  function organisms() {
    for (let i = 0; i < MAX; i++) { uf[i] = i; orgSize[i] = 0; }
    for (let k = 0; k < nBonds; k++) { const ra = find(bondA[k]), rb = find(bondB[k]); if (ra !== rb) uf[ra] = rb; }
    let n = 0, inOrg = 0, count = 0, largest = 0;
    for (let i = 0; i < MAX; i++) if (alive[i]) { n++; orgSize[find(i)]++; }
    for (let i = 0; i < MAX; i++) if (alive[i]) { const sz = orgSize[find(i)]; orgOf[i] = sz > 1 ? find(i) : -1; if (sz > 1) inOrg++; if (uf[i] === i && sz > 1) { count++; if (sz > largest) largest = sz; } }
    return { n, inOrg, count, largest, pctIn: n ? inOrg / n : 0, meanSize: count ? inOrg / count : 0 };
  }
  function stats() {
    let n = 0, maxGen = 0, gl = 0, sumNh = 0, maxNh = 0; const nhK = [0, 0];
    const fam = new Uint16Array(36);
    for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; n++; if (gen[i] > maxGen) maxGen = gen[i]; if (glue[i] > 0) gl++; sumNh += nh[i]; if (nh[i] > maxNh) maxNh = nh[i]; nhK[kind[i]] += nh[i]; if (kind[i] === 0) fam[Math.min(35, (hue[i] / 10) | 0)]++; }
    let families = 0; for (const f of fam) if (f >= Math.max(10, n * 0.02)) families++;
    return { brainG: nKind[0] ? nhK[0] / nKind[0] : 0, brainH: nKind[1] ? nhK[1] / nKind[1] : 0, hunters: nKind[1], grazers: nKind[0], kills, stung, reintroduced, brain: n ? sumNh / n : 0, maxBrain: maxNh, season: exp.seasons ? Math.sin(tick / C.SEASON * 6.2832) : 0, pop: n, tick, births, maxGen, sticky: n ? gl / n : 0, families, bonds: nBonds, gpu: true };
  }
  // save / restore: copy the whole state off the card (only living creatures' records and brains are kept)
  async function readBuf(src, size) { const rb = dev.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); const e = dev.createCommandEncoder(); e.copyBufferToBuffer(src, 0, rb, 0, size); Q.submit([e.finish()]); await rb.mapAsync(GPUMapMode.READ); const out = new Uint8Array(rb.getMappedRange()).slice(); rb.unmap(); rb.destroy(); return out.buffer; }
  async function snapshot() {
    await Q.onSubmittedWorkDone();
    const agU = new Uint32Array(await readBuf(B.ag, MAX * ST * 4)), Wf = new Float32Array(await readBuf(B.W, MAX * NW * 4));
    const idx = []; for (let i = 0; i < MAX; i++) if (agU[i * ST + 17]) idx.push(i);
    const n = idx.length, A = new Uint32Array(n * ST), Wk = new Float32Array(n * NW);
    idx.forEach((i, j) => { A.set(agU.subarray(i * ST, (i + 1) * ST), j * ST); Wk.set(Wf.subarray(i * NW, (i + 1) * NW), j * NW); });
    const ctr = new Int32Array(await readBuf(B.ctr, 64)), nl = Math.min(LMAX, Math.max(0, ctr[10]));
    return { meta: { engine: 'gpu', G, MAX, tick, seed, worldSeed, births, kills, stung, reintroduced, popPrev, spawnCount, blobs, params: Object.assign({}, params), exp: Object.assign({}, exp) },
      arrays: { idx: Int32Array.from(idx), ag: A, W: Wk, fd: new Int32Array(await readBuf(B.fd, CELLS * 12)), fld: new Float32Array(await readBuf(B.fld, CELLS * 8)), free: new Uint32Array(await readBuf(B.free, MAX * 4)), ctr, lt: new Float32Array((await readBuf(B.lt, LMAX * 16)).slice(0, Math.max(16, nl * 16))) } };
  }
  async function restore(snap) {
    const m = snap.meta, A = snap.arrays;
    if (m.engine !== 'gpu' || m.G !== G || m.MAX > MAX) throw new Error('This save is from a different world size.');
    await Q.onSubmittedWorkDone();
    tick = m.tick; seed = m.seed; worldSeed = m.worldSeed; births = m.births; kills = m.kills; stung = m.stung; reintroduced = m.reintroduced; popPrev = m.popPrev; spawnCount = m.spawnCount || 0;
    blobs = m.blobs.map(b => Object.assign({}, b)); Object.assign(params, m.params); Object.assign(exp, m.exp); buildCap();
    const agU = new Uint32Array(MAX * ST), Wf = new Float32Array(MAX * NW), I = A.idx;
    for (let j = 0; j < I.length; j++) { agU.set(A.ag.subarray(j * ST, (j + 1) * ST), I[j] * ST); Wf.set(A.W.subarray(j * NW, (j + 1) * NW), I[j] * NW); }
    Q.writeBuffer(B.ag, 0, agU); Q.writeBuffer(B.prv, 0, agU); Q.writeBuffer(B.W, 0, Wf); Q.writeBuffer(B.fd, 0, A.fd); Q.writeBuffer(B.fld, 0, A.fld); Q.writeBuffer(B.free, 0, A.free); Q.writeBuffer(B.ctr, 0, A.ctr);
    Q.writeBuffer(B.tmp, 0, new Uint8Array(B.tmp.size)); Q.writeBuffer(B.dek, 0, new Uint8Array(B.dek.size));
    const lt = new Float32Array(LMAX * 4); lt.set(A.lt.subarray(0, Math.min(A.lt.length, LMAX * 4))); Q.writeBuffer(B.lt, 0, lt);
    nLin = nLinRead = 0; rbCount = 0; await Q.onSubmittedWorkDone(); await readback();
  }
  async function brainOf(i) {
    const e = dev.createCommandEncoder(); e.copyBufferToBuffer(B.W, i * NW * 4, RB.W, 0, NW * 4); Q.submit([e.finish()]);
    await RB.W.mapAsync(GPUMapMode.READ); const w = new Float32Array(RB.W.getMappedRange()).slice(); RB.W.unmap(); return w;
  }
  // a new note clears food that is no longer on the pattern (using the last copy of the jelly; close enough)
  function setPattern(p) {
    pattern = p && p.n > 0 && p.m > 0 && p.n !== p.m ? { n: p.n | 0, m: p.m | 0 } : null; buildCap();
    if (pattern) { const f = new Int32Array(CELLS); for (let i = 0; i < CELLS; i++) { const v = Math.min(food[i], cap[i]); food[i] = v; f[i] = Math.round(v * 1000); } Q.writeBuffer(B.fd, 0, f); }
  }
  function setCellSize(v) { cs = Math.max(0.5, Math.min(2.5, +v || 1)); }
  return {
    setPattern, get pattern() { return pattern; }, setCellSize, get cellSize() { return cs; },
    gpu: true, adapterName: (adapter.info && (adapter.info.description || adapter.info.vendor)) || '',
    G, MAX, NI, NH, NO, NW, C, CX, CY, DR, params, exp,
    alive, px, py, hue, glue, energy, speak, kind, gen, age, nh, held, msg, mem, food, waste, cap, wall, lin, linPar, linHue, linBorn, linKind, bsize,
    get nLin() { return nLinRead; }, snapshot, restore,
    get pop() { return pop; }, get overflow() { return overflow; }, kinN, get tick() { return tick; }, get seed() { return worldSeed; }, get nBonds() { return nBonds; }, get lost() { return lost; },
    bondA, bondB, orgOf, orgSize, find,
    reset, stepN, readback, stats, organisms, brainOf,
    step() { stepN(1); },
    feed(x, y, r) { pendingFeed = { x, y, r }; },
    async done() { await Q.onSubmittedWorkDone(); },
  };
}
