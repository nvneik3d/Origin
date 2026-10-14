// Free Dish core: the Primordial Dish rules without the grid.
// Creatures have a real position, a heading and a velocity. They swim, push each other apart,
// can choose to stick to family, and can bite strangers. Same brain shape: 17 senses -> 8 neurons -> 7 outputs.
function createWorld(opt = {}) {
  const G = opt.G || 224;                 // food / signal field resolution (1 unit = one creature's width)
  const MAX = opt.MAX || 9000;
  // the brain can grow: room for 16 neurons, each creature uses its own number of them (a gene)
  const NI = 23, NH = 16, NO = 9, NW = NI * NH + NH * NO;
  const C = Object.assign({
    BASE_COST: 0.3, MOVE_COST: 0.14, TURN_COST: 0.04, EAT: 1.4, REPRO: 30, MAX_AGE: 900, FOOD_CAP: 6, KIN_BONUS: 0.08,
    ROOM: 0.13, HUNTERS0: 60, H_MIN: 6, H_REPRO: 45, H_GAIN: 0.7, H_MEAT: 5, H_TRY: 0.15, SHIELD: 1, STING: 1, STING_COST: 0.1, NEURON_COST: 0.002, BRAIN_MUT: 0.03, START_NH: 6, SEASON: 4000, SEASON_AMP: 0.5, ARMOUR: 0.5, PATCHES: 1, PATCH_R: 10, WASTE_MAKE: 0.05, WASTE_EAT: 0.06, WASTE_DECAY: 0.98, WASTE_HARM: 0.06, WASTE_CAP: 4, BOND_K: 0.5, BOND_REST: 1, BOND_BREAK: 2.2, GAP: 0.05, LETGO: 0, ALIGN: 0.2, GLUE_RATE: 0.1, SIZE_SHIELD: 0.5, KIN_CAP: 6, H_FLAT: 0, THRUST: 0.09, TURN: 0.32, DRAG: 0.82, BRAKE_DRAG: 0.55, VMAX: 0.6, PUSH: 0.45, GLUE: 0.09, MUT: 0.08, MUT_SD: 0.4,
  }, opt.C || {});
  const params = { food: 1, mut: 1 };
  const exp = { perks: true, hearing: true, glue: true, memory: true, waste: true, clonal: true, seasons: true, talkInside: true, hunters: true, fightBack: true, adhesion: true, contactBond: true, gap: false, align: false, sizeRefuge: false };
  const CX = G / 2, CY = G / 2, DR = G / 2 * 0.955;
  const CELLS = G * G;

  let seed = 1;
  function rand() { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }
  function gauss() { let u = 0; while (u === 0) u = rand(); const v = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v); }

  // fields
  const wall = new Uint8Array(CELLS);
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) if (Math.hypot(x + 0.5 - CX, y + 0.5 - CY) > DR) wall[y * G + x] = 1;
  const food = new Float32Array(CELLS), cap = new Float32Array(CELLS);
  let signal = new Float32Array(CELLS), signalB = new Float32Array(CELLS);
  // waste: every living thing leaks it; it spreads slowly, breaks down, and poisons whoever sits in it
  let waste = new Float32Array(CELLS), wasteB = new Float32Array(CELLS);
  // creatures
  const alive = new Uint8Array(MAX), px = new Float32Array(MAX), py = new Float32Array(MAX), vx = new Float32Array(MAX), vy = new Float32Array(MAX), th = new Float32Array(MAX);
  const energy = new Float32Array(MAX), hue = new Float32Array(MAX), age = new Uint32Array(MAX), gen = new Uint32Array(MAX);
  const glue = new Float32Array(MAX), speak = new Float32Array(MAX), bonds = new Uint8Array(MAX), lastOut = new Float32Array(MAX * NO);
  // memory: one number each creature writes every step and reads back the next; plus the food it tasted last step
  const mem = new Float32Array(MAX), lastFood = new Float32Array(MAX);
  // nh: how many neurons this creature has; msg: the signal it passes to the cells it holds; held: how many it holds
  // kind: 0 = grazer (eats the jelly's food), 1 = hunter (eats other cells)
  const kind = new Uint8Array(MAX), nKind = [0, 0]; let kills = 0, reintroduced = 0, stung = 0;
  const nh = new Uint8Array(MAX), msg = new Float32Array(MAX), held = new Uint8Array(MAX), bsize = new Uint16Array(MAX).fill(1);
  const W = new Float32Array(MAX * NW);
  // family tree: every cell carries a family id; a daughter whose colour has drifted 30° from her family's
  // founding colour starts a new branch. The tree is the record of which families split from which.
  const LMAX = 40000, lin = new Int32Array(MAX), linPar = new Int32Array(LMAX), linHue = new Float32Array(LMAX), linBorn = new Int32Array(LMAX), linKind = new Uint8Array(LMAX); let nLin = 0;
  function newLin(par, h, k) { if (nLin >= LMAX) return par; const id = nLin++; linPar[id] = par; linHue[id] = h; linBorn[id] = tick; linKind[id] = k; return id; }
  const freeList = new Int32Array(MAX); let freeTop = 0;
  // spatial hash: 3-unit buckets, rebuilt every step
  const S = 2, HG = Math.ceil(G / S), head = new Int32Array(HG * HG), next = new Int32Array(MAX);
  // pairs of family members holding on to each other this step (for drawing)
  const BMAX = 40000, bondA = new Int32Array(BMAX), bondB = new Int32Array(BMAX); let nBonds = 0;
  // lasting attachments (adhesion): each cell can hold up to 6 others with a spring that stays until
  // a cell lets go, the spring is stretched past breaking, or a cell dies
  const SL = 6, bp = new Int32Array(MAX * SL).fill(-1), nbp = new Uint8Array(MAX);
  const bondLog = { clonal: 0, contact: 0, letGo: 0, snapped: 0, died: 0 };
  function linked(a, b) { const o = a * SL; for (let s = 0; s < nbp[a]; s++) if (bp[o + s] === b) return true; return false; }
  function link(a, b) { if (a === b || nbp[a] >= SL || nbp[b] >= SL || linked(a, b)) return false; bp[a * SL + nbp[a]++] = b; bp[b * SL + nbp[b]++] = a; return true; }
  function dropSlot(a, s) { const o = a * SL, last = --nbp[a]; bp[o + s] = bp[o + last]; bp[o + last] = -1; }
  function unlink(a, s) { const b = bp[a * SL + s]; dropSlot(a, s); const o = b * SL; for (let t = 0; t < nbp[b]; t++) if (bp[o + t] === a) { dropSlot(b, t); break; } }

  let pop = 0, tick = 0, births = 0, blobs = [], worldSeed = 0;
  const inp = new Float32Array(NI), hid = new Float32Array(NH), out = new Float32Array(NO);
  const nb4 = new Float32Array(4), oth4 = new Float32Array(4), kinList = new Int32Array(8);
  const hueDist = (a, b) => { const d = Math.abs(a - b) % 360; return d > 180 ? 360 - d : d; };
  const gi = (x, y) => { let ix = x | 0, iy = y | 0; if (ix < 0) ix = 0; else if (ix >= G) ix = G - 1; if (iy < 0) iy = 0; else if (iy >= G) iy = G - 1; return iy * G + ix; };

  // sound mode: food grows only along the still lines of a vibrating plate (a Chladni pattern), one pattern per note
  let pattern = null; const SHAKE = 0.12;
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
      return;
    }
    for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
      let s = 0.04;
      for (const b of blobs) { const dx = x - b.x, dy = y - b.y; s += Math.exp(-(dx * dx + dy * dy) / (2 * b.r * b.r)); }
      cap[y * G + x] = wall[y * G + x] ? 0 : Math.min(1, s) * C.FOOD_CAP;
    }
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
  function alloc() { return freeTop ? freeList[--freeTop] : -1; }
  function spawnRandom(k = 0) {
    const r = Math.sqrt(rand()) * (DR - 1), a = rand() * 6.2832;
    const i = alloc(); if (i < 0) return -1;
    kind[i] = k; nKind[k]++;
    alive[i] = 1; px[i] = CX + Math.cos(a) * r; py[i] = CY + Math.sin(a) * r; vx[i] = vy[i] = 0; th[i] = rand() * 6.2832;
    energy[i] = 30; hue[i] = rand() * 360; age[i] = 0; gen[i] = 0; glue[i] = 0; speak[i] = 0; mem[i] = 0; lastFood[i] = -1; nh[i] = C.START_NH; msg[i] = 0; held[i] = 0;
    const b = i * NW; for (let k = 0; k < NW; k++) W[b + k] = gauss() * 0.8;
    lin[i] = nLin < LMAX ? newLin(-1, hue[i], kind[i]) : 0;
    pop++; return i;
  }
  function kill(i) {
    alive[i] = 0; pop--; nKind[kind[i]]--; freeList[freeTop++] = i;
    while (nbp[i]) { unlink(i, nbp[i] - 1); bondLog.died++; }
    const c = gi(px[i], py[i]); food[c] = Math.min(C.FOOD_CAP, food[c] + 2.5);
  }
  function reset(s, n) {
    worldSeed = s != null ? s >>> 0 : (Math.random() * 1e9) >>> 0; seed = worldSeed;
    alive.fill(0); signal.fill(0); signalB.fill(0); waste.fill(0); wasteB.fill(0); pop = 0; nKind[0] = nKind[1] = 0; kills = 0; reintroduced = 0; stung = 0; tick = 0; births = 0; nLin = 0; nBonds = 0; bp.fill(-1); nbp.fill(0); for (const k in bondLog) bondLog[k] = 0;
    freeTop = 0; for (let i = MAX - 1; i >= 0; i--) freeList[freeTop++] = i;
    blobs = [];
    const nb = Math.max(9, Math.round(9 * C.PATCHES * CELLS / 36864));
    for (let i = 0; i < nb; i++) { const r = Math.sqrt(rand()) * DR * 0.8, a = rand() * 6.2832; blobs.push({ x: CX + Math.cos(a) * r, y: CY + Math.sin(a) * r, r: C.PATCH_R + rand() * C.PATCH_R * 1.8, vx: (rand() - .5) * 0.02, vy: (rand() - .5) * 0.02 }); }
    buildCap(); food.set(cap);
    const start = n || Math.round(900 * CELLS / 36864);
    for (let i = 0; i < start; i++) spawnRandom(0);
    if (exp.hunters) for (let i = 0; i < Math.round(C.HUNTERS0 * CELLS / 36864); i++) spawnRandom(1);
  }

  function rebuildHash() {
    head.fill(-1);
    for (let i = 0; i < MAX; i++) {
      if (!alive[i]) continue;
      let hx = (px[i] / S) | 0, hy = (py[i] / S) | 0; if (hx < 0) hx = 0; else if (hx >= HG) hx = HG - 1; if (hy < 0) hy = 0; else if (hy >= HG) hy = HG - 1;
      const h = hy * HG + hx; next[i] = head[h]; head[h] = i;
    }
  }

  function step() {
    tick++;
    const regen = 0.012 * params.food;
    // seasons: food grows back fast in summer and barely at all in winter
    const season = exp.seasons ? 1 + C.SEASON_AMP * Math.sin(tick / C.SEASON * 6.2832) : 1;
    for (let i = 0; i < CELLS; i++) { const f = food[i], t = cap[i] * season; food[i] = f + (t - f) * regen; }
    for (let y = 1; y < G - 1; y++) {
      const row = y * G;
      for (let x = 1; x < G - 1; x++) {
        const i = row + x; if (wall[i]) { signalB[i] = 0; continue; }
        signalB[i] = (signal[i] * 0.52 + (signal[i - G] + signal[i + G] + signal[i - 1] + signal[i + 1]) * 0.115) * 0.93;
      }
    }
    { const t = signal; signal = signalB; signalB = t; }
    if (exp.waste) {
      const dk = C.WASTE_DECAY;
      for (let y = 1; y < G - 1; y++) {
        const row = y * G;
        for (let x = 1; x < G - 1; x++) {
          const i = row + x; if (wall[i]) { wasteB[i] = 0; continue; }
          wasteB[i] = (waste[i] * 0.8 + (waste[i - G] + waste[i + G] + waste[i - 1] + waste[i + 1]) * 0.05) * dk;
        }
      }
      const t = waste; waste = wasteB; wasteB = t;
    }
    if (tick % 200 === 0) driftBlobs();
    rebuildHash();
    nBonds = 0;
    const crowd = 0.6 + pop / (CELLS * C.ROOM);
    const start = (rand() * MAX) | 0;
    for (let k = 0; k < MAX; k++) {
      const i = (start + k) % MAX;
      if (!alive[i]) continue;
      const x = px[i], y = py[i], h = th[i], hu = hue[i], c = gi(x, y);

      // ---- sense ----
      // Like a real bacterium it can't smell left from right: it tastes the food where it is, and remembers
      // what it tasted a step ago, so it knows whether things are getting better or worse.
      //  0 food here   1 food change   2 waste here   3 waste ahead   4 its own memory
      //  5-8 family (+) or strangers (-) in front/right/behind/left   9 energy   10 noise   11 bias
      //  12 calls here   13-16 calls in front/right/behind/left
      const fNow = food[c] / C.FOOD_CAP;
      inp[0] = fNow;
      // compare here with where it was a moment ago (both tasted now, so its own eating doesn't fool it)
      inp[1] = exp.memory && lastFood[i] >= 0 ? Math.max(-1, Math.min(1, (food[c] - food[lastFood[i] | 0]) / C.FOOD_CAP * 6)) : 0;
      if (tick % 4 === 0) lastFood[i] = c;
      for (let d = 0; d < 4; d++) {
        const a = h + d * 1.5708, sc = gi(x + Math.cos(a) * 1.6 * cs, y + Math.sin(a) * 1.6 * cs);
        inp[13 + d] = exp.hearing ? Math.min(1, signal[sc] * 2) : 0;
      }
      nb4[0] = nb4[1] = nb4[2] = nb4[3] = 0;
      let fx = 0, fy = 0, kinN = 0, nKin = 0, prey = -1, preyD = 9, nHeld = 0, msgSum = 0, foe = -1, foeD = 9;
      const kd = kind[i]; oth4[0] = oth4[1] = oth4[2] = oth4[3] = 0;
      const RN = Math.ceil(cs), hx0 = Math.max(0, ((x / S) | 0) - RN), hx1 = Math.min(HG - 1, ((x / S) | 0) + RN), hy0 = Math.max(0, ((y / S) | 0) - RN), hy1 = Math.min(HG - 1, ((y / S) | 0) + RN);
      const ch = Math.cos(h), sh = Math.sin(h), gi_ = glue[i];
      for (let hy = hy0; hy <= hy1; hy++) for (let hx = hx0; hx <= hx1; hx++) {
        for (let j = head[hy * HG + hx]; j >= 0; j = next[j]) {
          if (j === i) continue;
          const dx = px[j] - x, dy = py[j] - y, d2 = dx * dx + dy * dy; if (d2 >= 4 * cs * cs) continue;
          const dr = Math.sqrt(d2) + 1e-6, d = dr / cs, same = kind[j] === kd, kin = same && hueDist(hue[j], hu) < 25;
          // which side is it on, relative to where I'm facing?
          const fwd = dx * ch + dy * sh, side = -dx * sh + dy * ch;
          const q = Math.abs(fwd) >= Math.abs(side) ? (fwd >= 0 ? 0 : 2) : (side >= 0 ? 1 : 3);
          const w = 1 - d / 2; if (same) nb4[q] += kin ? w : -w; else oth4[q] += w;
          if (d < 1) { const p = (1 - d) * C.PUSH; fx -= dx / dr * p; fy -= dy / dr * p; }  // bodies can't overlap
          if (kin && d < 1.45) { kinN++; if (nKin < 8) kinList[nKin++] = j; }
          // holding on: two family members who both choose to be sticky pull together
          if (kin && exp.glue && d < 1.95 && gi_ > 0 && glue[j] > 0) {
            const g = Math.min(gi_, glue[j]) * C.GLUE * (d - 1); fx += dx / dr * g; fy += dy / dr * g;
            if (exp.adhesion) { if (exp.contactBond && d < 1.1 && link(i, j)) bondLog.contact++; }
            else if (d < 1.5) { nHeld++; msgSum += msg[j]; if (j > i && nBonds < BMAX) { bondA[nBonds] = i; bondB[nBonds] = j; nBonds++; } }
          }
          // grazers bite strangers of their own kind; hunters go for grazers
          if (q === 0 && d < 1.3 && d < preyD && (kd === 1 ? !same : (same && !kin))) { prey = j; preyD = d; }
          if (kd === 0 && !same && q === 0 && d < 1.3 && d < foeD) { foe = j; foeD = d; } // a hunter right in front
        }
      }
      if (exp.adhesion) {
        // the cells it is attached to: springs pull it back to its partners; food flows along the joins
        const o = i * SL; let ax = 0, ay = 0;
        for (let s = nbp[i] - 1; s >= 0; s--) {
          const j = bp[o + s], dx = px[j] - x, dy = py[j] - y, d = Math.sqrt(dx * dx + dy * dy) + 1e-6;
          if (gi_ < C.LETGO || glue[j] < C.LETGO || !exp.glue) { unlink(i, s); bondLog.letGo++; continue; }
          if (d > C.BOND_BREAK * cs) { unlink(i, s); bondLog.snapped++; continue; }
          const g = C.BOND_K * (d - C.BOND_REST * cs); fx += dx / d * g; fy += dy / d * g;
          nHeld++; msgSum += msg[j]; ax += Math.cos(th[j]); ay += Math.sin(th[j]);
          if (exp.gap) { const t = (energy[i] - energy[j]) * C.GAP; energy[i] -= t; energy[j] += t; }
        }
        // attached cells slowly turn to face the same way as their partners, so a body can swim as one
        if (exp.align && nHeld) { let dA = Math.atan2(ay, ax) - h; dA = Math.atan2(Math.sin(dA), Math.cos(dA)); th[i] = h + dA * C.ALIGN; }
      }
      for (let q = 0; q < 4; q++) { inp[5 + q] = Math.max(-1, Math.min(1, nb4[q])); inp[19 + q] = Math.min(1, oth4[q]); } // 19-22: the other kind, front/right/behind/left
      inp[9] = energy[i] / 60; inp[10] = rand() * 2 - 1; inp[11] = 1; inp[12] = exp.hearing ? Math.min(1, signal[c] * 2) : 0;
      inp[2] = exp.waste ? Math.min(1, waste[c] / C.WASTE_CAP) : 0;
      inp[3] = exp.waste ? Math.min(1, waste[gi(x + ch * 1.6 * cs, y + sh * 1.6 * cs)] / C.WASTE_CAP) : 0;
      inp[4] = exp.memory ? mem[i] : 0;
      // inside a body: how many cells it is holding, and what they are signalling to it
      inp[17] = Math.min(1, nHeld / 6); inp[18] = exp.talkInside && nHeld ? msgSum / nHeld : 0; held[i] = nHeld;

      // ---- think ----
      const b = i * NW;
      const nn = nh[i];
      for (let n = 0; n < nn; n++) { let s = 0; const r = b + n * NI; for (let m = 0; m < NI; m++) s += inp[m] * W[r + m]; hid[n] = Math.tanh(s); }
      const ob = b + NI * NH, lo = i * NO;
      for (let o = 0; o < NO; o++) { let s = 0; const r = ob + o * NH; for (let n = 0; n < nn; n++) s += hid[n] * W[r + n]; out[o] = s; lastOut[lo + o] = s; }

      // ---- act: 0 swim, 1 turn, 2 stickiness, 3 brake, 4 bite, 5 share, 6 call ----
      const thrust = Math.max(0, Math.tanh(out[0])), turn = Math.tanh(out[1]) * C.TURN, braking = out[3] > 0.5;
      glue[i] += (Math.tanh(out[2]) - glue[i]) * C.GLUE_RATE; mem[i] = Math.tanh(out[7]); msg[i] = Math.tanh(out[8]);
      th[i] = th[i] + turn;
      let nvx = vx[i] + fx, nvy = vy[i] + fy;
      // sound mode: the vibration nudges every cell towards the nearest still line, like sand on a plate
      if (pattern) {
        const u = (px[i] - CX) / DR, w = (py[i] - CY) / DR, n = pattern.n * Math.PI, m = pattern.m * Math.PI;
        const cnu = Math.cos(n * u), cmw = Math.cos(m * w), cmu = Math.cos(m * u), cnw = Math.cos(n * w), a = cnu * cmw - cmu * cnw;
        const gu = -n * Math.sin(n * u) * cmw + m * Math.sin(m * u) * cnw, gw = -m * cnu * Math.sin(m * w) + n * cmu * Math.sin(n * w);
        const k = SHAKE * Math.max(-1, Math.min(1, a)) / (Math.hypot(gu, gw) + 1e-6); nvx -= k * gu; nvy -= k * gw;
      }
      if (!braking) { nvx += Math.cos(th[i]) * thrust * C.THRUST; nvy += Math.sin(th[i]) * thrust * C.THRUST; }
      const drag = braking ? C.BRAKE_DRAG : C.DRAG; nvx *= drag; nvy *= drag;
      const sp = Math.hypot(nvx, nvy); if (sp > C.VMAX) { nvx *= C.VMAX / sp; nvy *= C.VMAX / sp; }
      let nx = x + nvx, ny = y + nvy;
      const rx = nx - CX, ry = ny - CY, r = Math.hypot(rx, ry);
      if (r > DR - 0.5) { const k2 = (DR - 0.5) / r; nx = CX + rx * k2; ny = CY + ry * k2; const vr = (nvx * rx + nvy * ry) / r; if (vr > 0) { nvx -= vr * rx / r * 1.5; nvy -= vr * ry / r * 1.5; } }
      px[i] = nx; py[i] = ny; vx[i] = nvx; vy[i] = nvy;

      // ---- metabolism + eat ----
      energy[i] -= C.BASE_COST * (kd === 1 && C.H_FLAT ? C.H_FLAT : crowd) - (exp.perks ? Math.min(kinN, C.KIN_CAP) * C.KIN_BONUS : 0) + (braking ? 0 : thrust * C.MOVE_COST) + Math.abs(turn) * C.TURN_COST + nn * C.NEURON_COST;
      age[i]++;
      const ec = gi(nx, ny), take = kd === 1 ? 0 : Math.min(food[ec], C.EAT); food[ec] -= take; energy[i] += take;
      if (exp.waste) { energy[i] -= Math.min(waste[ec], C.WASTE_CAP * 2) * C.WASTE_HARM; waste[ec] += C.WASTE_MAKE + take * C.WASTE_EAT; }

      // ---- bite a stranger in front: groups defend each other ----
      if (kd === 1) {
        // a hunter engulfs a grazer whole, but a cell held on many sides is hard to tear out of its body
        if (out[4] > 0.5 && prey >= 0 && alive[prey]) {
          energy[i] -= C.H_TRY;
          if (rand() < 1 / (1 + C.SHIELD * held[prey]) / (exp.sizeRefuge ? 1 + (bsize[prey] - 1) * C.SIZE_SHIELD : 1)) { energy[i] += energy[prey] * C.H_GAIN + C.H_MEAT; kill(prey); kills++; }
        }
      } else if (exp.fightBack && out[4] > 0.5 && foe >= 0 && alive[foe]) {
        // fight back: a grazer strikes a hunter in front of it, harder the more family it is holding on to
        energy[i] -= C.STING_COST;
        energy[foe] -= C.STING * (1 + held[i]);
        if (energy[foe] <= 0) { kill(foe); stung++; }
      } else if (out[4] > 0.5 && prey >= 0 && alive[prey]) {
        energy[i] -= 0.1;
        let myK = Math.min(kinN, 3), theirK = 0;
        if (exp.perks) {
          const qx = px[prey], qy = py[prey];
          // count the prey's family close by (cheap: look in its own bucket)
          const qh = Math.min(HG - 1, (qy / S) | 0) * HG + Math.min(HG - 1, (qx / S) | 0);
          for (let j = head[qh]; j >= 0; j = next[j]) if (j !== prey && alive[j] && hueDist(hue[j], hue[prey]) < 25) { const dx = px[j] - qx, dy = py[j] - qy; if (dx * dx + dy * dy < 2.1) theirK++; }
          theirK = Math.min(theirK, 3);
        } else myK = 0;
        if (energy[i] * (1 + myK) > energy[prey] * (1 + theirK)) {
          const steal = Math.min(10, energy[prey] * 0.4) / (1 + C.ARMOUR * held[prey]); energy[prey] -= steal; energy[i] += steal * 0.7;
          if (energy[prey] <= 0) kill(prey);
        }
      }
      // ---- call ----
      if (out[6] > 0.3) { const v = Math.min(1, out[6] - 0.3); signal[ec] += v * 0.6; energy[i] -= 0.025 * v; speak[i] = v; } else speak[i] = 0;
      // ---- share with hungrier family touching it ----
      if (out[5] > 0.5 && energy[i] > 18) for (let m = 0; m < nKin; m++) { const j = kinList[m]; if (alive[j] && energy[j] < energy[i]) { energy[i] -= 1.5; energy[j] += 1.5; } }
      bonds[i] = Math.min(255, kinN);

      // ---- death ----
      if (energy[i] <= 0 || age[i] > C.MAX_AGE || (kd === 1 && !exp.hunters)) { kill(i); continue; }

      // ---- reproduce: the child buds off behind ----
      if (energy[i] > (kd === 1 ? C.H_REPRO : C.REPRO)) {
        const ch2 = alloc();
        if (ch2 >= 0) {
          const half = energy[i] * 0.45; energy[i] = half; energy[ch2] = half;
          const ba = th[i] + Math.PI + (rand() - 0.5) * 1.6;
          // a sticky parent's daughter doesn't separate: it stays attached and the body grows
          const stay = exp.clonal && exp.glue && glue[i] > 0, kick = stay ? 0 : 0.15, gap = (stay ? 0.9 : 0.55) * cs;
          kind[ch2] = kd; nKind[kd]++;
          alive[ch2] = 1; px[ch2] = nx + Math.cos(ba) * gap; py[ch2] = ny + Math.sin(ba) * gap; vx[ch2] = Math.cos(ba) * kick; vy[ch2] = Math.sin(ba) * kick;
          th[ch2] = stay && exp.align ? th[i] : th[i] + (rand() - 0.5) * 2; age[ch2] = 0; gen[ch2] = gen[i] + 1; glue[ch2] = glue[i]; speak[ch2] = 0; mem[ch2] = 0; lastFood[ch2] = -1; msg[ch2] = 0; held[ch2] = 0;
          nh[ch2] = nh[i]; if (rand() < C.BRAIN_MUT * params.mut) nh[ch2] = Math.max(1, Math.min(NH, nh[i] + (rand() < 0.5 ? -1 : 1)));
          const m = params.mut, pb = i * NW, cb = ch2 * NW;
          for (let w = 0; w < NW; w++) W[cb + w] = rand() < C.MUT * m ? W[pb + w] + gauss() * C.MUT_SD : W[pb + w];
          let hh = hue[i] + gauss() * 2.5 * m; if (rand() < 0.003 * m) hh = rand() * 360;
          hue[ch2] = (hh % 360 + 360) % 360;
          lin[ch2] = lin[i]; if (hueDist(hue[ch2], linHue[lin[i]]) >= 30) lin[ch2] = newLin(lin[i], hue[ch2], kd);
          nbp[ch2] = 0; bsize[ch2] = 1; if (stay && exp.adhesion && link(i, ch2)) bondLog.clonal++;
          pop++; births++;
        }
      }
    }
    if (exp.adhesion) { nBonds = 0; for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; const o = i * SL; for (let s = 0; s < nbp[i]; s++) { const j = bp[o + s]; if (j > i && nBonds < BMAX) { bondA[nBonds] = i; bondB[nBonds] = j; nBonds++; } } } }
    // how big is each cell's body? (used next step: big bodies are hard for hunters to tear cells out of)
    if (exp.sizeRefuge) { organisms(); for (let i = 0; i < MAX; i++) if (alive[i]) bsize[i] = orgOf[i] >= 0 ? Math.min(65535, orgSize[orgOf[i]]) : 1; }
    if (nKind[0] < 40) for (let i = 0; i < 60; i++) spawnRandom(0);
    // if the hunters die out, a few new ones drift in (counted, so it's visible)
    if (exp.hunters && nKind[1] < C.H_MIN) { for (let i = 0; i < C.H_MIN * 2; i++) spawnRandom(1); reintroduced++; }
  }

  // organisms: groups of cells that are holding on to each other right now (connected through bonds)
  const uf = new Int32Array(MAX), orgSize = new Int32Array(MAX), orgOf = new Int32Array(MAX);
  const find = a => { while (uf[a] !== a) { uf[a] = uf[uf[a]]; a = uf[a]; } return a; };
  function organisms() {
    for (let i = 0; i < MAX; i++) { uf[i] = i; orgSize[i] = 0; }
    for (let k = 0; k < nBonds; k++) {
      const a = bondA[k], b = bondB[k]; if (!alive[a] || !alive[b]) continue;
      const dx = px[a] - px[b], dy = py[a] - py[b]; if (dx * dx + dy * dy > 4) continue;
      const ra = find(a), rb = find(b); if (ra !== rb) uf[ra] = rb;
    }
    let n = 0, inOrg = 0, count = 0, largest = 0;
    for (let i = 0; i < MAX; i++) if (alive[i]) { n++; orgSize[find(i)]++; }
    for (let i = 0; i < MAX; i++) if (alive[i]) { const sz = orgSize[find(i)]; orgOf[i] = sz > 1 ? find(i) : -1; if (sz > 1) inOrg++; if (uf[i] === i && sz > 1) { count++; if (sz > largest) largest = sz; } }
    return { n, inOrg, count, largest, pctIn: n ? inOrg / n : 0, meanSize: count ? inOrg / count : 0 };
  }
  // a few numbers that say what kind of world this is
  function stats() {
    let n = 0, bonded = 0, maxGen = 0, sumKin = 0, gl = 0, sumNh = 0, maxNh = 0; const nhK = [0, 0];
    const fam = new Uint16Array(36);
    for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; n++; if (bonds[i] > 0) bonded++; sumKin += bonds[i]; if (gen[i] > maxGen) maxGen = gen[i]; if (glue[i] > 0) gl++; sumNh += nh[i]; if (nh[i] > maxNh) maxNh = nh[i]; nhK[kind[i]] += nh[i]; if (kind[i] === 0) fam[(hue[i] / 10) | 0]++; }
    let families = 0; for (const f of fam) if (f >= Math.max(10, n * 0.02)) families++;
    return { brainG: nKind[0] ? nhK[0] / nKind[0] : 0, brainH: nKind[1] ? nhK[1] / nKind[1] : 0, hunters: nKind[1], grazers: nKind[0], kills, stung, reintroduced, brain: n ? sumNh / n : 0, maxBrain: maxNh, season: exp.seasons ? Math.sin(tick / C.SEASON * 6.2832) : 0, pop: n, tick, births, bondedPct: n ? bonded / n : 0, kinPer: n ? sumKin / n : 0, maxGen, sticky: n ? gl / n : 0, families, bonds: nBonds, bondLog: Object.assign({}, bondLog) };
  }

  // save / restore the whole world (slots keep their numbers, so bonds and families stay valid)
  function snapshot() {
    const idx = []; for (let i = 0; i < MAX; i++) if (alive[i]) idx.push(i);
    const n = idx.length, I = Int32Array.from(idx), pick = (a, T, k = 1) => { const o = new T(n * k); for (let j = 0; j < n; j++) for (let q = 0; q < k; q++) o[j * k + q] = a[idx[j] * k + q]; return o; };
    return {
      meta: { engine: 'cpu', G, MAX, tick, seed, worldSeed, births, kills, reintroduced, stung, nKind: nKind.slice(), blobs, params: Object.assign({}, params), exp: Object.assign({}, exp), nLin, bondLog: Object.assign({}, bondLog) },
      arrays: { idx: I, px: pick(px, Float32Array), py: pick(py, Float32Array), vx: pick(vx, Float32Array), vy: pick(vy, Float32Array), th: pick(th, Float32Array), energy: pick(energy, Float32Array), hue: pick(hue, Float32Array),
        age: pick(age, Uint32Array), gen: pick(gen, Uint32Array), glue: pick(glue, Float32Array), mem: pick(mem, Float32Array), lastFood: pick(lastFood, Float32Array), nh: pick(nh, Uint8Array), msg: pick(msg, Float32Array),
        held: pick(held, Uint8Array), bsize: pick(bsize, Uint16Array), kind: pick(kind, Uint8Array), lin: pick(lin, Int32Array), nbp: pick(nbp, Uint8Array), bp: pick(bp, Int32Array, SL), W: pick(W, Float32Array, NW),
        food: food.slice(), signal: signal.slice(), waste: waste.slice(), free: freeList.slice(0, freeTop), speak: pick(speak, Float32Array),
        linPar: linPar.slice(0, nLin), linHue: linHue.slice(0, nLin), linBorn: linBorn.slice(0, nLin), linKind: linKind.slice(0, nLin) },
    };
  }
  function restore(snap) {
    const m = snap.meta, A = snap.arrays;
    if (m.G !== G || m.MAX > MAX) throw new Error('This save is from a different world size.');
    tick = m.tick; seed = m.seed; worldSeed = m.worldSeed; births = m.births; kills = m.kills; reintroduced = m.reintroduced; stung = m.stung; blobs = m.blobs.map(b => Object.assign({}, b));
    Object.assign(params, m.params); Object.assign(exp, m.exp); Object.assign(bondLog, m.bondLog || {});
    alive.fill(0); nbp.fill(0); bp.fill(-1); pop = 0; nKind[0] = nKind[1] = 0; nBonds = 0;
    const I = A.idx, n = I.length;
    for (let j = 0; j < n; j++) {
      const i = I[j]; alive[i] = 1; pop++;
      px[i] = A.px[j]; py[i] = A.py[j]; vx[i] = A.vx[j]; vy[i] = A.vy[j]; th[i] = A.th[j]; energy[i] = A.energy[j]; hue[i] = A.hue[j]; age[i] = A.age[j]; gen[i] = A.gen[j];
      glue[i] = A.glue[j]; mem[i] = A.mem[j]; lastFood[i] = A.lastFood[j]; nh[i] = A.nh[j]; msg[i] = A.msg[j]; held[i] = A.held[j]; bsize[i] = A.bsize[j]; kind[i] = A.kind[j]; nKind[kind[i]]++; lin[i] = A.lin[j];
      nbp[i] = A.nbp[j]; for (let q = 0; q < SL; q++) bp[i * SL + q] = A.bp[j * SL + q];
      W.set(A.W.subarray(j * NW, (j + 1) * NW), i * NW);
    }
    if (A.free) { freeTop = A.free.length; freeList.set(A.free); } else { freeTop = 0; for (let i = MAX - 1; i >= 0; i--) if (!alive[i]) freeList[freeTop++] = i; }
    if (A.speak) for (let j = 0; j < n; j++) speak[I[j]] = A.speak[j];
    food.set(A.food); signal.set(A.signal); waste.set(A.waste); buildCap();
    nLin = Math.min(LMAX, m.nLin); linPar.set(A.linPar.subarray(0, nLin)); linHue.set(A.linHue.subarray(0, nLin)); linBorn.set(A.linBorn.subarray(0, nLin)); linKind.set(A.linKind.subarray(0, nLin));
    nBonds = 0; for (let i = 0; i < MAX; i++) { if (!alive[i]) continue; for (let s = 0; s < nbp[i]; s++) { const j = bp[i * SL + s]; if (j > i && nBonds < BMAX) { bondA[nBonds] = i; bondB[nBonds] = j; nBonds++; } } }
  }

  // a new note clears food that is no longer on the pattern, so the cells have to move to the new lines
  function setPattern(p) { pattern = p && p.n > 0 && p.m > 0 && p.n !== p.m ? { n: p.n | 0, m: p.m | 0 } : null; buildCap(); if (pattern) for (let i = 0; i < CELLS; i++) if (food[i] > cap[i]) food[i] = cap[i]; }
  function setCellSize(v) { cs = Math.max(0.5, Math.min(2.5, +v || 1)); }
  return {
    setPattern, get pattern() { return pattern; }, setCellSize, get cellSize() { return cs; },
    snapshot, restore,
    G, MAX, NI, NH, NO, NW, C, CX, CY, DR, params, exp,
    alive, px, py, vx, vy, th, energy, hue, age, gen, glue, speak, bonds, W, lastOut, food, cap, wall,
    get signal() { return signal; }, get waste() { return waste; }, mem, nh, msg, held, bsize, kind, get pop() { return pop; }, get tick() { return tick; }, get seed() { return worldSeed; },
    get nBonds() { return nBonds; }, get nLin() { return nLin; }, lin, linPar, linHue, linBorn, linKind, bondA, bondB, bp, nbp, SL, bondLog,
    reset, step, stats, spawnRandom, organisms, orgOf, orgSize, find,
    feed(x, y, r) { for (let yy = Math.max(0, (y - r) | 0); yy <= Math.min(G - 1, (y + r) | 0); yy++) for (let xx = Math.max(0, (x - r) | 0); xx <= Math.min(G - 1, (x + r) | 0); xx++) { const d = Math.hypot(xx + 0.5 - x, yy + 0.5 - y); if (d < r && !wall[yy * G + xx]) food[yy * G + xx] = Math.min(C.FOOD_CAP * 1.5, food[yy * G + xx] + C.FOOD_CAP * (1 - d / r) * 1.2); } },
  };
}
if (typeof module !== 'undefined') module.exports = { createWorld };
