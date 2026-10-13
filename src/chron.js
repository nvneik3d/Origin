// ---- the chronicle: history of the run, notable events, and the family tree ----
// Runs in the worker next to either engine. It only reads the world, so it can't change what happens.
function hueName(h) {
  h = ((h % 360) + 360) % 360;
  const N = [[15, 'red'], [40, 'orange'], [65, 'gold'], [90, 'yellow-green'], [150, 'green'], [180, 'teal'], [205, 'cyan'], [250, 'blue'], [285, 'violet'], [320, 'magenta'], [345, 'pink'], [361, 'red']];
  for (const [lim, n] of N) if (h < lim) return n;
  return 'red';
}
const ordinal = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');
function createChronicle(w) {
  const PT = 10;                 // numbers per history point
  let every = 50, lastT = -1e9, pts = [], events = [], fresh = [], evId = 0;
  const lin = new Map();         // family id -> { peak, peakT, now, died }
  let cnt = new Int32Array(1024);
  const R = { huntersT: -1e9, body: 0, bodyAnnounced: 0, brain: 0, gen: 0, hunters: 0, leader: -1, leaderT: -1e9, crashT: -1e9, season: null, huntersAhead: false, firstBody: false };
  const GENS = [100, 250, 500, 1000, 2500, 5000, 10000];
  function add(kind, text, extra = {}) {
    const e = Object.assign({ id: ++evId, t: w.tick, kind, text }, extra);
    events.push(e); fresh.push(e); if (events.length > 400) events.shift();
  }
  function centroid(test) {
    let sx = 0, sy = 0, n = 0, slot = -1;
    for (let i = 0; i < w.MAX; i++) if (w.alive[i] && test(i)) { sx += w.px[i]; sy += w.py[i]; n++; if (slot < 0) slot = i; }
    return n ? { x: sx / n, y: sy / n, slot, n } : null;
  }
  function sample(force) {
    if (!force && w.tick - lastT < every) return;
    lastT = w.tick;
    const st = w.stats(), o = w.organisms(), t = w.tick;
    // the largest body and where it is
    let root = -1;
    if (o.largest > 1) for (let i = 0; i < w.MAX; i++) if (w.alive[i] && w.orgOf[i] >= 0 && w.orgSize[w.orgOf[i]] === o.largest) { root = w.orgOf[i]; break; }
    pts.push(t, st.grazers, st.hunters, o.pctIn, o.meanSize, o.largest, st.brainG, st.brainH, st.season, st.kills);
    if (pts.length / PT > 3000) { const k = []; for (let p = 0; p < pts.length / PT; p += 2) for (let j = 0; j < PT; j++) k.push(pts[p * PT + j]); pts = k; every *= 2; }

    // ---- events ----
    if (!R.firstBody && o.largest >= 10 && root >= 0) { R.firstBody = true; R.bodyAnnounced = o.largest; const c = centroid(i => w.orgOf[i] === root); add('body', `First body: ${o.largest} cells holding on to each other.`, c ? { x: c.x, y: c.y, slot: c.slot, hue: w.hue[c.slot] } : {}); }
    else if (R.firstBody && o.largest >= 20 && o.largest > R.bodyAnnounced * 1.3 && root >= 0) { R.bodyAnnounced = o.largest; const c = centroid(i => w.orgOf[i] === root); add('body', `New record: a body of ${o.largest} cells.`, c ? { x: c.x, y: c.y, slot: c.slot, hue: w.hue[c.slot] } : {}); }
    if (st.maxBrain > R.brain) {
      if (st.maxBrain >= 9 && R.brain > 0) { const c = centroid(i => w.nh[i] === st.maxBrain); add('brain', `Biggest brain yet: a ${w.kind[c ? c.slot : 0] === 1 ? 'hunter' : 'grazer'} with ${st.maxBrain} neurons.`, c ? { x: w.px[c.slot], y: w.py[c.slot], slot: c.slot, hue: w.hue[c.slot] } : {}); }
      R.brain = st.maxBrain;
    }
    for (const g of GENS) if (st.maxGen >= g && R.gen < g) { R.gen = g; add('gen', `Generation ${g.toLocaleString()}: the oldest family line now has ${g.toLocaleString()} ancestors.`); }
    // hunters dying out: say so the first time, then at most once every 3,000 steps
    if (st.reintroduced > R.hunters) { if (R.hunters === 0 || t - R.huntersT > 3000) { add('hunters', st.reintroduced === 1 ? 'Hunters died out. A few newcomers drifted in.' : `Hunters have now died out ${st.reintroduced} times. Each time a few newcomers drift in.`); R.huntersT = t; } R.hunters = st.reintroduced; }
    if (!R.huntersAhead && st.hunters > st.grazers && st.grazers > 0) { R.huntersAhead = true; add('hunters', `Hunters now outnumber grazers: ${st.hunters.toLocaleString()} to ${st.grazers.toLocaleString()}.`); }
    if (R.huntersAhead && st.hunters < st.grazers * 0.7) R.huntersAhead = false;
    // a crash: grazers fell well below their recent peak
    { const back = Math.max(0, pts.length / PT - Math.ceil(1000 / every)); let peak = 0; for (let p = back; p < pts.length / PT; p++) peak = Math.max(peak, pts[p * PT + 1]);
      if (peak > 300 && st.grazers < peak * 0.55 && t - R.crashT > 2000) { R.crashT = t; add('crash', `Crash: grazers fell from ${peak.toLocaleString()} to ${st.grazers.toLocaleString()}.`); } }
    if (st.season !== 0 || w.exp.seasons) {
      const L = w.C.SEASON || 4000, ph = ((t / L - 0.25) % 1 + 1) % 1, s = ph > 0.375 && ph < 0.625 ? 'winter' : ph < 0.125 || ph > 0.875 ? 'summer' : null;
      if (w.exp.seasons && s && s !== R.season) { if (R.season !== null || s === 'winter') add('season', s === 'winter' ? 'Winter: food barely grows back.' : 'Summer: food grows back fast again.'); R.season = s; }
    }

    // ---- families ----
    if (w.lin) {
      const nL = w.nLin; if (cnt.length < nL) cnt = new Int32Array(Math.max(nL, cnt.length * 2)); else cnt.fill(0, 0, nL);
      let grazers = 0;
      for (let i = 0; i < w.MAX; i++) if (w.alive[i]) { const l = w.lin[i]; if (l >= 0 && l < nL) cnt[l]++; if (!w.kind[i]) grazers++; }
      let lead = -1, leadN = 0;
      for (const [id, r] of lin) { if (r.now > 0 && cnt[id] === 0) { r.now = 0; r.died = t; if (r.peak >= 100) add('family', `The ${hueName(w.linHue[id])} ${w.linKind[id] ? 'hunter' : ''} family died out. At its peak it had ${r.peak.toLocaleString()} cells.`.replace('  ', ' '), { hue: w.linHue[id] }); } }
      for (let id = 0; id < nL; id++) {
        const c = cnt[id]; if (!c) continue;
        let r = lin.get(id); if (!r) { r = { peak: 0, peakT: t, now: 0, died: -1 }; lin.set(id, r); }
        r.now = c; r.died = -1; if (c > r.peak) { r.peak = c; r.peakT = t; }
        if (!w.linKind[id] && c > leadN) { lead = id; leadN = c; }
      }
      // forget small families that are gone; their branches are still in the engine's table
      if (lin.size > 3000) for (const [id, r] of lin) if (r.now === 0 && r.peak < 30) lin.delete(id);
      if (lead >= 0 && lead !== R.leader && grazers > 200 && leadN / grazers >= 0.25 && t - R.leaderT > 600) {
        const first = R.leader < 0; R.leader = lead; R.leaderT = t;
        const c = centroid(i => w.lin[i] === lead);
        add('family', first ? `The ${hueName(w.linHue[lead])} family leads: ${Math.round(leadN / grazers * 100)}% of grazers.` : `A new family took over: the ${hueName(w.linHue[lead])} family is now ${Math.round(leadN / grazers * 100)}% of grazers.`, c ? { x: c.x, y: c.y, slot: c.slot, hue: w.linHue[lead] } : { hue: w.linHue[lead] });
      }
    }
  }
  // the family tree: families that ever reached 30 cells, each joined to its nearest such ancestor
  function tree() {
    if (!w.lin) return null;
    const sig = new Set(); for (const [id, r] of lin) if (r.peak >= 30) sig.add(id);
    const ids = [...sig].sort((a, b) => lin.get(b).peak - lin.get(a).peak).slice(0, 160);
    const keep = new Set(ids), out = [];
    for (const id of ids) {
      let p = w.linPar[id], hops = 0; while (p >= 0 && !keep.has(p) && hops++ < 5000) p = w.linPar[p];
      const r = lin.get(id);
      out.push({ id, par: p >= 0 && keep.has(p) ? p : -1, hue: w.linHue[id], kind: w.linKind[id], born: w.linBorn[id], died: r.died, peak: r.peak, peakT: r.peakT, now: r.now });
    }
    return out;
  }
  return {
    sample, tree,
    takeFresh() { const f = fresh; fresh = []; return f; },
    history() { return { every, data: Float32Array.from(pts), PT }; },
    get events() { return events; },
    save() { return { every, lastT, pts, events, evId, R, lin: [...lin] }; },
    load(s) { every = s.every; lastT = s.lastT; pts = s.pts; events = s.events; evId = s.evId; Object.assign(R, s.R); lin.clear(); for (const [k, v] of s.lin) lin.set(k, v); fresh = events.slice(); },
    reset() { every = 50; lastT = -1e9; pts = []; events = []; fresh = []; lin.clear(); Object.assign(R, { huntersT: -1e9, body: 0, bodyAnnounced: 0, brain: 0, gen: 0, hunters: 0, leader: -1, leaderT: -1e9, crashT: -1e9, season: null, huntersAhead: false, firstBody: false }); },
  };
}
