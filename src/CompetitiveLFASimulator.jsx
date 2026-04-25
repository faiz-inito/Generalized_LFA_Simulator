import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════════
// COMPETITIVE LFA SIMULATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════════
//
// Competitive LFA mechanism (INVERTED signal vs sandwich):
//
//  Mobile species (same PDEs as sandwich):
//    A + P  ⇌  PA    (analyte sequesters detector in solution; Ka1/Kd1)
//
//  Test line — immobilised ANTIGEN (Ag_TL), captures FREE P only:
//    P  + Ag_TL  ⇌  PAg    SIGNAL species  (Ka2/Kd2)
//    PA cannot bind — detector paratope already occupied by analyte
//
//  Control line — anti-species antibody, captures any free P (same as sandwich):
//    P  + Rc  →  PC
//
//  Signal interpretation:
//    High [A]  →  PA forms in solution  →  little free P at test line  →  DIM  →  POSITIVE
//    Low  [A]  →  P stays free           →  P binds Ag_TL              →  BRIGHT →  NEGATIVE
//
//  T/C verdict (inverted from sandwich):
//    POSITIVE  when  T/C  <  threshold  (dim test line relative to control)
//    NEGATIVE  when  T/C  ≥  threshold  (bright test line — no analyte)
//
//  Physical model: identical to sandwich — upwind advection-diffusion,
//  Berli & Kler flow U(x)=c/x³, conjugate burst+decay release, porosity/tortuosity.
// ═══════════════════════════════════════════════════════════════════════════════

function gaussSmooth(arr, sigma = 1.2) {
  const n = arr.length;
  const out = new Float64Array(n);
  const r = Math.ceil(3 * sigma);
  const kernel = [];
  let ksum = 0;
  for (let k = -r; k <= r; k++) {
    const w = Math.exp(-(k * k) / (2 * sigma * sigma));
    kernel.push(w); ksum += w;
  }
  for (let i = 0; i < n; i++) {
    let v = 0;
    for (let k = -r; k <= r; k++) {
      const j = Math.max(0, Math.min(n - 1, i + k));
      v += arr[j] * kernel[k + r];
    }
    out[i] = v / ksum;
  }
  return out;
}

function runSimulation(up, kp, pp, nOutput = 160) {
  const { Ao, Po, Ag_tl, x_tl, x_cl, Ro_cl, t_end } = up;
  const { Ka1, Kd1, Ka2, Kd2, Ka_cl, Kd_cl, DA, DP, flow_c } = kp;
  const {
    strip_L, dx, phi, tau,
    conj_burst, conj_lambda,
    line_width_mm, sample_vol, U_max,
  } = pp;

  const strip_width_mm  = 4.0;
  const strip_thick_mm  = 0.135;
  const cross_section   = strip_width_mm * strip_thick_mm;
  const flow_rate_uL_s  = U_max * cross_section * 1e-3;
  const t_sample_end    = flow_rate_uL_s > 0 ? sample_vol / flow_rate_uL_s : t_end;

  const N   = Math.round(strip_L / dx);
  const tl  = Math.min(Math.round(x_tl / dx), N - 2);
  const cl_reachable = (x_cl + dx) < strip_L;
  const cl  = cl_reachable ? Math.min(Math.round(x_cl / dx), N - 2) : -1;
  const lw  = Math.max(1, Math.round(line_width_mm / dx));

  const DA_eff = DA * phi / tau;
  const DP_eff = DP * phi / tau;
  const dt = Math.min(t_end / 1500, 0.2);

  const x = Array.from({ length: N }, (_, i) => (i + 1) * dx);
  const U = x.map(xi => Math.min(flow_c / (xi ** 3), U_max));

  const i2  = 1 / (dx * dx);

  // State
  let A  = new Float64Array(N);
  let P  = new Float64Array(N);
  let PA = new Float64Array(N);

  // Test line: immobilised antigen Ag_TL captures free P → PAg (the signal)
  // Ag_free = Ag_tl - PAg
  let PAg_tl = 0;   // immobilised antigen-detector complex (SIGNAL)

  // Control line: anti-species Ab captures BOTH free P → CP  AND  PA complex → CAP
  // Rc_free is shared — both reactions consume receptor sites
  let Rc_free  = Ro_cl;
  let CP_cl    = 0;   // free-detector captured at control line
  let CAP_cl   = 0;   // analyte-loaded detector captured at control line

  // Pad flux
  let pad_A = 0, pad_P = 0, pad_PA = 0;

  const outputEvery = Math.max(1, Math.round((t_end / dt) / nOutput));
  const out = {
    times: [], A: [], P: [], PA: [],
    PAg: [],          // test line: AgP  (free P bound to immobilised antigen)
    CP:  [],          // control line: CP  (free P captured)
    CAP: [],          // control line: CAP (PA complex captured)
    PC:  [],          // total control signal = CP + CAP  (for T/C ratio)
    pad_A: [], pad_P: [], pad_PA: [],
    x, tl, cl, lw, N, cl_reachable,
    t_sample_end, flow_rate_uL_s,
  };

  let step = 0;
  const maxSteps = Math.round(t_end / dt);

  // ── Competitive assay — correct physical model ───────────────────────────
  // IDENTICAL strip architecture to sandwich:
  //   - P (gold-labelled antibody) is DRY on the conjugate pad, pre-loaded.
  //   - Sample flows through conjugate pad → reconstitutes P.
  //   - A and P then co-flow together through the NC membrane.
  //   - In solution: A + P ⇌ PA  (analyte sequesters detector)
  //
  // DIFFERENCE is ONLY at the test line:
  //   - Sandwich test line: immobilised antibody captures A (via PA complex)
  //   - Competitive test line: immobilised ANTIGEN captures FREE P only
  //     PA cannot bind (paratope already occupied by A)
  //
  // Net effect: high [A] → more PA in solution → less free P reaches test line
  //             → test line dims → POSITIVE result
  const conj_end = Math.round(0.20 * N);

  while (step <= maxSteps) {
    const t_now = step * dt;

    if (step % outputEvery === 0) {
      out.times.push(t_now);
      out.A.push(Array.from(A));
      out.P.push(Array.from(P));
      out.PA.push(Array.from(PA));
      out.PAg.push(PAg_tl);
      out.CP.push(CP_cl);
      out.CAP.push(CAP_cl);
      out.PC.push(CP_cl + CAP_cl);   // total control line gold = CP + CAP
      out.pad_A.push(pad_A);
      out.pad_P.push(pad_P);
      out.pad_PA.push(pad_PA);
    }

    // Pre-load conjugate pad with P on first step (after frame-0 snapshot)
    if (step === 0) {
      for (let j = 0; j < conj_end; j++) P[j] = Po;
    }

    // Inlet BCs:
    //   A[0] = Ao while sample flows  (analyte enters continuously from sample pad)
    //   PA[0] = 0  (no pre-formed complex at the inlet edge)
    //   P[0]  — NOT forced; depletes naturally as conjugate pad sweeps downstream
    const sampleFlowing = t_now <= t_sample_end;
    A[0]  = sampleFlowing ? Ao : 0.0;
    PA[0] = 0.0;
    // P[0] evolves freely from conjugate pad pre-load — do not override

    // ── Mobile reaction: A + P ⇌ PA (in solution everywhere) ─────────
    const F_PA = new Float64Array(N);
    for (let j = 0; j < N; j++)
      F_PA[j] = Ka1 * A[j] * P[j] - Kd1 * PA[j];

    // ── Test line: free P competes with PA for immobilised Ag_TL ─────
    // Only FREE P binds Ag_TL (PA cannot — paratope occupied)
    const Ag_free = Math.max(Ag_tl - PAg_tl, 0);
    let P_zone = 0, nz = 0;
    for (let j = Math.max(0, tl - lw); j <= Math.min(N - 1, tl + lw); j++) {
      P_zone += P[j]; nz++;
    }
    P_zone /= nz;
    // Rate of PAg formation (Ka2/Kd2 governs detector–antigen binding)
    const f_PAg = Ka2 * P_zone * Ag_free - Kd2 * PAg_tl;

    // ── Control line: anti-species Ab captures P AND PA ──────────────
    // Both share the same receptor pool Rc_free.
    // f_CP  = rate of CP  formation  (free P  binds Rc)
    // f_CAP = rate of CAP formation  (PA complex binds Rc)
    let f_CP = 0, f_CAP = 0;
    if (cl_reachable) {
      f_CP  = Ka_cl  * P[cl]  * Rc_free - Kd_cl * CP_cl;
      f_CAP = Ka_cl  * PA[cl] * Rc_free - Kd_cl * CAP_cl;
      // Note: Rc_free consumed by both; total rate consuming Rc = f_CP + f_CAP
    }

    // ── PDE RHS: upwind advection-diffusion ──────────────────────────
    const dA  = new Float64Array(N);
    const dP  = new Float64Array(N);
    const dPA = new Float64Array(N);

    for (let j = 0; j < N; j++) {
      const Uj  = U[j];
      const Al  = j === 0   ? A[0]   : A[j - 1];
      const Ar  = j === N-1 ? A[N-1] : A[j + 1];
      const Pl  = j === 0   ? P[0]   : P[j - 1];
      const Pr  = j === N-1 ? P[N-1] : P[j + 1];
      const PAl = j === 0   ? PA[0]  : PA[j - 1];
      const PAr = j === N-1 ? PA[N-1]: PA[j + 1];

      const advA  = Uj * (A[j]  - Al)  / dx;
      const advP  = Uj * (P[j]  - Pl)  / dx;
      const advPA = Uj * (PA[j] - PAl) / dx;

      const diffA  = DA_eff * (Ar  - 2*A[j]  + Al)  * i2;
      const diffP  = DP_eff * (Pr  - 2*P[j]  + Pl)  * i2;
      const diffPA = DP_eff * (PAr - 2*PA[j] + PAl) * i2;

      // A disappears → PA; P disappears → PA
      dA[j]  = diffA  - advA  - F_PA[j];
      dP[j]  = diffP  - advP  - F_PA[j];
      dPA[j] = diffPA - advPA + F_PA[j];
    }

    // Sink free P over test line zone (captured by Ag_TL)
    const zone_nodes = [];
    for (let j = Math.max(0, tl - lw); j <= Math.min(N - 1, tl + lw); j++)
      zone_nodes.push(j);
    zone_nodes.forEach(j => { dP[j] -= f_PAg / zone_nodes.length; });

    // Sink free P at control line (captured as CP)
    // Sink PA complex at control line (captured as CAP)
    if (cl_reachable) {
      dP[cl]  -= f_CP;
      dPA[cl] -= f_CAP;
    }

    // Euler step
    for (let j = 0; j < N; j++) {
      A[j]  = Math.max(0, A[j]  + dt * dA[j]);
      P[j]  = Math.max(0, P[j]  + dt * dP[j]);
      PA[j] = Math.max(0, PA[j] + dt * dPA[j]);
    }
    PAg_tl  = Math.max(0, PAg_tl  + dt * f_PAg);
    CP_cl   = Math.max(0, CP_cl   + dt * f_CP);
    CAP_cl  = Math.max(0, CAP_cl  + dt * f_CAP);
    // Rc_free is consumed by both CP and CAP formation
    Rc_free = Math.max(0, Rc_free - dt * (f_CP + f_CAP));

    const U_out = U[N - 1];
    pad_A  += U_out * A[N-1]  * dt;
    pad_P  += U_out * P[N-1]  * dt;
    pad_PA += U_out * PA[N-1] * dt;

    // Re-enforce inlet BCs after Euler step
    A[0]  = sampleFlowing ? Ao : 0.0;
    PA[0] = 0.0;
    // P[0] is not forced — it depletes from conjugate pad naturally
    step++;
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARAM METADATA
// ═══════════════════════════════════════════════════════════════════════════════
const KINETIC_META = [
  { key:"Ka1",    label:"Ka₁ (det-analyte on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:7.35e-4, sci:true },
  { key:"Kd1",    label:"Kd₁ (det-analyte off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:5.7e-5,  sci:true },
  { key:"Ka2",    label:"Ka₂ (det-antigen on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:7.35e-4, sci:true },
  { key:"Kd2",    label:"Kd₂ (det-antigen off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:5.7e-5,  sci:true },
  { key:"Ka_cl",  label:"Ka_cl (ctrl line on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:5.0e-4,  sci:true },
  { key:"Kd_cl",  label:"Kd_cl (ctrl line off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:4.0e-5,  sci:true },
  { key:"DA",     label:"Analyte diffusivity",   unit:"mm²/s",    min:1e-7, max:1e-1, def:1e-4,    sci:true },
  { key:"DP",     label:"Detector diffusivity",  unit:"mm²/s",    min:1e-8, max:1e-2, def:1e-6,    sci:true },
  { key:"flow_c", label:"Flow constant c",       unit:"mm⁴/s",    min:100,  max:5e4,  def:5327.75, sci:false},
];

const PHYSICAL_META = [
  { key:"strip_L",       label:"Strip length",        unit:"mm",  min:20,   max:200,  def:40,   sci:false, step:5    },
  { key:"dx",            label:"Grid spacing",         unit:"mm",  min:0.25, max:2,    def:0.5,  sci:false, step:0.25 },
  { key:"phi",           label:"Membrane porosity φ",  unit:"—",   min:0.1,  max:0.95, def:0.7,  sci:false, step:0.05 },
  { key:"tau",           label:"Tortuosity τ",         unit:"—",   min:1.0,  max:5.0,  def:1.5,  sci:false, step:0.1  },
  { key:"conj_burst",    label:"Conjugate burst β",    unit:"—",   min:0,    max:1,    def:0.4,  sci:false, step:0.05 },
  { key:"conj_lambda",   label:"Release rate λ",       unit:"s⁻¹", min:1e-4, max:0.1,  def:5e-3, sci:true,  step:1e-4 },
  { key:"line_width_mm", label:"Line half-width",      unit:"mm",  min:0.25, max:3,    def:0.5,  sci:false, step:0.25 },
  { key:"sample_vol",    label:"Sample volume",        unit:"µL",  min:5,    max:200,  def:75,   sci:false, step:5    },
  { key:"U_max",         label:"Max flow velocity",    unit:"mm/s",min:0.01, max:5,    def:0.8,  sci:false, step:0.05 },
];

function defaultKinetic() { return Object.fromEntries(KINETIC_META.map(m => [m.key, m.def])); }
function defaultPhysical() { return Object.fromEntries(PHYSICAL_META.map(m => [m.key, m.def])); }

// ═══════════════════════════════════════════════════════════════════════════════
// SLIDER UPPER-LIMIT METADATA
// ═══════════════════════════════════════════════════════════════════════════════
// Lets the user override slider upper bounds at runtime via the "Limits"
// sub-tab. Validation is permissive (any positive number above `min`); no
// hard ceiling enforced — see LimitRow for the full validation rules.
// ═══════════════════════════════════════════════════════════════════════════════
const SLIDER_LIMITS_META = [
  { key:"Po",    label:"Detector [P₀]",          unit:"nM", min:0.1, def:100 },
  { key:"Ag_tl", label:"Immob. antigen [Ag_TL]", unit:"nM", min:0.1, def:100 },
  { key:"Ro_cl", label:"Ctrl receptor [Rc]",     unit:"nM", min:0.1, def:100 },
];
function defaultSliderLimits() {
  return Object.fromEntries(SLIDER_LIMITS_META.map(m => [m.key, m.def]));
}

function validateParam(meta, rawStr) {
  const v = parseFloat(rawStr);
  if (isNaN(v)) return { ok: false, msg: "Not a number" };
  if (v < meta.min) return { ok: false, msg: `Min: ${meta.min}` };
  if (v > meta.max) return { ok: false, msg: `Max: ${meta.max}` };
  return { ok: true, v };
}

// ═══════════════════════════════════════════════════════════════════════════════
// THEME — amber-tinted dark (distinct from sandwich simulator's blue)
// ═══════════════════════════════════════════════════════════════════════════════
const T = {
  bg:"#070a08", surface:"#0b100d", card:"#0f1510",
  border:"#1a2820", border2:"#20352a", borderHi:"#2a4535",
  // Species colours
  A:"#38bdf8",    // analyte — cyan (same as sandwich)
  P:"#fbbf24",    // detector — amber (free P is the KEY signal species here)
  PA:"#f43f5e",   // analyte-detector complex — rose (sequesters P, reduces signal)
  PAg:"#4ade80",  // PAg at test line — green (SIGNAL in competitive = green)
  PC:"#a78bfa",   // control line — violet
  accent:"#fbbf24",
  text:"#dde4f0", muted:"#4a5a50", muted2:"#6b8070",
  err:"#f87171", ok:"#4ade80", warn:"#f59e0b",
};

const PADS = [
  { id:"sample", label:"Sample Pad",    sub:"sample application",  frac:[0.00,0.09], bg:"#0e1a0e", bdr:"#2a6040", acc:"#4ade80" },
  { id:"conj",   label:"Sample+Conj Pad", sub:"A & P co-enter together",  frac:[0.09,0.20], bg:"#1a1108", bdr:"#7a5010", acc:"#fbbf24" },
  { id:"nc",     label:"NC Membrane",   sub:"capillary flow zone",  frac:[0.20,0.83], bg:"#07100a", bdr:"#1a3520", acc:"#38bdf8" },
  { id:"abs",    label:"Absorbent Pad", sub:"wicking sink",         frac:[0.83,1.00], bg:"#091410", bdr:"#1a3a25", acc:"#60a5fa" },
];

// ═══════════════════════════════════════════════════════════════════════════════
// GAUSSIAN SMOOTH (display only — imported above)
// STRIP CANVAS
// ═══════════════════════════════════════════════════════════════════════════════
function LFAStrip({ simData, frameIdx, userParams }) {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !simData) return;
    const rect = canvas.getBoundingClientRect();
    const DPR  = window.devicePixelRatio || 1;
    const W = rect.width, H = rect.height;
    if (canvas.width !== W*DPR || canvas.height !== H*DPR) {
      canvas.width = W*DPR; canvas.height = H*DPR;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(DPR,0,0,DPR,0,0);
    ctx.clearRect(0,0,W,H);

    const LABEL_H = 48, AXIS_H = 24;
    const SY = LABEL_H, SH = H - LABEL_H - AXIS_H;

    // Pad regions
    PADS.forEach(r => {
      const x1 = r.frac[0]*W, rw = (r.frac[1]-r.frac[0])*W;
      const g = ctx.createLinearGradient(x1,SY,x1,SY+SH);
      g.addColorStop(0,r.bg+"ff"); g.addColorStop(1,r.bg+"bb");
      ctx.fillStyle=g; ctx.fillRect(x1,SY,rw,SH);
      ctx.strokeStyle=r.bdr; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.rect(x1+0.75,SY+0.75,rw-1.5,SH-1.5); ctx.stroke();
      ctx.save(); ctx.globalAlpha=0.04; ctx.strokeStyle="#fff"; ctx.lineWidth=1;
      for(let i=1;i<9;i++){
        ctx.beginPath();ctx.moveTo(x1+2,SY+(i/9)*SH);ctx.lineTo(x1+rw-2,SY+(i/9)*SH);ctx.stroke();
      }
      ctx.restore();
      const cx = x1+rw/2;
      ctx.font="bold 10px 'DM Mono',monospace"; ctx.fillStyle=r.acc; ctx.textAlign="center";
      ctx.fillText(r.label.toUpperCase(), cx, SY-20);
      ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle=T.muted;
      ctx.fillText(r.sub, cx, SY-8);
    });

    const fi = Math.min(frameIdx, simData.times.length - 1);
    const { x, tl, cl, N, cl_reachable, lw } = simData;
    const Af  = simData.A[fi];
    const Pf  = simData.P[fi];
    const PAf = simData.PA[fi];
    const PAg_v = simData.PAg[fi];
    const PC_v  = simData.PC[fi];
    const t_now = simData.times[fi];

    // x→canvas mapping (only NC membrane zone)
    const xTotalMM = x[x.length - 1];
    const ncX0 = PADS[2].frac[0] * W;
    const ncX1w = PADS[2].frac[1] * W;
    const cX0 = ncX0 + 4, cX1 = ncX1w - 4;
    const cW = cX1 - cX0;
    const cY1 = SY + 8, cY2 = SY + SH - 8, cH = cY2 - cY1;
    const mmToCx = mm => cX0 + (mm / xTotalMM) * cW;
    const toX = i => cX0 + (i / (N - 1)) * cW;

    // Flow front — needs its OWN mapper spanning the FULL canvas width (all pad zones)
    // mmToCx only covers the NC zone (20%-83%). The flow front must be able to
    // reach the absorbent pad region, so map 0..xTotalMM across the full W.
    const mmToFull = mm => (mm / xTotalMM) * W;

    const t_arr = simData.times[fi];
    const x4val = 21311 * t_arr - 69505;
    if (x4val > 0) {
      const ffMM = Math.min(Math.pow(x4val, 0.25), xTotalMM);
      const ffX  = mmToFull(ffMM);
      ctx.save();
      ctx.strokeStyle = "rgba(0,212,255,0.35)"; ctx.lineWidth = 1.2;
      ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(ffX,SY+2); ctx.lineTo(ffX,SY+SH-2); ctx.stroke();
      ctx.setLineDash([]); ctx.restore();
      ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle="rgba(0,212,255,0.65)";
      ctx.textAlign="center"; ctx.fillText("▲ flow front", ffX, SY+SH+14);
    }

    // Left axis: A and PA
    const leftMax = Math.max(...Af, ...PAf, 1e-30) * 1.15;
    const toYL = v => cY2 - Math.min(Math.max(v / leftMax, 0), 1) * cH;
    const L_TICKS = 4;
    for (let t = 0; t <= L_TICKS; t++) {
      const val = leftMax * (t / L_TICKS);
      const cy  = toYL(val);
      ctx.strokeStyle = "rgba(56,189,248,0.3)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cX0, cy); ctx.lineTo(cX0-5, cy); ctx.stroke();
      ctx.strokeStyle = "rgba(56,189,248,0.05)";
      ctx.beginPath(); ctx.moveTo(cX0, cy); ctx.lineTo(cX1, cy); ctx.stroke();
      ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle="rgba(56,189,248,0.7)";
      ctx.textAlign="right"; ctx.fillText(val.toExponential(1), cX0-7, cy+3);
    }
    ctx.strokeStyle="rgba(56,189,248,0.25)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cX0,cY1); ctx.lineTo(cX0,cY2); ctx.stroke();
    ctx.save();
    ctx.fillStyle="rgba(56,189,248,0.6)"; ctx.font="9px 'DM Mono',monospace";
    ctx.textAlign="center"; ctx.translate(ncX0+8, SY+SH/2); ctx.rotate(-Math.PI/2);
    ctx.fillText("[A],[PA] (nM)", 0, 0); ctx.restore();

    // Right axis: P (detector — KEY species in competitive)
    const rightMax = Math.max(...Pf, 1e-30) * 1.15;
    const toYR = v => cY2 - Math.min(Math.max(v / rightMax, 0), 1) * cH;
    const R_TICKS = 4;
    for (let t = 0; t <= R_TICKS; t++) {
      const val = rightMax * (t / R_TICKS);
      const cy  = toYR(val);
      ctx.strokeStyle="rgba(251,191,36,0.3)"; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(cX1,cy); ctx.lineTo(cX1+5,cy); ctx.stroke();
      ctx.strokeStyle="rgba(251,191,36,0.04)"; ctx.setLineDash([2,4]);
      ctx.beginPath(); ctx.moveTo(cX0,cy); ctx.lineTo(cX1,cy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle="rgba(251,191,36,0.7)";
      ctx.textAlign="left"; ctx.fillText(val.toExponential(1), cX1+7, cy+3);
    }
    ctx.strokeStyle="rgba(251,191,36,0.25)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cX1,cY1); ctx.lineTo(cX1,cY2); ctx.stroke();
    ctx.save();
    ctx.fillStyle="rgba(251,191,36,0.65)"; ctx.font="9px 'DM Mono',monospace";
    ctx.textAlign="center"; ctx.translate(ncX1w-8, SY+SH/2); ctx.rotate(Math.PI/2);
    ctx.fillText("[P] free det. (nM)", 0, 0); ctx.restore();

    // Draw profiles
    const drawProfile = (rawVals, color, toYfn) => {
      if (!rawVals?.length) return;
      const vals = gaussSmooth(rawVals, 1.5);
      const [ri,gi,bi] = [0,2,4].map(o=>parseInt(color.slice(1+o,3+o),16));
      ctx.save();
      ctx.beginPath(); ctx.rect(cX0, cY1-1, cW, cH+2); ctx.clip();
      ctx.beginPath(); ctx.moveTo(toX(0),cY2);
      for (let i=0;i<N;i++) ctx.lineTo(toX(i),toYfn(vals[i]));
      ctx.lineTo(toX(N-1),cY2); ctx.closePath();
      const fg=ctx.createLinearGradient(0,cY1,0,cY2);
      fg.addColorStop(0,`rgba(${ri},${gi},${bi},0.28)`);
      fg.addColorStop(0.6,`rgba(${ri},${gi},${bi},0.08)`);
      fg.addColorStop(1,`rgba(${ri},${gi},${bi},0.01)`);
      ctx.fillStyle=fg; ctx.fill();
      ctx.beginPath(); let penDown=false;
      for(let i=0;i<N;i++){
        const px=toX(i),py=toYfn(vals[i]);
        if(!penDown){ctx.moveTo(px,py);penDown=true;}else ctx.lineTo(px,py);
      }
      ctx.strokeStyle=`rgba(${ri},${gi},${bi},0.95)`; ctx.lineWidth=2;
      ctx.shadowColor=color; ctx.shadowBlur=4; ctx.stroke(); ctx.shadowBlur=0;
      ctx.restore();
    };

    // P on right axis (amber); A and PA on left axis
    drawProfile(Pf,  T.P,  toYR);
    drawProfile(PAf, T.PA, toYL);
    drawProfile(Af,  T.A,  toYL);

    ctx.strokeStyle="rgba(255,255,255,0.05)"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cX0,cY2); ctx.lineTo(cX1,cY2); ctx.stroke();

    // Test & Control lines
    const peakPAg = Math.max(...simData.PAg) || 1e-30;
    const peakPC  = Math.max(...simData.PC)  || 1e-30;
    const sharedScale = Math.max(peakPAg, peakPC);
    drawLFALine(ctx, mmToCx(userParams.x_tl), SY, SH, T.PAg, PAg_v/sharedScale, "TEST", PAg_v, "PAg");
    if (cl_reachable)
      drawLFALine(ctx, mmToCx(userParams.x_cl), SY, SH, T.PC, PC_v/sharedScale, "CTRL", PC_v, "PC");

    // x-axis ticks
    ctx.font="8px 'DM Mono',monospace"; ctx.textAlign="center";
    [0,10,20,30,40,50,60,70,80,90,100].forEach(mm=>{
      if(mm>xTotalMM)return;
      const cx=cX0+(mm/xTotalMM)*cW;
      if(cx<cX0-1||cx>cX1+1)return;
      ctx.fillStyle="rgba(255,255,255,0.15)";
      ctx.fillRect(cx-0.5,SY+SH-4,1,4);
      ctx.fillStyle=T.muted;
      ctx.fillText(`${mm}`, cx, SY+SH+13);
    });
    ctx.font="9px 'DM Mono',monospace"; ctx.fillStyle=T.muted2; ctx.textAlign="center";
    ctx.fillText("Position (mm)", (cX0+cX1)/2, H-2);
  }, [simData, frameIdx, userParams]);

  function drawLFALine(ctx, lx, sy, sh, color, intensity, label, value) {
    const int = Math.pow(Math.min(Math.max(intensity,0),1), 0.35);
    const [ri,gi,bi]=[0,2,4].map(o=>parseInt(color.slice(1+o,3+o),16));
    ctx.save();
    ctx.strokeStyle=`rgba(${ri},${gi},${bi},${0.18+int*0.82})`;
    ctx.lineWidth=2.5+int*8;
    ctx.shadowColor=color; ctx.shadowBlur=4+int*26;
    ctx.beginPath(); ctx.moveTo(lx,sy+2); ctx.lineTo(lx,sy+sh-2); ctx.stroke();
    ctx.shadowBlur=0;
    const pW=32,pH=15,pX=lx-pW/2,pY=sy-38;
    ctx.fillStyle=`rgba(${ri},${gi},${bi},${0.08+int*0.55})`;
    ctx.beginPath(); ctx.roundRect(pX,pY,pW,pH,4); ctx.fill();
    ctx.strokeStyle=`rgba(${ri},${gi},${bi},${0.3+int*0.55})`; ctx.lineWidth=1; ctx.stroke();
    ctx.font="bold 9px 'DM Mono',monospace";
    ctx.fillStyle=`rgba(${ri},${gi},${bi},${0.5+int*0.5})`; ctx.textAlign="center";
    ctx.fillText(label,lx,pY+10);
    if(value>1e-20){
      ctx.font="8px 'DM Mono',monospace";
      ctx.fillStyle=`rgba(${ri},${gi},${bi},${0.4+int*0.45})`;
      ctx.fillText(`${value.toExponential(1)} nM`,lx,sy+sh+13);
    }
    ctx.restore();
  }

  const rafRef=useRef(null);
  useEffect(()=>{
    cancelAnimationFrame(rafRef.current);
    rafRef.current=requestAnimationFrame(draw);
    return()=>cancelAnimationFrame(rafRef.current);
  },[draw]);
  useEffect(()=>{
    const ro=new ResizeObserver(()=>{cancelAnimationFrame(rafRef.current);rafRef.current=requestAnimationFrame(draw);});
    if(canvasRef.current) ro.observe(canvasRef.current.parentElement);
    return()=>ro.disconnect();
  },[draw]);

  return <canvas ref={canvasRef}
    style={{width:"100%",height:230,display:"block",borderRadius:10,
      border:`1px solid ${T.border2}`,background:T.surface}}/>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIGNAL CHART
// ═══════════════════════════════════════════════════════════════════════════════
function SignalChart({ simData, frameIdx, setFrameIdx, tSampleEnd }) {
  const canvasRef=useRef(null);
  useEffect(()=>{
    const canvas=canvasRef.current;
    if(!canvas||!simData)return;
    const DPR=window.devicePixelRatio||1;
    const W=canvas.offsetWidth,H=canvas.offsetHeight;
    canvas.width=W*DPR; canvas.height=H*DPR;
    const ctx=canvas.getContext("2d"); ctx.scale(DPR,DPR);
    const {times, PAg, PC, CP, CAP} = simData;
    const pad={t:14,r:16,b:36,l:62};
    const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
    const maxT=times[times.length-1];
    const maxY=Math.max(...PAg,...PC)||1;
    ctx.clearRect(0,0,W,H);
    const tx=t=>pad.l+(t/maxT)*iW, ty=v=>pad.t+iH-(v/maxY)*iH;
    for(let i=0;i<=4;i++){
      const yy=pad.t+(i/4)*iH;
      ctx.strokeStyle="rgba(255,255,255,0.04)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(W-pad.r,yy);ctx.stroke();
      ctx.font="9px 'DM Mono',monospace";ctx.fillStyle=T.muted2;ctx.textAlign="right";
      ctx.fillText((maxY*(1-i/4)).toExponential(1),pad.l-4,yy+3);
    }
    for(let i=0;i<=5;i++){
      const xx=pad.l+(i/5)*iW;
      ctx.strokeStyle="rgba(255,255,255,0.04)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,pad.t+iH);ctx.stroke();
      ctx.font="9px 'DM Mono',monospace";ctx.fillStyle=T.muted2;ctx.textAlign="center";
      ctx.fillText(`${Math.round(maxT*i/5)}s`,xx,H-pad.b+12);
    }
    const drawC=(vals,color,fa=0.22,dash=[])=>{
      const[ri,gi,bi]=[0,2,4].map(o=>parseInt(color.slice(1+o,3+o),16));
      ctx.beginPath();ctx.moveTo(tx(times[0]),ty(0));
      times.forEach((t,i)=>ctx.lineTo(tx(t),ty(vals[i])));
      ctx.lineTo(tx(times[times.length-1]),ty(0));ctx.closePath();
      const g=ctx.createLinearGradient(0,pad.t,0,pad.t+iH);
      g.addColorStop(0,`rgba(${ri},${gi},${bi},${fa})`);
      g.addColorStop(1,`rgba(${ri},${gi},${bi},0.02)`);
      ctx.fillStyle=g;ctx.fill();
      ctx.beginPath();ctx.setLineDash(dash);
      times.forEach((t,i)=>i===0?ctx.moveTo(tx(t),ty(vals[i])):ctx.lineTo(tx(t),ty(vals[i])));
      ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.shadowColor=color;ctx.shadowBlur=7;ctx.stroke();
      ctx.shadowBlur=0;ctx.setLineDash([]);
    };
    // Draw CP (dashed) and CAP (dashed) individually, then PC total (solid)
    if(CP)  drawC(CP,  T.PC, 0.08, [3,3]);
    if(CAP) drawC(CAP, T.PA, 0.08, [4,2]);
    drawC(PC, T.PC, 0.14);
    drawC(PAg,T.PAg,0.22);
    // Legend
    [[T.PAg,"Test [AgP]"],[T.PC,"Ctrl total [CP+CAP]"],[T.PC,"  [CP] free P",true],[T.PA,"  [CAP] loaded P",true]].forEach(([c,l,dashed],i)=>{
      const lx=pad.l+8,ly=pad.t+13+i*13;
      ctx.strokeStyle=c;ctx.lineWidth=dashed?1.5:2;
      ctx.setLineDash(dashed?[3,3]:[]);
      ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+14,ly);ctx.stroke();ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=c;ctx.textAlign="left";ctx.fillText(l,lx+17,ly+3);
    });
    const fi=Math.min(frameIdx,times.length-1);
    const cx=tx(times[fi]);
    ctx.strokeStyle="rgba(251,191,36,0.45)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(cx,pad.t);ctx.lineTo(cx,pad.t+iH);ctx.stroke();ctx.setLineDash([]);
    [[PAg,T.PAg],[PC,T.PC]].forEach(([arr,c])=>{
      ctx.beginPath();ctx.arc(cx,ty(arr[fi]),4.5,0,Math.PI*2);
      ctx.fillStyle=c;ctx.shadowColor=c;ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0;
    });
    ctx.strokeStyle=T.border2;ctx.lineWidth=1;ctx.strokeRect(pad.l,pad.t,iW,iH);
    if(tSampleEnd&&tSampleEnd<maxT){
      const sx=tx(tSampleEnd);
      ctx.save();ctx.strokeStyle="rgba(74,222,128,0.55)";ctx.lineWidth=1.2;
      ctx.setLineDash([3,4]);
      ctx.beginPath();ctx.moveTo(sx,pad.t);ctx.lineTo(sx,pad.t+iH);ctx.stroke();
      ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(74,222,128,0.7)";
      ctx.textAlign="center";ctx.fillText("sample end",sx,pad.t+9);
      ctx.restore();
    }
    ctx.save();ctx.fillStyle=T.muted2;ctx.font="9px 'DM Mono',monospace";
    ctx.translate(11,pad.t+iH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";
    ctx.fillText("Signal (nM)",0,0);ctx.restore();
  },[simData,frameIdx,tSampleEnd]);

  const handleClick=useCallback(e=>{
    if(!simData)return;
    const canvas=canvasRef.current;
    const rect=canvas.getBoundingClientRect();
    const DPR=window.devicePixelRatio||1;
    const clickX=(e.clientX-rect.left)*(canvas.width/rect.width)/DPR;
    const padL=62,padR=16;
    const iW=canvas.width/DPR-padL-padR;
    const frac=(clickX-padL)/iW;
    setFrameIdx(Math.max(0,Math.min(simData.times.length-1,Math.round(frac*(simData.times.length-1)))));
  },[simData,setFrameIdx]);

  return <canvas ref={canvasRef} onClick={handleClick}
    style={{width:"100%",height:155,display:"block",borderRadius:8,
      border:`1px solid ${T.border}`,background:T.surface,cursor:"crosshair"}}/>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SWEEP CHART  (T/C logic inverted: POSITIVE when T/C < threshold)
// ═══════════════════════════════════════════════════════════════════════════════
function SweepChart({ data, sweepParam }) {
  const canvasRef = useRef(null);
  const isLogX = sweepParam === "Ao";

  useEffect(()=>{
    const canvas=canvasRef.current;
    if(!canvas||!data||!data.length)return;
    const DPR=window.devicePixelRatio||1;
    const W=canvas.offsetWidth,H=canvas.offsetHeight;
    canvas.width=W*DPR; canvas.height=H*DPR;
    const ctx=canvas.getContext("2d"); ctx.scale(DPR,DPR);
    ctx.clearRect(0,0,W,H);

    const PAD={t:22,r:72,b:46,l:64};
    const IW=W-PAD.l-PAD.r, IH=H-PAD.t-PAD.b;

    const xs=data.map(d=>d.x), tcs=data.map(d=>d.tc);
    const rps=data.map(d=>d.rpa), pcs=data.map(d=>d.pc);
    const minX=xs[0], maxX=xs[xs.length-1];
    const maxTC=Math.max(...tcs,0.5);
    const maxSIG=Math.max(...rps,...pcs,1e-30);

    const tx=v=>{
      if(isLogX){const lo=Math.log10(Math.max(minX,1e-15)),hi=Math.log10(maxX);
        return PAD.l+((Math.log10(Math.max(v,1e-15))-lo)/(hi-lo))*IW;}
      return PAD.l+((v-minX)/(maxX-minX||1))*IW;
    };
    const tyL=v=>PAD.t+IH-Math.min(Math.max(v/maxTC,0),1)*IH;
    const tyR=v=>PAD.t+IH-Math.min(Math.max(v/maxSIG,0),1)*IH;

    // Left axis (T/C — amber)
    for(let i=0;i<=5;i++){
      const val=maxTC*i/5,cy=tyL(val);
      ctx.strokeStyle="rgba(251,191,36,0.07)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(PAD.l,cy);ctx.lineTo(W-PAD.r,cy);ctx.stroke();
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(251,191,36,0.75)";
      ctx.textAlign="right";ctx.fillText(val.toFixed(2),PAD.l-5,cy+3);
    }
    ctx.strokeStyle="rgba(251,191,36,0.25)";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(PAD.l,PAD.t);ctx.lineTo(PAD.l,PAD.t+IH);ctx.stroke();

    // Right axis (signal — cyan)
    for(let i=1;i<=4;i++){
      const val=maxSIG*i/4,cy=tyR(val);
      ctx.strokeStyle="rgba(56,189,248,0.04)";ctx.lineWidth=0.5;ctx.setLineDash([2,4]);
      ctx.beginPath();ctx.moveTo(PAD.l,cy);ctx.lineTo(W-PAD.r,cy);ctx.stroke();ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(56,189,248,0.65)";
      ctx.textAlign="left";ctx.fillText(val.toExponential(1),W-PAD.r+4,cy+3);
    }
    ctx.strokeStyle="rgba(56,189,248,0.2)";ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(W-PAD.r,PAD.t);ctx.lineTo(W-PAD.r,PAD.t+IH);ctx.stroke();

    // LOD line — INVERTED: POSITIVE when T/C < LOD threshold (default 0.8)
    const LOD = 0.8;
    if(LOD<=maxTC){
      const cy=tyL(LOD);
      ctx.strokeStyle="rgba(74,222,128,0.65)";ctx.lineWidth=1.5;ctx.setLineDash([6,4]);
      ctx.beginPath();ctx.moveTo(PAD.l,cy);ctx.lineTo(W-PAD.r,cy);ctx.stroke();ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(74,222,128,0.85)";
      ctx.textAlign="left";ctx.fillText("LOD  T/C = 0.8  (↓ below = POSITIVE)", PAD.l+4, cy-3);
    }

    // x ticks
    ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(180,180,200,0.7)";ctx.textAlign="center";
    if(isLogX){
      const lo=Math.floor(Math.log10(minX)),hi=Math.ceil(Math.log10(maxX));
      for(let e=lo;e<=hi;e++){const v=Math.pow(10,e);if(v<minX*0.5||v>maxX*2)continue;
        const cx=tx(v);if(cx<PAD.l||cx>W-PAD.r)continue;
        ctx.strokeStyle="rgba(255,255,255,0.07)";ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(cx,PAD.t);ctx.lineTo(cx,PAD.t+IH);ctx.stroke();
        ctx.fillText(`1e${e}`,cx,H-PAD.b+13);}
    } else {
      xs.forEach(v=>{const cx=tx(v);
        ctx.strokeStyle="rgba(255,255,255,0.06)";ctx.lineWidth=0.5;
        ctx.beginPath();ctx.moveTo(cx,PAD.t);ctx.lineTo(cx,PAD.t+IH);ctx.stroke();
        ctx.fillText(v<1?v.toFixed(1):v>=10?v.toFixed(0):v.toFixed(1),cx,H-PAD.b+13);});
    }

    // Fill + lines
    ctx.beginPath();ctx.moveTo(tx(xs[0]),PAD.t+IH);
    xs.forEach((v,i)=>ctx.lineTo(tx(v),tyL(tcs[i])));
    ctx.lineTo(tx(xs[xs.length-1]),PAD.t+IH);ctx.closePath();
    const gf=ctx.createLinearGradient(0,PAD.t,0,PAD.t+IH);
    gf.addColorStop(0,"rgba(251,191,36,0.20)");gf.addColorStop(1,"rgba(251,191,36,0)");
    ctx.fillStyle=gf;ctx.fill();

    const hasNoise=data.length>0&&data[0].tc_lo!==null;
    if(hasNoise){
      ctx.beginPath();
      xs.forEach((v,i)=>ctx.lineTo(tx(v),tyL(data[i].tc_hi??tcs[i])));
      [...xs].reverse().forEach((v,i)=>{const di=xs.length-1-i;ctx.lineTo(tx(v),tyL(data[di].tc_lo??tcs[di]));});
      ctx.closePath();ctx.fillStyle="rgba(251,191,36,0.13)";ctx.fill();
      ctx.beginPath();
      xs.forEach((v,i)=>ctx.lineTo(tx(v),tyR(data[i].rpa_hi??rps[i])));
      [...xs].reverse().forEach((v,i)=>{const di=xs.length-1-i;ctx.lineTo(tx(v),tyR(data[di].rpa_lo??rps[di]));});
      ctx.closePath();ctx.fillStyle="rgba(74,222,128,0.09)";ctx.fill();
    }

    const drawL=(vals,color,toY,lw,dash)=>{
      ctx.save();ctx.beginPath();
      vals.forEach((v,i)=>{const px=tx(xs[i]),py=toY(v);i===0?ctx.moveTo(px,py):ctx.lineTo(px,py);});
      ctx.strokeStyle=color;ctx.lineWidth=lw;ctx.setLineDash(dash||[]);
      ctx.shadowColor=color;ctx.shadowBlur=5;ctx.stroke();ctx.shadowBlur=0;ctx.setLineDash([]);ctx.restore();
    };
    drawL(rps,"rgba(74,222,128,0.7)",tyR,1.5,[4,3]);
    drawL(pcs,"rgba(167,139,250,0.7)",tyR,1.5,[2,4]);
    drawL(tcs,"rgba(251,191,36,1.0)",tyL,2.5);

    // Dots — green if T/C < LOD (POSITIVE in competitive), amber otherwise
    xs.forEach((v,i)=>{
      const px=tx(v),py=tyL(tcs[i]);
      const isPos=tcs[i]<LOD&&tcs[i]>0;
      ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);
      ctx.fillStyle=isPos?"rgba(74,222,128,0.9)":"rgba(251,191,36,0.9)";
      ctx.shadowColor=isPos?"#4ade80":"#fbbf24";ctx.shadowBlur=7;ctx.fill();ctx.shadowBlur=0;
    });

    // LOD crossing — competitive: T/C DESCENDS, detect drop below threshold
    for(let i=1;i<xs.length;i++){
      if(tcs[i-1]>=LOD&&tcs[i]<LOD){
        const frac=(LOD-tcs[i-1])/(tcs[i]-tcs[i-1]);
        const lodX=xs[i-1]+frac*(xs[i]-xs[i-1]);
        const lx=tx(lodX);
        ctx.strokeStyle="rgba(74,222,128,0.75)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
        ctx.beginPath();ctx.moveTo(lx,PAD.t);ctx.lineTo(lx,PAD.t+IH);ctx.stroke();ctx.setLineDash([]);
        ctx.font="bold 9px 'DM Mono',monospace";ctx.fillStyle="#4ade80";ctx.textAlign="center";
        const lbl=isLogX?`LOD ≈ ${lodX.toExponential(2)} nM`:`LOD ≈ ${lodX.toFixed(2)}`;
        ctx.fillText(lbl,lx,PAD.t+13);break;
      }
    }

    // Legend
    [["rgba(251,191,36,1)","T/C ratio (left axis)",[]],
     ["rgba(74,222,128,0.8)","[PAg] signal (right)",[4,3]],
     ["rgba(167,139,250,0.8)","[PC]  signal (right)",[2,4]],
     ["rgba(74,222,128,0.8)","LOD (T/C=0.8)",[6,4]],
    ].forEach(([color,label,dash],i)=>{
      const lx=PAD.l+4,ly=PAD.t+13+i*13;
      ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dash);
      ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+16,ly);ctx.stroke();ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=color;ctx.textAlign="left";
      ctx.fillText(label,lx+20,ly+3);
    });

    // Axis labels
    const xLabel=sweepParam==="Ao"?"Analyte [A₀] (nM)  —  log scale  (competition kicks in near Po)"
      :sweepParam==="Po"?"Detector [P₀] (nM)"
      :sweepParam==="Ag_tl"?"Immobilised Antigen [Ag_TL] (nM)"
      :"Test line position (mm)";
    ctx.font="9px 'DM Mono',monospace";ctx.fillStyle="rgba(180,180,200,0.65)";
    ctx.textAlign="center";ctx.fillText(xLabel,PAD.l+IW/2,H-3);
    ctx.save();ctx.font="9px 'DM Mono',monospace";ctx.fillStyle="rgba(251,191,36,0.7)";
    ctx.translate(11,PAD.t+IH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";
    ctx.fillText("T/C ratio",0,0);ctx.restore();
    ctx.save();ctx.font="9px 'DM Mono',monospace";ctx.fillStyle="rgba(74,222,128,0.6)";
    ctx.translate(W-7,PAD.t+IH/2);ctx.rotate(Math.PI/2);ctx.textAlign="center";
    ctx.fillText("Signal (nM)",0,0);ctx.restore();
    ctx.strokeStyle="rgba(40,50,70,1)";ctx.lineWidth=1;
    ctx.strokeRect(PAD.l,PAD.t,IW,IH);
  },[data,sweepParam,isLogX]);

  return <canvas ref={canvasRef}
    style={{width:"100%",height:300,display:"block",borderRadius:8,
      border:`1px solid ${T.border}`,background:T.surface}}/>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// REUSABLE UI ATOMS
// ═══════════════════════════════════════════════════════════════════════════════
function LogSlider({ label, value, logMin, logMax, onChange, unit, color }) {
  const logVal = Math.log10(Math.max(value, Math.pow(10, logMin)));
  const pct    = ((logVal - logMin) / (logMax - logMin)) * 100;
  const decades = [];
  for (let e = Math.ceil(logMin); e <= Math.floor(logMax); e++) decades.push(e);
  return (
    <div style={{marginBottom:16}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{color:T.muted2,fontSize:11}}>{label}</span>
        <span style={{color:color||T.accent,fontSize:11,fontWeight:700}}>
          {value.toExponential(2)} <span style={{color:T.muted,fontWeight:400}}>{unit}</span>
        </span>
      </div>
      <div style={{position:"relative",height:4,background:T.border,borderRadius:2,marginBottom:6}}>
        <div style={{position:"absolute",left:0,top:0,height:"100%",width:`${pct}%`,
          background:`linear-gradient(90deg,${color||T.accent}77,${color||T.accent})`,
          borderRadius:2,boxShadow:`0 0 7px ${color||T.accent}44`}}/>
        <div style={{position:"absolute",top:"50%",left:`${pct}%`,
          transform:"translate(-50%,-50%)",width:11,height:11,borderRadius:"50%",
          background:color||T.accent,pointerEvents:"none",
          boxShadow:`0 0 6px ${color||T.accent}`,border:"2px solid #0b100d"}}/>
        <input type="range" min={logMin} max={logMax}
          step={(logMax-logMin)/1000} value={logVal}
          onChange={e=>onChange(Math.pow(10,Number(e.target.value)))}
          style={{position:"absolute",inset:0,opacity:0,width:"100%",cursor:"pointer",margin:0,height:20,top:-8}}/>
      </div>
      <div style={{position:"relative",height:14}}>
        {decades.map(e=>{
          const tp=((e-logMin)/(logMax-logMin))*100;
          return (<div key={e} style={{position:"absolute",left:`${tp}%`,transform:"translateX(-50%)",textAlign:"center"}}>
            <div style={{width:1,height:4,background:T.border,margin:"0 auto 1px"}}/>
            <span style={{fontSize:7,color:T.muted,fontFamily:"'DM Mono',monospace",whiteSpace:"nowrap"}}>
              {e>=0?`1e+${e}`:`1e${e}`}</span>
          </div>);
        })}
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, unit, color, fmt }) {
  const pct=((value-min)/(max-min))*100;
  const disp=fmt?fmt(value):(step<1?value.toFixed(2):Math.round(value));
  return (
    <div style={{marginBottom:14}}>
      <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
        <span style={{color:T.muted2,fontSize:11}}>{label}</span>
        <span style={{color:color||T.accent,fontSize:11,fontWeight:700}}>
          {disp} <span style={{color:T.muted,fontWeight:400}}>{unit}</span>
        </span>
      </div>
      <div style={{position:"relative",height:4,background:T.border,borderRadius:2}}>
        <div style={{position:"absolute",left:0,top:0,height:"100%",width:`${pct}%`,
          background:`linear-gradient(90deg,${color||T.accent}77,${color||T.accent})`,
          borderRadius:2,boxShadow:`0 0 7px ${color||T.accent}44`}}/>
        <div style={{position:"absolute",top:"50%",left:`${pct}%`,
          transform:"translate(-50%,-50%)",width:11,height:11,borderRadius:"50%",
          background:color||T.accent,pointerEvents:"none",
          boxShadow:`0 0 6px ${color||T.accent}`,border:"2px solid #0b100d"}}/>
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e=>onChange(Number(e.target.value))}
          style={{position:"absolute",inset:0,opacity:0,width:"100%",cursor:"pointer",margin:0,height:20,top:-8}}/>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, color }) {
  return (
    <div style={{background:T.card,border:`1px solid ${color}22`,borderRadius:8,
      padding:"8px 11px",flex:1,minWidth:80}}>
      <div style={{color:T.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:3}}>{label}</div>
      <div style={{color,fontSize:13,fontWeight:700,fontFamily:"'DM Mono',monospace",
        textShadow:`0 0 8px ${color}55`}}>
        {value}<span style={{fontSize:9,color:T.muted,marginLeft:2}}>{unit}</span>
      </div>
    </div>
  );
}

function ParamRow({ meta, value, onChange }) {
  const [raw, setRaw]     = useState(meta.sci ? value.toExponential(2) : String(value));
  const [err, setErr]     = useState(null);
  const [dirty, setDirty] = useState(false);
  useEffect(()=>{
    setRaw(meta.sci ? value.toExponential(2) : String(Number(value.toFixed(6))));
    setErr(null); setDirty(false);
  },[value]);
  const commit = () => {
    const res = validateParam(meta, raw);
    if (!res.ok) { setErr(res.msg); return; }
    setErr(null); setDirty(false); onChange(res.v);
  };
  return (
    <div style={{marginBottom:8}}>
      <div style={{display:"flex",alignItems:"center",gap:6}}>
        <span style={{color:T.muted2,fontSize:10,flex:1,minWidth:0}}>{meta.label}</span>
        <div style={{position:"relative",display:"flex",alignItems:"center",gap:4}}>
          <input value={raw}
            onChange={e=>{setRaw(e.target.value);setDirty(true);setErr(null);}}
            onBlur={commit} onKeyDown={e=>e.key==="Enter"&&commit()}
            style={{background:T.surface,border:`1px solid ${err?T.err:dirty?T.accent:T.border}`,
              borderRadius:5,color:err?T.err:T.text,padding:"3px 6px",fontSize:10,
              fontFamily:"'DM Mono',monospace",width:100,outline:"none",transition:"border 0.15s"}}/>
          <span style={{color:T.muted,fontSize:9,minWidth:40}}>{meta.unit}</span>
        </div>
        {err&&<span style={{color:T.err,fontSize:9,whiteSpace:"nowrap"}}>⚠ {err}</span>}
        {!err&&dirty&&<span style={{color:T.ok,fontSize:9}}>↵</span>}
      </div>
      <div style={{color:T.muted,fontSize:8,marginTop:1,textAlign:"right",paddingRight:46}}>
        [{meta.min} – {meta.max}]
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// LIMIT ROW  (one row inside the Limits sub-tab)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Lets the user override the upper bound of a single slider.
//
// Validation rules (applied at commit, i.e. blur or Enter):
//   1. Must parse as a finite number (NaN / "" / Infinity rejected)
//   2. Must be strictly greater than the slider's hard floor `meta.min`
//   3. Must be a positive value
//
// On a successful commit we call onChange(newMax). The parent then:
//   • updates the limit in state, and
//   • clamps the current slider value down if it exceeds the new max
//     (so the slider thumb never falls off the right edge).
// ═══════════════════════════════════════════════════════════════════════════════
function LimitRow({ meta, currentValue, currentMax, onChange }) {
  const [raw, setRaw]     = useState(String(currentMax));
  const [err, setErr]     = useState(null);
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    setRaw(String(currentMax));
    setErr(null);
    setDirty(false);
  }, [currentMax]);

  const commit = () => {
    const v = parseFloat(raw);
    if (!isFinite(v))   { setErr("Not a valid number"); return; }
    if (v <= 0)         { setErr("Must be positive");   return; }
    if (v <= meta.min)  { setErr(`Must be > ${meta.min}`); return; }
    setErr(null);
    setDirty(false);
    onChange(v);
  };

  const clampWarn = !err && dirty && parseFloat(raw) < currentValue;

  return (
    <div style={{marginBottom:10}}>
      <div style={{display:"flex",alignItems:"center",gap:6}}>
        <span style={{color:T.muted2,fontSize:10,flex:1,minWidth:0}}>{meta.label}</span>
        <div style={{display:"flex",alignItems:"center",gap:4}}>
          <span style={{color:T.muted,fontSize:9}}>max</span>
          <input
            value={raw}
            onChange={e => { setRaw(e.target.value); setDirty(true); setErr(null); }}
            onBlur={commit}
            onKeyDown={e => e.key === "Enter" && commit()}
            style={{
              background:T.surface,
              border:`1px solid ${err?T.err:dirty?T.accent:T.border}`,
              borderRadius:5,
              color:err?T.err:T.text,
              padding:"3px 6px",
              fontSize:10,
              fontFamily:"'DM Mono',monospace",
              width:80, outline:"none", transition:"border 0.15s",
            }}
          />
          <span style={{color:T.muted,fontSize:9,minWidth:24}}>{meta.unit}</span>
        </div>
        {!err && dirty && <span style={{color:T.ok,fontSize:9}}>↵</span>}
      </div>
      <div style={{color:T.muted,fontSize:8,marginTop:2,paddingLeft:2,
        display:"flex",justifyContent:"space-between"}}>
        <span>min ≥ {meta.min} · current value: {Number(currentValue).toFixed(currentValue<10?2:1)}</span>
        <span>default: {meta.def}</span>
      </div>
      {err && (
        <div style={{color:T.err,fontSize:9,marginTop:2,fontStyle:"italic"}}>
          ⚠ {err}
        </div>
      )}
      {clampWarn && (
        <div style={{color:T.warn,fontSize:9,marginTop:2,fontStyle:"italic"}}>
          ⚠ current value ({currentValue}) will be clamped to {parseFloat(raw)}
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// OD ASSESSMENT  (inverted verdict logic)
// ═══════════════════════════════════════════════════════════════════════════════
function ODBars({ PAg_val, PCval, CPval, CAPval, peakPAg, peakPC, peakCP, peakCAP, clReachable }) {
  const sharedPeak = Math.max(peakPAg, peakPC, 1e-30);
  const tPct  = Math.min(100, (PAg_val / sharedPeak) * 100);
  const cPct  = clReachable ? Math.min(100, (PCval / sharedPeak) * 100) : 0;
  const cpPct = clReachable ? Math.min(100, ((CPval||0) / sharedPeak) * 100) : 0;
  const capPct= clReachable ? Math.min(100, ((CAPval||0) / sharedPeak) * 100) : 0;

  // COMPETITIVE: POSITIVE when T/C < 0.8 (dim test line)
  const LOD_TC = 0.8;
  const valid   = clReachable && peakPC > 0;
  const TC_ratio = valid ? peakPAg / peakPC : Infinity;
  // Positive = test line dimmer than 80% of control
  const positive = valid && TC_ratio < LOD_TC && TC_ratio > 0;

  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:10,padding:14}}>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:10}}>
        Optical Density Assessment
      </div>
      {/* Competitive inversion note */}
      <div style={{background:"#1a1500",border:"1px solid #fbbf2433",borderRadius:6,
        padding:"5px 10px",marginBottom:10,fontSize:9,color:"#fbbf24",lineHeight:1.5}}>
        ⚠ Competitive assay: DIM test line = POSITIVE.  BRIGHT test line = NEGATIVE.
      </div>
      <div style={{display:"flex",gap:12,alignItems:"stretch",marginBottom:10}}>
        {/* Mock strip */}
        <div style={{width:70,background:"#080d09",border:`1px solid ${T.border2}`,
          borderRadius:8,position:"relative",overflow:"hidden",minHeight:80}}>
          <div style={{position:"absolute",left:0,right:0,top:"38%",
            height:Math.max(2,3+tPct/12),background:T.PAg,
            opacity:0.15+tPct/100*0.85,boxShadow:`0 0 ${4+tPct/4}px ${T.PAg}`}}/>
          <div style={{position:"absolute",left:0,right:0,top:"63%",
            height:Math.max(2,3+cPct/12),background:T.PC,
            opacity:0.15+cPct/100*0.85,boxShadow:`0 0 ${4+cPct/4}px ${T.PC}`}}/>
          <div style={{position:"absolute",top:"35%",left:3,fontSize:7,color:T.PAg,fontFamily:"'DM Mono',monospace"}}>T</div>
          <div style={{position:"absolute",top:"60%",left:3,fontSize:7,color:T.PC,fontFamily:"'DM Mono',monospace"}}>C</div>
        </div>
        <div style={{flex:1,display:"flex",flexDirection:"column",gap:8}}>
          <div>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
              <span style={{color:T.PAg}}>Test (T) — [PAg]</span>
              <span style={{color:T.muted2,fontFamily:"'DM Mono',monospace"}}>{tPct.toFixed(0)}%  {PAg_val.toExponential(2)} nM</span>
            </div>
            <div style={{height:7,background:T.border,borderRadius:4}}>
              <div style={{height:"100%",width:`${tPct}%`,borderRadius:4,
                background:T.PAg,boxShadow:`0 0 7px ${T.PAg}55`,transition:"width 0.12s"}}/>
            </div>
          </div>
          {clReachable ? (
            <div>
              {/* Total control bar */}
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
                <span style={{color:T.PC}}>Control (C) — [CP + CAP]</span>
                <span style={{color:T.muted2,fontFamily:"'DM Mono',monospace"}}>{cPct.toFixed(0)}%  {PCval.toExponential(2)} nM</span>
              </div>
              {/* Stacked bar: CP (violet) + CAP (rose) */}
              <div style={{height:7,background:T.border,borderRadius:4,position:"relative",overflow:"hidden"}}>
                <div style={{position:"absolute",left:0,top:0,height:"100%",
                  width:`${cpPct}%`,background:T.PC,transition:"width 0.12s"}}/>
                <div style={{position:"absolute",left:`${cpPct}%`,top:0,height:"100%",
                  width:`${capPct}%`,background:T.PA,opacity:0.85,transition:"width 0.12s left 0.12s"}}/>
              </div>
              <div style={{display:"flex",gap:10,marginTop:3,fontSize:8,color:T.muted2}}>
                <span><span style={{color:T.PC}}>■</span> [CP] free P: {(CPval||0).toExponential(2)} nM</span>
                <span><span style={{color:T.PA}}>■</span> [CAP] AP complex: {(CAPval||0).toExponential(2)} nM</span>
              </div>
            </div>
          ) : (
            <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 0"}}>
              <div style={{width:20,height:2,background:T.muted,borderRadius:1}}/>
              <span style={{fontSize:10,color:T.muted}}>Control — outside strip</span>
            </div>
          )}
        </div>
      </div>

      {valid && (()=>{
        const tcStr = TC_ratio === Infinity ? "—" : TC_ratio.toFixed(3);
        const rows = [
          { label:"T/C ratio",         val:tcStr,
            note: TC_ratio<LOD_TC?"< 0.8 — POSITIVE (dim T line)":"≥ 0.8 — NEGATIVE (bright T line)",
            color: TC_ratio<LOD_TC ? T.ok : T.muted2 },
          { label:"Test peak [PAg]",   val:peakPAg.toExponential(2)+"nM",
            note:"free detector captured at immobilised antigen", color:T.PAg },
          { label:"Ctrl peak [PC]",    val:peakPC.toExponential(2)+"nM",
            note:"free detector captured at control Ab", color:T.PC },
        ];
        return (
          <div style={{background:T.surface,borderRadius:7,border:`1px solid ${T.border}`,
            padding:"8px 10px",marginBottom:8}}>
            <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:6}}>
              Signal Diagnostics
            </div>
            {rows.map(r=>(
              <div key={r.label} style={{display:"flex",alignItems:"baseline",gap:6,marginBottom:4,flexWrap:"wrap"}}>
                <span style={{fontSize:9,color:T.muted2,minWidth:110}}>{r.label}</span>
                <span style={{fontSize:10,fontFamily:"'DM Mono',monospace",fontWeight:700,color:r.color}}>{r.val}</span>
                <span style={{fontSize:9,color:T.muted,fontStyle:"italic"}}>{r.note}</span>
              </div>
            ))}
          </div>
        );
      })()}

      <div style={{
        background: !clReachable?"#10141a":positive?"#0a1a10":"#0a1020",
        border:`1px solid ${!clReachable?T.border:positive?"#166534":"#1e3a6a"}`,
        borderRadius:7,padding:"8px 12px",display:"flex",alignItems:"center",gap:10}}>
        <span style={{fontSize:18}}>
          {!clReachable?"📏":positive?"🔴":"🟢"}
        </span>
        <div>
          <div style={{fontSize:11,fontWeight:700,marginBottom:1,
            color:!clReachable?T.muted2:positive?"#f87171":"#86efac"}}>
            {!clReachable?"Control line outside strip"
              :positive?"Positive (analyte detected)"
              :"Negative (no analyte / below LOD)"}
          </div>
          <div style={{fontSize:10,color:T.muted2}}>
            {!clReachable?"Move control line inside strip or increase strip length"
              :positive?`T/C = ${TC_ratio.toFixed(3)} < ${LOD_TC} — test line suppressed by analyte`
              :`T/C = ${TC_ratio === Infinity ? "—" : TC_ratio.toFixed(3)} ≥ ${LOD_TC} — free P flowing to test line`}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// REPORT SECTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════════
function Section({ title, color, children }) {
  return (
    <div style={{marginBottom:20}}>
      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:10,
        paddingBottom:6,borderBottom:`1px solid ${color}44`}}>
        <div style={{width:3,height:16,background:color,borderRadius:2}}/>
        <span style={{fontSize:11,fontWeight:700,color,letterSpacing:1,
          textTransform:"uppercase",fontFamily:"'DM Mono',monospace"}}>{title}</span>
      </div>
      {children}
    </div>
  );
}
function Row({ label, value, unit, color, warn, note }) {
  return (
    <div style={{display:"flex",alignItems:"baseline",gap:8,
      padding:"4px 0",borderBottom:`1px solid ${T.border}`}}>
      <span style={{color:T.muted2,fontSize:10,minWidth:200,flexShrink:0}}>{label}</span>
      <span style={{color:warn?T.warn:(color||T.text),fontSize:11,
        fontFamily:"'DM Mono',monospace",fontWeight:600}}>{value}</span>
      {unit&&<span style={{color:T.muted,fontSize:9}}>{unit}</span>}
      {note&&<span style={{color:warn?T.warn:T.muted,fontSize:9,fontStyle:"italic",marginLeft:4}}>{note}</span>}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIM REPORT
// ═══════════════════════════════════════════════════════════════════════════════
function SimReport({ simData, userParams, kineticParams, physParams }) {
  if (!simData) return (
    <div style={{display:"flex",alignItems:"center",justifyContent:"center",
      height:300,color:T.muted,fontSize:12,fontFamily:"'DM Mono',monospace"}}>
      Run a simulation first to generate the report.
    </div>
  );

  const fi  = simData.times.length - 1;

  // ── Mobile species at final frame ─────────────────────────────────────────
  const Af  = simData.A[fi];
  const Pf  = simData.P[fi];
  const PAf = simData.PA[fi];

  // ── Immobilised species at final frame ────────────────────────────────────
  // Test line:    AgP  = detector captured by immobilised antigen
  // Control line: CP   = free detector captured by anti-species Ab
  //               CAP  = AP complex captured by anti-species Ab  (same Rc)
  //               PC   = CP + CAP  (total gold visible at control line)
  const AgP  = simData.PAg[fi];
  const CP   = simData.CP?.[fi]  ?? 0;
  const CAP  = simData.CAP?.[fi] ?? 0;
  const PC   = simData.PC[fi];     // = CP + CAP

  const t_end_actual = simData.times[fi];
  const { x, tl, cl, N, cl_reachable, t_sample_end } = simData;
  const { Ao, Po, Ag_tl, Ro_cl, x_tl, x_cl, t_end } = userParams;
  const { Ka1, Kd1, Ka2, Kd2, Ka_cl, Kd_cl } = kineticParams;
  const { strip_L, phi, tau, U_max, sample_vol } = physParams;
  const KD1 = Kd1 / Ka1;
  const KD2 = Kd2 / Ka2;
  const KD_cl = Kd_cl / Ka_cl;

  // ── Spatial stats ─────────────────────────────────────────────────────────
  const peakA   = Math.max(...Af),  peakA_x  = x[Af.indexOf(peakA)];
  const peakP   = Math.max(...Pf),  peakP_x  = x[Pf.indexOf(peakP)];
  const peakPA  = Math.max(...PAf), peakPA_x = x[PAf.indexOf(peakPA)];
  const A_at_tl  = Af[tl],  P_at_tl  = Pf[tl],  PA_at_tl  = PAf[tl];
  const A_at_cl  = cl_reachable ? Af[cl]  : 0;
  const P_at_cl  = cl_reachable ? Pf[cl]  : 0;
  const PA_at_cl = cl_reachable ? PAf[cl] : 0;

  const x4val = 21311 * t_end_actual - 69505;
  const flowFront = x4val > 0 ? Math.min(x4val ** 0.25, strip_L) : 0;

  // ── Line occupancy ────────────────────────────────────────────────────────
  const Ag_free   = Math.max(Ag_tl - AgP, 0);
  const Ag_occ    = Ag_tl > 0 ? (AgP / Ag_tl) * 100 : 0;
  // Total Rc consumed = CP + CAP
  const Rc_consumed = CP + CAP;
  const Rc_free   = Math.max(Ro_cl - Rc_consumed, 0);
  const Rc_occ    = Ro_cl > 0 ? (Rc_consumed / Ro_cl) * 100 : 0;
  const CAP_frac  = Rc_consumed > 0 ? (CAP / Rc_consumed) * 100 : 0;

  // ── Pad fluxes ────────────────────────────────────────────────────────────
  const padA  = simData.pad_A[fi];
  const padP  = simData.pad_P[fi];
  const padPA = simData.pad_PA[fi];

  // ── Verdict ───────────────────────────────────────────────────────────────
  const peakAgP = Math.max(...simData.PAg);
  const peakPC  = Math.max(...simData.PC);
  const TC      = peakPC > 0 ? peakAgP / peakPC : Infinity;
  const LOD_TC  = 0.8;
  const valid    = cl_reachable && peakPC > 0;
  const positive = valid && TC < LOD_TC && TC > 0;
  const verdictColor = !valid ? T.warn : positive ? T.err : T.PAg;
  const verdictText  = !valid ? "INVALID" : positive ? "POSITIVE" : "NEGATIVE";

  // fraction of detector that has been sequestered as AP at the test line
  const AP_frac_tl = (P_at_tl + PA_at_tl) > 0
    ? PA_at_tl / (P_at_tl + PA_at_tl) : 0;

  // ── Download ──────────────────────────────────────────────────────────────
  const downloadReport = () => {
    try {
      const lines = [
        "COMPETITIVE LFA SIMULATION REPORT",
        "=".repeat(60),
        `Generated: ${new Date().toISOString()}`,
        "",
        "SPECIES ACCOUNTING",
        "  Flowing  : P (free detector), AP (analyte-detector complex)",
        "  Test line: AgP  — free P captured by immobilised antigen",
        "  Ctrl line: CP   — free P captured by anti-species Ab",
        "             CAP  — AP complex captured by anti-species Ab",
        "             PC   — total gold at control = CP + CAP",
        "",
        "SIGNAL LOGIC",
        "  Low  [A] -> P free -> AgP high -> T bright -> T/C high -> NEGATIVE",
        "  High [A] -> AP forms -> P depleted -> AgP low -> T dim -> T/C low -> POSITIVE",
        "",
        "1. PARAMETERS",
        `Ao=${Ao.toExponential(2)} nM  Po=${Po.toExponential(2)} nM`,
        `Ag_tl=${Ag_tl} nM  Ro_cl=${Ro_cl} nM`,
        `x_tl=${x_tl} mm  x_cl=${x_cl} mm  t_end=${t_end} s`,
        `KD1=${KD1.toExponential(3)} nM  KD2=${KD2.toExponential(3)} nM  KD_cl=${KD_cl.toExponential(3)} nM`,
        `strip_L=${strip_L}mm  phi=${phi}  tau=${tau}  U_max=${U_max}mm/s`,
        "",
        "2. MOBILE SPECIES at t_end",
        `[P]  peak=${peakP.toExponential(3)} nM at ${peakP_x.toFixed(1)}mm | at TL: ${P_at_tl.toExponential(3)} nM | at CL: ${P_at_cl.toExponential(3)} nM`,
        `[AP] peak=${peakPA.toExponential(3)} nM at ${peakPA_x.toFixed(1)}mm | at TL: ${PA_at_tl.toExponential(3)} nM | at CL: ${PA_at_cl.toExponential(3)} nM`,
        `[A]  peak=${peakA.toExponential(3)} nM at ${peakA_x.toFixed(1)}mm`,
        "",
        "3. TEST LINE (AgP)",
        `AgP = ${AgP.toExponential(3)} nM  |  Ag_free = ${Ag_free.toExponential(3)} nM  |  occupancy = ${Ag_occ.toFixed(1)}%`,
        "",
        "4. CONTROL LINE",
        `CP  (free P captured)  = ${CP.toExponential(3)} nM`,
        `CAP (AP complex capt.) = ${CAP.toExponential(3)} nM`,
        `PC  (total gold)       = ${PC.toExponential(3)} nM`,
        `Rc occupancy = ${Rc_occ.toFixed(1)}%  |  CAP fraction of ctrl = ${CAP_frac.toFixed(1)}%`,
        "",
        "5. VERDICT",
        `T/C = ${TC === Infinity ? "inf" : TC.toFixed(4)}  (threshold ${LOD_TC})`,
        `Result: ${verdictText}`,
      ];
      const text     = lines.join("\n");
      const dataUri  = "data:text/plain;charset=utf-8," + encodeURIComponent(text);
      const a        = document.createElement("a");
      a.href         = dataUri;
      a.download     = "competitive_lfa_report.txt";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (err) {
      // Fallback: open report text in a new tab
      const text = `COMPETITIVE LFA REPORT\n\nAo=${Ao.toExponential(2)} nM  Po=${Po.toExponential(2)} nM  Ag_tl=${Ag_tl} nM\nAgP=${AgP.toExponential(3)} nM  CP=${CP.toExponential(3)} nM  CAP=${CAP.toExponential(3)} nM  PC=${PC.toExponential(3)} nM\nT/C=${TC===Infinity?"inf":TC.toFixed(4)}  Result: ${verdictText}`;
      const w = window.open("", "_blank");
      if (w) { w.document.write("<pre>" + text + "</pre>"); w.document.close(); }
    }
  };

  return (
    <div style={{fontFamily:"'DM Mono',monospace",fontSize:11}}>

      {/* Header */}
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <div style={{fontSize:9,color:T.muted,letterSpacing:2}}>COMPETITIVE LFA — SIMULATION REPORT</div>
        <button onClick={downloadReport}
          style={{background:T.surface,border:`1px solid ${T.border}`,color:T.muted2,
            padding:"4px 12px",borderRadius:6,cursor:"pointer",fontFamily:"inherit",fontSize:10}}>
          ↓ Download
        </button>
      </div>

      {/* Species map */}
      <div style={{background:"#0d1a0d",border:"1px solid #fbbf2433",borderRadius:8,
        padding:"10px 14px",marginBottom:14,fontSize:9,lineHeight:1.9}}>
        <div style={{color:"#fbbf24",fontWeight:700,marginBottom:4,letterSpacing:1}}>
          SPECIES ACCOUNTING
        </div>
        {[
          ["Flowing",    [["P","Free detector — key signal carrier",T.P],["AP","Analyte-detector complex — sequesters P",T.PA]]],
          ["Test line",  [["AgP","Free P bound to immobilised antigen → signal",T.PAg]]],
          ["Ctrl line",  [["CP","Free P captured by anti-species Ab",T.PC],["CAP","AP complex captured by anti-species Ab",T.PA],["PC = CP+CAP","Total gold at control line",T.PC]]],
        ].map(([loc, species])=>(
          <div key={loc} style={{display:"flex",gap:8,marginBottom:2,alignItems:"baseline"}}>
            <span style={{color:T.muted,minWidth:70,fontSize:8,textTransform:"uppercase",letterSpacing:1}}>{loc}</span>
            <span>
              {species.map(([sym,desc,c],i)=>(
                <span key={sym}>
                  <span style={{color:c,fontWeight:700}}>{sym}</span>
                  <span style={{color:T.muted2}}> {desc}</span>
                  {i<species.length-1 && <span style={{color:T.border2}}>  ·  </span>}
                </span>
              ))}
            </span>
          </div>
        ))}
      </div>

      {/* Section 1: Parameters */}
      <Section title="1. Assay Parameters" color={T.accent}>
        <Row label="Analyte [A₀]"                value={Ao.toExponential(2)}      unit="nM"/>
        <Row label="Detector [P₀]"               value={Po.toExponential(2)}      unit="nM"/>
        <Row label="Immobilised antigen [Ag_TL]"  value={`${Ag_tl}`}               unit="nM"/>
        <Row label="Control receptor [Rc₀]"       value={`${Ro_cl}`}               unit="nM"/>
        <Row label="Test line position x_tl"      value={`${x_tl}`}                unit="mm"/>
        <Row label="Control line position x_cl"   value={`${x_cl}`}                unit="mm"/>
        <Row label="Run time"                     value={`${t_end}`}               unit="s"/>
        <Row label="Sample exhausted at"          value={t_sample_end.toFixed(0)}  unit="s"
          note={t_sample_end < t_end ? "buffer washes after this" : "sample lasts full run"}/>
        <Row label="KD₁ (A-P in solution)"        value={KD1.toExponential(3)}     unit="nM"/>
        <Row label="KD₂ (P-Ag_TL at test line)"  value={KD2.toExponential(3)}     unit="nM"/>
        <Row label="KD_cl (P/AP-Rc at ctrl line)" value={KD_cl.toExponential(3)}   unit="nM"/>
        <Row label="Ao / KD₁ (competition ratio)" value={(Ao/KD1).toExponential(2)}
          color={Ao/KD1>1?T.PA:T.warn}
          note={Ao/KD1>1?"strong competition — P effectively sequestered":"weak competition — increase Ao or decrease KD₁"}/>
      </Section>

      {/* Section 2: Mobile species */}
      <Section title="2. Mobile Species — Flowing" color={T.P}>
        <div style={{fontSize:9,color:T.muted,marginBottom:8,fontStyle:"italic"}}>
          P and AP flow through the strip. AP cannot bind the test line antigen.
        </div>
        <Row label="Flow front at t_end"
          value={`${flowFront.toFixed(1)} mm`}
          color={flowFront>=x_cl?T.PAg:flowFront>=x_tl?T.warn:T.err}
          note={flowFront>=x_cl?"past both lines":flowFront>=x_tl?"past test line only":"did not reach test line"}/>
        <div style={{overflowX:"auto",marginTop:8}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
            <thead>
              <tr style={{background:T.surface}}>
                {["Species","Role","Peak conc.","Peak loc.","At test line","At ctrl line","Escaped to pad"].map(h=>(
                  <th key={h} style={{padding:"5px 8px",textAlign:"left",color:T.muted,
                    fontSize:8,fontWeight:600,borderBottom:`1px solid ${T.border2}`}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { sp:"[P] free det.", role:"signal carrier", c:T.P,
                  pk:peakP,   pkx:peakP_x,   tl_v:P_at_tl,   cl_v:P_at_cl,   pad:padP },
                { sp:"[AP] complex",  role:"sequesters P",   c:T.PA,
                  pk:peakPA,  pkx:peakPA_x,  tl_v:PA_at_tl,  cl_v:PA_at_cl,  pad:padPA },
                { sp:"[A] analyte",   role:"competes for P", c:T.A,
                  pk:peakA,   pkx:peakA_x,   tl_v:A_at_tl,   cl_v:A_at_cl,   pad:padA },
              ].map((r,i)=>(
                <tr key={r.sp} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                  <td style={{padding:"5px 8px",color:r.c,fontWeight:600,fontSize:10}}>{r.sp}</td>
                  <td style={{padding:"5px 8px",color:T.muted,fontSize:9,fontStyle:"italic"}}>{r.role}</td>
                  <td style={{padding:"5px 8px",fontFamily:"monospace",fontSize:10}}>{r.pk.toExponential(3)}</td>
                  <td style={{padding:"5px 8px",color:T.muted2,fontSize:10}}>{r.pkx.toFixed(1)} mm</td>
                  <td style={{padding:"5px 8px",fontFamily:"monospace",fontSize:10}}>{r.tl_v.toExponential(3)}</td>
                  <td style={{padding:"5px 8px",color:cl_reachable?T.text:T.muted,fontFamily:"monospace",fontSize:10}}>
                    {cl_reachable ? r.cl_v.toExponential(3) : "—"}</td>
                  <td style={{padding:"5px 8px",color:r.pad>1e-10?T.warn:T.muted2,fontFamily:"monospace",fontSize:10}}>
                    {r.pad.toExponential(2)} nM·mm</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{marginTop:8}}>
          <Row label="AP fraction at test line"
            value={`${(AP_frac_tl*100).toFixed(1)}%`}
            color={AP_frac_tl>0.5?T.PA:T.PAg}
            note={AP_frac_tl>0.5?"detector mostly AP-bound → test line dim → POSITIVE likely"
                               :"detector mostly free → test line bright → NEGATIVE"}/>
        </div>
      </Section>

      {/* Section 3: Test line */}
      <Section title="3. Test Line — Immobilised Antigen" color={T.PAg}>
        <div style={{background:T.surface,borderRadius:8,padding:12,border:`1px solid ${T.PAg}33`}}>
          <div style={{color:T.PAg,fontWeight:700,fontSize:11,marginBottom:8}}>
            AgP  —  free P captured by Ag_TL  —  x = {x_tl} mm
          </div>
          <Row label="[AgP] signal"              value={AgP.toExponential(3)}      unit="nM"  color={T.PAg}
            note="only free P contributes — AP cannot bind"/>
          <Row label="Free antigen [Ag_TL] left" value={Ag_free.toExponential(3)}  unit="nM"/>
          <Row label="Antigen occupancy"          value={`${Ag_occ.toFixed(1)}%`}
            color={Ag_occ>50?T.PAg:T.warn}
            note={Ag_occ>50?"antigen well occupied":"consider increasing Ag_tl"}/>
          <Row label="[P] free arriving at TL"   value={P_at_tl.toExponential(3)}  unit="nM"/>
          <Row label="[AP] arriving at TL"        value={PA_at_tl.toExponential(3)} unit="nM"
            note="cannot bind Ag_TL — passes through"/>
        </div>
      </Section>

      {/* Section 4: Control line */}
      <Section title="4. Control Line — Anti-species Ab" color={T.PC}>
        <div style={{background:T.surface,borderRadius:8,padding:12,
          border:`1px solid ${T.PC}33`,opacity:cl_reachable?1:0.5}}>
          <div style={{color:T.PC,fontWeight:700,fontSize:11,marginBottom:8}}>
            CP + CAP  —  captures P AND AP  —  x = {x_cl} mm
            {!cl_reachable&&<span style={{color:T.warn,fontSize:9,marginLeft:8}}>⚠ OUTSIDE STRIP</span>}
          </div>
          {cl_reachable ? <>
            <Row label="[CP]  free P captured"    value={CP.toExponential(3)}   unit="nM"  color={T.PC}
              note="free detector binds anti-species Ab"/>
            <Row label="[CAP] AP complex captured" value={CAP.toExponential(3)}  unit="nM"  color={T.PA}
              note="AP complex also binds — same detector backbone"/>
            <Row label="[PC]  total gold at ctrl"  value={PC.toExponential(3)}   unit="nM"
              color={T.PC} note="= CP + CAP — what the reader sees"/>
            <Row label="CAP as % of ctrl gold"     value={`${CAP_frac.toFixed(1)}%`}
              color={CAP_frac>30?T.PA:T.muted2}
              note={CAP_frac>30?"significant AP reaching control — high [A]":"mostly free P at control"}/>
            <Row label="Rc free remaining"         value={Rc_free.toExponential(3)} unit="nM"/>
            <Row label="Rc occupancy"              value={`${Rc_occ.toFixed(1)}%`}
              color={Rc_occ>5?T.PAg:T.warn}
              warn={Rc_occ<0.5}
              note={Rc_occ<0.5?"control nearly invisible — increase Ro_cl or Po":undefined}/>
            <Row label="[P] free arriving at CL"  value={P_at_cl.toExponential(3)}  unit="nM"/>
            <Row label="[AP] arriving at CL"       value={PA_at_cl.toExponential(3)} unit="nM"/>
          </> : (
            <div style={{color:T.warn,fontSize:10,padding:"8px 0"}}>
              Control line is outside the strip. Extend strip length or reduce x_cl.
            </div>
          )}
        </div>
      </Section>

      {/* Section 5: Verdict */}
      <Section title="5. Assay Verdict" color={verdictColor}>
        <div style={{display:"flex",gap:10,marginBottom:12,flexWrap:"wrap"}}>
          <div style={{background:`${verdictColor}15`,border:`1px solid ${verdictColor}55`,
            borderRadius:8,padding:"10px 16px",flex:1,minWidth:140}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>VERDICT</div>
            <div style={{fontSize:22,fontWeight:800,color:verdictColor,
              fontFamily:"'Syne',sans-serif"}}>{verdictText}</div>
            <div style={{fontSize:9,color:T.muted2,marginTop:2}}>
              {!valid ? "control absent/unreachable"
                : positive ? "T/C < 0.8 — test line suppressed by analyte"
                : "T/C ≥ 0.8 — free P reaching test line"}
            </div>
          </div>
          <div style={{background:T.surface,border:`1px solid ${T.border}`,
            borderRadius:8,padding:"10px 16px",flex:1,minWidth:120}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>T/C RATIO</div>
            <div style={{fontSize:18,fontWeight:700,
              color:TC<LOD_TC?T.PAg:T.muted2,
              fontFamily:"'DM Mono',monospace"}}>
              {TC===Infinity?"∞":TC.toFixed(3)}
            </div>
            <div style={{fontSize:9,color:T.muted,marginTop:2}}>
              threshold = {LOD_TC} (below = POSITIVE)
            </div>
          </div>
          <div style={{background:T.surface,border:`1px solid ${T.border}`,
            borderRadius:8,padding:"10px 16px",flex:1,minWidth:120}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>TEST / CTRL</div>
            <div style={{fontSize:11,fontFamily:"'DM Mono',monospace",lineHeight:1.8}}>
              <div style={{color:T.PAg}}>AgP  {peakAgP.toExponential(2)} nM</div>
              <div style={{color:T.PC}}>CP   {Math.max(...(simData.CP||[0])).toExponential(2)} nM</div>
              <div style={{color:T.PA}}>CAP  {Math.max(...(simData.CAP||[0])).toExponential(2)} nM</div>
              <div style={{color:T.PC,borderTop:`1px solid ${T.border}`,marginTop:2,paddingTop:2}}>
                PC   {peakPC.toExponential(2)} nM
              </div>
            </div>
          </div>
        </div>
        <Row label="AP fraction at test line"
          value={`${(AP_frac_tl*100).toFixed(1)}%`}
          color={AP_frac_tl>0.5?T.PA:T.PAg}
          note={AP_frac_tl>0.5?"detector mostly sequestered → dim test → POSITIVE"
                             :"detector mostly free → bright test → NEGATIVE"}/>
        <Row label="Ao / KD₁ (competition)"
          value={(Ao/KD1).toExponential(2)}
          color={Ao/KD1>1?T.PA:T.warn}
          note={Ao/KD1>1?"analyte sequesters P effectively":"weak binding — need higher Ao or lower KD₁"}/>
        <Row label="Po / KD₂ (test line binding)"
          value={(Po/KD2).toExponential(2)}
          color={Po/KD2>10?T.PAg:T.warn}
          note={Po/KD2>10?"P binds Ag_TL efficiently when free":"partial antigen capture"}/>
        <Row label="CAP/PC at control"
          value={`${CAP_frac.toFixed(1)}%`}
          color={CAP_frac>30?T.PA:T.muted2}
          note="% of control gold that is AP-loaded — increases with [A]"/>
        <Row label="Rc occupancy at control"
          value={`${Rc_occ.toFixed(1)}%`}
          color={Rc_occ>5?T.PAg:T.warn}
          warn={Rc_occ<0.5}
          note={Rc_occ<0.5?"control faint — increase Ro_cl":"control line visible ✓"}/>
      </Section>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function CompetitiveLFASimulator() {
  const [userParams, setUserParams] = useState({
    Ao: 1, Po: 6, Ag_tl: 10,
    Ro_cl: 8, x_tl: 15, x_cl: 28, t_end: 1800,
  });
  const [kineticParams, setKineticParams] = useState(defaultKinetic());
  const [physParams,    setPhysParams]    = useState(defaultPhysical());

  // ── Slider upper-limit overrides ───────────────────────────────────────────
  // The user can extend the upper bound of any concentration slider via the
  // "Limits" sub-tab. State here lives only for the session — no persistence.
  const [sliderLimits, setSliderLimits] = useState(defaultSliderLimits());

  // Updater that:
  //   1. writes the new max into sliderLimits, and
  //   2. clamps the corresponding userParams value down if it now exceeds max.
  // Without step 2 the slider thumb would render off-screen.
  const updateSliderLimit = useCallback((key, newMax) => {
    setSliderLimits(prev => ({ ...prev, [key]: newMax }));
    setUserParams(prev =>
      prev[key] > newMax ? { ...prev, [key]: newMax } : prev);
  }, []);

  const [simData,   setSimData]   = useState(null);
  const [running,   setRunning]   = useState(false);
  const [frameIdx,  setFrameIdx]  = useState(0);
  const [playing,   setPlaying]   = useState(false);
  const [tab,       setTab]       = useState("sim");
  const [paramTab,  setParamTab]  = useState("assay");
  const [sweepParam,setSweepParam]= useState("Ao");
  const [sweepData, setSweepData] = useState(null);
  const [sweeping,  setSweeping]  = useState(false);
  const [sweepConfig, setSweepConfig] = useState({
    Ao:     { logMin:-2, logMax:3,  nPoints:16 },
    Po:     { min:0.1,  max:50,    nSteps:12  },
    Ag_tl:  { min:1,    max:100,   nSteps:10  },
    x_tl:   { min:10,   max:70,    nSteps:10  },
  });
  const [noiseEnabled, setNoiseEnabled] = useState(false);
  const [noiseConfig, setNoiseConfig] = useState({
    cv_Ao:0.10, cv_Ka:0.15, cv_conj:0.10, cv_flow:0.08, n_replicates:5,
  });

  const intervalRef = useRef(null);
  useEffect(()=>{ handleRun(); }, []);

  useEffect(()=>{
    clearInterval(intervalRef.current);
    if(playing&&simData){
      intervalRef.current=setInterval(()=>{
        setFrameIdx(i=>{
          if(i>=simData.times.length-1){setPlaying(false);return i;}
          return i+1;
        });
      },50);
    }
    return()=>clearInterval(intervalRef.current);
  },[playing,simData]);

  const handleRun = useCallback(()=>{
    setRunning(true); setPlaying(false); setFrameIdx(0);
    setTimeout(()=>{
      try { setSimData(runSimulation(userParams,kineticParams,physParams,160)); }
      catch(e){ console.error(e); }
      setRunning(false);
    },15);
  },[userParams,kineticParams,physParams]);

  const randn=()=>{
    let u=0,v=0;
    while(u===0)u=Math.random();
    while(v===0)v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  };

  const handleSweep = useCallback(()=>{
    setSweeping(true); setSweepData(null);
    setTimeout(()=>{
      try {
        const cfg=sweepConfig[sweepParam];
        const points = sweepParam==="Ao"
          ? Array.from({length:cfg.nPoints},(_,i)=>
              Math.pow(10,cfg.logMin+(cfg.logMax-cfg.logMin)*i/(cfg.nPoints-1)))
          : Array.from({length:cfg.nSteps},(_,i)=>
              cfg.min+(cfg.max-cfg.min)*i/(cfg.nSteps-1));

        const data=points.map(v=>{
          const baseU={...userParams,[sweepParam]:v};
          if(!noiseEnabled){
            const r=runSimulation(baseU,kineticParams,physParams,60);
            const pag=r.PAg[r.PAg.length-1]||0;
            const pc =r.PC[r.PC.length-1] ||0;
            return{x:v,rpa:pag,pc,tc:pc>1e-30?pag/pc:Infinity,
                   rpa_lo:null,rpa_hi:null,tc_lo:null,tc_hi:null};
          }
          const nc=noiseConfig;
          const rpas=[],pcs=[],tcs=[];
          for(let i=0;i<nc.n_replicates;i++){
            const rn=()=>randn();
            const nu={...baseU,Ao:Math.max(0,baseU.Ao*(1+nc.cv_Ao*rn()))};
            const nk={...kineticParams,
              Ka1:Math.max(1e-8,kineticParams.Ka1*(1+nc.cv_Ka*rn())),
              Ka2:Math.max(1e-8,kineticParams.Ka2*(1+nc.cv_Ka*rn())),
              Kd1:Math.max(1e-8,kineticParams.Kd1*(1+nc.cv_Ka*rn())),
              Kd2:Math.max(1e-8,kineticParams.Kd2*(1+nc.cv_Ka*rn()))};
            const np={...physParams,
              conj_burst:Math.min(1,Math.max(0,physParams.conj_burst*(1+nc.cv_conj*rn()))),
              flow_c:Math.max(100,physParams.flow_c*(1+nc.cv_flow*rn()))};
            const r=runSimulation(nu,nk,np,60);
            const pag_=r.PAg[r.PAg.length-1]||0;
            const pc_ =r.PC[r.PC.length-1] ||0;
            rpas.push(pag_);pcs.push(pc_);tcs.push(pc_>1e-30?pag_/pc_:0);
          }
          const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
          const std =a=>{const m=mean(a);return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length);};
          const rm=mean(rpas),pm=mean(pcs),tm=mean(tcs),ts=std(tcs),rs=std(rpas);
          return{x:v,rpa:rm,pc:pm,tc:tm,
                 rpa_lo:Math.max(0,rm-rs),rpa_hi:rm+rs,
                 tc_lo:Math.max(0,tm-ts),tc_hi:tm+ts};
        });
        setSweepData(data);
      } catch(e){console.error("Sweep error:",e);}
      setSweeping(false);
    },20);
  },[userParams,kineticParams,physParams,sweepParam,sweepConfig,noiseEnabled,noiseConfig]);

  const fi      = simData ? Math.min(frameIdx,simData.times.length-1) : 0;
  const curT    = simData ? simData.times[fi]  : 0;
  const curPAg  = simData ? simData.PAg[fi]    : 0;
  const curPC   = simData ? simData.PC[fi]     : 0;
  const curCP   = simData ? (simData.CP?.[fi]  ?? 0) : 0;
  const curCAP  = simData ? (simData.CAP?.[fi] ?? 0) : 0;
  const peakPAg = simData ? Math.max(...simData.PAg) : 0;
  const peakPC  = simData ? Math.max(...simData.PC)  : 0;
  const peakCP  = simData ? Math.max(...(simData.CP  || [0])) : 0;
  const peakCAP = simData ? Math.max(...(simData.CAP || [0])) : 0;

  const set_up = k => v => setUserParams(p=>({...p,[k]:v}));

  return (
    <div style={{background:T.bg,minHeight:"100vh",color:T.text,
      fontFamily:"'DM Mono',monospace",padding:"14px 16px"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-thumb{background:#1a2820;border-radius:2px;}
        input[type=range]{accent-color:#fbbf24;}
      `}</style>

      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
        <div>
          <div style={{fontSize:9,color:T.muted,letterSpacing:4,textTransform:"uppercase",marginBottom:2}}>
            Phase 2 · Competitive Assay · Inverted Signal
          </div>
          <h1 style={{fontFamily:"'Syne',sans-serif",fontSize:22,fontWeight:800,letterSpacing:-0.5,
            background:"linear-gradient(110deg,#fbbf24,#4ade80)",
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            Competitive LFA Simulator
          </h1>
        </div>
        <div style={{display:"flex",gap:6}}>
          {[["sim","⚗ Simulation"],["sweep","⟳ Sweep"],["info","ƒ Model"],["report","📋 Report"]].map(([id,lbl])=>(
            <button key={id} onClick={()=>{if(id==="sweep")setSweepData(null);setTab(id);}}
              style={{background:tab===id?"#fbbf2418":T.card,
                border:`1px solid ${tab===id?"#fbbf24":T.border}`,
                color:tab===id?"#fbbf24":T.muted2,padding:"5px 12px",borderRadius:6,
                cursor:"pointer",fontFamily:"inherit",fontSize:11,transition:"all 0.15s"}}>
              {lbl}
            </button>
          ))}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"248px 1fr",gap:14}}>

        {/* LEFT PANEL */}
        <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14,
          display:"flex",flexDirection:"column",gap:0,maxHeight:"calc(100vh - 80px)",overflowY:"auto"}}>

          <div style={{display:"flex",gap:4,marginBottom:14,background:T.surface,borderRadius:7,padding:3}}>
            {[["assay","Assay"],["kinetic","Kinetics"],["physical","Physical"],["limits","Limits"]].map(([id,lbl])=>(
              <button key={id} onClick={()=>setParamTab(id)}
                style={{flex:1,background:paramTab===id?"#1a2a20":T.surface,
                  border:`1px solid ${paramTab===id?T.borderHi:"transparent"}`,
                  color:paramTab===id?T.text:T.muted,padding:"5px 0",borderRadius:5,
                  cursor:"pointer",fontFamily:"inherit",fontSize:10,transition:"all 0.15s"}}>
                {lbl}
              </button>
            ))}
          </div>

          {/* Assay params — Ag_tl replaces Ro in sandwich */}
          {paramTab==="assay" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
              Assay Parameters
            </div>
            <LogSlider label="Analyte [A₀]" value={userParams.Ao}
              logMin={-2} logMax={3} unit="nM" color={T.A}
              onChange={set_up("Ao")}/>
            <Slider label="Detector [P₀]"          value={userParams.Po}     min={0.1} max={sliderLimits.Po}    step={0.5}   unit="nM" color={T.P}   onChange={set_up("Po")}/>
            <Slider label="Immob. antigen [Ag_TL]" value={userParams.Ag_tl}  min={0.1} max={sliderLimits.Ag_tl} step={0.5}   unit="nM" color={T.PAg} onChange={set_up("Ag_tl")}/>
            <Slider label="Ctrl receptor [Rc]"      value={userParams.Ro_cl} min={0.1} max={sliderLimits.Ro_cl} step={0.5}   unit="nM" color={T.PC}  onChange={set_up("Ro_cl")}/>
            <Slider label="Test line x"             value={userParams.x_tl}  min={10}  max={50}  step={5}     unit="mm" color={T.PAg} onChange={set_up("x_tl")}/>
            <Slider label="Control line x"          value={userParams.x_cl}
              min={Math.min(userParams.x_tl+10, physParams.strip_L-5)}
              max={physParams.strip_L+20} step={5} unit="mm" color={T.PC} onChange={set_up("x_cl")}/>
            <Slider label="Run Time"                value={userParams.t_end} min={120} max={1800} step={60}   unit="s"  color={T.accent} onChange={set_up("t_end")}/>

            {/* Live competitive signal hint */}
            {simData && (() => {
              const tc = peakPC > 0 ? peakPAg / peakPC : Infinity;
              const pos = tc < 0.8 && tc > 0;
              return (
                <div style={{background:pos?"#0a1a10":"#0a1020",border:`1px solid ${pos?"#166534":"#1e3a6a"}`,
                  borderRadius:6,padding:"7px 10px",marginTop:4,marginBottom:8,fontSize:9,lineHeight:1.6}}>
                  <div style={{color:pos?"#86efac":"#93c5fd",fontWeight:700,marginBottom:2}}>
                    {pos?"🔴 POSITIVE — test line suppressed":"🟢 NEGATIVE — free P reaching test line"}
                  </div>
                  <div style={{color:T.muted2}}>
                    T/C = {tc===Infinity?"—":tc.toFixed(3)}  |  threshold 0.8
                  </div>
                </div>
              );
            })()}
          </>}

          {paramTab==="kinetic" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>Kinetic Parameters</div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Ka₁/Kd₁: analyte-detector in solution. Ka₂/Kd₂: detector-antigen at test line.
            </div>
            {KINETIC_META.map(m=>(
              <ParamRow key={m.key} meta={m} value={kineticParams[m.key]}
                onChange={v=>setKineticParams(p=>({...p,[m.key]:v}))}/>
            ))}
            <button onClick={()=>setKineticParams(defaultKinetic())}
              style={{marginTop:8,background:T.surface,border:`1px solid ${T.border}`,
                color:T.muted2,padding:"5px 0",borderRadius:6,cursor:"pointer",
                fontFamily:"inherit",fontSize:10,width:"100%"}}>
              Reset to Defaults
            </button>
          </>}

          {paramTab==="physical" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>Physical Parameters</div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Membrane & strip geometry. Same physics as sandwich model.
            </div>
            {PHYSICAL_META.map(m=>(
              <ParamRow key={m.key} meta={m} value={physParams[m.key]}
                onChange={v=>setPhysParams(p=>({...p,[m.key]:v}))}/>
            ))}
            <button onClick={()=>setPhysParams(defaultPhysical())}
              style={{marginTop:8,background:T.surface,border:`1px solid ${T.border}`,
                color:T.muted2,padding:"5px 0",borderRadius:6,cursor:"pointer",
                fontFamily:"inherit",fontSize:10,width:"100%"}}>
              Reset to Defaults
            </button>
          </>}

          {/* Limits params (extend slider upper bounds) */}
          {paramTab==="limits" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>Slider Upper Limits</div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Extend the upper bound of any concentration slider. Enter a positive
              number above the slider's floor, then press Enter or click away.
              If the current slider value is above your new max, it will be clamped down.
            </div>
            {SLIDER_LIMITS_META.map(m=>(
              <LimitRow key={m.key} meta={m}
                currentValue={userParams[m.key]}
                currentMax={sliderLimits[m.key]}
                onChange={v=>updateSliderLimit(m.key,v)}/>
            ))}
            <button onClick={()=>{
                // Reset all limits to defaults; then clamp any params that
                // would now sit above the restored defaults.
                const def = defaultSliderLimits();
                setSliderLimits(def);
                setUserParams(p => {
                  const next = {...p};
                  for (const k of Object.keys(def))
                    if (next[k] > def[k]) next[k] = def[k];
                  return next;
                });
              }}
              style={{marginTop:8,background:T.surface,border:`1px solid ${T.border}`,
                color:T.muted2,padding:"5px 0",borderRadius:6,cursor:"pointer",
                fontFamily:"inherit",fontSize:10,width:"100%"}}>
              Reset Limits to Defaults
            </button>
          </>}

          {/* Geometry warnings */}
          {(()=>{
            const warns=[];
            if(userParams.x_cl>=physParams.strip_L)
              warns.push(`Control line (${userParams.x_cl}mm) outside strip (${physParams.strip_L}mm)`);
            if(userParams.x_tl>=physParams.strip_L-5)
              warns.push(`Test line (${userParams.x_tl}mm) too close to strip end`);
            if(userParams.x_cl<=userParams.x_tl)
              warns.push("Control line must be downstream of test line");
            return warns.length>0?(
              <div style={{background:"#1a0f08",border:"1px solid #7f3a1a",borderRadius:7,padding:"8px 10px",marginBottom:8}}>
                {warns.map((w,i)=>(<div key={i} style={{color:"#fb923c",fontSize:9,lineHeight:1.6}}>⚠ {w}</div>))}
              </div>
            ):null;
          })()}

          <button onClick={handleRun} disabled={running} style={{
            width:"100%",padding:"9px 0",marginTop:14,
            background:running?"#1a2820":"linear-gradient(135deg,#fbbf24,#16a34a)",
            border:"none",color:running?T.muted:"#050805",
            borderRadius:8,cursor:running?"not-allowed":"pointer",
            fontFamily:"inherit",fontSize:12,fontWeight:700,
            transition:"all 0.2s",letterSpacing:0.5}}>
            {running?"⧗ Simulating…":"▶ Run Simulation"}
          </button>
        </div>

        {/* RIGHT PANEL */}
        <div style={{display:"flex",flexDirection:"column",gap:10}}>

          {/* SIMULATION TAB */}
          {tab==="sim" && <>
            <LFAStrip simData={simData} frameIdx={fi} userParams={userParams}/>

            {/* Playback controls */}
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:10,padding:10}}>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                <button onClick={()=>setPlaying(p=>!p)} disabled={!simData}
                  style={{background:playing?"#fbbf2422":"#1a2820",border:`1px solid ${playing?"#fbbf24":T.border}`,
                    color:playing?"#fbbf24":T.muted2,padding:"4px 14px",borderRadius:6,
                    cursor:simData?"pointer":"not-allowed",fontFamily:"inherit",fontSize:12}}>
                  {playing?"⏸":"▶"}
                </button>
                <input type="range" min={0} max={simData?simData.times.length-1:0}
                  value={fi} onChange={e=>setFrameIdx(Number(e.target.value))}
                  disabled={!simData}
                  style={{flex:1,accentColor:"#fbbf24"}}/>
                <span style={{color:T.accent,fontSize:11,fontFamily:"'DM Mono',monospace",minWidth:55,textAlign:"right"}}>
                  {curT.toFixed(0)}s
                </span>
              </div>
              <div style={{display:"flex",gap:6,flexWrap:"wrap"}}>
                <Stat label="Test [AgP]"    value={curPAg.toExponential(2)} unit="nM" color={T.PAg}/>
                <Stat label="Ctrl [CP]"     value={curCP.toExponential(2)}  unit="nM" color={T.PC}/>
                <Stat label="Ctrl [CAP]"    value={curCAP.toExponential(2)} unit="nM" color={T.PA}/>
                <Stat label="T/C ratio"
                  value={curPC>0?(curPAg/curPC).toFixed(3):"—"} unit=""
                  color={(curPC>0&&curPAg/curPC<0.8)?T.ok:T.muted2}/>
              </div>
            </div>

            {/* Signal chart */}
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:10,padding:10}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:6}}>
                Signal vs Time  <span style={{color:T.muted2,fontSize:8,fontWeight:400,letterSpacing:0}}>
                  (click chart to seek)</span>
              </div>
              <SignalChart simData={simData} frameIdx={fi} setFrameIdx={setFrameIdx}
                tSampleEnd={simData?.t_sample_end}/>
            </div>

            {/* OD assessment */}
            <ODBars PAg_val={curPAg} PCval={curPC} CPval={curCP} CAPval={curCAP}
              peakPAg={peakPAg} peakPC={peakPC} peakCP={peakCP} peakCAP={peakCAP}
              clReachable={simData?.cl_reachable??false}/>

            {/* Species legend */}
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:10,padding:12}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
                Species Legend
              </div>
              <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                {[
                  [T.A,  "[A]",    "Free analyte  (flowing)"],
                  [T.P,  "[P]",    "Free detector  (flowing) ← test line signal source"],
                  [T.PA, "[AP]",   "Analyte-detector complex  (flowing) ← sequesters P"],
                  [T.PAg,"[AgP]",  "Test line: free P captured by Ag_TL"],
                  [T.PC, "[CP]",   "Control line: free P captured"],
                  [T.PA, "[CAP]",  "Control line: AP complex captured"],
                ].map(([c,sym,desc])=>(
                  <div key={sym} style={{display:"flex",alignItems:"center",gap:5,
                    background:T.surface,borderRadius:6,padding:"4px 8px",
                    border:`1px solid ${c}22`}}>
                    <div style={{width:8,height:8,borderRadius:"50%",background:c,boxShadow:`0 0 4px ${c}`}}/>
                    <span style={{color:c,fontWeight:700,fontSize:9}}>{sym}</span>
                    <span style={{color:T.muted,fontSize:8}}>{desc}</span>
                  </div>
                ))}
              </div>
            </div>
          </>}

          {/* SWEEP TAB */}
          {tab==="sweep" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
                Parameter Sweep
              </div>
              {/* Inverted LOD note */}
              <div style={{background:"#1a1500",border:"1px solid #fbbf2433",borderRadius:6,
                padding:"5px 10px",marginBottom:12,fontSize:9,color:"#fbbf24"}}>
                Competitive: LOD crossing when T/C drops BELOW 0.8 (test line dims) = analyte detected
              </div>

              <div style={{display:"flex",gap:8,alignItems:"center",flexWrap:"wrap",marginBottom:12}}>
                <span style={{fontSize:10,color:T.muted2}}>Sweep parameter:</span>
                {[["Ao","[A₀] analyte"],["Po","[P₀] detector"],["Ag_tl","[Ag_TL] antigen"],["x_tl","Test line pos."]].map(([k,lbl])=>(
                  <button key={k} onClick={()=>setSweepParam(k)}
                    style={{background:sweepParam===k?"#fbbf2422":T.surface,
                      border:`1px solid ${sweepParam===k?"#fbbf24":T.border}`,
                      color:sweepParam===k?"#fbbf24":T.muted2,
                      padding:"4px 10px",borderRadius:5,cursor:"pointer",
                      fontFamily:"inherit",fontSize:10}}>
                    {lbl}
                  </button>
                ))}
              </div>

              {/* Sweep range config */}
              <div style={{background:T.surface,border:`1px solid ${T.border}`,borderRadius:8,
                padding:"10px 12px",marginBottom:12,fontSize:10}}>
                <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:8}}>
                  Range Configuration
                </div>
                {sweepParam==="Ao"?(
                  <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
                    {[["logMin","Log min"],["logMax","Log max"],["nPoints","Points"]].map(([k,lbl])=>(
                      <label key={k} style={{display:"flex",alignItems:"center",gap:4,color:T.muted2,fontSize:9}}>
                        {lbl}
                        <input type="number" value={sweepConfig.Ao[k]}
                          onChange={e=>setSweepConfig(c=>({...c,Ao:{...c.Ao,[k]:Number(e.target.value)}}))}
                          style={{width:50,background:T.card,border:`1px solid ${T.border}`,
                            borderRadius:4,color:T.text,padding:"2px 5px",fontSize:10,
                            fontFamily:"'DM Mono',monospace"}}/>
                      </label>
                    ))}
                  </div>
                ):(
                  <div style={{display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
                    {[["min","Min"],["max","Max"],["nSteps","Steps"]].map(([k,lbl])=>(
                      <label key={k} style={{display:"flex",alignItems:"center",gap:4,color:T.muted2,fontSize:9}}>
                        {lbl}
                        <input type="number" value={sweepConfig[sweepParam]?.[k]||""}
                          onChange={e=>setSweepConfig(c=>({...c,[sweepParam]:{...c[sweepParam],[k]:Number(e.target.value)}}))}
                          style={{width:50,background:T.card,border:`1px solid ${T.border}`,
                            borderRadius:4,color:T.text,padding:"2px 5px",fontSize:10,
                            fontFamily:"'DM Mono',monospace"}}/>
                      </label>
                    ))}
                  </div>
                )}
              </div>

              {/* Noise toggle */}
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12,flexWrap:"wrap"}}>
                <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",fontSize:10,color:T.muted2}}>
                  <input type="checkbox" checked={noiseEnabled}
                    onChange={e=>setNoiseEnabled(e.target.checked)}
                    style={{accentColor:"#fbbf24"}}/>
                  Enable noise / variability
                </label>
                {noiseEnabled && <>
                  {[["cv_Ao","CV Ao"],["cv_Ka","CV Ka"],["cv_conj","CV β"],["cv_flow","CV flow"]].map(([k,lbl])=>(
                    <label key={k} style={{display:"flex",alignItems:"center",gap:4,fontSize:9,color:T.muted2}}>
                      {lbl}
                      <input type="number" step={0.01} min={0} max={1}
                        value={noiseConfig[k]}
                        onChange={e=>setNoiseConfig(c=>({...c,[k]:Number(e.target.value)}))}
                        style={{width:38,background:T.card,border:`1px solid ${T.border}`,
                          borderRadius:4,color:T.text,padding:"2px 5px",
                          fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                    </label>
                  ))}
                  <span style={{fontSize:9,color:T.muted,fontStyle:"italic"}}>shaded band = ±1σ</span>
                </>}
                <button onClick={handleSweep} disabled={sweeping}
                  style={{marginLeft:"auto",background:sweeping?T.border:"linear-gradient(90deg,#fbbf24,#16a34a)",
                    border:"none",color:sweeping?T.muted:"#050805",
                    padding:"6px 18px",borderRadius:6,cursor:sweeping?"not-allowed":"pointer",
                    fontFamily:"inherit",fontSize:11,fontWeight:600}}>
                  {sweeping?"⧗ Running…":"Run Sweep"}
                </button>
              </div>

              {sweepData&&sweepData.length>0
                ?<SweepChart data={sweepData} sweepParam={sweepParam}/>
                :(
                  <div style={{height:300,display:"flex",alignItems:"center",
                    justifyContent:"center",flexDirection:"column",gap:8}}>
                    <span style={{fontSize:20}}>📉</span>
                    <span style={{color:T.muted,fontSize:11}}>Select a parameter and click Run Sweep</span>
                    <span style={{color:T.muted,fontSize:9}}>
                      A₀ sweep traces the dose-response — T/C DECREASES as [A₀] increases (competition)
                    </span>
                  </div>
                )
              }

              {/* Sweep table */}
              {sweepData&&sweepData.length>0&&(
                <div style={{marginTop:14}}>
                  <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:6}}>
                    Sweep Results
                  </div>
                  <div style={{overflowX:"auto"}}>
                    <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
                      <thead>
                        <tr style={{background:T.surface}}>
                          {[sweepParam==="Ao"?"[A₀] (nM)":sweepParam==="Po"?"P₀ (nM)":
                            sweepParam==="Ag_tl"?"Ag_TL (nM)":"x_tl (mm)",
                            "PAg (nM)","PC (nM)","T/C ratio","Verdict",
                          ].map(h=>(
                            <th key={h} style={{padding:"5px 10px",textAlign:"left",
                              color:T.muted,fontSize:9,fontWeight:600,
                              borderBottom:`1px solid ${T.border2}`}}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sweepData.map((d,i)=>{
                          const pag=typeof d.rpa==="number"?d.rpa:0;
                          const pc =typeof d.pc ==="number"?d.pc :0;
                          const tc =typeof d.tc ==="number"?d.tc :Infinity;
                          const xv =typeof d.x  ==="number"?d.x  :0;
                          const isPos=pc>0&&tc>0&&tc<0.8;
                          const isInv=pc<=1e-30;
                          return (
                            <tr key={i} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                              <td style={{padding:"4px 10px",color:T.accent,
                                fontFamily:"monospace",fontWeight:600}}>
                                {sweepParam==="Ao"?xv.toExponential(2):xv.toFixed(1)}
                              </td>
                              <td style={{padding:"4px 10px",color:T.PAg,fontFamily:"monospace"}}>
                                {pag.toExponential(2)}
                              </td>
                              <td style={{padding:"4px 10px",color:T.PC,fontFamily:"monospace"}}>
                                {pc.toExponential(2)}
                              </td>
                              <td style={{padding:"4px 10px",fontFamily:"monospace",fontWeight:700,
                                color:isInv?T.muted:isPos?T.ok:T.muted2}}>
                                {isInv?"—":tc===Infinity?"∞":tc.toFixed(3)}
                              </td>
                              <td style={{padding:"4px 10px"}}>
                                <span style={{
                                  background:isInv?"#111":isPos?"#0a1a10":"#0a1020",
                                  color:isInv?T.muted:isPos?T.ok:T.muted2,
                                  border:`1px solid ${isInv?T.border:isPos?T.ok+"44":T.border}`,
                                  borderRadius:4,padding:"1px 7px",fontSize:9}}>
                                  {isInv?"—":isPos?"POSITIVE":"NEGATIVE"}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* REPORT TAB */}
          {tab==="report" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
              <SimReport simData={simData} userParams={userParams}
                kineticParams={kineticParams} physParams={physParams}/>
            </div>
          )}

          {/* MODEL INFO TAB */}
          {tab==="info" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
                Reaction Model + Physical Assumptions
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:14}}>
                {[
                  {n:"①",eq:"A + P  ⇌  AP",        desc:"Analyte sequesters detector in solution — depletes free P available for test line",   c:T.PA},
                  {n:"②",eq:"P + Ag_TL ⇌ AgP",     desc:"FREE P binds immobilised antigen → SIGNAL. Dims as [A] rises.",                      c:T.PAg},
                  {n:"③",eq:"AP  ✗  Ag_TL",         desc:"AP cannot bind Ag_TL — paratope occupied. AP passes through test line silently.",     c:T.muted},
                  {n:"④",eq:"P  + Rc  ⇌  CP",       desc:"Free detector captured at control line. Always present; rises when [A] is LOW.",      c:T.PC},
                  {n:"⑤",eq:"AP + Rc  ⇌  CAP",      desc:"AP complex also captured at control — same detector backbone, same anti-species Ab.", c:T.PA},
                  {n:"⑥",eq:"PC = CP + CAP",         desc:"Total control gold = CP + CAP. This is the denominator of the T/C ratio.",           c:T.PC},
                ].map(r=>(
                  <div key={r.n} style={{background:T.surface,border:`1px solid ${r.c}22`,borderRadius:8,padding:"9px 11px"}}>
                    <div style={{display:"flex",gap:8,alignItems:"baseline",marginBottom:3}}>
                      <span style={{color:r.c,fontSize:14,fontFamily:"'Syne',sans-serif",fontWeight:700}}>{r.n}</span>
                      <span style={{color:T.text,fontSize:11}}>{r.eq}</span>
                    </div>
                    <div style={{color:T.muted2,fontSize:10}}>{r.desc}</div>
                  </div>
                ))}
              </div>

              {/* Species location table */}
              <div style={{marginBottom:14}}>
                <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
                  Species by Location
                </div>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
                  <thead>
                    <tr style={{background:T.surface}}>
                      {["Location","Species","Description","Signal role"].map(h=>(
                        <th key={h} style={{padding:"6px 10px",textAlign:"left",color:T.muted,
                          fontSize:9,fontWeight:600,borderBottom:`1px solid ${T.border2}`}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Flowing","P",  "Free detector",           "Depleted by A → less reaches test line",  T.P],
                      ["Flowing","AP", "Analyte-detector complex","Cannot bind Ag_TL — passes through",       T.PA],
                      ["Test line","AgP","P captured by Ag_TL",  "THE signal — decreases with [A]",          T.PAg],
                      ["Ctrl line","CP", "Free P captured by Rc","Rises when [A] is low (more free P)",      T.PC],
                      ["Ctrl line","CAP","AP captured by Rc",    "Rises when [A] is high (more AP flows)",   T.PA],
                      ["Ctrl line","PC = CP+CAP","Total ctrl gold","T/C denominator — anti-species Ab binds both", T.PC],
                    ].map(([loc,sp,desc,role,c],i)=>(
                      <tr key={sp+loc} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                        <td style={{padding:"5px 10px",color:T.muted2,fontSize:9}}>{loc}</td>
                        <td style={{padding:"5px 10px",color:c,fontWeight:700}}>{sp}</td>
                        <td style={{padding:"5px 10px",color:T.text,fontSize:9}}>{desc}</td>
                        <td style={{padding:"5px 10px",color:T.muted,fontSize:9,fontStyle:"italic"}}>{role}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Competitive vs Sandwich */}
              <div style={{marginBottom:14}}>
                <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
                  Competitive vs Sandwich
                </div>
                <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
                  <thead>
                    <tr style={{background:T.surface}}>
                      {["Feature","Sandwich","Competitive"].map(h=>(
                        <th key={h} style={{padding:"6px 10px",textAlign:"left",color:T.muted,
                          fontSize:9,fontWeight:600,borderBottom:`1px solid ${T.border2}`}}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Test line captures","Analyte A (via antibody)","Free detector P (via antigen)"],
                      ["Signal species","PA complex (sandwich)","Free P (competitive)"],
                      ["Signal vs [A]","Increases ↑","Decreases ↓"],
                      ["POSITIVE verdict","T/C ≥ 0.05","T/C < 0.80"],
                      ["NEGATIVE verdict","T/C < 0.05","T/C ≥ 0.80"],
                      ["High [A] result","BRIGHT test line","DIM test line"],
                      ["Low [A] result","DIM test line","BRIGHT test line"],
                    ].map(([f,s,c],i)=>(
                      <tr key={f} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                        <td style={{padding:"5px 10px",color:T.muted2,fontSize:9}}>{f}</td>
                        <td style={{padding:"5px 10px",color:"#60a5fa",fontSize:9}}>{s}</td>
                        <td style={{padding:"5px 10px",color:"#4ade80",fontSize:9}}>{c}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Physical Model</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:6}}>
                {[
                  ["Diffusivity","D_eff = D·φ/τ","porosity × tortuosity scaling"],
                  ["Flow","U(x) = min(c/x³, U_max)","Berli & Kler 2016 + cap"],
                  ["Flow position","x⁴ = 21311t − 69505","Rochester wet lab fit"],
                  ["Inlet model","Conjugate pad pre-load","P pre-loaded at t=0; A enters as Dirichlet BC at x=0"],
                  ["Co-flow","A and P enter together","both at x=0, already partially reacted"],
                  ["Line zone","±line_width nodes","distributed immobilised zone"],
                  ["Advection","1st-order upwind","CFL-stable for all U"],
                ].map(([k,v,d])=>(
                  <div key={k} style={{background:T.surface,border:`1px solid ${T.border}`,borderRadius:6,padding:"7px 10px"}}>
                    <div style={{color:T.accent,fontSize:10,fontWeight:700,marginBottom:1}}>{k}</div>
                    <div style={{color:T.text,fontSize:10}}>{v}</div>
                    <div style={{color:T.muted,fontSize:9,marginTop:1}}>{d}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
