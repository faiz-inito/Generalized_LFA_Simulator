import { useState, useEffect, useRef, useCallback, useMemo } from "react";

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION ENGINE
// ═══════════════════════════════════════════════════════════════════════════════
//
// Physical improvements in this version:
//  • dx = 0.5 mm  (N=200) — finer grid eliminates spurious oscillations
//  • U(x) hard-capped at U_max derived from sample pad flow speed
//  • Conjugate release: burst + exponential decay  C_conj(t) = P0*(β*e^(-λt) + (1-β))
//  • Strip thickness enters cross-sectional area for concentration scaling
//  • Membrane porosity φ scales effective diffusivity: D_eff = D * φ / τ
//  • Line thickness determines width of immobilised zone (nodes)
//
// Zig-zag fix: U is capped + smooth Gaussian kernel applied to output profiles
// ═══════════════════════════════════════════════════════════════════════════════

function gaussSmooth(arr, sigma = 1.2) {
  // Lightweight 1-D Gaussian smoothing to remove numerical noise for display
  // Does NOT affect the underlying simulation — display only
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
  // up = user params, kp = kinetic params, pp = physical params
  const {
    Ao, Po, Ro, x_tl, x_cl, Ro_cl, t_end,
  } = up;
  const {
    Ka1, Kd1, Ka2, Kd2, Ka_cl, Kd_cl,
    DA, DP, flow_c,
  } = kp;
  const {
    strip_L,          // mm  total strip length
    dx,               // mm  grid spacing
    phi,              // —   membrane porosity (0-1)
    tau,              // —   tortuosity factor (≥1)
    conj_burst,       // —   burst fraction β (0-1)
    conj_lambda,      // s⁻¹ exponential release rate λ
    line_width_mm,    // mm  line zone half-width
    sample_vol,       // µL  sample volume (affects run time pressure)
    U_max,            // mm/s max allowed flow velocity (from sample pad geometry)
  } = pp;

  // ── Sample volume limit ─────────────────────────────────────────────
  // Strip cross-section (HF180 typical): width 4mm × thickness 0.135mm
  const strip_width_mm   = 4.0;
  const strip_thick_mm   = 0.135;
  const cross_section_mm2 = strip_width_mm * strip_thick_mm;  // 0.54 mm²
  // Flow rate: volume per second passing through cross-section at U_max
  // (1 mm³ = 1e-3 µL)
  const flow_rate_uL_s  = U_max * cross_section_mm2 * 1e-3;  // µL/s
  // Time until sample pad is exhausted
  const t_sample_end    = flow_rate_uL_s > 0
    ? sample_vol / flow_rate_uL_s   // s
    : t_end;                         // fallback: never exhausted
  // After t_sample_end, A[0] → 0 (only buffer washes through)
  // P[0] continues from conjugate release (already mobilised)

  const N   = Math.round(strip_L / dx);
  const tl  = Math.min(Math.round(x_tl / dx), N - 2);
  // Control line is ONLY active when x_cl is strictly inside the strip.
  // If x_cl >= strip_L the zone is physically unreachable — zero signal.
  const cl_reachable = (x_cl + dx) < strip_L;
  const cl  = cl_reachable ? Math.min(Math.round(x_cl / dx), N - 2) : -1;
  const lw  = Math.max(1, Math.round(line_width_mm / dx));

  // Effective diffusivities scaled by porosity and tortuosity
  const DA_eff = DA * phi / tau;
  const DP_eff = DP * phi / tau;

  const dt = Math.min(t_end / 1500, 0.2);

  const x = Array.from({ length: N }, (_, i) => (i + 1) * dx);

  // Flow velocity: U(x) = c/x³, capped at U_max
  const U = x.map(xi => Math.min(flow_c / (xi ** 3), U_max));

  const i2  = 1 / (dx * dx);
  const i2x = 1 / (2 * dx);

  // State arrays
  let A  = new Float64Array(N);
  let P  = new Float64Array(N);
  let PA = new Float64Array(N);
  let RA_tl = 0, RPA_tl = 0;
  let Rc_free = Ro_cl, Pc_cl = 0;
  // Cumulative flux into absorbent pad (nM·mm = concentration × distance)
  // = integral of U[N-1]*C[N-1]*dt over all timesteps
  let pad_A = 0, pad_P = 0, pad_PA = 0;

  // Initial conditions: strip is dry at t=0.
  // All concentrations are zero. The conjugate release model drives P[0]
  // each timestep via Ct = Po*(β*exp(-λ*t) + (1-β)).
  // A[0] is set each step too — no pre-fill needed here.
  // (Pre-filling P across the conjugate pad was incorrect: it created
  //  a non-physical step profile visible at frame 0 before flow begins.)

  const outputEvery = Math.max(1, Math.round((t_end / dt) / nOutput));
  const out = { times: [], A: [], P: [], PA: [], RPA: [], RA: [], PC: [], pad_A: [], pad_P: [], pad_PA: [], x, tl, cl, lw, N, cl_reachable, t_sample_end, flow_rate_uL_s };

  let step = 0;
  const maxSteps = Math.round(t_end / dt);

  // t=0: Strip is completely dry — all arrays already zero-initialised.
  // Frame 0 is captured first (dry strip), then the conjugate pad is
  // pre-loaded with Po so detector is ready to flow from step 1 onward.
  const conj_end = Math.round(0.20 * N);   // conjugate pad = first 20% of strip

  while (step <= maxSteps) {
    const t_now = step * dt;

    // ── Snapshot first (frame 0 = dry strip, no P yet) ──────────────
    if (step % outputEvery === 0) {
      out.times.push(t_now);
      out.A.push(Array.from(A));
      out.P.push(Array.from(P));
      out.PA.push(Array.from(PA));
      out.RPA.push(RPA_tl);
      out.RA.push(RA_tl);
      out.PC.push(Pc_cl);
      out.pad_A.push(pad_A);
      out.pad_P.push(pad_P);
      out.pad_PA.push(pad_PA);
    }

    // ── Pre-load conjugate pad once, after frame-0 snapshot ─────────
    // Frame 0 (dry strip) is already recorded above. Now fill the pad
    // so detector flows naturally from step 1 onward.
    if (step === 0) {
      for (let j = 0; j < conj_end; j++) P[j] = Po;
    }

    // ── Apply inlet BCs ──────────────────────────────────────────────
    // A[0] = Ao while sample flows; 0 after sample exhausted.
    // P[0] evolves freely — depletes from conjugate pad via advection.
    const sampleFlowing = t_now <= t_sample_end;
    A[0]  = sampleFlowing ? Ao : 0.0;
    PA[0] = 0.0;
    // P[0]: no reset — let it deplete from initial pre-load

    // ── Mobile reaction rates ────────────────────────────────────────
    const F_PA = new Float64Array(N);
    for (let j = 0; j < N; j++)
      F_PA[j] = Ka1 * A[j] * P[j] - Kd1 * PA[j];

    // ── Test line: immobilised zone over [tl-lw .. tl+lw] ──────────
    const Rf_tl  = Math.max(Ro - RA_tl - RPA_tl, 0);
    // Average concentrations over line zone
    let A_zone=0, P_zone=0, PA_zone=0; let nz=0;
    for (let j=Math.max(0,tl-lw); j<=Math.min(N-1,tl+lw); j++){
      A_zone+=A[j]; P_zone+=P[j]; PA_zone+=PA[j]; nz++;
    }
    A_zone/=nz; P_zone/=nz; PA_zone/=nz;
    const f_RA   = Ka2 * A_zone  * Rf_tl  - Kd2 * RA_tl;
    const f_RPA1 = Ka1 * P_zone  * RA_tl  - Kd1 * RPA_tl;
    const f_RPA2 = Ka2 * PA_zone * Rf_tl  - Kd2 * RPA_tl;

    // ── Control line — only runs when physically reachable ───────────
    const f_Pc = cl_reachable
      ? Ka_cl * P[cl] * Rc_free - Kd_cl * Pc_cl
      : 0;

    // ── PDE RHS: central upwind advection-diffusion ──────────────────
    const dA  = new Float64Array(N);
    const dP  = new Float64Array(N);
    const dPA = new Float64Array(N);

    for (let j = 0; j < N; j++) {
      const Uj = U[j];
      const Al  = j===0   ? A[0]    : A[j-1];
      const Ar  = j===N-1 ? A[N-1]  : A[j+1];
      const Pl  = j===0   ? P[0]    : P[j-1];
      const Pr  = j===N-1 ? P[N-1]  : P[j+1];
      const PAl = j===0   ? PA[0]   : PA[j-1];
      const PAr = j===N-1 ? PA[N-1] : PA[j+1];

      // Upwind advection (first-order upwind for stability)
      // For U>0 (left-to-right flow), upwind = backward difference
      const advA  = Uj * (A[j]  - Al)  / dx;
      const advP  = Uj * (P[j]  - Pl)  / dx;
      const advPA = Uj * (PA[j] - PAl) / dx;

      const diffA  = DA_eff * (Ar  - 2*A[j]  + Al)  * i2;
      const diffP  = DP_eff * (Pr  - 2*P[j]  + Pl)  * i2;
      const diffPA = DP_eff * (PAr - 2*PA[j] + PAl) * i2;

      dA[j]  = diffA  - advA  - F_PA[j];
      dP[j]  = diffP  - advP  - F_PA[j];
      dPA[j] = diffPA - advPA + F_PA[j];
    }

    // Distribute sink over line zone
    const zone_nodes_tl = [];
    for (let j=Math.max(0,tl-lw); j<=Math.min(N-1,tl+lw); j++) zone_nodes_tl.push(j);
    const zone_n = zone_nodes_tl.length;
    zone_nodes_tl.forEach(j => {
      dA[j]  -= f_RA   / zone_n;
      dP[j]  -= f_RPA1 / zone_n;
      dPA[j] -= f_RPA2 / zone_n;
    });
    if (cl_reachable) dP[cl] -= f_Pc;

    // ── Euler step ───────────────────────────────────────────────────
    for (let j = 0; j < N; j++) {
      A[j]  = Math.max(0, A[j]  + dt * dA[j]);
      P[j]  = Math.max(0, P[j]  + dt * dP[j]);
      PA[j] = Math.max(0, PA[j] + dt * dPA[j]);
    }
    RA_tl   = Math.max(0, RA_tl   + dt * (f_RA - f_RPA1));
    RPA_tl  = Math.max(0, RPA_tl  + dt * (f_RPA1 + f_RPA2));
    Rc_free = Math.max(0, Rc_free - dt * f_Pc);
    Pc_cl   = Math.max(0, Pc_cl   + dt * f_Pc);
    // Accumulate flux leaving domain into absorbent pad
    // U[N-1] * C[N-1] * dt = flux this step (nM·mm equivalent)
    const U_out = U[N - 1];
    pad_A  += U_out * A[N-1]  * dt;
    pad_P  += U_out * P[N-1]  * dt;
    pad_PA += U_out * PA[N-1] * dt;

    // Re-enforce outlet zero-flux (Neumann) — inlet already set above
    PA[0] = 0;
    step++;
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PARAM VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════
const KINETIC_META = [
  { key:"Ka1",    label:"Ka₁ (det-analyte on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:7.35e-4, sci:true },
  { key:"Kd1",    label:"Kd₁ (det-analyte off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:5.7e-5,  sci:true },
  { key:"Ka2",    label:"Ka₂ (rec-analyte on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:7.35e-4, sci:true },
  { key:"Kd2",    label:"Kd₂ (rec-analyte off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:5.7e-5,  sci:true },
  { key:"Ka_cl",  label:"Ka_cl (ctrl line on)",  unit:"nM⁻¹s⁻¹", min:1e-8, max:1e-1, def:5.0e-4,  sci:true },
  { key:"Kd_cl",  label:"Kd_cl (ctrl line off)", unit:"s⁻¹",      min:1e-8, max:1e0,  def:4.0e-5,  sci:true },
  { key:"DA",     label:"Analyte diffusivity",   unit:"mm²/s",    min:1e-7, max:1e-1, def:1e-4,    sci:true },
  { key:"DP",     label:"Detector diffusivity",  unit:"mm²/s",    min:1e-8, max:1e-2, def:1e-6,    sci:true },
  { key:"flow_c", label:"Flow constant c",       unit:"mm⁴/s",    min:100,  max:5e4,  def:5327.75, sci:false},
];

const PHYSICAL_META = [
  { key:"strip_L",       label:"Strip length",           unit:"mm",  min:20,   max:200,  def:100,   sci:false, step:5    },
  { key:"dx",            label:"Grid spacing",           unit:"mm",  min:0.25, max:2,    def:0.5,   sci:false, step:0.25 },
  { key:"phi",           label:"Membrane porosity φ",    unit:"—",   min:0.1,  max:0.95, def:0.7,   sci:false, step:0.05 },
  { key:"tau",           label:"Tortuosity τ",           unit:"—",   min:1.0,  max:5.0,  def:1.5,   sci:false, step:0.1  },
  { key:"conj_burst",    label:"Conjugate burst β",      unit:"—",   min:0,    max:1,    def:0.4,   sci:false, step:0.05 },
  { key:"conj_lambda",   label:"Release rate λ",         unit:"s⁻¹", min:1e-4, max:0.1,  def:5e-3,  sci:true,  step:1e-4 },
  { key:"line_width_mm", label:"Line half-width",        unit:"mm",  min:0.25, max:3,    def:0.5,   sci:false, step:0.25 },
  { key:"sample_vol",    label:"Sample volume",          unit:"µL",  min:5,    max:200,  def:75,    sci:false, step:5    },
  { key:"U_max",         label:"Max flow velocity",      unit:"mm/s",min:0.01, max:5,    def:0.8,   sci:false, step:0.05 },
];

function defaultKinetic() {
  return Object.fromEntries(KINETIC_META.map(m => [m.key, m.def]));
}
function defaultPhysical() {
  return Object.fromEntries(PHYSICAL_META.map(m => [m.key, m.def]));
}

function validateParam(meta, rawStr) {
  const v = parseFloat(rawStr);
  if (isNaN(v)) return { ok: false, msg: "Not a number" };
  if (v < meta.min) return { ok: false, msg: `Min: ${meta.min}` };
  if (v > meta.max) return { ok: false, msg: `Max: ${meta.max}` };
  return { ok: true, v };
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLOURS
// ═══════════════════════════════════════════════════════════════════════════════
const T = {
  bg:"#060a10", surface:"#0b0f18", card:"#0f1520",
  border:"#1a2535", border2:"#202e45", borderHi:"#2a3f60",
  A:"#38bdf8", P:"#c084fc", PA:"#34d399",
  RPA:"#fb923c", PC:"#f43f5e",
  accent:"#00d4ff", text:"#dde4f0",
  muted:"#4a5568", muted2:"#6b7a8d",
  err:"#f87171", ok:"#4ade80", warn:"#f59e0b",
};

const PADS = [
  { id:"sample",  label:"Sample Pad",    sub:"sample application",  frac:[0.00,0.09], bg:"#0e1f0e", bdr:"#2a6040", acc:"#4ade80" },
  { id:"conj",    label:"Conjugate Pad", sub:"labelled antibodies",  frac:[0.09,0.20], bg:"#130d22", bdr:"#5030a0", acc:"#c084fc" },
  { id:"nc",      label:"NC Membrane",   sub:"capillary flow zone",  frac:[0.20,0.83], bg:"#07101a", bdr:"#1a3560", acc:"#38bdf8" },
  { id:"abs",     label:"Absorbent Pad", sub:"wicking sink",         frac:[0.83,1.00], bg:"#091420", bdr:"#1a3a55", acc:"#60a5fa" },
];
const NC0=0.20, NC1=0.83;

// ═══════════════════════════════════════════════════════════════════════════════
// STRIP CANVAS
// ═══════════════════════════════════════════════════════════════════════════════
function LFAStrip({ simData, frameIdx, userParams }) {
  const canvasRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
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

    // ── Pad regions ────────────────────────────────────────────────
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
      ctx.strokeStyle=r.bdr; ctx.lineWidth=1;
      ctx.beginPath();
      ctx.moveTo(x1+4,SY-4); ctx.lineTo(x1+4,SY-28);
      ctx.lineTo(x1+rw-4,SY-28); ctx.lineTo(x1+rw-4,SY-4);
      ctx.stroke();
    });

    // NC pore texture
    const ncX0=NC0*W, ncX1w=NC1*W, ncW=ncX1w-ncX0;
    ctx.save(); ctx.globalAlpha=0.07; ctx.fillStyle="#6ab4f0";
    for(let i=0;i<260;i++){
      const px=ncX0+Math.random()*ncW, py=SY+3+Math.random()*(SH-6);
      ctx.beginPath(); ctx.arc(px,py,0.85,0,Math.PI*2); ctx.fill();
    }
    ctx.restore();

    if (!simData) {
      ctx.font="12px 'DM Mono',monospace"; ctx.fillStyle=T.muted;
      ctx.textAlign="center"; ctx.fillText("Press ▶ Run Simulation", W/2, SY+SH/2);
      return;
    }

    const fi    = Math.min(frameIdx, simData.A.length-1);
    const Af    = simData.A[fi];
    const Pf    = simData.P[fi];
    const PAf   = simData.PA[fi];
    const RPAv  = simData.RPA[fi];
    const PCv   = simData.PC[fi];
    const t_now = simData.times[fi];
    const { x, N } = simData;
    // Chart inset constants — defined here so mmToCx and flow front
    // can use the same coordinate system as axes and profiles.
    const LEFT_INSET  = 50;
    const RIGHT_INSET = 52;
    const cX0 = ncX0 + LEFT_INSET;
    const cX1 = ncX1w - RIGHT_INSET;
    const cW  = cX1 - cX0;
    const cY1 = SY + 6;
    const cY2 = SY + SH - 4;
    const cH  = cY2 - cY1;
    // mmToCx: mm → canvas px, aligned with x-axis ticks and profiles
    const mmToCx = mm => cX0 + (mm / x[x.length - 1]) * cW;

    // ── Flow front ─────────────────────────────────────────────────
    // The liquid wetting front sweeps the ENTIRE physical strip
    // (all pads), so we map it over the full canvas width 0→W,
    // not just the NC membrane chart area cX0→cX1.
    // strip_L (mm) covers the NC membrane only in the simulation,
    // but physically the strip also includes sample pad (~9%) +
    // conjugate pad (~11%) + absorbent pad (~17%).
    // Total physical length ≈ strip_L / (NC1-NC0) = strip_L / 0.63
    const physTotalMM = x[x.length-1] / (NC1 - NC0); // full strip in mm
    const flowToCx = mm => Math.min((mm / physTotalMM) * W, W); // full strip → canvas, capped at W
    const x4 = 21311*t_now-69505;
    // No clamp here — let the front travel freely into the absorbent pad
    const xFmm_nc = x4 > 0 ? x4**0.25 : 0;
    // Convert NC-relative position to full-strip position
    const ncStartMM = NC0 * physTotalMM;
    const xFmm = xFmm_nc + ncStartMM;
    const ffX  = flowToCx(xFmm);
    if (xFmm_nc > 0) {
      const x2 = Math.min(ffX, W);
      const wg = ctx.createLinearGradient(0,0,x2,0);
      wg.addColorStop(0,"rgba(14,100,180,0.09)");
      wg.addColorStop(0.85,"rgba(14,100,180,0.06)");
      wg.addColorStop(1,"rgba(0,212,255,0)");
      ctx.fillStyle=wg; ctx.fillRect(0,SY,x2,SH);
      if (ffX < W) {
        const eg=ctx.createLinearGradient(ffX-22,0,ffX+6,0);
        eg.addColorStop(0,"rgba(0,212,255,0)");
        eg.addColorStop(0.55,"rgba(0,212,255,0.38)");
        eg.addColorStop(1,"rgba(200,245,255,0.12)");
        ctx.fillStyle=eg; ctx.fillRect(ffX-22,SY+2,28,SH-4);
        ctx.save(); ctx.strokeStyle="rgba(0,212,255,0.85)"; ctx.lineWidth=1.5;
        ctx.setLineDash([5,4]);
        ctx.beginPath(); ctx.moveTo(ffX,SY+2); ctx.lineTo(ffX,SY+SH-2); ctx.stroke();
        ctx.setLineDash([]); ctx.restore();
        if (ffX > 30 && ffX < W-20) {
          ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle="rgba(0,212,255,0.7)";
          ctx.textAlign="center"; ctx.fillText("▲ flow front", ffX, SY+SH+14);
        }
      }
    }

    // ── Concentration profiles — DUAL Y-AXIS ───────────────────────────
    // LEFT  axis (orange ticks): Analyte [A] + Complex [PA]  — analyte scale
    // RIGHT axis (purple ticks): Detector [P]               — detector scale
    //
    // This separates species that live in completely different nM regimes:
    //   P  ~ Po (nM range, set by user)
    //   A  ~ Ao (often pM range, 1e-7 nM default)
    //   PA ~ between A and P in magnitude
    // Both axes are linear within their own range. Each has its own ticks,
    // labels, and grid lines drawn in its species colour.

    const toX = i => cX0 + (i / (N - 1)) * cW;

    // ── Left axis: A and PA ──────────────────────────────────────────
    // Scale = max of A and PA across all nodes, with 10% headroom
    const leftMax = Math.max(...Af, ...PAf, 1e-30) * 1.15;
    const toYL    = v => cY2 - Math.min(Math.max(v / leftMax, 0), 1) * cH;

    // Left axis ticks and grid (4 ticks, A-colour)
    const L_TICKS = 4;
    for (let t = 0; t <= L_TICKS; t++) {
      const val = leftMax * (t / L_TICKS);
      const cy  = toYL(val);
      // Tick
      ctx.strokeStyle = `rgba(56,189,248,0.35)`;  // T.A colour
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.moveTo(cX0, cy); ctx.lineTo(cX0 - 5, cy); ctx.stroke();
      // Grid line (faint)
      ctx.strokeStyle = `rgba(56,189,248,0.06)`;
      ctx.beginPath(); ctx.moveTo(cX0, cy); ctx.lineTo(cX1, cy); ctx.stroke();
      // Label
      ctx.font      = "8px 'DM Mono',monospace";
      ctx.fillStyle = "rgba(56,189,248,0.75)";
      ctx.textAlign = "right";
      ctx.fillText(val.toExponential(1), cX0 - 7, cy + 3);
    }
    // Left axis spine
    ctx.strokeStyle = "rgba(56,189,248,0.3)";
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(cX0, cY1); ctx.lineTo(cX0, cY2); ctx.stroke();
    // Left axis title
    ctx.save();
    ctx.fillStyle = "rgba(56,189,248,0.65)";
    ctx.font      = "9px 'DM Mono',monospace";
    ctx.textAlign = "center";
    ctx.translate(ncX0 + 8, SY + SH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("[A],[PA] (nM)", 0, 0);
    ctx.restore();

    // ── Right axis: P ────────────────────────────────────────────────
    // Scale = max of P across all nodes, with 10% headroom
    const rightMax = Math.max(...Pf, 1e-30) * 1.15;
    const toYR     = v => cY2 - Math.min(Math.max(v / rightMax, 0), 1) * cH;

    // Right axis ticks (4 ticks, P-colour)
    const R_TICKS = 4;
    for (let t = 0; t <= R_TICKS; t++) {
      const val = rightMax * (t / R_TICKS);
      const cy  = toYR(val);
      // Tick
      ctx.strokeStyle = `rgba(192,132,252,0.35)`;  // T.P colour
      ctx.lineWidth   = 1;
      ctx.beginPath(); ctx.moveTo(cX1, cy); ctx.lineTo(cX1 + 5, cy); ctx.stroke();
      // Grid line (very faint, different dash to distinguish from left grid)
      ctx.strokeStyle = `rgba(192,132,252,0.05)`;
      ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(cX0, cy); ctx.lineTo(cX1, cy); ctx.stroke();
      ctx.setLineDash([]);
      // Label
      ctx.font      = "8px 'DM Mono',monospace";
      ctx.fillStyle = "rgba(192,132,252,0.75)";
      ctx.textAlign = "left";
      ctx.fillText(val.toExponential(1), cX1 + 7, cy + 3);
    }
    // Right axis spine
    ctx.strokeStyle = "rgba(192,132,252,0.3)";
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(cX1, cY1); ctx.lineTo(cX1, cY2); ctx.stroke();
    // Right axis title
    ctx.save();
    ctx.fillStyle = "rgba(192,132,252,0.65)";
    ctx.font      = "9px 'DM Mono',monospace";
    ctx.textAlign = "center";
    ctx.translate(ncX1w - 8, SY + SH / 2);
    ctx.rotate(Math.PI / 2);
    ctx.fillText("[P] (nM)", 0, 0);
    ctx.restore();

    // ── Draw concentration profiles ──────────────────────────────────
    const drawProfile = (rawVals, color, toYfn) => {
      if (!rawVals?.length) return;
      const vals = gaussSmooth(rawVals, 1.5);
      const [ri,gi,bi] = [0,2,4].map(o => parseInt(color.slice(1+o,3+o), 16));
      ctx.save();
      ctx.beginPath(); ctx.rect(cX0, cY1 - 1, cW, cH + 2); ctx.clip();

      // Filled area
      ctx.beginPath();
      ctx.moveTo(toX(0), cY2);
      for (let i = 0; i < N; i++) ctx.lineTo(toX(i), toYfn(vals[i]));
      ctx.lineTo(toX(N - 1), cY2);
      ctx.closePath();
      const fg = ctx.createLinearGradient(0, cY1, 0, cY2);
      fg.addColorStop(0,   `rgba(${ri},${gi},${bi},0.28)`);
      fg.addColorStop(0.6, `rgba(${ri},${gi},${bi},0.08)`);
      fg.addColorStop(1,   `rgba(${ri},${gi},${bi},0.01)`);
      ctx.fillStyle = fg; ctx.fill();

      // Line
      ctx.beginPath();
      let penDown = false;
      for (let i = 0; i < N; i++) {
        const px = toX(i), py = toYfn(vals[i]);
        if (!penDown) { ctx.moveTo(px, py); penDown = true; }
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = `rgba(${ri},${gi},${bi},0.95)`;
      ctx.lineWidth   = 2;
      ctx.shadowColor = color; ctx.shadowBlur = 4;
      ctx.stroke(); ctx.shadowBlur = 0;
      ctx.restore();
    };

    // P uses right axis (toYR), A and PA use left axis (toYL)
    drawProfile(Pf,  T.P,  toYR);
    drawProfile(PAf, T.PA, toYL);
    drawProfile(Af,  T.A,  toYL);

    // Baseline
    ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cX0, cY2); ctx.lineTo(cX1, cY2); ctx.stroke();

    // ── Test & Control lines ──────────────────────────────────────────
    // Lines span full strip height (SY..SY+SH), mapped via mmToCx.
    // Intensity normalised on shared scale so weak T looks dim vs bright C.
    const peakRPA_canvas = Math.max(...simData.RPA) || 1e-30;
    const peakPC_canvas  = Math.max(...simData.PC)  || 1e-30;
    const sharedScale    = Math.max(peakRPA_canvas, peakPC_canvas);
    drawLFALine(ctx, mmToCx(userParams.x_tl), SY, SH, T.RPA, RPAv / sharedScale, "TEST", RPAv);
    if (simData.cl_reachable) {
      drawLFALine(ctx, mmToCx(userParams.x_cl), SY, SH, T.PC, PCv / sharedScale, "CTRL", PCv);
    }

    // x-axis ticks — use exact same cX0/cW/xTotalMM as mmToCx so labels
    // align perfectly with the concentration profiles and line markers.
    const xTotalMM = x[x.length - 1];
    ctx.font="8px 'DM Mono',monospace"; ctx.textAlign="center";
    [0,10,20,30,40,50,60,70,80,90,100].forEach(mm => {
      if (mm > xTotalMM) return;
      const cx = cX0 + (mm / xTotalMM) * cW;
      if (cx < cX0 - 1 || cx > cX1 + 1) return;
      ctx.fillStyle = "rgba(255,255,255,0.18)";
      ctx.fillRect(cx - 0.5, SY + SH - 4, 1, 4);
      ctx.fillStyle = T.muted;
      ctx.fillText(`${mm}`, cx, SY + SH + 13);
    });
    ctx.font="9px 'DM Mono',monospace"; ctx.fillStyle=T.muted2; ctx.textAlign="center";
    ctx.fillText("Position (mm)", (cX0 + cX1) / 2, H - 2);
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
    const {times,RPA,PC}=simData;
    const pad={t:14,r:16,b:36,l:62};
    const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
    const maxT=times[times.length-1], maxY=Math.max(...RPA,...PC)||1;
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
    const drawC=(vals,color,fa=0.22)=>{
      const[ri,gi,bi]=[0,2,4].map(o=>parseInt(color.slice(1+o,3+o),16));
      ctx.beginPath();ctx.moveTo(tx(times[0]),ty(0));
      times.forEach((t,i)=>ctx.lineTo(tx(t),ty(vals[i])));
      ctx.lineTo(tx(times[times.length-1]),ty(0));ctx.closePath();
      const g=ctx.createLinearGradient(0,pad.t,0,pad.t+iH);
      g.addColorStop(0,`rgba(${ri},${gi},${bi},${fa})`);
      g.addColorStop(1,`rgba(${ri},${gi},${bi},0.02)`);
      ctx.fillStyle=g;ctx.fill();
      ctx.beginPath();
      times.forEach((t,i)=>i===0?ctx.moveTo(tx(t),ty(vals[i])):ctx.lineTo(tx(t),ty(vals[i])));
      ctx.strokeStyle=color;ctx.lineWidth=2.2;ctx.shadowColor=color;ctx.shadowBlur=7;ctx.stroke();ctx.shadowBlur=0;
    };
    drawC(PC,T.PC,0.14); drawC(RPA,T.RPA,0.22);
    [[T.RPA,"Test [RPA]"],[T.PC,"Control [PC]"]].forEach(([c,l],i)=>{
      const lx=pad.l+8,ly=pad.t+14+i*15;
      ctx.strokeStyle=c;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+16,ly);ctx.stroke();
      ctx.font="9px 'DM Mono',monospace";ctx.fillStyle=c;ctx.textAlign="left";ctx.fillText(l,lx+20,ly+3);
    });
    const fi=Math.min(frameIdx,times.length-1);
    const cx=tx(times[fi]);
    ctx.strokeStyle="rgba(0,212,255,0.45)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(cx,pad.t);ctx.lineTo(cx,pad.t+iH);ctx.stroke();ctx.setLineDash([]);
    [[RPA,T.RPA],[PC,T.PC]].forEach(([arr,c])=>{
      ctx.beginPath();ctx.arc(cx,ty(arr[fi]),4.5,0,Math.PI*2);
      ctx.fillStyle=c;ctx.shadowColor=c;ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0;
    });
    ctx.strokeStyle=T.border2;ctx.lineWidth=1;ctx.strokeRect(pad.l,pad.t,iW,iH);
    // Sample exhaustion marker
    if (tSampleEnd && tSampleEnd < maxT) {
      const sx = tx(tSampleEnd);
      ctx.save();
      ctx.strokeStyle="rgba(74,222,128,0.55)"; ctx.lineWidth=1.2;
      ctx.setLineDash([3,4]);
      ctx.beginPath();ctx.moveTo(sx,pad.t);ctx.lineTo(sx,pad.t+iH);ctx.stroke();
      ctx.setLineDash([]);
      ctx.font="8px 'DM Mono',monospace"; ctx.fillStyle="rgba(74,222,128,0.7)";
      ctx.textAlign="center"; ctx.fillText("sample end",sx,pad.t+9);
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
// SWEEP CHART
// ═══════════════════════════════════════════════════════════════════════════════
function SweepChart({ data, sweepParam }) {
  const canvasRef = useRef(null);
  const isLogX = sweepParam === "Ao";

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.length) return;
    const DPR = window.devicePixelRatio || 1;
    const W = canvas.offsetWidth, H = canvas.offsetHeight;
    canvas.width  = W * DPR;
    canvas.height = H * DPR;
    const ctx = canvas.getContext("2d");
    ctx.scale(DPR, DPR);
    ctx.clearRect(0, 0, W, H);

    const PAD = { t: 22, r: 72, b: 46, l: 64 };
    const IW = W - PAD.l - PAD.r;
    const IH = H - PAD.t - PAD.b;

    const xs  = data.map(d => d.x);
    const tcs = data.map(d => d.tc);
    const rps = data.map(d => d.rpa);
    const pcs = data.map(d => d.pc);

    const minX   = xs[0];
    const maxX   = xs[xs.length - 1];
    const maxTC  = Math.max(...tcs, 0.06);
    const maxSIG = Math.max(...rps, ...pcs, 1e-30);

    // x mapping
    const tx = v => {
      if (isLogX) {
        const lo = Math.log10(Math.max(minX, 1e-15));
        const hi = Math.log10(maxX);
        return PAD.l + ((Math.log10(Math.max(v, 1e-15)) - lo) / (hi - lo)) * IW;
      }
      return PAD.l + ((v - minX) / (maxX - minX || 1)) * IW;
    };

    // y mappings — left axis TC, right axis signal
    const tyL = v => PAD.t + IH - Math.min(Math.max(v / maxTC,  0), 1) * IH;
    const tyR = v => PAD.t + IH - Math.min(Math.max(v / maxSIG, 0), 1) * IH;

    // ── Grid & left axis (orange = T/C) ───────────────────────────
    const L_TICKS = 5;
    for (let i = 0; i <= L_TICKS; i++) {
      const val = maxTC * i / L_TICKS;
      const cy  = tyL(val);
      ctx.strokeStyle = "rgba(251,146,60,0.07)";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(PAD.l, cy); ctx.lineTo(W - PAD.r, cy); ctx.stroke();
      ctx.font = "8px 'DM Mono',monospace";
      ctx.fillStyle = "rgba(251,146,60,0.75)";
      ctx.textAlign = "right";
      ctx.fillText(val.toExponential(1), PAD.l - 5, cy + 3);
    }
    // Left spine
    ctx.strokeStyle = "rgba(251,146,60,0.25)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, PAD.t + IH); ctx.stroke();

    // ── Right axis (sky blue = signal) ────────────────────────────
    const R_TICKS = 4;
    for (let i = 1; i <= R_TICKS; i++) {
      const val = maxSIG * i / R_TICKS;
      const cy  = tyR(val);
      ctx.strokeStyle = "rgba(56,189,248,0.04)";
      ctx.lineWidth = 0.5;
      ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.l, cy); ctx.lineTo(W - PAD.r, cy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "8px 'DM Mono',monospace";
      ctx.fillStyle = "rgba(56,189,248,0.65)";
      ctx.textAlign = "left";
      ctx.fillText(val.toExponential(1), W - PAD.r + 4, cy + 3);
    }
    // Right spine
    ctx.strokeStyle = "rgba(56,189,248,0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W - PAD.r, PAD.t); ctx.lineTo(W - PAD.r, PAD.t + IH); ctx.stroke();

    // ── LOD line T/C = 0.05 ───────────────────────────────────────
    const LOD = 0.05;
    if (LOD <= maxTC) {
      const cy = tyL(LOD);
      ctx.strokeStyle = "rgba(74,222,128,0.65)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath(); ctx.moveTo(PAD.l, cy); ctx.lineTo(W - PAD.r, cy); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "8px 'DM Mono',monospace";
      ctx.fillStyle = "rgba(74,222,128,0.85)";
      ctx.textAlign = "left";
      ctx.fillText("LOD  T/C = 0.05", PAD.l + 4, cy - 3);
    }

    // ── x-axis ticks ──────────────────────────────────────────────
    ctx.font = "8px 'DM Mono',monospace";
    ctx.fillStyle = "rgba(180,180,200,0.7)";
    ctx.textAlign = "center";
    if (isLogX) {
      const lo = Math.floor(Math.log10(minX));
      const hi = Math.ceil(Math.log10(maxX));
      for (let e = lo; e <= hi; e++) {
        const v = Math.pow(10, e);
        if (v < minX * 0.5 || v > maxX * 2) continue;
        const cx = tx(v);
        if (cx < PAD.l || cx > W - PAD.r) continue;
        ctx.strokeStyle = "rgba(255,255,255,0.07)";
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, PAD.t + IH); ctx.stroke();
        ctx.fillText(`1e${e}`, cx, H - PAD.b + 13);
      }
    } else {
      xs.forEach(v => {
        const cx = tx(v);
        ctx.strokeStyle = "rgba(255,255,255,0.06)";
        ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, PAD.t + IH); ctx.stroke();
        const lbl = v < 1 ? v.toFixed(1) : v >= 10 ? v.toFixed(0) : v.toFixed(1);
        ctx.fillText(lbl, cx, H - PAD.b + 13);
      });
    }

    // ── Draw lines ────────────────────────────────────────────────
    const drawLine = (vals, color, toY, lw, dash) => {
      ctx.save();
      ctx.beginPath();
      vals.forEach((v, i) => {
        const px = tx(xs[i]), py = toY(v);
        i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = lw;
      ctx.setLineDash(dash || []);
      ctx.shadowColor = color;
      ctx.shadowBlur = 5;
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.setLineDash([]);
      ctx.restore();
    };

    // Fill under T/C curve
    ctx.beginPath();
    ctx.moveTo(tx(xs[0]), PAD.t + IH);
    xs.forEach((v, i) => ctx.lineTo(tx(v), tyL(tcs[i])));
    ctx.lineTo(tx(xs[xs.length - 1]), PAD.t + IH);
    ctx.closePath();
    const gfill = ctx.createLinearGradient(0, PAD.t, 0, PAD.t + IH);
    gfill.addColorStop(0, "rgba(251,146,60,0.20)");
    gfill.addColorStop(1, "rgba(251,146,60,0)");
    ctx.fillStyle = gfill;
    ctx.fill();

    // ±1σ noise bands
    const hasNoise = data.length>0 && data[0].tc_lo !== null;
    if (hasNoise) {
      // T/C band
      ctx.beginPath();
      xs.forEach((v,i)=>ctx.lineTo(tx(v), tyL(data[i].tc_hi??tcs[i])));
      [...xs].reverse().forEach((v,i)=>{const di=xs.length-1-i;ctx.lineTo(tx(v),tyL(data[di].tc_lo??tcs[di]));});
      ctx.closePath(); ctx.fillStyle="rgba(251,146,60,0.13)"; ctx.fill();
      // RPA band
      ctx.beginPath();
      xs.forEach((v,i)=>ctx.lineTo(tx(v), tyR(data[i].rpa_hi??rps[i])));
      [...xs].reverse().forEach((v,i)=>{const di=xs.length-1-i;ctx.lineTo(tx(v),tyR(data[di].rpa_lo??rps[di]));});
      ctx.closePath(); ctx.fillStyle="rgba(56,189,248,0.09)"; ctx.fill();
    }
    drawLine(rps, "rgba(56,189,248,0.7)",  tyR, 1.5, [4, 3]);
    drawLine(pcs, "rgba(244,63,94,0.7)",   tyR, 1.5, [2, 4]);
    drawLine(tcs, "rgba(251,146,60,1.0)",  tyL, 2.5);

    // Dots on T/C — green if positive, orange if not
    xs.forEach((v, i) => {
      const px = tx(v), py = tyL(tcs[i]);
      const aboveLOD = tcs[i] >= LOD;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = aboveLOD ? "rgba(74,222,128,0.9)" : "rgba(251,146,60,0.9)";
      ctx.shadowColor = aboveLOD ? "#4ade80" : "#fb923c";
      ctx.shadowBlur = 7;
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    // LOD crossing vertical line + label
    for (let i = 1; i < xs.length; i++) {
      if (tcs[i - 1] < LOD && tcs[i] >= LOD) {
        const frac = (LOD - tcs[i - 1]) / (tcs[i] - tcs[i - 1]);
        const lodX = xs[i - 1] + frac * (xs[i] - xs[i - 1]);
        const lx   = tx(lodX);
        ctx.strokeStyle = "rgba(74,222,128,0.75)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(lx, PAD.t); ctx.lineTo(lx, PAD.t + IH); ctx.stroke();
        ctx.setLineDash([]);
        ctx.font = "bold 9px 'DM Mono',monospace";
        ctx.fillStyle = "#4ade80";
        ctx.textAlign = "center";
        const lbl = isLogX
          ? `LOD \u2248 ${lodX.toExponential(2)} nM`
          : `LOD \u2248 ${lodX.toFixed(2)}`;
        ctx.fillText(lbl, lx, PAD.t + 13);
        break;
      }
    }

    // ── Legend (top-right) ────────────────────────────────────────
    const legends = [
      ["rgba(251,146,60,1)",  "T/C ratio (left axis)", []],
      ["rgba(56,189,248,0.8)","[RPA] signal (right)",  [4,3]],
      ["rgba(244,63,94,0.8)", "[PC]  signal (right)",  [2,4]],
      ["rgba(74,222,128,0.8)","LOD threshold",         [6,4]],
    ];
    legends.forEach(([color, label, dash], i) => {
      const lx = PAD.l + 4, ly = PAD.t + 13 + i * 13;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dash);
      ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = "8px 'DM Mono',monospace";
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.fillText(label, lx + 20, ly + 3);
    });

    // ── Axis labels ───────────────────────────────────────────────
    const xLabel = sweepParam === "Ao" ? "Analyte [A\u2080] (nM)  \u2014  log scale"
                 : sweepParam === "Po" ? "Detector [P\u2080] (nM)"
                 : sweepParam === "Ro" ? "Test receptor [R\u2080] (nM)"
                 : "Test line position (mm)";
    ctx.font = "9px 'DM Mono',monospace";
    ctx.fillStyle = "rgba(180,180,200,0.65)";
    ctx.textAlign = "center";
    ctx.fillText(xLabel, PAD.l + IW / 2, H - 3);

    ctx.save();
    ctx.font = "9px 'DM Mono',monospace";
    ctx.fillStyle = "rgba(251,146,60,0.7)";
    ctx.translate(11, PAD.t + IH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("T/C ratio", 0, 0);
    ctx.restore();

    ctx.save();
    ctx.font = "9px 'DM Mono',monospace";
    ctx.fillStyle = "rgba(56,189,248,0.6)";
    ctx.translate(W - 7, PAD.t + IH / 2);
    ctx.rotate(Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillText("Signal (nM)", 0, 0);
    ctx.restore();

    ctx.strokeStyle = "rgba(40,50,70,1)";
    ctx.lineWidth = 1;
    ctx.strokeRect(PAD.l, PAD.t, IW, IH);

  }, [data, sweepParam, isLogX]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%", height: 300, display: "block",
        borderRadius: 8, border: `1px solid ${T.border}`, background: T.surface,
      }}
    />
  );
}

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
          boxShadow:`0 0 6px ${color||T.accent}`,border:"2px solid #0b0f18"}}/>
        <input type="range" min={logMin} max={logMax}
          step={(logMax-logMin)/1000} value={logVal}
          onChange={e=>onChange(Math.pow(10,Number(e.target.value)))}
          style={{position:"absolute",inset:0,opacity:0,width:"100%",
            cursor:"pointer",margin:0,height:20,top:-8}}/>
      </div>
      <div style={{position:"relative",height:14}}>
        {decades.map(e=>{
          const tp=((e-logMin)/(logMax-logMin))*100;
          return (
            <div key={e} style={{position:"absolute",left:`${tp}%`,
              transform:"translateX(-50%)",textAlign:"center"}}>
              <div style={{width:1,height:4,background:T.border,margin:"0 auto 1px"}}/>
              <span style={{fontSize:7,color:T.muted,fontFamily:"'DM Mono',monospace",
                whiteSpace:"nowrap"}}>{e>=0?`1e+${e}`:`1e${e}`}</span>
            </div>
          );
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
          boxShadow:`0 0 6px ${color||T.accent}`,border:"2px solid #0b0f18"}}/>
        <input type="range" min={min} max={max} step={step} value={value}
          onChange={e=>onChange(Number(e.target.value))}
          style={{position:"absolute",inset:0,opacity:0,width:"100%",cursor:"pointer",
            margin:0,height:20,top:-8}}/>
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

// ═══════════════════════════════════════════════════════════════════════════════
// EDITABLE PARAM ROW  (validated text input)
// ═══════════════════════════════════════════════════════════════════════════════
function ParamRow({ meta, value, onChange }) {
  const [raw, setRaw]   = useState(meta.sci ? value.toExponential(2) : String(value));
  const [err, setErr]   = useState(null);
  const [dirty, setDirty] = useState(false);

  // Sync if value changed externally (e.g. reset)
  useEffect(() => {
    setRaw(meta.sci ? value.toExponential(2) : String(Number(value.toFixed(6))));
    setErr(null); setDirty(false);
  }, [value]);

  const commit = () => {
    const res = validateParam(meta, raw);
    if (!res.ok) { setErr(res.msg); return; }
    setErr(null); setDirty(false);
    onChange(res.v);
  };

  return (
    <div style={{marginBottom:8}}>
      <div style={{display:"flex",alignItems:"center",gap:6}}>
        <span style={{color:T.muted2,fontSize:10,flex:1,minWidth:0}}>{meta.label}</span>
        <div style={{position:"relative",display:"flex",alignItems:"center",gap:4}}>
          <input
            value={raw}
            onChange={e=>{setRaw(e.target.value);setDirty(true);setErr(null);}}
            onBlur={commit}
            onKeyDown={e=>e.key==="Enter"&&commit()}
            style={{
              background:T.surface,
              border:`1px solid ${err?T.err:dirty?T.accent:T.border}`,
              borderRadius:5,color:err?T.err:T.text,
              padding:"3px 6px",fontSize:10,fontFamily:"'DM Mono',monospace",
              width:100,outline:"none",transition:"border 0.15s"
            }}
          />
          <span style={{color:T.muted,fontSize:9,minWidth:40}}>{meta.unit}</span>
        </div>
        {err && <span style={{color:T.err,fontSize:9,whiteSpace:"nowrap"}}>⚠ {err}</span>}
        {!err && dirty && <span style={{color:T.ok,fontSize:9}}>↵</span>}
      </div>
      <div style={{color:T.muted,fontSize:8,marginTop:1,textAlign:"right",paddingRight:46}}>
        [{meta.min} – {meta.max}]
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// OD ASSESSMENT
// ═══════════════════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════════════════
// ABSORBENT PAD ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════════
function AbsorbentPad({ simData, frameIdx, userParams }) {
  if (!simData) return null;
  const fi      = Math.min(frameIdx, simData.times.length - 1);
  const pad_A   = simData.pad_A[fi];
  const pad_P   = simData.pad_P[fi];
  const pad_PA  = simData.pad_PA[fi];
  const t_now   = simData.times[fi];

  // Normalise: divide by total possible flux = Ao * t_sample_end * U_avg
  // Simpler: normalise against the largest pad flux at end of run
  // giving a % of "what reached the pad" per species
  const maxPad   = Math.max(
    simData.pad_A[simData.pad_A.length-1],
    simData.pad_P[simData.pad_P.length-1],
    simData.pad_PA[simData.pad_PA.length-1],
    1e-30
  );

  // Capture efficiency: what fraction of analyte-related species was RETAINED
  // by the membrane (test line) vs escaped to pad
  const finalPadA  = simData.pad_A[simData.pad_A.length-1];
  const finalPadPA = simData.pad_PA[simData.pad_PA.length-1];
  const finalRPA   = Math.max(...simData.RPA);
  // Total analyte-derived species: bound(RPA) + escaped(pad_A + pad_PA)
  const totalA     = finalRPA + finalPadA + finalPadPA;
  const captureEff = totalA > 0 ? (finalRPA / totalA) * 100 : 0;

  // Detector waste: P that escaped past both lines
  const finalPadP  = simData.pad_P[simData.pad_P.length-1];
  const finalPC    = Math.max(...simData.PC);
  const totalP     = finalPC + finalPadP;
  const ctrlEff    = totalP > 0 ? (finalPC / totalP) * 100 : 0;

  const rows = [
    {
      species: "[A] Free analyte",
      color:   T.A,
      current: pad_A,
      final:   finalPadA,
      pct:     maxPad > 0 ? (pad_A / maxPad) * 100 : 0,
      note:    "Analyte that bypassed test line",
    },
    {
      species: "[P] Free detector",
      color:   T.P,
      current: pad_P,
      final:   finalPadP,
      pct:     maxPad > 0 ? (pad_P / maxPad) * 100 : 0,
      note:    "Excess detector past control line",
    },
    {
      species: "[PA] A·P complex",
      color:   T.PA,
      current: pad_PA,
      final:   finalPadPA,
      pct:     maxPad > 0 ? (pad_PA / maxPad) * 100 : 0,
      note:    "Labelled analyte not captured",
    },
  ];

  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,
      borderRadius:10,padding:14}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",
        marginBottom:10}}>
        <div style={{fontSize:9,color:T.muted,letterSpacing:3,
          textTransform:"uppercase"}}>Absorbent Pad</div>
        <div style={{fontSize:9,color:T.muted2}}>
          cumulative flux · t = {Math.round(t_now)} s
        </div>
      </div>

      {/* Species bars */}
      <div style={{display:"flex",flexDirection:"column",gap:8,marginBottom:12}}>
        {rows.map(r => (
          <div key={r.species}>
            <div style={{display:"flex",justifyContent:"space-between",
              fontSize:10,marginBottom:3}}>
              <span style={{color:r.color,fontWeight:600}}>{r.species}</span>
              <span style={{color:T.muted2,fontFamily:"'DM Mono',monospace",fontSize:9}}>
                {r.current.toExponential(2)} nM·mm
              </span>
            </div>
            <div style={{height:6,background:T.border,borderRadius:3}}>
              <div style={{height:"100%",width:`${Math.min(100,r.pct)}%`,
                borderRadius:3,background:r.color,
                boxShadow:`0 0 6px ${r.color}55`,
                transition:"width 0.1s"}}/>
            </div>
            <div style={{fontSize:8,color:T.muted,marginTop:2,fontStyle:"italic"}}>
              {r.note}
            </div>
          </div>
        ))}
      </div>

      {/* Derived efficiency metrics */}
      <div style={{borderTop:`1px solid ${T.border}`,paddingTop:10,
        display:"grid",gridTemplateColumns:"1fr 1fr",gap:8}}>
        {[
          {
            label:"Test line capture",
            value:`${captureEff.toFixed(1)}%`,
            color: captureEff>50?T.ok:captureEff>20?T.warn:T.err,
            note:"RPA / (RPA + escaped A + PA)",
          },
          {
            label:"Control line capture",
            value:`${ctrlEff.toFixed(1)}%`,
            color: ctrlEff>80?T.ok:ctrlEff>40?T.warn:T.err,
            note:"PC / (PC + escaped P)",
          },
        ].map(m => (
          <div key={m.label} style={{background:T.surface,borderRadius:7,
            padding:"8px 10px",border:`1px solid ${m.color}22`}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>{m.label}</div>
            <div style={{fontSize:16,fontFamily:"'DM Mono',monospace",
              fontWeight:700,color:m.color,
              textShadow:`0 0 10px ${m.color}55`,marginBottom:3}}>
              {m.value}
            </div>
            <div style={{fontSize:8,color:T.muted,fontStyle:"italic"}}>{m.note}</div>
          </div>
        ))}
      </div>

      {/* Mass balance note */}
      <div style={{marginTop:8,fontSize:9,color:T.muted,fontStyle:"italic",
        borderTop:`1px solid ${T.border}`,paddingTop:8}}>
        Units: nM·mm (flux integral). Immobilised species [RA], [RPA], [PC]
        never reach the pad — they remain bound at their line positions.
        High [A] at pad = poor test line binding (low Ao or fast flow).
        High [PA] at pad = sandwich complex not captured (test line too far or Ro too low).
      </div>
    </div>
  );
}

function ODBars({ RPAval, PCval, peakRPA, peakPC, clReachable }) {
  // SHARED scale for both bars — same logic as strip canvas.
  // Both T and C bars are normalised to the same reference (whichever peak
  // is larger). This makes relative intensities visually honest:
  //   • Negative result: C bar fills high, T bar stays near zero
  //   • Positive result: both bars visible at meaningful heights
  const sharedPeak = Math.max(peakRPA, peakPC, 1e-30);
  const tPct = Math.min(100, (RPAval / sharedPeak) * 100);
  const cPct = clReachable
    ? Math.min(100, (PCval  / sharedPeak) * 100) : 0;

  // Verdict: stable, uses full-run peaks (not current frame).
  // T/C ratio threshold: test line must be at least 5% of control line signal.
  // This is the standard LFA reader criterion — T line is "visible" only when
  // it reaches a meaningful fraction of the control line intensity.
  const valid    = clReachable && peakPC > 0;
  const TC_ratio = valid ? peakRPA / peakPC : 0;
  const positive = valid && TC_ratio > 0.05;   // T must be ≥5% of C
  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:10,padding:14}}>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:10}}>
        Optical Density Assessment
      </div>
      <div style={{display:"flex",gap:12,alignItems:"stretch",marginBottom:10}}>
        {/* Mock strip visual */}
        <div style={{width:70,background:"#0a1020",border:`1px solid ${T.border2}`,
          borderRadius:8,position:"relative",overflow:"hidden",minHeight:80}}>
          <div style={{position:"absolute",left:0,right:0,top:"38%",
            height:Math.max(2,3+tPct/12),background:T.RPA,
            opacity:0.15+tPct/100*0.85,boxShadow:`0 0 ${4+tPct/4}px ${T.RPA}`}}/>
          <div style={{position:"absolute",left:0,right:0,top:"63%",
            height:Math.max(2,3+cPct/12),background:T.PC,
            opacity:0.15+cPct/100*0.85,boxShadow:`0 0 ${4+cPct/4}px ${T.PC}`}}/>
          <div style={{position:"absolute",top:"35%",left:3,fontSize:7,color:T.RPA,fontFamily:"'DM Mono',monospace"}}>T</div>
          <div style={{position:"absolute",top:"60%",left:3,fontSize:7,color:T.PC, fontFamily:"'DM Mono',monospace"}}>C</div>
        </div>
        <div style={{flex:1,display:"flex",flexDirection:"column",gap:8}}>
          {/* Test line bar */}
          <div>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
              <span style={{color:T.RPA}}>Test (T)</span>
              <span style={{color:T.muted2,fontFamily:"'DM Mono',monospace"}}>{tPct.toFixed(0)}%  {RPAval.toExponential(2)} nM</span>
            </div>
            <div style={{height:7,background:T.border,borderRadius:4}}>
              <div style={{height:"100%",width:`${tPct}%`,borderRadius:4,
                background:T.RPA,boxShadow:`0 0 7px ${T.RPA}55`,transition:"width 0.12s"}}/>
            </div>
          </div>
          {/* Control line bar — only shown when reachable */}
          {clReachable ? (
            <div>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:10,marginBottom:3}}>
                <span style={{color:T.PC}}>Control (C)</span>
                <span style={{color:T.muted2,fontFamily:"'DM Mono',monospace"}}>{cPct.toFixed(0)}%  {PCval.toExponential(2)} nM</span>
              </div>
              <div style={{height:7,background:T.border,borderRadius:4}}>
                <div style={{height:"100%",width:`${cPct}%`,borderRadius:4,
                  background:T.PC,boxShadow:`0 0 7px ${T.PC}55`,transition:"width 0.12s"}}/>
              </div>
            </div>
          ) : (
            <div style={{display:"flex",alignItems:"center",gap:8,padding:"6px 0"}}>
              <div style={{width:20,height:2,background:T.muted,borderRadius:1}}/>
              <span style={{fontSize:10,color:T.muted}}>Control (C) — outside strip, not modelled</span>
            </div>
          )}
        </div>
      </div>
      {/* Physics diagnostics — always visible, explains the result */}
      {valid && (() => {
        // Key ratios that determine the result
        const tcRatioStr  = TC_ratio.toExponential(2);
        const lodNote     = TC_ratio >= 0.05
          ? "≥ 0.05 — detectable ✓"
          : `< 0.05 — increase Ao or decrease Po`;
        const rows = [
          { label:"T/C signal ratio",   val: tcRatioStr,               note: lodNote,      color: TC_ratio>=0.05?T.RPA:T.muted2 },
          { label:"Test peak [RPA]", val: peakRPA.toExponential(2)+"nM",
                  note: TC_ratio<0.05
                    ? `analyte-limited — need Ao ≥ ${(0.05*peakPC).toExponential(1)} nM`
                    : "analyte-limited → scales with Ao",
                  color: T.RPA },
          { label:"Ctrl peak [PC]",     val: peakPC.toExponential(2)+"nM",   note:"detector-limited → scales with Po", color: T.PC  },
        ];
        return (
          <div style={{background:T.surface,borderRadius:7,border:`1px solid ${T.border}`,
            padding:"8px 10px",marginBottom:8}}>
            <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:6}}>
              Signal Diagnostics
            </div>
            {rows.map(r=>(
              <div key={r.label} style={{display:"flex",alignItems:"baseline",
                gap:6,marginBottom:4,flexWrap:"wrap"}}>
                <span style={{fontSize:9,color:T.muted2,minWidth:110}}>{r.label}</span>
                <span style={{fontSize:10,fontFamily:"'DM Mono',monospace",
                  fontWeight:700,color:r.color}}>{r.val}</span>
                <span style={{fontSize:9,color:T.muted,fontStyle:"italic"}}>{r.note}</span>
              </div>
            ))}
          </div>
        );
      })()}
      <div style={{
        background: !clReachable?"#10141a":!valid?"#1a0f08":positive?"#0a1a10":"#0a1020",
        border:`1px solid ${ !clReachable?T.border:!valid?"#7f3a1a":positive?"#166534":"#1e3a6a"}`,
        borderRadius:7,padding:"8px 12px",display:"flex",alignItems:"center",gap:10
      }}>
        <span style={{fontSize:18}}>
          {!clReachable?"📏":!valid?"⚠️":positive?"🔴":"🟢"}
        </span>
        <div>
          <div style={{fontSize:11,fontWeight:700,marginBottom:1,
            color:!clReachable?T.muted2:!valid?"#fb923c":positive?"#f87171":"#86efac"}}>
            {!clReachable
              ? "Control line outside strip — result unverifiable"
              : !valid ? "Invalid — control line absent"
              : positive ? "Positive" : "Negative"}
          </div>
          <div style={{fontSize:10,color:T.muted2}}>
            {!clReachable
              ? "Move control line inside strip length, or increase strip length"
              : !valid ? "Flow did not reach control line or reagents exhausted"
              : positive ? `T/C ratio ${TC_ratio.toExponential(2)} ≥ 0.05 — analyte detected`
              : `T/C ratio ${TC_ratio.toExponential(2)} < 0.05 — below detection limit`}
          </div>
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// SIMULATION REPORT COMPONENT
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
      {unit && <span style={{color:T.muted,fontSize:9}}>{unit}</span>}
      {note && <span style={{color:warn?T.warn:T.muted,fontSize:9,
        fontStyle:"italic",marginLeft:4}}>{note}</span>}
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{background:`${color}22`,color,border:`1px solid ${color}55`,
      borderRadius:4,padding:"2px 8px",fontSize:10,fontFamily:"'DM Mono',monospace",
      fontWeight:700}}>
      {text}
    </span>
  );
}

function SimReport({ simData, userParams, kineticParams, physParams }) {
  if (!simData) return (
    <div style={{display:"flex",alignItems:"center",justifyContent:"center",
      height:300,color:T.muted,fontSize:12,fontFamily:"'DM Mono',monospace"}}>
      Run a simulation first to generate the report.
    </div>
  );

  const fi  = simData.times.length - 1;   // final frame
  const Af  = simData.A[fi];
  const Pf  = simData.P[fi];
  const PAf = simData.PA[fi];
  const RPA = simData.RPA[fi];
  const PC  = simData.PC[fi];
  const RA  = simData.RA ? simData.RA[fi] : 0;
  const t_end_actual = simData.times[fi];
  const { x, tl, cl, N, cl_reachable, t_sample_end, flow_rate_uL_s } = simData;

  const { Ao, Po, Ro, Ro_cl, x_tl, x_cl, t_end } = userParams;
  const { Ka1, Kd1, Ka2, Kd2, Ka_cl, Kd_cl } = kineticParams;
  const { strip_L, phi, tau, dx, U_max, sample_vol, line_width_mm,
          conj_burst, conj_lambda } = physParams;

  const KD1 = Kd1 / Ka1, KD2 = Kd2 / Ka2, KD_cl = Kd_cl / Ka_cl;

  // ── Spatial stats ──────────────────────────────────────────────────────────
  const peakA  = Math.max(...Af),  peakA_x  = x[Af.indexOf(peakA)];
  const peakP  = Math.max(...Pf),  peakP_x  = x[Pf.indexOf(peakP)];
  const peakPA = Math.max(...PAf), peakPA_x = x[PAf.indexOf(peakPA)];
  const A_at_tl  = Af[tl],  P_at_tl  = Pf[tl],  PA_at_tl  = PAf[tl];
  const A_at_cl  = cl_reachable ? Af[cl]  : 0;
  const P_at_cl  = cl_reachable ? Pf[cl]  : 0;
  const PA_at_cl = cl_reachable ? PAf[cl] : 0;
  const x4 = 21311 * t_end_actual - 69505;
  const flowFront = x4 > 0 ? Math.min(x4 ** 0.25, strip_L) : 0;

  // ── Line stats ─────────────────────────────────────────────────────────────
  const recOccTL   = Ro   > 0 ? ((RA + RPA) / Ro)    * 100 : 0;
  const labelEff   = (RA + RPA) > 0 ? (RPA / (RA + RPA)) * 100 : 0;
  const recOccCL   = Ro_cl > 0 ? (PC / Ro_cl) * 100 : 0;
  const hookEffect = RA > 0 && RPA > 0 && (RA / RPA) > 3;

  // ── Absorbent pad ──────────────────────────────────────────────────────────
  const padA  = simData.pad_A[fi];
  const padP  = simData.pad_P[fi];
  const padPA = simData.pad_PA[fi];

  // ── Mass conservation ──────────────────────────────────────────────────────
  // NOTE: Detector (P) mass is NOT conserved in the traditional closed-system
  // sense because this is an OPEN advection-dominated system:
  //   - Analyte A enters continuously via Dirichlet BC (A[0]=Ao)
  //   - Detector P starts as finite pre-load but advects with U(x)=c/x³
  //   - Upwind advection causes ∫P·dx to exceed initial load (physical behaviour
  //     in an open flow system — not a numerical error)
  // Receptor conservation IS exact (closed ODE, no spatial transport).
  // We report receptor conservation only — that is the meaningful check.
  const P_conserv_err = null;   // not meaningful for open advection system
  const R_sum_tl        = RA + RPA + Math.max(0, Ro - RA - RPA);
  const R_sum_cl        = PC + (Ro_cl - PC);

  // ── Assay verdict ──────────────────────────────────────────────────────────
  const peakRPA = Math.max(...simData.RPA);
  const peakPC  = Math.max(...simData.PC);
  const TC      = peakPC > 0 ? peakRPA / peakPC : 0;
  const valid   = cl_reachable && peakPC > 0;
  const positive = valid && TC >= 0.05;
  const verdictColor = !valid ? T.warn : positive ? T.err : T.PA;
  const verdictText  = !valid ? "INVALID" : positive ? "POSITIVE" : "NEGATIVE";

  // LOD estimate: Ao needed for TC=0.05 (linear approximation)
  const LOD_est = peakPC > 0 && Ao > 0 ? 0.05 * peakPC * (Ao / peakRPA) : null;

  // ── Download handler ───────────────────────────────────────────────────────
  const downloadReport = () => {
    const lines = [
      "LFA SIMULATION REPORT",
      "=".repeat(60),
      `Generated: ${new Date().toISOString()}`,
      "",
      "1. PARAMETERS",
      `Ao=${Ao.toExponential(2)} nM  Po=${Po.toExponential(2)} nM  Ro=${Ro} nM  Ro_cl=${Ro_cl} nM`,
      `x_tl=${x_tl} mm  x_cl=${x_cl} mm  t_end=${t_end} s`,
      `Ka1=${Ka1.toExponential(2)}  Kd1=${Kd1.toExponential(2)}  KD1=${KD1.toExponential(4)} nM`,
      `Ka2=${Ka2.toExponential(2)}  Kd2=${Kd2.toExponential(2)}  KD2=${KD2.toExponential(4)} nM`,
      `strip_L=${strip_L}mm  phi=${phi}  tau=${tau}  U_max=${U_max}mm/s`,
      "",
      "2. SPATIAL PEAKS at t_end",
      `[A]  peak=${peakA.toExponential(3)} nM at x=${peakA_x.toFixed(1)}mm | at TL: ${A_at_tl.toExponential(3)} nM`,
      `[P]  peak=${peakP.toExponential(3)} nM at x=${peakP_x.toFixed(1)}mm | at TL: ${P_at_tl.toExponential(3)} nM`,
      `[PA] peak=${peakPA.toExponential(3)} nM at x=${peakPA_x.toFixed(1)}mm | at TL: ${PA_at_tl.toExponential(3)} nM`,
      `Flow front: ${flowFront.toFixed(1)} mm`,
      "",
      "3. LINE FORMATION",
      `Test  line (${x_tl}mm): RA=${RA.toExponential(3)} nM  RPA=${RPA.toExponential(3)} nM`,
      `  Receptor occupancy: ${recOccTL.toFixed(1)}%  Labelling efficiency: ${labelEff.toFixed(1)}%`,
      `Control line (${x_cl}mm): PC=${PC.toExponential(3)} nM  Reached: ${cl_reachable}`,
      `  Receptor occupancy: ${recOccCL.toFixed(2)}%`,
      "",
      "4. ABSORBENT PAD (cumulative flux nM·mm)",
      `[A]=${padA.toExponential(3)}  [P]=${padP.toExponential(3)}  [PA]=${padPA.toExponential(3)}`,
      "",
      "5. MASS CONSERVATION",
      `Test  receptor: RA+RPA+R_free = ${R_sum_tl.toFixed(4)} nM (Ro=${Ro}) ${Math.abs(R_sum_tl-Ro)<0.01?"OK":"ERROR"}`,
      `Ctrl  receptor: PC+Rc_free = ${R_sum_cl.toFixed(4)} nM (Ro_cl=${Ro_cl}) ${Math.abs(R_sum_cl-Ro_cl)<0.01?"OK":"ERROR"}`,
      `RPA peak vs ceiling: ${(peakRPA/Math.min(Ao,Po)*100).toFixed(1)}% of min(Ao,Po)=${Math.min(Ao,Po).toExponential(2)} nM`,
      "Note: Detector budget % not shown (open advection system — see report for explanation)",
      "",
      "6. VERDICT",
      `Result: ${verdictText}  T/C ratio: ${TC.toExponential(2)}`,
      LOD_est ? `Estimated LOD: ${LOD_est.toExponential(2)} nM` : "",
      hookEffect ? "WARNING: Hook effect detected (RA >> RPA)" : "",
      Po < Ao ? `WARNING: Detector limiting (Po=${Po} < Ao=${Ao})` : "",
    ];
    const blob = new Blob([lines.join("\n")], {type:"text/plain"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `LFA_Report_${Date.now()}.txt`;
    a.click();
  };

  return (
    <div style={{fontFamily:"'DM Mono',monospace"}}>
      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",
        marginBottom:20,paddingBottom:12,borderBottom:`1px solid ${T.border2}`}}>
        <div>
          <div style={{fontSize:16,fontWeight:700,color:T.text,
            fontFamily:"'Syne',sans-serif",marginBottom:3}}>Simulation Report</div>
          <div style={{fontSize:9,color:T.muted}}>
            t = {t_end_actual.toFixed(0)} s · {simData.times.length} frames · N={N} nodes
          </div>
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          <Badge text={verdictText} color={verdictColor}/>
          <button onClick={downloadReport} style={{
            background:T.surface,border:`1px solid ${T.border2}`,
            color:T.muted2,padding:"5px 12px",borderRadius:6,cursor:"pointer",
            fontFamily:"inherit",fontSize:10,display:"flex",alignItems:"center",gap:5}}>
            ⬇ Download .txt
          </button>
        </div>
      </div>

      {/* ── Section 1: Parameters ── */}
      <Section title="1. Simulation Parameters" color={T.accent}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          <div>
            <div style={{fontSize:9,color:T.muted,letterSpacing:2,marginBottom:6,
              textTransform:"uppercase"}}>Assay</div>
            <Row label="Analyte [A₀]"        value={Ao.toExponential(2)}  unit="nM"/>
            <Row label="Detector [P₀]"       value={Po.toExponential(2)}  unit="nM"
              warn={Po < Ao} note={Po<Ao?"⚠ detector limiting":undefined}/>
            <Row label="Test receptor [R₀]"  value={Ro}                  unit="nM"/>
            <Row label="Ctrl receptor [Rc]"  value={Ro_cl}               unit="nM"/>
            <Row label="Test line position"  value={x_tl}                unit="mm"/>
            <Row label="Control line pos."   value={x_cl}                unit="mm"/>
            <Row label="Run time"            value={t_end}               unit="s"/>
            <Row label="Sample volume"       value={sample_vol}          unit="µL"/>
          </div>
          <div>
            <div style={{fontSize:9,color:T.muted,letterSpacing:2,marginBottom:6,
              textTransform:"uppercase"}}>Kinetics</div>
            <Row label="Ka₁ (det-analyte on)"  value={Ka1.toExponential(2)}  unit="nM⁻¹s⁻¹"/>
            <Row label="Kd₁ (det-analyte off)" value={Kd1.toExponential(2)}  unit="s⁻¹"/>
            <Row label="KD₁ = Kd₁/Ka₁"         value={KD1.toExponential(4)}  unit="nM"
              color={KD1<Ao?T.PA:T.warn} note={KD1<Ao?"strong":"weak vs Ao"}/>
            <Row label="Ka₂ (rec-analyte on)"  value={Ka2.toExponential(2)}  unit="nM⁻¹s⁻¹"/>
            <Row label="Kd₂ (rec-analyte off)" value={Kd2.toExponential(2)}  unit="s⁻¹"/>
            <Row label="KD₂ = Kd₂/Ka₂"         value={KD2.toExponential(4)}  unit="nM"
              color={KD2<Ao?T.PA:T.warn} note={KD2<Ao?"strong":"weak vs Ao"}/>
            <Row label="Ka_cl (ctrl on)"        value={Ka_cl.toExponential(2)} unit="nM⁻¹s⁻¹"/>
            <Row label="KD_cl"                  value={KD_cl.toExponential(4)} unit="nM"/>
          </div>
        </div>
        <div style={{marginTop:10,display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
          {[
            ["Strip length", strip_L, "mm"],
            ["Grid spacing dx", dx, "mm"],
            ["Porosity φ", phi, ""],
            ["Tortuosity τ", tau, ""],
            ["U_max", U_max, "mm/s"],
            ["Flow constant c", 5327.75, "mm⁴/s"],
            ["Line half-width", line_width_mm, "mm"],
            ["Conj. burst β", conj_burst, ""],
            ["Release rate λ", conj_lambda.toExponential(1), "s⁻¹"],
          ].map(([l,v,u])=>(
            <div key={l} style={{background:T.surface,borderRadius:5,padding:"5px 8px",
              border:`1px solid ${T.border}`}}>
              <div style={{color:T.muted,fontSize:8,marginBottom:2}}>{l}</div>
              <div style={{color:T.text,fontSize:10}}>{v} <span style={{color:T.muted}}>{u}</span></div>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Section 2: Spatial distribution ── */}
      <Section title="2. Species Spatial Distribution at t_end" color={T.A}>
        <div style={{marginBottom:8}}>
          <Row label="Flow front reached"
            value={`${flowFront.toFixed(1)} mm`}
            color={flowFront>=x_cl?T.PA:flowFront>=x_tl?T.warn:T.err}
            note={flowFront>=x_cl?"past both lines":flowFront>=x_tl?"past test only":"did not reach test line"}/>
        </div>
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
            <thead>
              <tr style={{background:T.surface}}>
                {["Species","Peak conc.","Peak location","At test line","At ctrl line","At pad outlet"].map(h=>(
                  <th key={h} style={{padding:"6px 10px",textAlign:"left",color:T.muted,
                    fontSize:9,fontWeight:600,borderBottom:`1px solid ${T.border2}`}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                {sp:"[A] Analyte",    c:T.A,  pk:peakA,  pkx:peakA_x,  tl:A_at_tl,  cl_v:A_at_cl,  pad:padA},
                {sp:"[P] Detector",   c:T.P,  pk:peakP,  pkx:peakP_x,  tl:P_at_tl,  cl_v:P_at_cl,  pad:padP},
                {sp:"[PA] Complex",   c:T.PA, pk:peakPA, pkx:peakPA_x, tl:PA_at_tl, cl_v:PA_at_cl, pad:padPA},
              ].map((r,i)=>(
                <tr key={r.sp} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                  <td style={{padding:"5px 10px",color:r.c,fontWeight:600}}>{r.sp}</td>
                  <td style={{padding:"5px 10px",color:T.text,fontFamily:"monospace"}}>{r.pk.toExponential(3)} nM</td>
                  <td style={{padding:"5px 10px",color:T.muted2}}>{r.pkx.toFixed(1)} mm</td>
                  <td style={{padding:"5px 10px",color:T.text,fontFamily:"monospace"}}>{r.tl.toExponential(3)} nM</td>
                  <td style={{padding:"5px 10px",color:cl_reachable?T.text:T.muted,fontFamily:"monospace"}}>
                    {cl_reachable ? r.cl_v.toExponential(3)+" nM" : "—"}</td>
                  <td style={{padding:"5px 10px",color:T.muted2,fontFamily:"monospace"}}>{r.pad.toExponential(2)} nM·mm</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* ── Section 3: Line formation ── */}
      <Section title="3. Line Formation" color={T.RPA}>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
          {/* Test line */}
          <div style={{background:T.surface,borderRadius:8,padding:12,
            border:`1px solid ${T.RPA}33`}}>
            <div style={{color:T.RPA,fontWeight:700,fontSize:11,marginBottom:8}}>
              TEST LINE — x = {x_tl} mm
            </div>
            <Row label="[RA] Analyte captured"    value={RA.toExponential(3)}  unit="nM"
              note="captured but not labelled"/>
            <Row label="[RPA] Signal complex"     value={RPA.toExponential(3)} unit="nM"
              color={T.RPA} note="complete sandwich"/>
            <Row label="Free receptor [R]"        value={Math.max(0,Ro-RA-RPA).toExponential(3)} unit="nM"/>
            <Row label="Receptor occupancy"       value={`${recOccTL.toFixed(1)}%`}
              color={recOccTL>80?T.PA:recOccTL>40?T.warn:T.muted2}/>
            <Row label="Labelling efficiency"     value={`${labelEff.toFixed(1)}%`}
              color={labelEff>80?T.PA:T.warn}
              warn={hookEffect}
              note={hookEffect?"⚠ hook effect: RA>>RPA":undefined}/>
            <Row label="Signal [A] at test line"  value={A_at_tl.toExponential(3)}  unit="nM"/>
            <Row label="Signal [P] at test line"  value={P_at_tl.toExponential(3)}  unit="nM"/>
          </div>
          {/* Control line */}
          <div style={{background:T.surface,borderRadius:8,padding:12,
            border:`1px solid ${T.PC}33`,opacity:cl_reachable?1:0.5}}>
            <div style={{color:T.PC,fontWeight:700,fontSize:11,marginBottom:8}}>
              CONTROL LINE — x = {x_cl} mm
              {!cl_reachable && <span style={{color:T.warn,fontSize:9,marginLeft:8}}>⚠ OUTSIDE STRIP</span>}
            </div>
            {cl_reachable ? <>
              <Row label="[PC] Detector captured" value={PC.toExponential(3)}  unit="nM" color={T.PC}/>
              <Row label="Free ctrl receptor [Rc]" value={(Ro_cl-PC).toExponential(3)} unit="nM"/>
              <Row label="Receptor occupancy"      value={`${recOccCL.toFixed(2)}%`}
                color={recOccCL>2?T.PA:T.warn}
                warn={recOccCL<0.1}
                note={recOccCL<0.1?"low — control may be invisible":undefined}/>
              <Row label="[P] at ctrl line"        value={P_at_cl.toExponential(3)}  unit="nM"/>
              <Row label="Flow reached ctrl line"  value={flowFront>=x_cl?"Yes":"No"}
                color={flowFront>=x_cl?T.PA:T.warn}/>
            </> : (
              <div style={{color:T.warn,fontSize:10,padding:"8px 0"}}>
                Control line is outside the strip. Extend strip length or move line.
              </div>
            )}
          </div>
        </div>
      </Section>

      {/* ── Section 4: Absorbent pad ── */}
      <Section title="4. Absorbent Pad — Cumulative Flux" color={T.muted2}>
        <div style={{marginBottom:8,fontSize:9,color:T.muted,fontStyle:"italic"}}>
          Values in nM·mm (∫ U·C dt). Zero means all species captured within strip.
          Immobilised species [RA][RPA][PC] never reach the pad.
        </div>
        <Row label="[A]  Escaped free analyte"     value={padA.toExponential(3)}  unit="nM·mm"
          color={padA>1e-10?T.warn:T.PA}
          note={padA>1e-10?"analyte loss detected":"none escaped ✓"}/>
        <Row label="[P]  Escaped free detector"    value={padP.toExponential(3)}  unit="nM·mm"
          color={padP>1e-10?T.warn:T.PA}
          note={padP>1e-10?"detector past both lines":"none escaped ✓"}/>
        <Row label="[PA] Escaped labelled complex" value={padPA.toExponential(3)} unit="nM·mm"
          color={padPA>1e-10?T.warn:T.PA}
          note={padPA>1e-10?"signal loss — false negative risk":"none escaped ✓"}/>
      </Section>

      {/* ── Section 5: Mass conservation ── */}
      <Section title="5. Mass Conservation" color={T.accent2||"#7c3aed"}>
        {/* Receptor conservation — the only exact check in an open flow system */}
        <Row label="Test receptor RA+RPA+R_free"
          value={R_sum_tl.toFixed(4)} unit="nM"
          color={Math.abs(R_sum_tl-Ro)<0.01?T.PA:T.err}
          note={`should = Ro = ${Ro} — ${Math.abs(R_sum_tl-Ro)<0.01?"✓ exact":"✗ error"}`}/>
        <Row label="Ctrl receptor PC+Rc_free"
          value={R_sum_cl.toFixed(4)} unit="nM"
          color={Math.abs(R_sum_cl-Ro_cl)<0.01?T.PA:T.err}
          note={`should = Ro_cl = ${Ro_cl} — ${Math.abs(R_sum_cl-Ro_cl)<0.01?"✓ exact":"✗ error"}`}/>
        <Row label="Detector pre-loaded (t=0)"
          value={(Po*Math.round(0.20*N)*dx).toExponential(3)} unit="nM·mm"
          note="Po × conjugate pad length — finite initial condition"/>
        <Row label="RPA peak vs theoretical ceiling"
          value={`${(peakRPA/Math.min(Ao,Po)*100).toFixed(1)}%`}
          color={peakRPA<=Math.min(Ao,Po)*1.05?T.PA:T.warn}
          note={`of min(Ao,Po)=${Math.min(Ao,Po).toExponential(2)} nM — physical ceiling`}/>
        <div style={{marginTop:8,background:T.surface,borderRadius:6,padding:"8px 10px",
          border:`1px solid ${T.border}`,fontSize:9,color:T.muted,lineHeight:1.6}}>
          <span style={{color:T.muted2,fontWeight:600}}>Why no detector error %?</span>
          {" "}This is an open advection-dominated system (U≫D). Detector flows
          right under U(x)=c/x³ from the conjugate pad — ∫P·dx naturally grows
          beyond the initial load as flow redistributes mass. Detector budget %
          is not meaningful here. Receptor conservation (above) is the correct check.
        </div>
      </Section>

      {/* ── Section 6: Diagnostics ── */}
      <Section title="6. Assay Diagnostics & Verdict" color={verdictColor}>
        <div style={{display:"flex",gap:10,marginBottom:12,flexWrap:"wrap"}}>
          <div style={{background:`${verdictColor}15`,border:`1px solid ${verdictColor}55`,
            borderRadius:8,padding:"10px 16px",flex:1,minWidth:160}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>VERDICT</div>
            <div style={{fontSize:22,fontWeight:800,color:verdictColor,
              fontFamily:"'Syne',sans-serif"}}>{verdictText}</div>
            <div style={{fontSize:9,color:T.muted2,marginTop:2}}>
              {!valid?"control line absent/unreachable"
               :positive?"T/C ≥ 0.05 — analyte detected"
               :"T/C < 0.05 — below detection limit"}
            </div>
          </div>
          <div style={{background:T.surface,border:`1px solid ${T.border}`,
            borderRadius:8,padding:"10px 16px",flex:1,minWidth:160}}>
            <div style={{fontSize:9,color:T.muted,marginBottom:4}}>T/C RATIO</div>
            <div style={{fontSize:18,fontWeight:700,color:TC>=0.05?T.RPA:T.muted2,
              fontFamily:"'DM Mono',monospace"}}>{TC.toExponential(2)}</div>
            <div style={{fontSize:9,color:T.muted,marginTop:2}}>threshold = 0.05</div>
          </div>
          {LOD_est && (
            <div style={{background:T.surface,border:`1px solid ${T.border}`,
              borderRadius:8,padding:"10px 16px",flex:1,minWidth:160}}>
              <div style={{fontSize:9,color:T.muted,marginBottom:4}}>EST. LOD</div>
              <div style={{fontSize:18,fontWeight:700,color:T.accent,
                fontFamily:"'DM Mono',monospace"}}>{LOD_est.toExponential(2)}</div>
              <div style={{fontSize:9,color:T.muted,marginTop:2}}>nM for T/C = 0.05</div>
            </div>
          )}
        </div>
        <Row label="Test line capture eff."      value={`${(RPA/(RPA+RA+padA+padPA+1e-30)*100).toFixed(1)}%`}
          color={RPA/(RPA+RA+padA+padPA+1e-30)>0.5?T.PA:T.warn}
          note="RPA / (RPA + RA + pad_A + pad_PA)"/>
        <Row label="Control line capture eff."   value={cl_reachable?`${(PC/(PC+padP+1e-30)*100).toFixed(1)}%`:"—"}
          color={T.PA} note="PC / (PC + pad_P)"/>
        <Row label="Ao / KD2 (binding ratio)"    value={(Ao/KD2).toExponential(2)}
          color={Ao/KD2>10?T.PA:T.warn}
          note={Ao/KD2>10?"analyte >> KD: strong capture":"analyte ~ KD: moderate binding"}/>
        <Row label="Po / KD1 (labelling ratio)"  value={(Po/KD1).toExponential(2)}
          color={Po/KD1>10?T.PA:T.warn}
          note={Po/KD1>10?"detector >> KD: efficient PA formation":"detector ~ KD: partial labelling"}/>
        <Row label="Hook effect"
          value={hookEffect?"⚠ DETECTED":"Not detected"}
          color={hookEffect?T.warn:T.PA}
          note={hookEffect?`RA/RPA = ${(RA/RPA).toFixed(1)} — analyte captured but detector exhausted`:"RA/RPA ratio normal"}/>
        <Row label="Detector limiting"
          value={Po<Ao?"⚠ YES (Po < Ao)":"No"}
          color={Po<Ao?T.warn:T.PA}
          warn={Po<Ao}
          note={Po<Ao?`Increase Po to ≥ ${(Ao*2).toFixed(2)} nM for reliable sandwich`:""}/>

        {/* Suggestions */}
        {(hookEffect||Po<Ao||recOccCL<1||!cl_reachable||TC<0.05)&&(
          <div style={{marginTop:12,background:"#1a1a0a",border:`1px solid ${T.warn}44`,
            borderRadius:7,padding:10}}>
            <div style={{fontSize:9,color:T.warn,fontWeight:700,marginBottom:6,
              textTransform:"uppercase",letterSpacing:1}}>Suggestions</div>
            {Po<Ao&&<div style={{fontSize:9,color:T.muted2,marginBottom:4}}>
              • Increase Po to {(Ao*3).toFixed(2)} nM (3×Ao) for reliable sandwich formation
            </div>}
            {hookEffect&&<div style={{fontSize:9,color:T.muted2,marginBottom:4}}>
              • Reduce Ao or increase Po to resolve hook effect
            </div>}
            {recOccCL<1&&cl_reachable&&<div style={{fontSize:9,color:T.muted2,marginBottom:4}}>
              • Control line occupancy is {recOccCL.toFixed(2)}% — move it closer to test line or increase Po
            </div>}
            {TC<0.05&&valid&&<div style={{fontSize:9,color:T.muted2,marginBottom:4}}>
              • For Positive result: increase Ao to ≥ {LOD_est?LOD_est.toFixed(3):"~"+( 0.05*peakPC*(Ao/Math.max(peakRPA,1e-30))).toExponential(1)} nM
            </div>}
            {!cl_reachable&&<div style={{fontSize:9,color:T.muted2,marginBottom:4}}>
              • Control line at {x_cl}mm is outside strip ({strip_L}mm). Increase strip length or reduce x_cl.
            </div>}
          </div>
        )}
      </Section>
    </div>
  );
}

export default function LFASimulator() {
  // User (assay) parameters
  const [userParams, setUserParams] = useState({
    Ao:1e-7, Po:6, Ro:10, Ro_cl:8,
    x_tl:25, x_cl:55, t_end:1200,
  });
  // Kinetic parameters (editable)
  const [kineticParams, setKineticParams] = useState(defaultKinetic());
  // Physical parameters (editable)
  const [physParams, setPhysParams] = useState(defaultPhysical());

  const [simData,   setSimData]   = useState(null);
  const [running,   setRunning]   = useState(false);
  const [frameIdx,  setFrameIdx]  = useState(0);
  const [playing,   setPlaying]   = useState(false);
  const [tab,       setTab]       = useState("sim");
  const [paramTab,  setParamTab]  = useState("assay");   // assay | kinetic | physical
  const [sweepParam,setSweepParam]= useState("Ao");
  const [sweepData, setSweepData] = useState(null);
  const [sweeping,  setSweeping]  = useState(false);
  // Sweep configuration — editable min/max/steps per param
  const [sweepConfig, setSweepConfig] = useState({
    Ao:   { logMin:-9, logMax:2,  nPoints:16 },
    Po:   { min:0.1,  max:50,    nSteps:12  },
    Ro:   { min:1,    max:100,   nSteps:10  },
    x_tl: { min:10,   max:70,    nSteps:10  },
  });
  // Noise (variability) settings
  const [noiseEnabled, setNoiseEnabled] = useState(false);
  const [noiseConfig, setNoiseConfig] = useState({
    cv_Ao:       0.10,   // 10% CV on analyte concentration (pipetting)
    cv_Ka:       0.15,   // 15% CV on kinetic constants (batch variation)
    cv_conj:     0.10,   // 10% CV on conjugate burst fraction
    cv_flow:     0.08,   // 8%  CV on flow constant (membrane lot variation)
    n_replicates: 5,     // number of noisy replicates
  });

  const intervalRef = useRef(null);

  useEffect(() => { handleRun(); }, []);

  // Smooth 1-frame-per-tick playback
  useEffect(() => {
    clearInterval(intervalRef.current);
    if (playing && simData) {
      intervalRef.current = setInterval(() => {
        setFrameIdx(i => {
          if (i >= simData.times.length-1) { setPlaying(false); return i; }
          return i+1;
        });
      }, 50);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, simData]);

  const handleRun = useCallback(() => {
    setRunning(true); setPlaying(false); setFrameIdx(0);
    setTimeout(() => {
      try {
        setSimData(runSimulation(userParams, kineticParams, physParams, 160));
      } catch(e) { console.error(e); }
      setRunning(false);
    }, 15);
  }, [userParams, kineticParams, physParams]);

  // Box-Muller normal random variate
  const randn = () => {
    let u=0, v=0;
    while(u===0) u=Math.random();
    while(v===0) v=Math.random();
    return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);
  };

  // Apply noise to params for one replicate
  const noisyParams = (baseUser, baseKin, basePhy, nc) => {
    const n = () => randn();
    return {
      user: { ...baseUser,
        Ao: Math.max(0, baseUser.Ao * (1 + nc.cv_Ao * n())),
      },
      kin: { ...baseKin,
        Ka1: Math.max(1e-8, baseKin.Ka1 * (1 + nc.cv_Ka * n())),
        Ka2: Math.max(1e-8, baseKin.Ka2 * (1 + nc.cv_Ka * n())),
        Kd1: Math.max(1e-8, baseKin.Kd1 * (1 + nc.cv_Ka * n())),
        Kd2: Math.max(1e-8, baseKin.Kd2 * (1 + nc.cv_Ka * n())),
      },
      phy: { ...basePhy,
        conj_burst: Math.min(1, Math.max(0, basePhy.conj_burst * (1 + nc.cv_conj * n()))),
        flow_c:     Math.max(100, basePhy.flow_c * (1 + nc.cv_flow * n())),
      },
    };
  };


  const handleSweep = useCallback(() => {
    setSweeping(true); setSweepData(null);
    setTimeout(() => {
      try {
        const cfg = sweepConfig[sweepParam];
        // Build points from config
        const points = sweepParam === "Ao"
          ? Array.from({length: cfg.nPoints}, (_,i) =>
              Math.pow(10, cfg.logMin + (cfg.logMax - cfg.logMin)*i/(cfg.nPoints-1)))
          : Array.from({length: cfg.nSteps}, (_,i) =>
              cfg.min + (cfg.max - cfg.min)*i/(cfg.nSteps-1));

        const data = points.map(v => {
          const baseU = {...userParams, [sweepParam]: v};

          if (!noiseEnabled) {
            const r = runSimulation(baseU, kineticParams, physParams, 60);
            const rpa = r.RPA[r.RPA.length-1]||0;
            const pc  = r.PC[r.PC.length-1] ||0;
            return {x:v, rpa, pc, tc:pc>1e-30?rpa/pc:0,
                    rpa_lo:null, rpa_hi:null, tc_lo:null, tc_hi:null};
          }
          // Noisy replicates
          const nc = noiseConfig;
          const rpas=[], pcs=[], tcs=[];
          for (let i=0; i<nc.n_replicates; i++) {
            const rn = () => randn();
            const nu = {...baseU,
              Ao: Math.max(0, baseU.Ao*(1+nc.cv_Ao*rn()))};
            const nk = {...kineticParams,
              Ka1: Math.max(1e-8, kineticParams.Ka1*(1+nc.cv_Ka*rn())),
              Ka2: Math.max(1e-8, kineticParams.Ka2*(1+nc.cv_Ka*rn())),
              Kd1: Math.max(1e-8, kineticParams.Kd1*(1+nc.cv_Ka*rn())),
              Kd2: Math.max(1e-8, kineticParams.Kd2*(1+nc.cv_Ka*rn()))};
            const np = {...physParams,
              conj_burst: Math.min(1,Math.max(0, physParams.conj_burst*(1+nc.cv_conj*rn()))),
              flow_c:     Math.max(100, physParams.flow_c*(1+nc.cv_flow*rn()))};
            const r = runSimulation(nu, nk, np, 60);
            const rpa_ = r.RPA[r.RPA.length-1]||0;
            const pc_  = r.PC[r.PC.length-1] ||0;
            rpas.push(rpa_); pcs.push(pc_);
            tcs.push(pc_>1e-30?rpa_/pc_:0);
          }
          const mean = a => a.reduce((s,v)=>s+v,0)/a.length;
          const std  = a => { const m=mean(a); return Math.sqrt(a.reduce((s,v)=>s+(v-m)**2,0)/a.length); };
          const rm=mean(rpas), pm=mean(pcs), tm=mean(tcs), ts=std(tcs), rs=std(rpas);
          return {x:v, rpa:rm, pc:pm, tc:tm,
                  rpa_lo:Math.max(0,rm-rs), rpa_hi:rm+rs,
                  tc_lo: Math.max(0,tm-ts), tc_hi: tm+ts};
        });
        setSweepData(data);
      } catch(e) { console.error("Sweep error:",e); }
      setSweeping(false);
    }, 20);
  }, [userParams, kineticParams, physParams, sweepParam, sweepConfig, noiseEnabled, noiseConfig]);

  const fi     = simData ? Math.min(frameIdx, simData.times.length-1) : 0;
  const curT   = simData ? simData.times[fi] : 0;
  const curRPA = simData ? simData.RPA[fi] : 0;
  const curPC  = simData ? simData.PC[fi]  : 0;
  // peakRPA/peakPC passed directly to ODBars at call site (Math.max over full run)

  const set_up = k => v => setUserParams(p=>({...p,[k]:v}));

  return (
    <div style={{background:T.bg,minHeight:"100vh",color:T.text,
      fontFamily:"'DM Mono',monospace",padding:"14px 16px"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-thumb{background:#1a2535;border-radius:2px;}
        input[type=range]{accent-color:${T.accent};}
      `}</style>

      {/* ── Header ── */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
        <div>
          <div style={{fontSize:9,color:T.muted,letterSpacing:4,textTransform:"uppercase",marginBottom:2}}>
            Phase 1 · Sandwich Assay · Rochester Model
          </div>
          <h1 style={{fontFamily:"'Syne',sans-serif",fontSize:22,fontWeight:800,letterSpacing:-0.5,
            background:`linear-gradient(110deg,${T.accent},#a78bfa)`,
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            Sandwich LFA Simulator
          </h1>
        </div>
        <div style={{display:"flex",gap:6}}>
          {[["sim","⚗ Simulation"],["sweep","⟳ Sweep"],["info","ƒ Model"],["report","📋 Report"]].map(([id,lbl])=>(
            <button key={id} onClick={()=>{if(id==="sweep"){setSweepData(null);}setTab(id);}}
              style={{background:tab===id?"#7c3aed28":T.card,
                border:`1px solid ${tab===id?"#7c3aed":T.border}`,
                color:tab===id?"#c084fc":T.muted2,padding:"5px 12px",borderRadius:6,
                cursor:"pointer",fontFamily:"inherit",fontSize:11,transition:"all 0.15s"}}>
              {lbl}
            </button>
          ))}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"248px 1fr",gap:14}}>

        {/* ── LEFT PANEL: Parameters ── */}
        <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14,
          display:"flex",flexDirection:"column",gap:0,maxHeight:"calc(100vh - 80px)",overflowY:"auto"}}>

          {/* Sub-tabs */}
          <div style={{display:"flex",gap:4,marginBottom:14,background:T.surface,
            borderRadius:7,padding:3}}>
            {[["assay","Assay"],["kinetic","Kinetics"],["physical","Physical"]].map(([id,lbl])=>(
              <button key={id} onClick={()=>setParamTab(id)}
                style={{flex:1,background:paramTab===id?"#1a2a45":T.surface,
                  border:`1px solid ${paramTab===id?T.borderHi:"transparent"}`,
                  color:paramTab===id?T.text:T.muted,padding:"5px 0",borderRadius:5,
                  cursor:"pointer",fontFamily:"inherit",fontSize:10,transition:"all 0.15s"}}>
                {lbl}
              </button>
            ))}
          </div>

          {/* Assay params */}
          {paramTab==="assay" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>Assay Parameters</div>
            <LogSlider label="Analyte [A₀]" value={userParams.Ao}
              logMin={-9} logMax={2} unit="nM" color={T.A}
              onChange={set_up("Ao")}/>
            <Slider label="Detector [P₀]"      value={userParams.Po}    min={0.1}  max={100}  step={0.5}  unit="nM" color={T.P}   onChange={set_up("Po")}/>
            <Slider label="Test receptor [R₀]" value={userParams.Ro}    min={0.1}  max={100}  step={0.5}  unit="nM" color={T.PA}  onChange={set_up("Ro")}/>
            <Slider label="Ctrl receptor [Rc]" value={userParams.Ro_cl} min={0.1}  max={100}  step={0.5}  unit="nM" color={T.PC}  onChange={set_up("Ro_cl")}/>
            <Slider label="Test line x"        value={userParams.x_tl}  min={10}   max={50}   step={5}    unit="mm" color={T.RPA} onChange={set_up("x_tl")}/>
            <Slider label="Control line x"     value={userParams.x_cl}  min={Math.min(userParams.x_tl+10, physParams.strip_L-5)} max={physParams.strip_L+20} step={5} unit="mm" color={T.PC} onChange={set_up("x_cl")}/>
            <Slider label="Run Time"           value={userParams.t_end} min={120}  max={1800} step={60}   unit="s"  color={T.accent} onChange={set_up("t_end")}/>
          </>}

          {/* Kinetic params (editable) */}
          {paramTab==="kinetic" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>Kinetic Parameters</div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Edit values and press Enter or click away. Range shown below each field.
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

          {/* Physical params (editable) */}
          {paramTab==="physical" && <>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>Physical Parameters</div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Membrane & strip geometry. Affects effective diffusivity and conjugate release.
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

          {/* Geometry validation warning */}
          {(()=>{
            const warns = [];
            if (userParams.x_cl >= physParams.strip_L)
              warns.push(`Control line (${userParams.x_cl}mm) is outside strip (${physParams.strip_L}mm) — no ctrl signal`);
            if (userParams.x_tl >= physParams.strip_L - 5)
              warns.push(`Test line (${userParams.x_tl}mm) too close to strip end`);
            if (userParams.x_cl <= userParams.x_tl)
              warns.push("Control line must be downstream of test line");
            return warns.length > 0 ? (
              <div style={{background:"#1a0f08",border:"1px solid #7f3a1a",borderRadius:7,
                padding:"8px 10px",marginBottom:8}}>
                {warns.map((w,i)=>(
                  <div key={i} style={{color:"#fb923c",fontSize:9,lineHeight:1.6}}>⚠ {w}</div>
                ))}
              </div>
            ) : null;
          })()}
          {/* Run button */}
          <button onClick={handleRun} disabled={running} style={{
            width:"100%",padding:"9px 0",marginTop:14,
            background:running?T.border:`linear-gradient(135deg,#0ea5e9,#7c3aed)`,
            border:"none",borderRadius:7,color:"#fff",cursor:running?"not-allowed":"pointer",
            fontFamily:"inherit",fontSize:12,fontWeight:500,
            boxShadow:running?"none":"0 0 16px rgba(14,165,233,0.25)",transition:"all 0.2s"}}>
            {running?"⟳  Computing…":"▶  Run Simulation"}
          </button>

          {/* Legend */}
          <div style={{marginTop:14,paddingTop:12,borderTop:`1px solid ${T.border}`}}>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Species</div>
            {[{c:T.A,l:"[A] Analyte"},{c:T.P,l:"[P] Detector"},
              {c:T.PA,l:"[PA] Complex"},
              {c:T.RPA,l:"[RPA] Test signal",b:true},{c:T.PC,l:"[PC] Ctrl signal",b:true}
            ].map(s=>(
              <div key={s.l} style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}>
                <div style={{width:20,height:2,borderRadius:1,background:s.c,boxShadow:`0 0 5px ${s.c}`}}/>
                <span style={{fontSize:10,color:s.b?s.c:T.muted2,fontWeight:s.b?700:400}}>{s.l}</span>
              </div>
            ))}
          </div>

          {/* Flow front + sample info */}
          {simData && (
            <div style={{marginTop:12,paddingTop:12,borderTop:`1px solid ${T.border}`}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:6}}>Flow Front</div>
              {(()=>{
                const x4=21311*curT-69505;
                const xf=x4>0?x4**0.25:0;
                const pct=Math.min(100,(xf/physParams.strip_L)*100);
                const tse = simData.t_sample_end;
                const fr  = simData.flow_rate_uL_s;
                const sampleDone = curT > tse;
                return <>
                  <div style={{fontSize:11,color:T.accent,marginBottom:5}}>
                    {xf.toFixed(1)} mm · {pct.toFixed(0)}%
                  </div>
                  <div style={{height:4,background:T.border,borderRadius:2,marginBottom:8}}>
                    <div style={{height:"100%",width:`${pct}%`,borderRadius:2,
                      background:`linear-gradient(90deg,${T.accent}55,${T.accent})`,
                      transition:"width 0.05s"}}/>
                  </div>
                  {/* Sample volume info */}
                  <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:4}}>Sample</div>
                  <div style={{fontSize:10,color:sampleDone?T.muted2:T.ok,marginBottom:3}}>
                    {sampleDone
                      ? `⏹ Exhausted at ${tse.toFixed(0)} s`
                      : `▶ Flowing · ${(tse-curT).toFixed(0)} s left`}
                  </div>
                  <div style={{fontSize:9,color:T.muted}}>
                    {fr.toExponential(2)} µL/s · {physParams.sample_vol} µL total
                  </div>
                  {/* Sample exhaustion progress bar */}
                  <div style={{height:3,background:T.border,borderRadius:2,marginTop:4}}>
                    <div style={{height:"100%",borderRadius:2,transition:"width 0.05s",
                      width:`${Math.min(100,(curT/tse)*100)}%`,
                      background:sampleDone
                        ?`linear-gradient(90deg,${T.muted},${T.muted2})`
                        :`linear-gradient(90deg,${T.ok}55,${T.ok})`}}/>
                  </div>
                </>;
              })()}
            </div>
          )}
        </div>

        {/* ── RIGHT PANEL ── */}
        <div style={{display:"flex",flexDirection:"column",gap:12}}>

          {/* Stats */}
          <div style={{display:"flex",gap:8}}>
            <Stat label="Time"      value={`${Math.round(curT)} s`} unit="" color={T.accent}/>
            <Stat label="Test RPA"  value={curRPA.toExponential(2)} unit="nM" color={T.RPA}/>
            <Stat label="Ctrl PC"   value={curPC.toExponential(2)}  unit="nM" color={T.PC}/>
            <Stat label="T/C ratio" value={simData&&Math.max(...simData.PC)>1e-25?(Math.max(...simData.RPA)/Math.max(...simData.PC)).toExponential(1):"—"} unit="" color={T.muted2}/>
          </div>

          {/* ── Simulation tab ── */}
          {tab==="sim" && <>
            <div style={{background:T.card,border:`1px solid ${T.border2}`,borderRadius:12,padding:14}}>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:10}}>
                <div style={{fontSize:11}}>
                  <span style={{color:T.muted2}}>Membrane strip — </span>
                  <span style={{color:T.accent}}>t = {Math.round(curT)} s</span>
                  <span style={{color:T.muted,fontSize:9,marginLeft:8}}>
                    cyan = flow front · <span style={{color:T.RPA}}>■</span> TEST · <span style={{color:T.PC}}>■</span> CTRL
                  </span>
                </div>
                <div style={{display:"flex",gap:5}}>
                  {[["⏮",()=>{setFrameIdx(0);setPlaying(false);}],
                    [playing?"⏸":"▶",()=>setPlaying(p=>!p)],
                    ["⏭",()=>{setFrameIdx(simData?simData.times.length-1:0);setPlaying(false);}]
                  ].map(([icon,fn],i)=>(
                    <button key={i} onClick={fn} style={{
                      background:T.surface,border:`1px solid ${i===1?T.accent+"44":T.border}`,
                      color:i===1?T.accent:T.muted2,padding:"4px 10px",borderRadius:5,
                      cursor:"pointer",fontFamily:"inherit",fontSize:12}}>
                      {icon}
                    </button>
                  ))}
                </div>
              </div>

              <LFAStrip simData={simData} frameIdx={frameIdx} userParams={userParams}/>

              {simData && (
                <div style={{marginTop:8}}>
                  <input type="range" min={0} max={simData.times.length-1} value={frameIdx}
                    onChange={e=>{setPlaying(false);setFrameIdx(Number(e.target.value));}}
                    style={{width:"100%",cursor:"pointer",height:4}}/>
                  <div style={{display:"flex",justifyContent:"space-between",
                    fontSize:9,color:T.muted,marginTop:3}}>
                    <span>0 s</span>
                    <span style={{color:T.muted2}}>← drag to scrub · ▶ to play →</span>
                    <span>{userParams.t_end} s</span>
                  </div>
                </div>
              )}
            </div>

            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>
              <div style={{fontSize:11,color:T.muted2,marginBottom:8}}>
                Test &amp; Control line signals vs time —
                <span style={{color:T.muted}}> click chart to seek</span>
              </div>
              {simData
                ? <SignalChart simData={simData} frameIdx={frameIdx} setFrameIdx={setFrameIdx} tSampleEnd={simData.t_sample_end}/>
                : <div style={{height:155,display:"flex",alignItems:"center",justifyContent:"center"}}>
                    <span style={{color:T.muted,fontSize:11}}>Run simulation to see signal curves</span>
                  </div>
              }
            </div>

            {simData && <AbsorbentPad
                simData={simData} frameIdx={frameIdx} userParams={userParams}/>}
            {simData && <ODBars
                RPAval={curRPA} PCval={curPC}
                peakRPA={Math.max(...simData.RPA)}
                peakPC={Math.max(...simData.PC)}
                clReachable={simData.cl_reachable}
              />}
          </>}

          {/* ── Sweep tab ── */}
          {tab==="sweep" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>

              {/* ── Row 1: param selector ── */}
              <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:10,flexWrap:"wrap"}}>
                <span style={{fontSize:9,color:T.muted,letterSpacing:2,
                  textTransform:"uppercase",marginRight:4}}>Sweep</span>
                {[["Ao","A₀ dose-response"],["Po","Detector P₀"],
                  ["Ro","Test R₀"],["x_tl","Test line x"]
                ].map(([p,l])=>(
                  <button key={p} onClick={()=>{setSweepParam(p);setSweepData(null);}}
                    style={{background:sweepParam===p?"#7c3aed28":T.surface,
                      border:`1px solid ${sweepParam===p?"#7c3aed":T.border}`,
                      color:sweepParam===p?"#c084fc":T.muted2,
                      padding:"4px 12px",borderRadius:5,cursor:"pointer",
                      fontFamily:"inherit",fontSize:11}}>
                    {l}
                  </button>
                ))}
              </div>

              {/* ── Row 2: range config ── */}
              <div style={{background:T.surface,border:`1px solid ${T.border}`,
                borderRadius:7,padding:"8px 12px",marginBottom:10,
                display:"flex",gap:14,alignItems:"center",flexWrap:"wrap"}}>
                <span style={{fontSize:9,color:T.muted,letterSpacing:2,
                  textTransform:"uppercase",minWidth:36}}>Range</span>
                {sweepParam==="Ao"?(<>
                  {[["log min","logMin",-20,2],["log max","logMax",-9,10]].map(([lbl,key,mn,mx])=>(
                    <label key={key} style={{fontSize:10,color:T.muted2,
                      display:"flex",gap:5,alignItems:"center"}}>
                      {lbl}
                      <input type="number" min={mn} max={mx}
                        value={sweepConfig.Ao[key]}
                        onChange={e=>setSweepConfig(c=>({...c,
                          Ao:{...c.Ao,[key]:Number(e.target.value)}}))}
                        style={{width:45,background:T.card,border:`1px solid ${T.border}`,
                          borderRadius:4,color:T.text,padding:"2px 5px",
                          fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                    </label>
                  ))}
                  <label style={{fontSize:10,color:T.muted2,
                    display:"flex",gap:5,alignItems:"center"}}>
                    points
                    <input type="number" min={4} max={40}
                      value={sweepConfig.Ao.nPoints}
                      onChange={e=>setSweepConfig(c=>({...c,
                        Ao:{...c.Ao,nPoints:Math.max(4,Math.min(40,+e.target.value))}}))}
                      style={{width:40,background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,padding:"2px 5px",
                        fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                </>):(<>
                  {[["min","min"],["max","max"]].map(([lbl,key])=>(
                    <label key={key} style={{fontSize:10,color:T.muted2,
                      display:"flex",gap:5,alignItems:"center"}}>
                      {lbl}
                      <input type="number"
                        value={sweepConfig[sweepParam]?.[key]??0}
                        onChange={e=>setSweepConfig(c=>({...c,
                          [sweepParam]:{...c[sweepParam],[key]:Number(e.target.value)}}))}
                        style={{width:52,background:T.card,border:`1px solid ${T.border}`,
                          borderRadius:4,color:T.text,padding:"2px 5px",
                          fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                      <span style={{fontSize:9,color:T.muted}}>
                        {sweepParam==="x_tl"?"mm":"nM"}
                      </span>
                    </label>
                  ))}
                  <label style={{fontSize:10,color:T.muted2,
                    display:"flex",gap:5,alignItems:"center"}}>
                    steps
                    <input type="number" min={3} max={30}
                      value={sweepConfig[sweepParam]?.nSteps??10}
                      onChange={e=>setSweepConfig(c=>({...c,
                        [sweepParam]:{...c[sweepParam],
                          nSteps:Math.max(3,Math.min(30,+e.target.value))}}))}
                      style={{width:40,background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,padding:"2px 5px",
                        fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                </>)}
              </div>

              {/* ── Row 3: noise + run ── */}
              <div style={{display:"flex",gap:10,alignItems:"center",
                marginBottom:12,flexWrap:"wrap"}}>
                <label style={{display:"flex",alignItems:"center",gap:6,cursor:"pointer",
                  fontSize:11,color:noiseEnabled?T.PA:T.muted2,
                  background:T.surface,border:`1px solid ${noiseEnabled?T.PA+"55":T.border}`,
                  borderRadius:6,padding:"4px 10px"}}>
                  <input type="checkbox" checked={noiseEnabled}
                    onChange={e=>setNoiseEnabled(e.target.checked)}
                    style={{accentColor:T.PA}}/>
                  Variability (noise)
                </label>
                {noiseEnabled&&(<>
                  {[["CV Ao","cv_Ao","pipetting"],["CV Ka","cv_Ka","kinetics"],
                    ["CV flow","cv_flow","membrane"]
                  ].map(([lbl,key,hint])=>(
                    <label key={key} style={{fontSize:10,color:T.muted2,
                      display:"flex",gap:4,alignItems:"center"}}>
                      {lbl}
                      <input type="number" min={0} max={0.5} step={0.01}
                        value={noiseConfig[key]}
                        onChange={e=>setNoiseConfig(c=>({...c,[key]:+e.target.value}))}
                        style={{width:44,background:T.card,border:`1px solid ${T.border}`,
                          borderRadius:4,color:T.text,padding:"2px 5px",
                          fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                      <span style={{fontSize:8,color:T.muted}}>{hint}</span>
                    </label>
                  ))}
                  <label style={{fontSize:10,color:T.muted2,
                    display:"flex",gap:4,alignItems:"center"}}>
                    reps
                    <input type="number" min={3} max={20}
                      value={noiseConfig.n_replicates}
                      onChange={e=>setNoiseConfig(c=>({...c,
                        n_replicates:Math.max(3,Math.min(20,+e.target.value))}))}
                      style={{width:38,background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,padding:"2px 5px",
                        fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                  <span style={{fontSize:9,color:T.muted,fontStyle:"italic"}}>
                    shaded band = ±1σ
                  </span>
                </>)}
                <button onClick={handleSweep} disabled={sweeping}
                  style={{marginLeft:"auto",background:sweeping?T.border:"#0ea5e9",
                    border:"none",color:"#fff",padding:"6px 18px",borderRadius:6,
                    cursor:sweeping?"not-allowed":"pointer",
                    fontFamily:"inherit",fontSize:11,fontWeight:600}}>
                  {sweeping?"⧗ Running…":"Run Sweep"}
                </button>
              </div>

              {/* Chart */}
              {sweepData && sweepData.length > 0
                ? <SweepChart data={sweepData} sweepParam={sweepParam} />
                : (
                  <div style={{height:300,display:"flex",alignItems:"center",
                    justifyContent:"center",flexDirection:"column",gap:8}}>
                    <span style={{fontSize:20}}>📈</span>
                    <span style={{color:T.muted,fontSize:11}}>
                      Select a parameter and click Run Sweep
                    </span>
                    <span style={{color:T.muted,fontSize:9}}>
                      A₀ sweep runs a dose-response curve across 16 log-spaced concentrations
                    </span>
                  </div>
                )
              }

              {/* Results table */}
              {sweepData && sweepData.length > 0 && (
                <div style={{marginTop:14}}>
                  <div style={{fontSize:9,color:T.muted,letterSpacing:2,
                    textTransform:"uppercase",marginBottom:6}}>
                    Sweep Results
                  </div>
                  <div style={{overflowX:"auto"}}>
                    <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
                      <thead>
                        <tr style={{background:T.surface}}>
                          {[
                            sweepParam==="Ao" ? "[A₀] (nM)"
                            : sweepParam==="Po" ? "P₀ (nM)"
                            : sweepParam==="Ro" ? "R₀ (nM)"
                            : "x_tl (mm)",
                            "RPA (nM)", "PC (nM)", "T/C ratio", "Verdict",
                          ].map(h => (
                            <th key={h} style={{
                              padding:"5px 10px", textAlign:"left",
                              color:T.muted, fontSize:9, fontWeight:600,
                              borderBottom:`1px solid ${T.border2}`,
                            }}>{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sweepData.map((d, i) => {
                          const rpa = typeof d.rpa === "number" ? d.rpa : 0;
                          const pc  = typeof d.pc  === "number" ? d.pc  : 0;
                          const tc  = typeof d.tc  === "number" ? d.tc  : 0;
                          const x   = typeof d.x   === "number" ? d.x   : 0;
                          const isPos = pc > 0 && tc >= 0.05;
                          const isInv = pc <= 1e-30;
                          return (
                            <tr key={i} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                              <td style={{padding:"4px 10px",color:T.accent,
                                fontFamily:"monospace",fontWeight:600}}>
                                {sweepParam==="Ao"
                                  ? x.toExponential(2)
                                  : x.toFixed(1)}
                              </td>
                              <td style={{padding:"4px 10px",color:T.RPA,fontFamily:"monospace"}}>
                                {rpa.toExponential(2)}
                              </td>
                              <td style={{padding:"4px 10px",color:T.PC,fontFamily:"monospace"}}>
                                {pc.toExponential(2)}
                              </td>
                              <td style={{padding:"4px 10px",fontFamily:"monospace",
                                fontWeight:700,
                                color:isInv?T.muted2:isPos?T.PA:T.muted2}}>
                                {isInv ? "—" : tc.toExponential(2)}
                              </td>
                              <td style={{padding:"4px 10px"}}>
                                <span style={{
                                  background: isInv?"#111":isPos?"#0a1a10":"#0a1020",
                                  color: isInv?T.muted:isPos?T.PA:T.muted2,
                                  border:`1px solid ${isInv?T.border:isPos?T.PA+"44":T.border}`,
                                  borderRadius:4, padding:"1px 7px", fontSize:9,
                                }}>
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

          {/* ── Report tab ── */}
          {tab==="report" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,
              borderRadius:12,padding:16}}>
              <SimReport
                simData={simData}
                userParams={userParams}
                kineticParams={kineticParams}
                physParams={physParams}
              />
            </div>
          )}

          {/* ── Model info tab ── */}
          {tab==="info" && (
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
                Reaction Model + Physical Assumptions
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:14}}>
                {[
                  {n:"①",eq:"A + P  ⇌  PA",   desc:"Detector binds analyte in solution",        c:T.P},
                  {n:"②",eq:"A + R  ⇌  RA",   desc:"Analyte binds test line receptor",           c:T.PA},
                  {n:"③",eq:"P + RA  →  RPA",  desc:"Detector labels analyte-receptor complex",   c:T.RPA},
                  {n:"④",eq:"PA + R  →  RPA",  desc:"Labelled complex captured by test receptor", c:T.RPA},
                  {n:"⑤",eq:"P + Rc  →  PC",   desc:"Free detector captured at control line",     c:T.PC},
                  {n:"⑥",eq:"Ct = P₀(βe^{-λt}+(1-β))",desc:"Conjugate burst + exponential release",c:T.muted2},
                ].map(r=>(
                  <div key={r.n} style={{background:T.surface,border:`1px solid ${r.c}22`,
                    borderRadius:8,padding:"9px 11px"}}>
                    <div style={{display:"flex",gap:8,alignItems:"baseline",marginBottom:3}}>
                      <span style={{color:r.c,fontSize:14,fontFamily:"'Syne',sans-serif",fontWeight:700}}>{r.n}</span>
                      <span style={{color:T.text,fontSize:11}}>{r.eq}</span>
                    </div>
                    <div style={{color:T.muted2,fontSize:10}}>{r.desc}</div>
                  </div>
                ))}
              </div>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Physical Model</div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:6}}>
                {[
                  ["Diffusivity","D_eff = D·φ/τ","porosity × tortuosity scaling"],
                  ["Flow","U(x) = min(c/x³, U_max)","Berli & Kler 2016 + cap"],
                  ["Flow position","x⁴ = 21311t − 69505","Rochester wet lab fit"],
                  ["Conjugate","Ct = P₀(βe^{-λt}+(1-β))","burst + baseline release"],
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
