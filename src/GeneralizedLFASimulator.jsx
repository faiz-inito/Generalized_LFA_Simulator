import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════════════════
const T = {
  bg:"#060a10", surface:"#0b0f18", card:"#0f1520",
  border:"#1a2535", border2:"#202e45", borderHi:"#2a3f60",
  A:"#38bdf8", P:"#c084fc", PA:"#34d399",
  RPA:"#fb923c", PC:"#f43f5e",
  accent:"#00d4ff", text:"#dde4f0",
  muted:"#4a5568", muted2:"#6b7a8d",
  err:"#f87171", ok:"#4ade80", warn:"#f59e0b",
};
const LINE_COLORS = {
  sandwich:"#fb923c", competitive:"#f59e0b",
  reference:"#38bdf8", control:"#f43f5e",
};

// ═══════════════════════════════════════════════════════════════════════════
// GAUSSIAN SMOOTHER  (display only)
// ═══════════════════════════════════════════════════════════════════════════
function gaussSmooth(arr, sigma = 1.2) {
  const n = arr.length, out = new Float64Array(n);
  const r = Math.ceil(3*sigma), kernel = []; let ksum = 0;
  for (let k=-r;k<=r;k++){const w=Math.exp(-(k*k)/(2*sigma*sigma));kernel.push(w);ksum+=w;}
  for (let i=0;i<n;i++){let v=0;for(let k=-r;k<=r;k++){const j=Math.max(0,Math.min(n-1,i+k));v+=arr[j]*kernel[k+r];}out[i]=v/ksum;}
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// COUPLED MULTI-LINE SOLVER
//
// Architecture:
//   • One shared PDE field (A, P, PA) per CONJUGATE GROUP
//   • Lines are sinks distributed along their group's shared field
//   • Control line taps EVERY conjugate group simultaneously
//     (anti-species Ab captures any gold-labelled conjugate)
//   • Lines with the same conjId compete for the same [P] pool
//   • Lines with different conjIds are independent
//
// Mass conservation checks:
//   • Receptor: RA + RPA + Rfree = Ro  (exact — immobilised)
//   • Fixed Ag:  AgP + AgFree = fixedAg (exact — immobilised)
//   • Mobile (A, P, PA): open system — mass exits via absorbent pad
//     Conservation = Receptor conservation + pad flux accounting
// ═══════════════════════════════════════════════════════════════════════════

function runSimulation({ analytes, conjugates, lines, pp }) {
  const {
    strip_L, dx, phi, tau,
    line_width_mm, sample_vol, U_max,
    t_end, DA, DP, flow_c,
  } = pp;

  const N      = Math.round(strip_L / dx);
  const i2     = 1 / (dx * dx);
  const dt     = Math.min(t_end / 1500, 0.2);
  const DA_eff = DA * phi / tau;
  const DP_eff = DP * phi / tau;
  const lw     = Math.max(1, Math.round(line_width_mm / dx));

  // Sample timing
  const cross_mm2      = 4.0 * 0.135;
  const flow_rate_uL_s = U_max * cross_mm2 * 1e-3;
  const t_sample_end   = flow_rate_uL_s > 0 ? sample_vol / flow_rate_uL_s : t_end;

  // x-grid and flow velocity  U(x) = min(c/x³, U_max)
  const xGrid = Array.from({length:N}, (_,i) => (i+1)*dx);
  const U     = xGrid.map(xi => Math.min(flow_c / (xi**3), U_max));

  // Conjugate pad = first 20% of nodes
  const conj_end = Math.round(0.20 * N);

  // ── Identify control lines ───────────────────────────────────────────
  const ctrlLines = lines.filter(ln => ln.type === 'control');
  const nonCtrlLines = lines.filter(ln => ln.type !== 'control');

  // ── Build one PDE group per unique conjugate ─────────────────────────
  // Each group owns: A[], P[], PA[] fields shared by all its lines
  const conjGroups = conjugates.map(cj => {
    const analyte = analytes.find(a => a.id === cj.analyteId) || analytes[0];
    const Ao  = parseFloat(analyte?.Ao) || 0;
    const Po  = parseFloat(cj.Po)  || 6;
    const Ka1 = parseFloat(cj.Ka1) || 7.35e-4;
    const Kd1 = parseFloat(cj.Kd1) || 5.7e-5;

    // Non-control lines linked to this conjugate
    const groupLines = nonCtrlLines
      .filter(ln => ln.conjId === cj.id)
      .map(ln => ({
        ln,
        pos:     Math.min(Math.round(parseFloat(ln.pos)/dx), N-2),
        Ka2:     parseFloat(ln.Ka2) || 7.35e-4,
        Kd2:     parseFloat(ln.Kd2) || 5.7e-5,
        Ro:      parseFloat(ln.Ro)  || 0,
        fixedAg: parseFloat(ln.fixedAg) || 0,
        // immobilised state
        RA: 0, RPA: 0, Rfree: parseFloat(ln.Ro) || 0,
        AgFree: parseFloat(ln.fixedAg) || 0, AgP: 0,
        // time series
        sig_ts: [],
      }));

    return {
      cj, analyte, Ao, Po, Ka1, Kd1,
      // shared mobile fields
      A:  new Float64Array(N),
      P:  new Float64Array(N),
      PA: new Float64Array(N),
      // pad flux
      pad_A: 0, pad_P: 0, pad_PA: 0,
      // snapshots
      A_snap: [], P_snap: [], PA_snap: [],
      pad_A_ts: [], pad_P_ts: [], pad_PA_ts: [],
      groupLines,
    };
  });

  // ── Control line state — one per control line ────────────────────────
  // PC = sum of contributions from ALL conjugate groups
  const ctrlStates = ctrlLines.map(ln => ({
    ln,
    pos:    Math.min(Math.round(parseFloat(ln.pos)/dx), N-2),
    Ro:     parseFloat(ln.Ro) || 10,
    Rfree:  parseFloat(ln.Ro) || 10,
    // per-conjugate contribution tracking
    contributions: conjugates.map(cj => ({ conjId: cj.id, PC: 0, Ka_cl: 5e-4, Kd_cl: 4e-5 })),
    get PC() { return this.contributions.reduce((s,c)=>s+c.PC,0); },
    sig_ts: [],
  }));

  // ── Snapshot / timing ────────────────────────────────────────────────
  const nOutput    = 160;
  const outputEvery= Math.max(1, Math.round((t_end/dt)/nOutput));
  const maxSteps   = Math.round(t_end / dt);
  const sharedTimes= [];

  // ── Time loop ────────────────────────────────────────────────────────
  for (let step = 0; step <= maxSteps; step++) {
    const t_now = step * dt;
    const sampleFlowing = t_now <= t_sample_end;

    // ── Snapshot ──────────────────────────────────────────────────────
    if (step % outputEvery === 0) {
      sharedTimes.push(t_now);
      conjGroups.forEach(g => {
        g.A_snap.push(Array.from(g.A));
        g.P_snap.push(Array.from(g.P));
        g.PA_snap.push(Array.from(g.PA));
        g.pad_A_ts.push(g.pad_A);
        g.pad_P_ts.push(g.pad_P);
        g.pad_PA_ts.push(g.pad_PA);
        g.groupLines.forEach(gl => {
          const isComp = gl.ln.type==='competitive'||gl.ln.type==='reference';
          gl.sig_ts.push(isComp ? gl.AgP : gl.RPA);
        });
      });
      ctrlStates.forEach(cs => cs.sig_ts.push(cs.PC));
    }

    // ── Pre-load conjugate pad once ───────────────────────────────────
    if (step === 0) {
      conjGroups.forEach(g => {
        for (let j=0; j<conj_end; j++) g.P[j] = g.Po;
      });
    }

    // ── PDE step per conjugate group ──────────────────────────────────
    conjGroups.forEach(g => {
      const {A, P, PA, Ka1, Kd1, Ao} = g;

      // Inlet BCs
      A[0]  = sampleFlowing ? Ao : 0;
      PA[0] = 0;

      // Bulk reaction A + P ⇌ PA
      const F_PA = new Float64Array(N);
      for (let j=0; j<N; j++)
        F_PA[j] = Ka1*A[j]*P[j] - Kd1*PA[j];

      const dA  = new Float64Array(N);
      const dP  = new Float64Array(N);
      const dPA = new Float64Array(N);

      for (let j=0; j<N; j++) {
        const Uj  = U[j];
        const Al  = j===0   ? A[0]   : A[j-1];
        const Ar  = j===N-1 ? A[N-1] : A[j+1];
        const Pl  = j===0   ? P[0]   : P[j-1];
        const Pr  = j===N-1 ? P[N-1] : P[j+1];
        const PAl = j===0   ? PA[0]  : PA[j-1];
        const PAr = j===N-1 ? PA[N-1]: PA[j+1];
        dA[j]  = DA_eff*(Ar -2*A[j] +Al )*i2 - Uj*(A[j] -Al )/dx - F_PA[j];
        dP[j]  = DP_eff*(Pr -2*P[j] +Pl )*i2 - Uj*(P[j] -Pl )/dx - F_PA[j];
        dPA[j] = DP_eff*(PAr-2*PA[j]+PAl)*i2 - Uj*(PA[j]-PAl)/dx + F_PA[j];
      }

      // ── Line sinks — all lines in this group compete for same P ──────
      g.groupLines.forEach(gl => {
        const {pos, Ka2, Kd2} = gl;
        const isComp = gl.ln.type==='competitive'||gl.ln.type==='reference';
        const zS = Math.max(0, pos-lw), zE = Math.min(N-1, pos+lw);
        let Az=0, Pz=0, PAz=0, nz=0;
        for (let j=zS;j<=zE;j++){Az+=A[j];Pz+=P[j];PAz+=PA[j];nz++;}
        Az/=nz; Pz/=nz; PAz/=nz;

        let f_RA=0, f_RPA1=0, f_RPA2=0, f_AgP=0;
        if (!isComp) {
          f_RA   = Ka2*Az *gl.Rfree - Kd2*gl.RA;
          f_RPA1 = Ka1*Pz *gl.RA    - Kd1*gl.RPA;
          f_RPA2 = Ka2*PAz*gl.Rfree - Kd2*gl.RPA;
        } else {
          f_AgP  = Ka2*gl.AgFree*Pz - Kd2*gl.AgP;
        }

        const zn = zE-zS+1;
        for (let j=zS;j<=zE;j++) {
          if (!isComp) {
            dA[j]  -= f_RA   /zn;
            dP[j]  -= (f_RPA1+f_RPA2)/zn;
            dPA[j] -= f_RPA2 /zn;
          } else {
            dP[j]  -= f_AgP/zn;
          }
        }

        // Update immobilised
        if (!isComp) {
          gl.RA    = Math.max(0, gl.RA  + dt*(f_RA-f_RPA1));
          gl.RPA   = Math.max(0, gl.RPA + dt*(f_RPA1+f_RPA2));
          gl.Rfree = Math.max(0, gl.Ro  - gl.RA - gl.RPA);
        } else {
          gl.AgP    = Math.max(0, gl.AgP + dt*f_AgP);
          gl.AgFree = Math.max(0, gl.fixedAg - gl.AgP);
        }
      });

      // ── Control line sinks — tap this group's P ───────────────────────
      ctrlStates.forEach(cs => {
        const contrib = cs.contributions.find(c=>c.conjId===g.cj.id);
        if (!contrib) return;
        const zS=Math.max(0,cs.pos-lw), zE=Math.min(N-1,cs.pos+lw);
        let Pz=0, nz=0;
        for (let j=zS;j<=zE;j++){Pz+=P[j];nz++;}
        Pz/=nz;
        // Each group's Rfree needs to be tracked per-conjugate
        // Use shared Rfree scaled by number of conjugate groups
        const f_Pc = contrib.Ka_cl * cs.Rfree / conjugates.length * Pz - contrib.Kd_cl * contrib.PC;
        const zn=zE-zS+1;
        for (let j=zS;j<=zE;j++) dP[j] -= f_Pc/zn;
        contrib.PC = Math.max(0, contrib.PC + dt*f_Pc);
      });

      // Euler step
      for (let j=0;j<N;j++){
        A[j]  = Math.max(0, A[j]  + dt*dA[j]);
        P[j]  = Math.max(0, P[j]  + dt*dP[j]);
        PA[j] = Math.max(0, PA[j] + dt*dPA[j]);
      }

      // Update control Rfree (shared across all contributions)
      ctrlStates.forEach(cs => {
        const totalPC = cs.PC;
        cs.Rfree = Math.max(0, cs.Ro - totalPC);
      });

      // Absorbent pad flux
      g.pad_A  += U[N-1]*A[N-1] *dt;
      g.pad_P  += U[N-1]*P[N-1] *dt;
      g.pad_PA += U[N-1]*PA[N-1]*dt;
      PA[0] = 0;
    });
  }

  // ── Build lineResults (flat, sorted by position) ─────────────────────
  const lineResults = [];

  conjGroups.forEach(g => {
    g.groupLines.forEach(gl => {
      const isComp = gl.ln.type==='competitive'||gl.ln.type==='reference';
      const finalSig = isComp ? gl.AgP : gl.RPA;

      // Mass conservation check for this line's receptor/Ag
      const recSum = gl.RA + gl.RPA + gl.Rfree;
      const agSum  = gl.AgP + gl.AgFree;
      const massOK = isComp
        ? Math.abs(agSum - gl.fixedAg) < 0.01
        : Math.abs(recSum - gl.Ro) < 0.01;

      // Conjugate depletion: how much P was left at this line vs original Po
      // Take last snapshot P value at line node
      const lastSnap = g.P_snap[g.P_snap.length-1];
      const P_at_line = lastSnap ? lastSnap[gl.pos] : 0;
      const depletionPct = g.Po > 0 ? (1 - P_at_line/g.Po)*100 : 0;

      lineResults.push({
        id:        gl.ln.id,
        name:      gl.ln.name,
        type:      gl.ln.type,
        pos:       gl.ln.pos,
        conjId:    g.cj.id,
        conjName:  g.cj.name,
        analyteName: g.analyte?.name || '?',
        signal_ts: gl.sig_ts,
        finalSignal: finalSig,
        RA: gl.RA, RPA: gl.RPA, AgP: gl.AgP,
        Ro: gl.Ro, fixedAg: gl.fixedAg,
        massOK,
        depletionPct: Math.min(100, Math.max(0, depletionPct)),
        A_snap:  g.A_snap,
        P_snap:  g.P_snap,
        PA_snap: g.PA_snap,
        xGrid,
      });
    });
  });

  // Control lines
  ctrlStates.forEach(cs => {
    const recSum = cs.PC + cs.Rfree;
    lineResults.push({
      id:        cs.ln.id,
      name:      cs.ln.name,
      type:      'control',
      pos:       cs.ln.pos,
      conjId:    null,
      conjName:  'ALL conjugates',
      analyteName:'—',
      signal_ts: cs.sig_ts,
      finalSignal: cs.PC,
      RA: 0, RPA: 0, AgP: 0, PC: cs.PC,
      Ro: cs.Ro,
      contributions: cs.contributions,
      massOK: Math.abs(recSum - cs.Ro) < 0.1,
      depletionPct: 0,
      // use first group's snapshots for profile display
      A_snap:  conjGroups[0]?.A_snap || [],
      P_snap:  conjGroups[0]?.P_snap || [],
      PA_snap: conjGroups[0]?.PA_snap|| [],
      xGrid,
    });
  });

  // Sort by position
  lineResults.sort((a,b) => parseFloat(a.pos)-parseFloat(b.pos));

  // ── Conjugate group summary (for depletion warnings) ─────────────────
  const conjSummary = conjGroups.map(g => {
    const lastP = g.P_snap[g.P_snap.length-1];
    const P_at_ctrl = ctrlStates[0]
      ? (lastP?.[ctrlStates[0].pos] || 0) : 0;
    return {
      id:   g.cj.id,
      name: g.cj.name,
      Po:   g.Po,
      P_at_ctrl,
      depletedPct: g.Po>0 ? Math.min(100,(1-P_at_ctrl/g.Po)*100) : 0,
      pad_P: g.pad_P,
      pad_A: g.pad_A,
      pad_PA: g.pad_PA,
    };
  });

  return {
    times: sharedTimes,
    t_sample_end,
    flow_rate_uL_s,
    lineResults,
    conjSummary,
    xGrid,
    N,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// PHYSICAL PARAMS  (includes DA, DP, flow_c — kinetics panel removed)
// ═══════════════════════════════════════════════════════════════════════════
const PHYSICAL_META = [
  { key:"strip_L",       label:"Strip length",        unit:"mm",   min:20,   max:200,  def:100,     sci:false, step:5    },
  { key:"dx",            label:"Grid spacing",        unit:"mm",   min:0.25, max:2,    def:0.5,     sci:false, step:0.25 },
  { key:"phi",           label:"Porosity φ",          unit:"—",    min:0.1,  max:0.95, def:0.7,     sci:false, step:0.05 },
  { key:"tau",           label:"Tortuosity τ",        unit:"—",    min:1.0,  max:5.0,  def:1.5,     sci:false, step:0.1  },
  { key:"line_width_mm", label:"Line half-width",     unit:"mm",   min:0.25, max:3,    def:0.5,     sci:false, step:0.25 },
  { key:"sample_vol",    label:"Sample volume",       unit:"µL",   min:5,    max:200,  def:75,      sci:false, step:5    },
  { key:"U_max",         label:"Max flow velocity",   unit:"mm/s", min:0.01, max:5,    def:0.8,     sci:false, step:0.05 },
  { key:"t_end",         label:"Run time",            unit:"s",    min:120,  max:1800, def:1200,    sci:false, step:60   },
  { key:"DA",            label:"Analyte diffusivity", unit:"mm²/s",min:1e-7, max:1e-1, def:1e-4,    sci:true              },
  { key:"DP",            label:"Conj. diffusivity",   unit:"mm²/s",min:1e-8, max:1e-2, def:1e-6,    sci:true              },
  { key:"flow_c",        label:"Flow constant c",     unit:"mm⁴/s",min:100,  max:5e4,  def:5327.75, sci:false             },
];
const defPhysical = () => Object.fromEntries(PHYSICAL_META.map(m => [m.key, m.def]));

// ═══════════════════════════════════════════════════════════════════════════
// PER-ITEM CONCENTRATION CAPS
// ═══════════════════════════════════════════════════════════════════════════
// Each analyte carries its own `aoMax` (linear nM upper bound for its A₀
// LogSlider — the slider's logMax is log10(aoMax)). Each conjugate carries
// its own `poMax` (linear nM upper bound for its P₀ Slider). Caps live
// on the items themselves so add/delete needs no separate sync — the cap
// travels with the item.
//
// Edited only in the Limits tab. Defaults (below) are seeded into new items
// at creation time. The Reset button restores every cap to these defaults.
// ═══════════════════════════════════════════════════════════════════════════
const DEFAULT_AO_MAX = 100;   // nM — analyte A₀ slider top decade = 1e2
const DEFAULT_PO_MAX = 100;   // nM — conjugate P₀ slider top
const AO_MAX_FLOOR   = 0;     // any positive value > floor is accepted
const PO_MAX_FLOOR   = 0.1;   // matches the slider's hard min

const PADS = [
  { id:"sample", label:"Sample Pad",    sub:"sample application",  frac:[0.00,0.09], bg:"#0e1f0e", bdr:"#2a6040", acc:"#4ade80" },
  { id:"conj",   label:"Conjugate Pad", sub:"labelled antibodies",  frac:[0.09,0.20], bg:"#130d22", bdr:"#5030a0", acc:"#c084fc" },
  { id:"nc",     label:"NC Membrane",   sub:"capillary flow zone",  frac:[0.20,0.83], bg:"#07101a", bdr:"#1a3560", acc:"#38bdf8" },
  { id:"abs",    label:"Absorbent Pad", sub:"wicking sink",         frac:[0.83,1.00], bg:"#091420", bdr:"#1a3a55", acc:"#60a5fa" },
];

function validateParam(meta, raw) {
  const v = parseFloat(raw);
  if (isNaN(v)) return { ok:false, msg:"Not a number" };
  if (v < meta.min) return { ok:false, msg:`Min: ${meta.min}` };
  if (v > meta.max) return { ok:false, msg:`Max: ${meta.max}` };
  return { ok:true, v };
}

let _uid = 0;
const uid = () => `id${++_uid}`;

// ═══════════════════════════════════════════════════════════════════════════
// UI ATOMS
// ═══════════════════════════════════════════════════════════════════════════
function LogSlider({ label, value, logMin, logMax, onChange, unit, color }) {
  const logVal = Math.log10(Math.max(value, Math.pow(10,logMin)));
  const pct    = ((logVal-logMin)/(logMax-logMin))*100;
  const decades= [];
  for (let e=Math.ceil(logMin);e<=Math.floor(logMax);e++) decades.push(e);
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
        <input type="range" min={logMin} max={logMax} step={(logMax-logMin)/1000} value={logVal}
          onChange={e=>onChange(Math.pow(10,Number(e.target.value)))}
          style={{position:"absolute",inset:0,opacity:0,width:"100%",cursor:"pointer",margin:0,height:20,top:-8}}/>
      </div>
      <div style={{position:"relative",height:14}}>
        {decades.map(e=>{
          const tp=((e-logMin)/(logMax-logMin))*100;
          return (
            <div key={e} style={{position:"absolute",left:`${tp}%`,transform:"translateX(-50%)",textAlign:"center"}}>
              <div style={{width:1,height:4,background:T.border,margin:"0 auto 1px"}}/>
              <span style={{fontSize:7,color:T.muted,fontFamily:"'DM Mono',monospace",whiteSpace:"nowrap"}}>
                {e>=0?`1e+${e}`:`1e${e}`}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, unit, color, fmt }) {
  const pct  = ((value-min)/(max-min))*100;
  const disp = fmt?fmt(value):(step<1?value.toFixed(2):Math.round(value));
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
          style={{position:"absolute",inset:0,opacity:0,width:"100%",cursor:"pointer",margin:0,height:20,top:-8}}/>
      </div>
    </div>
  );
}

function Stat({ label, value, unit, color }) {
  return (
    <div style={{background:T.card,border:`1px solid ${color}22`,borderRadius:8,padding:"8px 11px",flex:1,minWidth:80}}>
      <div style={{color:T.muted,fontSize:9,textTransform:"uppercase",letterSpacing:1,marginBottom:3}}>{label}</div>
      <div style={{color,fontSize:13,fontWeight:700,fontFamily:"'DM Mono',monospace",textShadow:`0 0 8px ${color}55`}}>
        {value}<span style={{fontSize:9,color:T.muted,marginLeft:2}}>{unit}</span>
      </div>
    </div>
  );
}

function ParamRow({ meta, value, onChange }) {
  const [raw,setRaw]     = useState(meta.sci ? value.toExponential(2) : String(value));
  const [err,setErr]     = useState(null);
  const [dirty,setDirty] = useState(false);
  useEffect(()=>{
    setRaw(meta.sci?value.toExponential(2):String(Number(value.toFixed?value.toFixed(6):value)));
    setErr(null);setDirty(false);
  },[value]);
  const commit=()=>{
    const res=validateParam(meta,raw);
    if(!res.ok){setErr(res.msg);return;}
    setErr(null);setDirty(false);onChange(res.v);
  };
  return (
    <div style={{marginBottom:8}}>
      <div style={{display:"flex",alignItems:"center",gap:6}}>
        <span style={{color:T.muted2,fontSize:10,flex:1,minWidth:0}}>{meta.label}</span>
        <div style={{display:"flex",alignItems:"center",gap:4}}>
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

const RemoveBtn = ({ onClick, title="Remove" }) => (
  <button onClick={onClick} title={title}
    style={{background:"#3a1010",border:"1px solid #f8717155",color:"#f87171",cursor:"pointer",
      borderRadius:5,fontSize:10,fontWeight:700,padding:"3px 8px",flexShrink:0,
      fontFamily:"'DM Mono',monospace"}}
    onMouseEnter={e=>{e.currentTarget.style.background="#5a1a1a";e.currentTarget.style.borderColor="#f87171";}}
    onMouseLeave={e=>{e.currentTarget.style.background="#3a1010";e.currentTarget.style.borderColor="#f8717155";}}>
    ✕ Remove
  </button>
);

// ═══════════════════════════════════════════════════════════════════════════
// LIMIT ROW  (one row inside the Limits sub-tab)
// ═══════════════════════════════════════════════════════════════════════════
//
// Lets the user override the upper bound of one slider type. In the
// generalized simulator that means: the cap for the Po slider on every
// conjugate (one shared cap, applied across all conjugates).
//
// Validation rules (applied at commit, i.e. blur or Enter):
//   1. Must parse as a finite number (NaN / "" / Infinity rejected)
//   2. Must be strictly greater than the slider's hard floor `meta.min`
//   3. Must be a positive value
//
// On a successful commit we call onChange(newMax). The parent then updates
// the limit in state and clamps any conjugate Po that exceeds the new max.
// ═══════════════════════════════════════════════════════════════════════════
function LimitRow({ meta, currentMax, currentValuesPreview, onChange }) {
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

  // Will-be-clamped warning: any current value above the proposed new max
  const proposedMax = parseFloat(raw);
  const willClamp   = !err && dirty && isFinite(proposedMax) &&
                      currentValuesPreview.some(v => v > proposedMax);

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
        <span>min ≥ {meta.min} · current values: {currentValuesPreview.length
          ? currentValuesPreview.map(v=>{
              const n = Number(v);
              if (!isFinite(n) || n === 0) return "0";
              const abs = Math.abs(n);
              // Scientific for very small or very large; fixed otherwise
              if (abs < 0.01 || abs >= 1e4) return n.toExponential(2);
              return n.toFixed(abs < 10 ? 2 : 1);
            }).join(", ")
          : "—"}</span>
        <span>default: {meta.def}</span>
      </div>
      {err && (
        <div style={{color:T.err,fontSize:9,marginTop:2,fontStyle:"italic"}}>
          ⚠ {err}
        </div>
      )}
      {willClamp && (
        <div style={{color:T.warn,fontSize:9,marginTop:2,fontStyle:"italic"}}>
          ⚠ values above {proposedMax} will be clamped down
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// STRIP CANVAS
// ═══════════════════════════════════════════════════════════════════════════
function LFAStrip({ simData, frameIdx, lines, physParams }) {
  const canvasRef = useRef(null);
  const draw = useCallback(()=>{
    const canvas=canvasRef.current; if(!canvas)return;
    const rect=canvas.getBoundingClientRect();
    const DPR=window.devicePixelRatio||1;
    const W=rect.width,H=rect.height;
    if(canvas.width!==W*DPR||canvas.height!==H*DPR){canvas.width=W*DPR;canvas.height=H*DPR;}
    const ctx=canvas.getContext("2d");
    ctx.setTransform(DPR,0,0,DPR,0,0);
    ctx.clearRect(0,0,W,H);
    const LABEL_H=48,AXIS_H=24,SY=LABEL_H,SH=H-LABEL_H-AXIS_H;
    PADS.forEach(r=>{
      const x1=r.frac[0]*W,rw=(r.frac[1]-r.frac[0])*W;
      const g=ctx.createLinearGradient(x1,SY,x1,SY+SH);
      g.addColorStop(0,r.bg+"ff");g.addColorStop(1,r.bg+"bb");
      ctx.fillStyle=g;ctx.fillRect(x1,SY,rw,SH);
      ctx.strokeStyle=r.bdr;ctx.lineWidth=1.5;
      ctx.beginPath();ctx.rect(x1+.75,SY+.75,rw-1.5,SH-1.5);ctx.stroke();
      ctx.save();ctx.globalAlpha=.04;ctx.strokeStyle="#fff";ctx.lineWidth=1;
      for(let i=1;i<9;i++){ctx.beginPath();ctx.moveTo(x1+2,SY+(i/9)*SH);ctx.lineTo(x1+rw-2,SY+(i/9)*SH);ctx.stroke();}
      ctx.restore();
      const cx=x1+rw/2;
      ctx.font="bold 10px 'DM Mono',monospace";ctx.fillStyle=r.acc;ctx.textAlign="center";
      ctx.fillText(r.label.toUpperCase(),cx,SY-20);
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=T.muted;ctx.fillText(r.sub,cx,SY-8);
    });
    if(!simData)return;
    const fi=Math.min(frameIdx,simData.times.length-1);
    const t_now=simData.times[fi];
    const strip_L=physParams.strip_L;
    const ncX0=.20*W,ncX1w=(.83-.20)*W;
    const cX0=ncX0+36,cX1=ncX0+ncX1w-36,cW=cX1-cX0;
    const cY1=SY+6,cY2=SY+SH-6,cH=cY2-cY1;
    const mmToCx=mm=>cX0+(mm/strip_L)*cW;
    // flow front
    const x4=21311*t_now-69505;
    const xf=x4>0?x4**.25:0;
    const ffX=mmToCx(xf);
    if(ffX>cX0&&ffX<cX1+10){
      ctx.save();ctx.strokeStyle="rgba(0,212,255,0.45)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
      ctx.beginPath();ctx.moveTo(ffX,SY+2);ctx.lineTo(ffX,SY+SH-2);ctx.stroke();ctx.setLineDash([]);ctx.restore();
    }
    // profiles from first non-control result
    const disp=simData.lineResults.find(r=>r.type!=='control')||simData.lineResults[0];
    if(disp&&disp.A_snap?.[fi]){
      const Af =gaussSmooth(new Float64Array(disp.A_snap[fi]),1.5);
      const Pf =gaussSmooth(new Float64Array(disp.P_snap[fi]),1.5);
      const PAf=gaussSmooth(new Float64Array(disp.PA_snap[fi]),1.5);
      const toX=i=>cX0+(i/(disp.xGrid.length-1))*cW;
      const leftMax=Math.max(...Af,...PAf,1e-30)*1.15;
      const rightMax=Math.max(...Pf,1e-30)*1.15;
      const toYL=v=>cY2-Math.min(Math.max(v/leftMax,0),1)*cH;
      const toYR=v=>cY2-Math.min(Math.max(v/rightMax,0),1)*cH;
      // axes
      for(let t=0;t<=4;t++){
        const val=leftMax*(t/4),cy=toYL(val);
        ctx.strokeStyle="rgba(56,189,248,0.35)";ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(cX0,cy);ctx.lineTo(cX0-5,cy);ctx.stroke();
        ctx.strokeStyle="rgba(56,189,248,0.06)";
        ctx.beginPath();ctx.moveTo(cX0,cy);ctx.lineTo(cX1,cy);ctx.stroke();
        ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(56,189,248,0.75)";
        ctx.textAlign="right";ctx.fillText(val.toExponential(1),cX0-7,cy+3);
      }
      ctx.strokeStyle="rgba(56,189,248,0.3)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cX0,cY1);ctx.lineTo(cX0,cY2);ctx.stroke();
      for(let t=0;t<=4;t++){
        const val=rightMax*(t/4),cy=toYR(val);
        ctx.strokeStyle="rgba(192,132,252,0.35)";ctx.lineWidth=1;
        ctx.beginPath();ctx.moveTo(cX1,cy);ctx.lineTo(cX1+5,cy);ctx.stroke();
        ctx.strokeStyle="rgba(192,132,252,0.05)";ctx.setLineDash([2,4]);
        ctx.beginPath();ctx.moveTo(cX0,cy);ctx.lineTo(cX1,cy);ctx.stroke();ctx.setLineDash([]);
        ctx.font="8px 'DM Mono',monospace";ctx.fillStyle="rgba(192,132,252,0.75)";
        ctx.textAlign="left";ctx.fillText(val.toExponential(1),cX1+7,cy+3);
      }
      ctx.strokeStyle="rgba(192,132,252,0.3)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cX1,cY1);ctx.lineTo(cX1,cY2);ctx.stroke();
      const drawProf=(vals,color,toYfn)=>{
        if(!vals?.length)return;
        const [ri,gi,bi]=[0,2,4].map(o=>parseInt(color.slice(1+o,3+o),16));
        ctx.save();ctx.beginPath();ctx.rect(cX0,cY1-1,cW,cH+2);ctx.clip();
        ctx.beginPath();ctx.moveTo(toX(0),cY2);
        for(let i=0;i<vals.length;i++)ctx.lineTo(toX(i),toYfn(vals[i]));
        ctx.lineTo(toX(vals.length-1),cY2);ctx.closePath();
        const fg=ctx.createLinearGradient(0,cY1,0,cY2);
        fg.addColorStop(0,`rgba(${ri},${gi},${bi},0.28)`);
        fg.addColorStop(.6,`rgba(${ri},${gi},${bi},0.08)`);
        fg.addColorStop(1,`rgba(${ri},${gi},${bi},0.01)`);
        ctx.fillStyle=fg;ctx.fill();
        ctx.beginPath();let pd=false;
        for(let i=0;i<vals.length;i++){const px=toX(i),py=toYfn(vals[i]);if(!pd){ctx.moveTo(px,py);pd=true;}else ctx.lineTo(px,py);}
        ctx.strokeStyle=`rgba(${ri},${gi},${bi},0.95)`;ctx.lineWidth=2;ctx.shadowColor=color;ctx.shadowBlur=4;
        ctx.stroke();ctx.shadowBlur=0;ctx.restore();
      };
      drawProf(Pf,T.P,toYR);drawProf(PAf,T.PA,toYL);drawProf(Af,T.A,toYL);
    }
    // draw line markers
    const allSigs=simData.lineResults.map(r=>r.signal_ts[fi]||0);
    const sharedPeak=Math.max(...allSigs,1e-30);
    simData.lineResults.forEach(lr=>{
      const lx=mmToCx(parseFloat(lr.pos));
      const sig=lr.signal_ts[fi]||0;
      const col=LINE_COLORS[lr.type]||T.RPA;
      const int=Math.pow(Math.min(Math.max(sig/sharedPeak,0),1),.35);
      const [ri,gi,bi]=[0,2,4].map(o=>parseInt(col.slice(1+o,3+o),16));
      ctx.save();
      ctx.strokeStyle=`rgba(${ri},${gi},${bi},${.18+int*.82})`;
      ctx.lineWidth=2.5+int*8;ctx.shadowColor=col;ctx.shadowBlur=4+int*26;
      ctx.beginPath();ctx.moveTo(lx,SY+2);ctx.lineTo(lx,SY+SH-2);ctx.stroke();ctx.shadowBlur=0;
      const pW=36,pH=15,pX=lx-pW/2,pY=SY-38;
      ctx.fillStyle=`rgba(${ri},${gi},${bi},${.08+int*.55})`;
      ctx.beginPath();ctx.roundRect(pX,pY,pW,pH,4);ctx.fill();
      ctx.strokeStyle=`rgba(${ri},${gi},${bi},${.3+int*.55})`;ctx.lineWidth=1;ctx.stroke();
      ctx.font="bold 9px 'DM Mono',monospace";
      ctx.fillStyle=`rgba(${ri},${gi},${bi},${.5+int*.5})`;ctx.textAlign="center";
      ctx.fillText(lr.name,lx,pY+10);
      if(sig>1e-20){
        ctx.font="8px 'DM Mono',monospace";
        ctx.fillStyle=`rgba(${ri},${gi},${bi},${.4+int*.45})`;
        ctx.fillText(`${sig.toExponential(1)} nM`,lx,SY+SH+13);
      }
      ctx.restore();
    });
    // x-axis
    const xTotalMM=disp?.xGrid[disp.xGrid.length-1]||strip_L;
    ctx.font="8px 'DM Mono',monospace";ctx.textAlign="center";
    [0,10,20,30,40,50,60,70,80,90,100].forEach(mm=>{
      if(mm>xTotalMM)return;
      const cx=cX0+(mm/xTotalMM)*cW;
      if(cx<cX0-1||cx>cX1+1)return;
      ctx.fillStyle="rgba(255,255,255,0.18)";ctx.fillRect(cx-.5,SY+SH-4,1,4);
      ctx.fillStyle=T.muted;ctx.fillText(`${mm}`,cx,SY+SH+13);
    });
    ctx.font="9px 'DM Mono',monospace";ctx.fillStyle=T.muted2;ctx.textAlign="center";
    ctx.fillText("Position (mm)",(cX0+cX1)/2,H-2);
  },[simData,frameIdx,lines,physParams]);

  const rafRef=useRef(null);
  useEffect(()=>{cancelAnimationFrame(rafRef.current);rafRef.current=requestAnimationFrame(draw);return()=>cancelAnimationFrame(rafRef.current);},[draw]);
  useEffect(()=>{
    const ro=new ResizeObserver(()=>{cancelAnimationFrame(rafRef.current);rafRef.current=requestAnimationFrame(draw);});
    if(canvasRef.current)ro.observe(canvasRef.current.parentElement);return()=>ro.disconnect();
  },[draw]);
  return <canvas ref={canvasRef} style={{width:"100%",height:230,display:"block",borderRadius:10,border:`1px solid ${T.border2}`,background:T.surface}}/>;
}

// ═══════════════════════════════════════════════════════════════════════════
// SIGNAL CHART
// ═══════════════════════════════════════════════════════════════════════════
function SignalChart({ simData, frameIdx, setFrameIdx }) {
  const canvasRef=useRef(null);
  useEffect(()=>{
    const canvas=canvasRef.current;if(!canvas||!simData)return;
    const DPR=window.devicePixelRatio||1;const W=canvas.offsetWidth,H=canvas.offsetHeight;
    canvas.width=W*DPR;canvas.height=H*DPR;const ctx=canvas.getContext("2d");ctx.scale(DPR,DPR);
    const {times,lineResults}=simData;
    const pad={t:14,r:16,b:36,l:62};const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
    const maxT=times[times.length-1];
    const allSigs=lineResults.flatMap(r=>r.signal_ts);
    const maxY=Math.max(...allSigs,1e-30);
    ctx.clearRect(0,0,W,H);
    const tx=t=>pad.l+(t/maxT)*iW;const ty=v=>pad.t+iH-(v/maxY)*iH;
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
    lineResults.forEach((lr,idx)=>{
      const col=LINE_COLORS[lr.type]||T.RPA;
      const [ri,gi,bi]=[0,2,4].map(o=>parseInt(col.slice(1+o,3+o),16));
      ctx.beginPath();ctx.moveTo(tx(times[0]),ty(0));
      times.forEach((t,i)=>ctx.lineTo(tx(t),ty(lr.signal_ts[i]||0)));
      ctx.lineTo(tx(times[times.length-1]),ty(0));ctx.closePath();
      const g=ctx.createLinearGradient(0,pad.t,0,pad.t+iH);
      g.addColorStop(0,`rgba(${ri},${gi},${bi},0.18)`);g.addColorStop(1,`rgba(${ri},${gi},${bi},0.02)`);
      ctx.fillStyle=g;ctx.fill();
      ctx.beginPath();
      times.forEach((t,i)=>i===0?ctx.moveTo(tx(t),ty(lr.signal_ts[i]||0)):ctx.lineTo(tx(t),ty(lr.signal_ts[i]||0)));
      ctx.strokeStyle=col;ctx.lineWidth=2.2;ctx.shadowColor=col;ctx.shadowBlur=6;ctx.stroke();ctx.shadowBlur=0;
      const lx=pad.l+6,ly=pad.t+13+idx*14;
      ctx.setLineDash([]);ctx.strokeStyle=col;ctx.lineWidth=2;
      ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+14,ly);ctx.stroke();
      ctx.font="9px 'DM Mono',monospace";ctx.fillStyle=col;ctx.textAlign="left";
      ctx.fillText(`${lr.name} [${lr.type[0].toUpperCase()}]`,lx+18,ly+3);
    });
    const fi=Math.min(frameIdx,times.length-1);const cx=tx(times[fi]);
    ctx.strokeStyle="rgba(0,212,255,0.45)";ctx.lineWidth=1.5;ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(cx,pad.t);ctx.lineTo(cx,pad.t+iH);ctx.stroke();ctx.setLineDash([]);
    lineResults.forEach(lr=>{
      const col=LINE_COLORS[lr.type]||T.RPA;
      ctx.beginPath();ctx.arc(cx,ty(lr.signal_ts[fi]||0),4.5,0,Math.PI*2);
      ctx.fillStyle=col;ctx.shadowColor=col;ctx.shadowBlur=8;ctx.fill();ctx.shadowBlur=0;
    });
    ctx.strokeStyle=T.border2;ctx.lineWidth=1;ctx.strokeRect(pad.l,pad.t,iW,iH);
    ctx.save();ctx.fillStyle=T.muted2;ctx.font="9px 'DM Mono',monospace";
    ctx.translate(11,pad.t+iH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";
    ctx.fillText("Signal (nM)",0,0);ctx.restore();
  },[simData,frameIdx]);
  const handleClick=useCallback(e=>{
    if(!simData)return;
    const canvas=canvasRef.current;const rect=canvas.getBoundingClientRect();
    const DPR=window.devicePixelRatio||1;
    const clickX=(e.clientX-rect.left)*(canvas.width/rect.width)/DPR;
    const padL=62,padR=16;const iW=canvas.width/DPR-padL-padR;
    const frac=(clickX-padL)/iW;
    setFrameIdx(Math.max(0,Math.min(simData.times.length-1,Math.round(frac*(simData.times.length-1)))));
  },[simData,setFrameIdx]);
  return <canvas ref={canvasRef} onClick={handleClick}
    style={{width:"100%",height:170,display:"block",borderRadius:8,border:`1px solid ${T.border}`,background:T.surface,cursor:"crosshair"}}/>;
}

// ═══════════════════════════════════════════════════════════════════════════
// MASS CONSERVATION + DEPLETION PANEL
// ═══════════════════════════════════════════════════════════════════════════
function MassConservationPanel({ simData, conjugates }) {
  if (!simData) return null;
  const { lineResults, conjSummary } = simData;

  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
        Mass Conservation & Conjugate Budget
      </div>

      {/* Receptor conservation per line */}
      <div style={{fontSize:9,color:T.muted2,marginBottom:8,fontWeight:600}}>
        Immobilised species conservation  <span style={{color:T.muted,fontWeight:400}}>(exact check — should always pass)</span>
      </div>
      <div style={{display:"flex",flexDirection:"column",gap:5,marginBottom:14}}>
        {lineResults.map(lr=>{
          const col=LINE_COLORS[lr.type]||T.RPA;
          const isComp=lr.type==='competitive'||lr.type==='reference';
          const isCtrl=lr.type==='control';
          let lhs='', rhs='', ok=lr.massOK;
          if (isCtrl) {
            lhs=`PC + Rfree = ${(lr.PC||0).toExponential(2)} + ${Math.max(0,lr.Ro-(lr.PC||0)).toExponential(2)}`;
            rhs=`Ro = ${lr.Ro}`;
          } else if (isComp) {
            lhs=`AgP + AgFree = ${lr.AgP.toExponential(2)} + ${(lr.fixedAg-lr.AgP).toExponential(2)}`;
            rhs=`fixedAg = ${lr.fixedAg}`;
          } else {
            lhs=`RA + RPA + Rfree = ${lr.RA.toExponential(2)} + ${lr.RPA.toExponential(2)} + ${(lr.Ro-lr.RA-lr.RPA).toExponential(2)}`;
            rhs=`Ro = ${lr.Ro}`;
          }
          return (
            <div key={lr.id} style={{display:"flex",alignItems:"center",gap:8,
              background:T.surface,borderRadius:6,padding:"5px 10px",
              border:`1px solid ${ok?T.ok+"22":T.err+"44"}`}}>
              <span style={{color:col,fontWeight:700,fontSize:10,minWidth:24}}>{lr.name}</span>
              <span style={{color:T.muted2,fontSize:9,flex:1,fontFamily:"monospace"}}>{lhs} = {rhs}</span>
              <span style={{color:ok?T.ok:T.err,fontSize:11,fontWeight:700}}>{ok?"✓":"✗"}</span>
            </div>
          );
        })}
      </div>

      {/* Conjugate pool depletion */}
      <div style={{fontSize:9,color:T.muted2,marginBottom:8,fontWeight:600}}>
        Conjugate pool depletion at t_end
        <span style={{color:T.muted,fontWeight:400}}> — how much [P] each group lost to line capture</span>
      </div>
      <div style={{display:"flex",flexDirection:"column",gap:8}}>
        {conjSummary.map(cs=>{
          const depCol = cs.depletedPct>80?T.err:cs.depletedPct>40?T.warn:T.ok;
          return (
            <div key={cs.id} style={{background:T.surface,borderRadius:7,padding:"8px 12px",
              border:`1px solid ${depCol}22`}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                <span style={{color:T.P,fontWeight:700,fontSize:10}}>{cs.name}</span>
                <span style={{color:depCol,fontFamily:"monospace",fontSize:10,fontWeight:700}}>
                  {cs.depletedPct.toFixed(1)}% depleted
                </span>
              </div>
              <div style={{height:5,background:T.border,borderRadius:3,marginBottom:5}}>
                <div style={{height:"100%",width:`${cs.depletedPct}%`,borderRadius:3,
                  background:depCol,boxShadow:`0 0 6px ${depCol}55`,transition:"width 0.1s"}}/>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:6}}>
                {[
                  ["Initial Po",`${cs.Po} nM`,T.P],
                  ["Escaped to pad",`${cs.pad_P.toExponential(2)} nM·mm`,T.muted2],
                  ["Remaining at ctrl",`${cs.P_at_ctrl.toExponential(2)} nM`,depCol],
                ].map(([l,v,c])=>(
                  <div key={l} style={{background:T.card,borderRadius:5,padding:"4px 7px"}}>
                    <div style={{color:T.muted,fontSize:8,marginBottom:2}}>{l}</div>
                    <div style={{color:c,fontSize:9,fontFamily:"monospace"}}>{v}</div>
                  </div>
                ))}
              </div>
              {cs.depletedPct>80&&(
                <div style={{marginTop:6,fontSize:8,color:T.err,lineHeight:1.5}}>
                  ⚠ Severe depletion — downstream lines (R2, C) may be starved of conjugate.
                  Increase Po or reduce upstream receptor Ro.
                </div>
              )}
              {cs.depletedPct>40&&cs.depletedPct<=80&&(
                <div style={{marginTop:6,fontSize:8,color:T.warn,lineHeight:1.5}}>
                  ⚡ Moderate depletion — T/R normalisation may be affected. Check R line signals.
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Open system note */}
      <div style={{marginTop:12,background:T.surface,borderRadius:6,padding:"8px 10px",
        border:`1px solid ${T.border}`,fontSize:9,color:T.muted,lineHeight:1.6}}>
        <span style={{color:T.muted2,fontWeight:600}}>Why mobile species aren't globally conserved:</span>
        {" "}This is an open advection-dominated system. [A], [P], [PA] flow out at the absorbent pad
        under U(x)=c/x³. Global mass balance doesn't close — only receptor and fixed-Ag conservation
        (immobilised species) can be checked exactly. Pad flux accounts for all escaped mobile mass.
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULTS TABLE
// ═══════════════════════════════════════════════════════════════════════════
function ResultsTable({ simData }) {
  if (!simData) return null;
  const lrs=simData.lineResults;
  const refLines=lrs.filter(r=>r.type==="reference");
  const refMean=refLines.length?refLines.reduce((s,r)=>s+r.finalSignal,0)/refLines.length:null;
  const maxSig=Math.max(...lrs.map(r=>r.finalSignal),1e-30);
  return (
    <div>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
        Line Signals at t_end
      </div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
          <thead>
            <tr style={{background:T.surface}}>
              {["Line","Type","Pos","Conjugate","Signal (nM)","% peak","T/R ratio","Conservation","Verdict"].map(h=>(
                <th key={h} style={{padding:"5px 10px",textAlign:"left",color:T.muted,fontSize:9,
                  fontWeight:600,borderBottom:`1px solid ${T.border2}`,whiteSpace:"nowrap"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {lrs.map((lr,i)=>{
              const col=LINE_COLORS[lr.type]||T.RPA;
              const pct=(lr.finalSignal/maxSig*100).toFixed(1);
              const ratio=(lr.type==="sandwich"||lr.type==="competitive")&&refMean
                ?(lr.finalSignal/refMean).toFixed(4):"—";
              let verdict="";
              if      (lr.type==="control")     verdict=lr.finalSignal>1e-20?"✓ Valid":"✗ No flow";
              else if (lr.type==="reference")   verdict="Calibration";
              else if (lr.type==="sandwich")    verdict=lr.finalSignal>maxSig*.15?"POSITIVE":"NEGATIVE";
              else if (lr.type==="competitive") verdict=lr.finalSignal<maxSig*.30?"HIGH [A]":"LOW [A]";
              const vCol=verdict.includes("POS")||verdict.includes("HIGH")||verdict==="Calibration"||verdict==="✓ Valid"?T.ok:T.muted2;
              return (
                <tr key={lr.id} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                  <td style={{padding:"4px 10px",color:col,fontWeight:700}}>{lr.name}</td>
                  <td style={{padding:"4px 10px"}}>
                    <span style={{background:`${col}15`,color:col,border:`1px solid ${col}44`,
                      borderRadius:4,padding:"1px 6px",fontSize:9}}>{lr.type}</span>
                  </td>
                  <td style={{padding:"4px 10px",color:T.muted2}}>{lr.pos}mm</td>
                  <td style={{padding:"4px 10px",color:T.P,fontSize:9}}>{lr.conjName}</td>
                  <td style={{padding:"4px 10px",color:T.text,fontFamily:"monospace"}}>{lr.finalSignal.toExponential(3)}</td>
                  <td style={{padding:"4px 10px"}}>
                    <div style={{display:"flex",alignItems:"center",gap:5}}>
                      <div style={{width:50,height:5,background:T.border,borderRadius:3}}>
                        <div style={{width:`${pct}%`,height:"100%",borderRadius:3,background:col,boxShadow:`0 0 4px ${col}55`}}/>
                      </div>
                      <span style={{color:T.muted2,fontSize:9}}>{pct}%</span>
                    </div>
                  </td>
                  <td style={{padding:"4px 10px",color:T.PA,fontFamily:"monospace",fontSize:9}}>{ratio}</td>
                  <td style={{padding:"4px 10px"}}>
                    <span style={{color:lr.massOK?T.ok:T.err,fontSize:11}}>{lr.massOK?"✓":"✗"}</span>
                  </td>
                  <td style={{padding:"4px 10px"}}>
                    <span style={{color:vCol,fontSize:9}}>{verdict}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {refMean&&(
        <div style={{marginTop:10,background:T.surface,borderRadius:7,padding:"8px 12px",
          border:`1px solid ${T.PA}22`,fontSize:10,color:T.muted2}}>
          Reference mean: <span style={{color:T.PA,fontFamily:"monospace"}}>{refMean.toExponential(3)} nM</span>
          &nbsp;— T/R ratio corrects for conjugate lot-to-lot variation and sample volume differences
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// LINE BUILDER
// ═══════════════════════════════════════════════════════════════════════════
function LineBuilder({ lines, setLines, analytes, conjugates, physParams }) {
  const aOpts=analytes.map(a=>({v:a.id,l:a.name}));
  const cOpts=conjugates.map(c=>({v:c.id,l:c.name}));
  const typeOpts=[
    {v:"sandwich",    l:"Sandwich (↑ with [A])"},
    {v:"competitive", l:"Competitive (↓ with [A])"},
    {v:"reference",   l:"Reference (calibration)"},
    {v:"control",     l:"Control (validates flow)"},
  ];
  const upd=(id,f,v)=>setLines(p=>p.map(l=>l.id===id?{...l,[f]:v}:l));
  const add=()=>{
    const maxPos=Math.max(...lines.map(l=>parseFloat(l.pos)||0),20);
    setLines(p=>[...p,{
      id:uid(),name:`L${p.length+1}`,type:"sandwich",
      pos:Math.min(maxPos+5,physParams.strip_L-5),
      analyteId:analytes[0]?.id||"",conjId:conjugates[0]?.id||"",
      Ro:10,fixedAg:20,Ka2:7.35e-4,Kd2:5.7e-5,
    }]);
  };
  const rem=id=>setLines(p=>p.filter(l=>l.id!==id));

  return (
    <div>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
        Lines ({lines.length})
      </div>
      <div style={{display:"flex",flexDirection:"column",gap:8}}>
        {[...lines].sort((a,b)=>parseFloat(a.pos)-parseFloat(b.pos)).map(ln=>{
          const col=LINE_COLORS[ln.type]||T.RPA;
          const isComp=ln.type==="competitive"||ln.type==="reference";
          const isCtrl=ln.type==="control";
          return (
            <div key={ln.id} style={{background:T.surface,border:`1px solid ${col}33`,borderRadius:8,padding:10}}>
              {/* Row 1: name, type, remove */}
              <div style={{display:"flex",gap:6,alignItems:"center",marginBottom:8}}>
                <div style={{width:8,height:8,borderRadius:2,background:col,flexShrink:0}}/>
                <input value={ln.name} onChange={e=>upd(ln.id,"name",e.target.value)}
                  style={{background:"transparent",border:"none",color:T.text,fontSize:11,
                    fontWeight:700,width:40,outline:"none",fontFamily:"'DM Mono',monospace"}}/>
                <select value={ln.type} onChange={e=>upd(ln.id,"type",e.target.value)}
                  style={{flex:1,background:T.card,border:`1px solid ${T.border}`,borderRadius:4,
                    color:T.muted2,fontSize:9,padding:"2px 4px",fontFamily:"'DM Mono',monospace"}}>
                  {typeOpts.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
                </select>
                <RemoveBtn onClick={()=>rem(ln.id)} title="Remove this line"/>
              </div>
              {/* Row 2: position + analyte (hidden for ctrl) + conjugate */}
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                <label style={{fontSize:9,color:T.muted2}}>
                  Position (mm)
                  <input type="number" value={ln.pos} min={1} max={physParams.strip_L-2} step={1}
                    onChange={e=>upd(ln.id,"pos",parseFloat(e.target.value))}
                    style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                      borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
                </label>
                {!isCtrl ? (
                  <label style={{fontSize:9,color:T.muted2}}>
                    Analyte
                    <select value={ln.analyteId} onChange={e=>upd(ln.id,"analyteId",e.target.value)}
                      style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,fontSize:9,padding:"2px 4px",fontFamily:"'DM Mono',monospace"}}>
                      {aOpts.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
                    </select>
                  </label>
                ) : (
                  <div style={{fontSize:9,color:T.muted2,display:"flex",flexDirection:"column",justifyContent:"flex-end"}}>
                    <span style={{color:T.muted,fontSize:8,marginBottom:3}}>Analyte</span>
                    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:4,
                      padding:"3px 6px",color:T.muted,fontSize:9,fontStyle:"italic"}}>— not applicable</div>
                  </div>
                )}
                <label style={{fontSize:9,color:T.muted2}}>
                  Conjugate
                  <select value={ln.conjId} onChange={e=>upd(ln.id,"conjId",e.target.value)}
                    style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                      borderRadius:4,color:T.text,fontSize:9,padding:"2px 4px",fontFamily:"'DM Mono',monospace"}}>
                    {cOpts.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
                  </select>
                  {isCtrl&&<div style={{fontSize:8,color:T.muted,marginTop:2,fontStyle:"italic"}}>
                    taps ALL conjugate pools
                  </div>}
                </label>
                {/* Ro / fixedAg */}
                {isComp ? (
                  <label style={{fontSize:9,color:T.muted2}}>
                    Fixed Ag (nM)
                    <input type="number" value={ln.fixedAg} step="any"
                      onChange={e=>upd(ln.id,"fixedAg",parseFloat(e.target.value))}
                      style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                ) : (
                  <label style={{fontSize:9,color:T.muted2}}>
                    {isCtrl?"Receptor Rc (nM)":"Receptor Ro (nM)"}
                    <input type="number" value={ln.Ro} step="any"
                      onChange={e=>upd(ln.id,"Ro",parseFloat(e.target.value))}
                      style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                )}
              </div>
              {/* Row 3: Ka2/Kd2 (hidden for control) */}
              {!isCtrl ? (
                <div style={{marginTop:6,display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
                  <label style={{fontSize:9,color:T.muted2}}>
                    Ka₂ (nM⁻¹s⁻¹)
                    <input type="number" value={ln.Ka2} step="any"
                      onChange={e=>upd(ln.id,"Ka2",parseFloat(e.target.value))}
                      style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                  <label style={{fontSize:9,color:T.muted2}}>
                    Kd₂ (s⁻¹)
                    <input type="number" value={ln.Kd2} step="any"
                      onChange={e=>upd(ln.id,"Kd2",parseFloat(e.target.value))}
                      style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                        borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
                  </label>
                </div>
              ) : (
                <div style={{marginTop:6,background:T.card,border:`1px solid ${T.border}`,
                  borderRadius:5,padding:"5px 8px",fontSize:8,color:T.muted,lineHeight:1.6}}>
                  <span style={{color:T.PC,fontWeight:700}}>P + Rc → PC</span>
                  &nbsp;· accumulates [P] from ALL conjugate groups · uses conjugate Ka₁/Kd₁
                </div>
              )}
              {/* hint */}
              <div style={{marginTop:4,fontSize:8,color:T.muted,fontStyle:"italic"}}>
                {ln.type==="sandwich"   ?"Signal ↑ with [A] — competes for shared [P] with other lines in group"
                :ln.type==="competitive"?"Signal ↓ with [A] — Ag competes for shared [P]"
                :ln.type==="reference"  ?"Stable Ag anchor — shares [P] pool with same conjugate group"
                :                        "Captures residual [P] from ALL conjugate groups — validates run"}
              </div>
            </div>
          );
        })}
      </div>
      <button onClick={add}
        style={{marginTop:8,width:"100%",background:T.surface,border:`1px solid ${T.borderHi}`,
          color:T.muted2,borderRadius:6,padding:"6px 0",cursor:"pointer",
          fontFamily:"'DM Mono',monospace",fontSize:10}}>
        + Add line
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// ANALYTES & CONJUGATES BUILDER
// ═══════════════════════════════════════════════════════════════════════════
function AnCjBuilder({ analytes, setAnalytes, conjugates, setConjugates }) {
  const updA=(id,f,v)=>setAnalytes(p=>p.map(a=>a.id===id?{...a,[f]:v}:a));
  const updC=(id,f,v)=>setConjugates(p=>p.map(c=>c.id===id?{...c,[f]:v}:c));
  const addA=()=>setAnalytes(p=>[...p,{id:uid(),name:`Analyte ${String.fromCharCode(65+p.length)}`,Ao:0.01,aoMax:DEFAULT_AO_MAX}]);
  const addC=()=>setConjugates(p=>[...p,{id:uid(),name:`Conjugate ${p.length+1}`,Po:6,analyteId:analytes[0]?.id||"",Ka1:7.35e-4,Kd1:5.7e-5,poMax:DEFAULT_PO_MAX}]);
  const aOpts=analytes.map(a=>({v:a.id,l:a.name}));

  return (
    <div>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Analytes</div>
      {analytes.map(a=>(
        <div key={a.id} style={{background:T.surface,border:`1px solid ${T.A}22`,borderRadius:7,padding:8,marginBottom:6}}>
          <div style={{display:"flex",gap:6,alignItems:"center",marginBottom:6}}>
            <input value={a.name} onChange={e=>updA(a.id,"name",e.target.value)}
              style={{flex:1,background:"transparent",border:"none",color:T.A,fontSize:11,fontWeight:700,
                outline:"none",fontFamily:"'DM Mono',monospace"}}/>
            <RemoveBtn onClick={()=>setAnalytes(p=>p.filter(x=>x.id!==a.id))} title="Remove analyte"/>
          </div>
          <LogSlider label="[A₀] initial conc." value={parseFloat(a.Ao)||1e-7}
            logMin={-9} logMax={Math.log10(Math.max(parseFloat(a.aoMax)||DEFAULT_AO_MAX, 1e-8))} unit="nM" color={T.A} onChange={v=>updA(a.id,"Ao",v)}/>
        </div>
      ))}
      <button onClick={addA}
        style={{width:"100%",background:T.surface,border:`1px solid ${T.borderHi}`,color:T.muted2,
          borderRadius:6,padding:"5px 0",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:10,marginBottom:14}}>
        + Add analyte
      </button>

      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Conjugates</div>
      {/* Info box explaining competition */}
      <div style={{background:"#0a1a20",border:`1px solid ${T.P}22`,borderRadius:7,padding:"8px 10px",marginBottom:10,fontSize:9,color:T.muted2,lineHeight:1.6}}>
        <span style={{color:T.P,fontWeight:700}}>Competition rule:</span>
        {" "}Lines sharing the same conjugate compete for one [P] pool.
        Lines with different conjugates are independent.
        The control line taps <span style={{color:T.PC}}>all</span> conjugate pools simultaneously.
      </div>
      {conjugates.map(c=>(
        <div key={c.id} style={{background:T.surface,border:`1px solid ${T.P}22`,borderRadius:7,padding:8,marginBottom:6}}>
          <div style={{display:"flex",gap:6,alignItems:"center",marginBottom:6}}>
            <input value={c.name} onChange={e=>updC(c.id,"name",e.target.value)}
              style={{flex:1,background:"transparent",border:"none",color:T.P,fontSize:11,fontWeight:700,
                outline:"none",fontFamily:"'DM Mono',monospace"}}/>
            <RemoveBtn onClick={()=>setConjugates(p=>p.filter(x=>x.id!==c.id))} title="Remove conjugate"/>
          </div>
          <Slider label="[P₀] conj. conc." value={parseFloat(c.Po)||6}
            min={0.1} max={parseFloat(c.poMax)||DEFAULT_PO_MAX} step={0.5} unit="nM" color={T.P} onChange={v=>updC(c.id,"Po",v)}/>
          <label style={{fontSize:9,color:T.muted2,display:"flex",gap:6,alignItems:"center",marginBottom:6}}>
            Binds analyte:
            <select value={c.analyteId} onChange={e=>updC(c.id,"analyteId",e.target.value)}
              style={{flex:1,background:T.card,border:`1px solid ${T.border}`,borderRadius:4,
                color:T.text,fontSize:9,padding:"2px 4px",fontFamily:"'DM Mono',monospace"}}>
              {aOpts.map(o=><option key={o.v} value={o.v}>{o.l}</option>)}
            </select>
          </label>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
            <label style={{fontSize:9,color:T.muted2}}>
              Ka₁ (nM⁻¹s⁻¹)
              <input type="number" value={c.Ka1} step="any"
                onChange={e=>updC(c.id,"Ka1",parseFloat(e.target.value))}
                style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                  borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
            </label>
            <label style={{fontSize:9,color:T.muted2}}>
              Kd₁ (s⁻¹)
              <input type="number" value={c.Kd1} step="any"
                onChange={e=>updC(c.id,"Kd1",parseFloat(e.target.value))}
                style={{display:"block",width:"100%",background:T.card,border:`1px solid ${T.border}`,
                  borderRadius:4,color:T.text,fontSize:10,padding:"2px 5px",fontFamily:"'DM Mono',monospace"}}/>
            </label>
          </div>
        </div>
      ))}
      <button onClick={addC}
        style={{width:"100%",background:T.surface,border:`1px solid ${T.borderHi}`,color:T.muted2,
          borderRadius:6,padding:"5px 0",cursor:"pointer",fontFamily:"'DM Mono',monospace",fontSize:10}}>
        + Add conjugate
      </button>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SWEEP CHART + PANEL
// ═══════════════════════════════════════════════════════════════════════════
function SweepChart({ sweepData }) {
  const canvasRef=useRef(null);
  useEffect(()=>{
    const canvas=canvasRef.current;if(!canvas||!sweepData?.length)return;
    const DPR=window.devicePixelRatio||1;const W=canvas.offsetWidth,H=canvas.offsetHeight;
    canvas.width=W*DPR;canvas.height=H*DPR;const ctx=canvas.getContext("2d");ctx.scale(DPR,DPR);
    ctx.clearRect(0,0,W,H);
    const PAD={t:22,r:72,b:46,l:64};const IW=W-PAD.l-PAD.r,IH=H-PAD.t-PAD.b;
    const xs=sweepData.map(d=>d.Ao);
    const minX=xs[0],maxX=xs[xs.length-1];
    const allSigs=sweepData.flatMap(d=>d.lineSigs.map(s=>s.signal));
    const maxSIG=Math.max(...allSigs,1e-30);
    const lo=Math.log10(Math.max(minX,1e-20)),hi=Math.log10(maxX);
    const tx=v=>PAD.l+((Math.log10(Math.max(v,1e-20))-lo)/(hi-lo))*IW;
    const ty=v=>PAD.t+IH-Math.min(Math.max(v/maxSIG,0),1)*IH;
    for(let i=0;i<=4;i++){
      const val=maxSIG*i/4,cy=ty(val);
      ctx.strokeStyle="rgba(255,255,255,0.06)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(PAD.l,cy);ctx.lineTo(W-PAD.r,cy);ctx.stroke();
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=T.muted2;ctx.textAlign="right";
      ctx.fillText(val.toExponential(1),PAD.l-4,cy+3);
    }
    for(let e=Math.ceil(lo);e<=Math.floor(hi);e++){
      const cx=tx(Math.pow(10,e));
      ctx.strokeStyle="rgba(255,255,255,0.06)";ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cx,PAD.t);ctx.lineTo(cx,PAD.t+IH);ctx.stroke();
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=T.muted2;ctx.textAlign="center";
      ctx.fillText(`1e${e}`,cx,H-PAD.b+12);
    }
    const lineNames=sweepData[0].lineSigs.map(s=>({name:s.name,type:s.type}));
    lineNames.forEach((ln,li)=>{
      const col=LINE_COLORS[ln.type]||T.RPA;
      ctx.beginPath();
      sweepData.forEach((d,i)=>{
        const v=d.lineSigs[li]?.signal||0;
        i===0?ctx.moveTo(tx(d.Ao),ty(v)):ctx.lineTo(tx(d.Ao),ty(v));
      });
      ctx.strokeStyle=col;ctx.lineWidth=2.2;ctx.shadowColor=col;ctx.shadowBlur=5;ctx.stroke();ctx.shadowBlur=0;
      sweepData.forEach(d=>{
        const v=d.lineSigs[li]?.signal||0;
        ctx.beginPath();ctx.arc(tx(d.Ao),ty(v),3,0,Math.PI*2);
        ctx.fillStyle=col;ctx.shadowColor=col;ctx.shadowBlur=4;ctx.fill();ctx.shadowBlur=0;
      });
      const lx=W-PAD.r-80,ly=PAD.t+10+li*14;
      ctx.strokeStyle=col;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(lx,ly);ctx.lineTo(lx+12,ly);ctx.stroke();
      ctx.font="8px 'DM Mono',monospace";ctx.fillStyle=col;ctx.textAlign="left";ctx.fillText(ln.name,lx+15,ly+3);
    });
    ctx.strokeStyle="rgba(40,50,70,1)";ctx.lineWidth=1;ctx.strokeRect(PAD.l,PAD.t,IW,IH);
    ctx.font="9px 'DM Mono',monospace";ctx.fillStyle="rgba(180,180,200,0.65)";
    ctx.textAlign="center";ctx.fillText("[A₀] (nM) — log scale",PAD.l+IW/2,H-3);
    ctx.save();ctx.font="9px 'DM Mono',monospace";ctx.fillStyle="rgba(56,189,248,0.6)";
    ctx.translate(11,PAD.t+IH/2);ctx.rotate(-Math.PI/2);ctx.textAlign="center";ctx.fillText("Signal (nM)",0,0);ctx.restore();
  },[sweepData]);
  return <canvas ref={canvasRef} style={{width:"100%",height:280,display:"block",borderRadius:8,border:`1px solid ${T.border}`,background:T.surface}}/>;
}

function SweepPanel({ analytes, conjugates, lines, physParams }) {
  const [sweepAId,setSweepAId]=useState(analytes[0]?.id||"");
  const [logMin,setLogMin]=useState(-9);
  const [logMax,setLogMax]=useState(2);
  const [nPts,setNPts]=useState(14);
  const [sweepData,setSweepData]=useState(null);
  const [sweeping,setSweeping]=useState(false);

  const run=useCallback(()=>{
    setSweeping(true);setSweepData(null);
    setTimeout(()=>{
      try {
        const pts=Array.from({length:nPts},(_,i)=>Math.pow(10,logMin+(logMax-logMin)*i/(nPts-1)));
        const data=pts.map(Ao=>{
          const modA=analytes.map(a=>a.id===sweepAId?{...a,Ao}:a);
          const r=runSimulation({analytes:modA,conjugates,lines,pp:physParams});
          return {Ao,lineSigs:r.lineResults.map(lr=>({id:lr.id,name:lr.name,type:lr.type,signal:lr.finalSignal}))};
        });
        setSweepData(data);
      } catch(e){console.error(e);}
      setSweeping(false);
    },20);
  },[analytes,conjugates,lines,physParams,sweepAId,logMin,logMax,nPts]);

  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10,flexWrap:"wrap"}}>
        <span style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase"}}>Sweep analyte</span>
        <select value={sweepAId} onChange={e=>setSweepAId(e.target.value)}
          style={{background:T.surface,border:`1px solid ${T.border}`,borderRadius:5,
            color:T.text,padding:"3px 8px",fontSize:10,fontFamily:"'DM Mono',monospace"}}>
          {analytes.map(a=><option key={a.id} value={a.id}>{a.name}</option>)}
        </select>
        <span style={{fontSize:9,color:T.muted,marginLeft:8}}>log range</span>
        {[["min",logMin,setLogMin,-20,2],["max",logMax,setLogMax,-9,10]].map(([lbl,val,set,mn,mx])=>(
          <label key={lbl} style={{fontSize:9,color:T.muted2,display:"flex",gap:4,alignItems:"center"}}>
            {lbl}
            <input type="number" min={mn} max={mx} value={val} onChange={e=>set(Number(e.target.value))}
              style={{width:42,background:T.card,border:`1px solid ${T.border}`,borderRadius:4,
                color:T.text,padding:"2px 5px",fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
          </label>
        ))}
        <label style={{fontSize:9,color:T.muted2,display:"flex",gap:4,alignItems:"center"}}>
          pts
          <input type="number" min={4} max={30} value={nPts} onChange={e=>setNPts(Math.max(4,Math.min(30,+e.target.value)))}
            style={{width:40,background:T.card,border:`1px solid ${T.border}`,borderRadius:4,
              color:T.text,padding:"2px 5px",fontSize:10,fontFamily:"'DM Mono',monospace"}}/>
        </label>
        <button onClick={run} disabled={sweeping}
          style={{marginLeft:"auto",background:sweeping?T.border:"#0ea5e9",border:"none",
            color:"#fff",padding:"6px 18px",borderRadius:6,cursor:sweeping?"not-allowed":"pointer",
            fontFamily:"inherit",fontSize:11,fontWeight:600}}>
          {sweeping?"⧗ Running…":"Run Sweep"}
        </button>
      </div>
      {sweepData?<SweepChart sweepData={sweepData}/>
        :<div style={{height:280,display:"flex",alignItems:"center",justifyContent:"center",flexDirection:"column",gap:8}}>
          <span style={{fontSize:20}}>📈</span>
          <span style={{color:T.muted,fontSize:11}}>Select analyte and click Run Sweep</span>
          <span style={{color:T.muted,fontSize:9}}>All lines compete for their shared conjugate pool at each concentration point</span>
        </div>}
      {sweepData&&(
        <div style={{marginTop:14,overflowX:"auto"}}>
          <div style={{fontSize:9,color:T.muted,letterSpacing:2,textTransform:"uppercase",marginBottom:6}}>Sweep Results</div>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:10}}>
            <thead><tr style={{background:T.surface}}>
              {["[A₀] (nM)",...sweepData[0].lineSigs.map(s=>s.name)].map(h=>(
                <th key={h} style={{padding:"5px 10px",textAlign:"left",color:T.muted,fontSize:9,
                  fontWeight:600,borderBottom:`1px solid ${T.border2}`}}>{h}</th>
              ))}
            </tr></thead>
            <tbody>
              {sweepData.map((d,i)=>(
                <tr key={i} style={{background:i%2===0?T.surface+"44":"transparent"}}>
                  <td style={{padding:"4px 10px",color:T.accent,fontFamily:"monospace",fontWeight:600}}>{d.Ao.toExponential(2)}</td>
                  {d.lineSigs.map(s=>(
                    <td key={s.id} style={{padding:"4px 10px",color:LINE_COLORS[s.type]||T.RPA,fontFamily:"monospace"}}>
                      {s.signal.toExponential(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL INFO
// ═══════════════════════════════════════════════════════════════════════════
function ModelInfo() {
  return (
    <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:12}}>
        Reaction Model + Solver Architecture
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:8,marginBottom:14}}>
        {[
          {n:"①",eq:"A + P  ⇌  PA",    desc:"Conjugate binds analyte in solution (bulk, all nodes)",  c:T.P},
          {n:"②",eq:"A + R  ⇌  RA",    desc:"Analyte binds receptor (sandwich line zone only)",        c:T.PA},
          {n:"③",eq:"P + RA  →  RPA",   desc:"Conjugate labels receptor-analyte — signal ↑",           c:T.RPA},
          {n:"④",eq:"PA + R  →  RPA",   desc:"Labelled complex captured — signal ↑",                    c:T.RPA},
          {n:"⑤",eq:"Ag + P  ⇌  AgP",  desc:"Fixed Ag captures conjugate (comp/ref line) — signal ↓", c:T.warn},
          {n:"⑥",eq:"P + Rc  →  PC",    desc:"Anti-species Ab captures any conjugate (control line)",   c:T.PC},
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
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>
        Coupled Solver Architecture
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(2,1fr)",gap:6,marginBottom:10}}>
        {[
          ["One PDE per conjugate","Lines sharing conjId compete for one shared [P] field"],
          ["Control taps all pools","PC = sum of P captured from every conjugate group"],
          ["Mass conservation","RA+RPA+Rfree=Ro and AgP+AgFree=fixedAg checked exactly"],
          ["Open mobile system","A/P/PA conserved via pad flux accounting, not global sum"],
          ["Flow U(x)=min(c/x³,U_max)","Berli & Kler 2016 + velocity cap"],
          ["Depletion warning","Flags if >40% of [P] consumed before control line"],
        ].map(([k,v])=>(
          <div key={k} style={{background:T.surface,border:`1px solid ${T.border}`,borderRadius:6,padding:"7px 10px"}}>
            <div style={{color:T.accent,fontSize:10,fontWeight:700,marginBottom:1}}>{k}</div>
            <div style={{color:T.muted,fontSize:9}}>{v}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// REPORT
// ═══════════════════════════════════════════════════════════════════════════
function SimReport({ simData, analytes, conjugates, lines, physParams }) {
  if (!simData) return <div style={{padding:40,textAlign:"center",color:T.muted,fontSize:11}}>Run simulation to generate report.</div>;
  const lrs=simData.lineResults;
  const refLines=lrs.filter(r=>r.type==="reference");
  const refMean=refLines.length?refLines.reduce((s,r)=>s+r.finalSignal,0)/refLines.length:null;
  const massErrors=lrs.filter(lr=>!lr.massOK);

  const download=()=>{
    const rows=[
      "LFA GENERALISED SIMULATION REPORT","Generated: "+new Date().toISOString(),"",
      "── ANALYTES ──",...analytes.map(a=>`  ${a.name}: Ao=${parseFloat(a.Ao).toExponential(3)} nM`),
      "","── CONJUGATES ──",...conjugates.map(c=>`  ${c.name}: Po=${c.Po} nM  Ka1=${parseFloat(c.Ka1).toExponential(2)}  Kd1=${parseFloat(c.Kd1).toExponential(2)}`),
      "","── CONJUGATE COMPETITION GROUPS ──",
      ...simData.conjSummary.map(cs=>`  ${cs.name}: depleted ${cs.depletedPct.toFixed(1)}%  pad_P=${cs.pad_P.toExponential(2)} nM·mm`),
      "","── LINES & SIGNALS ──",
      ...lrs.map(lr=>`  ${lr.name} [${lr.type}] @ ${lr.pos}mm  conj=${lr.conjName}  signal=${lr.finalSignal.toExponential(4)} nM  massOK=${lr.massOK}${refMean?`  T/R=${(lr.finalSignal/refMean).toFixed(4)}`:""}`),
      "",refMean?`Reference mean: ${refMean.toExponential(4)} nM`:"",
      "","── MASS CONSERVATION ──",
      massErrors.length?massErrors.map(lr=>`  ✗ ${lr.name}`).join("\n"):"  ✓ All lines pass",
      "","── PHYSICAL PARAMS ──",
      `  strip_L=${physParams.strip_L}mm  phi=${physParams.phi}  tau=${physParams.tau}`,
      `  U_max=${physParams.U_max}mm/s  t_end=${physParams.t_end}s  sample=${physParams.sample_vol}µL`,
      `  DA=${physParams.DA}  DP=${physParams.DP}  flow_c=${physParams.flow_c}`,
    ];
    const blob=new Blob([rows.join("\n")],{type:"text/plain"});
    const a=document.createElement("a");a.href=URL.createObjectURL(blob);
    a.download=`LFA_Report_${Date.now()}.txt`;a.click();
  };

  return (
    <div style={{fontFamily:"'DM Mono',monospace"}}>
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:20,
        paddingBottom:12,borderBottom:`1px solid ${T.border2}`}}>
        <div>
          <div style={{fontSize:16,fontWeight:700,color:T.text,fontFamily:"'Syne',sans-serif",marginBottom:3}}>Simulation Report</div>
          <div style={{fontSize:9,color:T.muted}}>
            {lrs.length} lines · {simData.times.length} frames · {massErrors.length===0?"✓ all mass checks pass":`✗ ${massErrors.length} mass error(s)`}
          </div>
        </div>
        <button onClick={download}
          style={{background:T.surface,border:`1px solid ${T.border2}`,color:T.muted2,
            padding:"5px 12px",borderRadius:6,cursor:"pointer",fontFamily:"inherit",fontSize:10}}>
          ⬇ Download .txt
        </button>
      </div>

      {/* Conjugate competition summary */}
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Conjugate Competition Groups</div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(200px,1fr))",gap:8,marginBottom:16}}>
        {simData.conjSummary.map(cs=>{
          const depCol=cs.depletedPct>80?T.err:cs.depletedPct>40?T.warn:T.ok;
          const groupLines=lrs.filter(lr=>lr.conjId===cs.id);
          return (
            <div key={cs.id} style={{background:T.surface,border:`1px solid ${T.P}22`,borderRadius:7,padding:"8px 12px"}}>
              <div style={{color:T.P,fontWeight:700,fontSize:11,marginBottom:4}}>{cs.name}</div>
              <div style={{color:T.muted2,fontSize:9}}>Po = <span style={{color:T.text,fontFamily:"monospace"}}>{cs.Po} nM</span></div>
              <div style={{color:T.muted2,fontSize:9}}>Lines: <span style={{color:T.text}}>{groupLines.map(l=>l.name).join(", ")||"none"}</span></div>
              <div style={{color:depCol,fontSize:9,fontWeight:700,marginTop:4}}>{cs.depletedPct.toFixed(1)}% depleted</div>
            </div>
          );
        })}
        {/* Control line separate box */}
        {lrs.filter(l=>l.type==="control").map(cl=>(
          <div key={cl.id} style={{background:T.surface,border:`1px solid ${T.PC}22`,borderRadius:7,padding:"8px 12px"}}>
            <div style={{color:T.PC,fontWeight:700,fontSize:11,marginBottom:4}}>{cl.name} (Control)</div>
            <div style={{color:T.muted2,fontSize:9}}>Taps: <span style={{color:T.text}}>ALL conjugate pools</span></div>
            <div style={{color:T.muted2,fontSize:9}}>PC = <span style={{color:T.PC,fontFamily:"monospace"}}>{cl.finalSignal.toExponential(3)} nM</span></div>
          </div>
        ))}
      </div>

      {/* Line results */}
      <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:8}}>Line Results</div>
      <div style={{display:"flex",flexDirection:"column",gap:8,marginBottom:16}}>
        {lrs.map(lr=>{
          const col=LINE_COLORS[lr.type]||T.RPA;
          const isComp=lr.type==="competitive"||lr.type==="reference";
          return (
            <div key={lr.id} style={{background:T.surface,border:`1px solid ${col}33`,borderRadius:8,padding:12}}>
              <div style={{display:"flex",gap:10,alignItems:"center",marginBottom:8}}>
                <span style={{color:col,fontWeight:700,fontSize:13}}>{lr.name}</span>
                <span style={{background:`${col}15`,color:col,border:`1px solid ${col}44`,borderRadius:4,padding:"1px 6px",fontSize:9}}>{lr.type}</span>
                <span style={{color:T.muted2,fontSize:10}}>@ {lr.pos}mm · {lr.conjName}</span>
                <span style={{marginLeft:"auto",color:lr.massOK?T.ok:T.err,fontSize:11}}>{lr.massOK?"✓ mass OK":"✗ mass error"}</span>
                <span style={{color:col,fontSize:14,fontFamily:"monospace",fontWeight:700}}>{lr.finalSignal.toExponential(3)} nM</span>
              </div>
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8}}>
                {!isComp&&lr.type!=="control"&&<>
                  <div style={{background:T.card,borderRadius:6,padding:"6px 8px"}}>
                    <div style={{color:T.muted,fontSize:8,marginBottom:2}}>RA (unlabelled)</div>
                    <div style={{color:T.PA,fontFamily:"monospace",fontSize:10}}>{lr.RA.toExponential(3)} nM</div>
                  </div>
                  <div style={{background:T.card,borderRadius:6,padding:"6px 8px"}}>
                    <div style={{color:T.muted,fontSize:8,marginBottom:2}}>RPA signal</div>
                    <div style={{color:T.RPA,fontFamily:"monospace",fontSize:10}}>{lr.RPA.toExponential(3)} nM</div>
                  </div>
                </>}
                {isComp&&<div style={{background:T.card,borderRadius:6,padding:"6px 8px"}}>
                  <div style={{color:T.muted,fontSize:8,marginBottom:2}}>AgP signal</div>
                  <div style={{color:T.warn,fontFamily:"monospace",fontSize:10}}>{lr.AgP.toExponential(3)} nM</div>
                </div>}
                {lr.type==="control"&&<div style={{background:T.card,borderRadius:6,padding:"6px 8px"}}>
                  <div style={{color:T.muted,fontSize:8,marginBottom:2}}>PC (all conj.)</div>
                  <div style={{color:T.PC,fontFamily:"monospace",fontSize:10}}>{lr.finalSignal.toExponential(3)} nM</div>
                </div>}
                {refMean&&(lr.type==="sandwich"||lr.type==="competitive")&&(
                  <div style={{background:T.card,borderRadius:6,padding:"6px 8px"}}>
                    <div style={{color:T.muted,fontSize:8,marginBottom:2}}>T/R ratio</div>
                    <div style={{color:T.accent,fontFamily:"monospace",fontSize:10}}>{(lr.finalSignal/refMean).toFixed(4)}</div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
      {refMean&&(
        <div style={{background:`${T.PA}0a`,border:`1px solid ${T.PA}33`,borderRadius:8,padding:"10px 14px"}}>
          <div style={{color:T.PA,fontWeight:700,fontSize:11,marginBottom:4}}>Normalised Readout (T/R)</div>
          <div style={{fontSize:10,color:T.muted2,lineHeight:1.7}}>
            Reference mean = <span style={{color:T.PA,fontFamily:"monospace"}}>{refMean.toExponential(3)} nM</span><br/>
            T/R ratio corrects for conjugate lot variation, sample volume, and reader differences.
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
export default function LFASimulator() {
  const [analytes,    setAnalytes]    = useState([{ id:"a1", name:"Analyte A", Ao:1e-7, aoMax:DEFAULT_AO_MAX }]);
  const [conjugates,  setConjugates]  = useState([{ id:"c1", name:"Anti-A·Gold", Po:6, analyteId:"a1", Ka1:7.35e-4, Kd1:5.7e-5, poMax:DEFAULT_PO_MAX }]);
  const [lines,       setLines]       = useState([
    { id:"l1", name:"T1", type:"sandwich",    pos:25, analyteId:"a1", conjId:"c1", Ro:10,  fixedAg:0,  Ka2:7.35e-4, Kd2:5.7e-5 },
    { id:"l2", name:"R1", type:"reference",   pos:45, analyteId:"a1", conjId:"c1", Ro:0,   fixedAg:20, Ka2:7.35e-4, Kd2:5.7e-5 },
    { id:"l3", name:"R2", type:"reference",   pos:60, analyteId:"a1", conjId:"c1", Ro:0,   fixedAg:80, Ka2:7.35e-4, Kd2:5.7e-5 },
    { id:"l4", name:"C",  type:"control",     pos:75, analyteId:"a1", conjId:"c1", Ro:10,  fixedAg:0,  Ka2:7.35e-4, Kd2:5.7e-5 },
  ]);
  const [physParams,  setPhysParams]  = useState(defPhysical());

  // ── Per-item cap updaters ─────────────────────────────────────────────────
  // Lower an item's cap below its current Ao/Po → the value is clamped down.
  // Cap fields live on the items themselves; see DEFAULT_AO_MAX / DEFAULT_PO_MAX.
  const updateAnalyteCap = useCallback((id, newMax) => {
    setAnalytes(prev =>
      prev.map(a => {
        if (a.id !== id) return a;
        const cur = parseFloat(a.Ao);
        const clampedAo = isFinite(cur) && cur > newMax ? newMax : a.Ao;
        return { ...a, aoMax: newMax, Ao: clampedAo };
      })
    );
  }, []);
  const updateConjugateCap = useCallback((id, newMax) => {
    setConjugates(prev =>
      prev.map(c => {
        if (c.id !== id) return c;
        const cur = parseFloat(c.Po);
        const clampedPo = isFinite(cur) && cur > newMax ? newMax : c.Po;
        return { ...c, poMax: newMax, Po: clampedPo };
      })
    );
  }, []);

  const [simData,     setSimData]     = useState(null);
  const [running,     setRunning]     = useState(false);
  const [frameIdx,    setFrameIdx]    = useState(0);
  const [playing,     setPlaying]     = useState(false);
  const [tab,         setTab]         = useState("sim");
  const [leftTab,     setLeftTab]     = useState("lines");
  const intervalRef = useRef(null);

  useEffect(()=>{ handleRun(); },[]);

  useEffect(()=>{
    clearInterval(intervalRef.current);
    if(playing&&simData){
      intervalRef.current=setInterval(()=>{
        setFrameIdx(i=>{if(i>=simData.times.length-1){setPlaying(false);return i;}return i+1;});
      },50);
    }
    return()=>clearInterval(intervalRef.current);
  },[playing,simData]);

  const handleRun=useCallback(()=>{
    setRunning(true);setPlaying(false);setFrameIdx(0);
    setTimeout(()=>{
      try { setSimData(runSimulation({analytes,conjugates,lines,pp:physParams})); }
      catch(e){ console.error(e); }
      setRunning(false);
    },15);
  },[analytes,conjugates,lines,physParams]);

  const fi      = simData?Math.min(frameIdx,simData.times.length-1):0;
  const curT    = simData?simData.times[fi]:0;
  const firstTest = simData?.lineResults.find(r=>r.type==="sandwich"||r.type==="competitive");
  const firstCtrl = simData?.lineResults.find(r=>r.type==="control");
  const curSig  = firstTest?(firstTest.signal_ts[fi]||0):0;
  const curCtrl = firstCtrl?(firstCtrl.signal_ts[fi]||0):0;
  const refLines= simData?.lineResults.filter(r=>r.type==="reference")||[];
  const refMean = refLines.length?refLines.reduce((s,r)=>s+r.finalSignal,0)/refLines.length:null;
  const TC      = refMean&&firstTest?firstTest.finalSignal/refMean:null;

  // Depletion warning for header
  const deplWarn = simData?.conjSummary.some(cs=>cs.depletedPct>80);

  return (
    <div style={{background:T.bg,minHeight:"100vh",color:T.text,fontFamily:"'DM Mono',monospace",padding:"14px 16px"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:4px;}
        ::-webkit-scrollbar-thumb{background:#1a2535;border-radius:2px;}
        input[type=range]{accent-color:#00d4ff;}
        select{outline:none;}input{outline:none;}
      `}</style>

      {/* Header */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:14}}>
        <div>
          <div style={{fontSize:9,color:T.muted,letterSpacing:4,textTransform:"uppercase",marginBottom:2}}>
            Coupled solver · Multi-conjugate · Rochester Model
          </div>
          <h1 style={{fontFamily:"'Syne',sans-serif",fontSize:22,fontWeight:800,letterSpacing:-.5,
            background:`linear-gradient(110deg,${T.accent},#a78bfa)`,
            WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            LFA Simulator
          </h1>
        </div>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          {deplWarn&&(
            <div style={{background:"#3a1a00",border:"1px solid #f59e0b55",borderRadius:6,
              padding:"4px 10px",fontSize:9,color:T.warn}}>
              ⚡ Conjugate depletion detected
            </div>
          )}
          {[["sim","⚗ Simulation"],["sweep","⟳ Sweep"],["info","ƒ Model"],["report","📋 Report"]].map(([id,lbl])=>(
            <button key={id} onClick={()=>setTab(id)}
              style={{background:tab===id?"#7c3aed28":T.card,border:`1px solid ${tab===id?"#7c3aed":T.border}`,
                color:tab===id?"#c084fc":T.muted2,padding:"5px 12px",borderRadius:6,
                cursor:"pointer",fontFamily:"inherit",fontSize:11,transition:"all 0.15s"}}>
              {lbl}
            </button>
          ))}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"320px 1fr",gap:14}}>

        {/* LEFT PANEL */}
        <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14,
          display:"flex",flexDirection:"column",maxHeight:"calc(100vh - 80px)",overflowY:"auto"}}>

          {/* Sub-tabs: Kinetics removed */}
          <div style={{display:"flex",gap:3,marginBottom:14,background:T.surface,borderRadius:7,padding:3}}>
            {[["lines","Lines"],["ancj","A / Conj"],["physical","Physical"],["limits","Limits"]].map(([id,lbl])=>(
              <button key={id} onClick={()=>setLeftTab(id)}
                style={{flex:1,background:leftTab===id?"#1a2a45":T.surface,
                  border:`1px solid ${leftTab===id?T.borderHi:"transparent"}`,
                  color:leftTab===id?T.text:T.muted,padding:"4px 0",borderRadius:5,
                  cursor:"pointer",fontFamily:"inherit",fontSize:9,transition:"all 0.15s"}}>
                {lbl}
              </button>
            ))}
          </div>

          {leftTab==="lines"&&<LineBuilder lines={lines} setLines={setLines}
            analytes={analytes} conjugates={conjugates} physParams={physParams}/>}
          {leftTab==="ancj"&&<AnCjBuilder analytes={analytes} setAnalytes={setAnalytes}
            conjugates={conjugates} setConjugates={setConjugates}/>}
          {leftTab==="physical"&&<>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>
              Physical Parameters
            </div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:10,lineHeight:1.5}}>
              Includes diffusivities and flow constant — kinetics live on each conjugate / line card.
            </div>
            {PHYSICAL_META.map(m=>(
              <ParamRow key={m.key} meta={m} value={physParams[m.key]}
                onChange={v=>setPhysParams(p=>({...p,[m.key]:v}))}/>
            ))}
            <button onClick={()=>setPhysParams(defPhysical())}
              style={{marginTop:8,background:T.surface,border:`1px solid ${T.border}`,
                color:T.muted2,padding:"5px 0",borderRadius:6,cursor:"pointer",
                fontFamily:"inherit",fontSize:10,width:"100%"}}>
              Reset to Defaults
            </button>
          </>}

          {/* Limits tab — per-item slider upper bounds */}
          {leftTab==="limits"&&<>
            <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:4}}>
              Slider Upper Limits
            </div>
            <div style={{fontSize:9,color:T.muted2,marginBottom:12,lineHeight:1.5}}>
              Each <span style={{color:T.A}}>analyte</span> and <span style={{color:T.P}}>conjugate</span> carries
              its own upper bound. Enter a positive number above the slider's
              floor, then press Enter or click away. If you lower a cap below
              the current value, that value is clamped down. The new Ao max
              is a linear nM value — the slider's top decade is set to
              log₁₀ of it. Line Ro and fixedAg are already free-form number
              inputs (no max).
            </div>

            {/* Analytes section */}
            {analytes.length > 0 && <>
              <div style={{fontSize:9,color:T.A,letterSpacing:2,textTransform:"uppercase",marginBottom:6,marginTop:4}}>
                Analytes — A₀ caps
              </div>
              {analytes.map(a => {
                const meta = {
                  label: `${a.name}  [A₀]`,
                  unit:  "nM",
                  min:   AO_MAX_FLOOR,
                  def:   DEFAULT_AO_MAX,
                };
                const curMax = parseFloat(a.aoMax) || DEFAULT_AO_MAX;
                const preview = [parseFloat(a.Ao) || 0];
                return (
                  <LimitRow key={a.id} meta={meta}
                    currentMax={curMax}
                    currentValuesPreview={preview}
                    onChange={v => updateAnalyteCap(a.id, v)}/>
                );
              })}
            </>}

            {/* Conjugates section */}
            {conjugates.length > 0 && <>
              <div style={{fontSize:9,color:T.P,letterSpacing:2,textTransform:"uppercase",marginBottom:6,marginTop:10}}>
                Conjugates — P₀ caps
              </div>
              {conjugates.map(c => {
                const meta = {
                  label: `${c.name}  [P₀]`,
                  unit:  "nM",
                  min:   PO_MAX_FLOOR,
                  def:   DEFAULT_PO_MAX,
                };
                const curMax = parseFloat(c.poMax) || DEFAULT_PO_MAX;
                const preview = [parseFloat(c.Po) || 0];
                return (
                  <LimitRow key={c.id} meta={meta}
                    currentMax={curMax}
                    currentValuesPreview={preview}
                    onChange={v => updateConjugateCap(c.id, v)}/>
                );
              })}
            </>}

            {analytes.length === 0 && conjugates.length === 0 && (
              <div style={{color:T.muted,fontSize:10,textAlign:"center",padding:"12px 0"}}>
                No analytes or conjugates yet — add some in the A/Conj tab.
              </div>
            )}

            <button onClick={()=>{
                // Reset every analyte's aoMax and every conjugate's poMax
                // to defaults, clamping any current Ao/Po that sits above
                // the restored default.
                setAnalytes(prev =>
                  prev.map(a => {
                    const cur = parseFloat(a.Ao);
                    const Ao  = isFinite(cur) && cur > DEFAULT_AO_MAX ? DEFAULT_AO_MAX : a.Ao;
                    return { ...a, aoMax: DEFAULT_AO_MAX, Ao };
                  })
                );
                setConjugates(prev =>
                  prev.map(c => {
                    const cur = parseFloat(c.Po);
                    const Po  = isFinite(cur) && cur > DEFAULT_PO_MAX ? DEFAULT_PO_MAX : c.Po;
                    return { ...c, poMax: DEFAULT_PO_MAX, Po };
                  })
                );
              }}
              style={{marginTop:12,background:T.surface,border:`1px solid ${T.border}`,
                color:T.muted2,padding:"5px 0",borderRadius:6,cursor:"pointer",
                fontFamily:"inherit",fontSize:10,width:"100%"}}>
              Reset All Limits to Defaults
            </button>
          </>}

          {/* Geometry warnings */}
          {(()=>{
            const warns=[];
            lines.forEach(l=>{if(parseFloat(l.pos)>=physParams.strip_L)warns.push(`${l.name} (${l.pos}mm) outside strip (${physParams.strip_L}mm)`);});
            const ctrlFirst=lines.filter(l=>l.type!=="control").some(l=>{
              const ctrl=lines.find(c=>c.type==="control");
              return ctrl&&parseFloat(l.pos)>parseFloat(ctrl.pos);
            });
            if(ctrlFirst)warns.push("Control line should be the last (most downstream) line");
            return warns.length?(
              <div style={{background:"#1a0f08",border:"1px solid #7f3a1a",borderRadius:7,padding:"8px 10px",marginTop:8}}>
                {warns.map((w,i)=><div key={i} style={{color:"#fb923c",fontSize:9,lineHeight:1.6}}>⚠ {w}</div>)}
              </div>
            ):null;
          })()}

          {/* Flow front */}
          {simData&&(
            <div style={{marginTop:10,paddingTop:10,borderTop:`1px solid ${T.border}`}}>
              <div style={{fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:5}}>
                Flow Front · t={Math.round(curT)}s
              </div>
              {(()=>{
                const x4=21311*curT-69505;const xf=x4>0?x4**.25:0;
                const pct=Math.min(100,(xf/physParams.strip_L)*100);
                return <>
                  <div style={{fontSize:11,color:T.accent,marginBottom:4}}>{xf.toFixed(1)} mm · {pct.toFixed(0)}%</div>
                  <div style={{height:4,background:T.border,borderRadius:2}}>
                    <div style={{height:"100%",width:`${pct}%`,borderRadius:2,background:`linear-gradient(90deg,${T.accent}55,${T.accent})`}}/>
                  </div>
                </>;
              })()}
              <div style={{marginTop:10,fontSize:9,color:T.muted,letterSpacing:3,textTransform:"uppercase",marginBottom:5}}>Species</div>
              {[{c:T.A,l:"[A] Analyte"},{c:T.P,l:"[P] Conjugate (shared/group)"},{c:T.PA,l:"[PA] Complex"},
                ...Object.entries(LINE_COLORS).map(([k,c])=>({c,l:`signal (${k})`}))
              ].slice(0,6).map(s=>(
                <div key={s.l} style={{display:"flex",alignItems:"center",gap:7,marginBottom:5}}>
                  <div style={{width:16,height:2,borderRadius:1,background:s.c,boxShadow:`0 0 4px ${s.c}`}}/>
                  <span style={{fontSize:9,color:T.muted2}}>{s.l}</span>
                </div>
              ))}
            </div>
          )}

          {/* Run button */}
          <button onClick={handleRun} disabled={running}
            style={{width:"100%",padding:"9px 0",marginTop:14,
              background:running?T.border:"linear-gradient(135deg,#0ea5e9,#7c3aed)",
              border:"none",borderRadius:7,color:"#fff",cursor:running?"not-allowed":"pointer",
              fontFamily:"inherit",fontSize:12,fontWeight:500,
              boxShadow:running?"none":"0 0 16px rgba(14,165,233,0.25)",transition:"all 0.2s"}}>
            {running?"⟳  Computing…":"▶  Run Simulation"}
          </button>
        </div>

        {/* RIGHT PANEL */}
        <div style={{display:"flex",flexDirection:"column",gap:12}}>
          <div style={{display:"flex",gap:8}}>
            <Stat label="Time"      value={`${Math.round(curT)} s`}  unit=""   color={T.accent}/>
            <Stat label="Test sig." value={curSig.toExponential(2)}  unit="nM" color={T.RPA}/>
            <Stat label="Ctrl sig." value={curCtrl.toExponential(2)} unit="nM" color={T.PC}/>
            <Stat label="T/R ratio" value={TC?TC.toFixed(3):"—"}     unit=""   color={T.PA}/>
            <Stat label="Conj. groups" value={String(conjugates.length)} unit="" color={T.P}/>
          </div>

          {tab==="sim"&&<>
            <div style={{background:T.card,border:`1px solid ${T.border2}`,borderRadius:12,padding:14}}>
              <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:10}}>
                <div style={{fontSize:11}}>
                  <span style={{color:T.muted2}}>Membrane strip — </span>
                  <span style={{color:T.accent}}>t = {Math.round(curT)} s</span>
                  <span style={{color:T.muted,fontSize:9,marginLeft:8}}>
                    {lines.length} lines · {conjugates.length} conjugate group(s) · coupled solver
                  </span>
                </div>
                <div style={{display:"flex",gap:5}}>
                  {[["⏮",()=>{setFrameIdx(0);setPlaying(false);}],
                    [playing?"⏸":"▶",()=>setPlaying(p=>!p)],
                    ["⏭",()=>{setFrameIdx(simData?simData.times.length-1:0);setPlaying(false);}]
                  ].map(([icon,fn],i)=>(
                    <button key={i} onClick={fn}
                      style={{background:T.surface,border:`1px solid ${i===1?T.accent+"44":T.border}`,
                        color:i===1?T.accent:T.muted2,padding:"4px 10px",borderRadius:5,
                        cursor:"pointer",fontFamily:"inherit",fontSize:12}}>
                      {icon}
                    </button>
                  ))}
                </div>
              </div>
              <LFAStrip simData={simData} frameIdx={fi} lines={lines} physParams={physParams}/>
              {simData&&(
                <div style={{marginTop:8}}>
                  <input type="range" min={0} max={simData.times.length-1} value={fi}
                    onChange={e=>{setPlaying(false);setFrameIdx(Number(e.target.value));}}
                    style={{width:"100%",cursor:"pointer",height:4}}/>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:T.muted,marginTop:3}}>
                    <span>0 s</span>
                    <span style={{color:T.muted2}}>← drag to scrub · ▶ to play →</span>
                    <span>{physParams.t_end} s</span>
                  </div>
                </div>
              )}
            </div>

            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>
              <div style={{fontSize:11,color:T.muted2,marginBottom:8}}>
                Line signals vs time — <span style={{color:T.muted}}>click to seek</span>
              </div>
              {simData
                ?<SignalChart simData={simData} frameIdx={fi} setFrameIdx={setFrameIdx}/>
                :<div style={{height:170,display:"flex",alignItems:"center",justifyContent:"center"}}>
                  <span style={{color:T.muted,fontSize:11}}>Run simulation first</span>
                </div>}
            </div>

            {simData&&<div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:14}}>
              <ResultsTable simData={simData}/>
            </div>}

            {simData&&<MassConservationPanel simData={simData} conjugates={conjugates}/>}
          </>}

          {tab==="sweep"&&<SweepPanel analytes={analytes} conjugates={conjugates} lines={lines} physParams={physParams}/>}
          {tab==="info"&&<ModelInfo/>}
          {tab==="report"&&(
            <div style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,padding:16}}>
              <SimReport simData={simData} analytes={analytes} conjugates={conjugates}
                lines={lines} physParams={physParams}/>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
