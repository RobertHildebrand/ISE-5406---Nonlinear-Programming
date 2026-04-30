import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Play, Pause, RotateCcw, StepForward, CheckCircle2 } from "lucide-react";

/* ============================================================
   SGD / MOMENTUM / ADAM — IN-CLASS CODE STEPPER
   ISE 5406 (Nonlinear Programming)

   Pick an algorithm tab. Click "Step" to advance the highlighted
   code line. The 2D contour shows the loss surface; the trajectory
   updates each time the active line modifies w. The state panel
   below the plot shows g, m, v, m̂, v̂, and the effective step
   size — the actual numbers the active line just computed.

   Designed for live in-class demonstration — instructor advances
   the lecture without typing.
   ============================================================ */

// ---------------- RNG / math helpers ----------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randn = (rng) => {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const sigmoid = (z) => 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, z))));

// ---------------- Dataset (2D logistic regression) ----------------
function makeData(seed = 2, n = 40) {
  const rng = mulberry32(seed);
  const wTrue = [1.4, 0.6];
  const data = [];
  for (let i = 0; i < n; i++) {
    const x = [1.4 * randn(rng), 1.4 * randn(rng)];
    const z = wTrue[0] * x[0] + wTrue[1] * x[1];
    const yClean = z >= 0 ? 1 : -1;
    // 8% label flip → optimum is interior, slightly offset from wTrue
    const y = rng() < 0.08 ? -yClean : yClean;
    data.push({ x, y });
  }
  return { data, wTrue };
}

function lossOver(w, batch) {
  let s = 0;
  for (const p of batch) {
    const z = p.y * (w[0] * p.x[0] + w[1] * p.x[1]);
    s += Math.log(1 + Math.exp(-Math.max(-50, Math.min(50, z))));
  }
  return s / batch.length;
}
function gradOver(w, batch) {
  let g0 = 0,
    g1 = 0;
  for (const p of batch) {
    const z = p.y * (w[0] * p.x[0] + w[1] * p.x[1]);
    const c = -p.y * sigmoid(-z);
    g0 += c * p.x[0];
    g1 += c * p.x[1];
  }
  return [g0 / batch.length, g1 / batch.length];
}

// Deterministic mini-batch index list per step.
function makeBatchSampler(n, batchSize, seed) {
  return (step) => {
    const rng = mulberry32(seed * 1000003 + step + 1);
    const taken = new Set();
    const idx = [];
    while (idx.length < Math.min(batchSize, n)) {
      const i = Math.floor(rng() * n);
      if (!taken.has(i)) {
        taken.add(i);
        idx.push(i);
      }
    }
    return idx;
  };
}

// ---------------- Pseudocode strings ----------------
// We pre-pad with a leading "0:" sentinel so 1-based indexing into
// CODE arrays is natural (CODE_X[1] is line 1).
const CODE_SGD = [
  null,
  "w = [0.0, 0.0]",
  "for t in range(T):",
  "  batch = sample_batch(data)",
  "  g = grad_loss(w, batch)",
  "  w = w - lr * g",
];
const CODE_MOMENTUM = [
  null,
  "w = [0.0, 0.0]",
  "v = [0.0, 0.0]",
  "for t in range(T):",
  "  batch = sample_batch(data)",
  "  g = grad_loss(w, batch)",
  "  v = beta * v + g",
  "  w = w - lr * v",
];
const CODE_ADAM = [
  null,
  "w = [0.0, 0.0]",
  "m = [0.0, 0.0]",
  "v = [0.0, 0.0]",
  "t = 0",
  "for step in range(T):",
  "  t += 1",
  "  batch = sample_batch(data)",
  "  g = grad_loss(w, batch)",
  "  m = b1 * m + (1 - b1) * g",
  "  v = b2 * v + (1 - b2) * g**2",
  "  m_hat = m / (1 - b1**t)",
  "  v_hat = v / (1 - b2**t)",
  "  w = w - lr * m_hat / (sqrt(v_hat) + eps)",
];

// ---------------- Event planners ----------------
// Each event = { kind, line, ... }. `kind` drives the state-panel
// highlights and any animation; `line` is the 1-based code line.

function planSGD({ data, sampler, hp, maxSteps, w0 }) {
  const events = [];
  let w = [...w0];
  events.push({ kind: "init-w", line: 1, w: [...w] });
  for (let step = 0; step < maxSteps; step++) {
    events.push({ kind: "loop", line: 2, step });
    const idx = sampler(step);
    const batch = idx.map((i) => data[i]);
    events.push({ kind: "sample", line: 3, step, idx });
    const g = gradOver(w, batch);
    events.push({ kind: "grad", line: 4, step, idx, g });
    const newW = [w[0] - hp.lr * g[0], w[1] - hp.lr * g[1]];
    events.push({
      kind: "update-w",
      line: 5,
      step,
      idx,
      g,
      fromW: [...w],
      toW: newW,
    });
    w = newW;
  }
  events.push({ kind: "done", line: 2 });
  return events;
}

function planMomentum({ data, sampler, hp, maxSteps, w0 }) {
  const events = [];
  let w = [...w0];
  let v = [0, 0];
  events.push({ kind: "init-w", line: 1, w: [...w] });
  events.push({ kind: "init-v", line: 2, v: [...v] });
  for (let step = 0; step < maxSteps; step++) {
    events.push({ kind: "loop", line: 3, step });
    const idx = sampler(step);
    const batch = idx.map((i) => data[i]);
    events.push({ kind: "sample", line: 4, step, idx });
    const g = gradOver(w, batch);
    events.push({ kind: "grad", line: 5, step, idx, g });
    const newV = [hp.beta * v[0] + g[0], hp.beta * v[1] + g[1]];
    events.push({
      kind: "update-v",
      line: 6,
      step,
      idx,
      g,
      fromV: [...v],
      toV: newV,
    });
    v = newV;
    const newW = [w[0] - hp.lr * v[0], w[1] - hp.lr * v[1]];
    events.push({
      kind: "update-w",
      line: 7,
      step,
      idx,
      g,
      v: [...v],
      fromW: [...w],
      toW: newW,
    });
    w = newW;
  }
  events.push({ kind: "done", line: 3 });
  return events;
}

function planAdam({ data, sampler, hp, maxSteps, w0 }) {
  const events = [];
  let w = [...w0];
  let m = [0, 0];
  let v = [0, 0];
  let t = 0;
  events.push({ kind: "init-w", line: 1, w: [...w] });
  events.push({ kind: "init-m", line: 2, m: [...m] });
  events.push({ kind: "init-v", line: 3, v: [...v] });
  events.push({ kind: "init-t", line: 4, t });
  for (let step = 0; step < maxSteps; step++) {
    events.push({ kind: "loop", line: 5, step });
    t += 1;
    events.push({ kind: "t-inc", line: 6, step, t });
    const idx = sampler(step);
    const batch = idx.map((i) => data[i]);
    events.push({ kind: "sample", line: 7, step, idx });
    const g = gradOver(w, batch);
    events.push({ kind: "grad", line: 8, step, idx, g });
    const newM = [hp.b1 * m[0] + (1 - hp.b1) * g[0], hp.b1 * m[1] + (1 - hp.b1) * g[1]];
    events.push({
      kind: "update-m",
      line: 9,
      step,
      g,
      fromM: [...m],
      toM: newM,
    });
    m = newM;
    const newV = [
      hp.b2 * v[0] + (1 - hp.b2) * g[0] * g[0],
      hp.b2 * v[1] + (1 - hp.b2) * g[1] * g[1],
    ];
    events.push({
      kind: "update-v",
      line: 10,
      step,
      g,
      fromV: [...v],
      toV: newV,
    });
    v = newV;
    const bcM = 1 - Math.pow(hp.b1, t);
    const bcV = 1 - Math.pow(hp.b2, t);
    const mHat = [m[0] / bcM, m[1] / bcM];
    events.push({ kind: "mhat", line: 11, step, t, mHat });
    const vHat = [v[0] / bcV, v[1] / bcV];
    events.push({ kind: "vhat", line: 12, step, t, vHat });
    const stepSize = [
      (hp.lr * mHat[0]) / (Math.sqrt(vHat[0]) + hp.eps),
      (hp.lr * mHat[1]) / (Math.sqrt(vHat[1]) + hp.eps),
    ];
    const newW = [w[0] - stepSize[0], w[1] - stepSize[1]];
    events.push({
      kind: "update-w",
      line: 13,
      step,
      g,
      mHat,
      vHat,
      effStep: stepSize,
      fromW: [...w],
      toW: newW,
    });
    w = newW;
  }
  events.push({ kind: "done", line: 5 });
  return events;
}

// ---------------- Replay helpers ----------------
// Reconstruct (w, m, v, t) at the end of event idx.
function replayState(events, idx) {
  let w = [0, 0];
  let m = [0, 0];
  let v = [0, 0];
  let t = 0;
  let lastG = null;
  let lastIdx = null;
  let lastEffStep = null;
  let lastStep = -1;
  for (let k = 0; k <= Math.min(idx, events.length - 1); k++) {
    const ev = events[k];
    if (!ev) break;
    switch (ev.kind) {
      case "init-w":
        w = [...ev.w];
        break;
      case "init-m":
        m = [...ev.m];
        break;
      case "init-v":
        v = [...ev.v];
        break;
      case "init-t":
        t = ev.t;
        break;
      case "t-inc":
        t = ev.t;
        break;
      case "sample":
        lastIdx = ev.idx;
        lastStep = ev.step;
        break;
      case "grad":
        lastG = ev.g;
        break;
      case "update-m":
        m = [...ev.toM];
        break;
      case "update-v":
        v = [...ev.toV];
        break;
      case "update-w":
        w = [...ev.toW];
        if (ev.effStep) lastEffStep = ev.effStep;
        break;
      default:
        break;
    }
  }
  return { w, m, v, t, lastG, lastIdx, lastStep, lastEffStep };
}

// Trajectory of w values after each "update-w" event up to idx.
function trajectoryUpTo(events, idx) {
  const pts = [];
  let w0 = null;
  for (let k = 0; k <= Math.min(idx, events.length - 1); k++) {
    const ev = events[k];
    if (ev.kind === "init-w") {
      w0 = ev.w;
      pts.push([...w0]);
    } else if (ev.kind === "update-w") {
      pts.push([...ev.toW]);
    }
  }
  return pts;
}

// ---------------- Plot ----------------
const PLOT = { size: 380, lo: -1.5, hi: 3.5, loY: -1.5, hiY: 3.0 };
const dx = (x) => ((x - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const dy = (y) => PLOT.size - ((y - PLOT.loY) / (PLOT.hiY - PLOT.loY)) * PLOT.size;

// Marching-squares contours
function buildContour(fGrid, levels, nx, ny, xStart, yStart, xStep, yStep) {
  const segs = [];
  const interp = (a, b) => a / (a - b);
  for (const L of levels) {
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const v00 = fGrid[i * (ny + 1) + j] - L;
        const v10 = fGrid[(i + 1) * (ny + 1) + j] - L;
        const v01 = fGrid[i * (ny + 1) + (j + 1)] - L;
        const v11 = fGrid[(i + 1) * (ny + 1) + (j + 1)] - L;
        let idx = 0;
        if (v00 > 0) idx |= 1;
        if (v10 > 0) idx |= 2;
        if (v11 > 0) idx |= 4;
        if (v01 > 0) idx |= 8;
        if (idx === 0 || idx === 15) continue;
        const x0 = xStart + i * xStep;
        const x1 = xStart + (i + 1) * xStep;
        const y0 = yStart + j * yStep;
        const y1 = yStart + (j + 1) * yStep;
        const e = [];
        if ((v00 > 0) !== (v10 > 0)) e.push([x0 + xStep * interp(v00, v10), y0]);
        if ((v10 > 0) !== (v11 > 0)) e.push([x1, y0 + yStep * interp(v10, v11)]);
        if ((v01 > 0) !== (v11 > 0)) e.push([x0 + xStep * interp(v01, v11), y1]);
        if ((v00 > 0) !== (v01 > 0)) e.push([x0, y0 + yStep * interp(v00, v01)]);
        if (e.length >= 2) segs.push([e[0], e[1], L]);
        if (e.length === 4) segs.push([e[2], e[3], L]);
      }
    }
  }
  return segs;
}

// Pre-compute the loss grid + contour segments for the *full* dataset.
function useLossSurface(data) {
  return useMemo(() => {
    const N = 60;
    const xStep = (PLOT.hi - PLOT.lo) / N;
    const yStep = (PLOT.hiY - PLOT.loY) / N;
    const grid = new Float64Array((N + 1) * (N + 1));
    let mn = Infinity,
      mx = -Infinity;
    for (let i = 0; i <= N; i++) {
      const wx = PLOT.lo + i * xStep;
      for (let j = 0; j <= N; j++) {
        const wy = PLOT.loY + j * yStep;
        const L = lossOver([wx, wy], data);
        grid[i * (N + 1) + j] = L;
        if (L < mn) mn = L;
        if (L > mx) mx = L;
      }
    }
    const levels = [];
    for (let k = 1; k <= 8; k++) {
      levels.push(mn + ((mx - mn) * k) / 9);
    }
    const segs = buildContour(grid, levels, N, N, PLOT.lo, PLOT.loY, xStep, yStep);
    // Convert data segs to pixel segs.
    const pxSegs = segs.map(([a, b, L]) => [
      [dx(a[0]), dy(a[1])],
      [dx(b[0]), dy(b[1])],
      L,
    ]);
    return { grid, levels, pxSegs, mn, mx };
  }, [data]);
}

function ContourPlot({ data, surface, trajectory, currentW, batchIdx, wTrue }) {
  return (
    <svg
      width={PLOT.size}
      height={PLOT.size}
      style={{ background: "#fafafa", borderRadius: 6, border: "1px solid #eee" }}
    >
      {/* faint axes */}
      <line x1={dx(0)} y1={0} x2={dx(0)} y2={PLOT.size} stroke="#ddd" />
      <line x1={0} y1={dy(0)} x2={PLOT.size} y2={dy(0)} stroke="#ddd" />

      {/* contour lines */}
      {surface.pxSegs.map((s, k) => {
        const tNorm = (s[2] - surface.mn) / (surface.mx - surface.mn || 1);
        const grey = Math.floor(220 - 100 * tNorm);
        return (
          <line
            key={k}
            x1={s[0][0]}
            y1={s[0][1]}
            x2={s[1][0]}
            y2={s[1][1]}
            stroke={`rgb(${grey},${grey},${grey})`}
            strokeWidth={0.9}
          />
        );
      })}

      {/* trajectory */}
      {trajectory.length > 1 && (
        <polyline
          fill="none"
          stroke="#c8311c"
          strokeWidth={1.6}
          points={trajectory.map((p) => `${dx(p[0])},${dy(p[1])}`).join(" ")}
        />
      )}
      {/* trajectory dots */}
      {trajectory.slice(0, -1).map((p, k) => (
        <circle
          key={k}
          cx={dx(p[0])}
          cy={dy(p[1])}
          r={2}
          fill="#c8311c"
          opacity={0.5}
        />
      ))}

      {/* true w (asterisk) */}
      <g>
        <line
          x1={dx(wTrue[0]) - 7}
          y1={dy(wTrue[1])}
          x2={dx(wTrue[0]) + 7}
          y2={dy(wTrue[1])}
          stroke="#1f4e3d"
          strokeWidth={2}
        />
        <line
          x1={dx(wTrue[0])}
          y1={dy(wTrue[1]) - 7}
          x2={dx(wTrue[0])}
          y2={dy(wTrue[1]) + 7}
          stroke="#1f4e3d"
          strokeWidth={2}
        />
        <text
          x={dx(wTrue[0]) + 10}
          y={dy(wTrue[1]) - 6}
          fontSize={11}
          fill="#1f4e3d"
          fontFamily="monospace"
        >
          w*
        </text>
      </g>

      {/* current w */}
      <circle
        cx={dx(currentW[0])}
        cy={dy(currentW[1])}
        r={6}
        fill="#c8311c"
        stroke="#fff"
        strokeWidth={2}
      />

      {/* axis labels */}
      <text x={PLOT.size - 30} y={dy(0) - 6} fontSize={11} fill="#888" fontFamily="monospace">
        w₀
      </text>
      <text x={dx(0) + 6} y={14} fontSize={11} fill="#888" fontFamily="monospace">
        w₁
      </text>
    </svg>
  );
}

// ---------------- Code panel ----------------
function CodePanel({ codeLines, highlightedLine }) {
  const lineHeight = 22;
  return (
    <div
      style={{
        fontFamily:
          "'JetBrains Mono', Menlo, ui-monospace, monospace",
        fontSize: 13,
        background: "#1f1d1a",
        color: "#e8e2d4",
        padding: "12px 0",
        borderRadius: 8,
        overflow: "hidden",
        lineHeight: `${lineHeight}px`,
        minHeight: codeLines.length * lineHeight + 24,
      }}
    >
      {codeLines.map((line, i) => {
        if (i === 0) return null; // sentinel
        const active = i === highlightedLine;
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              background: active ? "#3b3526" : "transparent",
              borderLeft: active ? "3px solid #f5a524" : "3px solid transparent",
            }}
          >
            <span
              style={{
                width: 28,
                color: active ? "#f5a524" : "#7f7864",
                fontSize: 11,
                userSelect: "none",
              }}
            >
              {active ? "▶" : ""}
            </span>
            <span
              style={{
                width: 24,
                color: "#7f7864",
                textAlign: "right",
                marginRight: 14,
                fontSize: 11,
                userSelect: "none",
              }}
            >
              {i}
            </span>
            <span
              style={{
                color: active ? "#fff8e1" : "#e8e2d4",
                whiteSpace: "pre",
              }}
            >
              {line}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------- Stats panel ----------------
function Vec2({ value, label, color }) {
  if (!value) {
    return (
      <div style={statRow}>
        <span style={statLabel}>{label}</span>
        <span style={{ ...statValue, color: "#999" }}>—</span>
      </div>
    );
  }
  return (
    <div style={statRow}>
      <span style={statLabel}>{label}</span>
      <span style={{ ...statValue, color: color || "#111" }}>
        ({value[0].toFixed(3)}, {value[1].toFixed(3)})
      </span>
    </div>
  );
}
function Scalar({ value, label, format }) {
  return (
    <div style={statRow}>
      <span style={statLabel}>{label}</span>
      <span style={statValue}>{format ? format(value) : String(value)}</span>
    </div>
  );
}

const statRow = { display: "flex", justifyContent: "space-between", padding: "3px 0" };
const statLabel = { color: "#666", fontFamily: "monospace", fontSize: 12 };
const statValue = { fontFamily: "monospace", fontSize: 12, fontWeight: 600 };

// ---------------- Algorithm registry ----------------
const ALGORITHMS = {
  sgd: {
    name: "SGD",
    code: CODE_SGD,
    plan: planSGD,
    pyTorch: `opt = torch.optim.SGD(model.parameters(), lr=0.1)
for epoch in range(T):
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()`,
    defaultHp: { lr: 0.5 },
    hpControls: [
      { key: "lr", label: "lr", min: 0.01, max: 2.0, step: 0.01 },
    ],
  },
  momentum: {
    name: "SGD + Momentum",
    code: CODE_MOMENTUM,
    plan: planMomentum,
    pyTorch: `opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for epoch in range(T):
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()`,
    defaultHp: { lr: 0.1, beta: 0.9 },
    hpControls: [
      { key: "lr", label: "lr", min: 0.01, max: 1.0, step: 0.01 },
      { key: "beta", label: "β", min: 0, max: 0.99, step: 0.01 },
    ],
  },
  adam: {
    name: "Adam",
    code: CODE_ADAM,
    plan: planAdam,
    pyTorch: `opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(T):
    for x, y in loader:
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()`,
    defaultHp: { lr: 0.2, b1: 0.9, b2: 0.999, eps: 1e-8 },
    hpControls: [
      { key: "lr", label: "lr", min: 0.01, max: 1.0, step: 0.01 },
      { key: "b1", label: "β₁", min: 0, max: 0.999, step: 0.001 },
      { key: "b2", label: "β₂", min: 0, max: 0.9999, step: 0.0001 },
    ],
  },
};

// ---------------- Per-algorithm narration ----------------
function narrate(ev, alg) {
  if (!ev) return "";
  switch (ev.kind) {
    case "init-w":
      return "Initialise the parameter vector w to zeros.";
    case "init-m":
      return "Initialise the first-moment buffer m (running mean of the gradient).";
    case "init-v":
      return alg === "adam"
        ? "Initialise the second-moment buffer v (running mean of g²)."
        : "Initialise the velocity buffer v.";
    case "init-t":
      return "Initialise the step counter t = 0. Adam uses t for bias correction.";
    case "loop":
      return `Top of the training loop. Step ${ev.step + 1}.`;
    case "t-inc":
      return `Increment t. t = ${ev.t}. The bias-correction divisors below depend on this.`;
    case "sample":
      return `Sample a mini-batch (indices ${ev.idx.join(", ")}). Smaller batch → noisier g.`;
    case "grad":
      return `Compute g = ∇loss(w; batch) on this mini-batch. g = (${ev.g[0].toFixed(3)}, ${ev.g[1].toFixed(3)}).`;
    case "update-m":
      return `EMA of g: m ← β₁·m + (1−β₁)·g. New m = (${ev.toM[0].toFixed(3)}, ${ev.toM[1].toFixed(3)}).`;
    case "update-v":
      if (alg === "momentum")
        return `Velocity update v ← β·v + g. New v = (${ev.toV[0].toFixed(3)}, ${ev.toV[1].toFixed(3)}).`;
      return `EMA of g²: v ← β₂·v + (1−β₂)·g². New v = (${ev.toV[0].toFixed(4)}, ${ev.toV[1].toFixed(4)}).`;
    case "mhat":
      return `Bias-correct m: m̂ = m / (1 − β₁ᵗ). Without this, m would be biased toward 0 in the first few steps.`;
    case "vhat":
      return `Bias-correct v: v̂ = v / (1 − β₂ᵗ). Same reason — early-iteration bias correction.`;
    case "update-w":
      if (alg === "adam")
        return `Adaptive step: w ← w − lr · m̂ / (√v̂ + ε). Effective per-coordinate step size = (${ev.effStep[0].toFixed(4)}, ${ev.effStep[1].toFixed(4)}).`;
      if (alg === "momentum")
        return `Apply velocity: w ← w − lr · v.`;
      return `Take a gradient step: w ← w − lr · g.`;
    case "done":
      return "Training loop finished — out of budget.";
    default:
      return "";
  }
}

// ---------------- Main component ----------------
export default function SGDAdamTutorial() {
  const [algKey, setAlgKey] = useState("sgd");
  const [batchSize, setBatchSize] = useState(4);
  const [maxSteps, setMaxSteps] = useState(40);
  const [seed, setSeed] = useState(2);
  const [hp, setHp] = useState(ALGORITHMS.sgd.defaultHp);
  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(280); // ms per step

  // Dataset and surface — recompute when seed changes.
  const { data, wTrue } = useMemo(() => makeData(seed, 40), [seed]);
  const surface = useLossSurface(data);

  // Choose w0 ~ same starting corner for all algorithms.
  const w0 = useMemo(() => [-0.8, 2.5], []);

  // When the algorithm changes, snap hp to that algorithm's defaults.
  useEffect(() => {
    setHp(ALGORITHMS[algKey].defaultHp);
    setEvIdx(0);
    setRunning(false);
  }, [algKey]);

  // (Re)plan events whenever knobs change.
  const events = useMemo(() => {
    const sampler = makeBatchSampler(data.length, batchSize, seed);
    return ALGORITHMS[algKey].plan({
      data,
      sampler,
      hp,
      maxSteps,
      w0,
    });
  }, [data, algKey, hp, batchSize, maxSteps, seed, w0]);

  // Reset evIdx if it falls past the new event list.
  useEffect(() => {
    if (evIdx >= events.length) setEvIdx(events.length - 1);
  }, [events]); // eslint-disable-line react-hooks/exhaustive-deps

  // Auto-run.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setEvIdx((k) => {
        if (k + 1 >= events.length) {
          setRunning(false);
          return k;
        }
        return k + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, events.length]);

  const stepOnce = useCallback(() => {
    setEvIdx((k) => Math.min(events.length - 1, k + 1));
  }, [events.length]);

  const reset = useCallback(() => {
    setEvIdx(0);
    setRunning(false);
  }, []);

  const ev = events[evIdx];
  const state = useMemo(() => replayState(events, evIdx), [events, evIdx]);
  const trajectory = useMemo(() => trajectoryUpTo(events, evIdx), [events, evIdx]);
  const currentW = state.w;
  const currentLoss = lossOver(currentW, data);

  const alg = ALGORITHMS[algKey];
  const code = alg.code;

  return (
    <div style={{ maxWidth: 1180, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        SGD &amp; Adam — Code-Stepper Tutorial
      </h1>
      <p style={{ color: "#666", marginBottom: 22, maxWidth: 880 }}>
        Click the code line you want to teach, or just press <b>Step</b>. The
        active line is highlighted on the left and its effect is shown on the
        right: trajectory on the loss surface, plus the actual values of{" "}
        <code style={codeInline}>g</code>, <code style={codeInline}>m</code>,{" "}
        <code style={codeInline}>v</code>,{" "}
        <code style={codeInline}>m̂</code>, <code style={codeInline}>v̂</code>,
        and the effective per-coordinate step size.
      </p>

      {/* Algorithm tabs */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {Object.entries(ALGORITHMS).map(([k, a]) => (
          <button
            key={k}
            onClick={() => setAlgKey(k)}
            style={{
              ...tabBtn,
              ...(k === algKey ? tabBtnActive : {}),
            }}
          >
            {a.name}
          </button>
        ))}
      </div>

      {/* Main content: code | plot+stats */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(380px, 1fr) minmax(420px, 1fr)",
          gap: 24,
          alignItems: "flex-start",
        }}
      >
        {/* LEFT: code + narration */}
        <div>
          <CodePanel codeLines={code} highlightedLine={ev?.line || 1} />

          <div style={narrationBox}>
            <div style={{ fontSize: 12, color: "#777", marginBottom: 4, fontFamily: "monospace" }}>
              what this line does
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.5 }}>
              {narrate(ev, algKey)}
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            <button onClick={stepOnce} disabled={evIdx >= events.length - 1} style={btnPrimary}>
              <StepForward size={16} /> Step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={evIdx >= events.length - 1}
              style={btn}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={reset} style={btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={{ ...controlGroup, marginTop: 14 }}>
            <label style={label}>
              run speed (ms/step): <b>{speed}</b>
            </label>
            <input
              type="range"
              min={60}
              max={800}
              step={20}
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
              style={{ width: "100%" }}
            />
          </div>

          {/* Progress bar */}
          <div
            style={{
              marginTop: 12,
              height: 6,
              background: "#eee",
              borderRadius: 3,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                background: "#1f4e3d",
                width: `${(100 * (evIdx + 1)) / events.length}%`,
                transition: "width 0.1s",
              }}
            />
          </div>
          <div
            style={{
              fontSize: 11,
              color: "#888",
              fontFamily: "monospace",
              marginTop: 4,
            }}
          >
            event {evIdx + 1} / {events.length}
          </div>
        </div>

        {/* RIGHT: plot + stats */}
        <div>
          <ContourPlot
            data={data}
            surface={surface}
            trajectory={trajectory}
            currentW={currentW}
            wTrue={wTrue}
            batchIdx={state.lastIdx}
          />

          <div style={statBox}>
            <Vec2 label="w (current)" value={currentW} color="#c8311c" />
            <Scalar
              label="loss(w) (full data)"
              value={currentLoss}
              format={(x) => x.toFixed(4)}
            />
            <Vec2 label="g (mini-batch)" value={state.lastG} color="#0066aa" />
            {(algKey === "momentum" || algKey === "adam") && (
              <Vec2 label={algKey === "momentum" ? "v" : "m"} value={state.m.length ? state.m : null} />
            )}
            {algKey === "momentum" && <Vec2 label="" value={null} />}
            {algKey === "adam" && (
              <>
                <Vec2 label="v" value={state.v} />
                <Scalar label="t" value={state.t} />
                <Vec2
                  label="lr · m̂ / (√v̂ + ε)"
                  value={state.lastEffStep}
                  color="#1f4e3d"
                />
              </>
            )}
            {algKey === "momentum" && (
              <Vec2 label="v" value={state.v} />
            )}
            {state.lastStep != null && state.lastStep >= 0 && (
              <Scalar
                label="step #"
                value={state.lastStep + 1}
              />
            )}
            {state.lastIdx && (
              <Scalar
                label="batch indices"
                value={state.lastIdx.join(", ")}
              />
            )}
          </div>
        </div>
      </div>

      {/* HP and dataset controls */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 14,
          marginTop: 28,
          padding: 16,
          background: "#fafafa",
          borderRadius: 10,
          border: "1px solid #eee",
        }}
      >
        {alg.hpControls.map((c) => (
          <div key={c.key}>
            <label style={label}>
              {c.label}: <b>{(+hp[c.key]).toFixed(c.step < 0.01 ? 4 : c.step < 1 ? 3 : 0)}</b>
            </label>
            <input
              type="range"
              min={c.min}
              max={c.max}
              step={c.step}
              value={hp[c.key]}
              onChange={(e) => {
                setHp((p) => ({ ...p, [c.key]: +e.target.value }));
                setEvIdx(0);
                setRunning(false);
              }}
              style={{ width: "100%" }}
            />
          </div>
        ))}
        <div>
          <label style={label}>
            batch size: <b>{batchSize}</b>
          </label>
          <input
            type="range"
            min={1}
            max={Math.min(40, data.length)}
            step={1}
            value={batchSize}
            onChange={(e) => {
              setBatchSize(+e.target.value);
              setEvIdx(0);
              setRunning(false);
            }}
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <label style={label}>
            T (steps): <b>{maxSteps}</b>
          </label>
          <input
            type="range"
            min={5}
            max={120}
            step={1}
            value={maxSteps}
            onChange={(e) => {
              setMaxSteps(+e.target.value);
              setEvIdx(0);
              setRunning(false);
            }}
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <label style={label}>
            data seed: <b>{seed}</b>
          </label>
          <input
            type="range"
            min={1}
            max={20}
            step={1}
            value={seed}
            onChange={(e) => {
              setSeed(+e.target.value);
              setEvIdx(0);
              setRunning(false);
            }}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      {/* PyTorch comparison */}
      <div style={{ marginTop: 28 }}>
        <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>
          The same algorithm in PyTorch
        </h2>
        <p style={{ color: "#666", fontSize: 14, marginBottom: 10 }}>
          Everything you just stepped through is hidden inside{" "}
          <code style={codeInline}>opt.step()</code>. The PyTorch optimizer
          stores the buffers (m, v, t) as attributes of the optimizer object.
        </p>
        <pre
          style={{
            background: "#1f1d1a",
            color: "#e8e2d4",
            padding: "14px 18px",
            borderRadius: 8,
            fontSize: 13,
            fontFamily: "'JetBrains Mono', Menlo, monospace",
            lineHeight: 1.6,
            overflowX: "auto",
          }}
        >
          {alg.pyTorch}
        </pre>
      </div>

      {/* Pedagogical notes */}
      <div
        style={{
          marginTop: 24,
          padding: 16,
          background: "#fff8e1",
          borderRadius: 10,
          border: "1px solid #f5d68d",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 6 }}>Try this in lecture</div>
        <ul style={{ margin: 0, paddingLeft: 22, lineHeight: 1.6, fontSize: 14 }}>
          <li>
            Set <b>batch size = 1</b> — watch g jitter wildly between mini-batches.
            That's the variance term in the SGD analysis.
          </li>
          <li>
            On the <b>SGD</b> tab, push lr to ~1.5 — students see oscillation /
            divergence. Drop to ~0.3 and they see convergence. (Robbins-Monro
            says "decreasing step sizes" — this is why.)
          </li>
          <li>
            On <b>Momentum</b>, set β = 0 → identical to SGD. Crank β to 0.95
            and the velocity buffer carries through gradient noise.
          </li>
          <li>
            On <b>Adam</b>, the early steps show why bias correction matters:
            without it, m̂ ≈ m would be near zero and the first updates would
            be tiny.
          </li>
        </ul>
      </div>
    </div>
  );
}

// ---------------- Shared styles ----------------
const codeInline = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
const tabBtn = {
  padding: "8px 16px",
  border: "1px solid #ccc",
  borderRadius: 6,
  background: "#fff",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: 14,
};
const tabBtnActive = {
  background: "#111",
  color: "#fff",
  border: "1px solid #111",
};
const btn = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "7px 12px",
  borderRadius: 6,
  border: "1px solid #ccc",
  background: "#f7f7f7",
  cursor: "pointer",
  fontWeight: 500,
};
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
const controlGroup = { marginTop: 10 };
const label = {
  display: "block",
  fontSize: 12,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
};
const statBox = {
  marginTop: 14,
  padding: "10px 14px",
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 8,
};
const narrationBox = {
  marginTop: 14,
  padding: "12px 16px",
  background: "#f4efe6",
  borderRadius: 8,
  border: "1px solid #ddd",
};
