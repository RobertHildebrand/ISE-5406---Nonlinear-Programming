import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward, Zap } from "lucide-react";

/* ============================================================
   SVM & PERCEPTRON — INTERACTIVE DEMO
   For ISE 5405/5406. Two sections:
   1) Perceptron with a Step button (cycle through misclassified
      points, watch the separating line rotate into place).
   2) Kernel SVM trained via simplified SMO. Linear / polynomial /
      RBF kernels with adjustable parameters.
   ============================================================ */

// ---------------- RNG ----------------
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
  // Box–Muller
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ---------------- Datasets ----------------
function dsLinSeparable(seed = 1, n = 24) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: 1.4 + 0.55 * randn(rng), y: 1.0 + 0.55 * randn(rng), label: 1 });
  }
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: -1.4 + 0.55 * randn(rng), y: -1.0 + 0.55 * randn(rng), label: -1 });
  }
  return pts;
}
function dsLinTight(seed = 7, n = 30) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: 0.7 + 0.35 * randn(rng), y: 0.4 + 0.35 * randn(rng), label: 1 });
  }
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: -0.7 + 0.35 * randn(rng), y: -0.4 + 0.35 * randn(rng), label: -1 });
  }
  return pts;
}
function dsXOR(seed = 3, n = 60) {
  const rng = mulberry32(seed);
  const pts = [];
  const centers = [
    [1.2, 1.2, 1],
    [-1.2, -1.2, 1],
    [1.2, -1.2, -1],
    [-1.2, 1.2, -1],
  ];
  for (let i = 0; i < n; i++) {
    const c = centers[i % 4];
    pts.push({ x: c[0] + 0.4 * randn(rng), y: c[1] + 0.4 * randn(rng), label: c[2] });
  }
  return pts;
}
function dsMoons(seed = 4, n = 60) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    const t = Math.PI * (i / (n / 2 - 1));
    pts.push({
      x: 1.5 * Math.cos(t) - 0.6 + 0.12 * randn(rng),
      y: 1.5 * Math.sin(t) - 0.4 + 0.12 * randn(rng),
      label: 1,
    });
  }
  for (let i = 0; i < n / 2; i++) {
    const t = Math.PI * (i / (n / 2 - 1));
    pts.push({
      x: 1.5 * Math.cos(t) + 0.6 + 0.12 * randn(rng),
      y: -1.5 * Math.sin(t) + 0.4 + 0.12 * randn(rng),
      label: -1,
    });
  }
  return pts;
}
function dsCircles(seed = 5, n = 70) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    const t = 2 * Math.PI * (i / (n / 2));
    pts.push({
      x: 0.6 * Math.cos(t) + 0.1 * randn(rng),
      y: 0.6 * Math.sin(t) + 0.1 * randn(rng),
      label: 1,
    });
  }
  for (let i = 0; i < n / 2; i++) {
    const t = 2 * Math.PI * (i / (n / 2));
    pts.push({
      x: 1.8 * Math.cos(t) + 0.13 * randn(rng),
      y: 1.8 * Math.sin(t) + 0.13 * randn(rng),
      label: -1,
    });
  }
  return pts;
}

// ---------------- Kernels ----------------
function kernelFn(name, params) {
  if (name === "linear") {
    return (a, b) => a.x * b.x + a.y * b.y;
  }
  if (name === "poly") {
    const { gamma, coef0, degree } = params;
    return (a, b) => Math.pow(gamma * (a.x * b.x + a.y * b.y) + coef0, degree);
  }
  if (name === "rbf") {
    const { gamma } = params;
    return (a, b) => {
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      return Math.exp(-gamma * (dx * dx + dy * dy));
    };
  }
  return () => 0;
}

// ---------------- Simplified SMO (Platt, Stanford CS229 notes) ----------------
function smoTrain(data, K, C = 1.0, tol = 1e-3, maxPasses = 8, seed = 42) {
  const m = data.length;
  const alpha = new Array(m).fill(0);
  let b = 0;
  const rng = mulberry32(seed);
  // Cache the kernel matrix for speed.
  const Kmat = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      Kmat[i * m + j] = K(data[i], data[j]);
    }
  }
  const f = (i) => {
    let s = b;
    for (let j = 0; j < m; j++) s += alpha[j] * data[j].label * Kmat[j * m + i];
    return s;
  };
  let passes = 0;
  let iter = 0;
  while (passes < maxPasses && iter < 4000) {
    let numChanged = 0;
    for (let i = 0; i < m; i++) {
      const Ei = f(i) - data[i].label;
      const yi = data[i].label;
      if (
        (yi * Ei < -tol && alpha[i] < C) ||
        (yi * Ei > tol && alpha[i] > 0)
      ) {
        let j = i;
        while (j === i) j = Math.floor(rng() * m);
        const Ej = f(j) - data[j].label;
        const yj = data[j].label;
        const ai_old = alpha[i];
        const aj_old = alpha[j];
        let L, H;
        if (yi !== yj) {
          L = Math.max(0, alpha[j] - alpha[i]);
          H = Math.min(C, C + alpha[j] - alpha[i]);
        } else {
          L = Math.max(0, alpha[i] + alpha[j] - C);
          H = Math.min(C, alpha[i] + alpha[j]);
        }
        if (L === H) continue;
        const eta =
          2 * Kmat[i * m + j] - Kmat[i * m + i] - Kmat[j * m + j];
        if (eta >= 0) continue;
        let aj = alpha[j] - (yj * (Ei - Ej)) / eta;
        if (aj > H) aj = H;
        else if (aj < L) aj = L;
        if (Math.abs(aj - aj_old) < 1e-5) continue;
        alpha[j] = aj;
        alpha[i] = ai_old + yi * yj * (aj_old - aj);
        const b1 =
          b -
          Ei -
          yi * (alpha[i] - ai_old) * Kmat[i * m + i] -
          yj * (alpha[j] - aj_old) * Kmat[i * m + j];
        const b2 =
          b -
          Ej -
          yi * (alpha[i] - ai_old) * Kmat[i * m + j] -
          yj * (alpha[j] - aj_old) * Kmat[j * m + j];
        if (alpha[i] > 0 && alpha[i] < C) b = b1;
        else if (alpha[j] > 0 && alpha[j] < C) b = b2;
        else b = (b1 + b2) / 2;
        numChanged++;
      }
      iter++;
    }
    if (numChanged === 0) passes++;
    else passes = 0;
  }
  // Pre-compute support vector list (alpha > eps).
  const svIdx = [];
  for (let i = 0; i < m; i++) if (alpha[i] > 1e-6) svIdx.push(i);
  return { alpha, b, svIdx };
}

// ---------------- Plot helpers ----------------
const PLOT = { size: 460, lo: -3, hi: 3 };
const dataToPxX = (x) => ((x - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const dataToPxY = (y) => PLOT.size - ((y - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const pxToDataX = (px) => PLOT.lo + (px / PLOT.size) * (PLOT.hi - PLOT.lo);
const pxToDataY = (py) =>
  PLOT.lo + ((PLOT.size - py) / PLOT.size) * (PLOT.hi - PLOT.lo);

const COLOR_POS = "#1f4e3d";
const COLOR_NEG = "#c8311c";
const COLOR_POS_LIGHT = "#cfe5da";
const COLOR_NEG_LIGHT = "#f3d2cc";

function Axes() {
  return (
    <g>
      <rect width={PLOT.size} height={PLOT.size} fill="#fafafa" stroke="#d4d4d4" />
      <line
        x1={dataToPxX(0)}
        y1={0}
        x2={dataToPxX(0)}
        y2={PLOT.size}
        stroke="#bbb"
        strokeDasharray="3 3"
      />
      <line
        x1={0}
        y1={dataToPxY(0)}
        x2={PLOT.size}
        y2={dataToPxY(0)}
        stroke="#bbb"
        strokeDasharray="3 3"
      />
    </g>
  );
}

function PointDots({ data, highlightIdx = -1 }) {
  return (
    <g>
      {data.map((p, i) => (
        <circle
          key={i}
          cx={dataToPxX(p.x)}
          cy={dataToPxY(p.y)}
          r={i === highlightIdx ? 8 : 5.5}
          fill={p.label === 1 ? COLOR_POS : COLOR_NEG}
          stroke={i === highlightIdx ? "#000" : "white"}
          strokeWidth={i === highlightIdx ? 2.5 : 1.2}
        />
      ))}
    </g>
  );
}

// ============================================================
// PERCEPTRON SECTION
// ============================================================
const PERCEPTRON_DATASETS = {
  separable: { name: "Linearly Separable (easy)", make: () => dsLinSeparable(1, 24) },
  tight: { name: "Linearly Separable (tight)", make: () => dsLinTight(7, 30) },
};

function PerceptronViz() {
  const [datasetKey, setDatasetKey] = useState("separable");
  const [data, setData] = useState(() => PERCEPTRON_DATASETS.separable.make());
  const [w, setW] = useState([0.0, 0.0]);
  const [bias, setBias] = useState(0);
  const [iter, setIter] = useState(0);
  const [updates, setUpdates] = useState(0);
  const [lastIdx, setLastIdx] = useState(-1);
  const [running, setRunning] = useState(false);
  const [cursor, setCursor] = useState(0); // for cyclic scan
  const [converged, setConverged] = useState(false);

  const reset = useCallback(
    (key = datasetKey) => {
      setData(PERCEPTRON_DATASETS[key].make());
      setW([0, 0]);
      setBias(0);
      setIter(0);
      setUpdates(0);
      setLastIdx(-1);
      setCursor(0);
      setConverged(false);
      setRunning(false);
    },
    [datasetKey]
  );

  // Take exactly one perceptron step: scan from `cursor`, find first
  // misclassified point, update w and b. If a full pass finds nothing,
  // converged.
  const step = useCallback(() => {
    if (converged) return false;
    const n = data.length;
    let scanned = 0;
    let i = cursor;
    while (scanned < n) {
      const p = data[i];
      const score = w[0] * p.x + w[1] * p.y + bias;
      const margin = p.label * score;
      if (margin <= 0) {
        // misclassified — update
        setW([w[0] + p.label * p.x, w[1] + p.label * p.y]);
        setBias(bias + p.label);
        setLastIdx(i);
        setUpdates((u) => u + 1);
        setIter((k) => k + 1);
        setCursor((i + 1) % n);
        return true;
      }
      i = (i + 1) % n;
      scanned++;
    }
    // full pass with no update → converged
    setConverged(true);
    setRunning(false);
    setLastIdx(-1);
    setIter((k) => k + 1);
    return false;
  }, [data, w, bias, cursor, converged]);

  // Run-to-convergence: drive `step` on a timer.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      step();
    }, 110);
    return () => clearInterval(id);
  }, [running, step]);

  // Compute the line w0*x + w1*y + b = 0 within the plot box.
  const linePts = useMemo(() => {
    const [a, c] = w;
    const eps = 1e-9;
    if (Math.abs(a) < eps && Math.abs(c) < eps) return null;
    // Find two endpoints by intersecting with plot box.
    const lo = PLOT.lo,
      hi = PLOT.hi;
    const pts = [];
    // Intersect with x = lo, x = hi
    if (Math.abs(c) > eps) {
      const yLo = -(a * lo + bias) / c;
      const yHi = -(a * hi + bias) / c;
      if (yLo >= lo && yLo <= hi) pts.push([lo, yLo]);
      if (yHi >= lo && yHi <= hi) pts.push([hi, yHi]);
    }
    if (Math.abs(a) > eps) {
      const xLo = -(c * lo + bias) / a;
      const xHi = -(c * hi + bias) / a;
      if (xLo >= lo && xLo <= hi) pts.push([xLo, lo]);
      if (xHi >= lo && xHi <= hi) pts.push([xHi, hi]);
    }
    if (pts.length < 2) return null;
    return [pts[0], pts[1]];
  }, [w, bias]);

  const numMisclass = useMemo(() => {
    let m = 0;
    for (const p of data) {
      const s = w[0] * p.x + w[1] * p.y + bias;
      if (p.label * s <= 0) m++;
    }
    return m;
  }, [data, w, bias]);

  return (
    <section style={section}>
      <h2 style={h2}>1. Perceptron — Step Through the Algorithm</h2>
      <p style={p}>
        The perceptron scans points and updates whenever it finds a misclassified one:
        <code style={code}>w ← w + y·x</code> and <code style={code}>b ← b + y</code>.
        Click <b>Step</b> to advance one update. The next misclassified point is highlighted.
      </p>

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
        <svg width={PLOT.size} height={PLOT.size} style={{ background: "#fafafa", borderRadius: 6 }}>
          <Axes />
          {linePts && (
            <line
              x1={dataToPxX(linePts[0][0])}
              y1={dataToPxY(linePts[0][1])}
              x2={dataToPxX(linePts[1][0])}
              y2={dataToPxY(linePts[1][1])}
              stroke="#111"
              strokeWidth={2}
            />
          )}
          <PointDots data={data} highlightIdx={lastIdx} />
        </svg>

        <div style={{ minWidth: 280, flex: 1 }}>
          <div style={controlGroup}>
            <label style={label}>Dataset</label>
            <select
              value={datasetKey}
              onChange={(e) => {
                setDatasetKey(e.target.value);
                reset(e.target.value);
              }}
              style={select}
            >
              {Object.entries(PERCEPTRON_DATASETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            <button onClick={step} disabled={converged} style={btnPrimary}>
              <StepForward size={16} /> Step
            </button>
            <button onClick={() => setRunning((r) => !r)} disabled={converged} style={btn}>
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={() => reset()} style={btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={statBox}>
            <Stat label="iter" value={iter} />
            <Stat label="updates" value={updates} />
            <Stat label="misclassified" value={numMisclass} />
            <Stat label="w" value={`(${w[0].toFixed(2)}, ${w[1].toFixed(2)})`} />
            <Stat label="b" value={bias.toFixed(2)} />
            <Stat
              label="status"
              value={converged ? "✓ converged" : numMisclass === 0 ? "all correct" : "training"}
            />
          </div>

          <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
            <b>Note.</b> The perceptron only converges if the data is linearly separable.
            On the "tight" dataset, classes are closer together — convergence is still
            guaranteed, but the line wobbles more before settling.
          </p>
        </div>
      </div>
    </section>
  );
}

// ============================================================
// KERNEL SVM SECTION
// ============================================================
const SVM_DATASETS = {
  blobs: { name: "Two Blobs (linear-friendly)", make: () => dsLinSeparable(2, 30) },
  xor: { name: "XOR (needs nonlinear)", make: () => dsXOR(3, 60) },
  moons: { name: "Moons", make: () => dsMoons(4, 60) },
  circles: { name: "Concentric Circles", make: () => dsCircles(5, 70) },
};

const KERNELS = {
  linear: { name: "Linear  K(x,y) = x·y", params: [] },
  poly: {
    name: "Polynomial  (γ·x·y + r)^d",
    params: [
      { key: "degree", label: "degree d", min: 2, max: 6, step: 1, def: 3 },
      { key: "gamma", label: "γ", min: 0.1, max: 2, step: 0.1, def: 0.5 },
      { key: "coef0", label: "r", min: 0, max: 2, step: 0.1, def: 1 },
    ],
  },
  rbf: {
    name: "RBF (Gaussian)  exp(−γ‖x−y‖²)",
    params: [{ key: "gamma", label: "γ", min: 0.1, max: 5, step: 0.1, def: 1.0 }],
  },
};

function KernelSVMViz() {
  const [datasetKey, setDatasetKey] = useState("xor");
  const [data, setData] = useState(() => SVM_DATASETS.xor.make());
  const [kernelKey, setKernelKey] = useState("rbf");
  const [params, setParams] = useState({
    degree: 3,
    gamma: 1.0,
    coef0: 1,
  });
  const [C, setC] = useState(5.0);
  const [model, setModel] = useState(null);
  const [training, setTraining] = useState(false);

  const reloadDataset = (key) => {
    setDatasetKey(key);
    setData(SVM_DATASETS[key].make());
    setModel(null);
  };
  const setKernel = (key) => {
    setKernelKey(key);
    setModel(null);
  };
  const updateParam = (k, v) => {
    setParams((prev) => ({ ...prev, [k]: v }));
    setModel(null);
  };

  const train = useCallback(() => {
    setTraining(true);
    // Yield to the browser so the button renders the "Training…" state.
    setTimeout(() => {
      const K = kernelFn(kernelKey, params);
      const m = smoTrain(data, K, C);
      setModel({ ...m, kernelKey, params: { ...params }, K });
      setTraining(false);
    }, 20);
  }, [data, kernelKey, params, C]);

  // Build the decision-region tile grid + prediction at every point.
  const grid = useMemo(() => {
    if (!model) return null;
    const N = 70;
    const cell = PLOT.size / N;
    const tiles = [];
    const { alpha, b, svIdx, K } = model;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const xData = pxToDataX((i + 0.5) * cell);
        const yData = pxToDataY((j + 0.5) * cell);
        const probe = { x: xData, y: yData };
        let s = b;
        for (const k of svIdx) s += alpha[k] * data[k].label * K(data[k], probe);
        // Color by sign and saturation by distance (clipped).
        const mag = Math.min(1, Math.abs(s) / 1.5);
        const col = s > 0 ? COLOR_POS_LIGHT : COLOR_NEG_LIGHT;
        // We modulate opacity for the saturation effect.
        tiles.push({ i, j, color: col, opacity: 0.25 + 0.55 * mag });
      }
    }
    return { N, cell, tiles };
  }, [model, data]);

  // Decision-boundary contour: collect cell edges where sign flips.
  const contourSegs = useMemo(() => {
    if (!model) return [];
    const N = 90;
    const step = PLOT.size / N;
    const { alpha, b, svIdx, K } = model;
    const f = (px, py) => {
      const xd = pxToDataX(px);
      const yd = pxToDataY(py);
      const probe = { x: xd, y: yd };
      let s = b;
      for (const k of svIdx) s += alpha[k] * data[k].label * K(data[k], probe);
      return s;
    };
    // Sample on grid.
    const F = new Float64Array((N + 1) * (N + 1));
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        F[i * (N + 1) + j] = f(i * step, j * step);
      }
    }
    const segs = [];
    const interp = (a, b) => a / (a - b);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v00 = F[i * (N + 1) + j];
        const v10 = F[(i + 1) * (N + 1) + j];
        const v01 = F[i * (N + 1) + (j + 1)];
        const v11 = F[(i + 1) * (N + 1) + (j + 1)];
        let idx = 0;
        if (v00 > 0) idx |= 1;
        if (v10 > 0) idx |= 2;
        if (v11 > 0) idx |= 4;
        if (v01 > 0) idx |= 8;
        if (idx === 0 || idx === 15) continue;
        const x0 = i * step,
          x1 = (i + 1) * step;
        const y0 = j * step,
          y1 = (j + 1) * step;
        // edges: bottom, right, top, left
        const edges = {};
        if ((v00 > 0) !== (v10 > 0))
          edges.b = [x0 + step * interp(v00, v10), y0];
        if ((v10 > 0) !== (v11 > 0))
          edges.r = [x1, y0 + step * interp(v10, v11)];
        if ((v01 > 0) !== (v11 > 0))
          edges.t = [x0 + step * interp(v01, v11), y1];
        if ((v00 > 0) !== (v01 > 0))
          edges.l = [x0, y0 + step * interp(v00, v01)];
        const e = Object.values(edges);
        if (e.length >= 2) segs.push([e[0], e[1]]);
        if (e.length === 4) segs.push([e[2], e[3]]);
      }
    }
    return segs;
  }, [model, data]);

  const accuracy = useMemo(() => {
    if (!model) return null;
    const { alpha, b, svIdx, K } = model;
    let correct = 0;
    for (const p of data) {
      let s = b;
      for (const k of svIdx) s += alpha[k] * data[k].label * K(data[k], p);
      if (Math.sign(s) === p.label) correct++;
    }
    return correct / data.length;
  }, [model, data]);

  const kernelDef = KERNELS[kernelKey];

  return (
    <section style={section}>
      <h2 style={h2}>2. Kernel SVM — Choose Your Kernel</h2>
      <p style={p}>
        Pick a dataset and a kernel. Train with simplified SMO and watch the decision
        boundary fit. <b>Linear</b> can only draw a straight line; <b>polynomial</b> and{" "}
        <b>RBF</b> bend space to separate classes that aren't linearly separable.
      </p>

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
        <svg
          width={PLOT.size}
          height={PLOT.size}
          style={{ background: "#fafafa", borderRadius: 6 }}
        >
          <Axes />
          {grid &&
            grid.tiles.map((t, k) => (
              <rect
                key={k}
                x={t.i * grid.cell}
                y={(grid.N - 1 - t.j) * grid.cell}
                width={grid.cell + 0.5}
                height={grid.cell + 0.5}
                fill={t.color}
                opacity={t.opacity}
              />
            ))}
          {contourSegs.map((s, k) => (
            <line
              key={k}
              x1={s[0][0]}
              y1={s[0][1]}
              x2={s[1][0]}
              y2={s[1][1]}
              stroke="#111"
              strokeWidth={1.6}
            />
          ))}
          {/* Mark support vectors */}
          {model &&
            model.svIdx.map((k) => (
              <circle
                key={`sv-${k}`}
                cx={dataToPxX(data[k].x)}
                cy={dataToPxY(data[k].y)}
                r={10}
                fill="none"
                stroke="#111"
                strokeWidth={1.5}
                strokeDasharray="3 2"
                opacity={0.7}
              />
            ))}
          <PointDots data={data} />
        </svg>

        <div style={{ minWidth: 280, flex: 1 }}>
          <div style={controlGroup}>
            <label style={label}>Dataset</label>
            <select
              value={datasetKey}
              onChange={(e) => reloadDataset(e.target.value)}
              style={select}
            >
              {Object.entries(SVM_DATASETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          <div style={controlGroup}>
            <label style={label}>Kernel</label>
            <select
              value={kernelKey}
              onChange={(e) => setKernel(e.target.value)}
              style={select}
            >
              {Object.entries(KERNELS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          {kernelDef.params.map((pp) => (
            <div key={pp.key} style={controlGroup}>
              <label style={label}>
                {pp.label}: <b>{(+params[pp.key]).toFixed(pp.step < 1 ? 2 : 0)}</b>
              </label>
              <input
                type="range"
                min={pp.min}
                max={pp.max}
                step={pp.step}
                value={params[pp.key]}
                onChange={(e) => updateParam(pp.key, +e.target.value)}
                style={{ width: "100%" }}
              />
            </div>
          ))}

          <div style={controlGroup}>
            <label style={label}>
              C (soft-margin penalty): <b>{C.toFixed(1)}</b>
            </label>
            <input
              type="range"
              min={0.1}
              max={20}
              step={0.1}
              value={C}
              onChange={(e) => {
                setC(+e.target.value);
                setModel(null);
              }}
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            <button onClick={train} disabled={training} style={btnPrimary}>
              <Zap size={16} /> {training ? "Training…" : "Train"}
            </button>
            <button
              onClick={() => {
                setData(SVM_DATASETS[datasetKey].make());
                setModel(null);
              }}
              style={btn}
            >
              <RotateCcw size={16} /> New Sample
            </button>
          </div>

          <div style={statBox}>
            <Stat label="kernel" value={kernelKey} />
            <Stat label="C" value={C.toFixed(1)} />
            <Stat
              label="support vectors"
              value={model ? `${model.svIdx.length} / ${data.length}` : "—"}
            />
            <Stat
              label="training accuracy"
              value={accuracy != null ? `${(100 * accuracy).toFixed(1)}%` : "—"}
            />
          </div>

          <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
            <b>Tips.</b> XOR / Circles / Moons are not linearly separable — the linear
            kernel will plateau at low accuracy. Try RBF with γ ≈ 1, or polynomial of
            degree 2–3. Larger <code style={code}>C</code> = harder margin (less
            tolerance for misclassification). Dashed circles mark support vectors
            (data points that pin down the boundary).
          </p>
        </div>
      </div>
    </section>
  );
}

// ============================================================
// Shared UI bits
// ============================================================
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

const section = {
  padding: "32px 28px",
  marginBottom: 24,
  background: "#fff",
  borderRadius: 12,
  border: "1px solid #e7e7e7",
  boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
};
const h2 = { fontSize: 22, fontWeight: 800, marginBottom: 8 };
const p = { color: "#444", lineHeight: 1.55, marginBottom: 14, maxWidth: 720 };
const code = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
const controlGroup = { marginTop: 10 };
const label = {
  display: "block",
  fontSize: 13,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
};
const select = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: 6,
  border: "1px solid #ccc",
  background: "#fff",
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
const statBox = {
  marginTop: 14,
  padding: "8px 12px",
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 6,
};

// ============================================================
// Top-level
// ============================================================
export default function SVMDemo() {
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Perceptron & Kernel SVM
      </h1>
      <p style={{ color: "#666", marginBottom: 28 }}>
        Two interactive demos: step through the perceptron algorithm, then train
        kernel SVMs with your choice of kernel.
      </p>
      <PerceptronViz />
      <KernelSVMViz />
    </div>
  );
}
