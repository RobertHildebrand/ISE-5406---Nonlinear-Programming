import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Zap } from 'lucide-react';

// ============================================================
// Tiny digit dataset: 8x8 binary patterns for digits 0-9
// Each row is a 64-dim vector, label is one-hot of size 10
// ============================================================
const DIGIT_PATTERNS = {
  0: [
    "..####..",
    ".#....#.",
    "#......#",
    "#......#",
    "#......#",
    "#......#",
    ".#....#.",
    "..####..",
  ],
  1: [
    "...##...",
    "..###...",
    ".#.##...",
    "...##...",
    "...##...",
    "...##...",
    "...##...",
    ".######.",
  ],
  2: [
    ".#####..",
    "#.....#.",
    "......#.",
    ".....#..",
    "...##...",
    "..#.....",
    ".#......",
    "#######.",
  ],
  3: [
    ".#####..",
    "#.....#.",
    "......#.",
    "..####..",
    "......#.",
    "......#.",
    "#.....#.",
    ".#####..",
  ],
  4: [
    "....##..",
    "...###..",
    "..#.##..",
    ".#..##..",
    "#######.",
    "....##..",
    "....##..",
    "....##..",
  ],
  5: [
    "#######.",
    "#.......",
    "#.......",
    "######..",
    "......#.",
    "......#.",
    "#.....#.",
    ".#####..",
  ],
  6: [
    "..####..",
    ".#....#.",
    "#.......",
    "######..",
    "#.....#.",
    "#.....#.",
    ".#...#..",
    "..###...",
  ],
  7: [
    "#######.",
    "......#.",
    ".....#..",
    "....#...",
    "...#....",
    "..#.....",
    "..#.....",
    "..#.....",
  ],
  8: [
    ".####...",
    "#....#..",
    "#....#..",
    ".####...",
    "#....#..",
    "#....#..",
    "#....#..",
    ".####...",
  ],
  9: [
    ".####...",
    "#....#..",
    "#....#..",
    "#....#..",
    ".#####..",
    "......#.",
    ".....#..",
    ".####...",
  ],
};

// Build dataset: for each digit, create the base pattern + a few noisy copies
function buildDataset() {
  const data = [];
  for (let d = 0; d < 10; d++) {
    const grid = DIGIT_PATTERNS[d];
    const base = [];
    for (const row of grid) {
      for (const ch of row) base.push(ch === '#' ? 1 : 0);
    }
    // base copy
    data.push({ x: base.slice(), y: d });
    // noisy copies
    for (let n = 0; n < 3; n++) {
      const noisy = base.slice();
      // flip a few random pixels
      const flips = 3;
      for (let f = 0; f < flips; f++) {
        const idx = Math.floor(Math.random() * 64);
        // flip with small prob to keep digit recognizable
        if (Math.random() < 0.5) noisy[idx] = noisy[idx] > 0.5 ? 0 : 1;
      }
      data.push({ x: noisy, y: d });
    }
  }
  return data;
}

// ============================================================
// Neural network: 64 -> 16 -> 10 with ReLU + softmax
// All forward/backward done by hand so we can show internals
// ============================================================
const INPUT = 64;
const HIDDEN = 16;
const OUTPUT = 10;

function randn() {
  // Box-Muller
  const u = 1 - Math.random();
  const v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function initParams(seed) {
  // Re-seedable-ish: just call Math.random; for reproducibility we reset on init
  const W1 = new Float32Array(INPUT * HIDDEN);
  const b1 = new Float32Array(HIDDEN);
  const W2 = new Float32Array(HIDDEN * OUTPUT);
  const b2 = new Float32Array(OUTPUT);
  const s1 = Math.sqrt(2 / INPUT); // He init
  const s2 = Math.sqrt(2 / HIDDEN);
  for (let i = 0; i < W1.length; i++) W1[i] = randn() * s1;
  for (let i = 0; i < W2.length; i++) W2[i] = randn() * s2;
  return { W1, b1, W2, b2 };
}

function relu(x) { return x > 0 ? x : 0; }

function forward(params, x) {
  const { W1, b1, W2, b2 } = params;
  // hidden = ReLU(W1 x + b1)
  const z1 = new Float32Array(HIDDEN);
  const h = new Float32Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let s = b1[j];
    for (let i = 0; i < INPUT; i++) s += W1[i * HIDDEN + j] * x[i];
    z1[j] = s;
    h[j] = relu(s);
  }
  const z2 = new Float32Array(OUTPUT);
  for (let k = 0; k < OUTPUT; k++) {
    let s = b2[k];
    for (let j = 0; j < HIDDEN; j++) s += W2[j * OUTPUT + k] * h[j];
    z2[k] = s;
  }
  // softmax
  let m = -Infinity;
  for (let k = 0; k < OUTPUT; k++) if (z2[k] > m) m = z2[k];
  let sumE = 0;
  const p = new Float32Array(OUTPUT);
  for (let k = 0; k < OUTPUT; k++) { p[k] = Math.exp(z2[k] - m); sumE += p[k]; }
  for (let k = 0; k < OUTPUT; k++) p[k] /= sumE;
  return { z1, h, z2, p };
}

// Compute gradients for a mini-batch; returns averaged grads + avg loss + acc
function computeGrads(params, batch) {
  const { W1, W2 } = params;
  const gW1 = new Float32Array(INPUT * HIDDEN);
  const gb1 = new Float32Array(HIDDEN);
  const gW2 = new Float32Array(HIDDEN * OUTPUT);
  const gb2 = new Float32Array(OUTPUT);
  let totalLoss = 0;
  let correct = 0;
  for (const { x, y } of batch) {
    const { z1, h, p } = forward(params, x);
    // loss = -log p[y]
    totalLoss += -Math.log(Math.max(p[y], 1e-12));
    // pred
    let best = 0;
    for (let k = 1; k < OUTPUT; k++) if (p[k] > p[best]) best = k;
    if (best === y) correct++;
    // dL/dz2 = p - onehot(y)
    const dz2 = new Float32Array(OUTPUT);
    for (let k = 0; k < OUTPUT; k++) dz2[k] = p[k] - (k === y ? 1 : 0);
    // grads on W2, b2
    for (let k = 0; k < OUTPUT; k++) {
      gb2[k] += dz2[k];
      for (let j = 0; j < HIDDEN; j++) {
        gW2[j * OUTPUT + k] += dz2[k] * h[j];
      }
    }
    // dh = W2^T dz2; then dz1 = dh * (z1>0)
    const dz1 = new Float32Array(HIDDEN);
    for (let j = 0; j < HIDDEN; j++) {
      let s = 0;
      for (let k = 0; k < OUTPUT; k++) s += W2[j * OUTPUT + k] * dz2[k];
      dz1[j] = z1[j] > 0 ? s : 0;
    }
    for (let j = 0; j < HIDDEN; j++) {
      gb1[j] += dz1[j];
      for (let i = 0; i < INPUT; i++) {
        gW1[i * HIDDEN + j] += dz1[j] * x[i];
      }
    }
  }
  const N = batch.length;
  for (let i = 0; i < gW1.length; i++) gW1[i] /= N;
  for (let i = 0; i < gb1.length; i++) gb1[i] /= N;
  for (let i = 0; i < gW2.length; i++) gW2[i] /= N;
  for (let i = 0; i < gb2.length; i++) gb2[i] /= N;
  return { gW1, gb1, gW2, gb2, loss: totalLoss / N, acc: correct / N };
}

// ============================================================
// Optimizers (first-order methods)
// Each optimizer holds its own state and exposes step(params, grads, lr)
// ============================================================
function makeOptimizer(name) {
  const zerosLike = (a) => new Float32Array(a.length);
  if (name === 'SGD') {
    return {
      name,
      init() {},
      step(params, grads, lr) {
        for (const k of ['W1','b1','W2','b2']) {
          const p = params[k]; const g = grads['g'+k];
          for (let i = 0; i < p.length; i++) p[i] -= lr * g[i];
        }
      }
    };
  }
  if (name === 'Momentum') {
    let v = null;
    return {
      name,
      init(params) { v = { W1: zerosLike(params.W1), b1: zerosLike(params.b1), W2: zerosLike(params.W2), b2: zerosLike(params.b2) }; },
      step(params, grads, lr) {
        const beta = 0.9;
        for (const k of ['W1','b1','W2','b2']) {
          const p = params[k]; const g = grads['g'+k]; const vv = v[k];
          for (let i = 0; i < p.length; i++) {
            vv[i] = beta * vv[i] + g[i];
            p[i] -= lr * vv[i];
          }
        }
      }
    };
  }
  if (name === 'Nesterov') {
    let v = null;
    return {
      name,
      init(params) { v = { W1: zerosLike(params.W1), b1: zerosLike(params.b1), W2: zerosLike(params.W2), b2: zerosLike(params.b2) }; },
      step(params, grads, lr) {
        const beta = 0.9;
        for (const k of ['W1','b1','W2','b2']) {
          const p = params[k]; const g = grads['g'+k]; const vv = v[k];
          for (let i = 0; i < p.length; i++) {
            const vPrev = vv[i];
            vv[i] = beta * vv[i] + g[i];
            // Nesterov-style update: look-ahead
            p[i] -= lr * (beta * vv[i] + g[i] - beta * vPrev * 0); // simplified
            // We'll use the standard form: p -= lr*(beta*v + g)
          }
        }
      }
    };
  }
  if (name === 'RMSProp') {
    let s = null;
    return {
      name,
      init(params) { s = { W1: zerosLike(params.W1), b1: zerosLike(params.b1), W2: zerosLike(params.W2), b2: zerosLike(params.b2) }; },
      step(params, grads, lr) {
        const beta = 0.9, eps = 1e-8;
        for (const k of ['W1','b1','W2','b2']) {
          const p = params[k]; const g = grads['g'+k]; const ss = s[k];
          for (let i = 0; i < p.length; i++) {
            ss[i] = beta * ss[i] + (1 - beta) * g[i] * g[i];
            p[i] -= lr * g[i] / (Math.sqrt(ss[i]) + eps);
          }
        }
      }
    };
  }
  if (name === 'Adam') {
    let m = null, v = null, t = 0;
    return {
      name,
      init(params) {
        m = { W1: zerosLike(params.W1), b1: zerosLike(params.b1), W2: zerosLike(params.W2), b2: zerosLike(params.b2) };
        v = { W1: zerosLike(params.W1), b1: zerosLike(params.b1), W2: zerosLike(params.W2), b2: zerosLike(params.b2) };
        t = 0;
      },
      step(params, grads, lr) {
        const b1 = 0.9, b2 = 0.999, eps = 1e-8;
        t++;
        const c1 = 1 - Math.pow(b1, t);
        const c2 = 1 - Math.pow(b2, t);
        for (const k of ['W1','b1','W2','b2']) {
          const p = params[k]; const g = grads['g'+k]; const mm = m[k]; const vv = v[k];
          for (let i = 0; i < p.length; i++) {
            mm[i] = b1 * mm[i] + (1 - b1) * g[i];
            vv[i] = b2 * vv[i] + (1 - b2) * g[i] * g[i];
            const mh = mm[i] / c1;
            const vh = vv[i] / c2;
            p[i] -= lr * mh / (Math.sqrt(vh) + eps);
          }
        }
      }
    };
  }
}

const OPTIMIZERS = ['SGD', 'Momentum', 'RMSProp', 'Adam'];
const OPT_INFO = {
  SGD: 'Pure gradient descent: θ ← θ − η∇L. Simple, sensitive to step size.',
  Momentum: 'Accumulates a velocity: v ← βv + g, θ ← θ − ηv. Damps oscillations.',
  RMSProp: 'Per-coordinate scaling by √(EMA of g²). Adapts to curvature.',
  Adam: 'Momentum + RMSProp with bias correction. Robust default in deep learning.',
};
const OPT_COLOR = {
  SGD: '#e85d75',
  Momentum: '#f5b14a',
  RMSProp: '#5fb3e8',
  Adam: '#7ed957',
};

// ============================================================
// React component
// ============================================================
export default function NeuralNetViz() {
  const [dataset, setDataset] = useState(() => buildDataset());
  const [optimizer, setOptimizer] = useState('Adam');
  const [lr, setLr] = useState(0.05);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [paramsState, setParamsState] = useState(() => initParams());
  const [history, setHistory] = useState([]); // [{step, loss, acc}] per optimizer
  const [allHistories, setAllHistories] = useState({}); // optimizer -> history array
  const [activations, setActivations] = useState(null); // for displayed sample
  const [displayIdx, setDisplayIdx] = useState(0); // which sample to visualize
  const [speed, setSpeed] = useState(20); // steps per animation frame batch

  const rafRef = useRef(null);
  const optRef = useRef(null);
  const paramsRef = useRef(null);

  // Init optimizer & params
  const reset = useCallback(() => {
    const p = initParams();
    paramsRef.current = p;
    const opt = makeOptimizer(optimizer);
    opt.init(p);
    optRef.current = opt;
    setParamsState(p);
    setStep(0);
    setHistory([]);
    setRunning(false);
  }, [optimizer]);

  useEffect(() => { reset(); }, [reset]);

  // Get the 10 representative samples (one per digit, the clean version)
  const showcase = React.useMemo(() => {
    const out = [];
    for (let d = 0; d < 10; d++) {
      out.push(dataset.find(s => s.y === d));
    }
    return out;
  }, [dataset]);

  // Compute activations for the displayed sample whenever params change
  useEffect(() => {
    if (!paramsRef.current) return;
    const sample = showcase[displayIdx];
    if (!sample) return;
    const fwd = forward(paramsRef.current, sample.x);
    setActivations({ ...fwd, sample });
  }, [paramsState, displayIdx, showcase]);

  // Training loop using requestAnimationFrame
  useEffect(() => {
    if (!running) return;
    let cancelled = false;
    const tick = () => {
      if (cancelled) return;
      const batchSize = 16;
      let lastLoss = 0, lastAcc = 0;
      const newPoints = [];
      for (let s = 0; s < speed; s++) {
        const batch = [];
        for (let i = 0; i < batchSize; i++) {
          batch.push(dataset[Math.floor(Math.random() * dataset.length)]);
        }
        const grads = computeGrads(paramsRef.current, batch);
        optRef.current.step(paramsRef.current, grads, lr);
        lastLoss = grads.loss; lastAcc = grads.acc;
        newPoints.push({ step: 0, loss: lastLoss, acc: lastAcc });
      }
      setStep(prev => {
        const newStep = prev + speed;
        // attach correct step numbers
        const startStep = prev + 1;
        for (let i = 0; i < newPoints.length; i++) newPoints[i].step = startStep + i;
        setHistory(h => {
          const merged = h.concat(newPoints);
          // cap history length for perf
          return merged.length > 800 ? merged.slice(merged.length - 800) : merged;
        });
        return newStep;
      });
      // Trigger activation re-render by bumping a ref-like state
      setParamsState({ ...paramsRef.current });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => { cancelled = true; if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [running, dataset, lr, speed]);

  // Save history when stopping or switching optimizer
  const saveHistory = useCallback(() => {
    if (history.length > 0) {
      setAllHistories(prev => ({ ...prev, [optimizer]: history.slice() }));
    }
  }, [history, optimizer]);

  const handleOptChange = (newOpt) => {
    saveHistory();
    setOptimizer(newOpt);
  };

  const handleReset = () => {
    saveHistory();
    reset();
  };

  // ============================================================
  // Rendering helpers
  // ============================================================
  const renderDigit = (sample, isSelected, onClick) => {
    const cells = [];
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const v = sample.x[r * 8 + c];
        cells.push(
          <div key={`${r}-${c}`}
            className="aspect-square"
            style={{
              background: v > 0.5 ? '#1a1a1a' : '#fafaf5',
              border: '0.5px solid rgba(0,0,0,0.04)'
            }} />
        );
      }
    }
    return (
      <button
        onClick={onClick}
        className="flex flex-col items-center gap-1 p-2 rounded transition-all"
        style={{
          background: isSelected ? '#1a1a1a' : 'transparent',
          border: isSelected ? '2px solid #1a1a1a' : '2px solid transparent',
          cursor: 'pointer',
        }}>
        <div className="grid grid-cols-8 gap-0" style={{ width: 56, height: 56 }}>
          {cells}
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: 11,
          color: isSelected ? '#fafaf5' : '#666',
          letterSpacing: '0.05em'
        }}>
          y={sample.y}
        </div>
      </button>
    );
  };

  // Loss / accuracy chart (custom SVG)
  const renderChart = () => {
    const W = 520, H = 200, pad = 32;
    const histories = { ...allHistories };
    if (history.length > 0) histories[optimizer] = history;
    let maxStep = 1, maxLoss = 0.1;
    Object.values(histories).forEach(h => {
      h.forEach(pt => {
        if (pt.step > maxStep) maxStep = pt.step;
        if (pt.loss > maxLoss) maxLoss = pt.loss;
      });
    });
    maxLoss = Math.max(maxLoss, 2.5);
    const xScale = s => pad + ((W - pad * 2) * s) / maxStep;
    const yScale = l => pad + (H - pad * 2) * (1 - Math.min(l, maxLoss) / maxLoss);

    return (
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        {/* grid */}
        {[0, 0.5, 1, 1.5, 2].map((v, i) => v <= maxLoss && (
          <g key={i}>
            <line x1={pad} y1={yScale(v)} x2={W - pad} y2={yScale(v)}
              stroke="#1a1a1a" strokeOpacity="0.08" strokeDasharray="2 4" />
            <text x={pad - 6} y={yScale(v) + 4} textAnchor="end"
              style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: '#999' }}>
              {v.toFixed(1)}
            </text>
          </g>
        ))}
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#1a1a1a" strokeWidth="1" />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#1a1a1a" strokeWidth="1" />
        <text x={pad} y={pad - 10} style={{ fontFamily: 'Fraunces, serif', fontSize: 12, fill: '#1a1a1a', fontStyle: 'italic' }}>
          loss
        </text>
        <text x={W - pad} y={H - 8} textAnchor="end" style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, fill: '#999' }}>
          step {maxStep}
        </text>
        {/* lines per optimizer */}
        {Object.entries(histories).map(([name, h]) => {
          if (h.length < 2) return null;
          // smooth: take every Nth point
          const stride = Math.max(1, Math.floor(h.length / 200));
          const pts = [];
          for (let i = 0; i < h.length; i += stride) pts.push(h[i]);
          if (pts[pts.length - 1] !== h[h.length - 1]) pts.push(h[h.length - 1]);
          const d = pts.map((pt, i) => `${i === 0 ? 'M' : 'L'} ${xScale(pt.step)} ${yScale(pt.loss)}`).join(' ');
          return (
            <g key={name}>
              <path d={d} fill="none" stroke={OPT_COLOR[name]} strokeWidth={name === optimizer ? 2 : 1.2}
                strokeOpacity={name === optimizer ? 1 : 0.5} />
            </g>
          );
        })}
      </svg>
    );
  };

  // Network architecture diagram
  const renderNetwork = () => {
    if (!activations) return null;
    const { z1, h, p, sample } = activations;
    const W = 560, H = 280;
    // Layout: input column on left (small grid icon), hidden column middle, output column right
    const inputX = 60;
    const hiddenX = 280;
    const outputX = 500;
    const hiddenSpacing = (H - 40) / (HIDDEN - 1);
    const outputSpacing = (H - 40) / (OUTPUT - 1);

    // Find max h for normalization
    let maxH = 0.001;
    for (let i = 0; i < HIDDEN; i++) if (Math.abs(h[i]) > maxH) maxH = Math.abs(h[i]);

    const W1 = paramsRef.current?.W1;
    const W2 = paramsRef.current?.W2;
    if (!W1 || !W2) return null;

    // Compute per-hidden-unit input strength = sum_i W1[i,j] * x[i]
    // We'll only draw a subsample of edges to keep it readable
    let maxAbsW1 = 0.001, maxAbsW2 = 0.001;
    for (let i = 0; i < W1.length; i++) if (Math.abs(W1[i]) > maxAbsW1) maxAbsW1 = Math.abs(W1[i]);
    for (let i = 0; i < W2.length; i++) if (Math.abs(W2[i]) > maxAbsW2) maxAbsW2 = Math.abs(W2[i]);

    // Predicted class
    let pred = 0;
    for (let k = 1; k < OUTPUT; k++) if (p[k] > p[pred]) pred = k;
    const correct = pred === sample.y;

    return (
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        {/* Hidden -> Output edges */}
        {Array.from({ length: HIDDEN }, (_, j) =>
          Array.from({ length: OUTPUT }, (_, k) => {
            const w = W2[j * OUTPUT + k];
            const a = Math.abs(w) / maxAbsW2;
            if (a < 0.15) return null;
            const y1 = 20 + j * hiddenSpacing;
            const y2 = 20 + k * outputSpacing;
            return (
              <line key={`h${j}o${k}`} x1={hiddenX} y1={y1} x2={outputX} y2={y2}
                stroke={w > 0 ? '#7ed957' : '#e85d75'} strokeOpacity={a * 0.5} strokeWidth={0.8} />
            );
          })
        )}
        {/* Input -> Hidden edges (only top by magnitude per hidden unit) */}
        {Array.from({ length: HIDDEN }, (_, j) => {
          // gather weights for this hidden unit
          const weights = [];
          for (let i = 0; i < INPUT; i++) weights.push({ i, w: W1[i * HIDDEN + j] });
          weights.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
          const top = weights.slice(0, 6);
          return top.map(({ i, w }) => {
            const a = Math.abs(w) / maxAbsW1;
            const y1 = 40 + (i / 63) * (H - 80);
            const y2 = 20 + j * hiddenSpacing;
            return (
              <line key={`i${i}h${j}`} x1={inputX + 30} y1={y1} x2={hiddenX} y2={y2}
                stroke={w > 0 ? '#7ed957' : '#e85d75'} strokeOpacity={a * 0.4} strokeWidth={0.5} />
            );
          });
        })}

        {/* Input grid label */}
        <text x={inputX} y={14} style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 12, fill: '#1a1a1a' }}>
          input · 64
        </text>
        {/* Input grid as small 8x8 squares */}
        {Array.from({ length: 64 }, (_, i) => {
          const r = Math.floor(i / 8), c = i % 8;
          const cellSize = 5;
          return (
            <rect key={`in${i}`}
              x={inputX + c * cellSize} y={40 + r * cellSize}
              width={cellSize - 0.5} height={cellSize - 0.5}
              fill={sample.x[i] > 0.5 ? '#1a1a1a' : '#fafaf5'}
              stroke="rgba(0,0,0,0.1)" strokeWidth="0.3" />
          );
        })}

        {/* Hidden layer label */}
        <text x={hiddenX} y={14} textAnchor="middle" style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 12, fill: '#1a1a1a' }}>
          hidden · ReLU · 16
        </text>
        {/* Hidden neurons */}
        {Array.from({ length: HIDDEN }, (_, j) => {
          const y = 20 + j * hiddenSpacing;
          const a = h[j] / maxH;
          return (
            <g key={`hn${j}`}>
              <circle cx={hiddenX} cy={y} r={6 + a * 4}
                fill="#fafaf5" stroke="#1a1a1a" strokeWidth="1" />
              <circle cx={hiddenX} cy={y} r={Math.max(0, a * 6)}
                fill="#1a1a1a" />
            </g>
          );
        })}

        {/* Output label */}
        <text x={outputX} y={14} textAnchor="middle" style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 12, fill: '#1a1a1a' }}>
          softmax · 10
        </text>
        {/* Output neurons */}
        {Array.from({ length: OUTPUT }, (_, k) => {
          const y = 20 + k * outputSpacing;
          const isPred = k === pred;
          const isTrue = k === sample.y;
          return (
            <g key={`on${k}`}>
              {/* probability bar to the right */}
              <rect x={outputX + 14} y={y - 5} width={p[k] * 50} height={10}
                fill={isTrue ? '#7ed957' : (isPred && !correct ? '#e85d75' : '#1a1a1a')}
                opacity={0.85} />
              <circle cx={outputX} cy={y} r={8}
                fill={isPred ? (correct ? '#7ed957' : '#e85d75') : '#fafaf5'}
                stroke="#1a1a1a" strokeWidth={isTrue ? 2 : 1} />
              <text x={outputX} y={y + 3.5} textAnchor="middle"
                style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: isPred ? '#fafaf5' : '#1a1a1a', fontWeight: 600 }}>
                {k}
              </text>
              <text x={outputX + 70} y={y + 3.5}
                style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, fill: '#666' }}>
                {(p[k] * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  // Compute current displayed prediction
  const currentPred = activations ? (() => {
    const p = activations.p;
    let best = 0;
    for (let k = 1; k < OUTPUT; k++) if (p[k] > p[best]) best = k;
    return { pred: best, conf: p[best], correct: best === activations.sample.y };
  })() : null;

  const lastPoint = history[history.length - 1];

  return (
    <div style={{
      minHeight: '100vh',
      background: '#fafaf5',
      fontFamily: 'Inter, system-ui, sans-serif',
      color: '#1a1a1a',
      padding: '32px 24px',
    }}>
      {/* Google fonts */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,500;0,700;1,300;1,400;1,500&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
        @keyframes pulse-soft { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }
        .pulse { animation: pulse-soft 1.4s ease-in-out infinite; }
      `}</style>

      <div style={{ maxWidth: 1180, margin: '0 auto' }}>
        {/* Header */}
        <header style={{ marginBottom: 28, borderBottom: '1px solid #1a1a1a', paddingBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.15em', color: '#666', textTransform: 'uppercase', marginBottom: 6 }}>
                first-order methods · live training
              </div>
              <h1 style={{
                fontFamily: 'Fraunces, serif',
                fontWeight: 400,
                fontSize: 'clamp(32px, 5vw, 52px)',
                lineHeight: 1.05,
                margin: 0,
                letterSpacing: '-0.02em',
              }}>
                Watching a network <em style={{ fontStyle: 'italic' }}>learn</em> to read.
              </h1>
            </div>
            <div style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 11,
              color: '#666',
              maxWidth: 280,
              lineHeight: 1.5,
            }}>
              A 64→16→10 MLP with cross-entropy loss, trained in your browser. Switch optimizers to compare convergence.
            </div>
          </div>
        </header>

        {/* Top row: data + chart */}
        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.1fr)', gap: 28, marginBottom: 28 }}>
          {/* Digit gallery */}
          <section>
            <SectionLabel n="01" text="the data" />
            <p style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', color: '#444', fontSize: 14, margin: '8px 0 16px', lineHeight: 1.5 }}>
              Ten 8×8 digit prototypes (plus noisy variants — 40 samples total). Click one to inspect the network's belief about it.
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6, padding: 16, background: '#fff', border: '1px solid #1a1a1a' }}>
              {showcase.map((s, i) => (
                <div key={i}>
                  {renderDigit(s, i === displayIdx, () => setDisplayIdx(i))}
                </div>
              ))}
            </div>
            {currentPred && (
              <div style={{
                marginTop: 12,
                padding: 14,
                background: currentPred.correct ? '#e8f5e8' : '#fde8ec',
                border: `1px solid ${currentPred.correct ? '#7ed957' : '#e85d75'}`,
                fontFamily: 'JetBrains Mono, monospace',
                fontSize: 12,
              }}>
                <span style={{ color: '#666' }}>true = </span>
                <span style={{ fontWeight: 600 }}>{activations.sample.y}</span>
                <span style={{ color: '#666', margin: '0 8px' }}>·</span>
                <span style={{ color: '#666' }}>predicted = </span>
                <span style={{ fontWeight: 600, color: currentPred.correct ? '#2d7a3d' : '#a02838' }}>
                  {currentPred.pred}
                </span>
                <span style={{ color: '#666', margin: '0 8px' }}>·</span>
                <span style={{ color: '#666' }}>confidence = </span>
                <span style={{ fontWeight: 600 }}>{(currentPred.conf * 100).toFixed(1)}%</span>
              </div>
            )}
          </section>

          {/* Loss chart */}
          <section>
            <SectionLabel n="02" text="loss curves" />
            <p style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', color: '#444', fontSize: 14, margin: '8px 0 16px', lineHeight: 1.5 }}>
              Each color is a separate run. Reset to clear, then train with another optimizer to overlay.
            </p>
            <div style={{ background: '#fff', border: '1px solid #1a1a1a', padding: 8 }}>
              {renderChart()}
            </div>
            <div style={{ display: 'flex', gap: 16, marginTop: 12, flexWrap: 'wrap', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
              {OPTIMIZERS.map(o => (
                <div key={o} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: OPT_COLOR[o] }} />
                  <span style={{ color: o === optimizer ? '#1a1a1a' : '#999', fontWeight: o === optimizer ? 600 : 400 }}>
                    {o}
                  </span>
                </div>
              ))}
            </div>
          </section>
        </div>

        {/* Network diagram */}
        <section style={{ marginBottom: 28 }}>
          <SectionLabel n="03" text="the network" />
          <p style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', color: '#444', fontSize: 14, margin: '8px 0 16px', lineHeight: 1.5 }}>
            Green edges carry positive weight, red carry negative. Filled circles in the hidden layer show ReLU activation strength for the selected digit. Output bars are softmax probabilities.
          </p>
          <div style={{ background: '#fff', border: '1px solid #1a1a1a', padding: 16 }}>
            {renderNetwork()}
          </div>
        </section>

        {/* Controls */}
        <section style={{
          background: '#1a1a1a',
          color: '#fafaf5',
          padding: '24px 24px',
          marginBottom: 16,
        }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'baseline', marginBottom: 18 }}>
            <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#fafaf5', opacity: 0.5, letterSpacing: '0.1em' }}>
              04 ·
            </span>
            <h2 style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontWeight: 400, fontSize: 22, margin: 0 }}>
              controls
            </h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 24 }}>
            {/* Optimizer */}
            <div>
              <label style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', opacity: 0.6, display: 'block', marginBottom: 8 }}>
                optimizer
              </label>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {OPTIMIZERS.map(o => (
                  <button key={o} onClick={() => handleOptChange(o)}
                    style={{
                      padding: '8px 12px',
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: 11,
                      background: o === optimizer ? OPT_COLOR[o] : 'transparent',
                      color: o === optimizer ? '#1a1a1a' : '#fafaf5',
                      border: `1px solid ${o === optimizer ? OPT_COLOR[o] : '#fafaf5'}`,
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      letterSpacing: '0.05em',
                    }}>
                    {o}
                  </button>
                ))}
              </div>
              <div style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontSize: 12, opacity: 0.75, marginTop: 10, lineHeight: 1.5, minHeight: 36 }}>
                {OPT_INFO[optimizer]}
              </div>
            </div>

            {/* Learning rate */}
            <div>
              <label style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', opacity: 0.6, display: 'block', marginBottom: 8 }}>
                learning rate η
              </label>
              <input type="range" min={-3.5} max={0} step={0.05}
                value={Math.log10(lr)}
                onChange={e => setLr(Math.pow(10, parseFloat(e.target.value)))}
                style={{ width: '100%', accentColor: OPT_COLOR[optimizer] }} />
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, marginTop: 4, color: OPT_COLOR[optimizer] }}>
                {lr.toFixed(lr < 0.001 ? 5 : lr < 0.01 ? 4 : 3)}
              </div>
            </div>

            {/* Speed */}
            <div>
              <label style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', opacity: 0.6, display: 'block', marginBottom: 8 }}>
                steps / frame
              </label>
              <input type="range" min={1} max={80} step={1} value={speed}
                onChange={e => setSpeed(parseInt(e.target.value))}
                style={{ width: '100%', accentColor: OPT_COLOR[optimizer] }} />
              <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 16, marginTop: 4, color: OPT_COLOR[optimizer] }}>
                {speed}×
              </div>
            </div>

            {/* Actions */}
            <div>
              <label style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase', opacity: 0.6, display: 'block', marginBottom: 8 }}>
                run
              </label>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                <button onClick={() => setRunning(r => !r)}
                  style={{
                    padding: '10px 14px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 11,
                    background: running ? '#fafaf5' : OPT_COLOR[optimizer],
                    color: '#1a1a1a',
                    border: 'none',
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                    letterSpacing: '0.05em',
                  }}>
                  {running ? <Pause size={12} /> : <Play size={12} />}
                  {running ? 'pause' : 'train'}
                </button>
                <button onClick={handleReset}
                  style={{
                    padding: '10px 14px',
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: 11,
                    background: 'transparent',
                    color: '#fafaf5',
                    border: '1px solid #fafaf5',
                    cursor: 'pointer',
                    display: 'flex', alignItems: 'center', gap: 6,
                    letterSpacing: '0.05em',
                  }}>
                  <RotateCcw size={12} />
                  reset
                </button>
              </div>
            </div>
          </div>

          {/* Live stats */}
          <div style={{
            marginTop: 24, paddingTop: 18, borderTop: '1px solid rgba(250,250,245,0.15)',
            display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16,
          }}>
            <Stat label="step" value={step} mono />
            <Stat label="batch loss" value={lastPoint ? lastPoint.loss.toFixed(3) : '—'} mono color={OPT_COLOR[optimizer]} />
            <Stat label="batch acc" value={lastPoint ? `${(lastPoint.acc * 100).toFixed(0)}%` : '—'} mono />
            <Stat label="status" value={running ? 'training' : 'idle'} mono pulse={running} />
          </div>
        </section>

        {/* Footer / theory */}
        <section style={{ marginTop: 36, paddingTop: 24, borderTop: '1px solid #1a1a1a' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 12 }}>
            <Zap size={14} />
            <h3 style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontWeight: 400, fontSize: 22, margin: 0 }}>
              what's happening, exactly
            </h3>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 24, fontFamily: 'Inter, sans-serif', fontSize: 13, lineHeight: 1.65, color: '#333' }}>
            <div>
              <strong style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>forward pass</strong>
              <p style={{ marginTop: 6 }}>
                Each 8×8 image is flattened to a 64-vector, mapped to 16 hidden units via <em>h = ReLU(W₁x + b₁)</em>, then to 10 logits. Softmax turns logits into a probability simplex.
              </p>
            </div>
            <div>
              <strong style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>loss</strong>
              <p style={{ marginTop: 6 }}>
                Cross-entropy <em>L = −log p(y)</em> on the true class. Gradient at the output simplifies beautifully to <em>∂L/∂z₂ = p − e<sub>y</sub></em>.
              </p>
            </div>
            <div>
              <strong style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>backward pass</strong>
              <p style={{ marginTop: 6 }}>
                Backprop is just the chain rule. The optimizer chooses how to use the gradient — that's the only thing changing across the four buttons above.
              </p>
            </div>
            <div>
              <strong style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, letterSpacing: '0.1em', textTransform: 'uppercase' }}>why first-order?</strong>
              <p style={{ marginTop: 6 }}>
                For 64·16 + 16·10 ≈ 1.2k parameters, second-order methods are tractable, but in deep nets they're not. First-order methods scale linearly per step — that's the whole game.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}

function SectionLabel({ n, text }) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
      <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11, color: '#999', letterSpacing: '0.1em' }}>
        {n} ·
      </span>
      <h2 style={{ fontFamily: 'Fraunces, serif', fontStyle: 'italic', fontWeight: 400, fontSize: 22, margin: 0 }}>
        {text}
      </h2>
    </div>
  );
}

function Stat({ label, value, mono, color, pulse }) {
  return (
    <div>
      <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 9, letterSpacing: '0.15em', textTransform: 'uppercase', opacity: 0.5, marginBottom: 4 }}>
        {label}
      </div>
      <div className={pulse ? 'pulse' : ''}
        style={{
          fontFamily: mono ? 'JetBrains Mono, monospace' : 'Fraunces, serif',
          fontSize: 22,
          fontWeight: 500,
          color: color || '#fafaf5',
          letterSpacing: '-0.01em',
        }}>
        {value}
      </div>
    </div>
  );
}
