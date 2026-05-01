import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { Play, Pause, RotateCcw, StepForward, MousePointer2, Activity } from "lucide-react";

/* ============================================================
   FIRST-ORDER METHODS — INTERACTIVE COMPARISON
   For ISE 5405/5406. Built around 2D test functions so students
   can see trajectories, conditioning effects, and momentum.
   ============================================================ */

// ---------- Test functions: f, grad f, recommended view box, optimum ----------
const FUNCTIONS = {
  quad_ill: {
    name: "Ill-Conditioned Quadratic",
    formula: "f(x,y) = ½(x² + 25y²)",
    note: "Condition number κ = 25. Classic zig-zag for steepest descent.",
    f: (x, y) => 0.5 * (x * x + 25 * y * y),
    g: (x, y) => [x, 25 * y],
    box: [-5, 5, -2, 2],
    opt: [0, 0],
    start: [-4, 1.5],
    levels: [0.05, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64],
    // Per-function tuned hyperparameters. Defaults already work here.
    tuned: {},
  },
  quad_well: {
    name: "Well-Conditioned Quadratic",
    formula: "f(x,y) = ½(x² + y²)",
    note: "κ = 1. All first-order methods reach the optimum in one step (with line search).",
    f: (x, y) => 0.5 * (x * x + y * y),
    g: (x, y) => [x, y],
    box: [-5, 5, -5, 5],
    opt: [0, 0],
    start: [-4, 3],
    levels: [0.5, 2, 4.5, 8, 12.5, 18, 24.5, 32],
    tuned: {
      gd: { lr: 0.5 }, // optimal lr = 1 here; 0.5 shows convergence cleanly
      heavyball: { lr: 0.3 },
      nesterov: { lr: 0.3 },
    },
  },
  rosenbrock: {
    name: "Rosenbrock",
    formula: "f(x,y) = (1−x)² + 100(y−x²)²",
    note: "Banana-shaped narrow valley. Tests momentum and adaptive methods.",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: (x, y) => [
      -2 * (1 - x) - 400 * x * (y - x * x),
      200 * (y - x * x),
    ],
    box: [-2, 2, -1, 3],
    opt: [1, 1],
    start: [-1.5, 2],
    levels: [1, 5, 20, 50, 100, 200, 400, 800, 1500, 3000],
    // Rosenbrock has huge gradient norms — needs small lr or it explodes.
    tuned: {
      gd: { lr: 0.0015 },
      heavyball: { lr: 0.0008, momentum: 0.85 },
      nesterov: { lr: 0.0006, momentum: 0.88 },
      adagrad: { lr: 0.5 },
      rmsprop: { lr: 0.02, beta: 0.9 },
      adam: { lr: 0.05, b1: 0.9, b2: 0.999 },
    },
  },
  himmelblau: {
    name: "Himmelblau",
    formula: "f = (x²+y−11)² + (x+y²−7)²",
    note: "Four global minima. Trajectory depends strongly on start.",
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    g: (x, y) => [
      4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
      2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
    ],
    box: [-5, 5, -5, 5],
    opt: [3, 2],
    start: [0, 0],
    levels: [1, 5, 15, 40, 80, 150, 300, 600, 1200],
    tuned: {
      gd: { lr: 0.005 },
      heavyball: { lr: 0.003, momentum: 0.85 },
      nesterov: { lr: 0.004, momentum: 0.9 },
      rmsprop: { lr: 0.05, beta: 0.9 },
      adam: { lr: 0.1 },
    },
  },
  beale: {
    name: "Beale",
    formula: "(1.5−x+xy)² + (2.25−x+xy²)² + (2.625−x+xy³)²",
    note: "Sharp narrow valley off-axis. Min at (3, 0.5).",
    f: (x, y) => {
      const a = 1.5 - x + x * y;
      const b = 2.25 - x + x * y * y;
      const c = 2.625 - x + x * y * y * y;
      return a * a + b * b + c * c;
    },
    g: (x, y) => {
      const a = 1.5 - x + x * y;
      const b = 2.25 - x + x * y * y;
      const c = 2.625 - x + x * y * y * y;
      const dx = 2 * a * (y - 1) + 2 * b * (y * y - 1) + 2 * c * (y * y * y - 1);
      const dy = 2 * a * x + 2 * b * (2 * x * y) + 2 * c * (3 * x * y * y);
      return [dx, dy];
    },
    box: [-1, 4.5, -1.5, 1.5],
    opt: [3, 0.5],
    start: [2.5, 1.2],
    levels: [0.1, 1, 5, 15, 40, 100, 250, 600, 1500],
    // Beale has wildly varying gradient scale. Needs cautious step sizes.
    tuned: {
      gd: { lr: 0.005 },
      heavyball: { lr: 0.002, momentum: 0.85 },
      nesterov: { lr: 0.003, momentum: 0.9 },
      adagrad: { lr: 0.3 },
      rmsprop: { lr: 0.02, beta: 0.9 },
      adam: { lr: 0.05 },
    },
  },
  saddle: {
    name: "Saddle",
    formula: "f(x,y) = x² − y²",
    note: "Saddle at origin. With a tiny y-perturbation, watch escape rates differ.",
    f: (x, y) => x * x - y * y,
    g: (x, y) => [2 * x, -2 * y],
    box: [-3, 3, -3, 3],
    opt: null,
    start: [-2, 0.05],
    levels: [-8, -4, -2, -1, -0.25, 0.25, 1, 2, 4, 8],
    // Saddle's y direction has unbounded descent — keep step sizes small
    // so trajectories stay on screen and the comparison stays meaningful.
    tuned: {
      gd: { lr: 0.02 },
      heavyball: { lr: 0.01, momentum: 0.7 },
      nesterov: { lr: 0.01, momentum: 0.7 },
      adagrad: { lr: 0.3 },
      rmsprop: { lr: 0.05, beta: 0.9 },
      adam: { lr: 0.05 },
    },
  },
};

// ---------- Algorithms ----------
// Each returns: { name, color, init(x), step(state, gradFn) -> newState }
// state = { x: [..], iter, ...persistent }

const ALGOS = {
  gd: {
    name: "Gradient Descent",
    short: "GD",
    color: "#d4453d",
    params: { lr: 0.02 },
    init: (x0) => ({ x: x0.slice(), iter: 0 }),
    step: (s, g, p) => {
      const grad = g(s.x[0], s.x[1]);
      return {
        x: [s.x[0] - p.lr * grad[0], s.x[1] - p.lr * grad[1]],
        iter: s.iter + 1,
      };
    },
  },
  heavyball: {
    name: "Heavy Ball (Polyak)",
    short: "HB",
    color: "#e8843a",
    params: { lr: 0.015, momentum: 0.9 },
    init: (x0) => ({ x: x0.slice(), v: [0, 0], iter: 0 }),
    step: (s, g, p) => {
      const grad = g(s.x[0], s.x[1]);
      const v = [
        p.momentum * s.v[0] - p.lr * grad[0],
        p.momentum * s.v[1] - p.lr * grad[1],
      ];
      return { x: [s.x[0] + v[0], s.x[1] + v[1]], v, iter: s.iter + 1 };
    },
  },
  nesterov: {
    name: "Nesterov Accelerated",
    short: "NAG",
    color: "#c9a227",
    params: { lr: 0.015, momentum: 0.9 },
    init: (x0) => ({ x: x0.slice(), v: [0, 0], iter: 0 }),
    step: (s, g, p) => {
      // Lookahead point
      const lx = s.x[0] + p.momentum * s.v[0];
      const ly = s.x[1] + p.momentum * s.v[1];
      const grad = g(lx, ly);
      const v = [
        p.momentum * s.v[0] - p.lr * grad[0],
        p.momentum * s.v[1] - p.lr * grad[1],
      ];
      return { x: [s.x[0] + v[0], s.x[1] + v[1]], v, iter: s.iter + 1 };
    },
  },
  adagrad: {
    name: "AdaGrad",
    short: "AG",
    color: "#3f8c5c",
    params: { lr: 0.5 },
    init: (x0) => ({ x: x0.slice(), G: [1e-12, 1e-12], iter: 0 }),
    step: (s, g, p) => {
      const grad = g(s.x[0], s.x[1]);
      const G = [s.G[0] + grad[0] * grad[0], s.G[1] + grad[1] * grad[1]];
      return {
        x: [
          s.x[0] - (p.lr / Math.sqrt(G[0] + 1e-8)) * grad[0],
          s.x[1] - (p.lr / Math.sqrt(G[1] + 1e-8)) * grad[1],
        ],
        G,
        iter: s.iter + 1,
      };
    },
  },
  rmsprop: {
    name: "RMSProp",
    short: "RMS",
    color: "#3a7ca5",
    params: { lr: 0.05, beta: 0.9 },
    init: (x0) => ({ x: x0.slice(), G: [1e-12, 1e-12], iter: 0 }),
    step: (s, g, p) => {
      const grad = g(s.x[0], s.x[1]);
      const G = [
        p.beta * s.G[0] + (1 - p.beta) * grad[0] * grad[0],
        p.beta * s.G[1] + (1 - p.beta) * grad[1] * grad[1],
      ];
      return {
        x: [
          s.x[0] - (p.lr / Math.sqrt(G[0] + 1e-8)) * grad[0],
          s.x[1] - (p.lr / Math.sqrt(G[1] + 1e-8)) * grad[1],
        ],
        G,
        iter: s.iter + 1,
      };
    },
  },
  adam: {
    name: "Adam",
    short: "ADAM",
    color: "#7a4ba5",
    params: { lr: 0.1, b1: 0.9, b2: 0.999 },
    init: (x0) => ({ x: x0.slice(), m: [0, 0], v: [0, 0], iter: 0 }),
    step: (s, g, p) => {
      const grad = g(s.x[0], s.x[1]);
      const t = s.iter + 1;
      const m = [
        p.b1 * s.m[0] + (1 - p.b1) * grad[0],
        p.b1 * s.m[1] + (1 - p.b1) * grad[1],
      ];
      const v = [
        p.b2 * s.v[0] + (1 - p.b2) * grad[0] * grad[0],
        p.b2 * s.v[1] + (1 - p.b2) * grad[1] * grad[1],
      ];
      const mh = [m[0] / (1 - Math.pow(p.b1, t)), m[1] / (1 - Math.pow(p.b1, t))];
      const vh = [v[0] / (1 - Math.pow(p.b2, t)), v[1] / (1 - Math.pow(p.b2, t))];
      return {
        x: [
          s.x[0] - (p.lr * mh[0]) / (Math.sqrt(vh[0]) + 1e-8),
          s.x[1] - (p.lr * mh[1]) / (Math.sqrt(vh[1]) + 1e-8),
        ],
        m,
        v,
        iter: t,
      };
    },
  },
};

const ALGO_KEYS = Object.keys(ALGOS);

// Build params for a function: per-algorithm defaults overlaid with the
// function's tuned overrides.
function paramsFor(fnKey) {
  const fdef = FUNCTIONS[fnKey];
  const out = {};
  for (const k of ALGO_KEYS) {
    out[k] = { ...ALGOS[k].params, ...(fdef.tuned[k] || {}) };
  }
  return out;
}

// ---------- Curated presets per (function, algo) ----------
// kind ∈ {"good", "bad", "slow"}. Click a chip to splat values into params[k].
// The "good" entry doubles as the tuned default; "bad" demonstrates breakdown
// (divergence, oscillation); "slow" shows the other failure mode.
const PRESETS = {
  quad_ill: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.02 }, tip: "stable zigzag along the steep y-axis" },
      { name: "near-optimal", kind: "good", params: { lr: 0.075 }, tip: "lr ≈ 2/(L+m) = 2/26 — fastest stable rate" },
      { name: "diverges", kind: "bad", params: { lr: 0.085 }, tip: "lr > 2/L = 0.08 — y component blows up" },
      { name: "slow", kind: "slow", params: { lr: 0.003 }, tip: "step too small — many iters to reach origin" },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.015, momentum: 0.9 } },
      { name: "high mom", kind: "bad", params: { lr: 0.03, momentum: 0.97 }, tip: "momentum overshoots, oscillates wildly" },
      { name: "no mom", kind: "slow", params: { lr: 0.015, momentum: 0.0 }, tip: "degenerates to plain GD" },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.015, momentum: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.05, momentum: 0.95 } },
      { name: "weak mom", kind: "slow", params: { lr: 0.015, momentum: 0.3 } },
    ],
    adagrad: [
      { name: "tuned", kind: "good", params: { lr: 0.5 } },
      { name: "aggressive", kind: "good", params: { lr: 3.0 }, tip: "AdaGrad self-normalizes — large lr is OK" },
      { name: "tiny", kind: "slow", params: { lr: 0.05 } },
    ],
    rmsprop: [
      { name: "tuned", kind: "good", params: { lr: 0.05, beta: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.6, beta: 0.9 } },
      { name: "low beta", kind: "bad", params: { lr: 0.05, beta: 0.3 }, tip: "near-AdaGrad, but with high lr unstable" },
    ],
    adam: [
      { name: "tuned", kind: "good", params: { lr: 0.1, b1: 0.9, b2: 0.999 } },
      { name: "diverges", kind: "bad", params: { lr: 1.5, b1: 0.9, b2: 0.999 } },
      { name: "low b1", kind: "good", params: { lr: 0.1, b1: 0.3, b2: 0.999 }, tip: "less momentum — closer to RMSProp" },
    ],
  },
  quad_well: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.5 } },
      { name: "optimal", kind: "good", params: { lr: 1.0 }, tip: "lr = 1/L — converges in one step from any start" },
      { name: "diverges", kind: "bad", params: { lr: 2.05 }, tip: "lr > 2/L — bounces away" },
      { name: "slow", kind: "slow", params: { lr: 0.05 } },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.3, momentum: 0.9 } },
      { name: "oscillates", kind: "bad", params: { lr: 0.3, momentum: 0.99 }, tip: "near-unit momentum sustains overshoot" },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.3, momentum: 0.9 } },
      { name: "oscillates", kind: "bad", params: { lr: 0.5, momentum: 0.95 } },
    ],
    adagrad: [{ name: "tuned", kind: "good", params: { lr: 0.5 } }],
    rmsprop: [{ name: "tuned", kind: "good", params: { lr: 0.05, beta: 0.9 } }],
    adam: [{ name: "tuned", kind: "good", params: { lr: 0.1, b1: 0.9, b2: 0.999 } }],
  },
  rosenbrock: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.0015 }, tip: "creeps along the valley floor — converges very slowly" },
      { name: "diverges", kind: "bad", params: { lr: 0.005 }, tip: "valley curvature kicks the iterate out, gradient explodes" },
      { name: "slow", kind: "slow", params: { lr: 0.0003 } },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.0008, momentum: 0.85 } },
      { name: "diverges", kind: "bad", params: { lr: 0.003, momentum: 0.9 } },
      { name: "high mom", kind: "bad", params: { lr: 0.0008, momentum: 0.99 }, tip: "velocity outruns the valley — loops or diverges" },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.0006, momentum: 0.88 } },
      { name: "diverges", kind: "bad", params: { lr: 0.002, momentum: 0.9 }, tip: "lookahead lands in steep wall — gradient explodes" },
      { name: "high mom", kind: "bad", params: { lr: 0.0006, momentum: 0.99 } },
    ],
    adagrad: [
      { name: "tuned", kind: "good", params: { lr: 0.5 } },
      { name: "aggressive", kind: "good", params: { lr: 3.0 } },
    ],
    rmsprop: [
      { name: "tuned", kind: "good", params: { lr: 0.02, beta: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.2, beta: 0.9 } },
    ],
    adam: [
      { name: "tuned", kind: "good", params: { lr: 0.05, b1: 0.9, b2: 0.999 } },
      { name: "diverges", kind: "bad", params: { lr: 0.5, b1: 0.9, b2: 0.999 } },
      { name: "low b2", kind: "bad", params: { lr: 0.05, b1: 0.9, b2: 0.5 }, tip: "stale 2nd moment makes step erratic" },
    ],
  },
  himmelblau: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.005 } },
      { name: "diverges", kind: "bad", params: { lr: 0.02 }, tip: "Himmelblau's 4th-order growth amplifies large steps" },
      { name: "slow", kind: "slow", params: { lr: 0.001 } },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.003, momentum: 0.85 } },
      { name: "diverges", kind: "bad", params: { lr: 0.02, momentum: 0.9 } },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.004, momentum: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.02, momentum: 0.9 } },
    ],
    adagrad: [{ name: "tuned", kind: "good", params: { lr: 0.5 } }],
    rmsprop: [
      { name: "tuned", kind: "good", params: { lr: 0.05, beta: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.5, beta: 0.9 } },
    ],
    adam: [
      { name: "tuned", kind: "good", params: { lr: 0.1, b1: 0.9, b2: 0.999 } },
      { name: "diverges", kind: "bad", params: { lr: 1.0, b1: 0.9, b2: 0.999 } },
    ],
  },
  beale: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.005 } },
      { name: "diverges", kind: "bad", params: { lr: 0.02 }, tip: "Beale's gradient is huge off-valley — small lr is mandatory" },
      { name: "slow", kind: "slow", params: { lr: 0.001 } },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.002, momentum: 0.85 } },
      { name: "diverges", kind: "bad", params: { lr: 0.01, momentum: 0.9 } },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.003, momentum: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.015, momentum: 0.9 } },
    ],
    adagrad: [{ name: "tuned", kind: "good", params: { lr: 0.3 } }],
    rmsprop: [
      { name: "tuned", kind: "good", params: { lr: 0.02, beta: 0.9 } },
      { name: "diverges", kind: "bad", params: { lr: 0.2, beta: 0.9 } },
    ],
    adam: [
      { name: "tuned", kind: "good", params: { lr: 0.05, b1: 0.9, b2: 0.999 } },
      { name: "diverges", kind: "bad", params: { lr: 0.5, b1: 0.9, b2: 0.999 } },
    ],
  },
  saddle: {
    gd: [
      { name: "tuned", kind: "good", params: { lr: 0.02 }, tip: "with y₀ ≠ 0, escapes the saddle exponentially" },
      { name: "explodes", kind: "bad", params: { lr: 0.5 }, tip: "lr > 1 — |1+2lr|>1 in y, blows up to ±∞" },
      { name: "slow escape", kind: "slow", params: { lr: 0.005 } },
    ],
    heavyball: [
      { name: "tuned", kind: "good", params: { lr: 0.01, momentum: 0.7 } },
      { name: "explodes", kind: "bad", params: { lr: 0.05, momentum: 0.9 } },
    ],
    nesterov: [
      { name: "tuned", kind: "good", params: { lr: 0.01, momentum: 0.7 } },
      { name: "explodes", kind: "bad", params: { lr: 0.05, momentum: 0.9 } },
    ],
    adagrad: [{ name: "tuned", kind: "good", params: { lr: 0.3 } }],
    rmsprop: [
      { name: "tuned", kind: "good", params: { lr: 0.05, beta: 0.9 } },
      { name: "explodes", kind: "bad", params: { lr: 0.5, beta: 0.9 } },
    ],
    adam: [
      { name: "tuned", kind: "good", params: { lr: 0.05, b1: 0.9, b2: 0.999 } },
      { name: "explodes", kind: "bad", params: { lr: 0.5, b1: 0.9, b2: 0.999 } },
    ],
  },
};

// ---------- Contour drawing ----------
function drawContours(ctx, fn, box, levels, W, H) {
  const [xmin, xmax, ymin, ymax] = box;
  // Heatmap background
  const img = ctx.createImageData(W, H);
  let fmin = Infinity,
    fmax = -Infinity;
  const grid = new Float32Array(W * H);
  for (let py = 0; py < H; py++) {
    const y = ymax - (py / (H - 1)) * (ymax - ymin);
    for (let px = 0; px < W; px++) {
      const x = xmin + (px / (W - 1)) * (xmax - xmin);
      let v = fn(x, y);
      if (!isFinite(v)) v = 1e9;
      grid[py * W + px] = v;
      if (v < fmin) fmin = v;
      if (v > fmax) fmax = v;
    }
  }
  // Log-ish normalize for color
  const shift = fmin < 0 ? -fmin + 1 : 0;
  const lmin = Math.log(fmin + shift + 1e-9);
  const lmax = Math.log(fmax + shift + 1e-9);
  for (let i = 0; i < W * H; i++) {
    const v = grid[i];
    let t = (Math.log(v + shift + 1e-9) - lmin) / (lmax - lmin + 1e-12);
    t = Math.max(0, Math.min(1, t));
    // Cool→warm muted scheme: deep navy → cream
    const r = Math.round(20 + 220 * Math.pow(t, 1.4));
    const g = Math.round(28 + 200 * Math.pow(t, 1.6));
    const b = Math.round(48 + 150 * (1 - Math.pow(1 - t, 2)));
    const idx = i * 4;
    img.data[idx] = r;
    img.data[idx + 1] = g;
    img.data[idx + 2] = b;
    img.data[idx + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  // Contour lines via marching squares per level
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  for (const L of levels) {
    ctx.beginPath();
    for (let py = 0; py < H - 1; py += 2) {
      for (let px = 0; px < W - 1; px += 2) {
        const a = grid[py * W + px];
        const b = grid[py * W + px + 2];
        const c = grid[(py + 2) * W + px + 2];
        const d = grid[(py + 2) * W + px];
        const seg = (v1, v2, x1, y1, x2, y2) => {
          if ((v1 < L && v2 >= L) || (v1 >= L && v2 < L)) {
            const t = (L - v1) / (v2 - v1);
            return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
          }
          return null;
        };
        const pts = [];
        let s = seg(a, b, px, py, px + 2, py);
        if (s) pts.push(s);
        s = seg(b, c, px + 2, py, px + 2, py + 2);
        if (s) pts.push(s);
        s = seg(c, d, px + 2, py + 2, px, py + 2);
        if (s) pts.push(s);
        s = seg(d, a, px, py + 2, px, py);
        if (s) pts.push(s);
        if (pts.length >= 2) {
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
        }
      }
    }
    ctx.stroke();
  }
}

// ---------- Coordinate helpers ----------
function makeMap(box, W, H) {
  const [xmin, xmax, ymin, ymax] = box;
  return {
    toPx: (x, y) => [
      ((x - xmin) / (xmax - xmin)) * (W - 1),
      ((ymax - y) / (ymax - ymin)) * (H - 1),
    ],
    toData: (px, py) => [
      xmin + (px / (W - 1)) * (xmax - xmin),
      ymax - (py / (H - 1)) * (ymax - ymin),
    ],
  };
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function OptimDemo() {
  const [fnKey, setFnKey] = useState("quad_ill");
  const fnDef = FUNCTIONS[fnKey];

  const [start, setStart] = useState(fnDef.start);
  const [enabled, setEnabled] = useState({
    gd: true,
    heavyball: true,
    nesterov: true,
    adagrad: false,
    rmsprop: true,
    adam: true,
  });
  const [params, setParams] = useState(() => paramsFor("quad_ill"));
  const [maxIter, setMaxIter] = useState(200);
  const [speed, setSpeed] = useState(20); // iters per second
  const [running, setRunning] = useState(false);
  const [traj, setTraj] = useState(() => initTraj("quad_ill", fnDef.start));

  function initTraj(key, x0) {
    const fdef = FUNCTIONS[key];
    const out = {};
    for (const a of ALGO_KEYS) {
      const s0 = ALGOS[a].init(x0);
      out[a] = {
        states: [s0],
        path: [[x0[0], x0[1]]],
        fvals: [fdef.f(x0[0], x0[1])],
        gnorms: [Math.hypot(...fdef.g(x0[0], x0[1]))],
        diverged: false,
      };
    }
    return out;
  }

  // Reset trajectories whenever fn or start changes
  useEffect(() => {
    setTraj(initTraj(fnKey, start));
    setRunning(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fnKey, start[0], start[1]]);

  // Animation loop
  const rafRef = useRef(null);
  const lastStepTime = useRef(0);
  useEffect(() => {
    if (!running) return;
    const tick = (ts) => {
      const dt = ts - lastStepTime.current;
      const interval = 1000 / speed;
      if (dt >= interval) {
        lastStepTime.current = ts;
        setTraj((prev) => advanceAll(prev, fnDef, params, maxIter, enabled));
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    lastStepTime.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [running, speed, fnDef, params, maxIter, enabled]);

  // Auto-stop when everyone converged or hit maxIter
  useEffect(() => {
    if (!running) return;
    const allDone = ALGO_KEYS.filter((k) => enabled[k]).every((k) => {
      const t = traj[k];
      if (!t) return true;
      const last = t.states[t.states.length - 1];
      return (
        last.iter >= maxIter ||
        t.diverged ||
        t.gnorms[t.gnorms.length - 1] < 1e-6
      );
    });
    if (allDone) setRunning(false);
  }, [traj, running, enabled, maxIter]);

  function advanceAll(prev, fdef, params, maxIter, enabled) {
    const next = { ...prev };
    for (const a of ALGO_KEYS) {
      if (!enabled[a]) continue;
      const t = next[a];
      if (!t || t.diverged) continue;
      const last = t.states[t.states.length - 1];
      if (last.iter >= maxIter) continue;
      if (t.gnorms[t.gnorms.length - 1] < 1e-6) continue;
      let s2;
      try {
        s2 = ALGOS[a].step(last, fdef.g, params[a]);
      } catch (e) {
        next[a] = { ...t, diverged: true };
        continue;
      }
      const fv = fdef.f(s2.x[0], s2.x[1]);
      const gv = Math.hypot(...fdef.g(s2.x[0], s2.x[1]));
      if (
        !isFinite(s2.x[0]) ||
        !isFinite(s2.x[1]) ||
        Math.abs(s2.x[0]) > 1e6 ||
        Math.abs(s2.x[1]) > 1e6
      ) {
        next[a] = { ...t, diverged: true };
        continue;
      }
      next[a] = {
        states: [...t.states, s2],
        path: [...t.path, [s2.x[0], s2.x[1]]],
        fvals: [...t.fvals, fv],
        gnorms: [...t.gnorms, gv],
        diverged: false,
      };
    }
    return next;
  }

  function singleStep() {
    setTraj((p) => advanceAll(p, fnDef, params, maxIter, enabled));
  }
  function reset() {
    setRunning(false);
    setTraj(initTraj(fnKey, start));
  }

  // ------------- Canvas rendering -------------
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const W = 720,
    H = 520;
  const map = useMemo(() => makeMap(fnDef.box, W, H), [fnDef.box]);

  // Render contours when fn changes
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    drawContours(ctx, fnDef.f, fnDef.box, fnDef.levels, W, H);
  }, [fnDef]);

  // Render trajectories overlay every traj update
  useEffect(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    // Optimum marker
    if (fnDef.opt) {
      const [ox, oy] = map.toPx(fnDef.opt[0], fnDef.opt[1]);
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(ox - 6, oy);
      ctx.lineTo(ox + 6, oy);
      ctx.moveTo(ox, oy - 6);
      ctx.lineTo(ox, oy + 6);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(ox, oy, 4, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Start marker
    const [sx, sy] = map.toPx(start[0], start[1]);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.strokeStyle = "rgba(0,0,0,0.5)";
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Trajectories
    for (const a of ALGO_KEYS) {
      if (!enabled[a]) continue;
      const t = traj[a];
      if (!t || t.path.length < 1) continue;
      ctx.strokeStyle = ALGOS[a].color;
      ctx.lineWidth = 2;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < t.path.length; i++) {
        const [x, y] = t.path[i];
        const [px, py] = map.toPx(x, y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
      // Current point
      const [lx, ly] = t.path[t.path.length - 1];
      const [lpx, lpy] = map.toPx(lx, ly);
      ctx.fillStyle = ALGOS[a].color;
      ctx.beginPath();
      ctx.arc(lpx, lpy, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(0,0,0,0.6)";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }, [traj, enabled, fnDef, start, map]);

  // ------------- Click handler: set start -------------
  const handleCanvasClick = (e) => {
    const rect = overlayRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    const [x, y] = map.toData(px, py);
    setStart([x, y]);
  };

  // ------------- Convergence chart (f-value vs iter, log scale) -------------
  const chartRef = useRef(null);
  useEffect(() => {
    const c = chartRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const CW = c.width,
      CH = c.height;
    ctx.clearRect(0, 0, CW, CH);
    // Background
    ctx.fillStyle = "#1a1d28";
    ctx.fillRect(0, 0, CW, CH);

    const pad = { l: 44, r: 12, t: 14, b: 28 };
    const plotW = CW - pad.l - pad.r;
    const plotH = CH - pad.t - pad.b;

    // Find ranges. Plot ‖∇f‖ — always positive, the canonical first-order
    // metric, and works on the saddle (where f → −∞).
    let maxIt = 1;
    let fmin = Infinity,
      fmax = -Infinity;
    for (const a of ALGO_KEYS) {
      if (!enabled[a]) continue;
      const t = traj[a];
      if (!t) continue;
      maxIt = Math.max(maxIt, t.gnorms.length - 1);
      for (const v of t.gnorms) {
        if (v > 1e-15 && v < fmin) fmin = v;
        if (v > fmax && isFinite(v)) fmax = v;
      }
    }
    if (!isFinite(fmin) || !isFinite(fmax) || fmax <= 0) {
      fmin = 1e-6;
      fmax = 1;
    }
    // Ensure at least one decade of range so labels render
    if (fmax / fmin < 10) fmax = fmin * 10;
    const lmin = Math.log10(Math.max(fmin, 1e-12));
    const lmax = Math.log10(Math.max(fmax, fmin * 10));

    const xPx = (i) => pad.l + (i / Math.max(maxIt, 1)) * plotW;
    const yPx = (v) => {
      const lv = Math.log10(Math.max(v, 1e-12));
      return pad.t + (1 - (lv - lmin) / (lmax - lmin + 1e-12)) * plotH;
    };

    // Grid
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad.t + (g / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
    }
    // Axes
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.stroke();

    // Y-axis labels (log10 ticks)
    ctx.fillStyle = "rgba(230,225,210,0.65)";
    ctx.font = "10px ui-monospace, Menlo, monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const span = lmax - lmin;
    const step = span > 6 ? 2 : 1;
    for (let l = Math.ceil(lmin); l <= Math.floor(lmax); l += step) {
      const y = pad.t + (1 - (l - lmin) / (span + 1e-12)) * plotH;
      ctx.fillText(`10${superscript(l)}`, pad.l - 6, y);
      ctx.strokeStyle = "rgba(255,255,255,0.05)";
      ctx.beginPath();
      ctx.moveTo(pad.l, y);
      ctx.lineTo(pad.l + plotW, y);
      ctx.stroke();
    }
    // X-axis labels
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let k = 0; k <= 4; k++) {
      const it = Math.round((k / 4) * maxIt);
      const x = pad.l + (k / 4) * plotW;
      ctx.fillText(it, x, pad.t + plotH + 6);
    }
    // Axis titles
    ctx.fillStyle = "rgba(230,225,210,0.85)";
    ctx.font = "11px ui-monospace, Menlo, monospace";
    ctx.textAlign = "center";
    ctx.fillText("iteration", pad.l + plotW / 2, CH - 8);
    ctx.save();
    ctx.translate(11, pad.t + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("‖∇f‖", 0, 0);
    ctx.restore();

    // Curves
    for (const a of ALGO_KEYS) {
      if (!enabled[a]) continue;
      const t = traj[a];
      if (!t || t.gnorms.length < 2) continue;
      ctx.strokeStyle = ALGOS[a].color;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i < t.gnorms.length; i++) {
        const v = t.gnorms[i];
        const x = xPx(i);
        const y = yPx(Math.max(v, 1e-12));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [traj, enabled, fnDef]);

  // ------------- UI -------------
  return (
    <div style={S.root}>
      <style>{globalCss}</style>

      <header style={S.header}>
        <div>
          <div style={S.eyebrow}>ISE 5405 / 5406 — TEACHING DEMO</div>
          <h1 style={S.title}>First-Order Methods, Side by Side</h1>
          <div style={S.sub}>
            Pick a function. Click the contour plot to set <em>x⁰</em>. Press play.
          </div>
        </div>
        <div style={S.legend}>
          {ALGO_KEYS.map((k) => (
            <button
              key={k}
              onClick={() => setEnabled((e) => ({ ...e, [k]: !e[k] }))}
              style={{
                ...S.legendBtn,
                opacity: enabled[k] ? 1 : 0.35,
                borderColor: ALGOS[k].color,
              }}
              title={ALGOS[k].name}
            >
              <span style={{ ...S.legendSwatch, background: ALGOS[k].color }} />
              <span style={S.legendName}>{ALGOS[k].short}</span>
            </button>
          ))}
        </div>
      </header>

      <main style={S.main}>
        {/* LEFT PANEL — function & params */}
        <aside style={S.left}>
          <Section title="Test Function">
            <select
              value={fnKey}
              onChange={(e) => {
                const k = e.target.value;
                setFnKey(k);
                setStart(FUNCTIONS[k].start);
                setParams(paramsFor(k));
              }}
              style={S.select}
            >
              {Object.entries(FUNCTIONS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
            <div style={S.formula}>{fnDef.formula}</div>
            <div style={S.note}>{fnDef.note}</div>
            {fnDef.opt && (
              <div style={S.statRow}>
                <span style={S.statKey}>x*</span>
                <span style={S.statVal}>
                  ({fnDef.opt[0].toFixed(2)}, {fnDef.opt[1].toFixed(2)})
                </span>
              </div>
            )}
            <div style={S.statRow}>
              <span style={S.statKey}>x⁰</span>
              <span style={S.statVal}>
                ({start[0].toFixed(3)}, {start[1].toFixed(3)})
              </span>
            </div>
          </Section>

          <Section title="Run Control">
            <div style={S.controlRow}>
              <button
                style={{ ...S.btn, ...S.btnPrimary }}
                onClick={() => setRunning((r) => !r)}
              >
                {running ? <Pause size={14} /> : <Play size={14} />}
                {running ? "Pause" : "Play"}
              </button>
              <button style={S.btn} onClick={singleStep} disabled={running}>
                <StepForward size={14} /> Step
              </button>
              <button style={S.btn} onClick={reset}>
                <RotateCcw size={14} /> Reset
              </button>
            </div>
            <Slider
              label="Max iterations"
              value={maxIter}
              min={20}
              max={2000}
              step={10}
              onChange={setMaxIter}
              fmt={(v) => v.toString()}
            />
            <Slider
              label="Animation speed"
              value={speed}
              min={1}
              max={120}
              step={1}
              onChange={setSpeed}
              fmt={(v) => `${v} it/s`}
            />
          </Section>

          <Section title="Per-Algorithm Hyperparameters">
            {ALGO_KEYS.map((k) => (
              <div
                key={k}
                style={{
                  ...S.algoBlock,
                  borderLeftColor: ALGOS[k].color,
                  opacity: enabled[k] ? 1 : 0.4,
                }}
              >
                <div style={S.algoHead}>
                  <span style={S.algoName}>{ALGOS[k].name}</span>
                </div>
                {PRESETS[fnKey] && PRESETS[fnKey][k] && (
                  <div style={S.presetRow}>
                    {PRESETS[fnKey][k].map((preset) => (
                      <button
                        key={preset.name}
                        style={{ ...S.presetBtn, ...S.presetKind[preset.kind] }}
                        title={preset.tip || preset.name}
                        onClick={() =>
                          setParams((p) => ({
                            ...p,
                            [k]: {
                              ...ALGOS[k].params,
                              ...(fnDef.tuned[k] || {}),
                              ...preset.params,
                            },
                          }))
                        }
                      >
                        {preset.name}
                      </button>
                    ))}
                  </div>
                )}
                {Object.entries(params[k]).map(([pname, pval]) => {
                  // Adapt range so the slider always centers on the tuned value
                  const isLr = pname === "lr";
                  const sliderMax = isLr
                    ? Math.max(2.0, pval * 4)
                    : pname === "b2"
                    ? 0.9999
                    : 0.999;
                  const sliderStep = isLr
                    ? Math.max(0.0001, sliderMax / 1000)
                    : 0.001;
                  return (
                    <Slider
                      key={pname}
                      label={pname}
                      value={pval}
                      min={isLr ? 0.0001 : 0}
                      max={sliderMax}
                      step={sliderStep}
                      fmt={(v) => (isLr ? v.toFixed(5) : v.toFixed(3))}
                      onChange={(v) =>
                        setParams((p) => ({
                          ...p,
                          [k]: { ...p[k], [pname]: v },
                        }))
                      }
                    />
                  );
                })}
              </div>
            ))}
          </Section>
        </aside>

        {/* CENTER — plots */}
        <section style={S.center}>
          <div style={S.plotWrap}>
            <div style={S.hint}>
              <MousePointer2 size={12} />
              <span>click anywhere to reset starting point</span>
            </div>
            <canvas
              ref={canvasRef}
              width={W}
              height={H}
              style={S.canvasBack}
            />
            <canvas
              ref={overlayRef}
              width={W}
              height={H}
              style={S.canvasFront}
              onClick={handleCanvasClick}
            />
          </div>

          <div style={S.chartWrap}>
            <div style={S.chartTitle}>
              <Activity size={12} /> GRADIENT NORM — log scale
            </div>
            <canvas ref={chartRef} width={720} height={200} style={S.chart} />
          </div>
        </section>

        {/* RIGHT — telemetry */}
        <aside style={S.right}>
          <Section title="Live Telemetry">
            <div style={S.telemHeader}>
              <span style={{ flex: "0 0 48px" }}>algo</span>
              <span style={{ flex: "0 0 42px", textAlign: "right" }}>iter</span>
              <span style={{ flex: 1, textAlign: "right" }}>f(x)</span>
              <span style={{ flex: 1, textAlign: "right" }}>‖∇f‖</span>
            </div>
            {ALGO_KEYS.map((k) => {
              const t = traj[k];
              if (!t) return null;
              const iter = t.states[t.states.length - 1].iter;
              const fv = t.fvals[t.fvals.length - 1];
              const gv = t.gnorms[t.gnorms.length - 1];
              const converged = gv < 1e-6;
              return (
                <div
                  key={k}
                  style={{
                    ...S.telemRow,
                    opacity: enabled[k] ? 1 : 0.35,
                    borderLeftColor: ALGOS[k].color,
                  }}
                >
                  <span style={{ flex: "0 0 48px", color: ALGOS[k].color, fontWeight: 600 }}>
                    {ALGOS[k].short}
                  </span>
                  <span style={{ flex: "0 0 42px", textAlign: "right" }}>{iter}</span>
                  <span style={{ flex: 1, textAlign: "right" }}>
                    {fmtNum(fv)}
                  </span>
                  <span
                    style={{
                      flex: 1,
                      textAlign: "right",
                      color: t.diverged
                        ? "#ff6b6b"
                        : converged
                        ? "#7dd87d"
                        : "rgba(230,225,210,0.85)",
                    }}
                  >
                    {t.diverged ? "diverged" : fmtNum(gv)}
                  </span>
                </div>
              );
            })}
          </Section>

          <Section title="Notes for Class">
            <ul style={S.notes}>
              <li>
                On the ill-conditioned quadratic, watch GD zig-zag while
                momentum methods cut diagonally.
              </li>
              <li>
                On Rosenbrock, plain GD crawls along the valley floor; Nesterov
                and Adam handle the curvature far better.
              </li>
              <li>
                AdaGrad's per-coordinate accumulator monotonically decreases its
                effective step — useful intuition for why RMSProp's
                exponential averaging exists.
              </li>
              <li>
                On the saddle <em>x²−y²</em>, perturb y₀ slightly off zero and
                see escape time differ across methods.
              </li>
              <li>
                Try a bad lr (e.g. 0.5 on the ill-conditioned quadratic for GD)
                to demonstrate divergence.
              </li>
            </ul>
          </Section>
        </aside>
      </main>
    </div>
  );
}

// ---------- Small components ----------
function Section({ title, children }) {
  return (
    <div style={S.section}>
      <div style={S.sectionTitle}>{title}</div>
      {children}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, fmt }) {
  return (
    <div style={S.slider}>
      <div style={S.sliderTop}>
        <span style={S.sliderLabel}>{label}</span>
        <span style={S.sliderVal}>{fmt ? fmt(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={S.range}
      />
    </div>
  );
}

// ---------- Helpers ----------
function fmtNum(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-3 || a > 1e4) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : 3);
}
function superscript(n) {
  const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(n)
    .split("")
    .map((c) => map[c] || c)
    .join("");
}

// ---------- Styles ----------
const COL = {
  bg: "#0e1018",
  panel: "#171a24",
  panel2: "#1f2230",
  ink: "#e8e2d0",
  inkMute: "rgba(232,226,208,0.62)",
  rule: "rgba(232,226,208,0.10)",
  accent: "#d4453d",
};

const S = {
  root: {
    minHeight: "100vh",
    background: COL.bg,
    color: COL.ink,
    fontFamily: '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
  },
  header: {
    display: "flex",
    alignItems: "flex-end",
    justifyContent: "space-between",
    padding: "26px 32px 18px",
    borderBottom: `1px solid ${COL.rule}`,
    flexWrap: "wrap",
    gap: 16,
  },
  eyebrow: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.18em",
    color: COL.inkMute,
    marginBottom: 6,
  },
  title: {
    margin: 0,
    fontWeight: 500,
    fontSize: 32,
    letterSpacing: "-0.01em",
    fontStyle: "italic",
  },
  sub: {
    color: COL.inkMute,
    fontSize: 14,
    marginTop: 6,
    fontStyle: "italic",
  },
  legend: { display: "flex", gap: 8, flexWrap: "wrap" },
  legendBtn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "6px 10px",
    background: COL.panel,
    border: "1px solid",
    borderRadius: 0,
    cursor: "pointer",
    color: COL.ink,
    transition: "opacity 0.15s",
  },
  legendSwatch: {
    display: "inline-block",
    width: 10,
    height: 10,
    borderRadius: "50%",
  },
  legendName: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
    letterSpacing: "0.04em",
  },
  main: {
    display: "grid",
    gridTemplateColumns: "320px 1fr 320px",
    gap: 0,
    minHeight: "calc(100vh - 120px)",
  },
  left: {
    borderRight: `1px solid ${COL.rule}`,
    padding: "18px 20px",
    overflowY: "auto",
  },
  center: {
    padding: "18px 20px",
    display: "flex",
    flexDirection: "column",
    gap: 14,
    minWidth: 0,
  },
  right: {
    borderLeft: `1px solid ${COL.rule}`,
    padding: "18px 20px",
    overflowY: "auto",
  },
  section: { marginBottom: 22 },
  sectionTitle: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.22em",
    color: COL.inkMute,
    marginBottom: 12,
    paddingBottom: 6,
    borderBottom: `1px solid ${COL.rule}`,
  },
  select: {
    width: "100%",
    background: COL.panel,
    color: COL.ink,
    border: `1px solid ${COL.rule}`,
    padding: "8px 10px",
    fontSize: 14,
    fontFamily: "inherit",
    fontStyle: "italic",
  },
  formula: {
    marginTop: 10,
    padding: "10px 12px",
    background: COL.panel,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 12,
    color: COL.ink,
    border: `1px solid ${COL.rule}`,
    overflowX: "auto",
  },
  note: {
    marginTop: 8,
    color: COL.inkMute,
    fontSize: 12,
    lineHeight: 1.5,
    fontStyle: "italic",
  },
  statRow: {
    marginTop: 6,
    display: "flex",
    justifyContent: "space-between",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
  },
  statKey: { color: COL.inkMute, letterSpacing: "0.08em" },
  statVal: { color: COL.ink },
  controlRow: { display: "flex", gap: 6, marginBottom: 14 },
  btn: {
    flex: 1,
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 5,
    padding: "8px 10px",
    background: COL.panel,
    color: COL.ink,
    border: `1px solid ${COL.rule}`,
    cursor: "pointer",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
    letterSpacing: "0.08em",
    textTransform: "uppercase",
  },
  btnPrimary: {
    background: COL.accent,
    borderColor: COL.accent,
    color: "#fff",
  },
  slider: { marginBottom: 12 },
  sliderTop: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 4,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
  },
  sliderLabel: { color: COL.inkMute, letterSpacing: "0.06em" },
  sliderVal: { color: COL.ink },
  range: { width: "100%", accentColor: COL.accent },
  algoBlock: {
    paddingLeft: 10,
    marginBottom: 14,
    borderLeft: "3px solid",
  },
  algoHead: { marginBottom: 6 },
  algoName: {
    fontStyle: "italic",
    fontSize: 13,
  },
  presetRow: {
    display: "flex",
    flexWrap: "wrap",
    gap: 4,
    marginBottom: 8,
  },
  presetBtn: {
    padding: "2px 7px",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.04em",
    border: "1px solid",
    background: "transparent",
    cursor: "pointer",
    borderRadius: 0,
  },
  presetKind: {
    good: { color: "#7dd87d", borderColor: "rgba(125,216,125,0.45)" },
    bad: { color: "#ff8b8b", borderColor: "rgba(255,139,139,0.45)" },
    slow: { color: "rgba(232,226,208,0.55)", borderColor: "rgba(232,226,208,0.25)" },
  },
  plotWrap: {
    position: "relative",
    background: "#0a0c14",
    border: `1px solid ${COL.rule}`,
    aspectRatio: `${720} / ${520}`,
    width: "100%",
  },
  canvasBack: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    imageRendering: "pixelated",
  },
  canvasFront: {
    position: "absolute",
    inset: 0,
    width: "100%",
    height: "100%",
    cursor: "crosshair",
  },
  hint: {
    position: "absolute",
    top: 8,
    left: 10,
    zIndex: 2,
    display: "inline-flex",
    alignItems: "center",
    gap: 5,
    padding: "4px 8px",
    background: "rgba(10,12,20,0.6)",
    color: "rgba(232,226,208,0.7)",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.06em",
    border: `1px solid ${COL.rule}`,
  },
  chartWrap: {
    background: "#1a1d28",
    border: `1px solid ${COL.rule}`,
    padding: "10px 12px",
  },
  chartTitle: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.22em",
    color: COL.inkMute,
    marginBottom: 6,
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
  },
  chart: { width: "100%", display: "block" },
  telemHeader: {
    display: "flex",
    gap: 6,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10,
    letterSpacing: "0.12em",
    color: COL.inkMute,
    paddingBottom: 6,
    marginBottom: 4,
    borderBottom: `1px solid ${COL.rule}`,
  },
  telemRow: {
    display: "flex",
    gap: 6,
    padding: "5px 0 5px 8px",
    borderLeft: "3px solid",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
    marginBottom: 2,
  },
  notes: {
    fontSize: 13,
    lineHeight: 1.55,
    color: COL.inkMute,
    paddingLeft: 18,
    margin: 0,
    fontStyle: "italic",
  },
};

const globalCss = `
  body { margin: 0; }
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: rgba(232,226,208,0.15); }
  input[type=range] { height: 18px; }
  @media (max-width: 1100px) {
    main { grid-template-columns: 1fr !important; }
    aside { border: none !important; border-bottom: 1px solid rgba(232,226,208,0.10) !important; }
  }
`;
