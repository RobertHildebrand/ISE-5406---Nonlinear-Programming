import React, { useState, useEffect, useRef, useMemo } from "react";
import { Play, RotateCcw, StepForward, MousePointer2, ArrowRight } from "lucide-react";

/* ============================================================
   ANATOMY OF A FIRST-ORDER STEP
   Companion to the trajectory demo. Shows what each algorithm
   is actually *doing* at one point: the gradient it sees, the
   memory it carries, and how those compose into a step.
   ============================================================ */

const FUNCTIONS = {
  quad_ill: {
    name: "Ill-Conditioned Quadratic",
    formula: "½(x² + 25y²)",
    f: (x, y) => 0.5 * (x * x + 25 * y * y),
    g: (x, y) => [x, 25 * y],
    box: [-4, 4, -1.5, 1.5],
    levels: [0.05, 0.2, 0.5, 1, 2, 4, 8, 16],
    start: [-3, 0.8],
  },
  rosenbrock: {
    name: "Rosenbrock",
    formula: "(1−x)² + 100(y−x²)²",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
    box: [-2, 2, -1, 3],
    levels: [1, 5, 20, 50, 100, 200, 500, 1500],
    start: [-1.0, 1.5],
  },
  himmelblau: {
    name: "Himmelblau",
    formula: "(x²+y−11)² + (x+y²−7)²",
    f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
    g: (x, y) => [
      4 * x * (x * x + y - 11) + 2 * (x + y * y - 7),
      2 * (x * x + y - 11) + 4 * y * (x + y * y - 7),
    ],
    box: [-5, 5, -5, 5],
    levels: [1, 5, 15, 40, 80, 150, 300, 600],
    start: [-2, 1],
  },
  banana: {
    name: "Skewed Quadratic",
    formula: "½(x² + 10y² + 4xy)",
    f: (x, y) => 0.5 * (x * x + 10 * y * y + 4 * x * y),
    g: (x, y) => [x + 2 * y, 10 * y + 2 * x],
    box: [-4, 4, -2, 2],
    levels: [0.1, 0.5, 1, 2, 4, 8, 16],
    start: [-3, 1],
  },
};

// ---------- Algorithm step decomposers ----------
// Each returns { x_new, components: [{label, vec, color, kind, note}], formula: jsx, log: text }
// 'kind' controls how each component renders:
//   "vector"      — arrow from x
//   "vector_at"   — arrow from a different anchor (use 'from' field)
//   "point"       — circle marker (e.g. Nesterov lookahead)
//   "ellipse"     — preconditioner shown as scaled ellipse around x

const COL = {
  grad: "#d4453d",      // raw gradient
  mom: "#e8843a",       // momentum / velocity
  step: "#3a7ca5",      // final step
  precond: "#7a4ba5",   // preconditioner
  lookahead: "#c9a227", // Nesterov lookahead point
  ema: "#3f8c5c",       // EMA-related
};

const ALGOS = {
  gd: {
    name: "Gradient Descent",
    short: "GD",
    needsHistory: false,
    params: { lr: 0.05 },
    init: () => ({ iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const g = gFn(x[0], x[1]);
      const step = [-p.lr * g[0], -p.lr * g[1]];
      const x_new = [x[0] + step[0], x[1] + step[1]];
      return {
        x_new,
        new_state: { iter: state.iter + 1 },
        components: [
          { label: "∇f(x)", vec: g, anchor: x, color: COL.grad, kind: "vector",
            note: "the gradient at the current point" },
          { label: "−η∇f(x)", vec: step, anchor: x, color: COL.step, kind: "vector",
            note: "the actual step taken (length scaled by lr)" },
        ],
        formula: [
          { tex: "x", role: "var" },
          { tex: "_{k+1} = x_k - \\eta", role: "op" },
          { tex: "\\nabla f(x_k)", role: "grad" },
        ],
        narrative: "Pure descent. The step is just the negative gradient scaled by the learning rate η. No memory.",
        substitution: [
          ["∇f(x)", `(${num(g[0])}, ${num(g[1])})`],
          ["η", num(p.lr)],
          ["−η ∇f", `(${num(step[0])}, ${num(step[1])})`],
        ],
      };
    },
  },

  heavyball: {
    name: "Heavy Ball (Polyak Momentum)",
    short: "HB",
    needsHistory: true,
    params: { lr: 0.02, momentum: 0.85 },
    init: () => ({ v: [0, 0], iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const g = gFn(x[0], x[1]);
      const memTerm = [p.momentum * state.v[0], p.momentum * state.v[1]];
      const gradTerm = [-p.lr * g[0], -p.lr * g[1]];
      const v_new = [memTerm[0] + gradTerm[0], memTerm[1] + gradTerm[1]];
      const x_new = [x[0] + v_new[0], x[1] + v_new[1]];
      return {
        x_new,
        new_state: { v: v_new, iter: state.iter + 1 },
        components: [
          { label: "β·v_{k−1}", vec: memTerm, anchor: x, color: COL.mom, kind: "vector",
            note: "carry-over from previous velocity" },
          { label: "−η∇f(x)", vec: gradTerm, anchor: x, color: COL.grad, kind: "vector",
            note: "current gradient correction" },
          { label: "v_k (sum)", vec: v_new, anchor: x, color: COL.step, kind: "vector",
            note: "the two added together → the actual step" },
        ],
        narrative:
          "Two arrows: the velocity from the last iteration (orange) and the new gradient correction (red). Their vector sum (blue) is the step. With β = 0 you recover plain GD.",
        substitution: [
          ["v_{k−1}", `(${num(state.v[0])}, ${num(state.v[1])})`],
          ["β", num(p.momentum)],
          ["β·v_{k−1}", `(${num(memTerm[0])}, ${num(memTerm[1])})`],
          ["∇f(x_k)", `(${num(g[0])}, ${num(g[1])})`],
          ["−η∇f", `(${num(gradTerm[0])}, ${num(gradTerm[1])})`],
          ["v_k", `(${num(v_new[0])}, ${num(v_new[1])})`],
        ],
      };
    },
  },

  nesterov: {
    name: "Nesterov Accelerated Gradient",
    short: "NAG",
    needsHistory: true,
    params: { lr: 0.02, momentum: 0.85 },
    init: () => ({ v: [0, 0], iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const lookahead = [x[0] + p.momentum * state.v[0], x[1] + p.momentum * state.v[1]];
      const g_at_la = gFn(lookahead[0], lookahead[1]);
      const memTerm = [p.momentum * state.v[0], p.momentum * state.v[1]];
      const gradTerm = [-p.lr * g_at_la[0], -p.lr * g_at_la[1]];
      const v_new = [memTerm[0] + gradTerm[0], memTerm[1] + gradTerm[1]];
      const x_new = [x[0] + v_new[0], x[1] + v_new[1]];
      return {
        x_new,
        new_state: { v: v_new, iter: state.iter + 1 },
        components: [
          { label: "β·v_{k−1}", vec: memTerm, anchor: x, color: COL.mom, kind: "vector",
            note: "predicted move from momentum alone" },
          { label: "x + β·v (lookahead)", vec: [0, 0], anchor: lookahead, color: COL.lookahead, kind: "point",
            note: "the gradient is sampled HERE, not at x" },
          { label: "−η∇f at lookahead", vec: gradTerm, anchor: lookahead, color: COL.grad, kind: "vector",
            note: "correction sampled at the lookahead point" },
          { label: "v_k (sum)", vec: v_new, anchor: x, color: COL.step, kind: "vector",
            note: "final step from x" },
        ],
        narrative:
          "Same form as Heavy Ball, but the gradient is evaluated at the lookahead point x + β·v_{k−1} (yellow dot), not at x itself. This gives anticipatory correction — if momentum is overshooting, the lookahead gradient pulls back.",
        substitution: [
          ["v_{k−1}", `(${num(state.v[0])}, ${num(state.v[1])})`],
          ["lookahead", `(${num(lookahead[0])}, ${num(lookahead[1])})`],
          ["∇f(lookahead)", `(${num(g_at_la[0])}, ${num(g_at_la[1])})`],
          ["v_k", `(${num(v_new[0])}, ${num(v_new[1])})`],
        ],
      };
    },
  },

  adagrad: {
    name: "AdaGrad",
    short: "AG",
    needsHistory: true,
    params: { lr: 0.5 },
    init: () => ({ G: [1e-12, 1e-12], iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const g = gFn(x[0], x[1]);
      const G_new = [state.G[0] + g[0] * g[0], state.G[1] + g[1] * g[1]];
      const scale = [1 / Math.sqrt(G_new[0] + 1e-8), 1 / Math.sqrt(G_new[1] + 1e-8)];
      const rawStep = [-p.lr * g[0], -p.lr * g[1]];
      const condStep = [-p.lr * scale[0] * g[0], -p.lr * scale[1] * g[1]];
      const x_new = [x[0] + condStep[0], x[1] + condStep[1]];
      return {
        x_new,
        new_state: { G: G_new, iter: state.iter + 1 },
        components: [
          { label: "∇f(x)", vec: g, anchor: x, color: COL.grad, kind: "vector",
            note: "raw gradient" },
          { label: "−η∇f (no precond)", vec: rawStep, anchor: x, color: "rgba(212,69,61,0.4)",
            kind: "vector_dashed", note: "what GD would do" },
          { label: "preconditioner ellipse", anchor: x,
            ax: scale, color: COL.precond, kind: "ellipse",
            note: "axes ∝ 1/√G — directions with large past gradients are shortened" },
          { label: "actual step", vec: condStep, anchor: x, color: COL.step, kind: "vector",
            note: "GD step rescaled per-coordinate by 1/√G" },
        ],
        narrative:
          "AdaGrad accumulates squared gradients in G. Each coordinate's effective learning rate becomes η/√G_i, so coordinates that have seen large gradients shrink their step. The purple ellipse shows the per-coordinate scaling: short along directions with big history, long along quiet ones.",
        substitution: [
          ["∇f(x)", `(${num(g[0])}, ${num(g[1])})`],
          ["G_k (after update)", `(${num(G_new[0])}, ${num(G_new[1])})`],
          ["1/√G", `(${num(scale[0])}, ${num(scale[1])})`],
          ["step", `(${num(condStep[0])}, ${num(condStep[1])})`],
        ],
      };
    },
  },

  rmsprop: {
    name: "RMSProp",
    short: "RMS",
    needsHistory: true,
    params: { lr: 0.05, beta: 0.9 },
    init: () => ({ G: [1e-12, 1e-12], iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const g = gFn(x[0], x[1]);
      const G_new = [
        p.beta * state.G[0] + (1 - p.beta) * g[0] * g[0],
        p.beta * state.G[1] + (1 - p.beta) * g[1] * g[1],
      ];
      const scale = [1 / Math.sqrt(G_new[0] + 1e-8), 1 / Math.sqrt(G_new[1] + 1e-8)];
      const rawStep = [-p.lr * g[0], -p.lr * g[1]];
      const condStep = [-p.lr * scale[0] * g[0], -p.lr * scale[1] * g[1]];
      const x_new = [x[0] + condStep[0], x[1] + condStep[1]];
      return {
        x_new,
        new_state: { G: G_new, iter: state.iter + 1 },
        components: [
          { label: "∇f(x)", vec: g, anchor: x, color: COL.grad, kind: "vector" },
          { label: "−η∇f (no precond)", vec: rawStep, anchor: x, color: "rgba(212,69,61,0.4)",
            kind: "vector_dashed", note: "what GD would do" },
          { label: "EMA ellipse", anchor: x, ax: scale, color: COL.ema, kind: "ellipse",
            note: "axes ∝ 1/√G with EMA-windowed G" },
          { label: "actual step", vec: condStep, anchor: x, color: COL.step, kind: "vector" },
        ],
        narrative:
          "Like AdaGrad, but G is an exponential moving average instead of a sum. This solves AdaGrad's problem of monotonically shrinking step sizes — the preconditioner can adapt as the geometry changes.",
        substitution: [
          ["∇f", `(${num(g[0])}, ${num(g[1])})`],
          ["β·G_{k−1}", `(${num(p.beta * state.G[0])}, ${num(p.beta * state.G[1])})`],
          ["(1−β)·∇f²", `(${num((1-p.beta)*g[0]*g[0])}, ${num((1-p.beta)*g[1]*g[1])})`],
          ["G_k", `(${num(G_new[0])}, ${num(G_new[1])})`],
          ["1/√G_k", `(${num(scale[0])}, ${num(scale[1])})`],
          ["step", `(${num(condStep[0])}, ${num(condStep[1])})`],
        ],
      };
    },
  },

  adam: {
    name: "Adam",
    short: "ADAM",
    needsHistory: true,
    params: { lr: 0.1, b1: 0.9, b2: 0.999 },
    init: () => ({ m: [0, 0], v: [0, 0], iter: 0 }),
    decompose: (state, x, gFn, p) => {
      const g = gFn(x[0], x[1]);
      const t = state.iter + 1;
      const m = [p.b1 * state.m[0] + (1 - p.b1) * g[0], p.b1 * state.m[1] + (1 - p.b1) * g[1]];
      const v = [p.b2 * state.v[0] + (1 - p.b2) * g[0] * g[0], p.b2 * state.v[1] + (1 - p.b2) * g[1] * g[1]];
      const mh = [m[0] / (1 - Math.pow(p.b1, t)), m[1] / (1 - Math.pow(p.b1, t))];
      const vh = [v[0] / (1 - Math.pow(p.b2, t)), v[1] / (1 - Math.pow(p.b2, t))];
      const scale = [1 / (Math.sqrt(vh[0]) + 1e-8), 1 / (Math.sqrt(vh[1]) + 1e-8)];
      const step = [-p.lr * mh[0] * scale[0], -p.lr * mh[1] * scale[1]];
      const x_new = [x[0] + step[0], x[1] + step[1]];
      // Show what raw gradient direction would look like
      const rawDir = [-p.lr * g[0] * scale[0], -p.lr * g[1] * scale[1]];
      return {
        x_new,
        new_state: { m, v, iter: t },
        components: [
          { label: "∇f(x)", vec: g, anchor: x, color: COL.grad, kind: "vector",
            note: "raw gradient" },
          { label: "m̂_k (smoothed direction)", vec: mh, anchor: x, color: COL.mom, kind: "vector",
            note: "bias-corrected EMA of gradients (1st moment)" },
          { label: "1/√v̂ ellipse", anchor: x, ax: scale, color: COL.precond, kind: "ellipse",
            note: "bias-corrected EMA of squared gradients (2nd moment)" },
          { label: "actual step", vec: step, anchor: x, color: COL.step, kind: "vector",
            note: "−η·m̂/(√v̂+ε)" },
        ],
        narrative:
          "Adam = momentum (m̂) + RMSProp preconditioning (1/√v̂), with bias-correction so the early iterations aren't biased toward zero. The direction comes from m̂; the per-coordinate scale comes from 1/√v̂.",
        substitution: [
          ["∇f", `(${num(g[0])}, ${num(g[1])})`],
          ["m_k", `(${num(m[0])}, ${num(m[1])})`],
          ["v_k", `(${num(v[0])}, ${num(v[1])})`],
          ["m̂_k (bias-corr)", `(${num(mh[0])}, ${num(mh[1])})`],
          ["v̂_k (bias-corr)", `(${num(vh[0])}, ${num(vh[1])})`],
          ["step", `(${num(step[0])}, ${num(step[1])})`],
        ],
      };
    },
  },
};

const ALGO_KEYS = Object.keys(ALGOS);

// ---------- Contour drawing (shared with trajectory demo) ----------
function drawContours(ctx, fn, box, levels, W, H) {
  const [xmin, xmax, ymin, ymax] = box;
  const img = ctx.createImageData(W, H);
  let fmin = Infinity, fmax = -Infinity;
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
  const shift = fmin < 0 ? -fmin + 1 : 0;
  const lmin = Math.log(fmin + shift + 1e-9);
  const lmax = Math.log(fmax + shift + 1e-9);
  for (let i = 0; i < W * H; i++) {
    const v = grid[i];
    let t = (Math.log(v + shift + 1e-9) - lmin) / (lmax - lmin + 1e-12);
    t = Math.max(0, Math.min(1, t));
    const r = Math.round(20 + 220 * Math.pow(t, 1.4));
    const g = Math.round(28 + 200 * Math.pow(t, 1.6));
    const b = Math.round(48 + 150 * (1 - Math.pow(1 - t, 2)));
    const idx = i * 4;
    img.data[idx] = r; img.data[idx + 1] = g; img.data[idx + 2] = b; img.data[idx + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

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
        let s = seg(a, b, px, py, px + 2, py); if (s) pts.push(s);
        s = seg(b, c, px + 2, py, px + 2, py + 2); if (s) pts.push(s);
        s = seg(c, d, px + 2, py + 2, px, py + 2); if (s) pts.push(s);
        s = seg(d, a, px, py + 2, px, py); if (s) pts.push(s);
        if (pts.length >= 2) {
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
        }
      }
    }
    ctx.stroke();
  }
}

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
    // length scale for arrow → pixel conversion
    scaleX: (W - 1) / (xmax - xmin),
    scaleY: (H - 1) / (ymax - ymin),
  };
}

// Draw an arrow with arrowhead, given start & end pixel coords
function drawArrow(ctx, x1, y1, x2, y2, color, dashed = false, lw = 2.5) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  if (dashed) ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.setLineDash([]);
  // Arrowhead
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const ah = 8;
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - ah * Math.cos(ang - Math.PI / 7), y2 - ah * Math.sin(ang - Math.PI / 7));
  ctx.lineTo(x2 - ah * Math.cos(ang + Math.PI / 7), y2 - ah * Math.sin(ang + Math.PI / 7));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function num(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-3 || a > 1e4) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : 3);
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function StepAnatomy() {
  const [fnKey, setFnKey] = useState("quad_ill");
  const fnDef = FUNCTIONS[fnKey];
  const [algoKey, setAlgoKey] = useState("heavyball");
  const algoDef = ALGOS[algoKey];

  const [params, setParams] = useState(() => ({ ...algoDef.params }));
  const [point, setPoint] = useState(fnDef.start);
  const [state, setState] = useState(() => algoDef.init());
  const [history, setHistory] = useState([fnDef.start]); // chain of points
  const [arrowScale, setArrowScale] = useState(1.0);

  // Recompute decomposition from the current point + state (live)
  const decomp = useMemo(() => {
    try {
      return algoDef.decompose(state, point, fnDef.g, params);
    } catch (e) {
      return null;
    }
  }, [algoDef, state, point, fnDef, params]);

  // Reset whenever fn or algo changes
  useEffect(() => {
    setParams({ ...ALGOS[algoKey].params });
    setState(ALGOS[algoKey].init());
    setHistory([point]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [algoKey]);

  useEffect(() => {
    setPoint(fnDef.start);
    setHistory([fnDef.start]);
    setState(ALGOS[algoKey].init());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fnKey]);

  // Take the proposed step (commit it)
  function takeStep() {
    if (!decomp) return;
    setPoint(decomp.x_new);
    setState(decomp.new_state);
    setHistory((h) => [...h, decomp.x_new]);
  }

  function reset() {
    setPoint(fnDef.start);
    setState(algoDef.init());
    setHistory([fnDef.start]);
  }

  // Build up momentum: take 5 steps fast
  function buildHistory() {
    if (!algoDef.needsHistory) return;
    let p = point;
    let s = state;
    const newHist = [p];
    for (let i = 0; i < 5; i++) {
      const d = algoDef.decompose(s, p, fnDef.g, params);
      p = d.x_new;
      s = d.new_state;
      newHist.push(p);
    }
    setPoint(p);
    setState(s);
    setHistory((h) => [...h, ...newHist.slice(1)]);
  }

  // ---------- Canvas drawing ----------
  const W = 720, H = 520;
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const map = useMemo(() => makeMap(fnDef.box, W, H), [fnDef.box]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    drawContours(c.getContext("2d"), fnDef.f, fnDef.box, fnDef.levels, W, H);
  }, [fnDef]);

  useEffect(() => {
    const c = overlayRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, W, H);

    // Draw history trail
    if (history.length > 1) {
      ctx.strokeStyle = "rgba(255,255,255,0.35)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      for (let i = 0; i < history.length; i++) {
        const [px, py] = map.toPx(history[i][0], history[i][1]);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      // Past point markers
      for (let i = 0; i < history.length - 1; i++) {
        const [px, py] = map.toPx(history[i][0], history[i][1]);
        ctx.fillStyle = "rgba(255,255,255,0.35)";
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    if (!decomp) return;

    // Helper: project a data-space vector to pixel-space delta
    const vecPx = (v) => [v[0] * map.scaleX * arrowScale, -v[1] * map.scaleY * arrowScale];

    // Draw components, ellipses first (background), then dashed, then solid arrows, then point markers
    const ordered = [
      ...decomp.components.filter((c) => c.kind === "ellipse"),
      ...decomp.components.filter((c) => c.kind === "vector_dashed"),
      ...decomp.components.filter((c) => c.kind === "vector" || c.kind === "vector_at"),
      ...decomp.components.filter((c) => c.kind === "point"),
    ];

    for (const comp of ordered) {
      const [ax, ay] = comp.anchor || point;
      const [apx, apy] = map.toPx(ax, ay);
      if (comp.kind === "ellipse") {
        // Draw ellipse with semi-axes proportional to comp.ax (1/√G etc.)
        const [rxData, ryData] = comp.ax;
        const norm = Math.max(rxData, ryData);
        // Visual scale: largest axis = 0.5 of plot half-extent
        const visualR = 0.4 * Math.min(map.scaleX * (fnDef.box[1] - fnDef.box[0]) / 2,
                                        map.scaleY * (fnDef.box[3] - fnDef.box[2]) / 2);
        const rx = (rxData / norm) * visualR * 0.5;
        const ry = (ryData / norm) * visualR * 0.5;
        ctx.save();
        ctx.strokeStyle = comp.color;
        ctx.fillStyle = comp.color + "1a"; // ~10% alpha
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.ellipse(apx, apy, rx, ry, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      } else if (comp.kind === "point") {
        ctx.save();
        ctx.fillStyle = comp.color;
        ctx.strokeStyle = "rgba(0,0,0,0.6)";
        ctx.beginPath();
        ctx.arc(apx, apy, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        // Dashed line from x to lookahead
        const [xpx, xpy] = map.toPx(point[0], point[1]);
        ctx.strokeStyle = comp.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        ctx.moveTo(xpx, xpy);
        ctx.lineTo(apx, apy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      } else {
        const [dx, dy] = vecPx(comp.vec);
        drawArrow(
          ctx, apx, apy, apx + dx, apy + dy,
          comp.color,
          comp.kind === "vector_dashed",
          comp.kind === "vector_dashed" ? 1.5 : 2.5
        );
      }
    }

    // Current point marker (drawn last on top)
    const [cpx, cpy] = map.toPx(point[0], point[1]);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "rgba(0,0,0,0.7)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cpx, cpy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Proposed next point
    const [npx, npy] = map.toPx(decomp.x_new[0], decomp.x_new[1]);
    ctx.save();
    ctx.fillStyle = "rgba(58,124,165,0.4)";
    ctx.strokeStyle = COL.step;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([2, 2]);
    ctx.beginPath();
    ctx.arc(npx, npy, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }, [decomp, history, point, map, fnDef, arrowScale]);

  function handleClick(e) {
    const rect = overlayRef.current.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * W;
    const py = ((e.clientY - rect.top) / rect.height) * H;
    const [x, y] = map.toData(px, py);
    setPoint([x, y]);
    setHistory([[x, y]]);
    setState(algoDef.init()); // clicking = fresh start, no momentum
  }

  // ---------- UI ----------
  return (
    <div style={S.root}>
      <style>{globalCss}</style>

      <header style={S.header}>
        <div>
          <div style={S.eyebrow}>ISE 5405 / 5406 — STEP ANATOMY</div>
          <h1 style={S.title}>What's Inside a First-Order Step?</h1>
          <div style={S.sub}>
            See the gradient, momentum, and preconditioner that compose into a single update.
          </div>
        </div>
      </header>

      <main style={S.main}>
        {/* LEFT — controls */}
        <aside style={S.left}>
          <Section title="Function">
            <select
              value={fnKey}
              onChange={(e) => setFnKey(e.target.value)}
              style={S.select}
            >
              {Object.entries(FUNCTIONS).map(([k, v]) => (
                <option key={k} value={k}>{v.name}</option>
              ))}
            </select>
            <div style={S.formula}>f(x,y) = {fnDef.formula}</div>
          </Section>

          <Section title="Algorithm">
            <div style={S.algoGrid}>
              {ALGO_KEYS.map((k) => (
                <button
                  key={k}
                  onClick={() => setAlgoKey(k)}
                  style={{
                    ...S.algoBtn,
                    background: algoKey === k ? "#d4453d" : "#171a24",
                    color: algoKey === k ? "#fff" : "#e8e2d0",
                    borderColor: algoKey === k ? "#d4453d" : "rgba(232,226,208,0.10)",
                  }}
                >
                  {ALGOS[k].short}
                </button>
              ))}
            </div>
            <div style={S.algoFullName}>{algoDef.name}</div>
          </Section>

          <Section title="Hyperparameters">
            {Object.entries(params).map(([pname, pval]) => (
              <Slider
                key={pname}
                label={pname}
                value={pval}
                min={pname === "lr" ? 0.0001 : 0}
                max={pname === "lr" ? Math.max(2.0, pval * 4) : pname === "b2" ? 0.9999 : 0.999}
                step={pname === "lr" ? Math.max(0.0001, Math.max(2.0, pval * 4) / 1000) : 0.001}
                fmt={(v) => v.toFixed(4)}
                onChange={(v) => setParams((p) => ({ ...p, [pname]: v }))}
              />
            ))}
          </Section>

          <Section title="Step Controls">
            <div style={S.controlRow}>
              <button style={{ ...S.btn, ...S.btnPrimary }} onClick={takeStep}>
                <ArrowRight size={14} /> Take Step
              </button>
              <button style={S.btn} onClick={reset}>
                <RotateCcw size={14} /> Reset
              </button>
            </div>
            {algoDef.needsHistory && (
              <button
                style={{ ...S.btn, width: "100%", marginTop: 6 }}
                onClick={buildHistory}
              >
                <StepForward size={14} /> Build Up Momentum (5 steps)
              </button>
            )}
            <Slider
              label="Arrow display scale"
              value={arrowScale}
              min={0.1}
              max={5}
              step={0.05}
              fmt={(v) => `${v.toFixed(2)}×`}
              onChange={setArrowScale}
            />
          </Section>

          <Section title="Current State">
            <KV k="x" v={`(${num(point[0])}, ${num(point[1])})`} />
            <KV k="iter" v={state.iter} />
            {state.v && <KV k="v (velocity)" v={`(${num(state.v[0])}, ${num(state.v[1])})`} />}
            {state.G && <KV k="G (sq-grad)" v={`(${num(state.G[0])}, ${num(state.G[1])})`} />}
            {state.m && <KV k="m (1st mom)" v={`(${num(state.m[0])}, ${num(state.m[1])})`} />}
          </Section>
        </aside>

        {/* CENTER — plot */}
        <section style={S.center}>
          <div style={S.plotWrap}>
            <div style={S.hint}>
              <MousePointer2 size={12} />
              <span>click to set point — momentum will be reset</span>
            </div>
            <canvas ref={canvasRef} width={W} height={H} style={S.canvasBack} />
            <canvas
              ref={overlayRef}
              width={W}
              height={H}
              style={S.canvasFront}
              onClick={handleClick}
            />
          </div>

          {/* Legend chips for the components currently drawn */}
          {decomp && (
            <div style={S.legendBar}>
              {decomp.components.map((c, i) => (
                <span key={i} style={S.legendChip}>
                  <span
                    style={{
                      ...S.legendDot,
                      background: c.kind === "ellipse" || c.kind === "vector_dashed"
                        ? "transparent"
                        : c.color,
                      border: `2px ${c.kind === "vector_dashed" ? "dashed" : "solid"} ${c.color}`,
                    }}
                  />
                  <span>{c.label}</span>
                </span>
              ))}
            </div>
          )}
        </section>

        {/* RIGHT — math anatomy */}
        <aside style={S.right}>
          <Section title="The Update Rule">
            <UpdateRule algoKey={algoKey} />
          </Section>

          <Section title="What's Happening">
            <p style={S.narrative}>{decomp?.narrative}</p>
          </Section>

          <Section title="Substituted Values">
            <div style={S.substTable}>
              {decomp?.substitution.map(([k, v], i) => (
                <div key={i} style={S.substRow}>
                  <span style={S.substKey}>{k}</span>
                  <span style={S.substEq}>=</span>
                  <span style={S.substVal}>{v}</span>
                </div>
              ))}
            </div>
          </Section>

          <Section title="Result">
            <div style={S.resultBox}>
              <div style={{ color: "rgba(232,226,208,0.55)", fontSize: 11, marginBottom: 4 }}>
                x_{state.iter} = ({num(point[0])}, {num(point[1])})
              </div>
              <div style={{ color: COL.step, fontWeight: 600, fontSize: 13 }}>
                x_{state.iter + 1} = ({num(decomp?.x_new[0] ?? 0)}, {num(decomp?.x_new[1] ?? 0)})
              </div>
              <div style={{ color: "rgba(232,226,208,0.55)", fontSize: 11, marginTop: 4 }}>
                f → {num(fnDef.f((decomp?.x_new ?? point)[0], (decomp?.x_new ?? point)[1]))}
              </div>
            </div>
          </Section>
        </aside>
      </main>
    </div>
  );
}

// ---------- Update rule formula display per algorithm ----------
function UpdateRule({ algoKey }) {
  const rules = {
    gd: (
      <div>
        <div style={S.eq}>v_k = −η · ∇f(x_k)</div>
        <div style={S.eq}>x_{`{k+1}`} = x_k + v_k</div>
      </div>
    ),
    heavyball: (
      <div>
        <div style={S.eq}>v_k = β · v_{`{k−1}`} − η · ∇f(x_k)</div>
        <div style={S.eq}>x_{`{k+1}`} = x_k + v_k</div>
        <div style={S.eqNote}>memory term + gradient correction</div>
      </div>
    ),
    nesterov: (
      <div>
        <div style={S.eq}>x̃ = x_k + β · v_{`{k−1}`} <span style={S.eqAnnot}>(lookahead)</span></div>
        <div style={S.eq}>v_k = β · v_{`{k−1}`} − η · ∇f(x̃)</div>
        <div style={S.eq}>x_{`{k+1}`} = x_k + v_k</div>
        <div style={S.eqNote}>gradient evaluated at the lookahead point</div>
      </div>
    ),
    adagrad: (
      <div>
        <div style={S.eq}>g_k = ∇f(x_k)</div>
        <div style={S.eq}>G_k = G_{`{k−1}`} + g_k ⊙ g_k <span style={S.eqAnnot}>(per-coord)</span></div>
        <div style={S.eq}>x_{`{k+1}`} = x_k − η · g_k / (√G_k + ε)</div>
      </div>
    ),
    rmsprop: (
      <div>
        <div style={S.eq}>g_k = ∇f(x_k)</div>
        <div style={S.eq}>G_k = β · G_{`{k−1}`} + (1−β) · g_k ⊙ g_k</div>
        <div style={S.eq}>x_{`{k+1}`} = x_k − η · g_k / (√G_k + ε)</div>
        <div style={S.eqNote}>EMA window — adapts to changing geometry</div>
      </div>
    ),
    adam: (
      <div>
        <div style={S.eq}>m_k = β₁ m_{`{k−1}`} + (1−β₁) g_k</div>
        <div style={S.eq}>v_k = β₂ v_{`{k−1}`} + (1−β₂) g_k ⊙ g_k</div>
        <div style={S.eq}>m̂_k = m_k / (1 − β₁ᵏ),  v̂_k = v_k / (1 − β₂ᵏ)</div>
        <div style={S.eq}>x_{`{k+1}`} = x_k − η · m̂_k / (√v̂_k + ε)</div>
      </div>
    ),
  };
  return rules[algoKey];
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
        type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={S.range}
      />
    </div>
  );
}

function KV({ k, v }) {
  return (
    <div style={S.kvRow}>
      <span style={S.kvKey}>{k}</span>
      <span style={S.kvVal}>{v}</span>
    </div>
  );
}

// ---------- Styles ----------
const S = {
  root: {
    minHeight: "100vh",
    background: "#0e1018",
    color: "#e8e2d0",
    fontFamily: '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif',
  },
  header: {
    padding: "26px 32px 18px",
    borderBottom: "1px solid rgba(232,226,208,0.10)",
  },
  eyebrow: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10, letterSpacing: "0.18em", color: "rgba(232,226,208,0.62)", marginBottom: 6,
  },
  title: { margin: 0, fontWeight: 500, fontSize: 32, letterSpacing: "-0.01em", fontStyle: "italic" },
  sub: { color: "rgba(232,226,208,0.62)", fontSize: 14, marginTop: 6, fontStyle: "italic" },
  main: {
    display: "grid",
    gridTemplateColumns: "320px 1fr 320px",
    minHeight: "calc(100vh - 110px)",
  },
  left: { borderRight: "1px solid rgba(232,226,208,0.10)", padding: "18px 20px", overflowY: "auto" },
  center: { padding: "18px 20px", display: "flex", flexDirection: "column", gap: 14, minWidth: 0 },
  right: { borderLeft: "1px solid rgba(232,226,208,0.10)", padding: "18px 20px", overflowY: "auto" },
  section: { marginBottom: 22 },
  sectionTitle: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10, letterSpacing: "0.22em", color: "rgba(232,226,208,0.62)",
    marginBottom: 12, paddingBottom: 6, borderBottom: "1px solid rgba(232,226,208,0.10)",
  },
  select: {
    width: "100%", background: "#171a24", color: "#e8e2d0",
    border: "1px solid rgba(232,226,208,0.10)", padding: "8px 10px",
    fontSize: 14, fontFamily: "inherit", fontStyle: "italic",
  },
  formula: {
    marginTop: 10, padding: "10px 12px", background: "#171a24",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 12, border: "1px solid rgba(232,226,208,0.10)",
  },
  algoGrid: {
    display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6,
  },
  algoBtn: {
    padding: "10px 4px", border: "1px solid", cursor: "pointer",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11, letterSpacing: "0.06em", fontWeight: 600,
  },
  algoFullName: {
    marginTop: 10, fontStyle: "italic", fontSize: 13,
    color: "rgba(232,226,208,0.85)", textAlign: "center",
  },
  controlRow: { display: "flex", gap: 6 },
  btn: {
    flex: 1, display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 5,
    padding: "8px 10px", background: "#171a24", color: "#e8e2d0",
    border: "1px solid rgba(232,226,208,0.10)", cursor: "pointer",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11, letterSpacing: "0.08em", textTransform: "uppercase",
  },
  btnPrimary: { background: "#d4453d", borderColor: "#d4453d", color: "#fff" },
  slider: { marginBottom: 12 },
  sliderTop: {
    display: "flex", justifyContent: "space-between", marginBottom: 4,
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace', fontSize: 11,
  },
  sliderLabel: { color: "rgba(232,226,208,0.62)", letterSpacing: "0.06em" },
  sliderVal: { color: "#e8e2d0" },
  range: { width: "100%", accentColor: "#d4453d" },
  plotWrap: {
    position: "relative", background: "#0a0c14",
    border: "1px solid rgba(232,226,208,0.10)",
    aspectRatio: `${720} / ${520}`, width: "100%",
  },
  canvasBack: {
    position: "absolute", inset: 0, width: "100%", height: "100%", imageRendering: "pixelated",
  },
  canvasFront: {
    position: "absolute", inset: 0, width: "100%", height: "100%", cursor: "crosshair",
  },
  hint: {
    position: "absolute", top: 8, left: 10, zIndex: 2,
    display: "inline-flex", alignItems: "center", gap: 5,
    padding: "4px 8px", background: "rgba(10,12,20,0.6)",
    color: "rgba(232,226,208,0.7)",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 10, letterSpacing: "0.06em",
    border: "1px solid rgba(232,226,208,0.10)",
  },
  legendBar: {
    display: "flex", flexWrap: "wrap", gap: 12,
    padding: "10px 14px", background: "#171a24",
    border: "1px solid rgba(232,226,208,0.10)",
  },
  legendChip: {
    display: "inline-flex", alignItems: "center", gap: 6,
    fontSize: 12, color: "rgba(232,226,208,0.85)", fontStyle: "italic",
  },
  legendDot: {
    display: "inline-block", width: 10, height: 10, borderRadius: "50%",
  },
  narrative: {
    fontSize: 13, lineHeight: 1.55, color: "rgba(232,226,208,0.85)",
    margin: 0, fontStyle: "italic",
  },
  eq: {
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 12, color: "#e8e2d0", padding: "4px 0",
  },
  eqAnnot: {
    color: "rgba(232,226,208,0.5)", fontStyle: "italic", marginLeft: 6,
    fontFamily: '"Iowan Old Style", Georgia, serif',
  },
  eqNote: {
    fontStyle: "italic", color: "rgba(232,226,208,0.55)",
    fontSize: 11, marginTop: 6,
  },
  substTable: { display: "flex", flexDirection: "column", gap: 4 },
  substRow: {
    display: "flex", gap: 8, alignItems: "baseline",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11, padding: "3px 0",
    borderBottom: "1px dotted rgba(232,226,208,0.07)",
  },
  substKey: { color: "rgba(232,226,208,0.62)", flex: "0 0 auto", minWidth: 90 },
  substEq: { color: "rgba(232,226,208,0.4)" },
  substVal: { color: "#e8e2d0", flex: 1, textAlign: "right" },
  resultBox: {
    background: "#171a24", border: "1px solid rgba(232,226,208,0.10)",
    padding: "12px 14px",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
  },
  kvRow: {
    display: "flex", justifyContent: "space-between", padding: "3px 0",
    fontFamily: 'ui-monospace, "SF Mono", Menlo, monospace',
    fontSize: 11,
    borderBottom: "1px dotted rgba(232,226,208,0.07)",
  },
  kvKey: { color: "rgba(232,226,208,0.62)" },
  kvVal: { color: "#e8e2d0" },
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
