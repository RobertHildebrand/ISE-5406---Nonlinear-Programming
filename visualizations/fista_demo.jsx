import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   PROXIMAL GRADIENT (ISTA) and FISTA
   ISE 5406

   Solves the lasso-style 2D problem:

       min  ½ ‖A x − b‖² + λ ‖x‖₁

   ISTA   :  x_{k+1} = soft_{λτ}( x_k − τ Aᵀ (A x_k − b) )
   FISTA  :  add Nesterov momentum

       y_k    = x_k + ((t_{k-1} − 1) / t_k) (x_k − x_{k-1})
       x_{k+1} = soft_{λτ}( y_k − τ ∇g(y_k) )
       t_{k+1} = (1 + √(1 + 4 t_k²)) / 2

   The visualization shows level sets of g(x) = ½ ‖A x − b‖²
   and the iterate trajectory; the soft-thresholding box is
   highlighted explicitly so students see why ISTA produces
   exact zeros.
   ============================================================ */

// ============================================================
// Problem instance — 2D so we can plot
// ============================================================
const A = [
  [1.0, 0.4],
  [0.4, 1.0],
];
const B = [1.5, 1.2];
const LAMBDA = 0.6;

const ATA = [
  [A[0][0] ** 2 + A[1][0] ** 2, A[0][0] * A[0][1] + A[1][0] * A[1][1]],
  [A[0][0] * A[0][1] + A[1][0] * A[1][1], A[0][1] ** 2 + A[1][1] ** 2],
];
const ATB = [A[0][0] * B[0] + A[1][0] * B[1], A[0][1] * B[0] + A[1][1] * B[1]];

const matVec = (M, v) => [
  M[0][0] * v[0] + M[0][1] * v[1],
  M[1][0] * v[0] + M[1][1] * v[1],
];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const scl = (a, c) => [a[0] * c, a[1] * c];

// ½ ‖A x − b‖²
function g(x) {
  const r = sub(matVec(A, x), B);
  return 0.5 * (r[0] ** 2 + r[1] ** 2);
}
function gradG(x) {
  // Aᵀ(Ax − b)
  const r = sub(matVec(A, x), B);
  return [A[0][0] * r[0] + A[1][0] * r[1], A[0][1] * r[0] + A[1][1] * r[1]];
}
function f(x) {
  return g(x) + LAMBDA * (Math.abs(x[0]) + Math.abs(x[1]));
}
function softThresh(z, kappa) {
  return z.map((zi) => Math.sign(zi) * Math.max(0, Math.abs(zi) - kappa));
}

// Lipschitz constant of ∇g = ‖AᵀA‖₂ → largest eigenvalue
function lipschitzL() {
  const a = ATA[0][0];
  const b = ATA[1][1];
  const c = ATA[0][1];
  return 0.5 * (a + b + Math.sqrt((a - b) ** 2 + 4 * c * c));
}
const L = lipschitzL();
const TAU = 1.0 / L;

// True optimum: solved analytically via KKT (precomputed)
const X_STAR = (() => {
  // For tiny 2D lasso we can numerically solve via many ISTA steps
  let x = [0, 0];
  for (let k = 0; k < 5000; k++) {
    const z = sub(x, scl(gradG(x), TAU));
    x = softThresh(z, LAMBDA * TAU);
  }
  return x;
})();
const F_STAR = f(X_STAR);

// ============================================================
// Run iterations
// ============================================================
function runIterations(algo, maxIter, x0) {
  const history = [];
  let x = [...x0];
  let xPrev = [...x0];
  let t = 1;
  for (let k = 0; k < maxIter; k++) {
    let y = x;
    if (algo === "fista") {
      const tNext = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
      const beta = (t - 1) / tNext;
      y = [x[0] + beta * (x[0] - xPrev[0]), x[1] + beta * (x[1] - xPrev[1])];
      // Note: tNext is the new t, used in the NEXT iteration
    }
    const gradAtY = gradG(y);
    const z = sub(y, scl(gradAtY, TAU));
    const xNext = softThresh(z, LAMBDA * TAU);
    history.push({
      k,
      x: [...x],
      xPrev: [...xPrev],
      y: [...y],
      gradAtY,
      z,
      xNext: [...xNext],
      f: f(x),
    });
    if (algo === "fista") {
      t = (1 + Math.sqrt(1 + 4 * t * t)) / 2;
    }
    xPrev = [...x];
    x = [...xNext];
  }
  history.push({
    k: maxIter,
    x: [...x],
    xPrev: [...xPrev],
    y: [...x],
    f: f(x),
    final: true,
  });
  return history;
}

// ============================================================
// Code stepper
// ============================================================
const CODE_LINES = [
  null,
  "import numpy as np",
  "",
  "A = np.array([[1.0, 0.4],",
  "              [0.4, 1.0]])",
  "b   = np.array([1.5, 1.2])",
  "lam = 0.6",
  "L   = np.linalg.norm(A.T @ A, 2)   # Lipschitz constant of ∇g",
  "tau = 1.0 / L                       # safe step size",
  "",
  "def soft(z, kappa):                 # element-wise soft-threshold",
  "    return np.sign(z) * np.maximum(np.abs(z) - kappa, 0.0)",
  "",
  "x = np.zeros(2)",
  "x_prev = x.copy()",
  "t = 1.0",
  "",
  "for k in range(MAX_ITER):",
  "    # ── ISTA (vanilla proximal gradient) ────",
  "    grad = A.T @ (A @ x - b)",
  "    x_ista = soft(x - tau * grad, lam * tau)",
  "",
  "    # ── FISTA (Nesterov-accelerated) ────────",
  "    t_next = 0.5 * (1 + np.sqrt(1 + 4 * t**2))",
  "    beta   = (t - 1) / t_next",
  "    y      = x + beta * (x - x_prev)            # extrapolation",
  "    grad_y = A.T @ (A @ y - b)",
  "    x_fista = soft(y - tau * grad_y, lam * tau) # prox step at y",
  "",
  "    x_prev = x",
  "    x      = x_fista if FISTA else x_ista",
  "    t      = t_next",
  "",
  "print('x* =', x, ' f* =', 0.5*((A @ x - b)**2).sum() + lam*np.abs(x).sum())",
];

// ============================================================
// Main component
// ============================================================
export default function FistaDemo() {
  const [algo, setAlgo] = useState("fista");
  const [k, setK] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(500);
  const MAX_ITER = 50;

  const ista = useMemo(() => runIterations("ista", MAX_ITER, [-2.0, 1.5]), []);
  const fista = useMemo(() => runIterations("fista", MAX_ITER, [-2.0, 1.5]), []);
  const history = algo === "fista" ? fista : ista;

  useEffect(() => setK(0), [algo]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setK((kk) => {
        if (kk + 1 >= history.length) {
          setRunning(false);
          return kk;
        }
        return kk + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, history.length]);

  const step = useCallback(() => setK((kk) => Math.min(history.length - 1, kk + 1)), [history.length]);
  const reset = useCallback(() => {
    setK(0);
    setRunning(false);
  }, []);

  const state = history[k];

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Proximal Gradient & FISTA
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        The standard tool for sparse / non-smooth optimization. Each
        iteration takes a gradient step on the smooth part, then applies a
        proximal operator (here: soft-thresholding) for the non-smooth
        part. FISTA accelerates ISTA with Nesterov momentum, turning{" "}
        <Tex>{String.raw`O(1/k)`}</Tex> convergence into{" "}
        <Tex>{String.raw`O(1/k^2)`}</Tex> — the same rate as smooth
        gradient descent on a strongly convex problem.
      </p>

      <div style={problemBox}>
        <div style={{ fontWeight: 600, fontSize: 13, marginBottom: 4 }}>
          Problem
        </div>
        <Tex block>
          {String.raw`\min_{x \in \mathbb{R}^2}\;\; f(x) = \underbrace{\tfrac{1}{2}\|Ax - b\|_2^2}_{g(x):\,\text{smooth}} \;+\; \underbrace{\lambda \|x\|_1}_{h(x):\,\text{non-smooth}}`}
        </Tex>
        <div style={{ fontSize: 13, color: "#444", marginTop: 6 }}>
          With{" "}
          <Tex>{String.raw`A = \begin{bmatrix}1.0 & 0.4 \\ 0.4 & 1.0\end{bmatrix},\;\; b = (1.5,\, 1.2)^T,\;\; \lambda = 0.6,\;\; \tau = 1/L \approx ${TAU.toFixed(3)}`}</Tex>
          .
        </div>
        <div style={{ fontSize: 13, color: "#444", marginTop: 4 }}>
          True optimum: <Tex>{`x^\\star = (${X_STAR[0].toFixed(3)},\\, ${X_STAR[1].toFixed(3)}),\\;\\; f^\\star = ${F_STAR.toFixed(4)}`}</Tex>
        </div>
      </div>

      <div style={{ marginBottom: 14 }}>
        <label style={{ marginRight: 14, fontSize: 13 }}>
          <input type="radio" checked={algo === "ista"} onChange={() => setAlgo("ista")} />
          &nbsp;ISTA (no momentum)
        </label>
        <label style={{ fontSize: 13 }}>
          <input type="radio" checked={algo === "fista"} onChange={() => setAlgo("fista")} />
          &nbsp;FISTA (Nesterov momentum)
        </label>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)",
          gap: 22,
          alignItems: "flex-start",
        }}
      >
        <div>
          <ContourPlot history={history} kIdx={k} />

          <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
            <button onClick={step} disabled={k >= history.length - 1} style={btnPrimary}>
              <StepForward size={16} /> Step
            </button>
            <button onClick={() => setRunning((r) => !r)} disabled={k >= history.length - 1} style={btn}>
              {running ? <Pause size={16} /> : <Play size={16} />}{" "}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={reset} style={btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={{ marginTop: 12 }}>
            <label style={{ fontSize: 12, color: "#444", fontFamily: "monospace" }}>
              speed: <b>{speed} ms/step</b>
            </label>
            <input
              type="range"
              min={100}
              max={1500}
              step={50}
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
              style={{ width: "100%" }}
            />
          </div>

          <div style={{ marginTop: 6, fontSize: 11, color: "#888", fontFamily: "monospace" }}>
            iter {k} / {history.length - 1}
          </div>

          <ConvergenceChart ista={ista} fista={fista} kIdx={k} algo={algo} />
        </div>

        <div>
          <CodePanel codeLines={CODE_LINES} algo={algo} />
          <SoftThresholdViz state={state} />
          <StatePanel state={state} algo={algo} />
        </div>
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Contour plot of g + iterate trajectory
// ============================================================
function ContourPlot({ history, kIdx }) {
  const W = 480, H = 480;
  const padL = 50, padR = 16, padT = 18, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -2.5, xmax = 2.5;
  const ymin = -2.5, ymax = 2.5;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;

  // Contour lines of g(x) = ½‖Ax − b‖² (ellipses centered at A^{-1}b)
  // Generate by sampling
  const rings = [];
  const center = (() => {
    // Solve Ax = b  →  x = A⁻¹b
    const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    return [
      (A[1][1] * B[0] - A[0][1] * B[1]) / det,
      (-A[1][0] * B[0] + A[0][0] * B[1]) / det,
    ];
  })();
  for (let r = 0.2; r <= 3.0; r += 0.4) {
    const ring = [];
    for (let i = 0; i <= 80; i++) {
      const t = (i / 80) * 2 * Math.PI;
      ring.push([center[0] + r * Math.cos(t), center[1] + r * Math.sin(t)]);
    }
    rings.push(ring);
  }

  // ‖x‖₁ ≤ const : diamonds — show one for the active level
  const lambdaLevel = 1.0;
  const diamond = [
    [lambdaLevel, 0],
    [0, lambdaLevel],
    [-lambdaLevel, 0],
    [0, -lambdaLevel],
  ];

  const traj = history.slice(0, kIdx + 1);
  const trajPath = traj.map((s) => `${xs(s.x[0])},${ys(s.x[1])}`).join(" ");
  const state = history[kIdx];
  const yPx = state.y ? { x: xs(state.y[0]), y: ys(state.y[1]) } : null;
  const xPx = { x: xs(state.x[0]), y: ys(state.x[1]) };
  const zPx = state.z ? { x: xs(state.z[0]), y: ys(state.z[1]) } : null;
  const xnPx = state.xNext ? { x: xs(state.xNext[0]), y: ys(state.xNext[1]) } : null;

  return (
    <div style={panel}>
      <svg width={W} height={H}>
        {/* axes */}
        <line x1={padL} y1={ys(0)} x2={padL + chartW} y2={ys(0)} stroke="#bbb" />
        <line x1={xs(0)} y1={padT} x2={xs(0)} y2={padT + chartH} stroke="#bbb" />

        {/* contour rings of g */}
        {rings.map((ring, i) => (
          <polyline
            key={i}
            points={ring.map(([x, y]) => `${xs(x)},${ys(y)}`).join(" ")}
            fill="none"
            stroke="#e5d8b8"
            strokeWidth={1}
          />
        ))}

        {/* L1 diamond (one level) */}
        <polygon
          points={diamond.map(([x, y]) => `${xs(x)},${ys(y)}`).join(" ")}
          fill="rgba(122, 61, 160, 0.06)"
          stroke="#7a3da0"
          strokeWidth={1.2}
          strokeDasharray="4,2"
        />

        {/* Center of g */}
        <circle cx={xs(center[0])} cy={ys(center[1])} r={4} fill="#1f4e3d" stroke="#fff" strokeWidth={1.5} />
        <text x={xs(center[0]) + 8} y={ys(center[1]) - 6} fontSize={10} fontFamily="monospace" fill="#1f4e3d">
          A⁻¹b
        </text>

        {/* True optimum */}
        <circle cx={xs(X_STAR[0])} cy={ys(X_STAR[1])} r={6} fill="none" stroke="#c8311c" strokeWidth={2.5} />
        <text x={xs(X_STAR[0]) + 9} y={ys(X_STAR[1]) + 4} fontSize={11} fontFamily="monospace" fill="#c8311c">
          x⋆
        </text>

        {/* Trajectory */}
        {traj.length > 1 && (
          <polyline points={trajPath} fill="none" stroke="#0b3da0" strokeWidth={2} strokeDasharray="3,2" />
        )}

        {/* y_k (extrapolation in FISTA) */}
        {yPx && state.y && (state.y[0] !== state.x[0] || state.y[1] !== state.x[1]) && (
          <>
            <line x1={xPx.x} y1={xPx.y} x2={yPx.x} y2={yPx.y} stroke="#7a3da0" strokeWidth={2} />
            <circle cx={yPx.x} cy={yPx.y} r={5} fill="#7a3da0" stroke="#fff" strokeWidth={2} />
            <text x={yPx.x + 8} y={yPx.y - 8} fontSize={10} fontFamily="monospace" fill="#7a3da0">
              yₖ
            </text>
          </>
        )}

        {/* z = y - τ∇g(y)  (gradient step before prox) */}
        {zPx && (
          <>
            <line
              x1={yPx ? yPx.x : xPx.x}
              y1={yPx ? yPx.y : xPx.y}
              x2={zPx.x}
              y2={zPx.y}
              stroke="#888"
              strokeWidth={1.5}
              strokeDasharray="2,2"
            />
            <circle cx={zPx.x} cy={zPx.y} r={4} fill="#888" />
            <text x={zPx.x + 6} y={zPx.y - 4} fontSize={10} fontFamily="monospace" fill="#666">
              z = y−τ∇g
            </text>
          </>
        )}

        {/* x_{k+1} = soft(z) */}
        {xnPx && zPx && (
          <>
            <line x1={zPx.x} y1={zPx.y} x2={xnPx.x} y2={xnPx.y} stroke="#c8311c" strokeWidth={1.5} />
            <circle cx={xnPx.x} cy={xnPx.y} r={5} fill="#c8311c" stroke="#fff" strokeWidth={2} />
            <text x={xnPx.x + 8} y={xnPx.y + 12} fontSize={10} fontFamily="monospace" fill="#c8311c">
              xₖ₊₁ = soft(z)
            </text>
          </>
        )}

        {/* current x */}
        <circle cx={xPx.x} cy={xPx.y} r={6} fill="#0b3da0" stroke="#fff" strokeWidth={2} />
        <text x={xPx.x - 22} y={xPx.y + 4} fontSize={10} fontFamily="monospace" fill="#0b3da0">
          xₖ
        </text>

        {/* axis labels */}
        {[-2, -1, 0, 1, 2].map((v) => (
          <text key={`xl${v}`} x={xs(v)} y={padT + chartH + 14} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        {[-2, -1, 0, 1, 2].map((v) => (
          <text key={`yl${v}`} x={padL - 6} y={ys(v) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}

        {/* legend */}
        <g transform={`translate(${padL + 10}, ${padT + 10})`}>
          <rect width={170} height={86} fill="rgba(255,255,255,0.92)" stroke="#ccc" />
          <line x1={6} y1={14} x2={20} y2={14} stroke="#e5d8b8" strokeWidth={2} />
          <text x={26} y={18} fontSize={11}>level sets of g</text>
          <line x1={6} y1={32} x2={20} y2={32} stroke="#7a3da0" strokeDasharray="4,2" strokeWidth={1.5} />
          <text x={26} y={36} fontSize={11}>‖x‖₁ ball (one level)</text>
          <circle cx={13} cy={50} r={5} fill="#7a3da0" />
          <text x={26} y={54} fontSize={11}>extrapolation y</text>
          <circle cx={13} cy={66} r={4} fill="#888" />
          <text x={26} y={70} fontSize={11}>gradient step z</text>
          <circle cx={13} cy={82} r={5} fill="#c8311c" />
          <text x={26} y={86} fontSize={11}>prox(z) → x_(k+1)</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Soft-threshold visualization (1D shrinkage curve)
// ============================================================
function SoftThresholdViz({ state }) {
  const W = 420, H = 160;
  const padL = 50, padR = 12, padT = 12, padB = 28;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -2.5, xmax = 2.5;
  const ymin = -2.5, ymax = 2.5;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;
  const kappa = LAMBDA * TAU;

  const N = 100;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const x = xmin + (i / N) * (xmax - xmin);
    pts.push([x, Math.sign(x) * Math.max(0, Math.abs(x) - kappa)]);
  }

  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 6,
        }}
      >
        Soft-threshold (per coordinate, κ = λτ ≈ {kappa.toFixed(3)})
      </div>
      <Tex block>
        {String.raw`\mathrm{soft}_{\kappa}(z) = \mathrm{sign}(z) \cdot \max(|z| - \kappa,\, 0)`}
      </Tex>
      <svg width={W} height={H}>
        <line x1={padL} y1={ys(0)} x2={padL + chartW} y2={ys(0)} stroke="#bbb" />
        <line x1={xs(0)} y1={padT} x2={xs(0)} y2={padT + chartH} stroke="#bbb" />
        {/* dead-zone band */}
        <rect
          x={xs(-kappa)}
          y={padT}
          width={xs(kappa) - xs(-kappa)}
          height={chartH}
          fill="rgba(245, 165, 36, 0.18)"
        />
        <line x1={xs(-kappa)} y1={padT} x2={xs(-kappa)} y2={padT + chartH} stroke="#f5a524" strokeDasharray="3,2" />
        <line x1={xs(kappa)} y1={padT} x2={xs(kappa)} y2={padT + chartH} stroke="#f5a524" strokeDasharray="3,2" />
        <text x={xs(0)} y={padT + 12} textAnchor="middle" fontSize={10} fill="#a37300">
          dead zone
        </text>
        {/* identity for reference */}
        <line x1={xs(xmin)} y1={ys(xmin)} x2={xs(xmax)} y2={ys(xmax)} stroke="#ddd" strokeDasharray="4,3" />
        {/* soft curve */}
        <polyline
          points={pts.map(([x, y]) => `${xs(x)},${ys(y)}`).join(" ")}
          fill="none"
          stroke="#0b3da0"
          strokeWidth={2.5}
        />
        {/* current z and x for each coordinate */}
        {state.z &&
          state.z.map((zi, i) => {
            const xi = state.xNext[i];
            const color = i === 0 ? "#c8311c" : "#7a3da0";
            return (
              <g key={i}>
                <line x1={xs(zi)} y1={ys(0)} x2={xs(zi)} y2={ys(xi)} stroke={color} strokeWidth={1.5} />
                <line x1={xs(zi)} y1={ys(xi)} x2={xs(0)} y2={ys(xi)} stroke={color} strokeWidth={1.5} strokeDasharray="2,2" />
                <circle cx={xs(zi)} cy={ys(xi)} r={4} fill={color} stroke="#fff" strokeWidth={1.5} />
                <text x={xs(zi) + 6} y={ys(xi) - 4} fontSize={9} fontFamily="monospace" fill={color}>
                  {`z[${i}]→x[${i}]`}
                </text>
              </g>
            );
          })}
      </svg>
    </div>
  );
}

// ============================================================
// Convergence chart
// ============================================================
function ConvergenceChart({ ista, fista, kIdx, algo }) {
  const W = 480, H = 160;
  const padL = 50, padR = 8, padT = 12, padB = 26;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const N = ista.length;
  const istaLog = ista.map((s) => Math.log10(Math.max(1e-12, s.f - F_STAR)));
  const fistaLog = fista.map((s) => Math.log10(Math.max(1e-12, s.f - F_STAR)));
  const all = [...istaLog, ...fistaLog].filter(isFinite);
  const yMax = Math.max(...all);
  const yMin = Math.min(...all);
  const xs = (i) => padL + (i / Math.max(1, N - 1)) * chartW;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * chartH;
  const istaPath = istaLog.map((v, i) => `${xs(i)},${ys(v)}`).join(" ");
  const fistaPath = fistaLog.map((v, i) => `${xs(i)},${ys(v)}`).join(" ");

  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        log₁₀(f(xₖ) − f⋆) — ISTA vs FISTA
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="#bbb" />
        <polyline points={istaPath} fill="none" stroke="#888" strokeWidth={1.7} />
        <polyline points={fistaPath} fill="none" stroke="#0b3da0" strokeWidth={2.2} />
        <circle
          cx={xs(kIdx)}
          cy={ys(algo === "fista" ? fistaLog[kIdx] : istaLog[kIdx])}
          r={5}
          fill="#c8311c"
        />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMax.toFixed(1)}
        </text>
        <text x={padL - 4} y={padT + chartH} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMin.toFixed(1)}
        </text>
        <g transform={`translate(${padL + 12}, ${padT + 4})`}>
          <line x1={0} y1={6} x2={16} y2={6} stroke="#888" strokeWidth={1.7} />
          <text x={20} y={10} fontSize={10}>ISTA</text>
          <line x1={50} y1={6} x2={66} y2={6} stroke="#0b3da0" strokeWidth={2.2} />
          <text x={70} y={10} fontSize={10}>FISTA</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// State panel
// ============================================================
function StatePanel({ state, algo }) {
  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 8 }}>
        State at iteration k = {state.k}
      </div>
      <KV
        k="x_k"
        v={`(${state.x[0].toFixed(4)}, ${state.x[1].toFixed(4)})`}
      />
      {algo === "fista" && state.y && (
        <KV
          k="y_k (extrapolation)"
          v={`(${state.y[0].toFixed(4)}, ${state.y[1].toFixed(4)})`}
        />
      )}
      {state.gradAtY && (
        <KV
          k="∇g(y_k)"
          v={`(${state.gradAtY[0].toFixed(3)}, ${state.gradAtY[1].toFixed(3)})`}
        />
      )}
      {state.z && (
        <KV
          k="z = y − τ∇g(y)"
          v={`(${state.z[0].toFixed(4)}, ${state.z[1].toFixed(4)})`}
        />
      )}
      {state.xNext && (
        <KV
          k="x_{k+1} = soft(z)"
          v={`(${state.xNext[0].toFixed(4)}, ${state.xNext[1].toFixed(4)})`}
          highlight
        />
      )}
      <KV k="f(x_k)" v={state.f.toFixed(6)} highlight />
      <KV k="f⋆" v={F_STAR.toFixed(6)} />
      <KV k="suboptimality" v={(state.f - F_STAR).toExponential(3)} highlight />
    </div>
  );
}

function KV({ k, v, highlight }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "3px 0",
        borderBottom: "1px dotted #eee",
      }}
    >
      <span style={{ color: "#666", fontSize: 12, fontFamily: "monospace" }}>{k}</span>
      <span
        style={{
          fontSize: 12,
          fontFamily: "monospace",
          color: highlight ? "#c8311c" : "#222",
          fontWeight: highlight ? 700 : 400,
        }}
      >
        {v}
      </span>
    </div>
  );
}

// ============================================================
// Code panel — highlight which block is active for the chosen algo
// ============================================================
function CodePanel({ codeLines, algo }) {
  const lineHeight = 22;
  return (
    <div
      style={{
        fontFamily: "'JetBrains Mono', Menlo, ui-monospace, monospace",
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
        if (i === 0) return null;
        const isBlank = line === "";
        // Highlight the relevant algorithm block
        const inIsta = i >= 18 && i <= 20;
        const inFista = i >= 22 && i <= 28;
        const active = (algo === "ista" && inIsta) || (algo === "fista" && inFista);
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              minHeight: lineHeight,
              background: active ? "#3b3526" : "transparent",
              borderLeft: active ? "3px solid #f5a524" : "3px solid transparent",
            }}
          >
            <span
              style={{
                width: 28,
                color: "#7f7864",
                textAlign: "right",
                marginRight: 12,
                fontSize: 11,
                userSelect: "none",
              }}
            >
              {i}
            </span>
            <span
              style={{
                color: active ? "#fff8e1" : isBlank ? "#7f7864" : "#e8e2d4",
                whiteSpace: "pre",
              }}
            >
              {line || " "}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Pedagogical notes
// ============================================================
function PedagogicalNotes() {
  return (
    <div style={{ marginTop: 28, padding: 16, background: "#fff8e1", borderRadius: 10, border: "1px solid #f5d68d" }}>
      <div style={{ fontWeight: 700, marginBottom: 6 }}>
        <Terminal size={14} style={{ verticalAlign: "middle", marginRight: 6 }} />
        Notes for class
      </div>
      <ul style={{ margin: 0, paddingLeft: 22, lineHeight: 1.6, fontSize: 14, color: "#3d2f00" }}>
        <li>
          <b>Composite splitting.</b> The whole point of proximal gradient
          is that <Tex>{`f = g + h`}</Tex> with{" "}
          <Tex>{`g`}</Tex> smooth and <Tex>{`h`}</Tex> non-smooth-but-simple.
          Take a gradient step on <Tex>{`g`}</Tex>, then apply{" "}
          <Tex>{String.raw`\mathrm{prox}_{\tau h}`}</Tex> for <Tex>{`h`}</Tex>.
          For <Tex>{String.raw`h(x) = \lambda \|x\|_1`}</Tex> the prox is
          element-wise soft-thresholding.
        </li>
        <li>
          <b>Why exact zeros.</b> The dead zone{" "}
          <Tex>{String.raw`|z_i| \le \lambda\tau`}</Tex> in
          <Tex>{String.raw`\;\mathrm{soft}_{\lambda\tau}(z)\;`}</Tex>
          collapses small gradient noise to zero. Coordinates with weak
          signal stay zero; that's how lasso produces sparse solutions.
        </li>
        <li>
          <b>Step size <Tex>{String.raw`\tau`}</Tex>.</b> Safe choice:{" "}
          <Tex>{String.raw`\tau = 1/L`}</Tex> where{" "}
          <Tex>{String.raw`L = \|A^TA\|_2`}</Tex> is the Lipschitz constant of
          <Tex>{String.raw`\;\nabla g`}</Tex>. Backtracking line search
          works too if <Tex>{`L`}</Tex> is unknown.
        </li>
        <li>
          <b>Nesterov's trick.</b> FISTA's momentum coefficient{" "}
          <Tex>{String.raw`\beta_k = (t_{k-1} - 1)/t_k`}</Tex> with{" "}
          <Tex>{String.raw`t_{k+1} = (1+\sqrt{1+4t_k^2})/2`}</Tex> looks
          arbitrary but is precisely what gives the{" "}
          <Tex>{String.raw`O(1/k^2)`}</Tex> rate. Beck & Teboulle (2009).
        </li>
        <li>
          <b>Generalizes everywhere.</b> Replace <Tex>{`h`}</Tex> with any
          'simple' function and you get a different algorithm: indicator of
          a convex set → projected gradient. Group-lasso → block soft-
          thresholding. Nuclear norm → singular-value soft-thresholding
          for matrix completion.
        </li>
      </ul>
    </div>
  );
}

// ============================================================
// Style atoms
// ============================================================
const panel = {
  background: "#fafafa",
  border: "1px solid #ddd",
  borderRadius: 8,
  padding: 12,
};
const problemBox = {
  marginBottom: 16,
  padding: "12px 16px",
  background: "#f6f4ee",
  border: "1px solid #ece8dd",
  borderRadius: 8,
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
