import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Play, Pause, RotateCcw, Zap } from "lucide-react";

/* ============================================================
   INTERIOR-POINT / LOG-BARRIER METHOD — INTERACTIVE DEMO
   For ISE 5406. Visualizes the central path of the log-barrier
   subproblem
       B(x; t) = t * c^T x  -  Σ log(b_i - a_i^T x)
   for a 2D LP with polygonal feasible region.

   - Small t  → barrier dominates → x*(t) ≈ analytic center.
   - Large t  → objective dominates → x*(t) → LP optimum (vertex).
   The minimizers x*(t) trace out the "central path".
   ============================================================ */

// ---------------- LP problem ----------------
// min c^T x  subject to  A x <= b
// A pentagon-ish region. Each row is one inequality a_i^T x <= b_i.
// Constraints:
//   -x <= 0            (x >= 0)
//   -y <= 0            (y >= 0)
//    x + y <= 2
//    x     <= 1.5
//   2x + y <= 3
// We use c = (-1.05, -1) so the LP optimum is the unique vertex at (1, 1),
// the intersection of x₁ + x₂ = 2 and 2x₁ + x₂ = 3. (With exactly c = (-1,-1)
// the optimum would be tied along the edge x₁+x₂ = 2.)
const C_VEC = [-1.05, -1];
const A_MAT = [
  [-1, 0],
  [0, -1],
  [1, 1],
  [1, 0],
  [2, 1],
];
const B_VEC = [0, 0, 2, 1.5, 3];

// ---------------- Linear algebra helpers (2x2) ----------------
function dot2(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}
function add2(a, b) {
  return [a[0] + b[0], a[1] + b[1]];
}
function scl2(a, k) {
  return [a[0] * k, a[1] * k];
}
// Solve a 2x2 linear system H d = -g for d. Returns null if near-singular.
function solve2x2(H, g) {
  const a = H[0][0],
    b = H[0][1],
    c = H[1][0],
    d = H[1][1];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-14) return null;
  const inv = 1 / det;
  // d_x = ( -d * gx + b * gy ) / det  ... no wait, H^{-1} = 1/det * [[d,-b],[-c,a]]
  const idx = inv * (d * -g[0] + -b * -g[1]);
  const idy = inv * (-c * -g[0] + a * -g[1]);
  return [idx, idy];
}

// ---------------- Barrier function, gradient, Hessian ----------------
// Slack s_i = b_i - a_i^T x. Must be > 0 (strictly feasible).
function slacks(x) {
  const s = new Array(A_MAT.length);
  for (let i = 0; i < A_MAT.length; i++) {
    s[i] = B_VEC[i] - (A_MAT[i][0] * x[0] + A_MAT[i][1] * x[1]);
  }
  return s;
}
function isInterior(x) {
  for (let i = 0; i < A_MAT.length; i++) {
    if (B_VEC[i] - (A_MAT[i][0] * x[0] + A_MAT[i][1] * x[1]) <= 1e-12) return false;
  }
  return true;
}
function barrierGradHess(x, t) {
  // ∇B = t·c + Σ a_i / s_i ;  ∇²B = Σ a_i a_i^T / s_i^2
  const g = [t * C_VEC[0], t * C_VEC[1]];
  const H = [
    [0, 0],
    [0, 0],
  ];
  for (let i = 0; i < A_MAT.length; i++) {
    const a = A_MAT[i];
    const s = B_VEC[i] - (a[0] * x[0] + a[1] * x[1]);
    if (s <= 0) {
      // Outside interior — caller should handle.
      return null;
    }
    g[0] += a[0] / s;
    g[1] += a[1] / s;
    const inv2 = 1 / (s * s);
    H[0][0] += a[0] * a[0] * inv2;
    H[0][1] += a[0] * a[1] * inv2;
    H[1][0] += a[1] * a[0] * inv2;
    H[1][1] += a[1] * a[1] * inv2;
  }
  return { g, H };
}

// One damped Newton step starting at x for parameter t. Returns updated x.
function newtonStep(x, t) {
  const gh = barrierGradHess(x, t);
  if (!gh) return x;
  const { g, H } = gh;
  const d = solve2x2(H, g);
  if (!d) return x;
  // Determine the maximum step alpha such that x + alpha*d stays interior:
  // we need b_i - a_i^T (x + alpha d) > 0 for all i.
  // i.e. alpha * (a_i^T d) < b_i - a_i^T x = s_i.
  let alphaMax = 1.0;
  for (let i = 0; i < A_MAT.length; i++) {
    const a = A_MAT[i];
    const s = B_VEC[i] - (a[0] * x[0] + a[1] * x[1]);
    const ad = a[0] * d[0] + a[1] * d[1];
    if (ad > 0) {
      // step shrinks slack
      const cap = (0.99 * s) / ad;
      if (cap < alphaMax) alphaMax = cap;
    }
  }
  // Backtracking on the barrier value (Armijo-ish, but here we just ensure
  // feasibility & decrease).
  let alpha = alphaMax;
  const beta = 0.5;
  const objAt = (xx) => {
    let v = t * (C_VEC[0] * xx[0] + C_VEC[1] * xx[1]);
    for (let i = 0; i < A_MAT.length; i++) {
      const s = B_VEC[i] - (A_MAT[i][0] * xx[0] + A_MAT[i][1] * xx[1]);
      if (s <= 0) return Infinity;
      v -= Math.log(s);
    }
    return v;
  };
  const f0 = objAt(x);
  const gd = g[0] * d[0] + g[1] * d[1]; // < 0 expected for descent
  for (let k = 0; k < 30; k++) {
    const xNew = [x[0] + alpha * d[0], x[1] + alpha * d[1]];
    const f1 = objAt(xNew);
    if (f1 < f0 + 0.1 * alpha * gd) {
      return xNew;
    }
    alpha *= beta;
    if (alpha < 1e-12) break;
  }
  return x;
}

// Solve barrier subproblem: starting from a feasible x0, run several Newton
// iterations at parameter t.
function solveBarrier(x0, t, iters = 30) {
  let x = [x0[0], x0[1]];
  for (let k = 0; k < iters; k++) {
    const xNew = newtonStep(x, t);
    const dx = [xNew[0] - x[0], xNew[1] - x[1]];
    x = xNew;
    if (Math.hypot(dx[0], dx[1]) < 1e-9) break;
  }
  return x;
}

// ---------------- Polygon / geometry ----------------
// Compute polygon vertices by intersecting all pairs of constraints,
// keeping those that satisfy every constraint.
function computePolygon() {
  const verts = [];
  for (let i = 0; i < A_MAT.length; i++) {
    for (let j = i + 1; j < A_MAT.length; j++) {
      const a1 = A_MAT[i],
        a2 = A_MAT[j];
      const det = a1[0] * a2[1] - a1[1] * a2[0];
      if (Math.abs(det) < 1e-12) continue;
      const x = (a2[1] * B_VEC[i] - a1[1] * B_VEC[j]) / det;
      const y = (a1[0] * B_VEC[j] - a2[0] * B_VEC[i]) / det;
      // Check feasibility
      let ok = true;
      for (let k = 0; k < A_MAT.length; k++) {
        const lhs = A_MAT[k][0] * x + A_MAT[k][1] * y;
        if (lhs > B_VEC[k] + 1e-9) {
          ok = false;
          break;
        }
      }
      if (ok) verts.push([x, y]);
    }
  }
  // Sort by angle around centroid.
  let cx = 0,
    cy = 0;
  for (const v of verts) {
    cx += v[0];
    cy += v[1];
  }
  cx /= verts.length;
  cy /= verts.length;
  verts.sort((p, q) => Math.atan2(p[1] - cy, p[0] - cx) - Math.atan2(q[1] - cy, q[0] - cx));
  // De-duplicate
  const dedup = [];
  for (const v of verts) {
    if (
      dedup.length === 0 ||
      Math.hypot(v[0] - dedup[dedup.length - 1][0], v[1] - dedup[dedup.length - 1][1]) > 1e-7
    )
      dedup.push(v);
  }
  return dedup;
}

// Find the LP optimum vertex (the polygon vertex minimizing c^T x).
function lpOptimum(verts) {
  let best = verts[0];
  let bestVal = dot2(C_VEC, best);
  for (const v of verts) {
    const val = dot2(C_VEC, v);
    if (val < bestVal) {
      bestVal = val;
      best = v;
    }
  }
  return best;
}

// ---------------- Plot helpers ----------------
const PLOT = { size: 520, lo: -0.4, hi: 2.2 };
const dataToPxX = (x) => ((x - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const dataToPxY = (y) => PLOT.size - ((y - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;

const COLOR_FEAS = "#dfeae0";
const COLOR_FEAS_EDGE = "#5b8a6c";
const COLOR_PATH = "#1f4e3d";
const COLOR_X = "#c8311c";
const COLOR_OPT = "#0b3da0";
const COLOR_AC = "#9a4caa";
const COLOR_OBJ = "#bbb";

function Axes() {
  return (
    <g>
      <rect width={PLOT.size} height={PLOT.size} fill="#fafafa" stroke="#d4d4d4" />
      {/* Light gridlines */}
      {[0, 0.5, 1, 1.5, 2].map((g) => (
        <g key={g}>
          <line
            x1={dataToPxX(g)}
            y1={0}
            x2={dataToPxX(g)}
            y2={PLOT.size}
            stroke="#eee"
            strokeWidth={1}
          />
          <line
            x1={0}
            y1={dataToPxY(g)}
            x2={PLOT.size}
            y2={dataToPxY(g)}
            stroke="#eee"
            strokeWidth={1}
          />
        </g>
      ))}
      {/* Axes */}
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
      {/* Tick labels */}
      {[0, 1, 2].map((g) => (
        <g key={`tx-${g}`}>
          <text
            x={dataToPxX(g)}
            y={dataToPxY(0) + 14}
            fontSize="10"
            fill="#888"
            textAnchor="middle"
            fontFamily="monospace"
          >
            {g}
          </text>
        </g>
      ))}
      {[0, 1, 2].map((g) =>
        g === 0 ? null : (
          <text
            key={`ty-${g}`}
            x={dataToPxX(0) - 6}
            y={dataToPxY(g) + 3}
            fontSize="10"
            fill="#888"
            textAnchor="end"
            fontFamily="monospace"
          >
            {g}
          </text>
        )
      )}
    </g>
  );
}

// ---------------- Component ----------------
export default function BarrierDemo() {
  const verts = useMemo(() => computePolygon(), []);
  const xOpt = useMemo(() => lpOptimum(verts), [verts]);

  // Initial interior point — take centroid of vertices, which is interior for
  // a convex polygon.
  const x0 = useMemo(() => {
    let sx = 0,
      sy = 0;
    for (const v of verts) {
      sx += v[0];
      sy += v[1];
    }
    return [sx / verts.length, sy / verts.length];
  }, [verts]);

  // Analytic center: minimizer of -Σ log s_i  (equivalent to t = 0, but we
  // use a tiny t for numerical stability).
  const xAnalyticCenter = useMemo(() => solveBarrier(x0, 0, 60), [x0]);

  // Slider: log10(t) ∈ [-2, 3]
  const [logT, setLogT] = useState(0); // t = 1 by default
  const t = Math.pow(10, logT);

  // Solve x*(t) at current slider value.
  const xT = useMemo(() => solveBarrier(x0, t, 50), [x0, t]);

  // Pre-compute central path for plotting (30 log-spaced values).
  const centralPath = useMemo(() => {
    const N = 60;
    const lo = -2,
      hi = 3;
    const pts = [];
    // Start from x0 and warm-start through ascending t (smoother path).
    let x = [x0[0], x0[1]];
    for (let k = 0; k < N; k++) {
      const lt = lo + ((hi - lo) * k) / (N - 1);
      const tt = Math.pow(10, lt);
      x = solveBarrier(x, tt, 30);
      pts.push({ t: tt, x: [x[0], x[1]] });
    }
    return pts;
  }, [x0]);

  // Animation: sweep log10(t) from -2 → 3, leaving fading trace.
  const [animating, setAnimating] = useState(false);
  const [trace, setTrace] = useState([]); // list of {x, age}
  const animRef = useRef(null);

  useEffect(() => {
    if (!animating) return;
    let lt = -2;
    const stepDelta = 0.05;
    animRef.current = setInterval(() => {
      lt += stepDelta;
      if (lt > 3) {
        lt = 3;
        setAnimating(false);
      }
      setLogT(lt);
      setTrace((tr) => {
        const next = tr
          .map((p) => ({ ...p, age: p.age + 1 }))
          .filter((p) => p.age < 80);
        return [...next, { x: solveBarrier(x0, Math.pow(10, lt), 30), age: 0 }];
      });
    }, 60);
    return () => clearInterval(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animating]);

  const reset = () => {
    setAnimating(false);
    setTrace([]);
    setLogT(0);
  };

  // Objective level lines: c^T x = const, i.e. -x - y = k, equivalently x+y = -k.
  // Pick a few values straddling the optimum.
  const levelLines = useMemo(() => {
    const cTxStar = dot2(C_VEC, xOpt);
    const cTxCurrent = dot2(C_VEC, xT);
    const lines = [];
    // Lines spaced around current value.
    const span = 1.5;
    const n = 7;
    for (let k = 0; k < n; k++) {
      const v = cTxStar + 0.05 + (span * k) / (n - 1);
      // Level line c₁ x + c₂ y = v. Solve y = (v - c₁ x) / c₂.
      const yAtLo = (v - C_VEC[0] * PLOT.lo) / C_VEC[1];
      const yAtHi = (v - C_VEC[0] * PLOT.hi) / C_VEC[1];
      lines.push({
        v,
        x1: PLOT.lo,
        y1: yAtLo,
        x2: PLOT.hi,
        y2: yAtHi,
        isCurrent: Math.abs(v - cTxCurrent) < 0.05,
      });
    }
    return lines;
  }, [xOpt, xT]);

  // Distance to nearest constraint at current x
  const minSlack = useMemo(() => {
    const s = slacks(xT);
    return Math.min(...s);
  }, [xT]);

  // Polygon path string
  const polyD = useMemo(() => {
    return (
      "M " +
      verts
        .map((v) => `${dataToPxX(v[0]).toFixed(2)} ${dataToPxY(v[1]).toFixed(2)}`)
        .join(" L ") +
      " Z"
    );
  }, [verts]);

  // Central path polyline string
  const pathD = useMemo(() => {
    return centralPath
      .map((p, i) => {
        const px = dataToPxX(p.x[0]);
        const py = dataToPxY(p.x[1]);
        return `${i === 0 ? "M" : "L"} ${px.toFixed(2)} ${py.toFixed(2)}`;
      })
      .join(" ");
  }, [centralPath]);

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Interior-Point / Log-Barrier Method
      </h1>
      <p style={{ color: "#666", marginBottom: 28 }}>
        Watch the central path bend through the feasible polytope as the barrier
        parameter <code style={code}>t</code> changes. Small <code style={code}>t</code>:
        the barrier dominates and pulls the iterate to the analytic center. Large{" "}
        <code style={code}>t</code>: the objective dominates and pulls it toward the
        LP optimum (a vertex).
      </p>

      <section style={section}>
        <h2 style={h2}>The Problem</h2>
        <p style={p}>
          We minimize the linear objective <code style={code}>c<sup>T</sup>x</code> over
          a polygonal feasible region:
        </p>
        <pre style={pre}>
{`min   -1.05·x₁ − x₂
s.t.   x₁ ≥ 0,   x₂ ≥ 0
       x₁ + x₂ ≤ 2
       x₁     ≤ 1.5
      2x₁ + x₂ ≤ 3`}
        </pre>
        <p style={p}>
          The log-barrier subproblem is
        </p>
        <pre style={pre}>
{`B(x; t) = t · cᵀx  −  Σᵢ log(bᵢ − aᵢᵀx)`}
        </pre>
        <p style={p}>
          For each <code style={code}>t &gt; 0</code>, B is strictly convex on the
          interior of the feasible region, so it has a unique minimizer{" "}
          <code style={code}>x*(t)</code>. The set <code style={code}>{"{x*(t) : t > 0}"}</code>{" "}
          is the <b>central path</b>. It connects the <b>analytic center</b>{" "}
          (<code style={code}>t → 0</code>) to the <b>LP optimum vertex</b>{" "}
          (<code style={code}>t → ∞</code>). Interior-point methods follow this path.
        </p>
      </section>

      <section style={section}>
        <h2 style={h2}>Visualization</h2>
        <p style={p}>
          Drag the slider to change <code style={code}>log₁₀ t</code>. Or click{" "}
          <b>Animate</b> to sweep <code style={code}>t</code> from small to large.
          The grey diagonal lines are level sets of the objective; the curve through
          the polygon is the precomputed central path; the red dot is the current{" "}
          <code style={code}>x*(t)</code>.
        </p>

        <div
          style={{
            display: "flex",
            gap: 24,
            alignItems: "flex-start",
            flexWrap: "wrap",
            marginTop: 8,
          }}
        >
          <svg
            width={PLOT.size}
            height={PLOT.size}
            style={{ background: "#fafafa", borderRadius: 6, border: "1px solid #ddd" }}
          >
            <Axes />

            {/* Feasible polygon */}
            <path
              d={polyD}
              fill={COLOR_FEAS}
              fillOpacity={0.9}
              stroke={COLOR_FEAS_EDGE}
              strokeWidth={2}
            />

            {/* Objective level lines */}
            {levelLines.map((l, i) => (
              <line
                key={i}
                x1={dataToPxX(l.x1)}
                y1={dataToPxY(l.y1)}
                x2={dataToPxX(l.x2)}
                y2={dataToPxY(l.y2)}
                stroke={l.isCurrent ? "#666" : COLOR_OBJ}
                strokeWidth={l.isCurrent ? 1.5 : 0.8}
                strokeDasharray={l.isCurrent ? "none" : "2 4"}
                opacity={l.isCurrent ? 0.85 : 0.6}
              />
            ))}

            {/* Direction-of-improvement arrow */}
            <g>
              <defs>
                <marker
                  id="arr"
                  markerWidth="8"
                  markerHeight="8"
                  refX="6"
                  refY="3"
                  orient="auto"
                >
                  <path d="M0,0 L0,6 L6,3 z" fill="#666" />
                </marker>
              </defs>
              <line
                x1={dataToPxX(1.7)}
                y1={dataToPxY(0.2)}
                x2={dataToPxX(2.0)}
                y2={dataToPxY(0.5)}
                stroke="#666"
                strokeWidth={1.5}
                markerEnd="url(#arr)"
              />
              <text
                x={dataToPxX(1.9)}
                y={dataToPxY(0.18)}
                fontSize="11"
                fill="#666"
                fontFamily="monospace"
              >
                −c
              </text>
            </g>

            {/* Central path */}
            <path
              d={pathD}
              fill="none"
              stroke={COLOR_PATH}
              strokeWidth={2.2}
              strokeOpacity={0.85}
            />
            {/* Path marker dots */}
            {centralPath
              .filter((_, i) => i % 6 === 0)
              .map((p, k) => (
                <circle
                  key={`path-${k}`}
                  cx={dataToPxX(p.x[0])}
                  cy={dataToPxY(p.x[1])}
                  r={2}
                  fill={COLOR_PATH}
                  opacity={0.6}
                />
              ))}

            {/* Trace (animation echoes) */}
            {trace.map((tr, i) => (
              <circle
                key={`tr-${i}`}
                cx={dataToPxX(tr.x[0])}
                cy={dataToPxY(tr.x[1])}
                r={4}
                fill={COLOR_X}
                opacity={Math.max(0, 0.45 * (1 - tr.age / 80))}
              />
            ))}

            {/* Analytic center */}
            <circle
              cx={dataToPxX(xAnalyticCenter[0])}
              cy={dataToPxY(xAnalyticCenter[1])}
              r={6}
              fill={COLOR_AC}
              stroke="white"
              strokeWidth={1.5}
            />
            <text
              x={dataToPxX(xAnalyticCenter[0]) + 9}
              y={dataToPxY(xAnalyticCenter[1]) - 6}
              fontSize="11"
              fill={COLOR_AC}
              fontFamily="monospace"
              fontWeight={700}
            >
              analytic center
            </text>

            {/* LP optimum */}
            <circle
              cx={dataToPxX(xOpt[0])}
              cy={dataToPxY(xOpt[1])}
              r={7}
              fill="none"
              stroke={COLOR_OPT}
              strokeWidth={2.5}
            />
            <circle
              cx={dataToPxX(xOpt[0])}
              cy={dataToPxY(xOpt[1])}
              r={3}
              fill={COLOR_OPT}
            />
            <text
              x={dataToPxX(xOpt[0]) + 11}
              y={dataToPxY(xOpt[1]) + 4}
              fontSize="11"
              fill={COLOR_OPT}
              fontFamily="monospace"
              fontWeight={700}
            >
              LP optimum
            </text>

            {/* Current x*(t) */}
            <circle
              cx={dataToPxX(xT[0])}
              cy={dataToPxY(xT[1])}
              r={8}
              fill={COLOR_X}
              stroke="white"
              strokeWidth={2}
            />
            <text
              x={dataToPxX(xT[0]) - 12}
              y={dataToPxY(xT[1]) - 12}
              fontSize="11"
              fill={COLOR_X}
              fontFamily="monospace"
              fontWeight={700}
              textAnchor="end"
            >
              x*(t)
            </text>
          </svg>

          <div style={{ minWidth: 300, flex: 1 }}>
            <div style={controlGroup}>
              <label style={label}>
                log₁₀ t: <b>{logT.toFixed(2)}</b>{" "}
                <span style={{ color: "#888" }}>(t = {t.toExponential(2)})</span>
              </label>
              <input
                type="range"
                min={-2}
                max={3}
                step={0.05}
                value={logT}
                onChange={(e) => {
                  setAnimating(false);
                  setLogT(+e.target.value);
                }}
                style={{ width: "100%" }}
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 11,
                  color: "#888",
                  fontFamily: "monospace",
                  marginTop: 2,
                }}
              >
                <span>← analytic center</span>
                <span>LP optimum →</span>
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
              <button
                onClick={() => {
                  if (animating) {
                    setAnimating(false);
                  } else {
                    setTrace([]);
                    setLogT(-2);
                    setAnimating(true);
                  }
                }}
                style={btnPrimary}
              >
                {animating ? <Pause size={16} /> : <Play size={16} />}
                {animating ? "Pause" : "Animate"}
              </button>
              <button onClick={() => setTrace([])} style={btn}>
                <Zap size={16} /> Clear Trace
              </button>
              <button onClick={reset} style={btn}>
                <RotateCcw size={16} /> Reset
              </button>
            </div>

            <div style={statBox}>
              <Stat label="t" value={t.toExponential(3)} />
              <Stat
                label="x*(t)"
                value={`(${xT[0].toFixed(4)}, ${xT[1].toFixed(4)})`}
              />
              <Stat label="cᵀx*(t)" value={dot2(C_VEC, xT).toFixed(4)} />
              <Stat
                label="cᵀx_opt"
                value={dot2(C_VEC, xOpt).toFixed(4)}
              />
              <Stat
                label="gap to optimum"
                value={(dot2(C_VEC, xT) - dot2(C_VEC, xOpt)).toExponential(2)}
              />
              <Stat label="min slack" value={minSlack.toFixed(4)} />
              <Stat
                label="active constraint"
                value={activeConstraintName(xT)}
              />
            </div>

            <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
              <b>Try this.</b> Slide <code style={code}>t</code> all the way left:{" "}
              <code style={code}>x*(t)</code> sits near the analytic center, well
              inside the polytope. Slide right: it slides along the curve toward the
              corner. Notice the <b>gap to optimum</b> shrinks roughly like{" "}
              <code style={code}>m / t</code> where <code style={code}>m</code> is
              the number of constraints — that's the duality gap of the barrier
              method.
            </p>
          </div>
        </div>
      </section>

      <section style={section}>
        <h2 style={h2}>What's happening</h2>
        <p style={p}>
          The Newton step at the current iterate solves
        </p>
        <pre style={pre}>
{`∇²B(x; t) · d  =  −∇B(x; t)
∇B  = t·c  +  Σᵢ aᵢ / (bᵢ − aᵢᵀx)
∇²B = Σᵢ aᵢ aᵢᵀ / (bᵢ − aᵢᵀx)²`}
        </pre>
        <p style={p}>
          Each Newton step is then damped: we shrink the step size so every slack{" "}
          <code style={code}>bᵢ − aᵢᵀx</code> stays strictly positive (we never
          cross a constraint). That's why the iterate stays in the interior — and
          why the method is called <i>interior-point</i>.
        </p>
        <p style={p}>
          Practical interior-point solvers don't fix <code style={code}>t</code>;
          they solve a sequence of subproblems with increasing{" "}
          <code style={code}>t</code> (or, equivalently, decreasing{" "}
          <code style={code}>μ = 1/t</code>), warm-starting each one from the
          previous solution. The path you see here is precisely the trajectory
          they follow.
        </p>
      </section>
    </div>
  );
}

// Identify the closest-to-tight constraint at x.
function activeConstraintName(x) {
  const names = ["x₁ ≥ 0", "x₂ ≥ 0", "x₁+x₂ ≤ 2", "x₁ ≤ 1.5", "2x₁+x₂ ≤ 3"];
  const s = slacks(x);
  let mi = 0;
  for (let i = 1; i < s.length; i++) if (s[i] < s[mi]) mi = i;
  return `${names[mi]} (${s[mi].toFixed(3)})`;
}

// ============================================================
// Shared UI bits
// ============================================================
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>
        {label}
      </span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>
        {value}
      </span>
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
const pre = {
  background: "#f6f4ee",
  padding: "10px 14px",
  borderRadius: 6,
  fontFamily: "monospace",
  fontSize: 13,
  lineHeight: 1.5,
  overflowX: "auto",
  marginBottom: 14,
  border: "1px solid #ece8dd",
};
const controlGroup = { marginTop: 10 };
const label = {
  display: "block",
  fontSize: 13,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
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
