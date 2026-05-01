import React, { useState, useMemo, useCallback } from "react";
import { RotateCcw, Zap, Target } from "lucide-react";

/* ============================================================
   KKT CONDITIONS — INTERACTIVE 2D DEMO
   For ISE 5406 (Nonlinear Programming).

   Problem:
     minimize   f(x,y) = (x - cx)^2 + (y - cy)^2
     subject to g1(x,y) = x + 2y - b1 <= 0   (linear half-plane)
                g2(x,y) = x^2 + y^2 - r2^2 <= 0  (disk of radius r2)
                g3(x,y) = -x - b3 <= 0          ( x >= -b3 )

   At the constrained optimum x*, the KKT stationarity condition
     ∇f(x*) + Σ λ_i ∇g_i(x*) = 0,    λ_i >= 0,    λ_i g_i(x*) = 0
   must hold. We solve the KKT linear system on the active set
   via least squares, then display λ_i for each constraint.
   ============================================================ */

// ---------------- Plot helpers ----------------
const PLOT = { size: 480, lo: -2, hi: 2.5 };
const dataToPxX = (x) => ((x - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const dataToPxY = (y) =>
  PLOT.size - ((y - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;

const COLOR_F = "#2c5d8c";        // objective contours (blue)
const COLOR_FEASIBLE = "#cfe5da"; // light green
const COLOR_G1 = "#c8311c";       // red — linear half-plane
const COLOR_G2 = "#1f4e3d";       // green — disk
const COLOR_G3 = "#8c4a2c";       // brown — x >= 0 wall
const COLOR_GRAD_F = "#2c5d8c";
const COLOR_GRAD_G = "#a02822";

// ---------------- Problem setup ----------------
function makeProblem({ cx, cy, b1, r2, b3 }) {
  // f(x,y) = (x - cx)^2 + (y - cy)^2
  const f = (x, y) => (x - cx) * (x - cx) + (y - cy) * (y - cy);
  const grad_f = (x, y) => [2 * (x - cx), 2 * (y - cy)];

  // Each constraint: returns g(x,y) and its gradient ∇g(x,y).
  const constraints = [
    {
      name: "g₁: x + 2y ≤ " + b1.toFixed(2),
      short: "g₁",
      g: (x, y) => x + 2 * y - b1,
      grad: () => [1, 2],
      kind: "linear",
    },
    {
      name: "g₂: x² + y² ≤ " + (r2 * r2).toFixed(2) + "  (r=" + r2.toFixed(2) + ")",
      short: "g₂",
      g: (x, y) => x * x + y * y - r2 * r2,
      grad: (x, y) => [2 * x, 2 * y],
      kind: "disk",
      r: r2,
    },
    {
      name: "g₃: x ≥ " + (-b3).toFixed(2),
      short: "g₃",
      g: (x, y) => -x - b3,
      grad: () => [-1, 0],
      kind: "linear",
    },
  ];

  return { f, grad_f, constraints };
}

// ---------------- Projection onto feasible set ----------------
// Greedy projection: snap the point onto the boundary of any
// violated constraint, iterate a few rounds until stable. For
// these toy convex constraints this converges to the projection.
function project(x, y, constraints, iters = 30) {
  let px = x;
  let py = y;
  for (let t = 0; t < iters; t++) {
    let moved = false;
    for (const c of constraints) {
      const v = c.g(px, py);
      if (v > 1e-9) {
        if (c.kind === "linear") {
          // Project onto a*x + b*y = rhs.
          // Linear constraint has form g = n·x - rhs, so n = grad.
          const [nx, ny] = c.grad(px, py);
          const nn = nx * nx + ny * ny;
          // v = n·p - rhs, so step = v/||n||^2 * n
          px -= (v / nn) * nx;
          py -= (v / nn) * ny;
        } else if (c.kind === "disk") {
          // Project onto x^2 + y^2 = r^2: scale to radius r.
          const r = c.r;
          const rho = Math.sqrt(px * px + py * py);
          if (rho > 1e-12) {
            px *= r / rho;
            py *= r / rho;
          } else {
            px = r;
            py = 0;
          }
        }
        moved = true;
      }
    }
    if (!moved) break;
  }
  return [px, py];
}

function isFeasible(x, y, constraints, tol = 1e-7) {
  for (const c of constraints) {
    if (c.g(x, y) > tol) return false;
  }
  return true;
}

// ---------------- Projected gradient descent ----------------
function solvePGD(problem, x0 = 0, y0 = 0, lr = 0.05, iters = 200) {
  let [px, py] = project(x0, y0, problem.constraints);
  for (let t = 0; t < iters; t++) {
    const [gx, gy] = problem.grad_f(px, py);
    let xn = px - lr * gx;
    let yn = py - lr * gy;
    [xn, yn] = project(xn, yn, problem.constraints);
    const dx = xn - px;
    const dy = yn - py;
    px = xn;
    py = yn;
    if (Math.sqrt(dx * dx + dy * dy) < 1e-7) break;
  }
  return [px, py];
}

// ---------------- KKT multipliers via least squares on active set ----------------
// Stationarity: ∇f + Σ_{i in A} λ_i ∇g_i = 0
// Solve in least-squares sense. For 2 unknowns (∇f) and |A| multipliers,
// when |A| <= 2 we solve normal equations directly.
function solveKKTMultipliers(grad_f, activeGrads) {
  // grad_f: [g1, g2]; activeGrads: array of [g1, g2]
  // Set up A · λ = -grad_f where A is 2 x k (each column is ∇g_i).
  const k = activeGrads.length;
  if (k === 0) return [];
  // Build A^T A (k x k) and A^T b where b = -grad_f.
  const AtA = [];
  for (let i = 0; i < k; i++) {
    const row = [];
    for (let j = 0; j < k; j++) {
      row.push(
        activeGrads[i][0] * activeGrads[j][0] +
          activeGrads[i][1] * activeGrads[j][1]
      );
    }
    AtA.push(row);
  }
  const Atb = activeGrads.map(
    (gi) => -(gi[0] * grad_f[0] + gi[1] * grad_f[1])
  );
  // Solve k x k via Gaussian elimination (k is at most 3).
  const lambda = solveLinear(AtA, Atb);
  return lambda;
}

function solveLinear(A, b) {
  const n = A.length;
  // Augment.
  const M = A.map((row, i) => [...row, b[i]]);
  // Forward elimination with partial pivoting.
  for (let i = 0; i < n; i++) {
    let piv = i;
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
    }
    if (piv !== i) [M[i], M[piv]] = [M[piv], M[i]];
    if (Math.abs(M[i][i]) < 1e-12) {
      // Singular — return zeros.
      return new Array(n).fill(0);
    }
    for (let r = i + 1; r < n; r++) {
      const f = M[r][i] / M[i][i];
      for (let c = i; c <= n; c++) M[r][c] -= f * M[i][c];
    }
  }
  // Back substitution.
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return x;
}

// ---------------- SVG primitives ----------------
function Axes() {
  // Tick marks at integers.
  const ticks = [];
  for (let v = Math.ceil(PLOT.lo); v <= Math.floor(PLOT.hi); v++) {
    ticks.push(v);
  }
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
      {ticks.map((v) => (
        <g key={`tx-${v}`}>
          <line
            x1={dataToPxX(v)}
            y1={dataToPxY(0) - 3}
            x2={dataToPxX(v)}
            y2={dataToPxY(0) + 3}
            stroke="#999"
          />
          {v !== 0 && (
            <text
              x={dataToPxX(v)}
              y={dataToPxY(0) + 14}
              fontSize="10"
              fill="#888"
              textAnchor="middle"
              fontFamily="monospace"
            >
              {v}
            </text>
          )}
        </g>
      ))}
      {ticks.map((v) => (
        <g key={`ty-${v}`}>
          <line
            x1={dataToPxX(0) - 3}
            y1={dataToPxY(v)}
            x2={dataToPxX(0) + 3}
            y2={dataToPxY(v)}
            stroke="#999"
          />
          {v !== 0 && (
            <text
              x={dataToPxX(0) - 6}
              y={dataToPxY(v) + 3}
              fontSize="10"
              fill="#888"
              textAnchor="end"
              fontFamily="monospace"
            >
              {v}
            </text>
          )}
        </g>
      ))}
    </g>
  );
}

// Sample feasible region as a coarse grid of squares.
function FeasibleShade({ constraints }) {
  const N = 80;
  const cell = PLOT.size / N;
  const rects = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const x = PLOT.lo + ((i + 0.5) / N) * (PLOT.hi - PLOT.lo);
      const y = PLOT.lo + ((j + 0.5) / N) * (PLOT.hi - PLOT.lo);
      if (isFeasible(x, y, constraints, 1e-6)) {
        rects.push(
          <rect
            key={`${i}-${j}`}
            x={i * cell}
            y={PLOT.size - (j + 1) * cell}
            width={cell + 0.6}
            height={cell + 0.6}
            fill={COLOR_FEASIBLE}
            opacity={0.55}
          />
        );
      }
    }
  }
  return <g>{rects}</g>;
}

// Marching-squares-ish contour lines for f at fixed levels around the optimum.
function ObjectiveContours({ f, levels }) {
  const N = 110;
  const step = PLOT.size / N;
  const F = new Float64Array((N + 1) * (N + 1));
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const xd = PLOT.lo + (i / N) * (PLOT.hi - PLOT.lo);
      const yd = PLOT.lo + (j / N) * (PLOT.hi - PLOT.lo);
      F[i * (N + 1) + j] = f(xd, yd);
    }
  }
  const segs = [];
  const lerp = (a, b) => a / (a - b);
  for (const L of levels) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v00 = F[i * (N + 1) + j] - L;
        const v10 = F[(i + 1) * (N + 1) + j] - L;
        const v01 = F[i * (N + 1) + (j + 1)] - L;
        const v11 = F[(i + 1) * (N + 1) + (j + 1)] - L;
        let idx = 0;
        if (v00 > 0) idx |= 1;
        if (v10 > 0) idx |= 2;
        if (v11 > 0) idx |= 4;
        if (v01 > 0) idx |= 8;
        if (idx === 0 || idx === 15) continue;
        const x0 = i * step;
        const x1 = (i + 1) * step;
        const y0 = PLOT.size - j * step;
        const y1 = PLOT.size - (j + 1) * step;
        const edges = {};
        if ((v00 > 0) !== (v10 > 0))
          edges.b = [x0 + step * lerp(v00, v10), y0];
        if ((v10 > 0) !== (v11 > 0))
          edges.r = [x1, y0 - step * lerp(v10, v11)];
        if ((v01 > 0) !== (v11 > 0))
          edges.t = [x0 + step * lerp(v01, v11), y1];
        if ((v00 > 0) !== (v01 > 0))
          edges.l = [x0, y0 - step * lerp(v00, v01)];
        const e = Object.values(edges);
        if (e.length >= 2) segs.push([e[0], e[1], L]);
        if (e.length === 4) segs.push([e[2], e[3], L]);
      }
    }
  }
  const maxLevel = Math.max(...levels);
  return (
    <g>
      {segs.map((s, k) => {
        const t = s[2] / maxLevel;
        const opacity = 0.25 + 0.55 * (1 - t);
        return (
          <line
            key={k}
            x1={s[0][0]}
            y1={s[0][1]}
            x2={s[1][0]}
            y2={s[1][1]}
            stroke={COLOR_F}
            strokeWidth={1}
            opacity={opacity}
          />
        );
      })}
    </g>
  );
}

// Constraint boundaries.
function ConstraintBoundaries({ b1, r2, b3 }) {
  // g1: x + 2y = b1 → y = (b1 - x)/2, draw across plot.
  const g1 = (() => {
    const lo = PLOT.lo;
    const hi = PLOT.hi;
    const yLo = (b1 - lo) / 2;
    const yHi = (b1 - hi) / 2;
    return [
      [lo, yLo],
      [hi, yHi],
    ];
  })();

  // g2: circle of radius r2 centered at origin.
  const cx0 = dataToPxX(0);
  const cy0 = dataToPxY(0);
  // pixel radius along x:
  const rPx =
    ((r2) / (PLOT.hi - PLOT.lo)) * PLOT.size;

  // g3: x = -b3 vertical line.
  const g3x = -b3;

  return (
    <g>
      {/* g1 */}
      <line
        x1={dataToPxX(g1[0][0])}
        y1={dataToPxY(g1[0][1])}
        x2={dataToPxX(g1[1][0])}
        y2={dataToPxY(g1[1][1])}
        stroke={COLOR_G1}
        strokeWidth={2}
      />
      {/* g2 (circle) */}
      <circle
        cx={cx0}
        cy={cy0}
        r={rPx}
        fill="none"
        stroke={COLOR_G2}
        strokeWidth={2}
      />
      {/* g3 */}
      <line
        x1={dataToPxX(g3x)}
        y1={0}
        x2={dataToPxX(g3x)}
        y2={PLOT.size}
        stroke={COLOR_G3}
        strokeWidth={2}
      />
    </g>
  );
}

// ----------- Active-set cone (conic hull of active ∇gᵢ) -----------
// In 2D, the conic hull of {v₁, ..., vₖ} is one of:
//   • {0}              if k = 0       (interior optimum)
//   • a ray            if k = 1
//   • a wedge          if k ≥ 2 and the angular gap > π
//   • all of ℝ²        otherwise
// The KKT stationarity condition says −∇f lies inside this cone.
function ActiveCone({ centerPx, gradients, color = "#f5a524" }) {
  if (!gradients || gradients.length === 0) return null;

  // Normalize to unit directions and compute angles in DATA space.
  const dirs = gradients
    .map(([gx, gy]) => {
      const n = Math.hypot(gx, gy);
      return n > 1e-9 ? [gx / n, gy / n] : null;
    })
    .filter(Boolean);
  if (dirs.length === 0) return null;

  // Convert a data-space direction to a screen-space ray (y axis flipped).
  const screenAt = (theta, R) => [
    centerPx[0] + R * Math.cos(theta),
    centerPx[1] - R * Math.sin(theta),
  ];

  // Single active constraint → cone is exactly a ray.
  if (dirs.length === 1) {
    const theta = Math.atan2(dirs[0][1], dirs[0][0]);
    const R = PLOT.size * 1.5;
    const [ex, ey] = screenAt(theta, R);
    return (
      <line
        x1={centerPx[0]}
        y1={centerPx[1]}
        x2={ex}
        y2={ey}
        stroke={color}
        strokeWidth={10}
        strokeLinecap="round"
        opacity={0.18}
      />
    );
  }

  // ≥ 2 active constraints → look at angular extent.
  const angles = dirs.map((d) => Math.atan2(d[1], d[0])).sort((a, b) => a - b);
  let maxGap = 0;
  let maxGapStart = 0;
  for (let i = 0; i < angles.length; i++) {
    const next = i + 1 < angles.length ? angles[i + 1] : angles[0] + 2 * Math.PI;
    const gap = next - angles[i];
    if (gap > maxGap) {
      maxGap = gap;
      maxGapStart = angles[i];
    }
  }

  const R = PLOT.size * 1.5;

  // If the maximum angular gap is less than π, the cone covers all of ℝ²
  // (any direction is a non-negative combination of the active gradients).
  if (maxGap < Math.PI - 1e-3) {
    return (
      <rect
        x={0}
        y={0}
        width={PLOT.size}
        height={PLOT.size}
        fill={color}
        opacity={0.14}
      />
    );
  }

  // Otherwise the cone is the wedge from (maxGapStart + maxGap) to maxGapStart
  // going forward in angle (positive direction).
  const wedgeStart = maxGapStart + maxGap;
  const wedgeEnd = maxGapStart + 2 * Math.PI;
  const N = 24;
  const pts = [`${centerPx[0]},${centerPx[1]}`];
  for (let k = 0; k <= N; k++) {
    const theta = wedgeStart + (k / N) * (wedgeEnd - wedgeStart);
    const [px, py] = screenAt(theta, R);
    pts.push(`${px},${py}`);
  }
  return (
    <g>
      <polygon points={pts.join(" ")} fill={color} opacity={0.2} />
      <polygon
        points={pts.join(" ")}
        fill="none"
        stroke={color}
        strokeWidth={1.2}
        strokeDasharray="4 4"
        opacity={0.55}
      />
    </g>
  );
}

// Arrow with arrowhead.
function Arrow({ x0, y0, x1, y1, color, width = 2.2, label }) {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return null;
  const ux = dx / len;
  const uy = dy / len;
  // Arrowhead.
  const headLen = 9;
  const headW = 5;
  const bx = x1 - ux * headLen;
  const by = y1 - uy * headLen;
  const px = -uy;
  const py = ux;
  const h1x = bx + px * headW;
  const h1y = by + py * headW;
  const h2x = bx - px * headW;
  const h2y = by - py * headW;
  return (
    <g>
      <line
        x1={x0}
        y1={y0}
        x2={bx}
        y2={by}
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
      />
      <polygon
        points={`${x1},${y1} ${h1x},${h1y} ${h2x},${h2y}`}
        fill={color}
      />
      {label && (
        <text
          x={x1 + ux * 12}
          y={y1 + uy * 12}
          fontSize={12}
          fill={color}
          fontFamily="monospace"
          fontWeight={700}
          textAnchor="middle"
        >
          {label}
        </text>
      )}
    </g>
  );
}

// ============================================================
// MAIN COMPONENT
// ============================================================
export default function KKTDemo() {
  // Objective center (unconstrained optimum).
  const [cx, setCx] = useState(1.5);
  const [cy, setCy] = useState(1.0);

  // Constraint parameters.
  const [b1, setB1] = useState(2.0); // x + 2y <= b1
  const [r2, setR2] = useState(1.0); // x^2 + y^2 <= r2^2
  const [b3, setB3] = useState(0.0); // -x <= b3  i.e.  x >= -b3

  const problem = useMemo(
    () => makeProblem({ cx, cy, b1, r2, b3 }),
    [cx, cy, b1, r2, b3]
  );

  // Solve.
  const solution = useMemo(() => {
    const [xsRaw, ysRaw] = solvePGD(problem);

    // PGD's greedy projection ping-pongs near a vertex: whichever constraint
    // is projected onto LAST is exactly tight, the others are ~10⁻²ish off.
    // Snap to the intersection of all loosely-active constraints via Newton
    // before classifying the active set.
    const TOL_LOOSE = 0.05;
    const candidates = [];
    problem.constraints.forEach((c, idx) => {
      const v = c.g(xsRaw, ysRaw);
      if (v >= -TOL_LOOSE) candidates.push({ idx, c });
    });

    let xs = xsRaw,
      ys = ysRaw;
    for (let it = 0; it < 50 && candidates.length > 0; it++) {
      const grads = candidates.map(({ c }) => c.grad(xs, ys));
      const vals = candidates.map(({ c }) => c.g(xs, ys));
      const maxAbs = Math.max(...vals.map((v) => Math.abs(v)));
      if (maxAbs < 1e-12) break;
      if (candidates.length === 1) {
        const [a, b] = grads[0];
        const nn = a * a + b * b;
        if (nn < 1e-14) break;
        xs -= (vals[0] / nn) * a;
        ys -= (vals[0] / nn) * b;
      } else if (candidates.length === 2) {
        const [a1, b1] = grads[0];
        const [a2, b2] = grads[1];
        const det = a1 * b2 - a2 * b1;
        if (Math.abs(det) < 1e-12) break; // degenerate, keep PGD point
        const dx = (-vals[0] * b2 + vals[1] * b1) / det;
        const dy = (vals[0] * a2 - vals[1] * a1) / det;
        xs += dx;
        ys += dy;
        if (Math.hypot(dx, dy) < 1e-12) break;
      } else {
        // 3+ candidates in 2D is over-determined — fall back.
        break;
      }
    }
    // Safety: if the snap pushed us infeasible by more than the loose tol,
    // revert to the raw PGD point.
    let okSnap = true;
    for (const c of problem.constraints) {
      if (c.g(xs, ys) > 0.05) {
        okSnap = false;
        break;
      }
    }
    if (!okSnap) {
      xs = xsRaw;
      ys = ysRaw;
    }

    const [gx, gy] = problem.grad_f(xs, ys);
    // Determine active set with a tight post-snap tolerance.
    const tol = 5e-4;
    const active = [];
    const inactive = [];
    problem.constraints.forEach((c, idx) => {
      const v = c.g(xs, ys);
      if (Math.abs(v) < tol) active.push({ idx, c, v });
      else inactive.push({ idx, c, v });
    });
    // KKT multipliers on active set.
    const activeGrads = active.map(({ c }) => c.grad(xs, ys));
    const lambdaActive = solveKKTMultipliers([gx, gy], activeGrads);
    // Build full lambda vector (size 3).
    const lambda = [0, 0, 0];
    active.forEach(({ idx }, k) => {
      lambda[idx] = lambdaActive[k];
    });
    // Residual of stationarity.
    let rx = gx;
    let ry = gy;
    active.forEach(({ c }, k) => {
      const [a, b] = c.grad(xs, ys);
      rx += lambdaActive[k] * a;
      ry += lambdaActive[k] * b;
    });
    return {
      xs,
      ys,
      gradF: [gx, gy],
      active,
      inactive,
      lambda,
      residual: Math.sqrt(rx * rx + ry * ry),
    };
  }, [problem]);

  // Contour levels around the unconstrained optimum.
  const levels = useMemo(() => {
    const ls = [];
    for (let r = 0.2; r <= 4.0; r += 0.4) {
      ls.push(r * r);
    }
    return ls;
  }, []);

  const reset = useCallback(() => {
    setCx(1.5);
    setCy(1.0);
    setB1(2.0);
    setR2(1.0);
    setB3(0.0);
  }, []);

  // Preset scenarios.
  const presets = [
    {
      label: "g₂ active (disk binds)",
      apply: () => {
        setCx(1.5);
        setCy(1.0);
        setB1(2.5);
        setR2(1.0);
        setB3(0.5);
      },
    },
    {
      label: "g₁ active (line binds)",
      apply: () => {
        setCx(1.5);
        setCy(1.0);
        setB1(1.0);
        setR2(1.8);
        setB3(0.5);
      },
    },
    {
      label: "g₁ & g₂ both active",
      apply: () => {
        setCx(1.8);
        setCy(1.4);
        setB1(2.0);
        setR2(1.0);
        setB3(0.5);
      },
    },
    {
      label: "interior (no constraint active)",
      apply: () => {
        setCx(0.4);
        setCy(0.2);
        setB1(2.5);
        setR2(1.5);
        setB3(0.5);
      },
    },
  ];

  // Map data point to pixel.
  const optStarPx = [dataToPxX(solution.xs), dataToPxY(solution.ys)];
  const uncOptPx = [dataToPxX(cx), dataToPxY(cy)];

  // Gradient arrows: scale so a unit gradient is, say, 60 px.
  const gradPxScale = 28;
  const drawGradArrow = (x0, y0, gx, gy, color, label) => {
    const px0 = dataToPxX(x0);
    const py0 = dataToPxY(y0);
    // gradient in data → flip sign of y component for screen.
    const px1 = px0 + gx * gradPxScale;
    const py1 = py0 - gy * gradPxScale;
    return (
      <Arrow
        x0={px0}
        y0={py0}
        x1={px1}
        y1={py1}
        color={color}
        label={label}
      />
    );
  };

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        KKT Conditions — Constrained Optimization in 2D
      </h1>
      <p style={{ color: "#666", marginBottom: 28 }}>
        Watch the constrained minimum slide along an active constraint as you
        move the bounds. KKT stationarity says{" "}
        <b>−∇f lies inside the convex cone generated by the active ∇gᵢ</b>{" "}
        (shaded). The orange wedge IS that cone — when you tweak a slider so a
        new constraint becomes active, the cone widens to keep −∇f inside it.
      </p>

      <section style={section}>
        <h2 style={h2}>Problem</h2>
        <p style={p}>
          minimize&nbsp;<code style={code}>f(x,y) = (x − {cx.toFixed(2)})² + (y − {cy.toFixed(2)})²</code><br />
          subject to&nbsp;
          <span style={{ color: COLOR_G1 }}>
            <code style={code}>g₁: x + 2y − {b1.toFixed(2)} ≤ 0</code>
          </span>
          ,&nbsp;
          <span style={{ color: COLOR_G2 }}>
            <code style={code}>g₂: x² + y² − {(r2 * r2).toFixed(2)} ≤ 0</code>
          </span>
          ,&nbsp;
          <span style={{ color: COLOR_G3 }}>
            <code style={code}>g₃: −x − {b3.toFixed(2)} ≤ 0</code>
          </span>
          .
        </p>

        <div
          style={{
            display: "flex",
            gap: 24,
            alignItems: "flex-start",
            flexWrap: "wrap",
          }}
        >
          <svg
            width={PLOT.size}
            height={PLOT.size}
            style={{ background: "#fafafa", borderRadius: 6 }}
          >
            <Axes />
            <FeasibleShade constraints={problem.constraints} />

            {/* Active-set cone: −∇f must lie inside this region for KKT. */}
            <ActiveCone
              centerPx={optStarPx}
              gradients={solution.active.map(({ c }) => c.grad(solution.xs, solution.ys))}
            />

            <ObjectiveContours f={problem.f} levels={levels} />
            <ConstraintBoundaries b1={b1} r2={r2} b3={b3} />

            {/* Unconstrained optimum (open marker). */}
            <circle
              cx={uncOptPx[0]}
              cy={uncOptPx[1]}
              r={7}
              fill="white"
              stroke={COLOR_F}
              strokeWidth={2}
            />
            <text
              x={uncOptPx[0] + 11}
              y={uncOptPx[1] - 9}
              fontSize={11}
              fill={COLOR_F}
              fontFamily="monospace"
              fontWeight={700}
            >
              x̂ (uncon.)
            </text>

            {/* Constrained optimum (filled). */}
            <circle
              cx={optStarPx[0]}
              cy={optStarPx[1]}
              r={7.5}
              fill="#111"
              stroke="white"
              strokeWidth={1.5}
            />
            <text
              x={optStarPx[0] + 11}
              y={optStarPx[1] + 16}
              fontSize={11}
              fill="#111"
              fontFamily="monospace"
              fontWeight={700}
            >
              x* (con.)
            </text>

            {/* ∇f arrow at x*. */}
            {drawGradArrow(
              solution.xs,
              solution.ys,
              solution.gradF[0],
              solution.gradF[1],
              COLOR_GRAD_F,
              "∇f"
            )}

            {/* −∇f arrow at x* — must lie inside the active-set cone (shaded). */}
            {drawGradArrow(
              solution.xs,
              solution.ys,
              -solution.gradF[0],
              -solution.gradF[1],
              "#d4a017",
              "−∇f"
            )}

            {/* ∇g_i arrows at x* for active constraints. */}
            {solution.active.map(({ idx, c }) => {
              const [a, b] = c.grad(solution.xs, solution.ys);
              const colors = [COLOR_G1, COLOR_G2, COLOR_G3];
              return (
                <g key={`agrad-${idx}`}>
                  {drawGradArrow(
                    solution.xs,
                    solution.ys,
                    a,
                    b,
                    colors[idx],
                    `∇${c.short}`
                  )}
                </g>
              );
            })}

            {/* Show -Σ λ ∇g (should equal ∇f). Dashed companion arrow. */}
            {solution.active.length > 0 &&
              (() => {
                let mx = 0;
                let my = 0;
                solution.active.forEach(({ idx, c }) => {
                  const [a, b] = c.grad(solution.xs, solution.ys);
                  mx -= solution.lambda[idx] * a;
                  my -= solution.lambda[idx] * b;
                });
                const px0 = dataToPxX(solution.xs);
                const py0 = dataToPxY(solution.ys);
                const px1 = px0 + mx * gradPxScale;
                const py1 = py0 - my * gradPxScale;
                return (
                  <line
                    x1={px0}
                    y1={py0}
                    x2={px1}
                    y2={py1}
                    stroke="#444"
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    opacity={0.7}
                  />
                );
              })()}
          </svg>

          <div style={{ minWidth: 320, flex: 1 }}>
            <div style={controlGroup}>
              <label style={label}>
                obj. center cₓ: <b>{cx.toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={-1.5}
                max={2.5}
                step={0.05}
                value={cx}
                onChange={(e) => setCx(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>
            <div style={controlGroup}>
              <label style={label}>
                obj. center c_y: <b>{cy.toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={-1.5}
                max={2.5}
                step={0.05}
                value={cy}
                onChange={(e) => setCy(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>

            <div style={{ ...controlGroup, marginTop: 18 }}>
              <label style={{ ...label, color: COLOR_G1 }}>
                g₁ RHS: x + 2y ≤ <b>{b1.toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={-1.0}
                max={3.5}
                step={0.05}
                value={b1}
                onChange={(e) => setB1(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>
            <div style={controlGroup}>
              <label style={{ ...label, color: COLOR_G2 }}>
                g₂ radius r: <b>{r2.toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={0.3}
                max={2.0}
                step={0.05}
                value={r2}
                onChange={(e) => setR2(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>
            <div style={controlGroup}>
              <label style={{ ...label, color: COLOR_G3 }}>
                g₃ wall: x ≥ <b>{(-b3).toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={-1.5}
                max={1.5}
                step={0.05}
                value={-b3}
                onChange={(e) => setB3(-(+e.target.value))}
                style={{ width: "100%" }}
              />
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
              <button onClick={reset} style={btn}>
                <RotateCcw size={16} /> Reset
              </button>
              {presets.map((pr) => (
                <button key={pr.label} onClick={pr.apply} style={btnPrimary}>
                  <Zap size={16} /> {pr.label}
                </button>
              ))}
            </div>

            <div style={statBox}>
              <Stat label="x*" value={`(${solution.xs.toFixed(3)}, ${solution.ys.toFixed(3)})`} />
              <Stat label="f(x*)" value={problem.f(solution.xs, solution.ys).toFixed(4)} />
              <Stat
                label="‖∇f + Σλᵢ∇gᵢ‖"
                value={solution.residual.toFixed(4)}
              />
              <Stat label="∇f(x*)" value={`(${solution.gradF[0].toFixed(2)}, ${solution.gradF[1].toFixed(2)})`} />
            </div>

            <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
              <Target size={14} style={{ verticalAlign: "middle" }} /> Filled
              dot is the constrained optimum&nbsp;<b>x*</b>; the open marker is
              the unconstrained minimum&nbsp;<b>x̂</b>. Solid arrows show
              ∇f(x*) and the gradients ∇gᵢ of <b>active</b> constraints. The
              dashed arrow is&nbsp;−Σλᵢ∇gᵢ — when KKT holds, it equals ∇f.
            </p>
          </div>
        </div>
      </section>

      {/* KKT panel */}
      <section style={section}>
        <h2 style={h2}>KKT Conditions at x*</h2>
        <p style={p}>
          Stationarity: ∇f + Σ λᵢ ∇gᵢ = 0.&nbsp;&nbsp; Primal feasibility: gᵢ ≤
          0.&nbsp;&nbsp; Dual feasibility: λᵢ ≥ 0.&nbsp;&nbsp; Complementary
          slackness: λᵢ gᵢ = 0.
        </p>

        <table style={kktTable}>
          <thead>
            <tr>
              <th style={th}>i</th>
              <th style={th}>constraint</th>
              <th style={th}>gᵢ(x*)</th>
              <th style={th}>active?</th>
              <th style={th}>λᵢ</th>
              <th style={th}>λᵢ · gᵢ</th>
              <th style={th}>∇gᵢ(x*)</th>
            </tr>
          </thead>
          <tbody>
            {problem.constraints.map((c, i) => {
              const v = c.g(solution.xs, solution.ys);
              const isActive = solution.active.some((a) => a.idx === i);
              const grad = c.grad(solution.xs, solution.ys);
              const lam = solution.lambda[i];
              const colors = [COLOR_G1, COLOR_G2, COLOR_G3];
              return (
                <tr
                  key={i}
                  style={{
                    background: isActive ? "#fff8e6" : "#fff",
                  }}
                >
                  <td style={{ ...td, color: colors[i], fontWeight: 700 }}>
                    {i + 1}
                  </td>
                  <td style={{ ...td, color: colors[i] }}>{c.name}</td>
                  <td style={td}>{v.toFixed(4)}</td>
                  <td style={td}>
                    {isActive ? (
                      <span style={badgeActive}>ACTIVE</span>
                    ) : (
                      <span style={badgeInactive}>inactive</span>
                    )}
                  </td>
                  <td style={td}>{lam.toFixed(4)}</td>
                  <td style={td}>{(lam * v).toFixed(4)}</td>
                  <td style={td}>
                    ({grad[0].toFixed(2)}, {grad[1].toFixed(2)})
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>

        <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
          <b>How to read this.</b> A constraint is <b>active</b> when gᵢ(x*) ≈
          0 — the optimum sits on its boundary. Only active constraints
          carry a nonzero multiplier λᵢ; for inactive ones, complementary
          slackness forces λᵢ = 0. If KKT holds, the residual&nbsp;
          <code style={code}>‖∇f + Σ λᵢ ∇gᵢ‖</code> is essentially zero, and
          all multipliers are non-negative.
        </p>

        <div
          style={{
            marginTop: 12,
            padding: "10px 14px",
            background: "#f3f7fb",
            border: "1px solid #d6e2ed",
            borderRadius: 6,
            fontSize: 13,
            color: "#234",
          }}
        >
          <b>Try it.</b> (1) Shrink <b>g₂ radius</b> below ‖x̂‖ — the disk
          becomes binding; watch λ₂ rise and the optimum sit on the circle.
          (2) Slide <b>g₁ RHS</b> down past x̂ — the line cuts off x̂, the
          optimum moves onto the line, λ₁ &gt; 0. (3) Find the corner where
          the line touches the circle: <b>two</b> constraints active, two
          positive multipliers. (4) Push the unconstrained center inside
          the feasible region — every constraint inactive, every λᵢ = 0,
          and x* = x̂.
        </div>
      </section>

      {/* Legend section */}
      <section style={{ ...section, paddingTop: 20, paddingBottom: 20 }}>
        <h2 style={{ ...h2, fontSize: 16 }}>Legend</h2>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 18, fontSize: 13 }}>
          <LegendSwatch color={COLOR_F} label="∇f / contours of f" />
          <LegendSwatch color="#d4a017" label="−∇f" arrow />
          <LegendSwatch color={COLOR_FEASIBLE} label="feasible region" filled />
          <LegendSwatch color={COLOR_G1} label="g₁ boundary (line)" />
          <LegendSwatch color={COLOR_G2} label="g₂ boundary (circle)" />
          <LegendSwatch color={COLOR_G3} label="g₃ boundary (wall)" />
          <LegendSwatch color="#a02822" label="∇gᵢ (active)" arrow />
          <LegendSwatch color="#f5a524" label="normal cone at x*" wedge />
          <LegendSwatch color="#111" label="x* (constrained)" filled />
          <LegendSwatch color={COLOR_F} label="x̂ (unconstrained)" outline />
        </div>
        <p
          style={{
            marginTop: 14,
            color: "#555",
            fontSize: 13,
            lineHeight: 1.5,
            maxWidth: 720,
          }}
        >
          The shaded yellow wedge is the <b>normal cone</b> to the feasible
          set at <b>x*</b> — the conic hull of the active constraint
          gradients <code style={code}>{"{Σᵢ λᵢ ∇gᵢ(x*) : λᵢ ≥ 0, i ∈ A(x*)}"}</code>.
          KKT stationarity is exactly the geometric statement{" "}
          <b>−∇f ∈ N_C(x*)</b>: there's no feasible direction in which f decreases.
        </p>
      </section>
    </div>
  );
}

// ---------------- Small UI helpers ----------------
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function LegendSwatch({ color, label, filled, outline, arrow, wedge }) {
  let swatch;
  if (arrow) {
    swatch = (
      <svg width={22} height={16} style={{ display: "block" }}>
        <line x1={2} y1={8} x2={14} y2={8} stroke={color} strokeWidth={2.2} strokeLinecap="round" />
        <polygon points="20,8 14,5 14,11" fill={color} />
      </svg>
    );
  } else if (wedge) {
    // A small wedge — shaded fill + dashed outline, matching ActiveCone styling.
    swatch = (
      <svg width={20} height={16} style={{ display: "block" }}>
        <polygon points="3,14 17,2 17,14" fill={color} opacity={0.22} />
        <polygon
          points="3,14 17,2 17,14"
          fill="none"
          stroke={color}
          strokeWidth={1}
          strokeDasharray="3 2"
          opacity={0.7}
        />
      </svg>
    );
  } else {
    swatch = (
      <span
        style={{
          display: "inline-block",
          width: 16,
          height: 16,
          borderRadius: outline ? "50%" : 3,
          background: filled ? color : outline ? "white" : "transparent",
          border: outline
            ? `2px solid ${color}`
            : filled
            ? `1px solid ${color}`
            : `2px solid ${color}`,
        }}
      />
    );
  }
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      {swatch}
      <span style={{ color: "#444", fontFamily: "monospace" }}>{label}</span>
    </span>
  );
}

// ---------------- Styles ----------------
const section = {
  padding: "28px 26px",
  marginBottom: 24,
  background: "#fff",
  borderRadius: 12,
  border: "1px solid #e7e7e7",
  boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
};
const h2 = { fontSize: 20, fontWeight: 800, marginBottom: 8 };
const p = { color: "#444", lineHeight: 1.55, marginBottom: 14, maxWidth: 760 };
const code = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
const controlGroup = { marginTop: 8 };
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
  fontSize: 13,
};
const btnPrimary = {
  ...btn,
  background: "#111",
  color: "#fff",
  border: "1px solid #111",
};
const statBox = {
  marginTop: 14,
  padding: "8px 12px",
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 6,
};

const kktTable = {
  width: "100%",
  borderCollapse: "collapse",
  fontFamily: "monospace",
  fontSize: 13,
  marginTop: 8,
};
const th = {
  textAlign: "left",
  padding: "8px 10px",
  borderBottom: "2px solid #ddd",
  background: "#f7f7f7",
  fontWeight: 700,
};
const td = {
  padding: "8px 10px",
  borderBottom: "1px solid #eee",
};
const badgeActive = {
  display: "inline-block",
  padding: "2px 8px",
  background: "#fde68a",
  color: "#7c4a03",
  borderRadius: 10,
  fontSize: 11,
  fontWeight: 700,
};
const badgeInactive = {
  display: "inline-block",
  padding: "2px 8px",
  background: "#eee",
  color: "#888",
  borderRadius: 10,
  fontSize: 11,
  fontWeight: 600,
};
