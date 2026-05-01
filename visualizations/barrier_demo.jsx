import React, { useState, useEffect, useMemo, useRef } from "react";
import { Play, Pause, RotateCcw, Zap } from "lucide-react";

/* ============================================================
   INTERIOR-POINT / LOG-BARRIER METHOD — INTERACTIVE DEMO
   For ISE 5406. Visualizes the central path of the log-barrier
   subproblem
       B(x; t) = t · cᵀx  −  Σᵢ φᵢ(x)
   for several 2D problems: LPs, SOCPs, and SDPs.

     • LP   constraint  aᵢᵀx ≤ bᵢ           barrier  −log(bᵢ − aᵢᵀx)
     • SOC  constraint  ‖Cx+d‖ ≤ eᵀx+f      barrier  −log((eᵀx+f)² − ‖Cx+d‖²)
     • LMI  constraint  F₀+Σ xₖ Fₖ ⪰ 0      barrier  −log det F(x)

   The minimizers x*(t) trace out the "central path": small t pulls
   the iterate to the analytic center; large t pulls it toward the
   objective optimum on the boundary.
   ============================================================ */

// ============================================================
// Small N×N linear algebra (used by the LMI kernel)
// ============================================================
function matCopy(M) {
  return M.map((r) => r.slice());
}
function matMul(A, B) {
  const n = A.length;
  const m = B[0].length;
  const p = B.length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < p; k++) {
      const aik = A[i][k];
      for (let j = 0; j < m; j++) out[i][j] += aik * B[k][j];
    }
  }
  return out;
}
function matTrace(M) {
  let t = 0;
  for (let i = 0; i < M.length; i++) t += M[i][i];
  return t;
}
function matTraceMul(A, B) {
  // trace(A·B) for square N×N: Σᵢ Σⱼ A[i][j] · B[j][i]
  const n = A.length;
  let s = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += A[i][j] * B[j][i];
  return s;
}
function matInv(M) {
  const n = M.length;
  const A = M.map((row, i) => [
    ...row.slice(),
    ...new Array(n).fill(0).map((_, j) => (i === j ? 1 : 0)),
  ]);
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[pivot][i])) pivot = k;
    }
    if (Math.abs(A[pivot][i]) < 1e-14) return null;
    [A[i], A[pivot]] = [A[pivot], A[i]];
    const piv = A[i][i];
    for (let j = 0; j < 2 * n; j++) A[i][j] /= piv;
    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const f = A[k][i];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[k][j] -= f * A[i][j];
    }
  }
  return A.map((row) => row.slice(n));
}
function matDet(M) {
  const n = M.length;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  if (n === 3) {
    const a = M[0][0],
      b = M[0][1],
      c = M[0][2];
    const d = M[1][0],
      e = M[1][1],
      f = M[1][2];
    const g = M[2][0],
      h = M[2][1],
      i = M[2][2];
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  }
  // Generic LU
  const A = matCopy(M);
  let sign = 1;
  for (let i = 0; i < n; i++) {
    let pivot = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(A[k][i]) > Math.abs(A[pivot][i])) pivot = k;
    }
    if (pivot !== i) {
      [A[i], A[pivot]] = [A[pivot], A[i]];
      sign = -sign;
    }
    if (Math.abs(A[i][i]) < 1e-14) return 0;
    for (let k = i + 1; k < n; k++) {
      const f = A[k][i] / A[i][i];
      for (let j = i; j < n; j++) A[k][j] -= f * A[i][j];
    }
  }
  let d = sign;
  for (let i = 0; i < n; i++) d *= A[i][i];
  return d;
}
function minLeadingMinor(F) {
  // Sylvester's criterion: F ≻ 0 iff every leading principal minor > 0.
  // Return the smallest leading principal determinant (positive iff PSD).
  const n = F.length;
  let m = Infinity;
  for (let k = 1; k <= n; k++) {
    const sub = F.slice(0, k).map((r) => r.slice(0, k));
    const d = matDet(sub);
    if (d < m) m = d;
  }
  return m;
}

// ============================================================
// 2×2 solver (Hessian solve in the Newton step)
// ============================================================
function solve2x2(H, g) {
  const a = H[0][0],
    b = H[0][1],
    c = H[1][0],
    d = H[1][1];
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-14) return null;
  const inv = 1 / det;
  // d = -H⁻¹ g
  return [inv * (-d * g[0] + b * g[1]), inv * (c * g[0] - a * g[1])];
}

// ============================================================
// Constraint kernels
//   slack(c, x):              positive iff strictly interior
//   accumulate(c, x, g, H):   add this constraint's barrier ∇/∇² to g, H.
//                             returns false if not feasible at x.
// ============================================================
const KERNELS = {
  // Linear constraint: aᵀx ≤ b. Barrier φ = −log(b − aᵀx).
  linear: {
    slack: (c, x) => c.b - (c.a[0] * x[0] + c.a[1] * x[1]),
    accumulate: (c, x, g, H) => {
      const s = c.b - (c.a[0] * x[0] + c.a[1] * x[1]);
      if (s <= 0) return false;
      const inv = 1 / s,
        inv2 = inv * inv;
      g[0] += c.a[0] * inv;
      g[1] += c.a[1] * inv;
      H[0][0] += c.a[0] * c.a[0] * inv2;
      H[0][1] += c.a[0] * c.a[1] * inv2;
      H[1][0] += c.a[1] * c.a[0] * inv2;
      H[1][1] += c.a[1] * c.a[1] * inv2;
      return true;
    },
  },

  // Second-order cone: ‖Cx + d‖ ≤ eᵀx + f, with eᵀx + f > 0.
  // Barrier φ = −log(t² − uᵀu) where t = eᵀx + f, u = Cx + d.
  //   ∇ψ  = 2t·e − 2 Cᵀu
  //   ∇²ψ = 2 e eᵀ − 2 Cᵀ C
  //   ∇φ  = −∇ψ / ψ
  //   ∇²φ = −∇²ψ / ψ + ∇ψ ∇ψᵀ / ψ²
  soc: {
    slack: (c, x) => {
      const u0 = c.C[0][0] * x[0] + c.C[0][1] * x[1] + c.d[0];
      const u1 = c.C[1][0] * x[0] + c.C[1][1] * x[1] + c.d[1];
      const t = c.e[0] * x[0] + c.e[1] * x[1] + c.f;
      if (t <= 0) return -Infinity;
      return t * t - u0 * u0 - u1 * u1;
    },
    accumulate: (c, x, g, H) => {
      const u0 = c.C[0][0] * x[0] + c.C[0][1] * x[1] + c.d[0];
      const u1 = c.C[1][0] * x[0] + c.C[1][1] * x[1] + c.d[1];
      const t = c.e[0] * x[0] + c.e[1] * x[1] + c.f;
      if (t <= 0) return false;
      const psi = t * t - u0 * u0 - u1 * u1;
      if (psi <= 0) return false;
      // Cᵀu
      const ctu0 = c.C[0][0] * u0 + c.C[1][0] * u1;
      const ctu1 = c.C[0][1] * u0 + c.C[1][1] * u1;
      const gp0 = 2 * t * c.e[0] - 2 * ctu0;
      const gp1 = 2 * t * c.e[1] - 2 * ctu1;
      // CᵀC
      const ctc00 = c.C[0][0] * c.C[0][0] + c.C[1][0] * c.C[1][0];
      const ctc01 = c.C[0][0] * c.C[0][1] + c.C[1][0] * c.C[1][1];
      const ctc11 = c.C[0][1] * c.C[0][1] + c.C[1][1] * c.C[1][1];
      const Hp00 = 2 * c.e[0] * c.e[0] - 2 * ctc00;
      const Hp01 = 2 * c.e[0] * c.e[1] - 2 * ctc01;
      const Hp11 = 2 * c.e[1] * c.e[1] - 2 * ctc11;
      // ∇φ = −∇ψ/ψ
      g[0] += -gp0 / psi;
      g[1] += -gp1 / psi;
      // ∇²φ = ∇ψ ∇ψᵀ/ψ² − ∇²ψ/ψ
      const inv2 = 1 / (psi * psi);
      const invp = 1 / psi;
      H[0][0] += gp0 * gp0 * inv2 - Hp00 * invp;
      H[0][1] += gp0 * gp1 * inv2 - Hp01 * invp;
      H[1][0] += gp1 * gp0 * inv2 - Hp01 * invp;
      H[1][1] += gp1 * gp1 * inv2 - Hp11 * invp;
      return true;
    },
  },

  // Linear matrix inequality: F(x) = F₀ + x₁ F₁ + x₂ F₂  ⪰  0.
  // Barrier φ = −log det F(x).
  //   ∂φ/∂xₖ      = −trace(F⁻¹ Fₖ)
  //   ∂²φ/∂xⱼ∂xₖ  = +trace(F⁻¹ Fⱼ F⁻¹ Fₖ)
  lmi: {
    slack: (c, x) => {
      const F = lmiF(c, x);
      return minLeadingMinor(F);
    },
    accumulate: (c, x, g, H) => {
      const F = lmiF(c, x);
      if (minLeadingMinor(F) <= 0) return false;
      const Finv = matInv(F);
      if (!Finv) return false;
      // Mₖ = F⁻¹ Fₖ
      const M = c.Fs.map((Fk) => matMul(Finv, Fk));
      g[0] += -matTrace(M[0]);
      g[1] += -matTrace(M[1]);
      H[0][0] += matTraceMul(M[0], M[0]);
      const h01 = matTraceMul(M[0], M[1]);
      H[0][1] += h01;
      H[1][0] += h01;
      H[1][1] += matTraceMul(M[1], M[1]);
      return true;
    },
  },
};

function lmiF(c, x) {
  const n = c.F0.length;
  const F = c.F0.map((r) => r.slice());
  for (let k = 0; k < c.Fs.length; k++) {
    const xk = x[k];
    if (xk === 0) continue;
    const Fk = c.Fs[k];
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) F[i][j] += xk * Fk[i][j];
  }
  return F;
}

// ============================================================
// Generic problem operations
// ============================================================
function isFeasible(problem, x) {
  for (const c of problem.constraints) {
    if (KERNELS[c.type].slack(c, x) <= 1e-12) return false;
  }
  return true;
}
function gradHess(problem, x, t) {
  const g = [t * problem.c[0], t * problem.c[1]];
  const H = [
    [0, 0],
    [0, 0],
  ];
  for (const c of problem.constraints) {
    const ok = KERNELS[c.type].accumulate(c, x, g, H);
    if (!ok) return null;
  }
  return { g, H };
}
function barrierVal(problem, x, t) {
  let v = t * (problem.c[0] * x[0] + problem.c[1] * x[1]);
  for (const c of problem.constraints) {
    const s = KERNELS[c.type].slack(c, x);
    if (s <= 0) return Infinity;
    if (c.type === "lmi") {
      // φ = −log det F. Use the actual det (not the min-leading-minor proxy).
      const F = lmiF(c, x);
      const d = matDet(F);
      if (d <= 0) return Infinity;
      v -= Math.log(d);
    } else if (c.type === "soc") {
      v -= Math.log(s);
    } else {
      v -= Math.log(s);
    }
  }
  return v;
}

// One damped Newton step (backtracking line search; works for any kernel).
function newtonStep(problem, x, t) {
  const gh = gradHess(problem, x, t);
  if (!gh) return x;
  const { g, H } = gh;
  const d = solve2x2(H, g);
  if (!d) return x;
  let alpha = 1;
  const f0 = barrierVal(problem, x, t);
  const gd = g[0] * d[0] + g[1] * d[1];
  for (let k = 0; k < 60; k++) {
    const xNew = [x[0] + alpha * d[0], x[1] + alpha * d[1]];
    if (isFeasible(problem, xNew)) {
      const f1 = barrierVal(problem, xNew, t);
      if (isFinite(f1) && f1 < f0 + 0.1 * alpha * gd) return xNew;
    }
    alpha *= 0.5;
    if (alpha < 1e-14) break;
  }
  return x;
}

function solveBarrier(problem, x0, t, iters = 40) {
  let x = [x0[0], x0[1]];
  for (let k = 0; k < iters; k++) {
    const xn = newtonStep(problem, x, t);
    if (Math.hypot(xn[0] - x[0], xn[1] - x[1]) < 1e-10) {
      x = xn;
      break;
    }
    x = xn;
  }
  return x;
}

// ============================================================
// LP-only: vertex computation (for crisp polygon outline)
// ============================================================
function computeLPPolygon(problem) {
  const lin = problem.constraints.filter((c) => c.type === "linear");
  const A = lin.map((c) => c.a);
  const B = lin.map((c) => c.b);
  const verts = [];
  for (let i = 0; i < A.length; i++) {
    for (let j = i + 1; j < A.length; j++) {
      const det = A[i][0] * A[j][1] - A[i][1] * A[j][0];
      if (Math.abs(det) < 1e-12) continue;
      const x = (A[j][1] * B[i] - A[i][1] * B[j]) / det;
      const y = (A[i][0] * B[j] - A[j][0] * B[i]) / det;
      let ok = true;
      for (let k = 0; k < A.length; k++) {
        if (A[k][0] * x + A[k][1] * y > B[k] + 1e-9) {
          ok = false;
          break;
        }
      }
      if (ok) verts.push([x, y]);
    }
  }
  if (verts.length === 0) return [];
  let cx = 0,
    cy = 0;
  for (const v of verts) {
    cx += v[0];
    cy += v[1];
  }
  cx /= verts.length;
  cy /= verts.length;
  verts.sort(
    (p, q) =>
      Math.atan2(p[1] - cy, p[0] - cx) - Math.atan2(q[1] - cy, q[0] - cx)
  );
  const dedup = [];
  for (const v of verts) {
    if (
      dedup.length === 0 ||
      Math.hypot(
        v[0] - dedup[dedup.length - 1][0],
        v[1] - dedup[dedup.length - 1][1]
      ) > 1e-7
    )
      dedup.push(v);
  }
  return dedup;
}

function lpOptimum(problem, verts) {
  if (!verts.length) return null;
  let best = verts[0];
  let bestVal = problem.c[0] * best[0] + problem.c[1] * best[1];
  for (const v of verts) {
    const val = problem.c[0] * v[0] + problem.c[1] * v[1];
    if (val < bestVal) {
      bestVal = val;
      best = v;
    }
  }
  return best;
}

// ============================================================
// Problem registry
// ============================================================
const PROBLEMS = {
  lp_pentagon: {
    name: "LP — Pentagon",
    kind: "LP",
    objective: "min  −1.05·x₁ − x₂",
    constraintsText: [
      "x₁ ≥ 0,   x₂ ≥ 0",
      "x₁ + x₂ ≤ 2",
      "x₁ ≤ 1.5",
      "2x₁ + x₂ ≤ 3",
    ],
    blurb:
      "The classic LP example. Five linear constraints carve out a pentagon; the objective slopes toward the upper-right corner. The unique LP optimum is the vertex (1, 1).",
    c: [-1.05, -1],
    constraints: [
      { type: "linear", a: [-1, 0], b: 0, label: "x₁ ≥ 0" },
      { type: "linear", a: [0, -1], b: 0, label: "x₂ ≥ 0" },
      { type: "linear", a: [1, 1], b: 2, label: "x₁+x₂ ≤ 2" },
      { type: "linear", a: [1, 0], b: 1.5, label: "x₁ ≤ 1.5" },
      { type: "linear", a: [2, 1], b: 3, label: "2x₁+x₂ ≤ 3" },
    ],
    box: [-0.4, 2.2, -0.4, 2.2],
    optInit: [0.6, 0.6],
    optHint: [1, 1],
    optName: "LP optimum",
  },

  lp_triangle: {
    name: "LP — Triangle",
    kind: "LP",
    objective: "min  −0.5·x₁ − 2·x₂",
    constraintsText: ["x₁ ≥ 0,   x₂ ≥ 0", "x₁ + 2·x₂ ≤ 3"],
    blurb:
      "A minimal LP with three constraints. The central path is short and runs almost in a straight line from the analytic center to the unique optimum (0, 1.5). Useful as a sanity check before the more elaborate examples.",
    c: [-0.5, -2],
    constraints: [
      { type: "linear", a: [-1, 0], b: 0, label: "x₁ ≥ 0" },
      { type: "linear", a: [0, -1], b: 0, label: "x₂ ≥ 0" },
      { type: "linear", a: [1, 2], b: 3, label: "x₁+2x₂ ≤ 3" },
    ],
    box: [-0.3, 3.3, -0.3, 1.9],
    optInit: [0.5, 0.5],
    optHint: [0, 1.5],
    optName: "LP optimum",
  },

  socp_disk: {
    name: "SOCP — Disk",
    kind: "SOCP",
    objective: "min  −x₁ − x₂",
    constraintsText: ["‖x − (1, 1)‖₂ ≤ 1"],
    blurb:
      "A single second-order cone constraint produces a disk feasible region. The optimum sits on the boundary, exactly opposite the objective direction. Notice the central path is a straight line from the disk's center (the analytic center) toward that boundary point.",
    c: [-1, -1],
    constraints: [
      {
        type: "soc",
        C: [
          [1, 0],
          [0, 1],
        ],
        d: [-1, -1],
        e: [0, 0],
        f: 1,
        label: "‖x−(1,1)‖ ≤ 1",
      },
    ],
    box: [-0.4, 2.4, -0.4, 2.4],
    optInit: [1, 1],
    optHint: [1 + Math.SQRT1_2, 1 + Math.SQRT1_2],
    optName: "SOCP optimum",
  },

  socp_mixed: {
    name: "SOCP — Disk + Halfplanes",
    kind: "SOCP",
    objective: "min  −x₁ − 0.5·x₂",
    constraintsText: [
      "x₁ ≥ 0,   x₂ ≥ 0",
      "‖x − (1.2, 1.2)‖₂ ≤ 1.2",
    ],
    blurb:
      "Mix linear and conic constraints. The feasible region is the intersection of the first quadrant with a disk. The central path bends through the interior, hugging neither corner.",
    c: [-1, -0.5],
    constraints: [
      { type: "linear", a: [-1, 0], b: 0, label: "x₁ ≥ 0" },
      { type: "linear", a: [0, -1], b: 0, label: "x₂ ≥ 0" },
      {
        type: "soc",
        C: [
          [1, 0],
          [0, 1],
        ],
        d: [-1.2, -1.2],
        e: [0, 0],
        f: 1.2,
        label: "‖x−(1.2,1.2)‖ ≤ 1.2",
      },
    ],
    box: [-0.3, 2.7, -0.3, 2.7],
    optInit: [1.2, 1.2],
    optHint: null,
    optName: "SOCP optimum",
  },

  sdp_disk: {
    name: "SDP — Disk via 2×2 LMI",
    kind: "SDP",
    objective: "min  −x₁ − x₂",
    constraintsText: [
      "[[ 1+x₁,  x₂ ],",
      " [   x₂, 1−x₁ ]]  ⪰  0",
    ],
    blurb:
      "A 2×2 linear matrix inequality whose feasible set is the unit disk: F(x) ⪰ 0 iff (1+x₁)(1−x₁) − x₂² > 0. Same shape as a circular SOC, but the barrier is −log det F(x). Compare its central path to the SOCP disk above.",
    c: [-1, -1],
    constraints: [
      {
        type: "lmi",
        F0: [
          [1, 0],
          [0, 1],
        ],
        Fs: [
          [
            [1, 0],
            [0, -1],
          ],
          [
            [0, 1],
            [1, 0],
          ],
        ],
        label: "F(x) ⪰ 0",
      },
    ],
    box: [-1.3, 1.3, -1.3, 1.3],
    optInit: [0, 0],
    optHint: [Math.SQRT1_2, Math.SQRT1_2],
    optName: "SDP optimum",
  },

  sdp_parabolic: {
    name: "SDP — Parabolic Spectrahedron",
    kind: "SDP",
    objective: "min  −0.6·x₁ − x₂",
    constraintsText: [
      "F(x) = I + x₁·A + x₂·B  ⪰  0",
      "A = [[0,1,1],[1,0,0],[1,0,0]]",
      "B = [[0,0,0],[0,0,1],[0,1,0]]",
    ],
    blurb:
      "A 3×3 LMI gives a feasible region whose boundary is a curve no LP or SOCP can produce: the lower edge is a parabola (rank drops from 3 to 2) and the top is a line (the (1, 2)-block becomes singular). The central path is visibly non-smooth at the vertical edges where two constraints meet.",
    c: [-0.6, -1],
    constraints: [
      {
        type: "lmi",
        F0: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        Fs: [
          [
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
          ],
          [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
          ],
        ],
        label: "F(x) ⪰ 0",
      },
    ],
    box: [-1.05, 1.05, -1.1, 1.1],
    optInit: [0, 0.4],
    optHint: null,
    optName: "SDP optimum",
  },
};

const PROBLEM_KEYS = Object.keys(PROBLEMS);

// ============================================================
// Plot helpers
// ============================================================
const PLOT_SIZE = 520;

function makePlot(box) {
  const [xlo, xhi, ylo, yhi] = box;
  return {
    xlo,
    xhi,
    ylo,
    yhi,
    toPxX: (x) => ((x - xlo) / (xhi - xlo)) * PLOT_SIZE,
    toPxY: (y) => PLOT_SIZE - ((y - ylo) / (yhi - ylo)) * PLOT_SIZE,
  };
}

const COLOR_FEAS = "#dfeae0";
const COLOR_FEAS_EDGE = "#5b8a6c";
const COLOR_PATH = "#1f4e3d";
const COLOR_X = "#c8311c";
const COLOR_OPT = "#0b3da0";
const COLOR_AC = "#9a4caa";
const COLOR_OBJ = "#bbb";

// Rasterize the feasible region for problems that aren't pure LPs.
function FeasibilityCanvas({ problem, plot }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    const W = c.width,
      H = c.height;
    const img = ctx.createImageData(W, H);
    // Pre-decode COLOR_FEAS
    const fr = 0xdf,
      fg = 0xea,
      fb = 0xe0;
    for (let py = 0; py < H; py++) {
      const y = plot.yhi - (py / (H - 1)) * (plot.yhi - plot.ylo);
      for (let px = 0; px < W; px++) {
        const x = plot.xlo + (px / (W - 1)) * (plot.xhi - plot.xlo);
        const idx = (py * W + px) * 4;
        if (isFeasible(problem, [x, y])) {
          img.data[idx] = fr;
          img.data[idx + 1] = fg;
          img.data[idx + 2] = fb;
          img.data[idx + 3] = 230;
        } else {
          img.data[idx + 3] = 0;
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [problem, plot]);
  return (
    <canvas
      ref={ref}
      width={PLOT_SIZE}
      height={PLOT_SIZE}
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
    />
  );
}

// ============================================================
// Axes
// ============================================================
function Axes({ plot }) {
  const ticks = useMemo(() => {
    const span = Math.max(plot.xhi - plot.xlo, plot.yhi - plot.ylo);
    let step = 1;
    if (span <= 1.4) step = 0.25;
    else if (span <= 3.5) step = 0.5;
    else step = 1;
    const xt = [];
    const startX = Math.ceil(plot.xlo / step) * step;
    for (let x = startX; x <= plot.xhi + 1e-9; x += step) xt.push(+x.toFixed(6));
    const yt = [];
    const startY = Math.ceil(plot.ylo / step) * step;
    for (let y = startY; y <= plot.yhi + 1e-9; y += step) yt.push(+y.toFixed(6));
    return { xt, yt, step };
  }, [plot]);
  return (
    <g>
      <rect
        width={PLOT_SIZE}
        height={PLOT_SIZE}
        fill="#fafafa"
        stroke="#d4d4d4"
      />
      {ticks.xt.map((g) => (
        <line
          key={`gx-${g}`}
          x1={plot.toPxX(g)}
          y1={0}
          x2={plot.toPxX(g)}
          y2={PLOT_SIZE}
          stroke="#eee"
          strokeWidth={1}
        />
      ))}
      {ticks.yt.map((g) => (
        <line
          key={`gy-${g}`}
          x1={0}
          y1={plot.toPxY(g)}
          x2={PLOT_SIZE}
          y2={plot.toPxY(g)}
          stroke="#eee"
          strokeWidth={1}
        />
      ))}
      {plot.xlo <= 0 && plot.xhi >= 0 && (
        <line
          x1={plot.toPxX(0)}
          y1={0}
          x2={plot.toPxX(0)}
          y2={PLOT_SIZE}
          stroke="#bbb"
          strokeDasharray="3 3"
        />
      )}
      {plot.ylo <= 0 && plot.yhi >= 0 && (
        <line
          x1={0}
          y1={plot.toPxY(0)}
          x2={PLOT_SIZE}
          y2={plot.toPxY(0)}
          stroke="#bbb"
          strokeDasharray="3 3"
        />
      )}
      {ticks.xt
        .filter((g) => Math.abs(g) > 1e-9 || (plot.ylo <= 0 && plot.yhi >= 0))
        .map((g) => (
          <text
            key={`tx-${g}`}
            x={plot.toPxX(g)}
            y={
              plot.ylo <= 0 && plot.yhi >= 0
                ? plot.toPxY(0) + 14
                : PLOT_SIZE - 4
            }
            fontSize="10"
            fill="#888"
            textAnchor="middle"
            fontFamily="monospace"
          >
            {fmtTick(g)}
          </text>
        ))}
      {ticks.yt
        .filter((g) => Math.abs(g) > 1e-9 || !(plot.xlo <= 0 && plot.xhi >= 0))
        .map((g) => (
          <text
            key={`ty-${g}`}
            x={
              plot.xlo <= 0 && plot.xhi >= 0
                ? plot.toPxX(0) - 6
                : 4
            }
            y={plot.toPxY(g) + 3}
            fontSize="10"
            fill="#888"
            textAnchor={plot.xlo <= 0 && plot.xhi >= 0 ? "end" : "start"}
            fontFamily="monospace"
          >
            {fmtTick(g)}
          </text>
        ))}
    </g>
  );
}
function fmtTick(v) {
  if (Math.abs(v) < 1e-9) return "0";
  if (Number.isInteger(v)) return String(v);
  return v.toFixed(2).replace(/\.?0+$/, "");
}

// ============================================================
// Component
// ============================================================
export default function BarrierDemo() {
  const [problemKey, setProblemKey] = useState("lp_pentagon");
  const problem = PROBLEMS[problemKey];
  const plot = useMemo(() => makePlot(problem.box), [problem]);

  // LP-only polygon outline
  const lpVerts = useMemo(() => {
    if (problem.kind !== "LP") return [];
    return computeLPPolygon(problem);
  }, [problem]);
  const lpOpt = useMemo(
    () => (problem.kind === "LP" ? lpOptimum(problem, lpVerts) : null),
    [problem, lpVerts]
  );

  // Initial strictly feasible point (always falls back to optInit; for safety
  // we project lightly toward the analytic center if the user changes it).
  const x0 = problem.optInit;

  // Analytic center: barrier at t = 0 (we use a tiny t for numerics).
  const xAC = useMemo(() => solveBarrier(problem, x0, 0, 80), [problem, x0]);

  // Slider for log10(t)
  const [logT, setLogT] = useState(0);
  const t = Math.pow(10, logT);

  // Current x*(t)
  const xT = useMemo(() => solveBarrier(problem, x0, t, 60), [problem, x0, t]);

  // Pre-computed central path with warm-start
  const centralPath = useMemo(() => {
    const N = 70,
      lo = -2,
      hi = 4;
    const pts = [];
    let x = [x0[0], x0[1]];
    for (let k = 0; k < N; k++) {
      const lt = lo + ((hi - lo) * k) / (N - 1);
      const tt = Math.pow(10, lt);
      x = solveBarrier(problem, x, tt, 30);
      pts.push({ t: tt, x: [x[0], x[1]] });
    }
    return pts;
  }, [problem, x0]);

  // Reference "optimum" for displayed gap: exact LP vertex if known, else last
  // central-path point.
  const xOptDisplay = useMemo(() => {
    if (problem.optHint) return problem.optHint;
    if (lpOpt) return lpOpt;
    return centralPath[centralPath.length - 1].x;
  }, [problem, lpOpt, centralPath]);

  // Animation: sweep log10(t) and leave fading echoes.
  const [animating, setAnimating] = useState(false);
  const [trace, setTrace] = useState([]);
  const animRef = useRef(null);

  useEffect(() => {
    if (!animating) return;
    let lt = -2;
    const stepDelta = 0.05;
    animRef.current = setInterval(() => {
      lt += stepDelta;
      if (lt > 4) {
        lt = 4;
        setAnimating(false);
      }
      setLogT(lt);
      setTrace((tr) => {
        const next = tr
          .map((p) => ({ ...p, age: p.age + 1 }))
          .filter((p) => p.age < 80);
        return [
          ...next,
          { x: solveBarrier(problem, x0, Math.pow(10, lt), 30), age: 0 },
        ];
      });
    }, 60);
    return () => clearInterval(animRef.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [animating, problem]);

  // Reset trace and slider when the problem changes.
  useEffect(() => {
    setTrace([]);
    setLogT(0);
    setAnimating(false);
  }, [problemKey]);

  const reset = () => {
    setAnimating(false);
    setTrace([]);
    setLogT(0);
  };

  // Objective level lines (always linear in x).
  const levelLines = useMemo(() => {
    const cTxRef = problem.c[0] * xOptDisplay[0] + problem.c[1] * xOptDisplay[1];
    const cTxCur = problem.c[0] * xT[0] + problem.c[1] * xT[1];
    const lines = [];
    const span = 1.5 * Math.max(plot.xhi - plot.xlo, plot.yhi - plot.ylo) / 3;
    const n = 9;
    for (let k = 0; k < n; k++) {
      const v = cTxRef + 0.05 + (span * k) / (n - 1);
      // c₁ x + c₂ y = v  →  y = (v − c₁ x) / c₂  (skip if c₂ ≈ 0)
      let x1, y1, x2, y2;
      if (Math.abs(problem.c[1]) > 1e-9) {
        x1 = plot.xlo;
        x2 = plot.xhi;
        y1 = (v - problem.c[0] * x1) / problem.c[1];
        y2 = (v - problem.c[0] * x2) / problem.c[1];
      } else {
        x1 = x2 = v / problem.c[0];
        y1 = plot.ylo;
        y2 = plot.yhi;
      }
      lines.push({
        v,
        x1,
        y1,
        x2,
        y2,
        isCurrent: Math.abs(v - cTxCur) < 0.05,
      });
    }
    return lines;
  }, [problem, xT, xOptDisplay, plot]);

  // Telemetry: minimum slack and "active" constraint label.
  const slackInfo = useMemo(() => {
    let mi = -1,
      ms = Infinity;
    problem.constraints.forEach((c, i) => {
      const s = KERNELS[c.type].slack(c, xT);
      if (s < ms) {
        ms = s;
        mi = i;
      }
    });
    return {
      minSlack: ms,
      label: mi >= 0 ? problem.constraints[mi].label : "",
    };
  }, [problem, xT]);

  // SVG paths
  const polyD = useMemo(() => {
    if (problem.kind !== "LP" || lpVerts.length === 0) return null;
    return (
      "M " +
      lpVerts
        .map(
          (v) =>
            `${plot.toPxX(v[0]).toFixed(2)} ${plot.toPxY(v[1]).toFixed(2)}`
        )
        .join(" L ") +
      " Z"
    );
  }, [problem, lpVerts, plot]);

  const pathD = useMemo(
    () =>
      centralPath
        .map((p, i) => {
          const px = plot.toPxX(p.x[0]);
          const py = plot.toPxY(p.x[1]);
          return `${i === 0 ? "M" : "L"} ${px.toFixed(2)} ${py.toFixed(2)}`;
        })
        .join(" "),
    [centralPath, plot]
  );

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Interior-Point / Log-Barrier Method
      </h1>
      <p style={{ color: "#666", marginBottom: 20 }}>
        Pick an LP, SOCP, or SDP. Drag the barrier parameter <code style={code}>t</code>{" "}
        and watch the central path bend through the feasible region. Small{" "}
        <code style={code}>t</code> → barrier wins → analytic center. Large{" "}
        <code style={code}>t</code> → objective wins → boundary optimum.
      </p>

      <section style={section}>
        <div
          style={{
            display: "flex",
            gap: 16,
            alignItems: "center",
            flexWrap: "wrap",
            marginBottom: 12,
          }}
        >
          <label style={{ fontWeight: 700, fontSize: 15 }}>Problem:</label>
          <select
            value={problemKey}
            onChange={(e) => setProblemKey(e.target.value)}
            style={selectStyle}
          >
            {PROBLEM_KEYS.map((k) => (
              <option key={k} value={k}>
                {PROBLEMS[k].name}
              </option>
            ))}
          </select>
          <span
            style={{
              fontFamily: "monospace",
              fontSize: 12,
              color: "#fff",
              background: kindColor(problem.kind),
              padding: "3px 8px",
              borderRadius: 4,
              letterSpacing: "0.05em",
            }}
          >
            {problem.kind}
          </span>
        </div>

        <p style={{ ...p, fontSize: 14, marginBottom: 10 }}>{problem.blurb}</p>

        <pre style={pre}>
          {[problem.objective, "s.t.", ...problem.constraintsText.map((s) => "    " + s)].join(
            "\n"
          )}
        </pre>
      </section>

      <section style={section}>
        <h2 style={h2}>Visualization</h2>
        <p style={p}>
          Drag the slider to change <code style={code}>log₁₀ t</code>. Or click{" "}
          <b>Animate</b> to sweep <code style={code}>t</code> from small to large.
          The grey lines are level sets of the objective; the dark curve is the
          precomputed central path; the red dot is the current{" "}
          <code style={code}>x*(t)</code>; the purple dot is the analytic center;
          the blue circle (when shown) marks the optimum.
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
          <div
            style={{
              position: "relative",
              width: PLOT_SIZE,
              height: PLOT_SIZE,
              background: "#fafafa",
              borderRadius: 6,
              border: "1px solid #ddd",
              overflow: "hidden",
            }}
          >
            {problem.kind !== "LP" && (
              <FeasibilityCanvas problem={problem} plot={plot} />
            )}
            <svg
              width={PLOT_SIZE}
              height={PLOT_SIZE}
              style={{ position: "absolute", inset: 0 }}
            >
              <Axes plot={plot} />

              {/* LP polygon (crisp outline) */}
              {polyD && (
                <path
                  d={polyD}
                  fill={COLOR_FEAS}
                  fillOpacity={0.9}
                  stroke={COLOR_FEAS_EDGE}
                  strokeWidth={2}
                />
              )}
              {/* Non-LP boundary outline */}
              {problem.kind !== "LP" && (
                <BoundaryOutline problem={problem} plot={plot} />
              )}

              {/* Objective level lines */}
              {levelLines.map((l, i) => (
                <line
                  key={i}
                  x1={plot.toPxX(l.x1)}
                  y1={plot.toPxY(l.y1)}
                  x2={plot.toPxX(l.x2)}
                  y2={plot.toPxY(l.y2)}
                  stroke={l.isCurrent ? "#666" : COLOR_OBJ}
                  strokeWidth={l.isCurrent ? 1.5 : 0.8}
                  strokeDasharray={l.isCurrent ? "none" : "2 4"}
                  opacity={l.isCurrent ? 0.85 : 0.55}
                />
              ))}

              {/* −c arrow */}
              <ObjArrow problem={problem} plot={plot} />

              {/* Central path */}
              <path
                d={pathD}
                fill="none"
                stroke={COLOR_PATH}
                strokeWidth={2.2}
                strokeOpacity={0.85}
              />
              {centralPath
                .filter((_, i) => i % 8 === 0)
                .map((p, k) => (
                  <circle
                    key={`path-${k}`}
                    cx={plot.toPxX(p.x[0])}
                    cy={plot.toPxY(p.x[1])}
                    r={2}
                    fill={COLOR_PATH}
                    opacity={0.55}
                  />
                ))}

              {/* Animation echoes */}
              {trace.map((tr, i) => (
                <circle
                  key={`tr-${i}`}
                  cx={plot.toPxX(tr.x[0])}
                  cy={plot.toPxY(tr.x[1])}
                  r={4}
                  fill={COLOR_X}
                  opacity={Math.max(0, 0.45 * (1 - tr.age / 80))}
                />
              ))}

              {/* Analytic center */}
              <circle
                cx={plot.toPxX(xAC[0])}
                cy={plot.toPxY(xAC[1])}
                r={6}
                fill={COLOR_AC}
                stroke="white"
                strokeWidth={1.5}
              />
              <text
                x={plot.toPxX(xAC[0]) + 9}
                y={plot.toPxY(xAC[1]) - 6}
                fontSize="11"
                fill={COLOR_AC}
                fontFamily="monospace"
                fontWeight={700}
              >
                analytic center
              </text>

              {/* Optimum (only if known) */}
              {problem.optHint && (
                <>
                  <circle
                    cx={plot.toPxX(problem.optHint[0])}
                    cy={plot.toPxY(problem.optHint[1])}
                    r={7}
                    fill="none"
                    stroke={COLOR_OPT}
                    strokeWidth={2.5}
                  />
                  <circle
                    cx={plot.toPxX(problem.optHint[0])}
                    cy={plot.toPxY(problem.optHint[1])}
                    r={3}
                    fill={COLOR_OPT}
                  />
                  <text
                    x={plot.toPxX(problem.optHint[0]) + 11}
                    y={plot.toPxY(problem.optHint[1]) + 4}
                    fontSize="11"
                    fill={COLOR_OPT}
                    fontFamily="monospace"
                    fontWeight={700}
                  >
                    {problem.optName}
                  </text>
                </>
              )}

              {/* Current iterate */}
              <circle
                cx={plot.toPxX(xT[0])}
                cy={plot.toPxY(xT[1])}
                r={8}
                fill={COLOR_X}
                stroke="white"
                strokeWidth={2}
              />
              <text
                x={plot.toPxX(xT[0]) - 12}
                y={plot.toPxY(xT[1]) - 12}
                fontSize="11"
                fill={COLOR_X}
                fontFamily="monospace"
                fontWeight={700}
                textAnchor="end"
              >
                x*(t)
              </text>
            </svg>
          </div>

          <div style={{ minWidth: 320, flex: 1 }}>
            <div style={controlGroup}>
              <label style={label}>
                log₁₀ t: <b>{logT.toFixed(2)}</b>{" "}
                <span style={{ color: "#888" }}>
                  (t = {t.toExponential(2)})
                </span>
              </label>
              <input
                type="range"
                min={-2}
                max={4}
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
                <span>optimum →</span>
              </div>
            </div>

            <div
              style={{
                display: "flex",
                gap: 8,
                marginTop: 12,
                flexWrap: "wrap",
              }}
            >
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
              <Stat
                label="cᵀx*(t)"
                value={(problem.c[0] * xT[0] + problem.c[1] * xT[1]).toFixed(4)}
              />
              <Stat
                label="cᵀx_opt"
                value={(
                  problem.c[0] * xOptDisplay[0] +
                  problem.c[1] * xOptDisplay[1]
                ).toFixed(4)}
              />
              <Stat
                label="gap to optimum"
                value={(
                  problem.c[0] * xT[0] +
                  problem.c[1] * xT[1] -
                  (problem.c[0] * xOptDisplay[0] + problem.c[1] * xOptDisplay[1])
                ).toExponential(2)}
              />
              <Stat
                label="min slack"
                value={
                  isFinite(slackInfo.minSlack)
                    ? slackInfo.minSlack.toFixed(4)
                    : String(slackInfo.minSlack)
                }
              />
              <Stat
                label="active constraint"
                value={`${slackInfo.label}`}
              />
            </div>

            <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
              <b>Try this.</b> Compare the SOCP disk and SDP disk: same feasible
              region, but the LMI's <code style={code}>−log det</code> barrier
              produces a slightly different central path because the curvature
              of the barrier near the boundary is different. On the parabolic
              spectrahedron, the path bends — that's a curvature signal you
              can't get from any LP.
            </p>
            <p style={{ ...p, fontSize: 13 }}>
              For LP and SOCP, the duality gap on the central path is exactly{" "}
              <code style={code}>m / t</code>, where{" "}
              <code style={code}>m</code> is the number of constraints (or, for
              SDP, the matrix dimension). That's why doubling{" "}
              <code style={code}>t</code> halves the gap.
            </p>
          </div>
        </div>
      </section>

      <section style={section}>
        <h2 style={h2}>What's happening</h2>
        <p style={p}>
          For the chosen problem, B(x; t) is a strictly convex function on the
          interior of the feasible set. Each Newton step solves
        </p>
        <pre style={pre}>{`∇²B(x; t) · d  =  −∇B(x; t)`}</pre>
        <p style={p}>
          The gradient and Hessian come from the constraint kernels above. We
          then take a damped step <code style={code}>x ← x + α·d</code>,
          shrinking <code style={code}>α</code> by a factor of 2 (Armijo
          backtracking) until the new point is{" "}
          <i>strictly feasible</i> AND the barrier value decreases. That's why
          the iterate never crosses any constraint — it's an{" "}
          <i>interior-point</i> method, by construction.
        </p>
        <p style={p}>
          A practical IPM solves a sequence of subproblems with increasing{" "}
          <code style={code}>t</code> (or, equivalently, decreasing{" "}
          <code style={code}>μ = 1/t</code>), warm-starting each subproblem from
          the previous solution. The path you see precomputed here is exactly
          the trajectory the solver follows.
        </p>
      </section>
    </div>
  );
}

// ============================================================
// Smooth boundary outline for non-LP problems (marching squares
// on the feasibility indicator).
// ============================================================
function BoundaryOutline({ problem, plot }) {
  const segments = useMemo(() => {
    const N = 240;
    const grid = new Float32Array(N * N);
    for (let py = 0; py < N; py++) {
      const y = plot.ylo + (py / (N - 1)) * (plot.yhi - plot.ylo);
      for (let px = 0; px < N; px++) {
        const x = plot.xlo + (px / (N - 1)) * (plot.xhi - plot.xlo);
        // Use slack ≥ 0 as the indicator (positive = inside).
        let m = Infinity;
        for (const c of problem.constraints) {
          const s = KERNELS[c.type].slack(c, [x, y]);
          if (s < m) m = s;
        }
        grid[py * N + px] = isFinite(m) ? m : -1;
      }
    }
    const segs = [];
    const xAt = (i) => plot.xlo + (i / (N - 1)) * (plot.xhi - plot.xlo);
    const yAt = (j) => plot.ylo + (j / (N - 1)) * (plot.yhi - plot.ylo);
    for (let j = 0; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        const v00 = grid[j * N + i];
        const v10 = grid[j * N + i + 1];
        const v11 = grid[(j + 1) * N + i + 1];
        const v01 = grid[(j + 1) * N + i];
        const interp = (a, b, ax, ay, bx, by) => {
          const tt = a / (a - b);
          return [ax + tt * (bx - ax), ay + tt * (by - ay)];
        };
        const pts = [];
        const pushIf = (a, b, ax, ay, bx, by) => {
          if ((a > 0) !== (b > 0)) pts.push(interp(a, b, ax, ay, bx, by));
        };
        pushIf(v00, v10, xAt(i), yAt(j), xAt(i + 1), yAt(j));
        pushIf(v10, v11, xAt(i + 1), yAt(j), xAt(i + 1), yAt(j + 1));
        pushIf(v11, v01, xAt(i + 1), yAt(j + 1), xAt(i), yAt(j + 1));
        pushIf(v01, v00, xAt(i), yAt(j + 1), xAt(i), yAt(j));
        if (pts.length >= 2) segs.push([pts[0], pts[1]]);
      }
    }
    return segs;
  }, [problem, plot]);
  return (
    <g>
      {segments.map((s, k) => (
        <line
          key={k}
          x1={plot.toPxX(s[0][0])}
          y1={plot.toPxY(s[0][1])}
          x2={plot.toPxX(s[1][0])}
          y2={plot.toPxY(s[1][1])}
          stroke={COLOR_FEAS_EDGE}
          strokeWidth={1.4}
        />
      ))}
    </g>
  );
}

// ============================================================
// −c arrow (objective direction)
// ============================================================
function ObjArrow({ problem, plot }) {
  const cx = (plot.xlo + plot.xhi) / 2;
  const cy = plot.ylo + (plot.yhi - plot.ylo) * 0.08;
  const norm = Math.hypot(problem.c[0], problem.c[1]);
  const len = (plot.xhi - plot.xlo) * 0.08;
  const dx = (-problem.c[0] / norm) * len;
  const dy = (-problem.c[1] / norm) * len;
  return (
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
        x1={plot.toPxX(cx)}
        y1={plot.toPxY(cy)}
        x2={plot.toPxX(cx + dx)}
        y2={plot.toPxY(cy + dy)}
        stroke="#666"
        strokeWidth={1.5}
        markerEnd="url(#arr)"
      />
      <text
        x={plot.toPxX(cx + dx) + 4}
        y={plot.toPxY(cy + dy) - 2}
        fontSize="11"
        fill="#666"
        fontFamily="monospace"
      >
        −c
      </text>
    </g>
  );
}

// ============================================================
// Bits & pieces
// ============================================================
function Stat({ label, value }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "4px 0",
      }}
    >
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>
        {label}
      </span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}
function kindColor(k) {
  if (k === "LP") return "#1f4e3d";
  if (k === "SOCP") return "#0b3da0";
  if (k === "SDP") return "#7a3da0";
  return "#444";
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
  whiteSpace: "pre",
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
const selectStyle = {
  padding: "6px 10px",
  borderRadius: 6,
  border: "1px solid #ccc",
  fontSize: 14,
  background: "#fff",
  cursor: "pointer",
  fontFamily: "inherit",
};
