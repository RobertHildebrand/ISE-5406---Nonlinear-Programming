import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward } from "lucide-react";

/* NEWTON'S METHOD vs GRADIENT DESCENT — for ISE 5406.
   GD:     x_{k+1} = x_k − α ∇f(x_k)
   Newton: x_{k+1} = x_k − α_n · H(x_k)⁻¹ ∇f(x_k)
   At each Newton iterate we draw the level set of the local quadratic
   model m(d) = f(x) + gᵀd + ½ dᵀHd that passes through d=0; its center
   is where the full Newton step (α_n=1) lands. */

// Test functions: f, ∇f, ∇²f, view box, optimum, suggested start.
const FUNCTIONS = {
  quad_ill: {
    name: "Ill-Conditioned Quadratic",
    formula: "f(x,y) = ½ (x² + 25 y²)",
    note:
      "κ(H) = 25. Steepest descent zig-zags; Newton lands at (0,0) in one step.",
    f: (x, y) => 0.5 * (x * x + 25 * y * y),
    g: (x, y) => [x, 25 * y],
    H: (x, y) => [[1, 0], [0, 25]],
    box: [-5, 5, -2, 2],
    opt: [0, 0],
    start: [-4, 1.2],
    levels: [0.05, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64],
    gdLrDefault: 0.04,
  },
  quad_well: {
    name: "Well-Conditioned Bowl",
    formula: "f(x,y) = ½ (x² + y²)",
    note:
      "κ(H) = 1. With α=1 both Newton and GD converge in one step (and they coincide).",
    f: (x, y) => 0.5 * (x * x + y * y),
    g: (x, y) => [x, y],
    H: (x, y) => [[1, 0], [0, 1]],
    box: [-5, 5, -5, 5],
    opt: [0, 0],
    start: [-4, 3],
    levels: [0.5, 2, 4.5, 8, 12.5, 18, 24.5, 32],
    gdLrDefault: 0.5,
  },
  rosenbrock: {
    name: "Rosenbrock",
    formula: "f(x,y) = (1−x)² + 100 (y − x²)²",
    note:
      "Curved valley. Newton uses curvature info to cut across; GD crawls along the floor.",
    f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
    g: (x, y) => [
      -2 * (1 - x) - 400 * x * (y - x * x),
      200 * (y - x * x),
    ],
    H: (x, y) => [
      [2 - 400 * (y - x * x) + 800 * x * x, -400 * x],
      [-400 * x, 200],
    ],
    box: [-2, 2, -1, 3],
    opt: [1, 1],
    start: [-1.2, 1.5],
    levels: [1, 5, 20, 50, 100, 200, 400, 800, 1500, 3000],
    gdLrDefault: 0.0015,
  },
};

// 2x2 linear algebra (inline per brief).
// Inverse of [[a,b],[c,d]] = (1/det) * [[d,-b],[-c,a]]
function inv2(M) {
  const [[a, b], [c, d]] = M;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-14) return null;
  const k = 1 / det;
  return [[d * k, -b * k], [-c * k, a * k]];
}
// Symmetric 2x2 eigendecomp:  λ± = (a+d)/2 ± sqrt(((a-d)/2)² + b²);
// eigenvector for λ is (b, λ−a) when b ≠ 0. Returns columns of V as eigenvectors.
function eig2sym(M) {
  const a = M[0][0], d = M[1][1];
  const b = 0.5 * (M[0][1] + M[1][0]); // symmetrize defensively
  const s = Math.sqrt(((a - d) * 0.5) ** 2 + b * b);
  const lam1 = (a + d) * 0.5 + s; // larger
  const lam2 = (a + d) * 0.5 - s; // smaller
  let v1x, v1y, v2x, v2y;
  if (Math.abs(b) < 1e-12) {
    [v1x, v1y, v2x, v2y] = a >= d ? [1, 0, 0, 1] : [0, 1, 1, 0];
  } else {
    [v1x, v1y] = [b, lam1 - a];
    [v2x, v2y] = [b, lam2 - a];
    const n1 = Math.hypot(v1x, v1y), n2 = Math.hypot(v2x, v2y);
    v1x /= n1; v1y /= n1; v2x /= n2; v2y /= n2;
  }
  return { lam: [lam1, lam2], V: [[v1x, v2x], [v1y, v2y]] };
}
const matvec = (M, v) => [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]];

// ---- Step kernels --------------------------------------------------------
function gdStep(fdef, x, lr) {
  const g = fdef.g(x[0], x[1]);
  return { next: [x[0] - lr * g[0], x[1] - lr * g[1]], g };
}
// Modified Newton: when the smallest eigenvalue is below eps, add τI to keep H
// positive-definite (so −H⁻¹g remains a descent direction far from optimum).
function newtonStep(fdef, x, alpha, regularize = true) {
  const g = fdef.g(x[0], x[1]);
  let H = fdef.H(x[0], x[1]);
  if (regularize) {
    const { lam } = eig2sym(H);
    const eps = 1e-3;
    const minLam = Math.min(lam[0], lam[1]);
    if (minLam < eps) {
      const tau = eps - minLam;
      H = [[H[0][0] + tau, H[0][1]], [H[1][0], H[1][1] + tau]];
    }
  }
  const Hinv = inv2(H);
  if (!Hinv) return { next: [x[0] - alpha * g[0], x[1] - alpha * g[1]], g, H, dir: g };
  const dir = matvec(Hinv, g);
  return { next: [x[0] - alpha * dir[0], x[1] - alpha * dir[1]], g, H, dir };
}

// ---- Plot geometry ------------------------------------------------------
const PLOT_W = 520;
const PLOT_H = 520;

function makeMap(box) {
  const [xmin, xmax, ymin, ymax] = box;
  return {
    box,
    toPx: (x, y) => [
      ((x - xmin) / (xmax - xmin)) * PLOT_W,
      PLOT_H - ((y - ymin) / (ymax - ymin)) * PLOT_H,
    ],
    pxPerX: PLOT_W / (xmax - xmin),
    pxPerY: PLOT_H / (ymax - ymin),
  };
}

// ---- Contour generation (marching squares) ------------------------------
function buildContours(fdef, box, levels, N = 100) {
  const [xmin, xmax, ymin, ymax] = box;
  const grid = new Float64Array((N + 1) * (N + 1));
  const dx = (xmax - xmin) / N;
  const dy = (ymax - ymin) / N;
  for (let j = 0; j <= N; j++) {
    const y = ymin + j * dy;
    for (let i = 0; i <= N; i++) {
      const x = xmin + i * dx;
      grid[j * (N + 1) + i] = fdef.f(x, y);
    }
  }
  const segsByLevel = levels.map(() => []);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const v00 = grid[j * (N + 1) + i];
      const v10 = grid[j * (N + 1) + i + 1];
      const v11 = grid[(j + 1) * (N + 1) + i + 1];
      const v01 = grid[(j + 1) * (N + 1) + i];
      const x0 = xmin + i * dx;
      const x1 = xmin + (i + 1) * dx;
      const y0 = ymin + j * dy;
      const y1 = ymin + (j + 1) * dy;
      for (let li = 0; li < levels.length; li++) {
        const L = levels[li];
        let mask = 0;
        if (v00 > L) mask |= 1;
        if (v10 > L) mask |= 2;
        if (v11 > L) mask |= 4;
        if (v01 > L) mask |= 8;
        if (mask === 0 || mask === 15) continue;
        const interp = (a, b, ax, ay, bx, by) => {
          const t = (L - a) / (b - a);
          return [ax + t * (bx - ax), ay + t * (by - ay)];
        };
        const pts = [];
        if ((v00 > L) !== (v10 > L)) pts.push(interp(v00, v10, x0, y0, x1, y0));
        if ((v10 > L) !== (v11 > L)) pts.push(interp(v10, v11, x1, y0, x1, y1));
        if ((v11 > L) !== (v01 > L)) pts.push(interp(v11, v01, x1, y1, x0, y1));
        if ((v01 > L) !== (v00 > L)) pts.push(interp(v01, v00, x0, y1, x0, y0));
        if (pts.length >= 2) segsByLevel[li].push([pts[0], pts[1]]);
        if (pts.length === 4) segsByLevel[li].push([pts[2], pts[3]]);
      }
    }
  }
  return segsByLevel;
}

// ============================================================
export default function NewtonDemo() {
  const [fnKey, setFnKey] = useState("quad_ill");
  const fdef = FUNCTIONS[fnKey];

  const [start, setStart] = useState(fdef.start);
  const [gdLr, setGdLr] = useState(fdef.gdLrDefault);
  const [newtonAlpha, setNewtonAlpha] = useState(1.0);
  const [showModelEllipse, setShowModelEllipse] = useState(true);
  const [regularize, setRegularize] = useState(true);

  // Per-method paths
  const [gdPath, setGdPath] = useState([fdef.start.slice()]);
  const [newtonPath, setNewtonPath] = useState([fdef.start.slice()]);
  const [running, setRunning] = useState(false);

  // When fn changes: snap start, lr, and reset paths
  useEffect(() => {
    const f = FUNCTIONS[fnKey];
    setStart(f.start);
    setGdLr(f.gdLrDefault);
    setNewtonAlpha(1.0);
    setGdPath([f.start.slice()]);
    setNewtonPath([f.start.slice()]);
    setRunning(false);
  }, [fnKey]);

  // When start changes (sliders): reset both paths to the new start
  useEffect(() => {
    setGdPath([start.slice()]);
    setNewtonPath([start.slice()]);
  }, [start[0], start[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  // Single step
  const stepOnce = useCallback(() => {
    setGdPath((prev) => {
      const x = prev[prev.length - 1];
      const { next } = gdStep(fdef, x, gdLr);
      if (
        !isFinite(next[0]) ||
        !isFinite(next[1]) ||
        Math.abs(next[0]) > 1e6 ||
        Math.abs(next[1]) > 1e6
      )
        return prev;
      return [...prev, next];
    });
    setNewtonPath((prev) => {
      const x = prev[prev.length - 1];
      const { next } = newtonStep(fdef, x, newtonAlpha, regularize);
      if (
        !isFinite(next[0]) ||
        !isFinite(next[1]) ||
        Math.abs(next[0]) > 1e6 ||
        Math.abs(next[1]) > 1e6
      )
        return prev;
      return [...prev, next];
    });
  }, [fdef, gdLr, newtonAlpha, regularize]);

  // Run-to-convergence (animated)
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      stepOnce();
    }, 120);
    return () => clearInterval(id);
  }, [running, stepOnce]);

  // Auto-stop heuristic: both gradients tiny, OR each path has > 200 points.
  useEffect(() => {
    if (!running) return;
    const lastGd = gdPath[gdPath.length - 1];
    const lastN = newtonPath[newtonPath.length - 1];
    const gG = fdef.g(lastGd[0], lastGd[1]);
    const gN = fdef.g(lastN[0], lastN[1]);
    const done =
      (Math.hypot(gG[0], gG[1]) < 1e-6 && Math.hypot(gN[0], gN[1]) < 1e-6) ||
      gdPath.length > 400 ||
      newtonPath.length > 400;
    if (done) setRunning(false);
  }, [gdPath, newtonPath, fdef, running]);

  const reset = () => {
    setGdPath([start.slice()]);
    setNewtonPath([start.slice()]);
    setRunning(false);
  };

  // ---------- Geometry / contours (memoized) ----------
  const map = useMemo(() => makeMap(fdef.box), [fdef.box]);
  const contourSegs = useMemo(
    () => buildContours(fdef, fdef.box, fdef.levels, 110),
    [fdef]
  );

  // ---------- Current Newton iterate info: model ellipse, eigenvalues, etc. ----------
  const currentNewton = newtonPath[newtonPath.length - 1];
  const newtonInfo = useMemo(() => {
    const x = currentNewton;
    const g = fdef.g(x[0], x[1]);
    let H = fdef.H(x[0], x[1]);
    if (regularize) {
      const { lam } = eig2sym(H);
      const minLam = Math.min(lam[0], lam[1]);
      const eps = 1e-3;
      if (minLam < eps) {
        const tau = eps - minLam;
        H = [
          [H[0][0] + tau, H[0][1]],
          [H[1][0], H[1][1] + tau],
        ];
      }
    }
    const Hinv = inv2(H);
    const dir = Hinv ? matvec(Hinv, g) : [0, 0];
    const xNew = [x[0] - newtonAlpha * dir[0], x[1] - newtonAlpha * dir[1]];

    const eig = eig2sym(H);
    const [lam1, lam2] = eig.lam;
    const lamMax = Math.max(Math.abs(lam1), Math.abs(lam2));
    const lamMin = Math.min(Math.abs(lam1), Math.abs(lam2));
    const cond = lamMin > 0 ? lamMax / lamMin : Infinity;

    // Local quadratic model:  m(d) = f(x) + gᵀd + ½ dᵀ H d
    // Centered:  m(d) = f(xNew) + ½(d−d*)ᵀ H (d−d*)  with  d* = −H⁻¹g.
    // We draw the level set passing through d=0 (i.e. through x itself):
    //   ½(d*)ᵀ H (d*) = ½ gᵀ H⁻¹ g  =: ε
    // Semi-axes along eigenvectors v_i: a_i = sqrt(2ε / λ_i).
    let eps = 1;
    if (Hinv) {
      const Hg = matvec(Hinv, g);
      eps = Math.max(0.5 * (g[0] * Hg[0] + g[1] * Hg[1]), 1e-12);
    }
    const positive = lam1 > 0 && lam2 > 0;
    const a = positive ? Math.sqrt((2 * eps) / lam1) : 0;
    const b = positive ? Math.sqrt((2 * eps) / lam2) : 0;
    const angle = Math.atan2(eig.V[1][0], eig.V[0][0]);

    return {
      x, g, H, dir, xNew, cond, lam1, lam2,
      ellipse: { center: xNew, a, b, angle, positive },
      f: fdef.f(x[0], x[1]),
      gnorm: Math.hypot(g[0], g[1]),
    };
  }, [currentNewton, fdef, newtonAlpha, regularize]);

  // ---------- Step previews (arrows from each method's current iterate) ----------
  const gdPreview = useMemo(() => {
    const x = gdPath[gdPath.length - 1];
    const { next } = gdStep(fdef, x, gdLr);
    return { from: x, to: next };
  }, [gdPath, fdef, gdLr]);
  const newtonPreview = { from: newtonInfo.x, to: newtonInfo.xNew };

  // Colors
  const COL_GD = "#c8311c";
  const COL_NEWT = "#1f4e3d";
  const COL_MODEL = "#3a7ca5";

  // ---------- Precomputed pixel geometry for SVG layer ----------
  const [xmin, xmax, ymin, ymax] = fdef.box;
  const showXAxis = ymin <= 0 && ymax >= 0;
  const showYAxis = xmin <= 0 && xmax >= 0;
  const [oxPx, oyPx] = map.toPx(0, 0);
  const optPx = fdef.opt ? map.toPx(fdef.opt[0], fdef.opt[1]) : null;
  const startPx = map.toPx(start[0], start[1]);
  const ell = newtonInfo.ellipse;
  const [ellCx, ellCy] = map.toPx(ell.center[0], ell.center[1]);
  const ellRx = ell.a * map.pxPerX;
  const ellRy = ell.b * map.pxPerY;
  const ellAngDeg = (-ell.angle * 180) / Math.PI; // SVG y is flipped → negate
  const gdArrow = [
    map.toPx(gdPreview.from[0], gdPreview.from[1]),
    map.toPx(gdPreview.to[0], gdPreview.to[1]),
  ];
  const nArrow = [
    map.toPx(newtonPreview.from[0], newtonPreview.from[1]),
    map.toPx(newtonPreview.to[0], newtonPreview.to[1]),
  ];

  // ============================================================
  // Render
  // ============================================================
  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Newton's Method vs Gradient Descent
      </h1>
      <p style={{ color: "#666", marginBottom: 28 }}>
        Curvature info — the Hessian — turns a generic descent direction into a
        step adapted to the local quadratic model. Watch Newton land at the
        model's minimum (the center of the blue ellipse) while GD only knows
        slope.
      </p>

      <section style={section}>
        <h2 style={h2}>Side-by-side trajectories on a 2D test function</h2>
        <p style={p}>
          Both methods start from the same <code style={code}>x⁰</code>. GD uses{" "}
          <code style={code}>x ← x − α ∇f</code>; Newton uses{" "}
          <code style={code}>x ← x − α<sub>n</sub> H⁻¹ ∇f</code>. The blue
          ellipse at the current Newton iterate is the level set of the local
          quadratic model <code style={code}>m(d)</code> that passes through{" "}
          <code style={code}>d = 0</code> — its <em>center</em> is where the
          full Newton step lands.
        </p>

        <div
          style={{
            display: "flex",
            gap: 24,
            alignItems: "flex-start",
            flexWrap: "wrap",
          }}
        >
          {/* ============ SVG plot ============ */}
          <svg
            width={PLOT_W}
            height={PLOT_H}
            style={{ background: "#fafafa", borderRadius: 6 }}
          >
            <rect width={PLOT_W} height={PLOT_H} fill="#fafafa" stroke="#d4d4d4" />

            {/* Axes through origin (if visible) */}
            {showYAxis && (
              <line x1={oxPx} y1={0} x2={oxPx} y2={PLOT_H} stroke="#d8d8d8" strokeDasharray="3 3" />
            )}
            {showXAxis && (
              <line x1={0} y1={oyPx} x2={PLOT_W} y2={oyPx} stroke="#d8d8d8" strokeDasharray="3 3" />
            )}

            {/* Contours */}
            {contourSegs.map((segs, li) =>
              segs.map((seg, k) => {
                const [x1p, y1p] = map.toPx(seg[0][0], seg[0][1]);
                const [x2p, y2p] = map.toPx(seg[1][0], seg[1][1]);
                return (
                  <line
                    key={`c-${li}-${k}`}
                    x1={x1p}
                    y1={y1p}
                    x2={x2p}
                    y2={y2p}
                    stroke="#888"
                    strokeOpacity={0.45}
                    strokeWidth={li % 3 === 0 ? 1.0 : 0.6}
                  />
                );
              })
            )}

            {/* Optimum marker */}
            {optPx && (
              <g>
                <circle cx={optPx[0]} cy={optPx[1]} r={7} fill="none" stroke="#111" strokeWidth={1.5} />
                <line x1={optPx[0] - 5} y1={optPx[1]} x2={optPx[0] + 5} y2={optPx[1]} stroke="#111" />
                <line x1={optPx[0]} y1={optPx[1] - 5} x2={optPx[0]} y2={optPx[1] + 5} stroke="#111" />
              </g>
            )}

            {/* Local quadratic-model ellipse (at current Newton iterate).
                SVG y is flipped vs data y → negate eigenvector angle. */}
            {showModelEllipse && ell.positive && (
              <g>
                <ellipse
                  cx={ellCx}
                  cy={ellCy}
                  rx={ellRx}
                  ry={ellRy}
                  fill={COL_MODEL}
                  fillOpacity={0.1}
                  stroke={COL_MODEL}
                  strokeWidth={1.5}
                  strokeDasharray="5 4"
                  transform={`rotate(${ellAngDeg.toFixed(3)} ${ellCx} ${ellCy})`}
                />
                {/* Center = where full Newton step (α_n=1) would land */}
                <circle
                  cx={ellCx}
                  cy={ellCy}
                  r={3.5}
                  fill={COL_MODEL}
                  stroke="#fff"
                  strokeWidth={1.2}
                />
              </g>
            )}

            {/* Trajectories (polyline + iterate dots) */}
            <Trajectory path={gdPath} map={map} color={COL_GD} keyPrefix="gd" />
            <Trajectory path={newtonPath} map={map} color={COL_NEWT} keyPrefix="nw" />

            {/* Step-preview arrows from each method's current iterate */}
            <defs>
              <marker id="arrowGD" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={COL_GD} />
              </marker>
              <marker id="arrowN" viewBox="0 0 10 10" refX="8" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill={COL_NEWT} />
              </marker>
            </defs>
            <line
              x1={gdArrow[0][0]} y1={gdArrow[0][1]}
              x2={gdArrow[1][0]} y2={gdArrow[1][1]}
              stroke={COL_GD} strokeWidth={1.8} strokeDasharray="4 3"
              markerEnd="url(#arrowGD)" opacity={0.9}
            />
            <line
              x1={nArrow[0][0]} y1={nArrow[0][1]}
              x2={nArrow[1][0]} y2={nArrow[1][1]}
              stroke={COL_NEWT} strokeWidth={1.8} strokeDasharray="4 3"
              markerEnd="url(#arrowN)" opacity={0.9}
            />

            {/* Start marker */}
            <circle
              cx={startPx[0]} cy={startPx[1]} r={5}
              fill="#fff" stroke="#111" strokeWidth={1.5}
            />

            {/* Inset legend */}
            <g transform={`translate(${PLOT_W - 168}, 12)`}>
              <rect
                x={0}
                y={0}
                width={156}
                height={78}
                fill="rgba(255,255,255,0.85)"
                stroke="#ddd"
                rx={4}
              />
              <line x1={10} y1={18} x2={30} y2={18} stroke={COL_GD} strokeWidth={2.4} />
              <text x={36} y={22} fontSize={12} fontFamily="monospace" fill="#222">
                Gradient Descent
              </text>
              <line x1={10} y1={36} x2={30} y2={36} stroke={COL_NEWT} strokeWidth={2.4} />
              <text x={36} y={40} fontSize={12} fontFamily="monospace" fill="#222">
                Newton
              </text>
              <line
                x1={10}
                y1={54}
                x2={30}
                y2={54}
                stroke={COL_MODEL}
                strokeWidth={1.6}
                strokeDasharray="4 3"
              />
              <text x={36} y={58} fontSize={12} fontFamily="monospace" fill="#222">
                Quadratic model
              </text>
            </g>
          </svg>

          {/* ============ Controls column ============ */}
          <div style={{ minWidth: 300, flex: 1 }}>
            <div style={controlGroup}>
              <label style={label}>Test function</label>
              <select
                value={fnKey}
                onChange={(e) => setFnKey(e.target.value)}
                style={select}
              >
                {Object.entries(FUNCTIONS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.name}
                  </option>
                ))}
              </select>
              <div style={formula}>{fdef.formula}</div>
              <div style={note}>{fdef.note}</div>
            </div>

            <Slider
              label={<>GD step size α: <b>{gdLr.toFixed(5)}</b></>}
              min={0.0001}
              max={Math.max(0.5, fdef.gdLrDefault * 4)}
              step={Math.max(0.00005, fdef.gdLrDefault / 200)}
              value={gdLr}
              onChange={setGdLr}
            />
            <Slider
              label={<>Newton damping α<sub>n</sub>: <b>{newtonAlpha.toFixed(2)}</b></>}
              min={0.05}
              max={1.0}
              step={0.01}
              value={newtonAlpha}
              onChange={setNewtonAlpha}
            />
            <div style={miniNote}>
              α<sub>n</sub> = 1 is the pure Newton step; smaller values damp it
              (useful when far from the optimum).
            </div>

            <div style={{ ...controlGroup, display: "flex", gap: 12 }}>
              <div style={{ flex: 1 }}>
                <Slider
                  label={<>start x⁰ₓ: <b>{start[0].toFixed(2)}</b></>}
                  min={fdef.box[0]} max={fdef.box[1]}
                  step={(fdef.box[1] - fdef.box[0]) / 200}
                  value={start[0]}
                  onChange={(v) => setStart([v, start[1]])}
                />
              </div>
              <div style={{ flex: 1 }}>
                <Slider
                  label={<>start x⁰ᵧ: <b>{start[1].toFixed(2)}</b></>}
                  min={fdef.box[2]} max={fdef.box[3]}
                  step={(fdef.box[3] - fdef.box[2]) / 200}
                  value={start[1]}
                  onChange={(v) => setStart([start[0], v])}
                />
              </div>
            </div>

            <div style={{ ...controlGroup, display: "flex", gap: 14 }}>
              <label style={{ ...label, display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={showModelEllipse}
                  onChange={(e) => setShowModelEllipse(e.target.checked)}
                />
                show model ellipse
              </label>
              <label style={{ ...label, display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
                <input
                  type="checkbox"
                  checked={regularize}
                  onChange={(e) => setRegularize(e.target.checked)}
                />
                regularize H ≻ 0
              </label>
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
              <button onClick={stepOnce} style={btnPrimary}>
                <StepForward size={16} /> Step
              </button>
              <button onClick={() => setRunning((r) => !r)} style={btn}>
                {running ? <Pause size={16} /> : <Play size={16} />}
                {running ? "Pause" : "Run"}
              </button>
              <button onClick={reset} style={btn}>
                <RotateCcw size={16} /> Reset
              </button>
            </div>

            {/* ---------- Stats ---------- */}
            <div style={statBox}>
              <div style={statHeader}>At current Newton iterate</div>
              <Stat label="x" value={`(${newtonInfo.x[0].toFixed(3)}, ${newtonInfo.x[1].toFixed(3)})`} />
              <Stat label="f(x)" value={fmtNum(newtonInfo.f)} />
              <Stat label="‖∇f‖" value={fmtNum(newtonInfo.gnorm)} />
              <Stat label="eigvals(H)" value={`${fmtNum(newtonInfo.lam1)}, ${fmtNum(newtonInfo.lam2)}`} />
              <Stat label="κ(H)" value={isFinite(newtonInfo.cond) ? fmtNum(newtonInfo.cond) : "∞"} />
              <Stat label="H ≻ 0?" value={newtonInfo.ellipse.positive ? "yes" : "no (indef.)"} />
            </div>

            <div style={statBox}>
              <div style={statHeader}>Iteration counts</div>
              <Stat
                label="GD steps"
                value={`${gdPath.length - 1}  →  f = ${fmtNum(fdef.f(gdPath.at(-1)[0], gdPath.at(-1)[1]))}`}
              />
              <Stat
                label="Newton steps"
                value={`${newtonPath.length - 1}  →  f = ${fmtNum(fdef.f(newtonPath.at(-1)[0], newtonPath.at(-1)[1]))}`}
              />
            </div>

            <p style={{ ...p, marginTop: 14, fontSize: 13 }}>
              <b>Try this.</b> On the ill-conditioned quadratic, set α<sub>n</sub> = 1
              and click <b>Step</b> once — Newton lands at the optimum in a single
              iteration, while GD has to bounce between the steep walls. On
              Rosenbrock, watch the model ellipse stretch along the curved
              valley: each Newton step targets the bottom of the local parabola.
            </p>
          </div>
        </div>
      </section>

      <section style={section}>
        <h2 style={h2}>What the picture is telling you</h2>
        <ul style={bullets}>
          <li>
            <b>GD only knows slope.</b> It moves opposite the gradient. On
            ill-conditioned bowls the gradient points across the valley far more
            than along it, so GD ricochets.
          </li>
          <li>
            <b>Newton uses curvature.</b> <code style={code}>−H⁻¹∇f</code> is
            the exact minimizer of the local quadratic model{" "}
            <code style={code}>m(d) = f(x) + gᵀd + ½ dᵀ H d</code>. The blue
            ellipse is a level set of <code style={code}>m</code>; its{" "}
            <em>center</em> is exactly where the full Newton step (α
            <sub>n</sub>=1) lands.
          </li>
          <li>
            <b>Quadratic = exact.</b> When <code style={code}>f</code> itself is
            quadratic with constant Hessian (the first two test functions),
            Newton converges in one step from any start.
          </li>
          <li>
            <b>Indefinite Hessian.</b> Far from a minimum the Hessian can have
            negative eigenvalues — the "model ellipse" becomes a hyperbola. We
            regularize H ← H + τI (toggle above) so the demo keeps producing a
            descent direction.
          </li>
          <li>
            <b>Cost.</b> Each Newton step here is O(1) extra work (2×2 inverse).
            In high dimensions, the Hessian solve is the dominant cost — that's
            why quasi-Newton (BFGS, L-BFGS) and trust-region methods exist.
          </li>
        </ul>
      </section>
    </div>
  );
}

// ============================================================
// Small components / helpers
// ============================================================
function Trajectory({ path, map, color, keyPrefix }) {
  const pts = path.map((p) => map.toPx(p[0], p[1]));
  return (
    <g>
      <polyline
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
        points={pts.map((q) => `${q[0]},${q[1]}`).join(" ")}
      />
      {pts.map((q, i) => {
        const last = i === pts.length - 1;
        return (
          <circle
            key={`${keyPrefix}-${i}`}
            cx={q[0]}
            cy={q[1]}
            r={last ? 4.5 : 2.2}
            fill={color}
            stroke="#fff"
            strokeWidth={last ? 1.5 : 0.6}
          />
        );
      })}
    </g>
  );
}

function Slider({ label: lbl, min, max, step, value, onChange }) {
  return (
    <div style={controlGroup}>
      <label style={label}>{lbl}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 12 }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function fmtNum(v) {
  if (!isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-3 || a >= 1e4) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : 3);
}

// ============================================================
// Styles (mirrors svm_demo.jsx aesthetic)
// ============================================================
const section = {
  padding: "28px 26px", marginBottom: 22, background: "#fff",
  borderRadius: 12, border: "1px solid #e7e7e7",
  boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
};
const h2 = { fontSize: 21, fontWeight: 800, marginBottom: 8 };
const p = { color: "#444", lineHeight: 1.55, marginBottom: 14, maxWidth: 720 };
const code = {
  background: "#f0eee9", padding: "1px 6px", borderRadius: 4,
  fontFamily: "monospace", fontSize: 13,
};
const controlGroup = { marginTop: 10 };
const label = {
  display: "block", fontSize: 13, color: "#444",
  marginBottom: 4, fontFamily: "monospace",
};
const select = {
  width: "100%", padding: "6px 8px", borderRadius: 6,
  border: "1px solid #ccc", background: "#fff",
};
const btn = {
  display: "inline-flex", alignItems: "center", gap: 6,
  padding: "7px 12px", borderRadius: 6, border: "1px solid #ccc",
  background: "#f7f7f7", cursor: "pointer", fontWeight: 500,
};
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
const statBox = {
  marginTop: 12, padding: "8px 12px", background: "#fafafa",
  border: "1px solid #eee", borderRadius: 6,
};
const statHeader = {
  fontFamily: "monospace", fontSize: 11, letterSpacing: "0.08em",
  color: "#888", textTransform: "uppercase", marginBottom: 4,
};
const formula = {
  marginTop: 8, padding: "8px 10px", background: "#faf8f3",
  fontFamily: "monospace", fontSize: 12, color: "#222",
  border: "1px solid #ece9e0", borderRadius: 4,
};
const note = { marginTop: 6, fontSize: 12, color: "#666", lineHeight: 1.45 };
const miniNote = { marginTop: 4, fontSize: 11, color: "#888", lineHeight: 1.4 };
const bullets = {
  margin: 0, paddingLeft: 22, color: "#444",
  lineHeight: 1.6, fontSize: 14,
};
