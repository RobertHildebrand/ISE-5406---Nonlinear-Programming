import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward, Terminal } from "lucide-react";

/* ============================================================
   FRANK-WOLFE / FULLY-CORRECTIVE FRANK-WOLFE
   ISE 5406

   Demonstrates the conditional-gradient algorithm with active
   vertex tracking. Two variants:

     • Vanilla FW
         x_{k+1} = (1 - γ_k) x_k + γ_k s_k
         γ_k = 2 / (k+2)
         where s_k = argmin_{v ∈ V} ⟨∇f(x_k), v⟩

     • Fully-corrective FW (FCFW)
         S_{k+1} = S_k ∪ {s_k}
         x_{k+1} = argmin_{x ∈ conv(S_{k+1})} f(x)

   FCFW retains every vertex it has ever picked and re-optimizes
   over the convex hull each step. Same per-iteration linear-
   minimization oracle, but the inner QP closes the gap much
   faster — typical case: linear vs sublinear convergence.

   Visualization:
     • Pentagon feasible region (5 vertices)
     • Quadratic objective f(x) = ‖x − target‖²
     • Drag the target inside or outside the pentagon
     • Step through iterations; watch the active vertex set
       grow (Vanilla FW: monotone forgetful drift; FCFW:
       set grows and projection moves to the hull-closest
       point on each step).
   ============================================================ */

// ============================================================
// Geometry helpers
// ============================================================
const VERTICES = [
  { x: 1.0, y: 0.0 },
  { x: 0.309, y: 0.951 },
  { x: -0.809, y: 0.588 },
  { x: -0.809, y: -0.588 },
  { x: 0.309, y: -0.951 },
];

const dot = (a, b) => a.x * b.x + a.y * b.y;
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y });
const scale = (a, c) => ({ x: a.x * c, y: a.y * c });
const sqDist = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;
const norm = (a) => Math.sqrt(a.x ** 2 + a.y ** 2);

// Project p onto segment [a, b]
function projOntoSegment(p, a, b) {
  const ab = sub(b, a);
  const denom = dot(ab, ab);
  if (denom < 1e-12) return { ...a };
  let t = dot(sub(p, a), ab) / denom;
  t = Math.max(0, Math.min(1, t));
  return add(a, scale(ab, t));
}

// Is point p inside the convex hull of vs (a 2D polygon)?
// Uses cross-product sign test (vs assumed in CCW order or convex).
function isInsideConvHull(p, vs) {
  if (vs.length < 3) return false;
  // Compute centroid + check p is on the same side of every edge as the centroid.
  const cx = vs.reduce((s, v) => s + v.x, 0) / vs.length;
  const cy = vs.reduce((s, v) => s + v.y, 0) / vs.length;
  // Sort vertices by angle around centroid (so edges are in cyclic order)
  const sorted = [...vs].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i];
    const b = sorted[(i + 1) % sorted.length];
    const cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
    if (cross < -1e-9) return false;
  }
  return true;
}

// Project target onto convex hull of a set of vertices.
// For 2D with up to 5 vertices, we just check interior + every edge of
// the hull's boundary, and every vertex.
function projectOntoConvHull(target, vertices) {
  if (vertices.length === 0) return { x: 0, y: 0 };
  if (vertices.length === 1) return { ...vertices[0] };
  if (vertices.length === 2) return projOntoSegment(target, vertices[0], vertices[1]);
  if (isInsideConvHull(target, vertices)) return { ...target };

  // Sort by angle around centroid → hull order
  const cx = vertices.reduce((s, v) => s + v.x, 0) / vertices.length;
  const cy = vertices.reduce((s, v) => s + v.y, 0) / vertices.length;
  const sorted = [...vertices].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );

  let bestPt = sorted[0];
  let bestD = sqDist(target, sorted[0]);
  for (let i = 0; i < sorted.length; i++) {
    const a = sorted[i];
    const b = sorted[(i + 1) % sorted.length];
    const p = projOntoSegment(target, a, b);
    const d = sqDist(target, p);
    if (d < bestD) {
      bestD = d;
      bestPt = p;
    }
  }
  return bestPt;
}

// ============================================================
// FW iterations — precompute history
// ============================================================
function runFW(target, algo, maxIter) {
  const history = [];
  const start = VERTICES[0];
  let x = { x: start.x, y: start.y };
  let S = new Set([0]);
  for (let k = 0; k < maxIter; k++) {
    const grad = { x: 2 * (x.x - target.x), y: 2 * (x.y - target.y) };
    let minVal = Infinity, sIdx = 0;
    VERTICES.forEach((v, i) => {
      const d = grad.x * v.x + grad.y * v.y;
      if (d < minVal) {
        minVal = d;
        sIdx = i;
      }
    });
    const s = VERTICES[sIdx];
    // FW gap: <∇f(x), x − s>
    const gap = grad.x * (x.x - s.x) + grad.y * (x.y - s.y);
    const f = sqDist(x, target);
    history.push({
      k,
      x: { ...x },
      grad: { ...grad },
      sIdx,
      S: [...S],
      f,
      gap,
    });
    let nextX;
    if (algo === "vanilla") {
      const gamma = 2 / (k + 2);
      nextX = {
        x: (1 - gamma) * x.x + gamma * s.x,
        y: (1 - gamma) * x.y + gamma * s.y,
      };
      // Vanilla FW doesn't track an active set in the algorithm sense, but for
      // visualization we keep all vertices visited.
      S.add(sIdx);
    } else {
      // FCFW: project target onto conv(S ∪ {sIdx})
      S.add(sIdx);
      nextX = projectOntoConvHull(
        target,
        [...S].map((i) => VERTICES[i])
      );
    }
    x = nextX;
  }
  // Final state (post-step)
  const grad = { x: 2 * (x.x - target.x), y: 2 * (x.y - target.y) };
  history.push({
    k: maxIter,
    x: { ...x },
    grad,
    sIdx: null,
    S: [...S],
    f: sqDist(x, target),
    gap: 0,
    final: true,
  });
  return history;
}

// ============================================================
// Code lines (for the stepper)
// ============================================================
const CODE_LINES = [
  null,
  "import numpy as np",
  "from scipy.optimize import minimize",
  "",
  "# Pentagon vertices (rows of V are the extreme points)",
  "V = np.array([[1.000, 0.000],",
  "              [0.309, 0.951],",
  "              [-0.809, 0.588],",
  "              [-0.809, -0.588],",
  "              [0.309, -0.951]])",
  "target = np.array([0.6, 0.7])",
  "",
  "def f(x):    return ((x - target)**2).sum()",
  "def grad(x): return 2 * (x - target)",
  "",
  "x = V[0].copy()           # any extreme point is feasible",
  "S = {0}                   # ACTIVE VERTEX SET",
  "",
  "for k in range(MAX_ITER):",
  "    g = grad(x)",
  "    # Linear minimization oracle:",
  "    #   s = argmin_{v in V} <g, v>",
  "    s_idx = np.argmin(V @ g)",
  "    s = V[s_idx]",
  "",
  "    # Frank-Wolfe gap (a certificate of suboptimality)",
  "    gap = g @ (x - s)",
  "    if gap < TOL: break",
  "",
  "    # ── Vanilla FW ───────────────────────────────",
  "    gamma = 2.0 / (k + 2)",
  "    x_van = (1 - gamma) * x + gamma * s",
  "",
  "    # ── Fully-corrective FW (FCFW) ───────────────",
  "    S.add(s_idx)",
  "    # Inner QP: min ||sum_{i in S} λ_i V_i - target||²",
  "    #           s.t. λ ≥ 0,  sum λ = 1",
  "    res = minimize(",
  "        lambda l: ((l @ V[list(S)] - target)**2).sum(),",
  "        x0 = np.full(len(S), 1/len(S)),",
  "        constraints = [{'type':'eq', 'fun': lambda l: l.sum() - 1}],",
  "        bounds = [(0, 1)] * len(S),",
  "    )",
  "    x_fc = res.x @ V[list(S)]",
  "",
  "    # Pick the algorithm",
  "    x = x_fc if FCFW else x_van",
  "",
  "print('x* =', x, ' f* =', f(x))",
];

// Map a logical step to an "active code line"
function lineForStep(k) {
  // Cycle through the loop body lines as iterations proceed
  const base = 18; // start of 'for k in range...'
  const inner = [19, 22, 23, 26, 27, 30, 31, 35, 36, 41, 44]; // line subset
  return inner[k % inner.length];
}

// ============================================================
// Main component
// ============================================================
export default function FrankWolfeDemo() {
  const [target, setTarget] = useState({ x: 0.6, y: 0.7 });
  const [algo, setAlgo] = useState("fcfw");
  const [k, setK] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);
  const MAX_ITER = 15;

  const history = useMemo(() => runFW(target, algo, MAX_ITER), [target, algo]);

  useEffect(() => {
    setK(0);
  }, [target, algo]);

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

  const step = useCallback(
    () => setK((kk) => Math.min(history.length - 1, kk + 1)),
    [history.length]
  );
  const reset = useCallback(() => {
    setK(0);
    setRunning(false);
  }, []);

  const state = history[k];
  const fStar = sqDist(target, projectOntoConvHull(target, VERTICES));

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Frank-Wolfe with Active Vertex Tracking
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        The conditional-gradient algorithm, in slow motion. Each iteration
        calls a linear-minimization oracle (LMO) — find the polytope vertex
        that makes the steepest progress against the current gradient — and
        either takes a damped convex combination toward it (vanilla FW) or
        re-optimizes over the convex hull of every vertex picked so far
        (fully-corrective FW). Drag the target to see how the algorithm
        adapts.
      </p>

      <div style={controlBar}>
        <label style={{ marginRight: 12 }}>
          <input
            type="radio"
            name="algo"
            checked={algo === "vanilla"}
            onChange={() => setAlgo("vanilla")}
          />
          &nbsp;Vanilla FW (γ = 2/(k+2))
        </label>
        <label style={{ marginRight: 12 }}>
          <input
            type="radio"
            name="algo"
            checked={algo === "fcfw"}
            onChange={() => setAlgo("fcfw")}
          />
          &nbsp;Fully-corrective FW (re-optimize over conv(S))
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
          <PolygonViz
            target={target}
            setTarget={setTarget}
            state={state}
            history={history}
            kIdx={k}
          />

          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            <button
              onClick={step}
              disabled={k >= history.length - 1}
              style={btnPrimary}
            >
              <StepForward size={16} /> Step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={k >= history.length - 1}
              style={btn}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}{" "}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={reset} style={btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={{ marginTop: 12 }}>
            <label style={smallLabel}>
              speed (ms/step): <b>{speed}</b>
            </label>
            <input
              type="range"
              min={150}
              max={1500}
              step={50}
              value={speed}
              onChange={(e) => setSpeed(+e.target.value)}
              style={{ width: "100%" }}
            />
          </div>

          <div
            style={{
              marginTop: 8,
              fontSize: 11,
              color: "#888",
              fontFamily: "monospace",
            }}
          >
            iter {k} / {history.length - 1}
          </div>

          <ConvergenceChart history={history} kIdx={k} fStar={fStar} />
        </div>

        <div>
          <CodePanel codeLines={CODE_LINES} highlightedLine={state.final ? 47 : lineForStep(k)} />
          <StatePanel state={state} algo={algo} fStar={fStar} target={target} />
        </div>
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Polygon SVG visualization
// ============================================================
function PolygonViz({ target, setTarget, state, history, kIdx }) {
  const W = 480, H = 480;
  const cx = W / 2, cy = H / 2;
  const scaleP = 180; // world-to-pixel scale
  const wp = (p) => ({ x: cx + p.x * scaleP, y: cy - p.y * scaleP });
  const fromPx = (p) => ({ x: (p.x - cx) / scaleP, y: -(p.y - cy) / scaleP });

  const onMouseDown = (e) => {
    const handleMove = (mv) => {
      const rect = e.target.ownerSVGElement.getBoundingClientRect();
      const px = { x: mv.clientX - rect.left, y: mv.clientY - rect.top };
      setTarget(fromPx(px));
    };
    const handleUp = () => {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
    };
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
  };

  // Draw level sets of f(x) = ||x - target||²
  const levelRings = [];
  for (let r = 0.3; r <= 2.5; r += 0.3) {
    levelRings.push(r);
  }

  const polyPts = VERTICES.map(wp).map((p) => `${p.x},${p.y}`).join(" ");
  const activeVertexPts = state.S.map((i) => VERTICES[i]);
  const activeHull = activeVertexPts.length >= 3
    ? (() => {
        const c = activeVertexPts.reduce((s, v) => add(s, v), { x: 0, y: 0 });
        const center = scale(c, 1 / activeVertexPts.length);
        const sorted = [...activeVertexPts].sort(
          (a, b) =>
            Math.atan2(a.y - center.y, a.x - center.x) -
            Math.atan2(b.y - center.y, b.x - center.x)
        );
        return sorted;
      })()
    : activeVertexPts;
  const hullPts = activeHull.map(wp).map((p) => `${p.x},${p.y}`).join(" ");

  // Trajectory polyline up to k
  const traj = history.slice(0, kIdx + 1).map((h) => wp(h.x));
  const trajStr = traj.map((p) => `${p.x},${p.y}`).join(" ");

  const tgtPx = wp(target);
  const xPx = wp(state.x);
  const sPx = state.sIdx != null ? wp(VERTICES[state.sIdx]) : null;

  // Gradient arrow
  const gradLen = norm(state.grad);
  const gradEnd = gradLen > 1e-6
    ? add(state.x, scale(state.grad, -0.4 / Math.max(1, gradLen / 2)))
    : state.x;
  const gradEndPx = wp(gradEnd);

  return (
    <div style={{ background: "#fafafa", border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
      <svg width={W} height={H} style={{ display: "block", cursor: "crosshair" }}>
        {/* Level sets */}
        {levelRings.map((r, i) => (
          <circle
            key={i}
            cx={tgtPx.x}
            cy={tgtPx.y}
            r={r * scaleP}
            stroke="#e5d8b8"
            strokeWidth={1}
            fill="none"
            strokeDasharray="2,3"
          />
        ))}

        {/* Pentagon outline */}
        <polygon
          points={polyPts}
          fill="rgba(31, 78, 61, 0.05)"
          stroke="#1f4e3d"
          strokeWidth={1.5}
        />

        {/* Active hull (semi-transparent) */}
        {state.S.length >= 2 && (
          <polygon
            points={hullPts}
            fill="rgba(245, 165, 36, 0.18)"
            stroke="#f5a524"
            strokeWidth={2}
            strokeDasharray={state.S.length === 2 ? "4,3" : "0"}
          />
        )}

        {/* All polytope vertices */}
        {VERTICES.map((v, i) => {
          const p = wp(v);
          const inS = state.S.includes(i);
          const isS = state.sIdx === i;
          return (
            <g key={i}>
              <circle
                cx={p.x}
                cy={p.y}
                r={isS ? 9 : inS ? 7 : 5}
                fill={isS ? "#c8311c" : inS ? "#f5a524" : "#1f4e3d"}
                stroke="#fff"
                strokeWidth={2}
              />
              <text x={p.x + 10} y={p.y - 8} fontSize={11} fontFamily="monospace" fill="#444">
                v{i}
              </text>
            </g>
          );
        })}

        {/* Trajectory */}
        {traj.length > 1 && (
          <polyline
            points={trajStr}
            fill="none"
            stroke="#0b3da0"
            strokeWidth={2}
            strokeDasharray="3,2"
          />
        )}

        {/* Gradient arrow from x */}
        {gradLen > 1e-3 && (
          <>
            <line
              x1={xPx.x}
              y1={xPx.y}
              x2={gradEndPx.x}
              y2={gradEndPx.y}
              stroke="#7a3da0"
              strokeWidth={2}
            />
            <polygon
              points={`${gradEndPx.x},${gradEndPx.y} ${gradEndPx.x - 5},${gradEndPx.y - 5} ${gradEndPx.x - 5},${gradEndPx.y + 5}`}
              fill="#7a3da0"
            />
          </>
        )}

        {/* Selected vertex highlight */}
        {sPx && (
          <line
            x1={xPx.x}
            y1={xPx.y}
            x2={sPx.x}
            y2={sPx.y}
            stroke="#c8311c"
            strokeWidth={1.5}
            strokeDasharray="6,3"
          />
        )}

        {/* Current iterate */}
        <circle cx={xPx.x} cy={xPx.y} r={6} fill="#0b3da0" stroke="#fff" strokeWidth={2} />

        {/* Target (draggable) */}
        <circle
          cx={tgtPx.x}
          cy={tgtPx.y}
          r={9}
          fill="#fff"
          stroke="#c8311c"
          strokeWidth={3}
          style={{ cursor: "grab" }}
          onMouseDown={onMouseDown}
        />
        <text x={tgtPx.x + 12} y={tgtPx.y + 4} fontSize={12} fontFamily="monospace" fill="#c8311c" fontWeight={700}>
          target (drag)
        </text>

        {/* Legend */}
        <g transform={`translate(10, ${H - 80})`}>
          <rect x={0} y={0} width={150} height={70} fill="rgba(255,255,255,0.9)" stroke="#ccc" />
          <circle cx={12} cy={14} r={5} fill="#0b3da0" /><text x={22} y={18} fontSize={11}>iterate xₖ</text>
          <circle cx={12} cy={30} r={5} fill="#f5a524" /><text x={22} y={34} fontSize={11}>active set S</text>
          <circle cx={12} cy={46} r={5} fill="#c8311c" /><text x={22} y={50} fontSize={11}>sₖ (LMO)</text>
          <line x1={5} y1={62} x2={20} y2={62} stroke="#7a3da0" strokeWidth={2} /><text x={22} y={66} fontSize={11}>−∇f(xₖ)</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Convergence chart
// ============================================================
function ConvergenceChart({ history, kIdx, fStar }) {
  const W = 480, H = 140;
  const padL = 50, padR = 8, padT = 14, padB = 26;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const N = history.length;
  const fs = history.map((h) => Math.max(1e-12, h.f - fStar));
  const logFs = fs.map((v) => Math.log10(v));
  const yMin = Math.min(...logFs.filter((v) => isFinite(v)));
  const yMax = Math.max(...logFs.filter((v) => isFinite(v)), yMin + 1);
  const xs = (i) => padL + (i / Math.max(1, N - 1)) * chartW;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * chartH;
  const points = logFs.map((v, i) => `${xs(i)},${ys(v)}`).join(" ");
  return (
    <div
      style={{
        marginTop: 16,
        background: "#fafafa",
        border: "1px solid #ddd",
        borderRadius: 8,
        padding: 8,
      }}
    >
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 11,
          color: "#888",
          marginBottom: 4,
        }}
      >
        log₁₀(f(xₖ) − f*)
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="#bbb" />
        <polyline points={points} fill="none" stroke="#0b3da0" strokeWidth={2} />
        {logFs.map((v, i) => (
          <circle
            key={i}
            cx={xs(i)}
            cy={ys(v)}
            r={i === kIdx ? 4 : 2}
            fill={i === kIdx ? "#c8311c" : "#0b3da0"}
          />
        ))}
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMax.toFixed(1)}
        </text>
        <text x={padL - 4} y={padT + chartH} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMin.toFixed(1)}
        </text>
        <text x={padL + chartW} y={padT + chartH + 14} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          k = {N - 1}
        </text>
      </svg>
    </div>
  );
}

// ============================================================
// State panel
// ============================================================
function StatePanel({ state, algo, fStar, target }) {
  return (
    <div style={statePanelOuter}>
      <div style={sectionTitle}>State at iteration k = {state.k}</div>
      <KV k="x" v={`(${state.x.x.toFixed(4)}, ${state.x.y.toFixed(4)})`} />
      <KV k="f(x)" v={state.f.toFixed(6)} highlight />
      <KV k="f* (proj of target on Ω)" v={fStar.toFixed(6)} />
      <KV
        k="suboptimality"
        v={(state.f - fStar).toExponential(3)}
        highlight
      />
      <KV k="∇f(x)" v={`(${state.grad.x.toFixed(3)}, ${state.grad.y.toFixed(3)})`} />
      {state.sIdx != null && (
        <KV k="sₖ (LMO)" v={`v${state.sIdx} = (${VERTICES[state.sIdx].x.toFixed(2)}, ${VERTICES[state.sIdx].y.toFixed(2)})`} />
      )}
      <KV k="FW gap ⟨∇f, x − s⟩" v={state.gap.toExponential(3)} />
      <KV k="active set S" v={`{${state.S.map((i) => `v${i}`).join(", ")}}`} />
      <KV k="|S|" v={state.S.length} />
      <KV k="algorithm" v={algo === "vanilla" ? "Vanilla FW" : "Fully-corrective FW"} />
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
// Code panel
// ============================================================
function CodePanel({ codeLines, highlightedLine }) {
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
        marginBottom: 12,
      }}
    >
      {codeLines.map((line, i) => {
        if (i === 0) return null;
        const active = i === highlightedLine;
        const isBlank = line === "";
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              background: active ? "#3b3526" : "transparent",
              borderLeft: active ? "3px solid #f5a524" : "3px solid transparent",
              minHeight: lineHeight,
            }}
          >
            <span
              style={{
                width: 22,
                color: active ? "#f5a524" : "#7f7864",
                fontSize: 11,
                userSelect: "none",
              }}
            >
              {active ? "▶" : ""}
            </span>
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
    <div
      style={{
        marginTop: 28,
        padding: 16,
        background: "#fff8e1",
        borderRadius: 10,
        border: "1px solid #f5d68d",
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: 6 }}>
        <Terminal size={14} style={{ verticalAlign: "middle", marginRight: 6 }} />
        Notes for class
      </div>
      <ul style={{ margin: 0, paddingLeft: 22, lineHeight: 1.6, fontSize: 14, color: "#3d2f00" }}>
        <li>
          <b>The LMO is everything.</b> Frank-Wolfe needs only one oracle per
          iteration: <i>given a linear function, return the vertex that
          minimizes it.</i> No projection. No active-set logic. For simplex /
          ℓ₁-ball / nuclear-norm-ball / spectrahedron the LMO is closed-form.
        </li>
        <li>
          <b>Vanilla FW is sublinear.</b> f(x_k) − f* = O(1/k). Good enough
          for sparse signal recovery, weak enough that practitioners reach
          for variants.
        </li>
        <li>
          <b>FCFW closes the gap geometrically.</b> By keeping every chosen
          vertex in S and re-optimizing over conv(S), FCFW gets <i>linear</i>
          convergence on strongly convex objectives (assuming the LMO returns
          a vertex of the feasible polytope, not an arbitrary face). Cost: an
          inner QP each iteration.
        </li>
        <li>
          <b>Away-step FW (not shown).</b> Halfway between the two: at each
          step, choose between moving toward s_k or AWAY from the worst
          vertex in S. Also linearly convergent, with a smaller inner solve
          than FCFW.
        </li>
        <li>
          <b>FW gap is a free certificate.</b>{" "}
          <code style={inlineCode}>{"<∇f(x), x − s>"}</code> upper-bounds
          f(x) − f*. Use it as a stopping criterion — no need to track f* or
          run extra solves.
        </li>
        <li>
          <b>Why this matters in ML.</b> Frank-Wolfe is the algorithm behind
          coresets, conditional-gradient SVMs, OMP-style sparse coding, and
          convex optimization on probability simplexes. Anywhere you have a
          structured constraint set with a cheap LMO but expensive projection.
        </li>
      </ul>
    </div>
  );
}

// ============================================================
// Style atoms
// ============================================================
const controlBar = {
  marginBottom: 16,
  padding: "10px 14px",
  background: "#f6f4ee",
  border: "1px solid #ece8dd",
  borderRadius: 8,
  fontSize: 13,
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
const smallLabel = { display: "block", fontSize: 12, color: "#444", marginBottom: 4, fontFamily: "monospace" };
const sectionTitle = {
  fontFamily: "monospace",
  fontSize: 10,
  letterSpacing: "0.18em",
  color: "#888",
  marginBottom: 8,
  textTransform: "uppercase",
};
const statePanelOuter = {
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 8,
  padding: 14,
};
const inlineCode = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
