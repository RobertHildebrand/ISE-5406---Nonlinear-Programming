import React, { useState, useMemo, useCallback, useEffect, useRef } from "react";
import { Play, RotateCcw, Crosshair, FastForward } from "lucide-react";

/* ============================================================
   LINE SEARCH — INTERACTIVE DEMO
   For ISE 5406. Visualizes Armijo (sufficient decrease) and
   strong Wolfe (curvature) conditions used to pick a step size α
   along a descent direction d at the current iterate x.

     φ(α) = f(x + α · d)
     Armijo:        φ(α) ≤ φ(0) + c₁ · α · φ'(0)
     Strong Wolfe:  |φ'(α)| ≤ c₂ · |φ'(0)|

   ============================================================ */

// ---------------- Test functions ----------------
// Each function exposes f(x,y) and ∇f(x,y).
const FUNCTIONS = {
  quadratic: {
    name: "Quadratic  ½(x² + 5y²)",
    f: (x, y) => 0.5 * (x * x + 5 * y * y),
    grad: (x, y) => [x, 5 * y],
    domain: { lo: -3, hi: 3 },
    levels: [0.05, 0.2, 0.5, 1, 2, 4, 7, 11, 16, 22],
    defaultX: [-2.4, 1.6],
    defaultAlphaMax: 1.2,
  },
  rosenbrock: {
    name: "Rosenbrock  (1−x)² + 100(y−x²)²",
    f: (x, y) => {
      const a = 1 - x;
      const b = y - x * x;
      return a * a + 100 * b * b;
    },
    grad: (x, y) => {
      const a = 1 - x;
      const b = y - x * x;
      return [-2 * a - 400 * x * b, 200 * b];
    },
    domain: { lo: -2, hi: 2 },
    levels: [0.5, 2, 6, 15, 35, 80, 160, 320, 600, 1100],
    defaultX: [-1.2, 1.0],
    defaultAlphaMax: 0.005,
  },
  himmelblau: {
    name: "Himmelblau  (x²+y−11)² + (x+y²−7)²",
    f: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return a * a + b * b;
    },
    grad: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
    },
    domain: { lo: -5, hi: 5 },
    levels: [1, 4, 12, 30, 70, 150, 320, 650, 1200, 2200],
    defaultX: [-3.0, 2.5],
    defaultAlphaMax: 0.05,
  },
};

// ---------------- Plot geometry ----------------
const PLOT_SIZE = 380;

function makeMappers(domain) {
  const { lo, hi } = domain;
  return {
    xToPx: (x) => ((x - lo) / (hi - lo)) * PLOT_SIZE,
    yToPx: (y) => PLOT_SIZE - ((y - lo) / (hi - lo)) * PLOT_SIZE,
    pxToX: (px) => lo + (px / PLOT_SIZE) * (hi - lo),
    pxToY: (py) => lo + ((PLOT_SIZE - py) / PLOT_SIZE) * (hi - lo),
  };
}

// Marching-squares contour extraction for one level set.
function extractContour(F, N, level, step) {
  const segs = [];
  const interp = (a, b) => (level - a) / (b - a);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const v00 = F[i * (N + 1) + j];
      const v10 = F[(i + 1) * (N + 1) + j];
      const v01 = F[i * (N + 1) + (j + 1)];
      const v11 = F[(i + 1) * (N + 1) + (j + 1)];
      let idx = 0;
      if (v00 > level) idx |= 1;
      if (v10 > level) idx |= 2;
      if (v11 > level) idx |= 4;
      if (v01 > level) idx |= 8;
      if (idx === 0 || idx === 15) continue;
      const x0 = i * step, x1 = (i + 1) * step;
      const y0 = j * step, y1 = (j + 1) * step;
      const edges = [];
      if ((v00 > level) !== (v10 > level))
        edges.push([x0 + step * interp(v00, v10), y0]);
      if ((v10 > level) !== (v11 > level))
        edges.push([x1, y0 + step * interp(v10, v11)]);
      if ((v01 > level) !== (v11 > level))
        edges.push([x0 + step * interp(v01, v11), y1]);
      if ((v00 > level) !== (v01 > level))
        edges.push([x0, y0 + step * interp(v00, v01)]);
      if (edges.length >= 2) segs.push([edges[0], edges[1]]);
      if (edges.length === 4) segs.push([edges[2], edges[3]]);
    }
  }
  return segs;
}

// ============================================================
// Contour plot — f, current iterate x, descent direction d, marker x+αd
// ============================================================
function ContourPlot({ fnDef, x, alpha, d, onClickPick, pickMode }) {
  const map = useMemo(() => makeMappers(fnDef.domain), [fnDef]);
  const N = 80;

  // Sample f on a grid (in pixel coordinates).
  const F = useMemo(() => {
    const arr = new Float64Array((N + 1) * (N + 1));
    const step = PLOT_SIZE / N;
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        const xd = map.pxToX(i * step);
        const yd = map.pxToY(j * step);
        arr[i * (N + 1) + j] = fnDef.f(xd, yd);
      }
    }
    return arr;
  }, [fnDef, map]);

  const contours = useMemo(() => {
    const step = PLOT_SIZE / N;
    return fnDef.levels.map((lv) => ({
      level: lv,
      segs: extractContour(F, N, lv, step),
    }));
  }, [F, fnDef]);

  const xPx = map.xToPx(x[0]);
  const yPx = map.yToPx(x[1]);
  const tipX = x[0] + alpha * d[0];
  const tipY = x[1] + alpha * d[1];
  const tipXPx = map.xToPx(tipX);
  const tipYPx = map.yToPx(tipY);

  // Direction arrow extends a fixed visual length so direction is visible
  // even when α is tiny.
  const dNorm = Math.hypot(d[0], d[1]);
  const arrowLenPx = 80;
  const arrowEndX = xPx + (arrowLenPx * (d[0] / (dNorm || 1)));
  const arrowEndY = yPx - (arrowLenPx * (d[1] / (dNorm || 1)));

  const handleClick = (e) => {
    if (!pickMode || !onClickPick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    onClickPick([map.pxToX(px), map.pxToY(py)]);
  };

  return (
    <svg
      width={PLOT_SIZE}
      height={PLOT_SIZE}
      onClick={handleClick}
      style={{
        background: "#fafafa",
        borderRadius: 6,
        cursor: pickMode ? "crosshair" : "default",
        border: "1px solid #ddd",
      }}
    >
      <rect width={PLOT_SIZE} height={PLOT_SIZE} fill="#fafafa" />
      {/* axes */}
      <line
        x1={map.xToPx(0)}
        y1={0}
        x2={map.xToPx(0)}
        y2={PLOT_SIZE}
        stroke="#ddd"
        strokeDasharray="3 3"
      />
      <line
        x1={0}
        y1={map.yToPx(0)}
        x2={PLOT_SIZE}
        y2={map.yToPx(0)}
        stroke="#ddd"
        strokeDasharray="3 3"
      />
      {/* contour lines */}
      {contours.map((c, ci) =>
        c.segs.map((s, si) => (
          <line
            key={`c-${ci}-${si}`}
            x1={s[0][0]}
            y1={s[0][1]}
            x2={s[1][0]}
            y2={s[1][1]}
            stroke="#7aa9c4"
            strokeWidth={ci === 0 ? 1.4 : 1.0}
            opacity={0.55 + 0.04 * ci}
          />
        ))
      )}

      {/* line segment from x to x+α_max d (the slice we plot in φ) */}
      <line
        x1={xPx}
        y1={yPx}
        x2={arrowEndX}
        y2={arrowEndY}
        stroke="#1a1a1a"
        strokeWidth={1.4}
        strokeDasharray="2 4"
        opacity={0.5}
      />

      {/* descent-direction arrow */}
      <defs>
        <marker
          id="arrowhead"
          markerWidth="10"
          markerHeight="10"
          refX="6"
          refY="3"
          orient="auto"
        >
          <path d="M0,0 L0,6 L6,3 z" fill="#b04a2a" />
        </marker>
      </defs>
      <line
        x1={xPx}
        y1={yPx}
        x2={arrowEndX}
        y2={arrowEndY}
        stroke="#b04a2a"
        strokeWidth={2.2}
        markerEnd="url(#arrowhead)"
      />

      {/* tip = current iterate after step α */}
      <circle
        cx={tipXPx}
        cy={tipYPx}
        r={6}
        fill="#1f4e8c"
        stroke="white"
        strokeWidth={1.6}
      />

      {/* origin x */}
      <circle
        cx={xPx}
        cy={yPx}
        r={6}
        fill="#111"
        stroke="white"
        strokeWidth={1.6}
      />
    </svg>
  );
}

// ============================================================
// 1D plot of φ(α) with Armijo line, tangent, marker, and
// green/blue acceptance bands.
// ============================================================
function PhiPlot({
  fnDef,
  x,
  d,
  alphaMax,
  alpha,
  c1,
  c2,
  phiSamples,
  armijoOk,
  wolfeOk,
  bands,
}) {
  const W = 460;
  const H = PLOT_SIZE;
  const padL = 46, padR = 14, padT = 16, padB = 36;
  const innerW = W - padL - padR;
  const innerH = H - padT - padB;

  const phi0 = phiSamples[0].phi;
  const dphi0 = phiSamples[0].dphi;

  // y range: φ(α) plus the Armijo line endpoints so it is always on screen
  let yMin = Infinity, yMax = -Infinity;
  for (const s of phiSamples) {
    if (s.phi < yMin) yMin = s.phi;
    if (s.phi > yMax) yMax = s.phi;
  }
  // Include the Armijo line at α=0 and α=αmax
  const armijoMax = phi0 + c1 * alphaMax * dphi0;
  const tangentMax = phi0 + alphaMax * dphi0;
  yMin = Math.min(yMin, armijoMax, tangentMax);
  yMax = Math.max(yMax, phi0);
  const yPad = 0.08 * (yMax - yMin || 1);
  yMin -= yPad;
  yMax += yPad;

  const aToPx = (a) => padL + (a / alphaMax) * innerW;
  const yToPx = (y) => padT + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  // Build φ(α) polyline path.
  const phiPath = phiSamples
    .map((s, i) => `${i === 0 ? "M" : "L"}${aToPx(s.alpha).toFixed(2)},${yToPx(s.phi).toFixed(2)}`)
    .join(" ");

  // Current α and φ(α).
  const phiAtAlpha = (() => {
    // linear interp from samples for display marker
    const N = phiSamples.length - 1;
    const t = (alpha / alphaMax) * N;
    const i0 = Math.max(0, Math.min(N - 1, Math.floor(t)));
    const frac = t - i0;
    return phiSamples[i0].phi * (1 - frac) + phiSamples[i0 + 1].phi * frac;
  })();

  // Ticks
  const xTicks = [];
  const nXT = 5;
  for (let i = 0; i <= nXT; i++) xTicks.push((i / nXT) * alphaMax);
  const yTicks = [];
  const nYT = 4;
  for (let i = 0; i <= nYT; i++) yTicks.push(yMin + (i / nYT) * (yMax - yMin));

  return (
    <svg
      width={W}
      height={H}
      style={{ background: "#fafafa", borderRadius: 6, border: "1px solid #ddd" }}
    >
      {/* axes */}
      <line x1={padL} y1={padT} x2={padL} y2={padT + innerH} stroke="#888" />
      <line x1={padL} y1={padT + innerH} x2={padL + innerW} y2={padT + innerH} stroke="#888" />

      {/* gridlines + tick labels */}
      {xTicks.map((t, i) => (
        <g key={`xt-${i}`}>
          <line
            x1={aToPx(t)}
            y1={padT}
            x2={aToPx(t)}
            y2={padT + innerH}
            stroke="#eaeaea"
          />
          <text
            x={aToPx(t)}
            y={padT + innerH + 14}
            fontSize="10"
            fill="#666"
            textAnchor="middle"
            fontFamily="monospace"
          >
            {t.toFixed(alphaMax < 0.05 ? 4 : alphaMax < 0.5 ? 3 : 2)}
          </text>
        </g>
      ))}
      {yTicks.map((t, i) => (
        <g key={`yt-${i}`}>
          <line
            x1={padL}
            y1={yToPx(t)}
            x2={padL + innerW}
            y2={yToPx(t)}
            stroke="#eaeaea"
          />
          <text
            x={padL - 4}
            y={yToPx(t) + 3}
            fontSize="10"
            fill="#666"
            textAnchor="end"
            fontFamily="monospace"
          >
            {Math.abs(t) > 1000 || (Math.abs(t) < 0.01 && t !== 0)
              ? t.toExponential(1)
              : t.toFixed(2)}
          </text>
        </g>
      ))}

      <text
        x={padL + innerW / 2}
        y={H - 6}
        fontSize="11"
        fill="#444"
        textAnchor="middle"
        fontFamily="monospace"
      >
        α (step size)
      </text>
      <text
        x={12}
        y={padT + innerH / 2}
        fontSize="11"
        fill="#444"
        textAnchor="middle"
        fontFamily="monospace"
        transform={`rotate(-90, 12, ${padT + innerH / 2})`}
      >
        φ(α) = f(x + αd)
      </text>

      {/* Armijo acceptance band on x-axis */}
      {bands.armijo.map(([a0, a1], i) => (
        <rect
          key={`ab-${i}`}
          x={aToPx(a0)}
          y={padT + innerH - 8}
          width={Math.max(1, aToPx(a1) - aToPx(a0))}
          height={8}
          fill="#2e9b55"
          opacity={0.55}
        />
      ))}
      {/* Wolfe acceptance band */}
      {bands.wolfe.map(([a0, a1], i) => (
        <rect
          key={`wb-${i}`}
          x={aToPx(a0)}
          y={padT + innerH - 18}
          width={Math.max(1, aToPx(a1) - aToPx(a0))}
          height={8}
          fill="#1f6fb8"
          opacity={0.6}
        />
      ))}

      {/* Tangent at α = 0 (steepest descent direction) */}
      <line
        x1={aToPx(0)}
        y1={yToPx(phi0)}
        x2={aToPx(alphaMax)}
        y2={yToPx(phi0 + alphaMax * dphi0)}
        stroke="#888"
        strokeWidth={1.4}
        strokeDasharray="4 4"
      />

      {/* Armijo line: y = φ(0) + c1 α φ'(0) */}
      <line
        x1={aToPx(0)}
        y1={yToPx(phi0)}
        x2={aToPx(alphaMax)}
        y2={yToPx(phi0 + c1 * alphaMax * dphi0)}
        stroke="#2e9b55"
        strokeWidth={1.8}
      />

      {/* φ(α) curve */}
      <path d={phiPath} fill="none" stroke="#1a1a1a" strokeWidth={2.2} />

      {/* φ(0) horizontal reference */}
      <line
        x1={aToPx(0)}
        y1={yToPx(phi0)}
        x2={aToPx(alphaMax)}
        y2={yToPx(phi0)}
        stroke="#bbb"
        strokeDasharray="2 3"
      />

      {/* Vertical α marker */}
      <line
        x1={aToPx(alpha)}
        y1={padT}
        x2={aToPx(alpha)}
        y2={padT + innerH}
        stroke="#b04a2a"
        strokeWidth={1.4}
        strokeDasharray="3 3"
      />
      <circle
        cx={aToPx(alpha)}
        cy={yToPx(phiAtAlpha)}
        r={6}
        fill="#b04a2a"
        stroke="white"
        strokeWidth={1.6}
      />
      <text
        x={aToPx(alpha) + 8}
        y={yToPx(phiAtAlpha) - 8}
        fontSize="11"
        fill="#b04a2a"
        fontFamily="monospace"
      >
        φ({alpha.toFixed(alphaMax < 0.05 ? 4 : 3)}) = {phiAtAlpha.toFixed(3)}
      </text>

      {/* Status pills (top-left of plot area) */}
      <g transform={`translate(${padL + 8}, ${padT + 8})`}>
        <rect
          x={0}
          y={0}
          width={86}
          height={20}
          rx={4}
          fill={armijoOk ? "#2e9b55" : "#e0e0e0"}
        />
        <text
          x={43}
          y={14}
          fontSize="11"
          fontFamily="monospace"
          fill={armijoOk ? "#fff" : "#666"}
          textAnchor="middle"
          fontWeight={600}
        >
          Armijo {armijoOk ? "✓" : "✗"}
        </text>
        <rect
          x={92}
          y={0}
          width={86}
          height={20}
          rx={4}
          fill={wolfeOk ? "#1f6fb8" : "#e0e0e0"}
        />
        <text
          x={135}
          y={14}
          fontSize="11"
          fontFamily="monospace"
          fill={wolfeOk ? "#fff" : "#666"}
          textAnchor="middle"
          fontWeight={600}
        >
          Wolfe {wolfeOk ? "✓" : "✗"}
        </text>
      </g>
    </svg>
  );
}

// ============================================================
// Top-level demo
// ============================================================
export default function LineSearchDemo() {
  const [fnKey, setFnKey] = useState("quadratic");
  const fnDef = FUNCTIONS[fnKey];

  const [x, setX] = useState(fnDef.defaultX);
  const [alphaMax, setAlphaMax] = useState(fnDef.defaultAlphaMax);
  const [alpha, setAlpha] = useState(fnDef.defaultAlphaMax * 0.5);
  const [c1, setC1] = useState(1e-4);
  const [c2, setC2] = useState(0.9);
  const [pickMode, setPickMode] = useState(false);

  // animation state for backtracking
  const [backtracking, setBacktracking] = useState(false);
  const btTimer = useRef(null);

  // Reset state when the function changes.
  useEffect(() => {
    setX(fnDef.defaultX);
    setAlphaMax(fnDef.defaultAlphaMax);
    setAlpha(fnDef.defaultAlphaMax * 0.5);
    setBacktracking(false);
    if (btTimer.current) {
      clearInterval(btTimer.current);
      btTimer.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fnKey]);

  // Direction d = -∇f(x). Lock for clarity.
  const grad = useMemo(() => fnDef.grad(x[0], x[1]), [fnDef, x]);
  const d = useMemo(() => [-grad[0], -grad[1]], [grad]);

  // φ samples over [0, αmax]
  const NS = 240;
  const phiSamples = useMemo(() => {
    const out = [];
    for (let i = 0; i <= NS; i++) {
      const a = (i / NS) * alphaMax;
      const xa = x[0] + a * d[0];
      const ya = x[1] + a * d[1];
      const phi = fnDef.f(xa, ya);
      const ga = fnDef.grad(xa, ya);
      const dphi = ga[0] * d[0] + ga[1] * d[1];
      out.push({ alpha: a, phi, dphi });
    }
    return out;
  }, [fnDef, x, d, alphaMax]);

  const phi0 = phiSamples[0].phi;
  const dphi0 = phiSamples[0].dphi;

  // Helper: sample-index → does Armijo / Wolfe hold here?
  const armijoBand = useMemo(() => {
    // Connected α-intervals where φ(α) ≤ φ(0) + c1 α φ'(0).
    const segs = [];
    let inSeg = false;
    let segStart = 0;
    for (const s of phiSamples) {
      const ok = s.phi <= phi0 + c1 * s.alpha * dphi0 + 1e-12;
      if (ok && !inSeg) {
        inSeg = true;
        segStart = s.alpha;
      } else if (!ok && inSeg) {
        inSeg = false;
        segs.push([segStart, s.alpha]);
      }
    }
    if (inSeg) segs.push([segStart, alphaMax]);
    return segs;
  }, [phiSamples, c1, phi0, dphi0, alphaMax]);

  const wolfeBand = useMemo(() => {
    // Strong Wolfe: requires Armijo AND |φ'(α)| ≤ c2 |φ'(0)|.
    const target = c2 * Math.abs(dphi0);
    const segs = [];
    let inSeg = false;
    let segStart = 0;
    for (const s of phiSamples) {
      const armOk = s.phi <= phi0 + c1 * s.alpha * dphi0 + 1e-12;
      const curvOk = Math.abs(s.dphi) <= target + 1e-12;
      const ok = armOk && curvOk;
      if (ok && !inSeg) {
        inSeg = true;
        segStart = s.alpha;
      } else if (!ok && inSeg) {
        inSeg = false;
        segs.push([segStart, s.alpha]);
      }
    }
    if (inSeg) segs.push([segStart, alphaMax]);
    return segs;
  }, [phiSamples, c1, c2, phi0, dphi0, alphaMax]);

  // Eval φ and φ' at the current α (for status pills).
  const phiAtAlpha = useMemo(() => {
    const xa = x[0] + alpha * d[0];
    const ya = x[1] + alpha * d[1];
    return fnDef.f(xa, ya);
  }, [fnDef, x, d, alpha]);
  const dphiAtAlpha = useMemo(() => {
    const xa = x[0] + alpha * d[0];
    const ya = x[1] + alpha * d[1];
    const ga = fnDef.grad(xa, ya);
    return ga[0] * d[0] + ga[1] * d[1];
  }, [fnDef, x, d, alpha]);

  const armijoOk = phiAtAlpha <= phi0 + c1 * alpha * dphi0 + 1e-9;
  const wolfeOk = armijoOk && Math.abs(dphiAtAlpha) <= c2 * Math.abs(dphi0) + 1e-9;

  // ---------- Backtracking animation ----------
  const startBacktracking = useCallback(() => {
    if (btTimer.current) clearInterval(btTimer.current);
    let a = Math.min(1.0, alphaMax);
    setAlpha(a);
    setBacktracking(true);
    btTimer.current = setInterval(() => {
      // Re-evaluate Armijo at current a.
      const xa = x[0] + a * d[0];
      const ya = x[1] + a * d[1];
      const phiA = fnDef.f(xa, ya);
      const ok = phiA <= phi0 + c1 * a * dphi0 + 1e-9;
      if (ok || a < 1e-9) {
        clearInterval(btTimer.current);
        btTimer.current = null;
        setBacktracking(false);
        return;
      }
      a = a * 0.5;
      setAlpha(a);
    }, 380);
  }, [alphaMax, x, d, fnDef, c1, phi0, dphi0]);

  useEffect(() => {
    return () => {
      if (btTimer.current) clearInterval(btTimer.current);
    };
  }, []);

  const stopBacktracking = useCallback(() => {
    if (btTimer.current) {
      clearInterval(btTimer.current);
      btTimer.current = null;
    }
    setBacktracking(false);
  }, []);

  // Clicking on contour to set x (when pickMode is on).
  const handlePick = useCallback(
    (newX) => {
      setX(newX);
      setPickMode(false);
    },
    []
  );

  // Take-step button: commit x ← x + α d, reset α to default mid-range.
  const takeStep = useCallback(() => {
    setX([x[0] + alpha * d[0], x[1] + alpha * d[1]]);
    setAlpha(alphaMax * 0.5);
  }, [x, alpha, d, alphaMax]);

  const resetAll = useCallback(() => {
    stopBacktracking();
    setX(fnDef.defaultX);
    setAlphaMax(fnDef.defaultAlphaMax);
    setAlpha(fnDef.defaultAlphaMax * 0.5);
    setC1(1e-4);
    setC2(0.9);
  }, [fnDef, stopBacktracking]);

  // ---------- styles ----------
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
  const labelStyle = {
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
  const Stat = ({ label, value }) => (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 13 }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>{value}</span>
    </div>
  );

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Line Search — Armijo & Wolfe Conditions
      </h1>
      <p style={{ color: "#666", marginBottom: 24 }}>
        Pick a step size α along a descent direction <code style={code}>d = −∇f(x)</code>.
        The Armijo and Wolfe conditions tell us which α's are "good enough."
      </p>

      <section style={section}>
        <h2 style={h2}>Visualizing the slice φ(α) = f(x + αd)</h2>
        <p style={p}>
          The left plot shows the contour map of <code style={code}>f</code> with the
          current iterate <b>x</b> (black) and a descent direction <b>d</b> (red arrow,
          pointing toward decrease). The blue dot is{" "}
          <code style={code}>x + αd</code> for the slider's α. The right plot is the 1D
          slice φ(α) along the dashed line. Compare φ(α) (black) against the{" "}
          <span style={{ color: "#2e9b55", fontWeight: 600 }}>Armijo line</span>{" "}
          (sufficient decrease) and the dashed{" "}
          <span style={{ color: "#666" }}>tangent</span> at α=0 (best possible local
          rate). The <span style={{ color: "#2e9b55", fontWeight: 600 }}>green band</span>{" "}
          marks α's that satisfy Armijo; the{" "}
          <span style={{ color: "#1f6fb8", fontWeight: 600 }}>blue band</span> marks α's
          that satisfy strong Wolfe (Armijo + curvature).
        </p>

        <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <ContourPlot
              fnDef={fnDef}
              x={x}
              alpha={alpha}
              d={d}
              onClickPick={handlePick}
              pickMode={pickMode}
            />
            <PhiPlot
              fnDef={fnDef}
              x={x}
              d={d}
              alphaMax={alphaMax}
              alpha={alpha}
              c1={c1}
              c2={c2}
              phiSamples={phiSamples}
              armijoOk={armijoOk}
              wolfeOk={wolfeOk}
              bands={{ armijo: armijoBand, wolfe: wolfeBand }}
            />
          </div>

          <div style={{ minWidth: 280, flex: 1 }}>
            <div style={controlGroup}>
              <label style={labelStyle}>Function</label>
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
            </div>

            <div style={controlGroup}>
              <label style={labelStyle}>
                α (step size): <b>{alpha.toFixed(alphaMax < 0.05 ? 5 : 4)}</b> /{" "}
                {alphaMax.toFixed(alphaMax < 0.05 ? 4 : 2)}
              </label>
              <input
                type="range"
                min={0}
                max={alphaMax}
                step={alphaMax / 500}
                value={alpha}
                onChange={(e) => setAlpha(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>

            <div style={controlGroup}>
              <label style={labelStyle}>
                α<sub>max</sub> (slice length): <b>{alphaMax.toFixed(alphaMax < 0.05 ? 4 : 3)}</b>
              </label>
              <input
                type="range"
                min={Math.max(1e-4, fnDef.defaultAlphaMax * 0.1)}
                max={fnDef.defaultAlphaMax * 5}
                step={fnDef.defaultAlphaMax * 0.01}
                value={alphaMax}
                onChange={(e) => {
                  const v = +e.target.value;
                  setAlphaMax(v);
                  if (alpha > v) setAlpha(v);
                }}
                style={{ width: "100%" }}
              />
            </div>

            <div style={controlGroup}>
              <label style={labelStyle}>
                c₁ (Armijo): <b>{c1.toExponential(1)}</b>
              </label>
              <input
                type="range"
                min={1e-4}
                max={0.5}
                step={1e-4}
                value={c1}
                onChange={(e) => setC1(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>

            <div style={controlGroup}>
              <label style={labelStyle}>
                c₂ (Wolfe curvature): <b>{c2.toFixed(2)}</b>
              </label>
              <input
                type="range"
                min={0.1}
                max={0.99}
                step={0.01}
                value={c2}
                onChange={(e) => setC2(+e.target.value)}
                style={{ width: "100%" }}
              />
            </div>

            <div style={controlGroup}>
              <label style={labelStyle}>
                x: <b>({x[0].toFixed(3)}, {x[1].toFixed(3)})</b>
              </label>
              <div style={{ display: "flex", gap: 6 }}>
                <input
                  type="range"
                  min={fnDef.domain.lo}
                  max={fnDef.domain.hi}
                  step={(fnDef.domain.hi - fnDef.domain.lo) / 200}
                  value={x[0]}
                  onChange={(e) => setX([+e.target.value, x[1]])}
                  style={{ width: "50%" }}
                />
                <input
                  type="range"
                  min={fnDef.domain.lo}
                  max={fnDef.domain.hi}
                  step={(fnDef.domain.hi - fnDef.domain.lo) / 200}
                  value={x[1]}
                  onChange={(e) => setX([x[0], +e.target.value])}
                  style={{ width: "50%" }}
                />
              </div>
            </div>

            <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
              <button
                onClick={() => (backtracking ? stopBacktracking() : startBacktracking())}
                style={btnPrimary}
              >
                <FastForward size={16} />
                {backtracking ? "Stop" : "Backtrack (×½)"}
              </button>
              <button onClick={takeStep} style={btn} disabled={backtracking}>
                <Play size={16} /> Take step
              </button>
              <button
                onClick={() => setPickMode((m) => !m)}
                style={pickMode ? btnPrimary : btn}
              >
                <Crosshair size={16} /> {pickMode ? "Click plot…" : "Pick x"}
              </button>
              <button onClick={resetAll} style={btn}>
                <RotateCcw size={16} /> Reset
              </button>
            </div>

            <div style={statBox}>
              <Stat label="f(x)" value={phi0.toFixed(4)} />
              <Stat label="‖∇f(x)‖" value={Math.hypot(grad[0], grad[1]).toFixed(4)} />
              <Stat label="φ'(0) = ∇f·d" value={dphi0.toFixed(4)} />
              <Stat label="φ(α)" value={phiAtAlpha.toFixed(4)} />
              <Stat label="φ'(α)" value={dphiAtAlpha.toFixed(4)} />
              <Stat
                label="Armijo: φ(α) ≤ φ(0)+c₁αφ'(0)"
                value={armijoOk ? "✓ holds" : "✗ fails"}
              />
              <Stat
                label="|φ'(α)| ≤ c₂|φ'(0)|"
                value={
                  Math.abs(dphiAtAlpha) <= c2 * Math.abs(dphi0) + 1e-9
                    ? "✓ holds"
                    : "✗ fails"
                }
              />
              <Stat label="Strong Wolfe" value={wolfeOk ? "✓ holds" : "✗ fails"} />
            </div>
          </div>
        </div>

        <div style={{ marginTop: 18, fontSize: 13, color: "#444", lineHeight: 1.55 }}>
          <p style={{ marginBottom: 8 }}>
            <b>What to try.</b>
          </p>
          <ul style={{ paddingLeft: 22, marginTop: 0 }}>
            <li>
              Drag the <b>α</b> slider and watch the red marker glide along φ(α). Where
              it sits below the green Armijo line, sufficient decrease holds.
            </li>
            <li>
              Push <b>c₁</b> from 10⁻⁴ toward 0.5 — the green Armijo line tilts steeper
              and the green band shrinks. Big c₁ ⇒ stricter decrease requirement.
            </li>
            <li>
              Lower <b>c₂</b> toward 0.1 — the Wolfe (blue) band shrinks, demanding the
              slope nearly vanish. With c₂ → 1, Wolfe is easy to satisfy.
            </li>
            <li>
              Click <b>Backtrack (×½)</b> on Rosenbrock with a large α<sub>max</sub>.
              You'll see α get repeatedly halved until Armijo first holds — the standard
              backtracking line search.
            </li>
            <li>
              Switch to <b>Rosenbrock</b> and pick an x in the curved valley with the{" "}
              <i>Pick x</i> button: φ(α) becomes strongly nonconvex along the line, so
              the Wolfe band can be a small interval near a "good" curvature point.
            </li>
          </ul>
          <p style={{ marginTop: 10 }}>
            <b>Recall.</b> The Armijo (sufficient decrease) condition is{" "}
            <code style={code}>f(x + αd) ≤ f(x) + c₁ α ∇f(x)·d</code>. The strong Wolfe
            curvature condition is <code style={code}>|∇f(x+αd)·d| ≤ c₂ |∇f(x)·d|</code>.
            Together they define a non-empty interval of acceptable α whenever{" "}
            <code style={code}>d</code> is a descent direction and{" "}
            <code style={code}>0 &lt; c₁ &lt; c₂ &lt; 1</code>.
          </p>
        </div>
      </section>
    </div>
  );
}
