import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Play, Pause, RotateCcw, StepForward, CheckCircle2 } from "lucide-react";

/* ============================================================
   PERCEPTRON TUTORIAL — CODE STEPPER
   ISE 5406 (Nonlinear Programming)

   Pedagogical: students follow Python-like pseudocode line by
   line on the left while a 2D scatter plot animates on the right.
   Each click of "Next step" advances to the next semantically
   meaningful highlighted line of the algorithm.

   This file is self-contained (does not depend on svm_demo.jsx).
   ============================================================ */

// ---------------- RNG ----------------
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const randn = (rng) => {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ---------------- Datasets ----------------
function dsEasy(seed = 1, n = 16) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: 1.5 + 0.45 * randn(rng), y: 1.1 + 0.45 * randn(rng), label: 1 });
  }
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: -1.5 + 0.45 * randn(rng), y: -1.0 + 0.45 * randn(rng), label: -1 });
  }
  return pts;
}
function dsTight(seed = 7, n = 22) {
  const rng = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: 0.7 + 0.35 * randn(rng), y: 0.4 + 0.35 * randn(rng), label: 1 });
  }
  for (let i = 0; i < n / 2; i++) {
    pts.push({ x: -0.7 + 0.35 * randn(rng), y: -0.4 + 0.35 * randn(rng), label: -1 });
  }
  return pts;
}
function dsHarder(seed = 11, n = 40) {
  // Still linearly separable (perceptron requires that), but with a thin
  // margin and more points so the trajectory is more interesting.
  const rng = mulberry32(seed);
  const pts = [];
  // True boundary: y = 0.4*x + 0.3 (margin ~ 0.25 on each side)
  let placed = 0;
  while (placed < n) {
    const x = -2.4 + 4.8 * rng();
    const y = -2.2 + 4.4 * rng();
    const sgn = y - (0.4 * x + 0.3);
    if (Math.abs(sgn) < 0.3) continue; // enforce margin
    pts.push({ x, y, label: sgn > 0 ? 1 : -1 });
    placed++;
  }
  return pts;
}

const DATASETS = {
  easy: { name: "Easy (linearly separable)", make: () => dsEasy(1, 16) },
  tight: { name: "Tight margin", make: () => dsTight(7, 22) },
  harder: { name: "Harder (more points)", make: () => dsHarder(11, 40) },
};

// ---------------- Plot helpers ----------------
const PLOT = { size: 440, lo: -3, hi: 3 };
const dataToPxX = (x) => ((x - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;
const dataToPxY = (y) => PLOT.size - ((y - PLOT.lo) / (PLOT.hi - PLOT.lo)) * PLOT.size;

const COLOR_POS = "#1f4e3d";
const COLOR_NEG = "#c8311c";
const COLOR_HL = "#f5a524";

// Compute the two endpoints of the decision line w·x + b = 0 inside
// the plot box. Returns null if w is essentially zero.
function decisionLine(w, b) {
  const [a, c] = w;
  const eps = 1e-9;
  if (Math.abs(a) < eps && Math.abs(c) < eps) return null;
  const lo = PLOT.lo,
    hi = PLOT.hi;
  const pts = [];
  if (Math.abs(c) > eps) {
    const yLo = -(a * lo + b) / c;
    const yHi = -(a * hi + b) / c;
    if (yLo >= lo && yLo <= hi) pts.push([lo, yLo]);
    if (yHi >= lo && yHi <= hi) pts.push([hi, yHi]);
  }
  if (Math.abs(a) > eps) {
    const xLo = -(c * lo + b) / a;
    const xHi = -(c * hi + b) / a;
    if (xLo >= lo && xLo <= hi) pts.push([xLo, lo]);
    if (xHi >= lo && xHi <= hi) pts.push([xHi, hi]);
  }
  if (pts.length < 2) return null;
  return [pts[0], pts[1]];
}

// ---------------- Pseudocode (literal, line-numbered) ----------------
const CODE_LINES = [
  "w = [0, 0]",
  "b = 0",
  "while not converged:",
  "  converged = True",
  "  for (x, y) in data:",
  "    score = w · x + b",
  "    if y * score <= 0:           # misclassified",
  "      w = w + y * x",
  "      b = b + y",
  "      converged = False",
  "# done — perceptron has separated the data",
];

// ---------------- Step planner -----------------------------------------
// We pre-compute a deterministic list of highlighted-line "events" so
// that Step / Run advance through them one at a time. This keeps the
// rendering simple: each event records (lineIdx, action) and a small
// payload describing how to mutate w, b, the inspected point, etc.
//
// Event kinds:
//   init-w   -> highlight line 1, set w := [0,0]
//   init-b   -> highlight line 2, set b := 0
//   while    -> highlight line 3 (top of outer loop, new pass)
//   pass-set -> highlight line 4 (converged = True for this pass)
//   pick     -> highlight line 5, mark point i as the inspected one
//   score    -> highlight line 6, compute score for point i
//   check    -> highlight line 7, evaluate y*score sign
//   update-w -> highlight line 8, update w (animated)
//   update-b -> highlight line 9, update b (animated)
//   nopass   -> highlight line 10, set pass-converged=False, bump update counter
//   done     -> highlight line 11, "converged"
function planEvents(data, maxPasses = 30) {
  const events = [];
  const push = (kind, extra = {}) => events.push({ kind, ...extra });

  let w = [0, 0];
  let b = 0;

  push("init-w", { w: [0, 0] });
  push("init-b", { b: 0 });

  for (let pass = 1; pass <= maxPasses; pass++) {
    push("while", { pass });
    push("pass-set", { pass });
    let passConverged = true;

    for (let i = 0; i < data.length; i++) {
      const p = data[i];
      push("pick", { i, pass });
      const score = w[0] * p.x + w[1] * p.y + b;
      push("score", { i, pass, score });
      const margin = p.label * score;
      push("check", { i, pass, score, margin, mis: margin <= 0 });
      if (margin <= 0) {
        const newW = [w[0] + p.label * p.x, w[1] + p.label * p.y];
        push("update-w", { i, pass, score, fromW: [...w], toW: newW });
        w = newW;
        const newB = b + p.label;
        push("update-b", { i, pass, score, fromB: b, toB: newB });
        b = newB;
        push("nopass", { i, pass, score });
        passConverged = false;
      }
    }

    if (passConverged) {
      push("done", { pass });
      break;
    }
  }

  // Safety: if we ran out of passes without converging, still emit done so
  // the stepper has a terminal event. (Shouldn't happen on the bundled
  // datasets — they're all linearly separable.)
  if (events[events.length - 1].kind !== "done") {
    push("done", { pass: maxPasses, exhausted: true });
  }
  return events;
}

// Map an event to the code line number (1-based) it highlights.
function eventLine(ev) {
  switch (ev.kind) {
    case "init-w": return 1;
    case "init-b": return 2;
    case "while": return 3;
    case "pass-set": return 4;
    case "pick": return 5;
    case "score": return 6;
    case "check": return 7;
    case "update-w": return 8;
    case "update-b": return 9;
    case "nopass": return 10;
    case "done": return 11;
    default: return 1;
  }
}

// =======================================================================
// MAIN COMPONENT
// =======================================================================
export default function PerceptronTutorial() {
  const [datasetKey, setDatasetKey] = useState("easy");
  const [data, setData] = useState(() => DATASETS.easy.make());
  const [events, setEvents] = useState(() => planEvents(DATASETS.easy.make()));
  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);

  // Animated decision-line state. We linearly interpolate w and b from
  // their pre-update to post-update values across ~300ms whenever an
  // update-w / update-b event becomes the current event.
  const [animW, setAnimW] = useState([0, 0]);
  const [animB, setAnimB] = useState(0);
  const animRef = useRef({ raf: 0 });

  // Reset everything when the dataset changes.
  const rebuild = useCallback((key) => {
    const d = DATASETS[key].make();
    setData(d);
    setEvents(planEvents(d));
    setEvIdx(0);
    setAnimW([0, 0]);
    setAnimB(0);
    setRunning(false);
    cancelAnimationFrame(animRef.current.raf);
  }, []);

  // -------- Walk to a specific event index --------
  // Recompute the "logical" w, b at event idx by replaying events from
  // the start. The event array is small, so this is cheap.
  function replayUpTo(idx) {
    let w = [0, 0];
    let b = 0;
    let inspected = -1;
    let lastScore = null;
    let updates = 0;
    let pass = 0;
    for (let k = 0; k <= idx; k++) {
      const ev = events[k];
      if (!ev) break;
      switch (ev.kind) {
        case "init-w": w = [0, 0]; break;
        case "init-b": b = 0; break;
        case "while": pass = ev.pass; break;
        case "pick": inspected = ev.i; break;
        case "score": lastScore = ev.score; break;
        case "update-w": w = ev.toW; break;
        case "update-b": b = ev.toB; break;
        case "nopass": updates += 1; break;
        default: break;
      }
    }
    return { w, b, inspected, lastScore, updates, pass };
  }

  // Snapshot at the *end* of the current event (i.e. after it executes).
  const snap = useMemo(() => replayUpTo(evIdx), [evIdx, events]);

  const currentEvent = events[evIdx] || events[events.length - 1];
  const highlightedLine = eventLine(currentEvent);
  const isDone = currentEvent.kind === "done";

  // -------- Animate w / b changes --------
  // When the current event is an update, smoothly interpolate from its
  // "from" value to its "to" value over ~300ms. For other events, snap
  // immediately to the logical w/b at this point in execution.
  useEffect(() => {
    cancelAnimationFrame(animRef.current.raf);
    const ev = currentEvent;
    if (ev.kind === "update-w") {
      const t0 = performance.now();
      const dur = 300;
      const fromW = ev.fromW;
      const toW = ev.toW;
      // b stays fixed during a w-update.
      setAnimB(snap.b);
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        setAnimW([
          fromW[0] + (toW[0] - fromW[0]) * eased,
          fromW[1] + (toW[1] - fromW[1]) * eased,
        ]);
        if (t < 1) animRef.current.raf = requestAnimationFrame(tick);
      };
      animRef.current.raf = requestAnimationFrame(tick);
    } else if (ev.kind === "update-b") {
      const t0 = performance.now();
      const dur = 300;
      const fromB = ev.fromB;
      const toB = ev.toB;
      setAnimW(snap.w); // already at the post-w-update value
      const tick = (now) => {
        const t = Math.min(1, (now - t0) / dur);
        const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
        setAnimB(fromB + (toB - fromB) * eased);
        if (t < 1) animRef.current.raf = requestAnimationFrame(tick);
      };
      animRef.current.raf = requestAnimationFrame(tick);
    } else {
      setAnimW(snap.w);
      setAnimB(snap.b);
    }
    return () => cancelAnimationFrame(animRef.current.raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [evIdx, events]);

  // -------- Auto-run timer --------
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setEvIdx((i) => {
        if (i >= events.length - 1) {
          setRunning(false);
          return i;
        }
        return i + 1;
      });
    }, 250);
    return () => clearInterval(id);
  }, [running, events.length]);

  const stepOnce = useCallback(() => {
    setEvIdx((i) => Math.min(i + 1, events.length - 1));
  }, [events.length]);

  const reset = useCallback(() => {
    setEvIdx(0);
    setAnimW([0, 0]);
    setAnimB(0);
    setRunning(false);
  }, []);

  // -------- Derived plot data --------
  const linePts = useMemo(() => decisionLine(animW, animB), [animW, animB]);

  // The narration string shown beneath the stats — explains in plain
  // English what just happened.
  const narration = useMemo(() => {
    const ev = currentEvent;
    const i = ev.i;
    const p = i != null ? data[i] : null;
    switch (ev.kind) {
      case "init-w":
        return "Initialize the weight vector w to zero. With w = 0, every score is 0.";
      case "init-b":
        return "Initialize the bias b to zero.";
      case "while":
        return `Begin pass #${ev.pass}. We will scan all points; if any is misclassified we'll need another pass.`;
      case "pass-set":
        return "Optimistically assume this pass will converge — we'll set this flag back to False if any update happens.";
      case "pick":
        return `Pick the next point: index ${i}, x = (${p.x.toFixed(2)}, ${p.y.toFixed(2)}), label y = ${p.label > 0 ? "+1" : "-1"}.`;
      case "score":
        return `Compute score = w·x + b = ${ev.score.toFixed(3)}. The sign of this number is what the perceptron predicts.`;
      case "check": {
        const m = ev.margin;
        if (ev.mis) {
          return `y·score = ${m.toFixed(3)} ≤ 0 — misclassified! Apply the update rule.`;
        }
        return `y·score = ${m.toFixed(3)} > 0 — correctly classified. Skip to the next point.`;
      }
      case "update-w":
        return `Update weights: w ← w + y·x. The decision line tilts so this point is closer to the correct side.`;
      case "update-b":
        return `Update bias: b ← b + y. The decision line shifts.`;
      case "nopass":
        return "Mark this pass as not converged — we made an update, so we'll need another sweep.";
      case "done":
        return ev.exhausted
          ? "Stopped: max passes reached without convergence (data may not be separable)."
          : "✓ Converged! A full pass made no updates, so every point is classified correctly.";
      default:
        return "";
    }
  }, [currentEvent, data]);

  // -------- Render --------
  return (
    <section style={S.section}>
      <h2 style={S.h2}>Perceptron — Code Stepper Tutorial</h2>
      <p style={S.p}>
        Step through the online perceptron algorithm one line of pseudocode at a
        time. The plot on the right shows the current decision line
        <code style={S.code}>w·x + b = 0</code>; the highlighted point is the
        one currently being inspected. Whenever a misclassified point triggers
        the update rule
        <code style={S.code}>w ← w + y·x</code>, the line animates into
        its new position.
      </p>

      <div style={S.row}>
        {/* ---------------- LEFT: code panel ---------------- */}
        <div style={S.codePanel}>
          <div style={S.codeHeader}>perceptron.py</div>
          <pre style={S.codeBody}>
            {CODE_LINES.map((line, idx) => {
              const lineNo = idx + 1;
              const active = lineNo === highlightedLine;
              return (
                <div
                  key={lineNo}
                  style={{
                    ...S.codeLine,
                    background: active ? "#fff3bf" : "transparent",
                    borderLeft: active ? "3px solid #f59f00" : "3px solid transparent",
                  }}
                >
                  <span style={S.gutter}>
                    <span style={S.gutterMarker}>{active ? "▶" : " "}</span>
                    <span style={S.lineNo}>{String(lineNo).padStart(2, " ")}</span>
                  </span>
                  <span style={S.codeText}>{line}</span>
                </div>
              );
            })}
          </pre>

          <div style={{ ...S.controls, marginTop: 14 }}>
            <button
              onClick={stepOnce}
              disabled={evIdx >= events.length - 1}
              style={S.btnPrimary}
            >
              <StepForward size={16} /> Next step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={evIdx >= events.length - 1}
              style={S.btn}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={reset} style={S.btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={S.controlGroup}>
            <label style={S.label}>Dataset</label>
            <select
              value={datasetKey}
              onChange={(e) => {
                setDatasetKey(e.target.value);
                rebuild(e.target.value);
              }}
              style={S.select}
            >
              {Object.entries(DATASETS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          <div style={S.progressWrap}>
            <div style={S.progressLabel}>
              Event {evIdx + 1} / {events.length}
            </div>
            <div style={S.progressTrack}>
              <div
                style={{
                  ...S.progressFill,
                  width: `${((evIdx + 1) / events.length) * 100}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* ---------------- RIGHT: plot + stats ---------------- */}
        <div style={S.plotPanel}>
          <svg
            width={PLOT.size}
            height={PLOT.size}
            style={{ background: "#fafafa", borderRadius: 6, border: "1px solid #e5e5e5" }}
          >
            {/* axes */}
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

            {/* decision line */}
            {linePts && (
              <line
                x1={dataToPxX(linePts[0][0])}
                y1={dataToPxY(linePts[0][1])}
                x2={dataToPxX(linePts[1][0])}
                y2={dataToPxY(linePts[1][1])}
                stroke="#111"
                strokeWidth={2.2}
              />
            )}

            {/* data points */}
            {data.map((p, i) => {
              const isHL = i === snap.inspected && !isDone;
              return (
                <g key={i}>
                  {isHL && (
                    <circle
                      cx={dataToPxX(p.x)}
                      cy={dataToPxY(p.y)}
                      r={13}
                      fill="none"
                      stroke={COLOR_HL}
                      strokeWidth={2.5}
                    />
                  )}
                  <circle
                    cx={dataToPxX(p.x)}
                    cy={dataToPxY(p.y)}
                    r={isHL ? 7 : 5.5}
                    fill={p.label === 1 ? COLOR_POS : COLOR_NEG}
                    stroke={isHL ? "#000" : "white"}
                    strokeWidth={isHL ? 2 : 1.2}
                  />
                </g>
              );
            })}

            {/* "convergence" badge */}
            {isDone && !currentEvent.exhausted && (
              <g>
                <rect
                  x={PLOT.size / 2 - 90}
                  y={PLOT.size - 44}
                  width={180}
                  height={30}
                  rx={6}
                  fill="#e6fcf5"
                  stroke="#0ca678"
                />
                <text
                  x={PLOT.size / 2}
                  y={PLOT.size - 24}
                  textAnchor="middle"
                  fontSize={14}
                  fontWeight={600}
                  fill="#0b7a5a"
                >
                  ✓ converged
                </text>
              </g>
            )}
          </svg>

          {/* ---------- stats ---------- */}
          <div style={S.statBox}>
            <Stat label="w" value={`(${snap.w[0].toFixed(2)}, ${snap.w[1].toFixed(2)})`} />
            <Stat label="b" value={snap.b.toFixed(2)} />
            <Stat
              label="last score"
              value={snap.lastScore == null ? "—" : snap.lastScore.toFixed(3)}
            />
            <Stat label="updates" value={snap.updates} />
            <Stat label="pass" value={snap.pass || 0} />
            <Stat
              label="status"
              value={
                isDone
                  ? currentEvent.exhausted
                    ? "stopped"
                    : "converged"
                  : "training"
              }
            />
          </div>

          {/* ---------- narration ---------- */}
          <div style={S.narration}>
            {isDone && !currentEvent.exhausted && (
              <CheckCircle2
                size={18}
                style={{ color: "#0ca678", marginRight: 6, verticalAlign: "-3px" }}
              />
            )}
            {narration}
          </div>
        </div>
      </div>

      <div style={S.notes}>
        <p style={S.p}>
          <b>How to read this.</b> The perceptron makes one update per
          misclassified point. Each update tilts the separating line so that
          the offending point moves closer to (and eventually past) the
          boundary on its correct side. If the data is linearly separable,
          Novikoff’s theorem guarantees the algorithm halts in a finite
          number of updates — you’re watching that argument unfold.
        </p>
      </div>
    </section>
  );
}

// -------- small subcomponents --------
function Stat({ label, value }) {
  return (
    <div style={S.stat}>
      <div style={S.statLabel}>{label}</div>
      <div style={S.statValue}>{value}</div>
    </div>
  );
}

// -------- styles --------
const monoFamily =
  '"JetBrains Mono", "Fira Code", "SF Mono", Menlo, Consolas, "Liberation Mono", monospace';

const S = {
  section: {
    maxWidth: 1100,
    margin: "0 auto",
    padding: "24px 28px",
    fontFamily:
      '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    color: "#1f2933",
  },
  h2: {
    fontSize: 22,
    fontWeight: 700,
    margin: "0 0 8px",
    letterSpacing: "-0.01em",
  },
  p: {
    fontSize: 14.5,
    lineHeight: 1.55,
    color: "#333",
    margin: "8px 0 18px",
  },
  code: {
    fontFamily: monoFamily,
    fontSize: 13,
    background: "#f1f3f5",
    padding: "1px 5px",
    borderRadius: 3,
    margin: "0 3px",
  },
  row: {
    display: "flex",
    gap: 24,
    alignItems: "flex-start",
    flexWrap: "wrap",
  },
  // --- code panel ---
  codePanel: {
    flex: "1 1 460px",
    minWidth: 420,
    background: "#fdfdfd",
    border: "1px solid #e1e4e8",
    borderRadius: 8,
    overflow: "hidden",
    boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
  },
  codeHeader: {
    background: "#f5f7fa",
    padding: "8px 14px",
    fontFamily: monoFamily,
    fontSize: 12,
    color: "#495057",
    borderBottom: "1px solid #e1e4e8",
    letterSpacing: "0.02em",
  },
  codeBody: {
    margin: 0,
    padding: "10px 0",
    fontFamily: monoFamily,
    fontSize: 13.5,
    lineHeight: "1.55",
    background: "#ffffff",
  },
  codeLine: {
    display: "flex",
    alignItems: "flex-start",
    padding: "2px 0",
    transition: "background 120ms linear",
  },
  gutter: {
    flexShrink: 0,
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    paddingLeft: 6,
    paddingRight: 12,
    color: "#868e96",
    userSelect: "none",
  },
  gutterMarker: {
    display: "inline-block",
    width: 12,
    color: "#f59f00",
    fontWeight: 700,
  },
  lineNo: {
    display: "inline-block",
    minWidth: 22,
    textAlign: "right",
    fontFamily: monoFamily,
    fontSize: 12,
    color: "#adb5bd",
  },
  codeText: {
    whiteSpace: "pre",
    color: "#1f2933",
  },
  // --- plot panel ---
  plotPanel: {
    flex: "1 1 460px",
    minWidth: 420,
    display: "flex",
    flexDirection: "column",
    gap: 12,
  },
  // --- controls ---
  controls: {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    padding: "0 14px 14px",
  },
  btn: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "7px 12px",
    fontSize: 13,
    background: "white",
    border: "1px solid #ced4da",
    borderRadius: 6,
    cursor: "pointer",
    color: "#212529",
  },
  btnPrimary: {
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    padding: "7px 12px",
    fontSize: 13,
    background: "#1971c2",
    color: "white",
    border: "1px solid #1864ab",
    borderRadius: 6,
    cursor: "pointer",
    fontWeight: 600,
  },
  controlGroup: {
    padding: "0 14px 14px",
    display: "flex",
    alignItems: "center",
    gap: 10,
  },
  label: {
    fontSize: 12,
    color: "#495057",
    fontWeight: 600,
    letterSpacing: "0.02em",
    textTransform: "uppercase",
  },
  select: {
    flex: 1,
    padding: "6px 8px",
    fontSize: 13,
    border: "1px solid #ced4da",
    borderRadius: 6,
    background: "white",
  },
  // --- progress bar ---
  progressWrap: {
    padding: "0 14px 14px",
  },
  progressLabel: {
    fontSize: 11.5,
    color: "#868e96",
    marginBottom: 4,
    fontFamily: monoFamily,
  },
  progressTrack: {
    height: 5,
    background: "#e9ecef",
    borderRadius: 3,
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    background: "linear-gradient(90deg, #1971c2, #339af0)",
    transition: "width 180ms ease-out",
  },
  // --- stats ---
  statBox: {
    display: "grid",
    gridTemplateColumns: "repeat(3, 1fr)",
    gap: 8,
    padding: 12,
    background: "#f8f9fa",
    border: "1px solid #e1e4e8",
    borderRadius: 8,
  },
  stat: {
    background: "white",
    border: "1px solid #e9ecef",
    borderRadius: 6,
    padding: "6px 10px",
  },
  statLabel: {
    fontSize: 10.5,
    color: "#868e96",
    textTransform: "uppercase",
    letterSpacing: "0.06em",
    fontWeight: 600,
  },
  statValue: {
    fontSize: 14,
    fontFamily: monoFamily,
    color: "#212529",
    marginTop: 2,
  },
  // --- narration ---
  narration: {
    padding: "10px 14px",
    background: "#f1f3f5",
    border: "1px solid #dee2e6",
    borderRadius: 8,
    fontSize: 13.5,
    lineHeight: 1.5,
    color: "#343a40",
    minHeight: 48,
  },
  notes: {
    marginTop: 22,
    paddingTop: 14,
    borderTop: "1px solid #e9ecef",
  },
};
