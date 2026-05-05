import React, { useState, useMemo } from "react";
import { Terminal, RotateCcw } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   INTERACTIVE SIMPLEX TABLEAU
   ISE 5406

   Inspired by gilp.henryrobbins.com — but with an integrated
   sensitivity / duality view, click-to-pivot interaction, and
   a built-in 'practice' mode where students pick the entering
   and leaving variables themselves.

   Problem (modifiable inline):
       max  c · x
       s.t. A x <= b,  x >= 0
   Add slacks → tableau in canonical form. Click any negative
   reduced-cost column → ratio test highlights eligible rows →
   click a row to pivot. Watch the feasible region plot's
   highlighted vertex jump correspondingly.
   ============================================================ */

// ============================================================
// Default problem
//   max  3 x1 + 5 x2
//   s.t. 2 x1 + x2  <= 8
//        x1 + 3 x2  <= 6
//        x1, x2 >= 0
// Optimal: (x1, x2) = (3.6, 0.8), z = 14.8
// ============================================================
const DEFAULT_C = [3, 5];
const DEFAULT_A = [
  [2, 1],
  [1, 3],
];
const DEFAULT_B = [8, 6];

// ============================================================
// Tableau state
// Columns: x1, x2, s1, s2, RHS
// Rows: 0..m-1 = constraints, last = z (objective)
// 'basis' is array of basic variable INDICES (one per constraint row).
// In canonical form for max, the z-row holds NEGATIVE reduced costs of
// non-basic variables (and 0 for basic ones). Optimal when no negative.
// ============================================================
function buildInitialTableau(c, A, b) {
  const m = A.length;
  const n = c.length;
  const totalCols = n + m + 1; // n decision + m slack + RHS
  const T = Array.from({ length: m + 1 }, () => Array(totalCols).fill(0));
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) T[i][j] = A[i][j];
    T[i][n + i] = 1; // slack
    T[i][totalCols - 1] = b[i];
  }
  for (let j = 0; j < n; j++) T[m][j] = -c[j]; // negative reduced costs of decision vars
  // Slack reduced costs are 0 (basic in initial tableau).
  // Initial basis: the slacks.
  const basis = [];
  for (let i = 0; i < m; i++) basis.push(n + i);
  return { T, basis, n, m };
}

// Pivot tableau on (row, col)
function pivot(T, row, col) {
  const newT = T.map((r) => [...r]);
  const piv = newT[row][col];
  for (let j = 0; j < newT[0].length; j++) newT[row][j] /= piv;
  for (let i = 0; i < newT.length; i++) {
    if (i === row) continue;
    const factor = newT[i][col];
    if (Math.abs(factor) < 1e-12) continue;
    for (let j = 0; j < newT[0].length; j++) newT[i][j] -= factor * newT[row][j];
  }
  return newT;
}

function ratioTest(T, m, col) {
  const totalCols = T[0].length;
  const RHS = totalCols - 1;
  const ratios = [];
  for (let i = 0; i < m; i++) {
    const a = T[i][col];
    const r = T[i][RHS];
    if (a > 1e-9) ratios.push({ row: i, ratio: r / a, ok: true });
    else ratios.push({ row: i, ratio: Infinity, ok: false });
  }
  let minRow = -1, minRatio = Infinity;
  ratios.forEach((r) => {
    if (r.ok && r.ratio < minRatio - 1e-9) {
      minRatio = r.ratio;
      minRow = r.row;
    }
  });
  return { ratios, minRow };
}

function reducedCosts(T, m) {
  return T[m]; // last row
}

function isOptimal(T, m, n) {
  for (let j = 0; j < n; j++) if (T[m][j] < -1e-9) return false;
  // Slack columns may also have negative reduced costs (bland's rule etc),
  // but for max problems with c>=0 and standard form they won't.
  return true;
}

function variableLabel(idx, n) {
  if (idx < n) return `x${idx + 1}`;
  return `s${idx - n + 1}`;
}

// ============================================================
// Main component
// ============================================================
export default function SimplexTableauDemo() {
  const [c] = useState(DEFAULT_C);
  const [A] = useState(DEFAULT_A);
  const [b] = useState(DEFAULT_B);
  const initial = useMemo(() => buildInitialTableau(c, A, b), [c, A, b]);

  // History of (T, basis) so we can step backward
  const [history, setHistory] = useState([initial]);
  const [stepIdx, setStepIdx] = useState(0);
  const cur = history[stepIdx];
  const { T, basis, n, m } = cur;

  // Pivot column the user has selected (null = none)
  const [hoverCol, setHoverCol] = useState(null);
  const [practiceMode, setPracticeMode] = useState(false);
  const [feedback, setFeedback] = useState(null);

  const optimal = isOptimal(T, m, n);
  const rcosts = reducedCosts(T, m);

  // Suggested entering column (most-negative reduced cost) & leaving row
  const suggestedCol = useMemo(() => {
    let bestJ = -1, bestVal = -1e-9;
    for (let j = 0; j < n + m; j++) {
      if (rcosts[j] < bestVal) {
        bestVal = rcosts[j];
        bestJ = j;
      }
    }
    return bestJ;
  }, [rcosts, n, m]);
  const suggestedRow = useMemo(() => {
    if (suggestedCol < 0) return -1;
    return ratioTest(T, m, suggestedCol).minRow;
  }, [T, m, suggestedCol]);

  function doPivot(row, col) {
    const newT = pivot(T, row, col);
    const newBasis = [...basis];
    newBasis[row] = col;
    const newHistory = history.slice(0, stepIdx + 1);
    newHistory.push({ T: newT, basis: newBasis, n, m });
    setHistory(newHistory);
    setStepIdx(newHistory.length - 1);
    setHoverCol(null);
    setFeedback(null);
  }

  function handleColumnClick(j) {
    if (j >= n + m) return;
    if (rcosts[j] >= -1e-9) {
      if (practiceMode)
        setFeedback({
          ok: false,
          msg: `Column ${variableLabel(j, n)} has reduced cost ${rcosts[j].toFixed(2)} ≥ 0 — picking it would not improve the objective.`,
        });
      return;
    }
    if (practiceMode) {
      if (j === suggestedCol)
        setFeedback({
          ok: true,
          msg: `Correct! ${variableLabel(j, n)} has the most-negative reduced cost (${rcosts[j].toFixed(2)}). Now pick the leaving row.`,
        });
      else {
        const dantzig = variableLabel(suggestedCol, n);
        setFeedback({
          ok: true,
          msg: `${variableLabel(j, n)} is valid (reduced cost ${rcosts[j].toFixed(2)}), but Dantzig's rule prefers the most-negative: ${dantzig}.`,
        });
      }
    }
    setHoverCol(j);
  }

  function handleRowClick(i) {
    if (hoverCol === null) return;
    const { minRow } = ratioTest(T, m, hoverCol);
    if (T[i][hoverCol] <= 1e-9) {
      if (practiceMode)
        setFeedback({
          ok: false,
          msg: `Row ${i + 1} has aᵢⱼ ≤ 0 in column ${variableLabel(hoverCol, n)} — invalid pivot.`,
        });
      return;
    }
    if (practiceMode && i !== minRow)
      setFeedback({
        ok: false,
        msg: `Row ${i + 1} would give negative RHS after pivot. The min-ratio test points to row ${minRow + 1}.`,
      });
    else doPivot(i, hoverCol);
  }

  function reset() {
    setHistory([initial]);
    setStepIdx(0);
    setHoverCol(null);
    setFeedback(null);
  }

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Interactive Simplex Tableau
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        A practice tool for simplex pivoting. Click any column with a
        negative reduced cost to enter it into the basis; the ratio test
        highlights the eligible leaving rows. Click one to pivot — the
        feasible-region plot's highlighted vertex jumps to the new basic
        feasible solution. Toggle <i>practice mode</i> to get feedback
        when you pick the wrong column or row.
      </p>

      <div style={problemBox}>
        <Tex block>
          {String.raw`\begin{aligned} \max\;\; & 3 x_1 + 5 x_2 \\ \text{s.t.}\;\; & 2 x_1 + x_2 \le 8 \\ & x_1 + 3 x_2 \le 6 \\ & x_1, x_2 \ge 0 \end{aligned}`}
        </Tex>
      </div>

      <div style={{ marginBottom: 12, display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ fontSize: 13 }}>
          <input
            type="checkbox"
            checked={practiceMode}
            onChange={(e) => {
              setPracticeMode(e.target.checked);
              setFeedback(null);
            }}
          />
          &nbsp;Practice mode (validate my pivot picks)
        </label>
        <button onClick={reset} style={btn}>
          <RotateCcw size={14} /> Reset to initial tableau
        </button>
        <button
          onClick={() => setStepIdx(Math.max(0, stepIdx - 1))}
          disabled={stepIdx === 0}
          style={btn}
        >
          ← Undo pivot
        </button>
        <span style={{ fontSize: 12, fontFamily: "monospace", color: "#666" }}>
          step {stepIdx} / {history.length - 1}
        </span>
      </div>

      {feedback && (
        <div
          style={{
            marginBottom: 12,
            padding: "8px 12px",
            background: feedback.ok ? "#e8f5e9" : "#fde8e8",
            border: `1px solid ${feedback.ok ? "#7dd87d" : "#c8311c"}`,
            borderRadius: 6,
            fontSize: 13,
          }}
        >
          {feedback.msg}
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)",
          gap: 22,
          alignItems: "flex-start",
        }}
      >
        <div>
          <Tableau
            T={T}
            basis={basis}
            n={n}
            m={m}
            hoverCol={hoverCol}
            onColumnClick={handleColumnClick}
            onRowClick={handleRowClick}
            optimal={optimal}
            suggestedCol={suggestedCol}
            suggestedRow={suggestedRow}
            practiceMode={practiceMode}
          />
          <BasisStatus basis={basis} T={T} n={n} m={m} optimal={optimal} />
        </div>
        <div>
          <FeasibleRegionPlot T={T} basis={basis} n={n} m={m} A={A} b={b} />
          {optimal && <SensitivityPanel T={T} basis={basis} n={n} m={m} A={A} b={b} c={c} />}
        </div>
      </div>

      <PivotRulesPanel />
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Tableau view
// ============================================================
function Tableau({ T, basis, n, m, hoverCol, onColumnClick, onRowClick, optimal, suggestedCol, suggestedRow, practiceMode }) {
  const totalCols = n + m + 1;
  const rcosts = T[m];

  // For ratio test highlighting
  let ratioInfo = null;
  if (hoverCol !== null && hoverCol >= 0 && hoverCol < n + m) {
    ratioInfo = ratioTest(T, m, hoverCol);
  }

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 8 }}>
        Simplex tableau {optimal ? "(optimal)" : ""}
      </div>
      <table style={{ borderCollapse: "collapse", fontFamily: "monospace", fontSize: 13, width: "100%" }}>
        <thead>
          <tr>
            <th style={thLeft}>basis</th>
            {Array.from({ length: n + m }, (_, j) => {
              const isHover = j === hoverCol;
              const isSuggest = !practiceMode && j === suggestedCol;
              const negative = rcosts[j] < -1e-9;
              return (
                <th
                  key={j}
                  onClick={() => onColumnClick(j)}
                  style={{
                    ...th,
                    cursor: negative ? "pointer" : "default",
                    background: isHover ? "#f5a524" : isSuggest ? "#fff4c8" : negative ? "#e8eef5" : "#fff",
                    color: isHover ? "#1f1d1a" : "#222",
                  }}
                >
                  {variableLabel(j, n)}
                </th>
              );
            })}
            <th style={th}>RHS</th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: m }, (_, i) => {
            const r = ratioInfo?.ratios[i];
            const eligible = r && r.ok;
            const isMinRatio = r && eligible && i === ratioInfo.minRow;
            const rowBg = isMinRatio ? "#e8f5e9" : eligible ? "#fff8e1" : "transparent";
            return (
              <tr
                key={i}
                onClick={() => eligible && onRowClick(i)}
                style={{
                  background: rowBg,
                  cursor: eligible ? "pointer" : "default",
                }}
              >
                <td style={tdLabel}>{variableLabel(basis[i], n)}</td>
                {Array.from({ length: n + m }, (_, j) => {
                  const v = T[i][j];
                  const isPivot = j === hoverCol && eligible;
                  return (
                    <td
                      key={j}
                      style={{
                        ...td,
                        background: isPivot ? "#f5a524" : isMinRatio && j === hoverCol ? "#7dd87d" : "transparent",
                        color: isPivot ? "#1f1d1a" : "#222",
                        fontWeight: j === hoverCol && eligible ? 700 : 400,
                      }}
                    >
                      {fmt(v)}
                    </td>
                  );
                })}
                <td style={{ ...td, fontWeight: 700 }}>{fmt(T[i][totalCols - 1])}</td>
                {hoverCol !== null && (
                  <td style={{ ...td, color: eligible ? "#0b3da0" : "#c8311c", fontStyle: "italic" }}>
                    {eligible ? `ratio = ${r.ratio.toFixed(3)}` : `aᵢⱼ ≤ 0`}
                  </td>
                )}
              </tr>
            );
          })}
          <tr style={{ borderTop: "2px solid #444" }}>
            <td style={tdLabel}>z</td>
            {Array.from({ length: n + m }, (_, j) => (
              <td
                key={j}
                style={{
                  ...td,
                  color: rcosts[j] < -1e-9 ? "#c8311c" : "#222",
                  fontWeight: rcosts[j] < -1e-9 ? 700 : 400,
                }}
              >
                {fmt(rcosts[j])}
              </td>
            ))}
            <td style={{ ...td, fontWeight: 700 }}>{fmt(T[m][totalCols - 1])}</td>
          </tr>
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 12, color: "#555" }}>
        z-row shows negative reduced costs (column j with{" "}
        <Tex>{`-c_j + c_B^T B^{-1} A_j`}</Tex>). A negative entry means
        the corresponding variable would improve the objective if it
        entered the basis.
      </div>
    </div>
  );
}

function fmt(v) {
  if (Math.abs(v) < 1e-10) return "0";
  if (Math.abs(v - Math.round(v)) < 1e-9) return String(Math.round(v));
  return v.toFixed(3);
}

const th = {
  padding: "6px 8px",
  border: "1px solid #ccc",
  background: "#f0f0f0",
  fontWeight: 700,
};
const thLeft = { ...th, textAlign: "left" };
const td = {
  padding: "5px 8px",
  border: "1px solid #ddd",
  textAlign: "right",
};
const tdLabel = { ...td, textAlign: "left", fontWeight: 700, color: "#555" };

// ============================================================
// Basis status panel
// ============================================================
function BasisStatus({ basis, T, n, m, optimal }) {
  const totalCols = T[0].length;
  const x = Array(n + m).fill(0);
  for (let i = 0; i < m; i++) x[basis[i]] = T[i][totalCols - 1];
  return (
    <div style={{ ...panel, marginTop: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Current basic feasible solution
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 13 }}>
        <tbody>
          {Array.from({ length: n + m }, (_, j) => (
            <tr key={j}>
              <td style={{ padding: "2px 6px", color: "#555" }}>
                {variableLabel(j, n)}
              </td>
              <td
                style={{
                  padding: "2px 6px",
                  textAlign: "right",
                  color: x[j] > 1e-9 ? "#c8311c" : "#888",
                  fontWeight: x[j] > 1e-9 ? 700 : 400,
                }}
              >
                {fmt(x[j])}
              </td>
              <td style={{ padding: "2px 6px", color: "#666", fontStyle: "italic" }}>
                {x[j] > 1e-9 ? "(basic)" : "(non-basic)"}
              </td>
            </tr>
          ))}
          <tr style={{ borderTop: "2px solid #444" }}>
            <td style={{ padding: "2px 6px", fontWeight: 700 }}>z</td>
            <td style={{ padding: "2px 6px", textAlign: "right", fontWeight: 700, color: "#c8311c" }}>
              {fmt(T[m][totalCols - 1])}
            </td>
            <td>{optimal && <span style={{ color: "#1f4e3d", fontWeight: 700 }}>★ OPTIMAL</span>}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// Feasible region plot — highlight current vertex
// ============================================================
function FeasibleRegionPlot({ T, basis, n, m, A, b }) {
  const W = 480, H = 480;
  const padL = 50, padR = 16, padT = 18, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -1, xmax = 9;
  const ymin = -1, ymax = 9;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;

  // Current vertex from tableau
  const totalCols = T[0].length;
  const x1 = (() => {
    for (let i = 0; i < m; i++) if (basis[i] === 0) return T[i][totalCols - 1];
    return 0;
  })();
  const x2 = (() => {
    for (let i = 0; i < m; i++) if (basis[i] === 1) return T[i][totalCols - 1];
    return 0;
  })();

  // Vertices of the feasible region (pre-computed since A, b fixed)
  const vertices = useMemo(() => {
    const lines = [
      [A[0][0], A[0][1], b[0]],
      [A[1][0], A[1][1], b[1]],
      [1, 0, 0],
      [0, 1, 0],
    ];
    const verts = [];
    for (let i = 0; i < lines.length; i++) {
      for (let j = i + 1; j < lines.length; j++) {
        const [a1, b1, c1] = lines[i];
        const [a2, b2, c2] = lines[j];
        const det = a1 * b2 - a2 * b1;
        if (Math.abs(det) < 1e-9) continue;
        const x = (c1 * b2 - c2 * b1) / det;
        const y = (a1 * c2 - a2 * c1) / det;
        if (
          x >= -1e-9 &&
          y >= -1e-9 &&
          A[0][0] * x + A[0][1] * y <= b[0] + 1e-9 &&
          A[1][0] * x + A[1][1] * y <= b[1] + 1e-9
        ) {
          verts.push({ x: Math.max(0, x), y: Math.max(0, y) });
        }
      }
    }
    return verts;
  }, [A, b]);

  const cx = vertices.reduce((s, v) => s + v.x, 0) / vertices.length;
  const cy = vertices.reduce((s, v) => s + v.y, 0) / vertices.length;
  const polyPts = [...vertices].sort(
    (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
  );

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Feasible region — current vertex highlighted
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={ys(0)} x2={padL + chartW} y2={ys(0)} stroke="#888" />
        <line x1={xs(0)} y1={padT} x2={xs(0)} y2={padT + chartH} stroke="#888" />

        <polygon
          points={polyPts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" ")}
          fill="rgba(31, 78, 61, 0.12)"
          stroke="#1f4e3d"
          strokeWidth={1.5}
        />

        {/* All vertices */}
        {vertices.map((v, i) => (
          <circle
            key={i}
            cx={xs(v.x)}
            cy={ys(v.y)}
            r={5}
            fill="#fff"
            stroke="#1f4e3d"
            strokeWidth={1.5}
          />
        ))}

        {/* Current vertex */}
        <circle cx={xs(x1)} cy={ys(x2)} r={9} fill="#c8311c" stroke="#fff" strokeWidth={2.5} />
        <text x={xs(x1) + 12} y={ys(x2) + 4} fontSize={12} fontFamily="monospace" fill="#c8311c" fontWeight={700}>
          ({fmt(x1)}, {fmt(x2)})
        </text>

        {/* Axis labels */}
        {[0, 2, 4, 6, 8].map((v) => (
          <text key={`xl${v}`} x={xs(v)} y={padT + chartH + 14} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        {[0, 2, 4, 6, 8].map((v) => (
          <text key={`yl${v}`} x={padL - 6} y={ys(v) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        <text x={padL + chartW - 6} y={ys(0) - 6} textAnchor="end" fontSize={11} fontFamily="monospace" fill="#666">x₁</text>
        <text x={xs(0) + 8} y={padT + 12} fontSize={11} fontFamily="monospace" fill="#666">x₂</text>
      </svg>
    </div>
  );
}

// ============================================================
// Sensitivity & duality panel (only when optimal)
// ============================================================
function SensitivityPanel({ T, basis, n, m, A, b, c }) {
  const totalCols = T[0].length;
  // Dual values are reduced costs of the slack variables (with sign convention).
  // In the canonical 'min cz, Ax = b, x>=0' form with slacks added, the
  // optimal dual for constraint i is the z-row entry of the slack column.
  const pi = [];
  for (let i = 0; i < m; i++) pi.push(T[m][n + i]);
  // Reduced costs of original variables
  const rc = [];
  for (let j = 0; j < n; j++) rc.push(T[m][j]);

  return (
    <div style={{ ...panel, marginTop: 12, background: "#fff8e1", borderColor: "#f5d68d" }}>
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
        Optimal — read off duality + sensitivity from the tableau
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 13, borderCollapse: "collapse" }}>
        <tbody>
          <tr style={{ borderBottom: "1px solid #ccc" }}>
            <td style={{ padding: 4, color: "#555" }}>shadow price π₁ (constraint 1)</td>
            <td style={{ padding: 4, textAlign: "right", color: "#0b3da0", fontWeight: 700 }}>{pi[0].toFixed(4)}</td>
          </tr>
          <tr style={{ borderBottom: "1px solid #ccc" }}>
            <td style={{ padding: 4, color: "#555" }}>shadow price π₂ (constraint 2)</td>
            <td style={{ padding: 4, textAlign: "right", color: "#7a3da0", fontWeight: 700 }}>{pi[1].toFixed(4)}</td>
          </tr>
          <tr>
            <td style={{ padding: 4, color: "#555" }}>reduced cost (x₁)</td>
            <td style={{ padding: 4, textAlign: "right" }}>{rc[0].toFixed(4)}</td>
          </tr>
          <tr>
            <td style={{ padding: 4, color: "#555" }}>reduced cost (x₂)</td>
            <td style={{ padding: 4, textAlign: "right" }}>{rc[1].toFixed(4)}</td>
          </tr>
          <tr style={{ borderTop: "2px solid #444" }}>
            <td style={{ padding: 4 }}>
              dual obj <Tex>{`b^T \\pi`}</Tex>
            </td>
            <td style={{ padding: 4, textAlign: "right", fontWeight: 700, color: "#c8311c" }}>
              {(b[0] * pi[0] + b[1] * pi[1]).toFixed(4)}
            </td>
          </tr>
        </tbody>
      </table>
      <div style={{ marginTop: 8, fontSize: 13, color: "#3d2f00", lineHeight: 1.5 }}>
        The slack-column entries of the z-row ARE the optimal dual values
        — that's how simplex solvers extract <Tex>{`\\pi`}</Tex> for free.
        Strong duality: primal obj = dual obj at optimum (verify above).
      </div>
    </div>
  );
}

// ============================================================
// Pivot rules panel
// ============================================================
function PivotRulesPanel() {
  return (
    <div style={{ ...panel, marginTop: 18 }}>
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
        Pivot rules — reference card
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18, fontSize: 13 }}>
        <div>
          <b>Entering variable (column):</b>
          <ul style={{ paddingLeft: 22, margin: "4px 0", lineHeight: 1.6 }}>
            <li><b>Dantzig:</b> most-negative reduced cost (default — fast to compute, few iterations).</li>
            <li><b>Bland's rule:</b> smallest-index column with negative reduced cost (anti-cycling).</li>
            <li><b>Steepest edge:</b> normalize reduced cost by direction-norm (fewer iters, expensive per-step).</li>
            <li><b>Devex:</b> approximation of steepest edge (the default in CPLEX/Gurobi).</li>
          </ul>
        </div>
        <div>
          <b>Leaving variable (row):</b>
          <ul style={{ paddingLeft: 22, margin: "4px 0", lineHeight: 1.6 }}>
            <li><b>Min-ratio test:</b> over rows with{" "}
              <Tex>{`a_{ij} > 0`}</Tex>, pick the row with smallest{" "}
              <Tex>{`b_i / a_{ij}`}</Tex>. This keeps the next BFS feasible.
            </li>
            <li><b>Bland's rule:</b> smallest-index basic variable (with valid pivot).</li>
            <li><b>Lex / lexicographic:</b> tie-break by lex order on the row — guarantees no cycling.</li>
          </ul>
        </div>
      </div>
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
          <b>Geometric intuition.</b> Each tableau corresponds to a vertex
          of the feasible polytope. Pivoting moves to an ADJACENT vertex
          along an edge of the polytope. The simplex method walks
          vertex-to-vertex toward the optimum.
        </li>
        <li>
          <b>Why the ratio test.</b> Increasing the entering variable
          forces basic variables to change; the min-ratio bounds how far
          you can go before a basic variable hits zero. The minimum
          determines which basic variable becomes non-basic.
        </li>
        <li>
          <b>Cycling and degeneracy.</b> If two ratios tie (degenerate),
          some pivot rules can cycle indefinitely. Bland's rule and the
          lex rule prevent cycling but are slower in practice.
        </li>
        <li>
          <b>Dual simplex.</b> Same tableau, but pivoted to maintain dual
          feasibility instead of primal. Useful when adding cuts
          (post-LP-modification) — the previous optimal basis is now
          infeasible, and dual simplex restores it cheaply.
        </li>
        <li>
          <b>Why this matters in MIP.</b> Branch-and-bound runs ONE LP per
          node. Warm-starting via dual simplex (using the parent's
          basis) is what makes B&B tractable — restarting primal simplex
          would be 10–100× slower.
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
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid #ccc",
  background: "#f7f7f7",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: 13,
};
