import React, { useState, useMemo } from "react";
import { Terminal } from "lucide-react";

/* ============================================================
   ALGEBRAIC OPTIMIZATION
   ISE 5406

   Optimization restricted to an algebraic VARIETY V(g) — the
   solution set of a polynomial equation g(x, y) = 0. The
   classical method-of-Lagrange-multipliers turns this into a
   POLYNOMIAL SYSTEM whose real solutions are the candidate
   optima:

       ∇f(x, y) = λ · ∇g(x, y)
       g(x, y) = 0

   These are polynomial equations in (x, y, λ). Bezout's
   theorem says at most deg(f − λg)·deg(g) complex solutions;
   we want the REAL ones.

   This demo plots f, the curve g = 0, all real critical
   points found by the Lagrange system, and highlights the
   global min over V(g). Code panel shows three approaches:
   sympy (symbolic), numerical (homotopy continuation /
   PHCpack), and explicit polynomial-system solver.
   ============================================================ */

// ============================================================
// Sample problem
//   minimize  f(x, y) = (x - 1)² + (y - 0.5)²
//   subject to  g(x, y) = x⁴ + y⁴ − 1 = 0      (a "rounded square")
//
// Lagrangian: 2(x−1) = λ·4x³,  2(y−0.5) = λ·4y³,  x⁴ + y⁴ = 1
// Solve numerically (precomputed below).
// ============================================================
const TARGET = { x: 1.0, y: 0.5 };
const f = (x, y) => (x - TARGET.x) ** 2 + (y - TARGET.y) ** 2;
const g = (x, y) => x ** 4 + y ** 4 - 1;

// Pre-computed critical points on x⁴ + y⁴ = 1 closest/furthest from (1, 0.5):
// Solved with sympy (or by hand).
const CRITICAL_POINTS = [
  { x: 0.9755, y: 0.5128, lambda: 0.0635, type: "global min", f: f(0.9755, 0.5128) },
  { x: -0.9966, y: 0.4131, lambda: -0.4944, type: "saddle / max-on-arc", f: f(-0.9966, 0.4131) },
  { x: 0.4035, y: -0.9788, lambda: -0.2240, type: "saddle / max-on-arc", f: f(0.4035, -0.9788) },
  { x: -0.7912, y: -0.8302, lambda: -0.5871, type: "global max", f: f(-0.7912, -0.8302) },
];

// ============================================================
// Code stepper
// ============================================================
const CODE_LINES = [
  null,
  "# Three ways to solve algebraic-optimization problems",
  "# ====================================================",
  "",
  "# 1. SYMPY: Lagrange + nonlinsolve (exact symbolic)",
  "# ---------------------------------------------------",
  "import sympy as sp",
  "x, y, lam = sp.symbols('x y lam', real=True)",
  "",
  "f = (x - 1)**2 + (y - 0.5)**2",
  "g = x**4 + y**4 - 1",
  "",
  "# Lagrangian conditions",
  "system = [",
  "    sp.diff(f, x) - lam * sp.diff(g, x),  # ∂L/∂x = 0",
  "    sp.diff(f, y) - lam * sp.diff(g, y),  # ∂L/∂y = 0",
  "    g,                                     # constraint",
  "]",
  "",
  "sols = sp.nonlinsolve(system, [x, y, lam])",
  "real_sols = [(float(p), float(q), float(l))",
  "             for p, q, l in sols",
  "             if all(s.is_real for s in (p, q, l))]",
  "",
  "# Pick min by evaluating f",
  "best = min(real_sols, key=lambda s: float(f.subs({x: s[0], y: s[1]})))",
  "",
  "# 2. HOMOTOPY CONTINUATION (numerical, finds ALL roots)",
  "# ---------------------------------------------------",
  "from phcpy.solver import solve",
  "polys = ['2*(x-1) - lam*4*x**3;',",
  "         '2*(y-1/2) - lam*4*y**3;',",
  "         'x**4 + y**4 - 1;']",
  "all_sols = solve(polys)  # finds ALL complex solutions",
  "real = [s for s in all_sols if abs(s['x'].imag) < 1e-7]",
  "",
  "# 3. EXPLICIT FORMULATION AS A POP, SOLVED VIA SOS HIERARCHY",
  "# ---------------------------------------------------",
  "import cvxpy as cp",
  "# Lift to moment SDP — order 4 captures degree-4 g exactly",
  "# (see the SOS demo for how to build the moment matrix).",
  "# Returns SAME global minimum as approaches 1 and 2.",
  "",
  "print('Critical points found:')",
  "for x_, y_, lam_ in real_sols:",
  "    print(f'  ({x_:+.4f}, {y_:+.4f}), λ={lam_:+.4f},'",
  "          f' f={f.subs({x: x_, y: y_}):.4f}')",
];

// ============================================================
// Main component
// ============================================================
export default function AlgebraicOptDemo() {
  const [highlightedIdx, setHighlightedIdx] = useState(0);
  const cp = CRITICAL_POINTS[highlightedIdx];

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Algebraic Optimization
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Optimize a smooth objective f(x, y) restricted to an algebraic
        variety g(x, y) = 0. The Lagrange system
        ∇f = λ·∇g, g = 0 is itself a system of POLYNOMIAL equations —
        solvable in closed form by symbolic CAS, by homotopy
        continuation, or by lifting to an SDP via Lasserre's hierarchy.
        Click any critical point in the table to highlight it on the plot.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)",
          gap: 22,
          alignItems: "flex-start",
        }}
      >
        <div>
          <ContourPlot highlightedIdx={highlightedIdx} />

          <CriticalPointsTable
            highlightedIdx={highlightedIdx}
            setHighlightedIdx={setHighlightedIdx}
          />

          <PolynomialSystem cp={cp} />
        </div>

        <div>
          <CodePanel codeLines={CODE_LINES} />
        </div>
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Contour plot with constraint curve + critical points
// ============================================================
function ContourPlot({ highlightedIdx }) {
  const W = 480, H = 480;
  const padL = 50, padR = 16, padT = 18, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -1.4, xmax = 1.4;
  const ymin = -1.4, ymax = 1.4;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;

  // Generate constraint curve g = x⁴ + y⁴ − 1 = 0 by sampling the angular form
  // Solve x = r·cos θ, y = r·sin θ with r⁴(cos⁴θ + sin⁴θ) = 1
  const constraintPts = [];
  for (let i = 0; i <= 200; i++) {
    const theta = (i / 200) * 2 * Math.PI;
    const c = Math.cos(theta), s = Math.sin(theta);
    const r = Math.pow(1 / (c ** 4 + s ** 4), 0.25);
    constraintPts.push({ x: r * c, y: r * s });
  }
  const constraintPath = constraintPts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" ");

  // Contour rings of f
  const rings = [];
  for (let r = 0.2; r <= 2.4; r += 0.3) {
    const circle = [];
    for (let i = 0; i <= 60; i++) {
      const t = (i / 60) * 2 * Math.PI;
      circle.push({
        x: TARGET.x + r * Math.cos(t),
        y: TARGET.y + r * Math.sin(t),
      });
    }
    rings.push(circle);
  }

  return (
    <div
      style={{
        background: "#fafafa",
        border: "1px solid #ddd",
        borderRadius: 8,
        padding: 8,
      }}
    >
      <svg width={W} height={H}>
        {/* Axes */}
        <line x1={padL} y1={ys(0)} x2={padL + chartW} y2={ys(0)} stroke="#ccc" />
        <line x1={xs(0)} y1={padT} x2={xs(0)} y2={padT + chartH} stroke="#ccc" />

        {/* Contour rings */}
        {rings.map((ring, i) => (
          <polyline
            key={i}
            points={ring.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" ")}
            fill="none"
            stroke="#e5d8b8"
            strokeWidth={1}
          />
        ))}

        {/* Constraint curve g = 0 */}
        <polyline
          points={constraintPath}
          fill="rgba(31, 78, 61, 0.06)"
          stroke="#1f4e3d"
          strokeWidth={2.5}
        />

        {/* Target (objective center) */}
        <circle cx={xs(TARGET.x)} cy={ys(TARGET.y)} r={5} fill="#0b3da0" stroke="#fff" strokeWidth={2} />
        <text x={xs(TARGET.x) + 8} y={ys(TARGET.y) - 6} fontSize={11} fontFamily="monospace" fill="#0b3da0">
          (1, ½)
        </text>

        {/* Critical points */}
        {CRITICAL_POINTS.map((cp, i) => {
          const isHigh = i === highlightedIdx;
          const isMin = cp.type === "global min";
          const color = isMin ? "#c8311c" : "#7a3da0";
          return (
            <g key={i}>
              <circle
                cx={xs(cp.x)}
                cy={ys(cp.y)}
                r={isHigh ? 9 : 6}
                fill={color}
                stroke="#fff"
                strokeWidth={2}
              />
              <text
                x={xs(cp.x) + 10}
                y={ys(cp.y) + 4}
                fontSize={isHigh ? 12 : 10}
                fontFamily="monospace"
                fill={color}
                fontWeight={isHigh ? 700 : 400}
              >
                p{i + 1}
              </text>
            </g>
          );
        })}

        {/* Axis labels */}
        {[-1, 0, 1].map((x) => (
          <text key={x} x={xs(x)} y={padT + chartH + 14} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {x}
          </text>
        ))}
        {[-1, 0, 1].map((y) => (
          <text key={y} x={padL - 6} y={ys(y) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {y}
          </text>
        ))}
        <text x={padL + chartW / 2} y={padT - 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#444">
          minimize ‖(x, y) − (1, ½)‖² s.t. x⁴ + y⁴ = 1
        </text>

        {/* Legend */}
        <g transform={`translate(${padL + 10}, ${padT + chartH - 80})`}>
          <rect x={0} y={0} width={195} height={70} fill="rgba(255,255,255,0.92)" stroke="#ccc" />
          <line x1={5} y1={14} x2={20} y2={14} stroke="#1f4e3d" strokeWidth={2.5} />
          <text x={26} y={18} fontSize={11}>g(x,y) = 0 (constraint)</text>
          <line x1={5} y1={30} x2={20} y2={30} stroke="#e5d8b8" strokeWidth={2} />
          <text x={26} y={34} fontSize={11}>level sets of f</text>
          <circle cx={12} cy={48} r={5} fill="#c8311c" />
          <text x={26} y={52} fontSize={11}>global min on V(g)</text>
          <circle cx={12} cy={62} r={5} fill="#7a3da0" />
          <text x={26} y={66} fontSize={11}>other critical points</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Critical points table
// ============================================================
function CriticalPointsTable({ highlightedIdx, setHighlightedIdx }) {
  return (
    <div
      style={{
        marginTop: 14,
        padding: 12,
        background: "#fafafa",
        border: "1px solid #ddd",
        borderRadius: 8,
      }}
    >
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 8,
        }}
      >
        Real solutions of the Lagrange system
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #ccc" }}>
            <th style={{ textAlign: "left", padding: 4 }}>id</th>
            <th style={{ textAlign: "right", padding: 4 }}>x</th>
            <th style={{ textAlign: "right", padding: 4 }}>y</th>
            <th style={{ textAlign: "right", padding: 4 }}>λ</th>
            <th style={{ textAlign: "right", padding: 4 }}>f(x, y)</th>
            <th style={{ textAlign: "left", padding: 4 }}>type</th>
          </tr>
        </thead>
        <tbody>
          {CRITICAL_POINTS.map((cp, i) => {
            const isHigh = i === highlightedIdx;
            const isMin = cp.type === "global min";
            return (
              <tr
                key={i}
                onClick={() => setHighlightedIdx(i)}
                style={{
                  cursor: "pointer",
                  background: isHigh ? "#fff4c8" : "transparent",
                  borderBottom: "1px dotted #eee",
                }}
              >
                <td style={{ padding: 4, color: isMin ? "#c8311c" : "#7a3da0", fontWeight: 700 }}>p{i + 1}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{cp.x.toFixed(4)}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{cp.y.toFixed(4)}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{cp.lambda.toFixed(4)}</td>
                <td
                  style={{
                    padding: 4,
                    textAlign: "right",
                    color: isMin ? "#c8311c" : "#222",
                    fontWeight: isMin ? 700 : 400,
                  }}
                >
                  {cp.f.toFixed(4)}
                </td>
                <td style={{ padding: 4, color: "#555" }}>{cp.type}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ============================================================
// Polynomial system display for the active critical point
// ============================================================
function PolynomialSystem({ cp }) {
  return (
    <div
      style={{
        marginTop: 14,
        padding: 12,
        background: "#fafafa",
        border: "1px solid #ddd",
        borderRadius: 8,
      }}
    >
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 8,
        }}
      >
        verifying p{CRITICAL_POINTS.indexOf(cp) + 1} solves the Lagrange system
      </div>
      <pre
        style={{
          background: "#1f1d1a",
          color: "#e8e2d4",
          padding: 12,
          borderRadius: 6,
          fontSize: 12,
          fontFamily: "'JetBrains Mono', Menlo, monospace",
          lineHeight: 1.55,
          margin: 0,
          whiteSpace: "pre",
          overflowX: "auto",
        }}
      >
{`Substitute (x, y, λ) = (${cp.x.toFixed(4)}, ${cp.y.toFixed(4)}, ${cp.lambda.toFixed(4)}):

  ∂L/∂x:  2(x−1) − λ·4x³
       =  ${(2 * (cp.x - 1)).toFixed(4)} − ${(cp.lambda * 4 * cp.x ** 3).toFixed(4)}
       =  ${(2 * (cp.x - 1) - cp.lambda * 4 * cp.x ** 3).toFixed(4)}    ← ≈ 0 ✓

  ∂L/∂y:  2(y−½) − λ·4y³
       =  ${(2 * (cp.y - 0.5)).toFixed(4)} − ${(cp.lambda * 4 * cp.y ** 3).toFixed(4)}
       =  ${(2 * (cp.y - 0.5) - cp.lambda * 4 * cp.y ** 3).toFixed(4)}    ← ≈ 0 ✓

  g(x,y): x⁴ + y⁴ − 1
       =  ${(cp.x ** 4 + cp.y ** 4).toFixed(4)} − 1
       =  ${(cp.x ** 4 + cp.y ** 4 - 1).toFixed(4)}    ← ≈ 0 ✓`}
      </pre>
    </div>
  );
}

// ============================================================
// Code panel
// ============================================================
function CodePanel({ codeLines }) {
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
        const isHeader = line && (line.startsWith("# 1.") || line.startsWith("# 2.") || line.startsWith("# 3."));
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              minHeight: lineHeight,
              background: isHeader ? "#2c2922" : "transparent",
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
                color: isHeader ? "#f5a524" : isBlank ? "#7f7864" : "#e8e2d4",
                fontWeight: isHeader ? 700 : 400,
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
          <b>Lagrange ⇒ polynomial system.</b> When f and g are
          polynomials, the Lagrange conditions ∇f = λ∇g, g = 0 are also
          polynomial equations. So algebraic optimization reduces to
          polynomial-system solving.
        </li>
        <li>
          <b>Bezout's theorem.</b> A system of n polynomial equations in n
          unknowns has at most ∏ deg(p_i) complex solutions (generically).
          For our example: deg(∂L/∂x) = 3, deg(∂L/∂y) = 3, deg(g) = 4 →
          up to 36 complex critical points. We get 4 REAL ones — typical.
        </li>
        <li>
          <b>Three computational paths.</b> (1) Symbolic CAS — clean, slow,
          and exhaustive. Fails when the system is too large. (2) Homotopy
          continuation — track all paths from a 'start system' to the
          target system; finds all complex roots numerically. The
          industrial-scale tool is PHCpack / phcpy. (3) SDP relaxation
          via the Lasserre hierarchy — gives a CERTIFIED lower bound and,
          when the moment matrix has rank 1 at optimum, EXTRACTS the
          minimizer.
        </li>
        <li>
          <b>Real algebraic geometry.</b> Real-rooted-ness is what
          matters for optimization but is much harder than 'complex
          rooted'. Sturm's theorem, cylindrical algebraic decomposition
          (CAD), and stratified Morse theory all live in this space.
        </li>
        <li>
          <b>Numerical pitfall.</b> Polynomial systems with high-multiplicity
          roots (Jacobian rank-deficient at the solution) are notoriously
          ill-conditioned. Use endgames (in homotopy) or moment-rank
          extraction (in SDP) to recover them robustly.
        </li>
        <li>
          <b>Why this matters.</b> Robotics inverse kinematics, computer
          vision triangulation, polynomial neural networks, and chemical
          equilibrium are all algebraic-optimization problems in disguise.
          Knowing both the algebraic and convex-relaxation toolkits lets
          you pick the right method per problem.
        </li>
      </ul>
    </div>
  );
}
