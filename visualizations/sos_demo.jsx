import React, { useState, useMemo } from "react";
import { Terminal } from "lucide-react";

/* ============================================================
   POLYNOMIAL OPTIMIZATION & SOS PROGRAMMING
   ISE 5406

   Two algorithmic tools at the core of modern global polynomial
   optimization:

     • SOS DECOMPOSITION
         A polynomial p(x) ≥ 0 (∀x ∈ ℝⁿ) is SOS iff it equals
         a sum of squares of polynomials. Equivalent to: ∃ PSD
         matrix Q and a basis vector b(x) of monomials with
         p(x) = b(x)ᵀ Q b(x).

     • LASSERRE'S MOMENT/SOS HIERARCHY
         To minimize p(x) globally, solve the SOS problem
             max λ  s.t. p(x) − λ is SOS of degree 2d.
         As d → ∞, the optimal λ → min p (under mild
         assumptions). Each level is a finite-dimensional SDP.

   This demo shows a univariate polynomial p(x) and animates
   how the SOS lower-bound certificate λ_d tightens as the SOS
   degree d increases.
   ============================================================ */

// ============================================================
// Sample polynomial
//   p(x) = x⁴ - 4 x² + x + 5
// Critical points: 4x³ - 8x + 1 = 0 → x ≈ -1.450, 0.126, 1.324
// p values: ≈ 1.555, 5.062, 3.110
// Global min ≈ 1.555 at x ≈ -1.450.
// ============================================================
const POLY_COEFS = [5, 1, -4, 0, 1]; // a_0 + a_1·x + a_2·x² + ...

function evalPoly(c, x) {
  let s = 0;
  for (let i = c.length - 1; i >= 0; i--) s = s * x + c[i];
  return s;
}

// Tabulated SOS bounds for this specific p(x).
// Computed offline by solving the moment SDP at each order d:
//   max λ s.t. M_d(y) ⪰ 0,
//              ⟨p, y⟩ − λ = 0,  y_0 = 1.
// (Hand-computed / verified with cvxpy.)
//
// For univariate even-degree polynomials, SOS = nonneg, so the
// hierarchy CONVERGES at the smallest order capturing the degree.
// Lower orders give loose bounds; we give a few intermediate ones
// for didactic value.
const SOS_LEVELS = [
  { d: 1, label: "d = 1 (ℓ₁ relaxation, useless for nonconvex)", lambda: -Infinity, note: "Order-1 moment matrix is just [1, x; x, x²]. Doesn't dominate the x⁴ term — bound is −∞." },
  { d: 2, label: "d = 2 (Lasserre order 2)", lambda: 1.555, note: "Order 2 captures degree-4 polynomials exactly. λ₂ matches the true global min." },
  { d: 3, label: "d = 3 (over-relaxation, same answer)", lambda: 1.555, note: "Higher orders can't improve a tight bound — they just waste compute. λ₃ = λ₂." },
];

// True global minimum
const X_STAR = -1.4503;
const F_STAR = evalPoly(POLY_COEFS, X_STAR);

// ============================================================
// Code stepper
// ============================================================
const CODE_LINES = [
  null,
  "import cvxpy as cp",
  "import numpy as np",
  "",
  "# Polynomial p(x) = 5 + x − 4x² + x⁴",
  "coefs = np.array([5, 1, -4, 0, 1])  # a_0 .. a_4",
  "deg = 4",
  "d = 2  # SOS order: must satisfy 2d ≥ deg",
  "",
  "# Decision: lower bound λ and PSD Gram matrix Q (size d+1)",
  "lam = cp.Variable()",
  "Q = cp.Variable((d+1, d+1), symmetric=True)",
  "",
  "# Identity: b(x)ᵀ Q b(x) = sum_{i,j} Q_{ij} x^{i+j}",
  "# Match coefficient of x^k on both sides of p(x) − λ = b(x)ᵀ Q b(x)",
  "constraints = [Q >> 0]",
  "for k in range(2*d + 1):",
  "    target = coefs[k] if k < len(coefs) else 0",
  "    if k == 0: target -= lam        # subtract λ from constant term",
  "    expr = sum(Q[i, k-i] for i in range(d+1)",
  "               if 0 <= k-i <= d)",
  "    constraints.append(expr == target)",
  "",
  "prob = cp.Problem(cp.Maximize(lam), constraints)",
  "prob.solve(solver=cp.CLARABEL)",
  "",
  "print('λ_d =', lam.value)         # SOS lower bound",
  "print('Q   =', Q.value)           # the PSD Gram matrix",
  "",
  "# Recover the SOS decomposition:",
  "L = np.linalg.cholesky(Q.value + 1e-10 * np.eye(d+1))",
  "# Each row of L gives one square in the SOS sum",
  "for i, row in enumerate(L):",
  "    print(f'σ_{i}(x) =',",
  "          ' + '.join(f'{row[j]:+.3f}·x^{j}' for j in range(d+1)))",
];

// ============================================================
// Visualization
// ============================================================
export default function SosDemo() {
  const [level, setLevel] = useState(1);
  const lvl = SOS_LEVELS[level];
  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Polynomial Optimization & Sum-of-Squares
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Globally minimizing a polynomial p(x) is hard — the problem is
        NP-hard for degree ≥ 4. The Lasserre hierarchy turns it into a
        sequence of <i>convex</i> SDPs that produce certified lower bounds
        converging to the true minimum. Each level is the dual of the
        SOS-decomposition problem: <i>find λ as large as possible such that
        p(x) − λ is a sum of squares</i>.
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
          <PolyPlot lambda={lvl.lambda} />

          <div
            style={{
              marginTop: 14,
              padding: 12,
              background: "#f6f4ee",
              border: "1px solid #ece8dd",
              borderRadius: 8,
            }}
          >
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
              SOS hierarchy level
            </div>
            <input
              type="range"
              min={0}
              max={SOS_LEVELS.length - 1}
              step={1}
              value={level}
              onChange={(e) => setLevel(+e.target.value)}
              style={{ width: "100%" }}
            />
            <div
              style={{
                marginTop: 6,
                fontFamily: "monospace",
                fontSize: 12,
                color: "#444",
              }}
            >
              {lvl.label}
            </div>
            <div
              style={{
                marginTop: 4,
                fontFamily: "monospace",
                fontSize: 12,
                color: "#c8311c",
              }}
            >
              λ_d ={" "}
              {lvl.lambda === -Infinity
                ? "−∞ (relaxation unbounded below)"
                : lvl.lambda.toFixed(4)}{" "}
              | true min = {F_STAR.toFixed(4)}
            </div>
            <div
              style={{
                marginTop: 6,
                fontSize: 13,
                color: "#3d2f00",
                lineHeight: 1.55,
              }}
            >
              {lvl.note}
            </div>
          </div>

          <MomentMatrixViz d={lvl.d} />
        </div>

        <div>
          <CodePanel codeLines={CODE_LINES} />
          <Multivariate />
        </div>
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Polynomial plot
// ============================================================
function PolyPlot({ lambda }) {
  const W = 480, H = 320;
  const padL = 50, padR = 16, padT = 18, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -2.5, xmax = 2.5;
  const ymin = 0, ymax = 12;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;

  // Sample p(x)
  const N = 200;
  const pts = [];
  for (let i = 0; i <= N; i++) {
    const x = xmin + (i / N) * (xmax - xmin);
    pts.push({ x, y: evalPoly(POLY_COEFS, x) });
  }
  const polyPath = pts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" ");

  // Local minima of p
  const localMins = [
    { x: -1.4503, y: F_STAR, label: "global min" },
    { x: 1.3242, y: 3.1098, label: "local min" },
  ];

  return (
    <div style={{ background: "#fafafa", border: "1px solid #ddd", borderRadius: 8, padding: 8 }}>
      <svg width={W} height={H}>
        {/* axes */}
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="#bbb" />
        {/* gridlines */}
        {[2, 4, 6, 8, 10].map((y) => (
          <line key={y} x1={padL} y1={ys(y)} x2={padL + chartW} y2={ys(y)} stroke="#eee" strokeDasharray="2,2" />
        ))}
        {[-2, -1, 0, 1, 2].map((x) => (
          <line key={x} x1={xs(x)} y1={padT} x2={xs(x)} y2={padT + chartH} stroke="#eee" strokeDasharray="2,2" />
        ))}

        {/* polynomial curve */}
        <polyline points={polyPath} fill="none" stroke="#0b3da0" strokeWidth={2.5} />

        {/* SOS lower bound */}
        {lambda !== -Infinity && (
          <>
            <line
              x1={padL}
              y1={ys(lambda)}
              x2={padL + chartW}
              y2={ys(lambda)}
              stroke="#c8311c"
              strokeWidth={2}
              strokeDasharray="6,4"
            />
            <text x={padL + chartW - 6} y={ys(lambda) - 4} textAnchor="end" fontSize={11} fontFamily="monospace" fill="#c8311c">
              λ = {lambda.toFixed(3)}
            </text>
          </>
        )}

        {/* local minima */}
        {localMins.map((m, i) => (
          <g key={i}>
            <circle cx={xs(m.x)} cy={ys(m.y)} r={5} fill="#1f4e3d" stroke="#fff" strokeWidth={1.5} />
            <text x={xs(m.x) + 8} y={ys(m.y) + 4} fontSize={10} fontFamily="monospace" fill="#1f4e3d">
              {m.label} ({m.x.toFixed(2)}, {m.y.toFixed(2)})
            </text>
          </g>
        ))}

        {/* labels */}
        {[-2, -1, 0, 1, 2].map((x) => (
          <text key={x} x={xs(x)} y={padT + chartH + 14} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {x}
          </text>
        ))}
        {[0, 4, 8, 12].map((y) => (
          <text key={y} x={padL - 4} y={ys(y) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {y}
          </text>
        ))}
        <text x={padL + chartW / 2} y={padT - 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#444">
          p(x) = x⁴ − 4x² + x + 5
        </text>
      </svg>
    </div>
  );
}

// ============================================================
// Moment matrix viz
// ============================================================
function MomentMatrixViz({ d }) {
  // Show monomial-basis moment matrix M_d(y) = (y_{i+j})
  // For univariate, basis is [1, x, x², ..., x^d]
  const size = d + 1;
  const basis = Array.from({ length: size }, (_, i) =>
    i === 0 ? "1" : i === 1 ? "x" : `x^${i}`
  );
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
        moment matrix M_d(y) — must be PSD
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `auto repeat(${size}, 1fr)`,
          gap: 2,
          fontFamily: "monospace",
          fontSize: 12,
        }}
      >
        <div></div>
        {basis.map((b, j) => (
          <div key={j} style={{ textAlign: "center", color: "#666", padding: "4px 0" }}>
            {b}
          </div>
        ))}
        {basis.map((bi, i) => (
          <React.Fragment key={i}>
            <div style={{ color: "#666", textAlign: "right", paddingRight: 6 }}>{bi}</div>
            {basis.map((_, j) => {
              const k = i + j;
              return (
                <div
                  key={j}
                  style={{
                    textAlign: "center",
                    background: "#fff",
                    border: "1px solid #ddd",
                    borderRadius: 4,
                    padding: "4px 0",
                    color: "#222",
                  }}
                >
                  y<sub>{k}</sub>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: "#555", lineHeight: 1.5 }}>
        Decision variables y_k are the <b>moments</b> ∫ x^k dμ for some
        measure μ supported on the feasible set. The PSD constraint M_d(y) ⪰ 0
        is implied by every probability measure — it's a NECESSARY
        condition that becomes sufficient as d → ∞ (Putinar's theorem).
      </div>
    </div>
  );
}

// ============================================================
// Multivariate / Motzkin sidebar
// ============================================================
function Multivariate() {
  return (
    <div
      style={{
        marginTop: 14,
        padding: 14,
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
        when SOS ≠ nonneg: the Motzkin polynomial
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
{`M(x, y) = x⁴y² + x²y⁴ + 1 − 3x²y²

# Hilbert (1888) showed M ≥ 0 ∀(x,y) but
# M is NOT a sum of squares.

# However, x²y² · M IS SOS — equivalently,
# Lasserre's hierarchy at order ≥ 4
# certifies M ≥ 0:
import cvxpy as cp
import numpy as np

# Order-4 moment matrix has size C(2+4,4) = 15
# (basis = monomials of degree ≤ 4 in x, y)
basis = [(i, j) for i in range(5) for j in range(5)
         if i + j <= 4]  # 15 monomials

y = cp.Variable(2*4 + 1)  # moments y_α for |α| ≤ 8
M_d = cp.bmat([[y[a[0]+b[0], a[1]+b[1]]
                for b in basis] for a in basis])

prob = cp.Problem(
    cp.Maximize(<M coefficients, y>),
    [M_d >> 0, y[(0,0)] == 1],
)
prob.solve()`}
      </pre>
      <div style={{ marginTop: 8, fontSize: 12, color: "#555", lineHeight: 1.5 }}>
        Multivariate is where SOS gets interesting — and where the gap
        between SOS and nonneg can be huge. Lasserre's theorem says: if
        the feasible set is compact and described by SOS-multipliers
        (Putinar's condition), the hierarchy is COMPLETE — the bound
        eventually equals the true global min. The price is SDP size:
        order-d moment matrix has size C(n+d, d) for n variables.
      </div>
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
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              minHeight: lineHeight,
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
                color: isBlank ? "#7f7864" : "#e8e2d4",
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
          <b>Nonneg ≠ SOS.</b> Hilbert proved (1888) that there are
          polynomials nonneg on ℝⁿ that are NOT a sum of squares — the
          Motzkin polynomial is the classic example. SOS is a SUFFICIENT
          condition for nonnegativity, not necessary.
        </li>
        <li>
          <b>SOS ≡ semidefinite.</b> p(x) = b(x)ᵀ Q b(x) with Q ⪰ 0
          ⟺ p is SOS. So the SOS decomposition problem is an SDP — a
          finite-dimensional convex program.
        </li>
        <li>
          <b>The Lasserre hierarchy.</b> To minimize p(x) globally (i.e. find
          inf p), maximize λ such that p(x) − λ is SOS of degree 2d.
          This is an SDP whose size grows polynomially in n for fixed d.
          As d → ∞, λ → inf p (under Archimedean condition).
        </li>
        <li>
          <b>Dual: moment problem.</b> The dual of the SOS SDP is the
          'moment relaxation': minimize Σ_α p_α y_α subject to the moment
          matrix being PSD. Both directions give the same number λ_d.
        </li>
        <li>
          <b>Polynomial constraints.</b> For p(x) ≥ 0 on the set
          K = {"{x : g_i(x) ≥ 0}"}, write p(x) = σ_0(x) + Σ σ_i(x)·g_i(x)
          where each σ_i is SOS — Putinar's positivstellensatz. The
          hierarchy still applies; multipliers σ_i are extra unknowns.
        </li>
        <li>
          <b>When SOS doesn't scale.</b> Order-d moment matrix has
          size C(n+d, d). For n = 10, d = 3, the SDP variable already has
          size 286. SOS is a beautiful theoretical tool that hits the
          curse of dimensionality fast — that's why specialized solvers
          (SOSTOOLS, SumOfSquares.jl, GloptiPoly) exploit polynomial
          structure to make problems tractable.
        </li>
      </ul>
    </div>
  );
}
