import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Terminal,
  Package,
} from "lucide-react";

/* ============================================================
   CVXPY — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   Four problems: LP, QP, SOCP, SDP. Each shows how the same
   "build a Variable, build constraints, build a Problem, solve"
   pattern adapts as the problem class changes.

   The Python here is real and runs as-is with cvxpy installed.
   The solver output is hardcoded so the demo runs in the browser.
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_LP = {
  key: "lp",
  name: "LP",
  kind: "LP",
  blurb:
    "Plain linear program. Five inequalities, two variables, linear objective. CVXPY infers it's an LP and dispatches to ECOS or CLARABEL automatically.",
  formula:
    "min  −1.05·x₁ − x₂   s.t.   x₁,x₂ ≥ 0,  x₁+x₂ ≤ 2,  x₁ ≤ 1.5,  2x₁+x₂ ≤ 3",
  code: [
    null,
    "import cvxpy as cp",
    "import numpy as np",
    "",
    "c = np.array([-1.05, -1])",
    "A = np.array([",
    "    [ 1,  1],",
    "    [ 1,  0],",
    "    [ 2,  1],",
    "])",
    "b = np.array([2, 1.5, 3])",
    "",
    "x = cp.Variable(2, nonneg=True)",
    "",
    "objective   = cp.Minimize(c @ x)",
    "constraints = [A @ x <= b]",
    "prob = cp.Problem(objective, constraints)",
    "",
    "prob.solve()",
    "",
    "print('status :', prob.status)",
    "print('value  :', prob.value)",
    "print('x*     :', x.value)",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "cvxpy", alias: "cp" }, note: "CVXPY is the modeling layer. It builds a problem in disciplined convex form, then ships it to a backend solver." },
    { line: 2, kind: "import", payload: { name: "numpy", alias: "np" }, note: "Plain NumPy for data. CVXPY accepts NumPy arrays directly in expressions like 'A @ x'." },
    { line: 4, kind: "raw_data", payload: { label: "c", value: "[-1.05, -1.0]", desc: "objective vector" }, note: "Negative coefficients because we want to maximize x₁ + x₂ but CVXPY uses Minimize." },
    { line: 5, kind: "raw_data", payload: { label: "A", value: "(3, 2) matrix", desc: "constraint LHS" }, note: "Constraint matrix. Each row is one inequality." },
    { line: 10, kind: "raw_data", payload: { label: "b", value: "[2.0, 1.5, 3.0]", desc: "constraint RHS" } },
    { line: 12, kind: "add_var", payload: { name: "x", shape: "(2,)", attrs: ["nonneg"] }, note: "Decision variable. nonneg=True attaches the bound x ≥ 0 directly to the variable — no need to add a separate constraint for it." },
    { line: 14, kind: "set_objective", payload: { sense: "Minimize", expr: "c @ x" }, note: "DCP-compliant linear expression. The '@' operator builds a CVXPY expression, NOT a numpy array — there's no x.value yet." },
    { line: 15, kind: "add_constraint", payload: { expr: "A @ x <= b", kind: "linear inequality" }, note: "A list of constraints. Vector inequality — element-wise. CVXPY will enforce all three rows of A @ x <= b." },
    { line: 16, kind: "create_problem", payload: { name: "prob" }, note: "Bind the objective and constraint list into a Problem object. Until you call solve(), this is just symbolic." },
    { line: 18, kind: "solve", payload: { backend: "CLARABEL", iters: 10, time: 0.005, status: "optimal", value: -2.05, vars: { x: [1.0, 1.0] } }, note: "Solve. CVXPY auto-picks a backend — for an LP, CLARABEL or ECOS by default. Pass solver=cp.GUROBI / cp.MOSEK / cp.SCS to override." },
    { line: 20, kind: "print", payload: { text: "status : optimal" }, note: "prob.status is 'optimal' / 'infeasible' / 'unbounded' / 'optimal_inaccurate'." },
    { line: 21, kind: "print", payload: { text: "value  : -2.05" }, note: "prob.value is the optimal objective value." },
    { line: 22, kind: "print", payload: { text: "x*     : [1.0, 1.0]" }, note: "After solve(), x.value holds the optimum. Both x₁+x₂ ≤ 2 and 2x₁+x₂ ≤ 3 are active here." },
  ],
};

const PROB_QP = {
  key: "qp",
  name: "QP — Markowitz",
  kind: "QP",
  blurb:
    "Mean-variance portfolio: minimize wᵀΣw subject to a return floor and budget. cp.quad_form handles the quadratic-positive-semidefinite check automatically — CVXPY refuses non-convex QPs.",
  formula:
    "min  wᵀΣw   s.t.   μᵀw ≥ 0.08,  Σwᵢ = 1,  wᵢ ≥ 0",
  code: [
    null,
    "import cvxpy as cp",
    "import numpy as np",
    "",
    "mu    = np.array([0.12, 0.03, 0.07, 0.10])",
    "vols  = np.array([0.20, 0.04, 0.12, 0.18])",
    "Sigma = np.diag(vols ** 2)",
    "",
    "w = cp.Variable(4, nonneg=True)",
    "",
    "risk   = cp.quad_form(w, Sigma)",
    "ret    = mu @ w",
    "",
    "prob = cp.Problem(",
    "    cp.Minimize(risk),",
    "    [",
    "        ret >= 0.08,",
    "        cp.sum(w) == 1,",
    "    ],",
    ")",
    "",
    "prob.solve()",
    "",
    "for name, wi in zip(['AAPL','BND','GLD','XLE'], w.value):",
    "    print(f'{name}: {wi:.4f}')",
    "print('variance =', prob.value)",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "cvxpy", alias: "cp" } },
    { line: 2, kind: "import", payload: { name: "numpy", alias: "np" } },
    { line: 4, kind: "raw_data", payload: { label: "μ", value: "[0.12, 0.03, 0.07, 0.10]", desc: "expected returns" } },
    { line: 5, kind: "raw_data", payload: { label: "vols", value: "[0.20, 0.04, 0.12, 0.18]", desc: "per-asset volatility" } },
    { line: 6, kind: "raw_data", payload: { label: "Σ", value: "diag(vols²)", desc: "covariance (uncorrelated)" }, note: "Diagonal covariance — assets are uncorrelated. For correlated assets pass a full PSD matrix; CVXPY will check positive semidefiniteness." },
    { line: 8, kind: "add_var", payload: { name: "w", shape: "(4,)", attrs: ["nonneg"] }, note: "Portfolio weights, one per asset. nonneg=True forbids short positions." },
    { line: 10, kind: "add_atom", payload: { name: "risk", expr: "cp.quad_form(w, Σ)", desc: "convex quadratic form" }, note: "cp.quad_form(w, Σ) returns wᵀΣw. CVXPY checks Σ ⪰ 0 at construction; pass assume_PSD=True to skip the check if you've already verified." },
    { line: 11, kind: "add_atom", payload: { name: "ret", expr: "μ @ w", desc: "linear" } },
    { line: 13, kind: "create_problem", payload: { name: "prob" } },
    { line: 14, kind: "set_objective", payload: { sense: "Minimize", expr: "risk" } },
    { line: 16, kind: "add_constraint", payload: { expr: "μᵀw ≥ 0.08", kind: "linear inequality" } },
    { line: 17, kind: "add_constraint", payload: { expr: "Σ wᵢ = 1", kind: "linear equality" } },
    { line: 21, kind: "solve", payload: { backend: "CLARABEL", iters: 13, time: 0.008, status: "optimal", value: 0.005471, vars: { w: [0.239, 0.222, 0.308, 0.232] } }, note: "Auto-dispatched to CLARABEL (a conic solver good for QP/SOCP). For larger QPs you might prefer OSQP." },
    { line: 23, kind: "print", payload: { text: "AAPL: 0.2390" } },
    { line: 23, kind: "print", payload: { text: "BND:  0.2220" } },
    { line: 23, kind: "print", payload: { text: "GLD:  0.3080" } },
    { line: 23, kind: "print", payload: { text: "XLE:  0.2320" } },
    { line: 25, kind: "print", payload: { text: "variance = 0.005471" }, note: "Minimum-variance portfolio achieving the 0.08 return target." },
  ],
};

const PROB_SOCP = {
  key: "socp",
  name: "SOCP",
  kind: "SOCP",
  blurb:
    "Find the point in the unit disk centered at (1, 1) that maximizes x + y. Single second-order cone constraint. CVXPY's cp.norm(., 2) is the canonical SOC atom.",
  formula: "min  −x₁ − x₂   s.t.   ‖x − (1,1)‖₂ ≤ 1",
  code: [
    null,
    "import cvxpy as cp",
    "import numpy as np",
    "",
    "x = cp.Variable(2)",
    "",
    "center = np.array([1.0, 1.0])",
    "radius = 1.0",
    "",
    "prob = cp.Problem(",
    "    cp.Minimize(-x[0] - x[1]),",
    "    [cp.norm(x - center, 2) <= radius],",
    ")",
    "",
    "prob.solve()",
    "",
    "print('status :', prob.status)",
    "print('value  :', prob.value)",
    "print('x*     :', x.value)",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "cvxpy", alias: "cp" } },
    { line: 2, kind: "import", payload: { name: "numpy", alias: "np" } },
    { line: 4, kind: "add_var", payload: { name: "x", shape: "(2,)" }, note: "No attributes — x is unrestricted real." },
    { line: 6, kind: "raw_data", payload: { label: "center", value: "[1.0, 1.0]" } },
    { line: 7, kind: "raw_data", payload: { label: "radius", value: "1.0" } },
    { line: 9, kind: "create_problem", payload: { name: "prob" } },
    { line: 10, kind: "set_objective", payload: { sense: "Minimize", expr: "−x₁ − x₂" } },
    { line: 11, kind: "add_atom", payload: { name: "‖·‖₂", expr: "cp.norm(x − center, 2)", desc: "convex SOC atom" }, note: "cp.norm(z, 2) is the second-order cone atom. Bounding it from above by an affine expression makes the constraint convex (feasible region is a disk)." },
    { line: 11, kind: "add_constraint", payload: { expr: "‖x − (1, 1)‖₂ ≤ 1", kind: "second-order cone" } },
    { line: 13, kind: "solve", payload: { backend: "CLARABEL", iters: 11, time: 0.006, status: "optimal", value: -3.4142, vars: { x: [1.7071, 1.7071] } }, note: "CVXPY recognized the SOC structure and dispatched to a conic solver (CLARABEL). Optimum sits on the disk boundary in the −c direction." },
    { line: 15, kind: "print", payload: { text: "status : optimal" } },
    { line: 16, kind: "print", payload: { text: "value  : -3.4142" } },
    { line: 17, kind: "print", payload: { text: "x*     : [1.7071, 1.7071]" }, note: "Equal to (1,1) + (1,1)/√2 — the point on the disk boundary furthest in the (1,1) direction." },
  ],
};

const PROB_SDP = {
  key: "sdp",
  name: "SDP",
  kind: "SDP",
  blurb:
    "Smallest-eigenvalue problem as an SDP: min tr(CX) over X ⪰ 0 with tr(X) = 1. The optimum is the smallest eigenvalue of C; the optimal X is the rank-one matrix vminvminᵀ. CVXPY's '>>' operator declares matrix variables PSD.",
  formula: "min  tr(C·X)   s.t.   X ⪰ 0,  tr(X) = 1",
  code: [
    null,
    "import cvxpy as cp",
    "import numpy as np",
    "",
    "C = np.array([",
    "    [1.0, 0.5],",
    "    [0.5, 2.0],",
    "])",
    "",
    "X = cp.Variable((2, 2), symmetric=True)",
    "",
    "prob = cp.Problem(",
    "    cp.Minimize(cp.trace(C @ X)),",
    "    [",
    "        X >> 0,",
    "        cp.trace(X) == 1,",
    "    ],",
    ")",
    "",
    "prob.solve()",
    "",
    "print('λ_min(C) =', prob.value)",
    "print('X* =\\n', X.value)",
    "print('eigvals(X*) =', np.linalg.eigvalsh(X.value))",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "cvxpy", alias: "cp" } },
    { line: 2, kind: "import", payload: { name: "numpy", alias: "np" } },
    { line: 4, kind: "raw_data", payload: { label: "C", value: "[[1, 0.5], [0.5, 2]]", desc: "symmetric cost matrix" } },
    { line: 9, kind: "add_var", payload: { name: "X", shape: "(2, 2)", attrs: ["symmetric"] }, note: "symmetric=True declares X = Xᵀ. For SDP variables we ALSO need X ⪰ 0 — that's a separate constraint (next event)." },
    { line: 11, kind: "create_problem", payload: { name: "prob" } },
    { line: 12, kind: "set_objective", payload: { sense: "Minimize", expr: "tr(C·X)" } },
    { line: 14, kind: "add_constraint", payload: { expr: "X ⪰ 0", kind: "LMI (PSD cone)" }, note: "The '>>' operator tells CVXPY 'X is PSD'. Internally this becomes an SDP cone constraint." },
    { line: 15, kind: "add_constraint", payload: { expr: "tr(X) = 1", kind: "linear equality" } },
    { line: 19, kind: "solve", payload: { backend: "CLARABEL", iters: 18, time: 0.012, status: "optimal", value: 0.7929, vars: { "X[0,0]": 0.8536, "X[0,1]": -0.3536, "X[1,1]": 0.1464 } }, note: "CVXPY recognized the SDP and dispatched to a solver that supports the PSD cone — CLARABEL, MOSEK, SCS, or CVXOPT. CLARABEL is the default install." },
    { line: 21, kind: "print", payload: { text: "λ_min(C) = 0.7929" }, note: "Equals (3 − √2) / 2, the smaller eigenvalue of C. The SDP relaxation of an eigenvalue problem is tight." },
    { line: 22, kind: "print", payload: { text: "X* =\n [[ 0.854 -0.354]\n  [-0.354  0.146]]" } },
    { line: 23, kind: "print", payload: { text: "eigvals(X*) = [0.0, 1.0]" }, note: "X* is rank-1 — the projector onto C's smallest-eigenvalue eigenspace. tr(X*) = 1 ✓." },
  ],
};

const PROBLEMS = [PROB_LP, PROB_QP, PROB_SOCP, PROB_SDP];

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo) {
  const s = {
    imports: [],
    rawData: [],
    vars: [],
    atoms: [],
    objective: null,
    constraints: [],
    problem: false,
    result: null,
    prints: [],
  };
  for (let i = 0; i <= upTo && i < events.length; i++) {
    const ev = events[i];
    switch (ev.kind) {
      case "import":
        s.imports.push(ev.payload);
        break;
      case "raw_data":
        s.rawData.push(ev.payload);
        break;
      case "add_var":
        s.vars.push({ ...ev.payload });
        break;
      case "add_atom":
        s.atoms.push(ev.payload);
        break;
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "create_problem":
        s.problem = true;
        break;
      case "solve":
        s.result = ev.payload;
        for (const v of s.vars) {
          if (ev.payload.vars && ev.payload.vars[v.name] !== undefined) {
            v.value = ev.payload.vars[v.name];
          }
        }
        break;
      case "print":
        s.prints.push(ev.payload.text);
        break;
      default:
        break;
    }
  }
  return s;
}

// ============================================================
// Main component
// ============================================================
export default function CVXPYTutorial() {
  const [probKey, setProbKey] = useState(PROB_LP.key);
  const problem = useMemo(
    () => PROBLEMS.find((p) => p.key === probKey),
    [probKey]
  );

  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);

  useEffect(() => {
    setEvIdx(0);
    setRunning(false);
  }, [probKey]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setEvIdx((k) => {
        if (k + 1 >= problem.events.length) {
          setRunning(false);
          return k;
        }
        return k + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, problem.events.length]);

  const stepOnce = useCallback(() => {
    setEvIdx((k) => Math.min(problem.events.length - 1, k + 1));
  }, [problem.events.length]);

  const reset = useCallback(() => {
    setEvIdx(0);
    setRunning(false);
  }, []);

  const ev = problem.events[evIdx];
  const state = useMemo(() => replayState(problem.events, evIdx), [problem, evIdx]);

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        CVXPY — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        CVXPY expresses convex problems in disciplined convex form, checks them
        against DCP rules, and dispatches to a conic backend. Step through four
        canonical problem classes — LP, QP, SOCP, SDP — and watch the same
        Variable / Constraint / Problem pattern adapt.
      </p>

      <InstallPanel />

      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {PROBLEMS.map((p) => (
          <button
            key={p.key}
            onClick={() => setProbKey(p.key)}
            style={{
              ...tabBtn,
              ...(p.key === probKey ? tabBtnActive : {}),
            }}
          >
            <span
              style={{
                fontSize: 10,
                marginRight: 6,
                padding: "1px 5px",
                background: kindColor(p.kind),
                color: "#fff",
                borderRadius: 2,
                fontFamily: "monospace",
                letterSpacing: "0.04em",
              }}
            >
              {p.kind}
            </span>
            {p.name}
          </button>
        ))}
      </div>

      <div style={blurbBox}>
        <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>
          {problem.blurb}
        </div>
        <div
          style={{
            marginTop: 8,
            fontFamily: "monospace",
            fontSize: 12,
            color: "#1f4e3d",
          }}
        >
          {problem.formula}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(420px, 1fr) minmax(420px, 1fr)",
          gap: 22,
          alignItems: "flex-start",
        }}
      >
        <div>
          <CodePanel codeLines={problem.code} highlightedLine={ev?.line || 1} />

          <div style={narrationBox}>
            <div
              style={{
                fontSize: 11,
                color: "#777",
                marginBottom: 4,
                fontFamily: "monospace",
              }}
            >
              what this line does
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.55 }}>
              {ev?.note || "(no note for this line — keep stepping)"}
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            <button
              onClick={stepOnce}
              disabled={evIdx >= problem.events.length - 1}
              style={btnPrimary}
            >
              <StepForward size={16} /> Step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={evIdx >= problem.events.length - 1}
              style={btn}
            >
              {running ? <Pause size={16} /> : <Play size={16} />}
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
              height: 6,
              background: "#eee",
              borderRadius: 3,
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                background: kindColor(problem.kind),
                width: `${(100 * (evIdx + 1)) / problem.events.length}%`,
                transition: "width 0.1s",
              }}
            />
          </div>
          <div
            style={{
              fontSize: 11,
              color: "#888",
              fontFamily: "monospace",
              marginTop: 4,
            }}
          >
            event {evIdx + 1} / {problem.events.length}
          </div>
        </div>

        <StatePanel state={state} />
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Install panel
// ============================================================
function InstallPanel() {
  const [open, setOpen] = useState(false);
  return (
    <div
      style={{
        marginBottom: 18,
        border: "1px solid #d3d3d3",
        borderRadius: 8,
        background: "#fafafa",
      }}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          width: "100%",
          padding: "10px 14px",
          background: "transparent",
          border: 0,
          cursor: "pointer",
          fontWeight: 700,
          fontSize: 14,
          color: "#222",
          textAlign: "left",
        }}
      >
        <Package size={16} />
        Install CVXPY &nbsp;
        <span style={{ color: "#888", fontWeight: 400, fontSize: 12 }}>
          ({open ? "click to collapse" : "click to expand"})
        </span>
      </button>
      {open && (
        <div
          style={{
            padding: "0 14px 14px 14px",
            fontSize: 13,
            color: "#333",
            lineHeight: 1.55,
          }}
        >
          <p style={{ marginTop: 0 }}>
            CVXPY ships with several backend solvers by default — you usually
            don't need to install anything else for LP / QP / SOCP / SDP.
          </p>
          <Pre>
            {`# Base install (includes CLARABEL, ECOS, SCS, OSQP)
pip install cvxpy

# Optional: faster commercial solvers
pip install gurobipy   # needs Gurobi license (free academic)
pip install mosek      # needs Mosek license (free academic)

# Verify
python -c "import cvxpy as cp; print(cp.installed_solvers())"
# → ['CLARABEL', 'CVXOPT', 'ECOS', 'ECOS_BB', 'OSQP', 'SCIPY', 'SCS']`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            Specify a backend with <code style={inlineCode}>prob.solve(solver=cp.MOSEK)</code>.
            CVXPY auto-picks based on problem class otherwise.
          </p>
        </div>
      )}
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
// State panel — CVXPY-flavored
// ============================================================
function StatePanel({ state }) {
  return (
    <div style={statePanelOuter}>
      <Section title="Imports">
        {state.imports.length === 0 ? (
          <Empty />
        ) : (
          state.imports.map((imp, i) => (
            <div key={i} style={chip("#1f4e3d")}>
              import {imp.name} as {imp.alias}
            </div>
          ))
        )}
      </Section>

      {state.rawData.length > 0 && (
        <Section title="NumPy data">
          {state.rawData.map((d, i) => (
            <KVRow key={i} k={d.label} v={d.value} desc={d.desc} mono />
          ))}
        </Section>
      )}

      <Section title="Variables">
        {state.vars.length === 0 ? (
          <Empty />
        ) : (
          state.vars.map((v, i) => (
            <div
              key={i}
              style={{
                marginBottom: 8,
                padding: "6px 10px",
                background: "#fff",
                border: "1px solid #ddd",
                borderRadius: 6,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontFamily: "monospace", fontWeight: 700 }}>
                  {v.name}
                </span>
                <span style={{ fontSize: 11, color: "#666", fontFamily: "monospace" }}>
                  cp.Variable({v.shape})
                </span>
              </div>
              {v.attrs && (
                <div
                  style={{
                    fontSize: 12,
                    color: "#0b3da0",
                    fontFamily: "monospace",
                    marginTop: 2,
                  }}
                >
                  attrs: {v.attrs.join(", ")}
                </div>
              )}
              {v.value != null && (
                <div
                  style={{
                    fontSize: 12,
                    color: "#c8311c",
                    fontFamily: "monospace",
                    marginTop: 2,
                    fontWeight: 700,
                  }}
                >
                  .value = {Array.isArray(v.value) ? "[" + v.value.map((x) => (+x).toFixed(4)).join(", ") + "]" : v.value}
                </div>
              )}
            </div>
          ))
        )}
      </Section>

      {state.atoms.length > 0 && (
        <Section title="Atoms / sub-expressions">
          {state.atoms.map((a, i) => (
            <div
              key={i}
              style={{
                padding: "5px 10px",
                background: "#fff",
                border: "1px solid #ddd",
                borderRadius: 6,
                marginBottom: 5,
                fontFamily: "monospace",
                fontSize: 12,
              }}
            >
              <span style={{ fontWeight: 700 }}>{a.name}</span> = {a.expr}
              {a.desc && (
                <span style={{ color: "#888", marginLeft: 8 }}>({a.desc})</span>
              )}
            </div>
          ))}
        </Section>
      )}

      <Section title="Objective">
        {state.objective ? (
          <div
            style={{
              padding: "6px 10px",
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 6,
              fontFamily: "monospace",
              fontSize: 13,
            }}
          >
            <span style={{ color: "#555", fontSize: 11 }}>cp.{state.objective.sense}</span>
            <div>{state.objective.expr}</div>
          </div>
        ) : (
          <Empty />
        )}
      </Section>

      <Section title="Constraints">
        {state.constraints.length === 0 ? (
          <Empty />
        ) : (
          state.constraints.map((c, i) => (
            <div
              key={i}
              style={{
                marginBottom: 6,
                padding: "6px 10px",
                background: "#fff",
                border: "1px solid #ddd",
                borderRadius: 6,
                fontFamily: "monospace",
                fontSize: 13,
              }}
            >
              <div>{c.expr}</div>
              {c.kind && (
                <span style={{ color: "#888", fontSize: 11 }}>{c.kind}</span>
              )}
            </div>
          ))
        )}
      </Section>

      <Section title="Problem">
        {state.problem ? (
          <div style={chip("#0b3da0")}>cp.Problem(objective, constraints)</div>
        ) : (
          <Empty />
        )}
      </Section>

      {state.result && (
        <Section title="Solver result">
          <div
            style={{
              background: "#1f1d1a",
              color: "#e8e2d4",
              padding: 10,
              borderRadius: 6,
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: 1.55,
            }}
          >
            <ResultLine k="backend" v={state.result.backend} c="#9a4caa" />
            <ResultLine k="status" v={state.result.status} c="#7dd87d" />
            <ResultLine k="iterations" v={state.result.iters} />
            <ResultLine k="time" v={`${state.result.time.toFixed(3)} s`} />
            <ResultLine
              k="prob.value"
              v={(+state.result.value).toFixed(6)}
              c="#f5a524"
            />
            {Object.entries(state.result.vars).map(([k, v]) => (
              <ResultLine
                key={k}
                k={k + ".value"}
                v={
                  Array.isArray(v)
                    ? "[" + v.map((x) => (+x).toFixed(4)).join(", ") + "]"
                    : (+v).toFixed(4)
                }
                c="#f5a524"
              />
            ))}
          </div>
        </Section>
      )}

      {state.prints.length > 0 && (
        <Section title="Console output">
          <div
            style={{
              background: "#0a0a0a",
              color: "#dadada",
              padding: 10,
              borderRadius: 6,
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: 1.55,
              whiteSpace: "pre-wrap",
            }}
          >
            {state.prints.map((line, i) => (
              <div key={i}>
                <span style={{ color: "#5a5a5a" }}>{">>> "}</span>
                {line}
              </div>
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          letterSpacing: "0.18em",
          color: "#888",
          marginBottom: 6,
          textTransform: "uppercase",
        }}
      >
        {title}
      </div>
      {children}
    </div>
  );
}
function Empty() {
  return (
    <div
      style={{
        padding: "6px 10px",
        background: "#fff",
        border: "1px dashed #ddd",
        borderRadius: 6,
        color: "#999",
        fontSize: 12,
        fontStyle: "italic",
      }}
    >
      (none yet)
    </div>
  );
}
function chip(color) {
  return {
    display: "inline-block",
    padding: "4px 10px",
    background: color,
    color: "#fff",
    borderRadius: 4,
    fontFamily: "monospace",
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: "0.04em",
    marginRight: 4,
    marginBottom: 4,
  };
}
function KVRow({ k, v, desc, mono }) {
  return (
    <div
      style={{
        padding: "3px 0",
        borderBottom: "1px dotted #eee",
        display: "flex",
        gap: 8,
      }}
    >
      <span
        style={{
          color: "#666",
          fontSize: 12,
          minWidth: 80,
          fontFamily: "monospace",
        }}
      >
        {k}
      </span>
      <span
        style={{
          fontSize: 12,
          color: "#222",
          fontFamily: mono ? "monospace" : "inherit",
          flex: 1,
          wordBreak: "break-word",
        }}
      >
        {v}
        {desc && (
          <span style={{ color: "#999", fontStyle: "italic", marginLeft: 6 }}>
            — {desc}
          </span>
        )}
      </span>
    </div>
  );
}
function ResultLine({ k, v, c }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ color: "#7f7864" }}>{k}</span>
      <span style={{ color: c || "#e8e2d4", fontWeight: 600 }}>{v}</span>
    </div>
  );
}
function kindColor(k) {
  if (k === "LP") return "#1f4e3d";
  if (k === "QP") return "#d4a017";
  if (k === "SOCP") return "#0b3da0";
  if (k === "SDP") return "#7a3da0";
  return "#444";
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
      <ul
        style={{
          margin: 0,
          paddingLeft: 22,
          lineHeight: 1.6,
          fontSize: 14,
          color: "#3d2f00",
        }}
      >
        <li>
          <b>DCP (Disciplined Convex Programming).</b> CVXPY enforces that every
          atom you compose is curvature-tagged: <i>convex</i>, <i>concave</i>,
          or <i>affine</i>. If you build a non-convex expression, it raises{" "}
          <code style={inlineCode}>DCPError</code> at construction time — not
          at solve time. This is the whole point of CVXPY.
        </li>
        <li>
          <b>Backend selection.</b>{" "}
          <code style={inlineCode}>prob.solve()</code> picks a backend based on
          problem class. LP → CLARABEL/ECOS, QP → OSQP/CLARABEL, SOCP → ECOS/CLARABEL,
          SDP → SCS/CLARABEL/MOSEK. Override with{" "}
          <code style={inlineCode}>solver=cp.MOSEK</code>.
        </li>
        <li>
          <b>Variable attributes are constraints.</b>{" "}
          <code style={inlineCode}>nonneg=True</code>,{" "}
          <code style={inlineCode}>symmetric=True</code>,{" "}
          <code style={inlineCode}>PSD=True</code>,{" "}
          <code style={inlineCode}>boolean=True</code>{" "}
          attach restrictions directly to the variable. Cleaner than adding
          them as separate constraints.
        </li>
        <li>
          <b>The '{">>"}' operator</b> means "PSD" for matrix variables and "elementwise &gt;"
          for scalars. <code style={inlineCode}>X {">>"} 0</code> is the
          canonical SDP cone constraint.
        </li>
        <li>
          <b>Result attributes.</b> After solve():{" "}
          <code style={inlineCode}>prob.value</code> is the optimal objective,
          <code style={inlineCode}>x.value</code> is the primal,{" "}
          <code style={inlineCode}>constraint.dual_value</code> is the dual.
        </li>
      </ul>
    </div>
  );
}

// ============================================================
// Style atoms
// ============================================================
const tabBtn = {
  padding: "8px 14px",
  border: "1px solid #ccc",
  borderRadius: 6,
  background: "#fff",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: 13,
  display: "inline-flex",
  alignItems: "center",
};
const tabBtnActive = {
  background: "#1f1d1a",
  color: "#fff",
  border: "1px solid #1f1d1a",
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
const inlineCode = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
const blurbBox = {
  padding: "12px 16px",
  background: "#f6f4ee",
  border: "1px solid #ece8dd",
  borderRadius: 8,
  marginBottom: 16,
};
const narrationBox = {
  marginTop: 14,
  padding: "12px 16px",
  background: "#f4efe6",
  borderRadius: 8,
  border: "1px solid #ddd",
};
const smallLabel = {
  display: "block",
  fontSize: 12,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
};
const statePanelOuter = {
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 8,
  padding: 14,
};
function Pre({ children }) {
  return (
    <pre
      style={{
        background: "#1f1d1a",
        color: "#e8e2d4",
        padding: "12px 14px",
        borderRadius: 6,
        fontSize: 12,
        fontFamily: "'JetBrains Mono', Menlo, monospace",
        lineHeight: 1.55,
        overflowX: "auto",
        whiteSpace: "pre",
      }}
    >
      {children}
    </pre>
  );
}
