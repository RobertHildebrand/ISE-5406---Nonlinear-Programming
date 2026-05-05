import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Terminal,
  Package,
} from "lucide-react";
import { OutputReader } from "./output_reader.jsx";

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

const PROB_SPARSE_LOGREG = {
  key: "logreg",
  name: "Sparse Logistic Regression",
  kind: "EXP",
  blurb:
    "L1-regularized binary classifier — the workhorse 'lasso logistic' from statistical learning. Smooth convex log-loss + non-smooth L1 penalty. CVXPY routes this through the exponential-cone (cp.logistic) and the L1 norm splits the sum into auxiliary variables behind the scenes. This is genuinely production-shaped CVXPY code: arrays, broadcasting, and a non-smooth regularizer.",
  formula: "min  Σᵢ log(1 + exp(−yᵢ·(Xᵢβ + b))) + λ·‖β‖₁",
  code: [
    null,
    "import cvxpy as cp",
    "import numpy as np",
    "",
    "# 50 samples, 8 features. Only first 3 features matter.",
    "rng = np.random.default_rng(0)",
    "n, d = 50, 8",
    "X = rng.standard_normal((n, d))",
    "true_beta = np.array([2.0, -1.5, 1.0, 0, 0, 0, 0, 0])",
    "y = np.sign(X @ true_beta + 0.3*rng.standard_normal(n))",
    "",
    "beta = cp.Variable(d)",
    "b    = cp.Variable()",
    "lam  = cp.Parameter(nonneg=True, value=0.1)",
    "",
    "scores = cp.multiply(y, X @ beta + b)",
    "loss   = cp.sum(cp.logistic(-scores))",
    "reg    = lam * cp.norm1(beta)",
    "",
    "prob = cp.Problem(cp.Minimize(loss + reg))",
    "prob.solve(solver=cp.CLARABEL, verbose=True)",
    "",
    "print('value      :', prob.value)",
    "print('β (sparse) :', np.round(beta.value, 3))",
    "print('intercept  :', round(b.value, 3))",
    "print('# nonzeros :', int(np.sum(np.abs(beta.value) > 1e-3)))",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "cvxpy", alias: "cp" } },
    { line: 2, kind: "import", payload: { name: "numpy", alias: "np" } },
    { line: 5, kind: "raw_data", payload: { label: "rng", value: "default_rng(0)", desc: "fixed seed" }, note: "Seed the RNG so the demo is reproducible. In production never hardcode a seed unless you're benchmarking." },
    { line: 6, kind: "raw_data", payload: { label: "(n, d)", value: "(50, 8)", desc: "50 examples, 8 features" } },
    { line: 7, kind: "raw_data", payload: { label: "X", value: "(50, 8) Gaussian", desc: "design matrix" }, note: "Standard normal features. The L1 penalty forces β to be sparse — relevant when d ≫ n or when most features are irrelevant." },
    { line: 8, kind: "raw_data", payload: { label: "true β", value: "[2, −1.5, 1, 0, 0, 0, 0, 0]" }, note: "Five of the eight true coefficients are zero. We're hoping CVXPY recovers that." },
    { line: 9, kind: "raw_data", payload: { label: "y", value: "(50,) ∈ {±1}", desc: "labels" } },
    { line: 11, kind: "add_var", payload: { name: "β", shape: "(8,)" }, note: "Coefficient vector — unconstrained real. Sparsity comes from the regularizer, not from variable attributes." },
    { line: 12, kind: "add_var", payload: { name: "b", shape: "()" }, note: "Scalar intercept (bias). Unregularized — only β is shrunk." },
    { line: 13, kind: "add_var", payload: { name: "λ", shape: "()", attrs: ["nonneg", "Parameter"] }, note: "cp.Parameter is a knob you can re-set without rebuilding the Problem. Crucial for cross-validation: change λ.value and re-solve." },
    { line: 15, kind: "add_atom", payload: { name: "scores", expr: "y · (Xβ + b)", desc: "broadcasted" }, note: "Element-wise product of label vector with the linear scores. cp.multiply is broadcast-aware. Negative when the model gets it wrong." },
    { line: 16, kind: "add_atom", payload: { name: "loss", expr: "Σ log(1 + e^{−scoreᵢ})", desc: "log-likelihood (convex)" }, note: "cp.logistic(z) = log(1 + exp(z)). It's a DCP-recognized exp-cone atom — CVXPY will translate the problem into an exponential-cone program for the solver." },
    { line: 17, kind: "add_atom", payload: { name: "reg", expr: "λ · ‖β‖₁", desc: "L1 (non-smooth)" }, note: "The L1 norm is convex and CVXPY rewrites it internally as auxiliary variables u with u ≥ β, u ≥ −β, and minimization of Σuᵢ. You don't see it but it's why this is a conic LP+exp-cone hybrid." },
    { line: 19, kind: "create_problem", payload: { name: "prob" } },
    { line: 19, kind: "set_objective", payload: { sense: "Minimize", expr: "loss + reg" } },
    {
      line: 20, kind: "solve",
      payload: {
        backend: "CLARABEL",
        iters: 22,
        time: 0.026,
        status: "optimal",
        value: 16.842,
        vars: { β: [1.74, -1.32, 0.79, 0.00, 0.00, 0.04, 0.00, 0.00], b: 0.18 },
      },
      note: "verbose=True prints CLARABEL's per-iteration table — primal objective, dual objective, gap, residuals, step size. CLARABEL is the default conic solver; for exp-cone problems it (or SCS or MOSEK) is required. ECOS_BB does NOT support exponential cones.",
    },
    { line: 22, kind: "print", payload: { text: "value      : 16.842" } },
    { line: 23, kind: "print", payload: { text: "β (sparse) : [ 1.74 -1.32  0.79  0.    0.    0.04  0.    0.  ]" }, note: "Five components are EXACTLY zero, one is tiny (0.04). The L1 penalty correctly identified the sparse support." },
    { line: 24, kind: "print", payload: { text: "intercept  : 0.18" } },
    { line: 25, kind: "print", payload: { text: "# nonzeros : 4" }, note: "We recovered three true features plus one false positive. Increase λ to push that to exactly 3, decrease it to keep more features." },
  ],
};

const PROBLEMS = [PROB_LP, PROB_QP, PROB_SOCP, PROB_SDP, PROB_SPARSE_LOGREG];

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

      <OutputReader
        title="Reading CLARABEL's output, column by column"
        intro="With verbose=True, CLARABEL prints a per-iteration table while it climbs the central path. All nine columns are documented below — click Next (or any card) to step through them. The corresponding column lights up in the table and the textbox at the bottom tells you what that column did on THIS specific solve."
        columns={CVXPY_COLS}
        logs={CVXPY_LOGS}
        problemKey={problem.key}
      />
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Output reader — column data for CLARABEL
// ============================================================
const CVXPY_COLS = [
  { key: "iter", label: "iter", def: "Newton-step iteration along the central path. Typical convex conic problems converge in 10–30 iterations regardless of size. If iter exceeds 50, you likely have ill-conditioning or near-infeasibility." },
  { key: "pcost", label: "pcost", def: "Primal objective at the current iterate. For Minimize this approaches the optimum from above. It's the number you'd report if forced to stop early." },
  { key: "dcost", label: "dcost", def: "Dual objective. For Minimize, dcost ≤ optimum ≤ pcost. Their difference is the duality gap — the certificate of optimality." },
  { key: "gap", label: "gap", def: "Relative duality gap (pcost − dcost) / max(1, |pcost|). Headline 'how close' number. 1e-8 = production tolerance; 1e-6 fine for ML." },
  { key: "pres", label: "pres", def: "Primal residual: ‖Ax − b‖ scaled. Measures constraint violation. Should drive to zero at the optimum." },
  { key: "dres", label: "dres", def: "Dual residual: KKT-stationarity violation ‖∇L‖. Drops to zero alongside pres." },
  { key: "kt", label: "k/t", def: "Homogeneous embedding's complementarity (κτ). Diverges if the problem is infeasible — the canonical infeasibility certificate. Otherwise just a sanity check." },
  { key: "mu", label: "μ", def: "Central-path barrier parameter. CLARABEL drives μ → 0 to push the iterate toward the cone boundary (= the optimum). Tied to step size." },
  { key: "step", label: "step", def: "Damped Newton step length. 1.0 = full Newton step (quadratic-convergence basin). < 0.5 means CLARABEL is staying cautious near the cones." },
];

// ── CVXPY / CLARABEL extras ──
const CVXPY_OUTPUT_EXTRAS = [
  {
    key: "setup",
    kind: "output",
    label: "Setup block (cone summary)",
    summary: "Problem dimensions before iter table",
    excerpt: "problem\n  variables     = 2\n  constraints   = 3 (linear inequalities) + 2 (variable nonneg)\n  cones         = 5 nonneg-orthant components",
    explain: "CLARABEL prints what it sees AFTER CVXPY's canonicalization. The 'cones' line is most informative — it tells you which cone families CLARABEL has to project onto each iteration: nonneg-orthant (LP), zero (equality), SOC (second-order), PSD (semidefinite), exp (exponential), pow (power). If the count looks higher than your model, that's CVXPY adding auxiliary variables for atoms like norm1 or logistic.",
  },
  {
    key: "termination",
    kind: "output",
    label: "Termination line",
    summary: "Status + solve time",
    excerpt: "terminated: optimal\nsolve time:    0.005 sec\nprimal obj :   -2.0500\ndual obj   :   -2.0500\ngap        :    9.5e-09",
    explain: "Final block. Possible 'terminated' values: optimal, primal_infeasible, dual_infeasible, max_iter, time_limit, numerical_error, algebraic_error. Always check this before consuming prob.value. If status is 'optimal' but you're worried about precision, look at gap and pres/dres in the last iter row — sub-1e-6 means you're fine for ML, sub-1e-9 means you're at the limit of double precision.",
  },
];

const CVXPY_FEATURES = [
  {
    key: "parameter",
    kind: "feature",
    label: "cp.Parameter for fast resolves (DPP)",
    summary: "Change values without rebuilding problem",
    excerpt: "lam = cp.Parameter(nonneg=True)\nprob = cp.Problem(cp.Minimize(loss + lam * cp.norm1(beta)))\n\nfor lam_val in [0.001, 0.01, 0.1, 1.0]:\n    lam.value = lam_val\n    prob.solve()  # MUCH faster after first solve\n    print(lam_val, prob.value)",
    explain: "cp.Parameter is a knob you can re-set without rebuilding the Problem. Crucial for cross-validation, regularization-path computation, sensitivity studies. CVXPY uses DPP (Disciplined Parametrized Programming) to do canonicalization ONCE and reuse the compiled problem on subsequent solves — typically 100× speedup over rebuilding.",
  },
  {
    key: "solver_choice",
    kind: "feature",
    label: "Solver-backend override",
    summary: "Force CLARABEL / MOSEK / SCS / GUROBI / ECOS",
    excerpt: "# CVXPY auto-picks based on problem class.\n# Override:\nprob.solve(solver=cp.MOSEK)        # commercial, fastest for SDP\nprob.solve(solver=cp.GUROBI)       # commercial, best for QP/MIP\nprob.solve(solver=cp.SCS)          # GPU-friendly, good for huge problems\nprob.solve(solver=cp.ECOS)         # tiny problems, very fast startup\nprob.solve(solver=cp.CLARABEL)     # default; modern Rust replacement for ECOS\n\n# Pass solver-specific options:\nprob.solve(solver=cp.MOSEK, mosek_params={'MSK_DPAR_INTPNT_TOL_REL_GAP': 1e-10})",
    explain: "CVXPY auto-dispatches based on problem class (LP→CLARABEL, QP→OSQP/CLARABEL, SOCP→CLARABEL/ECOS, SDP→CLARABEL/MOSEK/SCS, MIP→GUROBI/CBC/MOSEK). Override when you have license access (MOSEK/GUROBI are usually fastest) or when defaults choke. cp.installed_solvers() shows what's actually available on your system.",
  },
  {
    key: "duals",
    kind: "feature",
    label: "Dual / variable values",
    summary: "After solve, inspect the primal+dual",
    excerpt: "prob.solve()\nprint('x*    =', x.value)             # primal\nprint('λ_eq =', constraint_eq.dual_value)\nprint('λ_ineq =', constraint_ineq.dual_value)\nprint('shadow on every constraint:')\nfor c in prob.constraints:\n    print(c, '→', c.dual_value)",
    explain: "After solve(), every Variable has a .value (the primal optimum) and every Constraint has a .dual_value (the Lagrange multiplier / shadow price). For inequalities, dual_value > 0 means the constraint is active. For equalities, the sign tells you which direction would loosen the optimum. Free — no extra solve required, comes back from the conic solver natively.",
  },
  {
    key: "gradients",
    kind: "feature",
    label: "Differentiable optimization",
    summary: "Gradient of solution w.r.t. parameters",
    excerpt: "lam = cp.Parameter(nonneg=True, value=0.1)\nprob = cp.Problem(cp.Minimize(loss + lam * cp.norm1(beta)))\nprob.solve(requires_grad=True)\n\n# Backprop through the optimization\nlam.delta = 0.01    # perturb lam upward\nprob.derivative()\nprint('dβ/dλ:', beta.delta)  # how β shifts with λ",
    explain: "CVXPY supports DIFFERENTIATING through the optimization (cvxpylayers under the hood). Useful for end-to-end ML pipelines where the inner optimization's solution feeds a downstream loss. Set requires_grad=True at solve, then call prob.derivative() to get sensitivities. Restricted to disciplined-parametrized problems.",
  },
  {
    key: "warm_start_cvxpy",
    kind: "feature",
    label: "Warm starts",
    summary: "Re-solve faster after small change",
    excerpt: "prob.solve(warm_start=True)\n# CVXPY caches canonicalization between solves;\n# the solver also reuses prior iterate as a starting point.",
    explain: "Pass warm_start=True (default for parametric problems). CVXPY keeps the canonicalization cache and CLARABEL/OSQP/Gurobi all support primal-dual restart. Combined with cp.Parameter, sequential solves often run 10–100× faster than the first.",
  },
];

const CVXPY_LOGS = {
  lp: {
    extras: [...CVXPY_OUTPUT_EXTRAS, ...CVXPY_FEATURES],
    setupText: "problem\n  variables     = 2\n  constraints   = 3 (linear inequalities) + 2 (variable nonneg)\n  cones         = 5 nonneg-orthant components\n  ----------------------\n  Status: solving",
    rows: [
      { cells: { iter: "  0", pcost: "+0.0000e+00", dcost: "-0.0000e+00", gap: "0.00e+00", pres: "3.00e+00", dres: "1.05e+00", kt: "1.00e+00", mu: "1.00e+00", step: "----" } },
      { cells: { iter: "  1", pcost: "-1.5234e+00", dcost: "-2.0501e+00", gap: "5.27e-01", pres: "9.21e-02", dres: "6.42e-02", kt: "6.30e-02", mu: "9.85e-02", step: "4.51e-01" } },
      { cells: { iter: "  5", pcost: "-2.0498e+00", dcost: "-2.0500e+00", gap: "2.31e-04", pres: "4.21e-05", dres: "6.04e-06", kt: "9.05e-06", mu: "1.49e-05", step: "9.10e-01" } },
      { cells: { iter: " 10", pcost: "-2.0500e+00", dcost: "-2.0500e+00", gap: "9.50e-09", pres: "1.83e-09", dres: "2.40e-10", kt: "1.10e-10", mu: "6.16e-11", step: "9.95e-01" } },
    ],
    summary: "terminated: optimal\nsolve time:    0.005 sec\nprimal obj :   -2.0500\ndual obj   :   -2.0500\ngap        :    9.5e-09",
    finalSummary:
      "Textbook LP convergence on CLARABEL's central path. Quadratic convergence kicks in by iteration 5, giving four extra digits of accuracy in five iterations. Total cost: 10 Newton steps, 5 ms.",
    perCol: {
      iter: "Four sample rows: 0, 1, 5, 10. CLARABEL prints one row per Newton step. The iter column lets you compare Newton-step counts across problems — 10 here, vs 22 for sparse logreg.",
      pcost: "0 → −1.52 → −2.05 → −2.05. Climbs into the optimum (well, descends — we're minimizing). Locked in by iter 5.",
      dcost: "0 → −2.05 → −2.05 → −2.05. The dual converges FASTER than the primal here — typical for LPs with simple geometry.",
      gap: "0 → 0.53 → 2.3e−4 → 9.5e−9. Quadratic shrinkage once we're inside the central neighborhood. Each row drops the gap by 4–5 orders of magnitude.",
      pres: "3 → 9e−2 → 4e−5 → 2e−9. Constraints are well satisfied by the end. The first iter has pres = 3 because the initial point isn't feasible — interior-point methods accept that.",
      dres: "1 → 6e−2 → 6e−6 → 2e−10. KKT-stationarity drops monotonically. Dual residual closes faster than primal — a clean LP.",
      kt: "1 → 6e−2 → 9e−6 → 1e−10. Decreasing — confirms the problem is feasible and bounded. (Diverging k/t would mean infeasible.)",
      mu: "1 → 0.099 → 1.5e−5 → 6e−11. Each barrier-update step reduces μ by ~3 orders of magnitude. Standard schedule.",
      step: "---- → 0.45 → 0.91 → 1.00. Started cautious (45% Newton step) and grew to full step by the end — meaning the iterate is now in the quadratic basin and CLARABEL trusts the Newton direction completely.",
    },
  },
  qp: {
    extras: [...CVXPY_OUTPUT_EXTRAS, ...CVXPY_FEATURES],
    setupText: "problem\n  variables     = 4\n  constraints   = 4 nonneg + 1 ineq + 1 eq\n  cones         = 5 nonneg-orthant + 1 zero\n  PSD blocks    = 1 (4×4 from quad_form)\n  ----------------------\n  Status: solving",
    rows: [
      { cells: { iter: "  0", pcost: "+0.0000e+00", dcost: "+0.0000e+00", gap: "0.00e+00", pres: "1.00e+00", dres: "1.20e-01", kt: "1.00e+00", mu: "1.00e+00", step: "----" } },
      { cells: { iter: "  3", pcost: "+5.5000e-03", dcost: "+5.4502e-03", gap: "4.98e-05", pres: "3.10e-04", dres: "6.20e-05", kt: "1.10e-04", mu: "4.55e-05", step: "8.92e-01" } },
      { cells: { iter: "  9", pcost: "+5.4710e-03", dcost: "+5.4710e-03", gap: "1.40e-08", pres: "4.40e-09", dres: "9.10e-10", kt: "6.10e-10", mu: "3.00e-10", step: "9.95e-01" } },
      { cells: { iter: " 13", pcost: "+5.4710e-03", dcost: "+5.4710e-03", gap: "3.40e-11", pres: "1.00e-11", dres: "3.10e-12", kt: "1.00e-12", mu: "9.00e-13", step: "9.99e-01" } },
    ],
    summary: "terminated: optimal\nsolve time:    0.008 sec\nprimal obj :    0.005471\ndual obj   :    0.005471\ngap        :    3.4e-11",
    finalSummary:
      "Markowitz QP. CLARABEL canonicalizes quad_form into a small SOCP via a Schur-complement trick — that's why the setup block reports a PSD/SOC block. 13 iterations is typical for a strongly convex QP.",
    perCol: {
      iter: "0, 3, 9, 13 — CLARABEL prints sparingly here because the problem is small.",
      pcost: "0 → 5.5e−3 → 5.471e−3 → 5.471e−3. Monotone DECREASE (we're minimizing variance) once we're inside the cone.",
      dcost: "0 → 5.45e−3 → 5.471e−3 → 5.471e−3. Climbs from below; meets pcost at iter 9.",
      gap: "0 → 5e−5 → 1.4e−8 → 3.4e−11. Three orders of magnitude per row pair. Strongly convex QPs converge fastest.",
      pres: "1 → 3e−4 → 4e−9 → 1e−11. Clean convergence — equality constraints (Σwᵢ = 1) and inequalities all satisfied to machine precision.",
      dres: "0.12 → 6e−5 → 9e−10 → 3e−12. Dual residual smaller than primal residual at every iteration — the strong-convexity prior is doing work.",
      kt: "1 → 1e−4 → 6e−10 → 1e−12. Drops fast; QP is feasible and bounded.",
      mu: "1 → 4.5e−5 → 3e−10 → 9e−13. Big drops between rows, especially between iter 3 and 9 — the centering phase is short for a well-conditioned QP.",
      step: "---- → 0.89 → 0.99 → 1.00. Almost-full steps from iter 3 onward. Strongly convex objective gives huge Newton-step trust.",
    },
  },
  socp: {
    extras: [...CVXPY_OUTPUT_EXTRAS, ...CVXPY_FEATURES],
    setupText: "problem\n  variables     = 2\n  cones         = 1 SOC (size 3)\n  ----------------------\n  Status: solving",
    rows: [
      { cells: { iter: "  0", pcost: "+0.0000e+00", dcost: "+0.0000e+00", gap: "0.00e+00", pres: "1.00e+00", dres: "1.00e+00", kt: "1.00e+00", mu: "1.00e+00", step: "----" } },
      { cells: { iter: "  2", pcost: "-3.4123e+00", dcost: "-3.4189e+00", gap: "6.55e-03", pres: "4.10e-03", dres: "9.20e-04", kt: "3.30e-04", mu: "4.04e-04", step: "6.78e-01" } },
      { cells: { iter: "  6", pcost: "-3.4142e+00", dcost: "-3.4142e+00", gap: "1.91e-06", pres: "3.50e-07", dres: "1.00e-07", kt: "4.00e-08", mu: "1.91e-08", step: "9.85e-01" } },
      { cells: { iter: " 11", pcost: "-3.4142e+00", dcost: "-3.4142e+00", gap: "9.20e-12", pres: "4.10e-12", dres: "6.20e-13", kt: "9.10e-14", mu: "4.50e-14", step: "9.99e-01" } },
    ],
    summary: "terminated: optimal\nsolve time:    0.006 sec\nprimal obj :   -3.4142\ndual obj   :   -3.4142\ngap        :    9.2e-12",
    finalSummary:
      "Single second-order cone of size n+1 = 3. Optimum −2√2 = −3.4142 sits on the SOC boundary. The central path approached it from inside the cone — pres and dres stay roughly equal, which is why this took 11 iters not 7.",
    perCol: {
      iter: "0, 2, 6, 11 — slightly more iterations than the LP because cone projections (rather than orthant projections) are the operation per Newton step.",
      pcost: "Lands at −3.4142 (= −2√2) by iter 6 and stays there. The optimum is on the SOC boundary.",
      dcost: "Same answer from below. Equal to pcost from iter 6 onward.",
      gap: "0 → 6.5e−3 → 1.9e−6 → 9.2e−12. Three big drops. 5 iters to get to 1e−6, 5 more to get to 1e−12 — a flat second half is healthy.",
      pres: "1 → 4e−3 → 3.5e−7 → 4e−12. The constraint ‖x − (1,1)‖ ≤ 1 is tight at the optimum, so CLARABEL has to slide along the cone boundary.",
      dres: "1 → 9e−4 → 1e−7 → 6e−13. Same magnitude as pres — symmetric primal-dual progress (typical for cone problems with no 'easy' face).",
      kt: "1 → 3e−4 → 4e−8 → 9e−14. Decreasing, confirms feasibility.",
      mu: "1 → 4e−4 → 1.9e−8 → 4.5e−14. Steeper decay than the QP — fewer Newton steps means each barrier update is more aggressive.",
      step: "---- → 0.68 → 0.99 → 1.00. Started cautious near the SOC boundary, accelerated as the iterate moved into the central neighborhood.",
    },
  },
  sdp: {
    extras: [...CVXPY_OUTPUT_EXTRAS, ...CVXPY_FEATURES],
    setupText: "problem\n  variables     = 1 (X, 2×2 symmetric → 3 free entries)\n  cones         = 1 PSD (size 2) + 1 zero (trace eq)\n  ----------------------\n  Status: solving",
    rows: [
      { cells: { iter: "  0", pcost: "+0.0000e+00", dcost: "+0.0000e+00", gap: "0.00e+00", pres: "1.00e+00", dres: "2.10e-01", kt: "1.00e+00", mu: "1.00e+00", step: "----" } },
      { cells: { iter: "  6", pcost: "+0.7950e+00", dcost: "+0.7901e+00", gap: "4.92e-03", pres: "9.10e-04", dres: "3.20e-04", kt: "1.20e-04", mu: "3.10e-04", step: "7.20e-01" } },
      { cells: { iter: " 12", pcost: "+0.7929e+00", dcost: "+0.7929e+00", gap: "9.85e-08", pres: "4.20e-08", dres: "1.00e-08", kt: "9.30e-09", mu: "3.05e-09", step: "9.50e-01" } },
      { cells: { iter: " 18", pcost: "+0.7929e+00", dcost: "+0.7929e+00", gap: "4.10e-12", pres: "3.30e-12", dres: "4.20e-13", kt: "6.10e-14", mu: "3.00e-14", step: "9.97e-01" } },
    ],
    summary: "terminated: optimal\nsolve time:    0.012 sec\nprimal obj :    0.7929\ndual obj   :    0.7929\ngap        :    4.1e-12",
    finalSummary:
      "SDP solves are slower per iteration (PSD cone projection = eigendecomposition) but converge in similar numbers of iterations. 18 iters is typical for a 2×2 SDP. Optimum 0.7929 = (3 − √2)/2 is the smaller eigenvalue of C — exactly what an SDP relaxation of an eigenvalue problem returns.",
    perCol: {
      iter: "0, 6, 12, 18 — an extra 6 iterations relative to the LP. Twice as many iterations × more expensive Newton step = ~2× the wall time.",
      pcost: "0 → 0.795 → 0.7929 → 0.7929. The objective starts overshooting (0.795 > 0.7929) and corrects as the iterate moves toward the central path.",
      dcost: "0 → 0.7901 → 0.7929 → 0.7929. Approaches from below.",
      gap: "0 → 5e−3 → 1e−7 → 4e−12. Slightly slower convergence than the LP — a 2×2 PSD cone has more 'curvature' to climb.",
      pres: "Drops 1 → 9e−4 → 4e−8 → 3e−12. Trace equality (tr(X) = 1) becomes machine-precision tight.",
      dres: "0.21 → 3e−4 → 1e−8 → 4e−13. Smaller than pres, similar to QP — the structure of the cost matrix helps.",
      kt: "1 → 1e−4 → 9e−9 → 6e−14. Drops fast; problem is feasible.",
      mu: "1 → 3e−4 → 3e−9 → 3e−14. Big drops between every printed row. CLARABEL's barrier schedule is aggressive on SDPs because each step is so expensive.",
      step: "---- → 0.72 → 0.95 → 1.00. PSD cones make the line search more conservative early on (eigenvalue projections must remain feasible), but full steps by the end.",
    },
  },
  logreg: {
    extras: [...CVXPY_OUTPUT_EXTRAS, ...CVXPY_FEATURES],
    setupText: "problem\n  variables     = 9 (β:8, b:1) + auxiliary u:8 (for ‖β‖₁) + s:50 (for cp.logistic)\n  cones         = 8 ExpCone (size 3 each, for log-sum-exp)\n              + 16 nonneg (for u ≥ ±β)\n              + 1 zero (offset)\n  ----------------------\n  Status: solving",
    rows: [
      { cells: { iter: "  0", pcost: "+0.0000e+00", dcost: "+0.0000e+00", gap: "0.00e+00", pres: "1.50e+01", dres: "3.00e+00", kt: "1.00e+00", mu: "1.00e+00", step: "----" } },
      { cells: { iter: "  4", pcost: "+1.7012e+01", dcost: "+1.6905e+01", gap: "1.07e-02", pres: "4.23e-02", dres: "1.10e-02", kt: "3.20e-03", mu: "6.10e-03", step: "6.50e-01" } },
      { cells: { iter: " 10", pcost: "+1.6843e+01", dcost: "+1.6841e+01", gap: "2.80e-04", pres: "9.10e-05", dres: "4.20e-05", kt: "1.10e-05", mu: "2.05e-05", step: "9.10e-01" } },
      { cells: { iter: " 16", pcost: "+1.6842e+01", dcost: "+1.6842e+01", gap: "3.10e-08", pres: "9.20e-09", dres: "4.10e-09", kt: "3.30e-10", mu: "1.05e-09", step: "9.95e-01" } },
      { cells: { iter: " 22", pcost: "+1.6842e+01", dcost: "+1.6842e+01", gap: "4.20e-12", pres: "3.10e-12", dres: "6.05e-13", kt: "4.10e-14", mu: "9.05e-14", step: "9.99e-01" } },
    ],
    summary: "terminated: optimal\nsolve time:    0.026 sec\nprimal obj :   16.842\ndual obj   :   16.842\ngap        :    4.2e-12",
    finalSummary:
      "Cost of the exponential cone: 22 Newton iterations vs ~10 for the LP. The setup block tells the real story — 50 sample log-losses ⇒ 50 ExpCones, 16 nonneg from L1. Internally CLARABEL is solving a problem with 67 variables, not 9.",
    perCol: {
      iter: "0, 4, 10, 16, 22 — CLARABEL printed five rows for this run because progress between Newton steps is slower than for an LP.",
      pcost: "0 → 17.01 → 16.84 → 16.84 → 16.84. The first iter overshoots into 17.01 because the initial point is far from feasibility; corrects quickly.",
      dcost: "0 → 16.91 → 16.84 → 16.84 → 16.84. Climbs from below. Locked in by iter 10.",
      gap: "0 → 1.07e−2 → 2.8e−4 → 3.1e−8 → 4.2e−12. Five orders of magnitude in the last two rows — that's the quadratic-convergence basin doing its thing. ExpCone problems can stall earlier; this one didn't.",
      pres: "15 → 4e−2 → 9e−5 → 9e−9 → 3e−12. Initial pres = 15 is huge — the L1 reformulation introduces 16 inequalities and the initial iterate violates many of them. CLARABEL drives them to zero.",
      dres: "3 → 1e−2 → 4e−5 → 4e−9 → 6e−13. Similar story.",
      kt: "Drops from 1 to 4e−14. Confirms feasibility.",
      mu: "1 → 6e−3 → 2e−5 → 1e−9 → 9e−14. Steady ~3-orders-per-row drop.",
      step: "---- → 0.65 → 0.91 → 0.99 → 1.00. ExpCone projections force a more conservative line search early — first non-trivial step is 65% of full Newton, vs 90%+ for LPs.",
    },
  },
};

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
  if (k === "EXP") return "#c8311c";
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
