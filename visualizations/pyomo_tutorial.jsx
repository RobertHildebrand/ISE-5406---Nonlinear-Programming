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
   PYOMO + IPOPT — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   Three problems show how to express different NLPs in Pyomo
   and solve them with IPOPT. Step through the code line by
   line; the right panel shows the model being built up
   (variables, objective, constraints) and finally the solver
   output.

   The Python here is real — paste it into a notebook with
   pyomo + ipopt installed and it runs. The "solver output"
   is hardcoded so the demo runs in the browser.
   ============================================================ */

// ============================================================
// Problem registry
// Each problem provides: code lines (1-indexed), event sequence,
// and the canned IPOPT output to display.
// ============================================================

const PROB_QP = {
  key: "qp",
  name: "Constrained QP — Closest Feasible Point",
  blurb:
    "Minimize the squared distance from (1, 2), but you can only choose points with x, y ≥ 0 and x + y ≤ 1. The unconstrained minimum lies outside the feasible set, so the budget constraint binds at the optimum.",
  formula: "min  (x − 1)² + (y − 2)²    s.t.   x ≥ 0,  y ≥ 0,  x + y ≤ 1",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "model = ConcreteModel()",
    "model.x = Var(initialize=0.5, bounds=(0, None))",
    "model.y = Var(initialize=0.5, bounds=(0, None))",
    "model.obj = Objective(",
    "    expr=(model.x - 1)**2 + (model.y - 2)**2",
    ")",
    "model.budget = Constraint(expr=model.x + model.y <= 1)",
    "",
    "solver = SolverFactory('ipopt')",
    "result = solver.solve(model, tee=False)",
    "",
    "print('status      :', result.solver.status)",
    "print('termination :', result.solver.termination_condition)",
    "print('x*          =', value(model.x))",
    "print('y*          =', value(model.y))",
    "print('obj*        =', value(model.obj))",
  ],
  events: [
    { line: 1, kind: "import", note: "Pull every public name from pyomo.environ. This is the standard 'star import' for Pyomo — gives you ConcreteModel, Var, Objective, Constraint, SolverFactory, value, Set, etc." },
    { line: 3, kind: "create_model", note: "Build a ConcreteModel — a model where every set / parameter / variable index is fixed at construction time. (Use AbstractModel only when you want to instantiate the same template against multiple data files.)" },
    { line: 4, kind: "add_var", payload: { name: "x", init: 0.5, lb: 0, ub: "None" }, note: "Declare scalar variable x. initialize sets the value IPOPT starts from; bounds=(0, None) means x ≥ 0 with no upper bound. The starting point matters for nonconvex problems." },
    { line: 5, kind: "add_var", payload: { name: "y", init: 0.5, lb: 0, ub: "None" }, note: "Same for y. Both starting points lie inside the feasible region — IPOPT requires a strictly feasible start." },
    { line: 6, kind: "set_objective", payload: { expr: "(x − 1)² + (y − 2)²", sense: "minimize" }, note: "Define the objective. Pyomo's '**' operator builds a symbolic expression — no numeric evaluation yet. Default sense is minimize." },
    { line: 9, kind: "add_constraint", payload: { name: "budget", expr: "x + y ≤ 1" }, note: "Inequality constraint. Pyomo recognizes <=, >=, == in expr=. Each constraint must be attached to model.<name> so Pyomo can find it during solve." },
    { line: 11, kind: "create_solver", payload: { name: "ipopt" }, note: "Look up the IPOPT solver. SolverFactory needs the ipopt binary on your PATH (or set executable=...). Other choices: 'glpk', 'cbc', 'gurobi', 'bonmin', 'couenne'." },
    { line: 12, kind: "solve", payload: { iters: 12, time: 0.014, status: "ok", term: "optimal", obj: 2.0, vars: { x: 0.0, y: 1.0 } }, note: "Solve. tee=False suppresses the IPOPT iteration log; set tee=True in class to show the actual line-by-line log. The result object holds status / termination — they're not the answer; the answer lives on model itself." },
    { line: 14, kind: "print", payload: { text: "status      : ok" }, note: "result.solver.status is a coarse outcome ('ok', 'warning', 'error'). For real diagnostics, look at termination_condition." },
    { line: 15, kind: "print", payload: { text: "termination : optimal" }, note: "termination_condition can be 'optimal', 'infeasible', 'maxIterations', 'unbounded', 'feasible', etc. Check this in production code." },
    { line: 16, kind: "print", payload: { text: "x*          = 0.0" }, note: "value(model.x) extracts the float from the symbolic variable. After solve() the variable's .value is set to the optimum." },
    { line: 17, kind: "print", payload: { text: "y*          = 1.0" }, note: "Both budget (x + y ≤ 1) and the lower bound x ≥ 0 are active here. The KKT multiplier on budget is 2 (the marginal cost of tightening the budget by one unit)." },
    { line: 18, kind: "print", payload: { text: "obj*        = 2.0" }, note: "value(model.obj) evaluates the objective at the current variable values. (1, 2) is two units away from (0, 1) — squared distance 2. Done." },
  ],
};

const PROB_DISK = {
  key: "disk",
  name: "NLP — Closest Point on a Disk",
  blurb:
    "Find the point in the disk centered at (2, 0) of radius 1 that's closest to the origin. Quadratic objective, quadratic inequality constraint — a true (non-LP) NLP.",
  formula: "min  x² + y²   s.t.   (x − 2)² + y² ≤ 1",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "model = ConcreteModel()",
    "model.x = Var(initialize=2.0)",
    "model.y = Var(initialize=0.5)",
    "model.obj = Objective(expr=model.x**2 + model.y**2)",
    "model.disk = Constraint(",
    "    expr=(model.x - 2)**2 + model.y**2 <= 1",
    ")",
    "",
    "solver = SolverFactory('ipopt')",
    "result = solver.solve(model)",
    "",
    "print('x* =', value(model.x))",
    "print('y* =', value(model.y))",
    "print('dist² =', value(model.obj))",
    "",
    "# Dual on the disk constraint (Lagrange multiplier)",
    "model.dual = Suffix(direction=Suffix.IMPORT)",
    "result = solver.solve(model)",
    "print('λ_disk =', model.dual[model.disk])",
  ],
  events: [
    { line: 1, kind: "import", note: "Standard star import." },
    { line: 3, kind: "create_model", note: "Concrete model again." },
    { line: 4, kind: "add_var", payload: { name: "x", init: 2.0, lb: "None", ub: "None" }, note: "Free scalar. No bounds — IPOPT can take any real x." },
    { line: 5, kind: "add_var", payload: { name: "y", init: 0.5, lb: "None", ub: "None" }, note: "Free scalar. Initial point (2.0, 0.5) is strictly inside the disk: (2-2)² + 0.5² = 0.25 ≤ 1. Good — IPOPT needs a strictly feasible start (with mild tolerance)." },
    { line: 6, kind: "set_objective", payload: { expr: "x² + y²", sense: "minimize" }, note: "Distance squared from origin. Smooth & strictly convex — easy for IPOPT." },
    { line: 7, kind: "add_constraint", payload: { name: "disk", expr: "(x − 2)² + y² ≤ 1" }, note: "Quadratic inequality. The feasible set is convex (a disk) but the constraint function isn't linear — this needs a real NLP solver, not LP." },
    { line: 11, kind: "create_solver", payload: { name: "ipopt" }, note: "IPOPT — interior-point. The barrier method visualization elsewhere in this collection shows what's happening inside this call." },
    { line: 12, kind: "solve", payload: { iters: 9, time: 0.011, status: "ok", term: "optimal", obj: 1.0, vars: { x: 1.0, y: 0.0 } }, note: "Solve. From (2.0, 0.5), IPOPT marches toward the origin, hits the disk boundary, and slides along it to (1, 0)." },
    { line: 14, kind: "print", payload: { text: "x* = 1.0" }, note: "On the boundary of the disk: (1-2)² + 0² = 1. ✓" },
    { line: 15, kind: "print", payload: { text: "y* = 0.0" }, note: "Symmetry: the closest point lies on the x-axis between origin and disk center." },
    { line: 16, kind: "print", payload: { text: "dist² = 1.0" }, note: "Squared distance from origin to (1, 0) is 1 — and the disk boundary is at distance 1 from origin along the x-axis. Tight." },
    { line: 19, kind: "add_suffix", payload: { name: "dual", direction: "IMPORT" }, note: "Suffix is Pyomo's machinery for solver-side data (duals, slacks, etc.). direction=IMPORT means 'pull values back from the solver after solve'." },
    { line: 20, kind: "solve", payload: { iters: 1, time: 0.003, status: "ok", term: "optimal", obj: 1.0, vars: { x: 1.0, y: 0.0 }, duals: { disk: 1.0 } }, note: "Re-solve so IPOPT can fill in the dual. This is the only reliable way to get duals out of pyomo + ipopt — declare Suffix BEFORE solving." },
    { line: 21, kind: "print", payload: { text: "λ_disk = 1.0" }, note: "The Lagrange multiplier on the disk constraint. Interpret: tightening the disk radius by 1 unit would increase the optimum value by ~1 unit (the rate, locally)." },
  ],
};

const PROB_PORTFOLIO = {
  key: "portfolio",
  name: "QP — Markowitz Portfolio",
  blurb:
    "Minimize portfolio variance subject to a target expected return and budget. Indexed variables, sums-over-sets, equality and inequality constraints — the idiomatic Pyomo flavor for medium-size NLPs.",
  formula:
    "min  Σᵢ σᵢ² wᵢ²   s.t.   Σᵢ μᵢ wᵢ ≥ 0.08,   Σᵢ wᵢ = 1,   wᵢ ∈ [0, 1]",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "assets = ['AAPL', 'BND', 'GLD', 'XLE']",
    "mu     = {'AAPL': 0.12, 'BND': 0.03, 'GLD': 0.07, 'XLE': 0.10}",
    "sigma  = {'AAPL': 0.20, 'BND': 0.04, 'GLD': 0.12, 'XLE': 0.18}",
    "TARGET_RETURN = 0.08",
    "",
    "model = ConcreteModel()",
    "model.A = Set(initialize=assets)",
    "model.w = Var(model.A, bounds=(0, 1), initialize=0.25)",
    "",
    "model.variance = Objective(",
    "    expr=sum(sigma[a]**2 * model.w[a]**2 for a in model.A)",
    ")",
    "model.return_floor = Constraint(",
    "    expr=sum(mu[a] * model.w[a] for a in model.A) >= TARGET_RETURN",
    ")",
    "model.budget = Constraint(",
    "    expr=sum(model.w[a] for a in model.A) == 1",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "for a in assets:",
    "    print(f'{a}: w = {value(model.w[a]):.4f}')",
    "print('variance =', value(model.variance))",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "assets", value: "['AAPL', 'BND', 'GLD', 'XLE']" }, note: "Plain Python list. Pyomo can take Python data directly — no need to wrap in Param objects unless you need solver-side parametric updates." },
    { line: 4, kind: "raw_data", payload: { label: "μ (expected return)", value: "{AAPL: 0.12, BND: 0.03, GLD: 0.07, XLE: 0.10}" }, note: "Returns dict keyed by asset name." },
    { line: 5, kind: "raw_data", payload: { label: "σ (vol)", value: "{AAPL: 0.20, BND: 0.04, GLD: 0.12, XLE: 0.18}" }, note: "Volatilities. We're treating assets as uncorrelated — variance is just Σ σᵢ² wᵢ². For correlated assets you'd use a covariance matrix and a quadratic form wᵀΣw." },
    { line: 6, kind: "raw_data", payload: { label: "TARGET_RETURN", value: "0.08" }, note: "Minimum annual expected return the portfolio must clear." },
    { line: 8, kind: "create_model" },
    { line: 9, kind: "add_set", payload: { name: "A", value: "{AAPL, BND, GLD, XLE}" }, note: "Pyomo Set. Indexed variables / constraints reference it. (You can also pass the list directly to Var(...), but a named Set keeps the model self-describing.)" },
    { line: 10, kind: "add_var", payload: { name: "w", indexed: "A", init: 0.25, lb: 0, ub: 1 }, note: "Indexed variable: model.w[a] for each a in A. Same bounds and init for every index — pass dicts to vary them per-index." },
    { line: 12, kind: "set_objective", payload: { expr: "Σₐ σ[a]² · w[a]²", sense: "minimize" }, note: "Generator expression sums over the set. Pyomo turns this into a single symbolic expression of degree 2 in w — IPOPT sees it as a smooth NLP." },
    { line: 15, kind: "add_constraint", payload: { name: "return_floor", expr: "Σₐ μ[a] · w[a] ≥ 0.08" }, note: "Linear inequality. Note we use Python's >= directly — Pyomo overloads it to build constraint expressions." },
    { line: 18, kind: "add_constraint", payload: { name: "budget", expr: "Σₐ w[a] = 1" }, note: "Equality constraint via ==. With both equality (budget) and inequality (return floor), IPOPT handles a mix without fuss." },
    { line: 22, kind: "solve", payload: { iters: 16, time: 0.022, status: "ok", term: "optimal", obj: 0.00547, vars: { w_AAPL: 0.239, w_BND: 0.222, w_GLD: 0.308, w_XLE: 0.232 } }, note: "Solve. SolverFactory(...).solve(model) is one-line shorthand. The QP is convex so IPOPT converges quickly." },
    { line: 24, kind: "print", payload: { text: "AAPL: w = 0.2390" } },
    { line: 24, kind: "print", payload: { text: "BND:  w = 0.2220" } },
    { line: 24, kind: "print", payload: { text: "GLD:  w = 0.3080" } },
    { line: 24, kind: "print", payload: { text: "XLE:  w = 0.2320" } },
    { line: 26, kind: "print", payload: { text: "variance = 0.005471" }, note: "Minimum-variance portfolio meeting the 0.08 return floor. Notice GLD (low vol, decent return) gets the largest weight." },
  ],
};

const PROBLEMS = [PROB_QP, PROB_DISK, PROB_PORTFOLIO];

// ============================================================
// State replay — fold events up to evIdx into a "model" object
// ============================================================
function replayState(events, upTo) {
  const s = {
    imported: false,
    model: false,
    rawData: [],
    sets: [],
    vars: [],
    objective: null,
    constraints: [],
    suffixes: [],
    solverName: null,
    result: null,
    duals: null,
    prints: [],
  };
  for (let i = 0; i <= upTo && i < events.length; i++) {
    const ev = events[i];
    switch (ev.kind) {
      case "import":
        s.imported = true;
        break;
      case "raw_data":
        s.rawData.push(ev.payload);
        break;
      case "create_model":
        s.model = true;
        break;
      case "add_set":
        s.sets.push(ev.payload);
        break;
      case "add_var":
        s.vars.push({ ...ev.payload });
        break;
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "add_suffix":
        s.suffixes.push(ev.payload);
        break;
      case "create_solver":
        s.solverName = ev.payload.name;
        break;
      case "solve":
        s.solverName = s.solverName || "ipopt";
        s.result = {
          status: ev.payload.status,
          term: ev.payload.term,
          iters: ev.payload.iters,
          time: ev.payload.time,
          obj: ev.payload.obj,
          vars: ev.payload.vars,
        };
        if (ev.payload.duals) s.duals = ev.payload.duals;
        // Bind values back onto vars
        for (const v of s.vars) {
          if (v.indexed) {
            const indexedKey = `w_${v.indexed}`; // not generic; portfolio specific
            v.values = {};
            for (const k in ev.payload.vars) {
              if (k.startsWith(v.name + "_")) {
                v.values[k.slice(v.name.length + 1)] = ev.payload.vars[k];
              }
            }
          } else if (ev.payload.vars[v.name] !== undefined) {
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
export default function PyomoTutorial() {
  const [probKey, setProbKey] = useState(PROB_QP.key);
  const problem = useMemo(
    () => PROBLEMS.find((p) => p.key === probKey),
    [probKey]
  );

  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);

  // When problem changes, reset position
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
  const state = useMemo(
    () => replayState(problem.events, evIdx),
    [problem, evIdx]
  );

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Pyomo + IPOPT — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Step through real Pyomo code line by line. The right panel shows the
        model being assembled — variables, objective, constraints — and the
        IPOPT solver output once <code style={inlineCode}>solver.solve(model)</code> runs.
        Pick a problem to switch examples.
      </p>

      <InstallPanel />

      {/* Problem tabs */}
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 14,
          flexWrap: "wrap",
        }}
      >
        {PROBLEMS.map((p) => (
          <button
            key={p.key}
            onClick={() => setProbKey(p.key)}
            style={{
              ...tabBtn,
              ...(p.key === probKey ? tabBtnActive : {}),
            }}
          >
            {p.name}
          </button>
        ))}
      </div>

      {/* Problem blurb */}
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
            whiteSpace: "pre-wrap",
          }}
        >
          {problem.formula}
        </div>
      </div>

      {/* Two-column layout */}
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
              {ev?.note ||
                "(no note for this line — keep stepping)"}
            </div>
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              marginTop: 14,
              flexWrap: "wrap",
            }}
          >
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
                background: "#1f4e3d",
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
        Install Pyomo + IPOPT &nbsp;
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
            Pyomo is just a Python package; IPOPT is a separate binary.
          </p>
          <Pre>
            {`# 1. Pyomo (any environment)
pip install pyomo

# 2. IPOPT itself — pick one
#    macOS:
brew install ipopt

#    conda (cross-platform, simplest):
conda install -c conda-forge ipopt

#    Ubuntu / Debian:
sudo apt install coinor-libipopt-dev

# 3. Verify
ipopt -v        # IPOPT prints its version
python -c "from pyomo.environ import SolverFactory; \\
           print(SolverFactory('ipopt').available())"`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            If <code style={inlineCode}>SolverFactory('ipopt').available()</code>{" "}
            returns <code style={inlineCode}>False</code>, Pyomo can't find the
            binary on PATH. Pass{" "}
            <code style={inlineCode}>executable='/full/path/to/ipopt'</code> to
            point at it explicitly.
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
              borderLeft: active
                ? "3px solid #f5a524"
                : "3px solid transparent",
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
// State panel
// ============================================================
function StatePanel({ state }) {
  return (
    <div style={statePanelOuter}>
      <Section title="Imports">
        {state.imported ? (
          <div style={chip("#1f4e3d")}>pyomo.environ ✓</div>
        ) : (
          <Empty />
        )}
      </Section>

      {state.rawData.length > 0 && (
        <Section title="Python data">
          {state.rawData.map((d, i) => (
            <KVRow key={i} k={d.label} v={d.value} mono />
          ))}
        </Section>
      )}

      <Section title="Model">
        {state.model ? (
          <div style={chip("#0b3da0")}>ConcreteModel</div>
        ) : (
          <Empty />
        )}
      </Section>

      {(state.sets.length > 0 || state.model) && state.sets.length > 0 && (
        <Section title="Sets">
          {state.sets.map((s, i) => (
            <KVRow key={i} k={`model.${s.name}`} v={s.value} mono />
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
                  model.{v.name}
                  {v.indexed ? `[${v.indexed}]` : ""}
                </span>
                <span style={{ fontSize: 11, color: "#666" }}>
                  Var{v.indexed ? " (indexed)" : ""}
                </span>
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "#555",
                  fontFamily: "monospace",
                  marginTop: 2,
                }}
              >
                init={v.init}, bounds=({fmt(v.lb)}, {fmt(v.ub)})
              </div>
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
                  value = {v.value}
                </div>
              )}
              {v.values && (
                <div
                  style={{
                    fontSize: 12,
                    color: "#c8311c",
                    fontFamily: "monospace",
                    marginTop: 2,
                  }}
                >
                  {Object.entries(v.values).map(([k, val]) => (
                    <div key={k}>
                      <b>
                        [{k}]
                      </b>{" "}
                      = {(+val).toFixed(4)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </Section>

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
            <span style={{ color: "#555", fontSize: 11 }}>
              {state.objective.sense}
            </span>
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
              <span style={{ color: "#555", fontSize: 11 }}>
                model.{c.name}
              </span>
              <div>{c.expr}</div>
            </div>
          ))
        )}
      </Section>

      {state.suffixes.length > 0 && (
        <Section title="Suffixes (solver attachments)">
          {state.suffixes.map((s, i) => (
            <KVRow
              key={i}
              k={`model.${s.name}`}
              v={`Suffix(direction=Suffix.${s.direction})`}
              mono
            />
          ))}
        </Section>
      )}

      <Section title="Solver">
        {state.solverName ? (
          <div style={chip("#7a3da0")}>SolverFactory('{state.solverName}')</div>
        ) : (
          <Empty />
        )}
      </Section>

      {state.result && (
        <Section title="IPOPT result">
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
            <ResultLine k="status" v={state.result.status} c="#7dd87d" />
            <ResultLine k="termination" v={state.result.term} c="#7dd87d" />
            <ResultLine k="iterations" v={state.result.iters} />
            <ResultLine k="time" v={`${state.result.time.toFixed(3)} s`} />
            <ResultLine
              k="objective"
              v={(+state.result.obj).toFixed(6)}
              c="#f5a524"
            />
            {Object.entries(state.result.vars).map(([k, v]) => (
              <ResultLine key={k} k={k} v={(+v).toFixed(4)} c="#f5a524" />
            ))}
            {state.duals && (
              <>
                <div
                  style={{
                    color: "#7f7864",
                    fontSize: 11,
                    margin: "6px 0 2px 0",
                  }}
                >
                  duals (Lagrange multipliers)
                </div>
                {Object.entries(state.duals).map(([k, v]) => (
                  <ResultLine key={k} k={`λ_${k}`} v={(+v).toFixed(4)} c="#9a4caa" />
                ))}
              </>
            )}
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
  };
}
function KVRow({ k, v, mono }) {
  return (
    <div
      style={{
        display: "flex",
        gap: 10,
        padding: "3px 0",
        borderBottom: "1px dotted #eee",
      }}
    >
      <span
        style={{
          color: "#666",
          fontSize: 12,
          minWidth: 130,
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
function fmt(x) {
  if (x === "None" || x == null) return "None";
  return x;
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
          <b>ConcreteModel vs AbstractModel.</b> Concrete is what you'll use 95%
          of the time — data is fixed at construction. Abstract is a template
          for repeating the same model with different .dat files.
        </li>
        <li>
          <b>Pyomo doesn't solve.</b> It builds expressions and ships them to a
          solver via NL/LP/MPS file (default for IPOPT: an .nl file). The solver
          binary does the work.
        </li>
        <li>
          <b>tee=True</b> in <code style={inlineCode}>solver.solve(model, tee=True)</code> streams
          the IPOPT iteration log live — show this in lecture so students see
          what an interior-point solver actually prints.
        </li>
        <li>
          <b>Duals require a Suffix.</b> By default, Pyomo throws away IPOPT's
          dual variables. Declare{" "}
          <code style={inlineCode}>model.dual = Suffix(direction=Suffix.IMPORT)</code>{" "}
          before solve to read them back.
        </li>
        <li>
          <b>Initial point matters.</b> IPOPT needs a strictly feasible
          start with respect to bounds. For nonconvex NLPs (not these
          examples), the initial point determines which local minimum you find.
        </li>
        <li>
          <b>Indexed variables and sums.</b> Generator expressions inside{" "}
          <code style={inlineCode}>sum(...)</code> are how you express{" "}
          <em>any</em> "for all i" structure — objectives, constraints, even
          multi-dimensional indices.
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
};
const tabBtnActive = {
  background: "#1f4e3d",
  color: "#fff",
  border: "1px solid #1f4e3d",
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
