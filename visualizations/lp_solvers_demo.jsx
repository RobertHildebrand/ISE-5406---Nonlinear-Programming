import React, { useState } from "react";
import { Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   LP SOLVERS — same problem, four languages
   ISE 5406

   PuLP, AMPL, Gurobi (gurobipy), CPLEX (docplex). Each tab
   shows the SAME LP — wood/labor production planning — with
   identical results, so students can directly compare API
   ergonomics.
   ============================================================ */

// Problem (matches duality_sensitivity_demo so cross-references work):
//   max  3 x + 5 y
//   s.t. 2 x +   y <= 8     (wood)
//          x + 3 y <= 6     (labor)
//        x, y >= 0
// Optimum: (x, y) = (3.6, 0.8), z = 14.8
// Duals : pi_wood = 0.8, pi_labor = 1.4

const LANGS = [
  { key: "pulp", label: "PuLP", color: "#1f4e3d" },
  { key: "ampl", label: "AMPL", color: "#0b3da0" },
  { key: "gurobi", label: "Gurobi (gurobipy)", color: "#a40000" },
  { key: "cplex", label: "CPLEX (docplex)", color: "#7a3da0" },
];

const CODES = {
  pulp: `from pulp import LpProblem, LpVariable, LpMaximize, LpStatus

m = LpProblem("production", LpMaximize)

x = LpVariable("x", lowBound=0)
y = LpVariable("y", lowBound=0)

# Objective
m += 3*x + 5*y, "profit"

# Constraints (named so we can read duals back)
m += 2*x + 1*y <= 8, "wood"
m += 1*x + 3*y <= 6, "labor"

# Default solver is CBC (open source)
m.solve()

print("status :", LpStatus[m.status])
print("z*     :", m.objective.value())
print("x*, y* :", x.value(), y.value())

# Duality / sensitivity
for c in m.constraints.values():
    print(f"  {c.name}: π = {c.pi}, slack = {c.slack}")
for v in m.variables():
    print(f"  rc({v.name}) = {v.dj}")`,

  ampl: `# production.mod
var x >= 0;
var y >= 0;

maximize profit: 3*x + 5*y;

subject to wood:  2*x + 1*y <= 8;
subject to labor: 1*x + 3*y <= 6;

# production.run (driver)
model production.mod;
option solver gurobi;     # or cplex / cbc / highs
solve;

display x, y, profit;
display wood.dual, labor.dual;     # shadow prices
display wood.slack, labor.slack;
display x.rc, y.rc;                # reduced costs

# sensitivity ranges (with the right solver)
option presolve 0;
display wood.up, wood.down;        # RHS upper / lower range
display x.up, x.down;              # obj-coef upper / lower range`,

  gurobi: `import gurobipy as gp
from gurobipy import GRB

m = gp.Model("production")

x = m.addVar(name="x", lb=0)
y = m.addVar(name="y", lb=0)

m.setObjective(3*x + 5*y, GRB.MAXIMIZE)

wood  = m.addConstr(2*x + 1*y <= 8, "wood")
labor = m.addConstr(1*x + 3*y <= 6, "labor")

m.optimize()

print("status :", m.Status)             # 2 = OPTIMAL
print("z*     :", m.ObjVal)
print("x*, y* :", x.X, y.X)

# Duality
for c in m.getConstrs():
    print(f"  {c.ConstrName}: π = {c.Pi}, slack = {c.Slack}")
for v in m.getVars():
    print(f"  rc({v.VarName}) = {v.RC}")

# Sensitivity ranges (Gurobi-specific attributes)
for c in m.getConstrs():
    print(f"  {c.ConstrName}: RHS in [{c.SARHSLow}, {c.SARHSUp}]")
for v in m.getVars():
    print(f"  obj({v.VarName}) in [{v.SAObjLow}, {v.SAObjUp}]")`,

  cplex: `from docplex.mp.model import Model

m = Model(name="production")

x = m.continuous_var(name="x", lb=0)
y = m.continuous_var(name="y", lb=0)

m.maximize(3*x + 5*y)

wood  = m.add_constraint(2*x + 1*y <= 8, ctname="wood")
labor = m.add_constraint(1*x + 3*y <= 6, ctname="labor")

sol = m.solve()
print("status :", m.solve_details.status)
print("z*     :", sol.objective_value)
print("x*, y* :", sol[x], sol[y])

# Duality / sensitivity (linear-only, after solving an LP)
duals  = m.dual_values([wood, labor])
slacks = m.slack_values([wood, labor])
rcs    = m.reduced_costs([x, y])

for ct, pi, sl in zip([wood, labor], duals, slacks):
    print(f"  {ct.name}: π = {pi}, slack = {sl}")
for v, rc in zip([x, y], rcs):
    print(f"  rc({v.name}) = {rc}")

# Sensitivity ranges via the CPLEX engine
sa = m.sensitivity()
print(sa.constraints_rhs([wood, labor]))   # [(low, high), (low, high)]
print(sa.objective_coefs([x, y]))`,
};

const RESULT = {
  status: "OPTIMAL",
  z: 14.8,
  x: 3.6,
  y: 0.8,
  duals: { wood: 0.8, labor: 1.4 },
  slacks: { wood: 0.0, labor: 0.0 },
  rc: { x: 0.0, y: 0.0 },
  sa_rhs: { wood: [4.0, 12.0], labor: [4.0, 24.0] },
  sa_obj: { x: [1.667, 10.0], y: [1.5, 9.0] },
};

// ============================================================
// Main component
// ============================================================
export default function LpSolversDemo() {
  const [tab, setTab] = useState("pulp");
  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        LP Modelers — Same Problem, Four Languages
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        The classic production-planning LP, written four ways:
        PuLP (Python), AMPL (algebraic modeling language), Gurobi's native
        Python (gurobipy), and CPLEX's docplex. Identical math, identical
        answer, very different APIs. Switch tabs to see how each modeler
        spells out variables, constraints, solves, and dual extraction.
      </p>

      <div style={problemBox}>
        <Tex block>
          {String.raw`\begin{aligned} \max\;\; & 3 x + 5 y \\ \text{s.t.}\;\; & 2 x + y \le 8 \quad (\text{wood}) \\ & x + 3 y \le 6 \quad (\text{labor}) \\ & x, y \ge 0 \end{aligned}`}
        </Tex>
        <div style={{ fontSize: 13, color: "#444", marginTop: 6 }}>
          Optimum: <Tex>{`(x^\\star, y^\\star) = (3.6, 0.8),\\;\\; z^\\star = 14.8`}</Tex>.
          Both constraints are tight, so the slacks are zero and the
          reduced costs of <Tex>{`x`}</Tex> and{" "}
          <Tex>{`y`}</Tex> are zero.
        </div>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {LANGS.map((l) => (
          <button
            key={l.key}
            onClick={() => setTab(l.key)}
            style={{
              padding: "8px 14px",
              border: "1px solid #ccc",
              borderRadius: 6,
              cursor: "pointer",
              background: tab === l.key ? l.color : "#fff",
              color: tab === l.key ? "#fff" : "#222",
              fontWeight: tab === l.key ? 700 : 500,
            }}
          >
            {l.label}
          </button>
        ))}
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
          <div style={{ fontSize: 12, fontFamily: "monospace", color: "#888", marginBottom: 4 }}>
            {LANGS.find((l) => l.key === tab).label} model
          </div>
          <pre
            style={{
              background: "#1f1d1a",
              color: "#e8e2d4",
              padding: 14,
              borderRadius: 8,
              fontSize: 12,
              fontFamily: "'JetBrains Mono', Menlo, monospace",
              lineHeight: 1.55,
              whiteSpace: "pre",
              overflowX: "auto",
              margin: 0,
              maxHeight: 700,
              overflowY: "auto",
            }}
          >
            {CODES[tab]}
          </pre>
        </div>
        <div>
          <ResultPanel />
          <InstallNotes tab={tab} />
        </div>
      </div>

      <ApiCheatSheet />
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Result panel — same numbers regardless of language
// ============================================================
function ResultPanel() {
  return (
    <div style={panel}>
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
        Result (same in every modeler)
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 13, borderCollapse: "collapse" }}>
        <tbody>
          <KV k="status" v={RESULT.status} />
          <KV k="z*" v={RESULT.z.toFixed(4)} highlight />
          <KV k="x*" v={RESULT.x.toFixed(4)} />
          <KV k="y*" v={RESULT.y.toFixed(4)} />
          <KV k="π (wood)" v={RESULT.duals.wood.toFixed(4)} accent="#0b3da0" />
          <KV k="π (labor)" v={RESULT.duals.labor.toFixed(4)} accent="#7a3da0" />
          <KV k="slack (wood)" v={RESULT.slacks.wood.toFixed(4)} />
          <KV k="slack (labor)" v={RESULT.slacks.labor.toFixed(4)} />
          <KV k="rc (x)" v={RESULT.rc.x.toFixed(4)} />
          <KV k="rc (y)" v={RESULT.rc.y.toFixed(4)} />
        </tbody>
      </table>
      <div style={{ marginTop: 10, fontFamily: "monospace", fontSize: 11, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase" }}>
        Sensitivity ranges
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 13, borderCollapse: "collapse" }}>
        <tbody>
          <KV k="b₁ (wood) RHS range" v={`[${RESULT.sa_rhs.wood[0]}, ${RESULT.sa_rhs.wood[1]}]`} />
          <KV k="b₂ (labor) RHS range" v={`[${RESULT.sa_rhs.labor[0]}, ${RESULT.sa_rhs.labor[1]}]`} />
          <KV k="c_x (obj) range" v={`[${RESULT.sa_obj.x[0]}, ${RESULT.sa_obj.x[1]}]`} />
          <KV k="c_y (obj) range" v={`[${RESULT.sa_obj.y[0]}, ${RESULT.sa_obj.y[1]}]`} />
        </tbody>
      </table>
    </div>
  );
}

function KV({ k, v, highlight, accent }) {
  return (
    <tr style={{ borderBottom: "1px dotted #eee" }}>
      <td style={{ padding: "3px 6px", color: "#666" }}>{k}</td>
      <td
        style={{
          padding: "3px 6px",
          textAlign: "right",
          color: accent || (highlight ? "#c8311c" : "#222"),
          fontWeight: highlight ? 700 : 400,
        }}
      >
        {v}
      </td>
    </tr>
  );
}

// ============================================================
// Per-language install notes
// ============================================================
function InstallNotes({ tab }) {
  const notes = {
    pulp: {
      title: "PuLP",
      body: `pip install pulp

# Default solver is CBC (bundled). For Gurobi/CPLEX:
m.solve(GUROBI(msg=0))
m.solve(CPLEX_CMD())
m.solve(HiGHS())   # the new fast open-source default`,
      tip: "PuLP uses Python's overloaded operators (≤ via <=) and adds constraints with += on the model object. Free, embedded CBC, easy ramp-up — the default 'first taste' modeler in OR courses.",
    },
    ampl: {
      title: "AMPL",
      body: `# AMPL is a separate language — install via:
# (1) Free 'community edition' from ampl.com
# (2) Conda: conda install -c conda-forge ampl
# (3) Python wrapper: pip install amplpy

# Run from CLI:
ampl production.run

# Or from Python via amplpy:
from amplpy import AMPL
ampl = AMPL()
ampl.read("production.mod")
ampl.solve(solver="gurobi")`,
      tip: "AMPL is an algebraic modeling language — the model is data-independent; the same .mod file runs on different .dat files. Industry standard for huge models with set-indexed structure. Steeper learning curve but unmatched scalability.",
    },
    gurobi: {
      title: "Gurobi",
      body: `pip install gurobipy

# Free academic license at portal.gurobi.com:
grbgetkey YOUR-LICENSE-KEY    # writes ~/gurobi.lic

# Verify
python -c "import gurobipy as gp; gp.Model().optimize()"

# Restricted-size models (≤2000 vars/cons) work without a license.`,
      tip: "Gurobi's native API is fast and feature-rich — sensitivity attributes (.SARHSLow, .SAObjUp), warm starts (.Start), callbacks, branching priorities. Industry standard for commercial MIP.",
    },
    cplex: {
      title: "CPLEX (docplex)",
      body: `pip install docplex   # the modern Python wrapper

# Free for academics: ibm.com/academic
# Or use the Community Edition (≤1000 vars/cons).

# CPLEX engine itself comes from a separate install
# (CPLEX Studio). docplex's pip wheel bundles a tiny
# embedded engine for the community edition.`,
      tip: "CPLEX's docplex is the modern object-oriented Python interface. Sensitivity analysis comes through the Sensitivity object (m.sensitivity()) returning structured ranges. Strongest at extremely large LP/QP and on warm-start-heavy workloads.",
    },
  };
  const note = notes[tab];
  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
        Install / setup — {note.title}
      </div>
      <pre
        style={{
          background: "#1f1d1a",
          color: "#e8e2d4",
          padding: 10,
          borderRadius: 6,
          fontSize: 11,
          fontFamily: "'JetBrains Mono', Menlo, monospace",
          lineHeight: 1.55,
          whiteSpace: "pre",
          overflowX: "auto",
          margin: "6px 0",
        }}
      >
        {note.body}
      </pre>
      <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>{note.tip}</div>
    </div>
  );
}

// ============================================================
// API cheat sheet — mapping common operations
// ============================================================
function ApiCheatSheet() {
  const ROWS = [
    ["create model", "LpProblem(...)", "model production.mod;", "gp.Model(...)", "Model(...)"],
    ["continuous var ≥ 0", "LpVariable(name, lowBound=0)", "var x >= 0;", "m.addVar(name=...)", "m.continuous_var(name=...)"],
    ["binary var", "LpVariable(name, cat='Binary')", "var x binary;", "m.addVar(vtype=GRB.BINARY)", "m.binary_var(name=...)"],
    ["objective", "m += 3*x + 5*y", "maximize z: 3*x + 5*y;", "m.setObjective(3*x + 5*y, GRB.MAXIMIZE)", "m.maximize(3*x + 5*y)"],
    ["constraint", "m += 2*x + y <= 8", "subject to c: 2*x + y <= 8;", "m.addConstr(2*x + y <= 8)", "m.add_constraint(2*x + y <= 8)"],
    ["solve", "m.solve()", "solve;", "m.optimize()", "m.solve()"],
    ["primal value", "x.value()", "x.val", "x.X", "sol[x]"],
    ["objective value", "m.objective.value()", "z.val (or z)", "m.ObjVal", "sol.objective_value"],
    ["dual / shadow price", "c.pi", "c.dual", "c.Pi", "m.dual_values([c])"],
    ["slack", "c.slack", "c.slack", "c.Slack", "m.slack_values([c])"],
    ["reduced cost", "v.dj", "v.rc", "v.RC", "m.reduced_costs([v])"],
    ["RHS sens. range", "(not built-in)", "c.up / c.down", "c.SARHSLow / .SARHSUp", "m.sensitivity().constraints_rhs(...)"],
    ["obj coef sens.", "(not built-in)", "v.up / v.down", "v.SAObjLow / .SAObjUp", "m.sensitivity().objective_coefs(...)"],
  ];
  return (
    <div style={{ ...panel, marginTop: 18 }}>
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
        API cheat sheet
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", fontFamily: "monospace", fontSize: 12, borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "#f0f0f0" }}>
              <th style={cellHead}>operation</th>
              <th style={cellHead}>PuLP</th>
              <th style={cellHead}>AMPL</th>
              <th style={cellHead}>gurobipy</th>
              <th style={cellHead}>docplex</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row, i) => (
              <tr key={i} style={{ borderBottom: "1px dotted #eee" }}>
                {row.map((cell, j) => (
                  <td key={j} style={j === 0 ? { ...cell, fontWeight: 700, padding: 6 } : { padding: 6 }}>
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const cellHead = { padding: 6, textAlign: "left", fontWeight: 700, borderBottom: "1px solid #ccc" };

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
          <b>Modeler vs solver.</b> All four wrappers ultimately ship the
          model to a SOLVER (CBC, HiGHS, Gurobi, CPLEX, …). Modelers
          handle syntax, indexing, sparsity. Solvers handle the math.
        </li>
        <li>
          <b>Pick PuLP or AMPL for teaching.</b> PuLP is one pip install
          away — perfect for homework. AMPL has the cleanest mathematical
          syntax and runs on any backend (free for academic use), making
          it ideal for showcasing structure.
        </li>
        <li>
          <b>Pick gurobipy or docplex for production.</b> Both expose
          attributes/methods for solver-specific features (callbacks,
          warm starts, sensitivity ranges) that PuLP and AMPL don't —
          essential when you're squeezing a real model.
        </li>
        <li>
          <b>Sensitivity is solver-specific.</b> PuLP returns duals
          (.pi) but doesn't compute SAOBJ/SARHS ranges. Use Gurobi or
          CPLEX (or AMPL with a solver that supports them) when you
          need the full sensitivity analysis.
        </li>
        <li>
          <b>Open-source alternatives.</b> HiGHS (highspy), SciPy linprog,
          Pyomo + GLPK, CVXPY + CLARABEL all solve LPs without commercial
          licenses. HiGHS in particular is competitive with Gurobi on
          many benchmarks.
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
