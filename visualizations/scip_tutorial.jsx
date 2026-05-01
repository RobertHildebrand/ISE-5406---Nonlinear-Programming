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
   SCIP (PySCIPOpt) — CODE STEPPER TUTORIAL
   ISE 5406

   SCIP is a state-of-the-art branch-and-bound / branch-and-cut
   solver for MIP and MINLP. Three problems:
     • Knapsack       (MILP)
     • Set cover      (MILP, indicator + cover constraints)
     • Rectangle      (MINLP — bilinear constraint on integers)
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_KNAPSACK = {
  key: "knapsack",
  name: "Knapsack",
  kind: "MILP",
  blurb:
    "Pure 0/1 knapsack: pick a subset of items maximizing value subject to a weight cap. The textbook MILP. SCIP solves it by branch-and-bound on the LP relaxation.",
  formula:
    "max  Σ vᵢ xᵢ   s.t.   Σ wᵢ xᵢ ≤ 50,   xᵢ ∈ {0, 1}",
  code: [
    null,
    "from pyscipopt import Model, quicksum",
    "",
    "values   = [60, 100, 120, 80]",
    "weights  = [10,  20,  30, 25]",
    "capacity = 50",
    "",
    "m = Model('knapsack')",
    "",
    "x = [m.addVar(vtype='B', name=f'x_{i}')",
    "     for i in range(len(values))]",
    "",
    "m.setObjective(",
    "    quicksum(values[i] * x[i] for i in range(len(values))),",
    "    'maximize',",
    ")",
    "m.addCons(",
    "    quicksum(weights[i] * x[i] for i in range(len(values))) <= capacity",
    ")",
    "",
    "m.optimize()",
    "",
    "print('status :', m.getStatus())",
    "print('obj    :', m.getObjVal())",
    "for i, xi in enumerate(x):",
    "    print(f'  item {i}: x = {m.getVal(xi):.0f}')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "pyscipopt", what: "Model, quicksum" }, note: "PySCIPOpt is the official Python interface to SCIP. Model is the problem container; quicksum is a faster substitute for Python's built-in sum()." },
    { line: 3, kind: "raw_data", payload: { label: "values", value: "[60, 100, 120, 80]", desc: "per-item value" } },
    { line: 4, kind: "raw_data", payload: { label: "weights", value: "[10, 20, 30, 25]", desc: "per-item weight" } },
    { line: 5, kind: "raw_data", payload: { label: "capacity", value: "50", desc: "knapsack capacity" } },
    { line: 7, kind: "create_model", payload: { name: "knapsack" }, note: "An empty SCIP Model. The string is just a name for the LP/MPS file SCIP will write internally." },
    { line: 9, kind: "add_vars", payload: { name: "x", count: 4, vtype: "B", desc: "binary (0/1) variables, one per item" }, note: "vtype='B' = binary. Other choices: 'I' = integer, 'C' = continuous (default). For each Var, SCIP knows it must branch on integer values." },
    { line: 12, kind: "set_objective", payload: { sense: "maximize", expr: "Σ valuesᵢ · xᵢ" }, note: "quicksum is to Python's sum what numpy.sum is to a list comprehension — much faster when building thousands of terms." },
    { line: 16, kind: "add_constraint", payload: { name: "weight_cap", expr: "Σ weightsᵢ · xᵢ ≤ 50", kind: "linear inequality" }, note: "addCons takes a Python expression built from Var objects. Operator overloading (≤, ≥, ==) builds the constraint." },
    { line: 20, kind: "solve", payload: {
        status: "optimal",
        nodes: 1,
        time: 0.002,
        gap: 0.0,
        obj: 220.0,
        primalbound: 220.0,
        dualbound: 220.0,
        vars: { "x_0": 0, "x_1": 1, "x_2": 1, "x_3": 0 },
      }, note: "SCIP solves at the root node — the LP relaxation already integers out. Primal bound = 220 (best feasible), dual bound = 220 (best LP bound) → gap = 0%, optimal." },
    { line: 22, kind: "print", payload: { text: "status : optimal" } },
    { line: 23, kind: "print", payload: { text: "obj    : 220.0" } },
    { line: 24, kind: "print", payload: { text: "  item 0: x = 0" } },
    { line: 24, kind: "print", payload: { text: "  item 1: x = 1" } },
    { line: 24, kind: "print", payload: { text: "  item 2: x = 1" } },
    { line: 24, kind: "print", payload: { text: "  item 3: x = 0" }, note: "Items 1 (w=20, v=100) and 2 (w=30, v=120). Total weight 50 = capacity. Total value 220." },
  ],
};

const PROB_SETCOVER = {
  key: "setcover",
  name: "Set Cover",
  kind: "MILP",
  blurb:
    "Cover every item with at least one selected set. Indicator variables, one constraint per item. Classic MILP that's NP-hard in general but tiny instances are trivial for SCIP.",
  formula:
    "min  Σⱼ cⱼ yⱼ   s.t.   Σ_{j: i ∈ Sⱼ} yⱼ ≥ 1  ∀i,   yⱼ ∈ {0, 1}",
  code: [
    null,
    "from pyscipopt import Model, quicksum",
    "",
    "items = [0, 1, 2, 3, 4]",
    "sets  = {",
    "    'A': [0, 1, 2],   # cost 6",
    "    'B': [1, 3],      # cost 4",
    "    'C': [2, 4],      # cost 5",
    "    'D': [0, 3, 4],   # cost 7",
    "    'E': [3, 4],      # cost 3",
    "}",
    "cost = {'A': 6, 'B': 4, 'C': 5, 'D': 7, 'E': 3}",
    "",
    "m = Model('setcover')",
    "y = {j: m.addVar(vtype='B', name=f'y_{j}') for j in sets}",
    "",
    "m.setObjective(",
    "    quicksum(cost[j] * y[j] for j in sets),",
    "    'minimize',",
    ")",
    "",
    "for i in items:",
    "    covers = [j for j, S in sets.items() if i in S]",
    "    m.addCons(quicksum(y[j] for j in covers) >= 1,",
    "              name=f'cover_{i}')",
    "",
    "m.optimize()",
    "",
    "for j in sets:",
    "    chosen = m.getVal(y[j])",
    "    if chosen > 0.5:",
    "        print(f'  pick set {j} (cost {cost[j]})')",
    "print('total cost =', m.getObjVal())",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "pyscipopt", what: "Model, quicksum" } },
    { line: 3, kind: "raw_data", payload: { label: "items", value: "[0, 1, 2, 3, 4]" } },
    { line: 4, kind: "raw_data", payload: { label: "sets", value: "5 sets covering subsets of items" } },
    { line: 12, kind: "raw_data", payload: { label: "cost", value: "{A:6, B:4, C:5, D:7, E:3}" } },
    { line: 14, kind: "create_model", payload: { name: "setcover" } },
    { line: 15, kind: "add_vars", payload: { name: "y", count: 5, vtype: "B", desc: "1 if set is selected" }, note: "Dict comprehension over set names — SCIP doesn't care about iteration order." },
    { line: 17, kind: "set_objective", payload: { sense: "minimize", expr: "Σⱼ cost[j] · y[j]" } },
    { line: 22, kind: "add_constraint", payload: { name: "cover_0", expr: "y_A + y_D ≥ 1", kind: "covering" }, note: "Item 0 is in sets A and D. At least one of {y_A, y_D} must be 1. We add one constraint per item." },
    { line: 22, kind: "add_constraint", payload: { name: "cover_1", expr: "y_A + y_B ≥ 1", kind: "covering" } },
    { line: 22, kind: "add_constraint", payload: { name: "cover_2", expr: "y_A + y_C ≥ 1", kind: "covering" } },
    { line: 22, kind: "add_constraint", payload: { name: "cover_3", expr: "y_B + y_D + y_E ≥ 1", kind: "covering" } },
    { line: 22, kind: "add_constraint", payload: { name: "cover_4", expr: "y_C + y_D + y_E ≥ 1", kind: "covering" } },
    { line: 26, kind: "solve", payload: {
        status: "optimal",
        nodes: 1,
        time: 0.003,
        gap: 0.0,
        obj: 9.0,
        primalbound: 9.0,
        dualbound: 9.0,
        vars: { "y_A": 1, "y_B": 0, "y_C": 0, "y_D": 0, "y_E": 1 },
      }, note: "SCIP finds the LP optimum is integral at the root → no branching needed. {A, E} covers items 0,1,2 (via A) and 3,4 (via E) for total cost 9." },
    { line: 28, kind: "print", payload: { text: "  pick set A (cost 6)" } },
    { line: 28, kind: "print", payload: { text: "  pick set E (cost 3)" } },
    { line: 31, kind: "print", payload: { text: "total cost = 9.0" } },
  ],
};

const PROB_MINLP = {
  key: "minlp",
  name: "Rectangle (MINLP)",
  kind: "MINLP",
  blurb:
    "Find integer side lengths of a rectangle with area ≥ 100 minimizing perimeter. The constraint L·W is bilinear — a true MINLP. SCIP handles it with spatial branch-and-bound on the nonlinear part plus normal MILP branching on integers.",
  formula: "min  2(L + W)   s.t.   L · W ≥ 100,   L, W ∈ {1, ..., 20}",
  code: [
    null,
    "from pyscipopt import Model",
    "",
    "m = Model('rectangle')",
    "L = m.addVar(vtype='I', name='L', lb=1, ub=20)",
    "W = m.addVar(vtype='I', name='W', lb=1, ub=20)",
    "",
    "m.setObjective(2*L + 2*W, 'minimize')",
    "",
    "# Bilinear constraint — this is the MINLP part",
    "m.addCons(L * W >= 100)",
    "",
    "m.optimize()",
    "",
    "print('status :', m.getStatus())",
    "print('L      =', m.getVal(L))",
    "print('W      =', m.getVal(W))",
    "print('area   =', m.getVal(L) * m.getVal(W))",
    "print('perim  =', m.getObjVal())",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "pyscipopt", what: "Model" }, note: "We don't need quicksum — only two variables and a single bilinear constraint." },
    { line: 3, kind: "create_model", payload: { name: "rectangle" } },
    { line: 4, kind: "add_vars", payload: { name: "L", count: 1, vtype: "I", desc: "integer length, 1 ≤ L ≤ 20" }, note: "vtype='I' for general integer (not binary). SCIP will branch on fractional values during the solve." },
    { line: 5, kind: "add_vars", payload: { name: "W", count: 1, vtype: "I", desc: "integer width, 1 ≤ W ≤ 20" } },
    { line: 7, kind: "set_objective", payload: { sense: "minimize", expr: "2L + 2W" }, note: "Linear objective. The nonlinearity is in the constraint." },
    { line: 10, kind: "add_constraint", payload: { name: "area", expr: "L · W ≥ 100", kind: "bilinear (nonconvex)" }, note: "The product L·W is nonconvex. SCIP applies spatial branch-and-bound: it splits the (L, W) box into smaller boxes, replacing L·W with linear underestimators (McCormick envelopes) on each." },
    { line: 12, kind: "solve", payload: {
        status: "optimal",
        nodes: 7,
        time: 0.018,
        gap: 0.0,
        obj: 40.0,
        primalbound: 40.0,
        dualbound: 40.0,
        vars: { L: 10, W: 10 },
      }, note: "SCIP explores 7 B&B nodes. The optimum is the most square-ish integer rectangle: L = W = 10, area = 100, perimeter = 40." },
    { line: 14, kind: "print", payload: { text: "status : optimal" } },
    { line: 15, kind: "print", payload: { text: "L      = 10.0" } },
    { line: 16, kind: "print", payload: { text: "W      = 10.0" } },
    { line: 17, kind: "print", payload: { text: "area   = 100.0" } },
    { line: 18, kind: "print", payload: { text: "perim  = 40.0" } },
  ],
};

const PROBLEMS = [PROB_KNAPSACK, PROB_SETCOVER, PROB_MINLP];

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo) {
  const s = {
    imports: [],
    rawData: [],
    model: null,
    vars: [],
    objective: null,
    constraints: [],
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
      case "create_model":
        s.model = ev.payload.name;
        break;
      case "add_vars":
        s.vars.push({ ...ev.payload });
        break;
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "solve":
        s.result = ev.payload;
        for (const v of s.vars) {
          if (ev.payload.vars) {
            const vals = {};
            for (const k in ev.payload.vars) {
              if (k === v.name || k.startsWith(v.name + "_")) {
                const subkey = k === v.name ? v.name : k;
                vals[subkey] = ev.payload.vars[k];
              }
            }
            if (Object.keys(vals).length > 0) v.values = vals;
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
export default function ScipTutorial() {
  const [probKey, setProbKey] = useState(PROB_KNAPSACK.key);
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
        SCIP / PySCIPOpt — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        SCIP is one of the strongest open-source MIP / MINLP solvers. Three
        problems show its API: a textbook knapsack, set cover, and a small
        nonconvex MINLP. Watch how the same Model / addVar / addCons /
        optimize pattern handles all three.
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
          gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)",
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
// State panel
// ============================================================
function StatePanel({ state }) {
  return (
    <div style={statePanelOuter}>
      <Section title="Imports">
        {state.imports.length === 0 ? (
          <Empty />
        ) : (
          state.imports.map((imp, i) => (
            <span key={i} style={chip("#1f4e3d")}>
              from {imp.name} import {imp.what}
            </span>
          ))
        )}
      </Section>

      {state.rawData.length > 0 && (
        <Section title="Python data">
          {state.rawData.map((d, i) => (
            <KVRow key={i} k={d.label} v={d.value} desc={d.desc} mono />
          ))}
        </Section>
      )}

      <Section title="Model">
        {state.model ? (
          <span style={chip("#0b3da0")}>Model('{state.model}')</span>
        ) : (
          <Empty />
        )}
      </Section>

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
                  {v.count > 1 && ` × ${v.count}`}
                </span>
                <span
                  style={{
                    fontSize: 11,
                    fontFamily: "monospace",
                    color: vtypeColor(v.vtype),
                    background: vtypeBg(v.vtype),
                    padding: "1px 6px",
                    borderRadius: 3,
                  }}
                >
                  vtype={v.vtype}
                </span>
              </div>
              {v.desc && (
                <div style={{ fontSize: 12, color: "#555", marginTop: 2 }}>
                  {v.desc}
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
                      <b>{k}</b> = {val}
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
              <div>{c.expr}</div>
              <span style={{ color: "#888", fontSize: 11 }}>
                {c.name && `${c.name} · `}
                {c.kind}
              </span>
            </div>
          ))
        )}
      </Section>

      {state.result && (
        <Section title="SCIP solve result">
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
            <ResultLine k="primal bound" v={(+state.result.primalbound).toFixed(4)} c="#f5a524" />
            <ResultLine k="dual bound" v={(+state.result.dualbound).toFixed(4)} c="#f5a524" />
            <ResultLine k="MIP gap" v={`${(state.result.gap * 100).toFixed(2)}%`} />
            <ResultLine k="B&B nodes" v={state.result.nodes} />
            <ResultLine k="solve time" v={`${state.result.time.toFixed(3)} s`} />
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

function vtypeColor(t) {
  if (t === "B") return "#fff";
  if (t === "I") return "#fff";
  if (t === "C") return "#fff";
  return "#fff";
}
function vtypeBg(t) {
  if (t === "B") return "#0b3da0"; // binary — blue
  if (t === "I") return "#7a3da0"; // integer — purple
  if (t === "C") return "#1f4e3d"; // continuous — green
  return "#666";
}

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
          <b>Variable types.</b>{" "}
          <code style={inlineCode}>vtype='B'</code> (binary 0/1),{" "}
          <code style={inlineCode}>vtype='I'</code> (general integer with
          bounds), <code style={inlineCode}>vtype='C'</code> (continuous —
          default). The vtype is what triggers branching.
        </li>
        <li>
          <b>Primal vs dual bound.</b> Primal bound = best feasible solution
          found so far. Dual bound = best LP relaxation bound. They squeeze
          together as branching proceeds; gap = 0 means proven optimal.
        </li>
        <li>
          <b>Knapsack solves at the root.</b> The LP relaxation of this
          knapsack happens to be integral (rare but possible for small
          instances). For larger knapsacks SCIP may explore hundreds of nodes.
        </li>
        <li>
          <b>MINLP via spatial branch-and-bound.</b> SCIP handles{" "}
          <code style={inlineCode}>L * W</code> by replacing it locally with
          McCormick envelopes (linear under/overestimators), branching on the
          tightest box, and refining as it goes.
        </li>
        <li>
          <b>Showing SCIP's iteration log.</b>{" "}
          <code style={inlineCode}>m.hideOutput()</code>{" "}
          /{" "}
          <code style={inlineCode}>m.showOutput()</code>{" "}
          toggle the streaming SCIP log. Shown by default — set hideOutput
          before solve() if it's distracting.
        </li>
        <li>
          <b>Time limit and gap.</b>{" "}
          <code style={inlineCode}>m.setParam('limits/time', 30)</code>,{" "}
          <code style={inlineCode}>m.setParam('limits/gap', 0.01)</code>. SCIP
          stops at the FIRST limit hit and returns the best solution found.
        </li>
      </ul>
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
        Install SCIP + PySCIPOpt &nbsp;
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
            PySCIPOpt is a Python wrapper around SCIP — you need both.
          </p>
          <Pre>
            {`# EASIEST — conda installs SCIP + Python bindings together:
conda install -c conda-forge pyscipopt

# pip route (you'll need SCIP on your system first):
brew install scip                  # macOS
sudo apt install libscip-dev       # Ubuntu / Debian
pip install pyscipopt

# Verify
python -c "from pyscipopt import Model; print(Model().version())"`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            Academic users get a no-cost license automatically. SCIP is free
            for non-commercial use.
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================
// CodePanel — same template
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
            <span style={{ width: 22, color: active ? "#f5a524" : "#7f7864", fontSize: 11, userSelect: "none" }}>
              {active ? "▶" : ""}
            </span>
            <span style={{ width: 28, color: "#7f7864", textAlign: "right", marginRight: 12, fontSize: 11, userSelect: "none" }}>
              {i}
            </span>
            <span style={{ color: active ? "#fff8e1" : isBlank ? "#7f7864" : "#e8e2d4", whiteSpace: "pre" }}>
              {line || " "}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ============================================================
// Bits & pieces
// ============================================================
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
    marginRight: 5,
    marginBottom: 5,
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
  if (k === "MILP") return "#1f4e3d";
  if (k === "MINLP") return "#7a3da0";
  return "#444";
}

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
