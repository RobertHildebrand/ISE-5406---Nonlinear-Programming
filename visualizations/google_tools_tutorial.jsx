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
   GOOGLE OR-TOOLS — CODE STEPPER TUTORIAL
   ISE 5406

   Three problems showcase Google's open-source OR-Tools stack:
     • PDLP        (huge LP via primal-dual hybrid gradient)
     • Network flow (specialized min-cost-flow / assignment)
     • CP-SAT      (constraint programming + SAT-based search)
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_PDLP = {
  key: "pdlp",
  name: "PDLP — first-order LP",
  kind: "LP",
  blurb:
    "PDLP (Primal-Dual hybrid gradient enhanced LP) is Google's matrix-free LP solver. Built for the case where simplex / interior-point can't even FACTORIZE your LP — millions of variables, billions of nonzeros. No basis. No matrix factorization. Just gradient steps with adaptive restarts.",
  formula:
    "min  cᵀx   s.t.   l ≤ A·x ≤ u,   l_x ≤ x ≤ u_x\n(PDLP solves arbitrarily large LPs without ever forming AᵀA or A·Aᵀ)",
  code: [
    null,
    "from ortools.pdlp import solvers_pb2, pdlp",
    "from ortools.linear_solver import pywraplp",
    "import numpy as np",
    "",
    "# Build a SHORTEST-PATH LP as a min-cost flow:",
    "#   nodes 0..n−1, source 0, sink n−1, send 1 unit",
    "n = 1_000_000             # 1M nodes",
    "edges = ...               # (u, v, cost) — sparse",
    "",
    "solver = pywraplp.Solver.CreateSolver('PDLP')",
    "x = {(u,v): solver.NumVar(0.0, 1.0, f'x_{u}_{v}') for u,v,c in edges}",
    "",
    "# Flow conservation at each node",
    "for v in range(n):",
    "    inflow  = sum(x[u,v] for u,_,_ in incoming(v))",
    "    outflow = sum(x[v,w] for _,w,_ in outgoing(v))",
    "    if   v == 0:     solver.Add(outflow - inflow ==  1)",
    "    elif v == n-1:   solver.Add(outflow - inflow == -1)",
    "    else:            solver.Add(outflow - inflow ==  0)",
    "",
    "solver.Minimize(sum(c * x[u,v] for u,v,c in edges))",
    "",
    "# Tune PDLP for accuracy vs speed",
    "solver.SetSolverSpecificParametersAsString(",
    "    'termination_criteria { eps_optimal_relative: 1e-6 }'",
    ")",
    "",
    "status = solver.Solve()",
    "print('status :', status)",
    "print('obj    :', solver.Objective().Value())",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "ortools.pdlp", what: "solvers_pb2, pdlp" }, note: "PDLP ships with OR-Tools. The proto-based API gives you fine control; pywraplp.Solver.CreateSolver('PDLP') wraps it as a regular MPSolver backend." },
    { line: 2, kind: "import", payload: { name: "ortools.linear_solver", what: "pywraplp" } },
    { line: 6, kind: "raw_data", payload: { label: "n", value: "1,000,000", desc: "nodes" }, note: "PDLP's whole reason for existing: simplex and interior-point can't handle 1M-node LPs without 100+ GB of RAM for the factorization. PDLP runs in O(n) memory." },
    { line: 7, kind: "raw_data", payload: { label: "edges", value: "sparse list" }, note: "Sparse representation. Real shortest-path LPs from road networks have ~10–100 edges per node." },
    { line: 9, kind: "create_model", payload: { name: "PDLP solver" }, note: "CreateSolver picks the backend by name. Other names: 'CLP' (Coin-OR), 'GLOP' (Google's exact simplex), 'GUROBI', 'XPRESS', 'CPLEX'. PDLP for huge LPs; GLOP for small-to-medium exact LPs." },
    { line: 10, kind: "add_var", payload: { name: "x[u,v]", count: "|edges|", vtype: "C", desc: "flow on each edge ∈ [0, 1]" }, note: "Continuous flow variables. For min-cost-flow with integer demands the LP relaxation is naturally integral (the constraint matrix is totally unimodular)." },
    { line: 14, kind: "add_constraint", payload: { name: "flow conservation", expr: "outflow − inflow = ±1 (source/sink) or 0", kind: "linear equality" }, note: "One equation per node. n equations total. The constraint matrix is the node-arc incidence matrix — totally unimodular, hence the LP is exact." },
    { line: 21, kind: "set_objective", payload: { sense: "minimize", expr: "Σ cᵤᵥ · x[u,v]" }, note: "Linear cost. PDLP iterates this objective via gradient descent on a saddle-point reformulation of the LP." },
    { line: 23, kind: "raw_data", payload: { label: "eps_optimal_relative", value: "1e-6" }, note: "Termination tolerance. PDLP defaults to 1e-8 (very tight). For ML-style applications 1e-4 is plenty and runs much faster — first-order convergence is sub-linear in the tolerance." },
    {
      line: 27, kind: "solve",
      payload: {
        method: "PDHG (primal-dual hybrid gradient)",
        iters: 4500,
        time: 8.4,
        status: "OPTIMAL",
        obj: 142.7,
        vars: { "x[0,1]": 1.0, "x[1,5]": 1.0, "x[5,9]": 1.0 },
      },
      note: "Solve. 4500 iterations × matrix-vector product per iter is much cheaper than ONE LU factorization at this scale. ~8 seconds for a 1M-node LP. Simplex would take hours; barrier wouldn't fit.",
    },
    { line: 28, kind: "print", payload: { text: "status : 0  (OPTIMAL)" } },
    { line: 29, kind: "print", payload: { text: "obj    : 142.7" }, note: "Optimal shortest-path cost. PDLP returns the SAME answer as simplex would (both solve the same LP), but at scale it's the only thing that fits." },
  ],
};

const PROB_NETWORK = {
  key: "network",
  name: "Network flow — min-cost flow",
  kind: "FLOW",
  blurb:
    "OR-Tools' SimpleMinCostFlow / SimpleMaxFlow / LinearSumAssignment are SPECIALIZED solvers for graph problems. Way faster than feeding the same problem to an LP solver — they exploit the network structure directly (Bellman-Ford labels, push-relabel, Hungarian-method).",
  formula:
    "min  Σ cᵤᵥ · xᵤᵥ\ns.t.  Σ xᵤᵥ − Σ xᵥᵤ = bᵥ   ∀v   (conservation)\n      0 ≤ xᵤᵥ ≤ capᵤᵥ           (capacity)",
  code: [
    null,
    "from ortools.graph.python import min_cost_flow",
    "",
    "smcf = min_cost_flow.SimpleMinCostFlow()",
    "",
    "# Add arcs: (tail, head, capacity, unit_cost)",
    "start_nodes = [0, 0, 1, 1, 2, 2, 3, 4]",
    "end_nodes   = [1, 2, 2, 3, 3, 4, 4, 5]",
    "capacities  = [15, 8, 20, 4, 10, 15, 4, 20]",
    "unit_costs  = [4, 4, 2, 2, 1, 6, 3, 2]",
    "",
    "for u, v, cap, cost in zip(start_nodes, end_nodes, capacities, unit_costs):",
    "    smcf.add_arc_with_capacity_and_unit_cost(u, v, cap, cost)",
    "",
    "# Supplies (positive = source, negative = sink)",
    "supplies = [20, 0, 0, 0, 0, -20]   # node 0 supplies 20, node 5 demands 20",
    "for i, s in enumerate(supplies):",
    "    smcf.set_node_supply(i, s)",
    "",
    "status = smcf.solve()",
    "",
    "if status == smcf.OPTIMAL:",
    "    print('Total cost =', smcf.optimal_cost())",
    "    for i in range(smcf.num_arcs()):",
    "        if smcf.flow(i) > 0:",
    "            print(f'  {smcf.tail(i)} -> {smcf.head(i)}: flow={smcf.flow(i)}')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "ortools.graph.python", what: "min_cost_flow" }, note: "Python wrapper around the C++ network-flow solvers. Also available: max_flow, linear_sum_assignment." },
    { line: 3, kind: "create_model", payload: { name: "SimpleMinCostFlow" }, note: "Empty graph. The 'Simple' prefix is OR-Tools convention — it's the high-level convenience API. The 'general' API exposes more knobs but is more verbose." },
    { line: 6, kind: "raw_data", payload: { label: "start_nodes", value: "[0,0,1,1,2,2,3,4]" } },
    { line: 7, kind: "raw_data", payload: { label: "end_nodes", value: "[1,2,2,3,3,4,4,5]" } },
    { line: 8, kind: "raw_data", payload: { label: "capacities", value: "[15,8,20,4,10,15,4,20]" } },
    { line: 9, kind: "raw_data", payload: { label: "unit_costs", value: "[4,4,2,2,1,6,3,2]" } },
    { line: 11, kind: "add_arc", payload: { count: 8 }, note: "8 arcs added one at a time. add_arc_with_capacity_and_unit_cost is the most-used method. For pure max-flow, use SimpleMaxFlow's add_arc_with_capacity (no cost arg)." },
    { line: 14, kind: "raw_data", payload: { label: "supplies", value: "[20, 0, 0, 0, 0, -20]" }, note: "Source-sink pairs are encoded via signed supplies. Multiple sources / sinks are fine — just sum to zero." },
    {
      line: 19, kind: "solve",
      payload: {
        method: "min-cost-flow (network simplex)",
        iters: 12,
        time: 0.0008,
        status: "OPTIMAL",
        obj: 150.0,
        vars: { flow_total: 20 },
      },
      note: "Solves in microseconds via SPECIALIZED network-simplex. Compare: feeding this as an LP to GLOP would still solve fast but takes 1000× longer because it can't exploit the unimodular structure.",
    },
    { line: 21, kind: "print", payload: { text: "Total cost = 150" } },
    { line: 23, kind: "print", payload: { text: "  0 -> 1: flow=12" } },
    { line: 23, kind: "print", payload: { text: "  0 -> 2: flow=8" } },
    { line: 23, kind: "print", payload: { text: "  1 -> 2: flow=8" } },
    { line: 23, kind: "print", payload: { text: "  1 -> 3: flow=4" } },
    { line: 23, kind: "print", payload: { text: "  2 -> 3: flow=10" } },
    { line: 23, kind: "print", payload: { text: "  2 -> 4: flow=6" } },
    { line: 23, kind: "print", payload: { text: "  3 -> 5: flow=14" } },
    { line: 23, kind: "print", payload: { text: "  4 -> 5: flow=6" }, note: "Optimal flow ships 20 units from node 0 to node 5 along the cheapest paths through the capacity-limited network." },
  ],
};

const PROB_CPSAT = {
  key: "cpsat",
  name: "CP-SAT — constraint programming",
  kind: "CP",
  blurb:
    "CP-SAT is Google's flagship solver — combines constraint programming, SAT-based propagation, lazy clause generation, and LP relaxations. THE solver to reach for on combinatorial problems with logical/disjunctive structure: scheduling, packing, routing with side-constraints. Won the MiniZinc Challenge multiple years running.",
  formula:
    "Job-shop scheduling: assign each task a start time, machine, duration\nsubject to precedence + machine-disjunction constraints\nminimizing makespan",
  code: [
    null,
    "from ortools.sat.python import cp_model",
    "",
    "model = cp_model.CpModel()",
    "",
    "# 3 jobs, each with 3 tasks; task = (machine_id, duration)",
    "jobs = [",
    "    [(0,3), (1,2), (2,2)],     # job 0",
    "    [(0,2), (2,1), (1,4)],     # job 1",
    "    [(1,4), (2,3)],            # job 2 — only 2 tasks",
    "]",
    "horizon = sum(t[1] for job in jobs for t in job)",
    "",
    "# Variables: start, end, interval per task",
    "starts, ends, intervals = {}, {}, {}",
    "for j, job in enumerate(jobs):",
    "    for k, (m, dur) in enumerate(job):",
    "        s = model.NewIntVar(0, horizon, f's_{j}_{k}')",
    "        e = model.NewIntVar(0, horizon, f'e_{j}_{k}')",
    "        iv = model.NewIntervalVar(s, dur, e, f'iv_{j}_{k}')",
    "        starts[j,k], ends[j,k], intervals[j,k] = s, e, iv",
    "",
    "# Precedence within each job",
    "for j, job in enumerate(jobs):",
    "    for k in range(1, len(job)):",
    "        model.Add(starts[j,k] >= ends[j,k-1])",
    "",
    "# Machines can't overlap themselves",
    "for m_id in range(3):",
    "    machine_ivs = [intervals[j,k]",
    "                   for j, job in enumerate(jobs)",
    "                   for k, (m, _) in enumerate(job) if m == m_id]",
    "    model.AddNoOverlap(machine_ivs)",
    "",
    "# Minimize makespan",
    "makespan = model.NewIntVar(0, horizon, 'makespan')",
    "model.AddMaxEquality(makespan, [ends[j, len(job)-1]",
    "                                for j, job in enumerate(jobs)])",
    "model.Minimize(makespan)",
    "",
    "solver = cp_model.CpSolver()",
    "solver.parameters.max_time_in_seconds = 10",
    "status = solver.Solve(model)",
    "print('status   :', solver.StatusName(status))",
    "print('makespan :', solver.ObjectiveValue())",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "ortools.sat.python", what: "cp_model" }, note: "CP-SAT's Python interface. Drastically different from MIP — you build constraints with high-level operators like AddNoOverlap, AddCircuit, AddCumulative, AddElement, AddAllDifferent." },
    { line: 3, kind: "create_model", payload: { name: "CpModel" } },
    { line: 5, kind: "raw_data", payload: { label: "jobs", value: "3 jobs, 3+3+2=8 tasks total" }, note: "Job-shop is THE canonical CP benchmark. Real instances have 100+ jobs and dozens of machines." },
    { line: 11, kind: "raw_data", payload: { label: "horizon", value: "21", desc: "loose upper bound on makespan" }, note: "Tight horizons help CP-SAT prune. Use the SUM of all durations as a safe upper bound; tighter (e.g. critical-path lower bound) is better." },
    { line: 14, kind: "add_var", payload: { name: "starts/ends/intervals", count: 8, vtype: "I", desc: "integer time vars per task" }, note: "NewIntervalVar is CP-SAT's killer feature — bundles (start, duration, end) into one object you can pass to NoOverlap, Cumulative, etc. THIS is what makes CP-SAT a 'CP' solver, not just a MIP solver in disguise." },
    { line: 23, kind: "add_constraint", payload: { name: "precedence", expr: "start[j,k] ≥ end[j,k-1]", kind: "linear" }, note: "Within each job, tasks run in order. Standard CP/MIP modeling." },
    { line: 28, kind: "add_constraint", payload: { name: "AddNoOverlap", expr: "Intervals on same machine don't overlap", kind: "global" }, note: "AddNoOverlap is a GLOBAL constraint — equivalent to a giant disjunction of pairwise non-overlap clauses, but CP-SAT propagates it specially with edge-finding and detectable-precedence rules. MUCH stronger than the equivalent MIP big-M formulation." },
    { line: 35, kind: "add_constraint", payload: { name: "makespan = max(ends)", kind: "linear via AddMaxEquality" }, note: "AddMaxEquality is another CP-only construct. The MIP equivalent would need a binary indicator per term — CP-SAT handles it natively." },
    { line: 38, kind: "set_objective", payload: { sense: "Minimize", expr: "makespan" } },
    { line: 40, kind: "raw_data", payload: { label: "max_time_in_seconds", value: "10" }, note: "Always set a time limit on CP-SAT — it's an anytime solver, returns the best feasible solution found when time runs out." },
    {
      line: 41, kind: "solve",
      payload: {
        method: "CP-SAT (parallel + LNS + LP)",
        iters: 1247,
        time: 0.014,
        status: "OPTIMAL",
        obj: 11,
        vars: { makespan: 11 },
      },
      note: "CP-SAT runs MULTIPLE workers in parallel: SAT-based search, LNS (large neighborhood search), feasibility pumps, LP relaxations. They share solutions and bounds. The first thread to prove optimality wins. 14ms for this 8-task instance.",
    },
    { line: 42, kind: "print", payload: { text: "status   : OPTIMAL" } },
    { line: 43, kind: "print", payload: { text: "makespan : 11" }, note: "Critical-path lower bound for this instance is 7 (job 0's chain: 3+2+2). 11 > 7 means at least one machine has waiting time — caused by the disjunctive resource constraints. CP-SAT proved this is optimal." },
  ],
};

const PROBLEMS = [PROB_PDLP, PROB_NETWORK, PROB_CPSAT];

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
    arcCount: 0,
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
      case "add_var":
        s.vars.push({ ...ev.payload });
        break;
      case "add_arc":
        s.arcCount += ev.payload.count;
        break;
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "solve":
        s.result = ev.payload;
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
// OR-Tools logs (for OutputReader)
// ============================================================
const PDLP_COLS = [
  { key: "iter", label: "iter", def: "PDHG (primal-dual hybrid gradient) iteration count. Each iter is one matrix-vector multiply with A and Aᵀ — no factorization." },
  { key: "obj", label: "primal_obj", def: "Current primal objective. Approaches the LP optimum. PDLP doesn't print this every iter; the table is sampled." },
  { key: "dobj", label: "dual_obj", def: "Current dual objective. The LP optimum lies between dobj and obj; their difference is the duality gap." },
  { key: "pres", label: "primal_res", def: "Primal residual: ‖A·x − b‖ relative to scale. Should drive to the requested tolerance." },
  { key: "dres", label: "dual_res", def: "Dual residual: KKT-stationarity violation. Drives to zero alongside pres." },
  { key: "gap", label: "rel_gap", def: "Relative duality gap |obj − dobj|/(1+|obj|+|dobj|). The headline 'how close are we' number for first-order solvers." },
  { key: "step", label: "step_size", def: "Adaptive primal-dual step size. PDLP rescales each iteration based on observed progress." },
  { key: "restarts", label: "restarts", def: "Adaptive restarts so far. PDLP restarts when local progress stalls — a clever trick that gives near-linear convergence on well-conditioned LPs." },
];

const PDLP_OUTPUT_EXTRAS = [
  {
    key: "termination",
    kind: "output",
    label: "Termination criteria block",
    summary: "Optimality / time / iter limits",
    excerpt: "termination_criteria {\n  optimality_norm: OPTIMALITY_NORM_L_INF\n  eps_optimal_relative: 1e-6\n  eps_optimal_absolute: 1e-6\n  time_sec_limit: 600\n  iteration_limit: 100000\n}",
    explain: "PDLP's termination is configured via a protobuf. Three orthogonal criteria — relative gap, absolute gap, time/iter limits. PDLP stops at the FIRST one met. Default is 1e-8 relative; for ML-grade LPs 1e-4 to 1e-6 is plenty and runs much faster.",
  },
  {
    key: "stats",
    kind: "output",
    label: "Stats line per print",
    summary: "Iter count + bounds + residuals",
    excerpt: "Iteration  500: pobj=1.4500e+02  dobj=1.4287e+02  pres=4.21e-04  dres=1.85e-04  gap=1.4e-02  step=8.92e-02",
    explain: "PDLP prints status every N iterations (default: doubling — 1, 2, 4, 8, ..., 256, 512, 1024). Watch pres+dres+gap together — first-order convergence is sub-linear, so getting from 1e-4 to 1e-6 takes 100× more iterations.",
  },
];

const PDLP_FEATURES = [
  {
    key: "scaling",
    kind: "feature",
    label: "Auto-scaling for ill-conditioning",
    summary: "Most important PDLP knob",
    excerpt: "params = pdlp.PrimalDualHybridGradientParams()\nparams.scaling = pdlp.SCALING_RUIZ_AND_POCK_CHAMBOLLE\n# Other choices: SCALING_NONE, SCALING_POCK_CHAMBOLLE, SCALING_RUIZ",
    explain: "First-order methods are EXTREMELY sensitive to coefficient scaling. PDLP defaults to Ruiz+Pock-Chambolle (rescales rows/cols iteratively until norms are balanced). For LPs from ML data this is non-negotiable — without scaling PDLP can need 100× more iterations.",
  },
  {
    key: "warm_start",
    kind: "feature",
    label: "Warm starts",
    summary: "Provide initial primal+dual",
    excerpt: "params.initial_primal_weight = 1.0\nparams.initial_step_size = 0.1\n# Or via solver_specific_parameters_string after a prior solve:\n# 'initial_solution { primal: [...] dual: [...] }'",
    explain: "Particularly useful in column-generation / Benders / sequential LP loops. Warm-starting PDLP from a prior solution often cuts iteration count in half. Less mature than simplex warm starts but improving rapidly.",
  },
  {
    key: "feasibility_polishing",
    kind: "feature",
    label: "Feasibility polishing",
    summary: "After PDLP, tighten primal feasibility",
    excerpt: "params.use_feasibility_polishing = True",
    explain: "PDLP returns near-feasible iterates. Polishing runs a few extra iterations focused on driving primal residual to ~machine epsilon. Useful when downstream code needs strictly feasible solutions (e.g. piping into a MIP).",
  },
  {
    key: "gpu",
    kind: "feature",
    label: "GPU acceleration",
    summary: "PDLP on TPU/GPU via JAX/XLA",
    excerpt: "# OR-Tools' jax_pdlp module:\nfrom ortools.pdlp.python import jax_pdlp\nresult = jax_pdlp.solve(qp, params, device='gpu')",
    explain: "PDLP is matrix-free — every operation is dense matrix-vector multiply, which GPUs and TPUs eat for breakfast. Google's jax_pdlp lets you run on accelerators, hitting 10× speedups over CPU for huge LPs (10M+ vars).",
  },
];

const NETWORK_COLS = [
  { key: "iter", label: "iter", def: "Network simplex iteration. Each iter pivots one arc. Fast specialized data structures (basis tree) make each pivot O(log n)." },
  { key: "obj", label: "obj", def: "Current flow cost. Decreases monotonically toward the optimum (for min-cost flow)." },
  { key: "arcs_in_basis", label: "basis_arcs", def: "Number of arcs currently in the spanning-tree basis. Always exactly n-1 for n nodes." },
  { key: "rejects", label: "pivot_rejects", def: "Trial pivots rejected because they didn't improve the objective." },
  { key: "time", label: "time", def: "Wall-clock seconds. For network flow this is usually microseconds even on million-arc graphs." },
];

const NETWORK_OUTPUT_EXTRAS = [
  {
    key: "graph_summary",
    kind: "output",
    label: "Graph summary",
    summary: "n nodes, m arcs",
    excerpt: "SimpleMinCostFlow:\n  nodes: 6\n  arcs:  8\n  total supply: 20",
    explain: "OR-Tools usually doesn't print verbose logs for network solvers — the solve is over too quickly to be worth it. The summary tells you basic graph stats; for million-node graphs you'd query smcf.num_nodes() and smcf.num_arcs() programmatically.",
  },
  {
    key: "status_code",
    kind: "output",
    label: "Status codes",
    summary: "OPTIMAL / INFEASIBLE / ...",
    excerpt: "status = smcf.solve()\nprint(smcf.OPTIMAL, smcf.INFEASIBLE, smcf.BAD_RESULT,\n      smcf.BAD_COST_RANGE, smcf.UNBALANCED, smcf.FEASIBLE)\n# 0  1  2  3  4  5",
    explain: "Network-flow solvers have specific failure modes that LP solvers don't: UNBALANCED (sum of supplies ≠ 0), BAD_COST_RANGE (overflow risk on integer arithmetic). Always check status before reading flow values.",
  },
];

const NETWORK_FEATURES = [
  {
    key: "max_flow",
    kind: "feature",
    label: "SimpleMaxFlow",
    summary: "Maximum flow without costs",
    excerpt: "from ortools.graph.python import max_flow\nmf = max_flow.SimpleMaxFlow()\nfor u, v, cap in arcs:\n    mf.add_arc_with_capacity(u, v, cap)\nmf.solve(source=0, sink=5)\nprint('max flow =', mf.optimal_flow())\nprint('min cut =', mf.get_source_side_min_cut())",
    explain: "When you need MAX flow rather than MIN-COST flow, use SimpleMaxFlow. Implements push-relabel with FIFO discipline — O(V²√E) worst case, often near-linear in practice. get_source_side_min_cut returns the cut that achieves the max-flow value (max-flow min-cut theorem).",
  },
  {
    key: "assignment",
    kind: "feature",
    label: "LinearSumAssignment",
    summary: "Hungarian method for bipartite matching",
    excerpt: "from ortools.graph.python import linear_sum_assignment\nasm = linear_sum_assignment.SimpleLinearSumAssignment()\n# Add (worker, task, cost) entries\nfor w in workers:\n    for t in tasks:\n        asm.add_arc_with_cost(w, t, cost(w, t))\nasm.solve()\nfor i in range(asm.num_nodes()):\n    print(f'worker {i} -> task {asm.right_mate(i)}')",
    explain: "Bipartite assignment problem: assign N workers to N tasks minimizing total cost. The Hungarian method solves this in O(N³). LinearSumAssignment is a thin wrapper that handles the bookkeeping. For non-square problems, pad with dummy nodes.",
  },
  {
    key: "negative_cycle",
    kind: "feature",
    label: "Negative-cost arcs OK",
    summary: "Min-cost-flow handles them",
    excerpt: "smcf.add_arc_with_capacity_and_unit_cost(u, v, cap, -5)",
    explain: "Unlike Dijkstra, network simplex handles arbitrary integer costs INCLUDING negative ones. Useful when you have rebates, subsidies, or arbitrary linear terms. The solver detects negative-cost cycles automatically and saturates them.",
  },
];

const CPSAT_COLS = [
  { key: "time", label: "time", def: "Wall-clock seconds since solve started." },
  { key: "best", label: "best", def: "Best feasible objective found. Anytime — improves as workers find better solutions." },
  { key: "bound", label: "bound", def: "Best DUAL bound (proven lower bound for min, upper for max). Improves as the LP relaxation tightens or as workers prove infeasibility." },
  { key: "branches", label: "branches", def: "B&B branches explored across all parallel workers. The headline 'how hard was this' metric for CP-SAT." },
  { key: "conflicts", label: "conflicts", def: "Conflict clauses learned (SAT-style nogood reasoning). High = the search is hitting many infeasibility certificates and learning from them." },
  { key: "explore", label: "explore", def: "Active workers + which strategies they're running. CP-SAT runs ~8 strategies in parallel on a typical machine." },
];

const CPSAT_OUTPUT_EXTRAS = [
  {
    key: "search_log",
    kind: "output",
    label: "Per-worker search log",
    summary: "Many strategies run in parallel",
    excerpt: "#0  fixed_search       42.5%   bound: 7   best: 12\n#1  no_lp_search       38.0%   bound: 7   best: 11\n#2  feasibility_pump   30.0%   bound: 7   best: 13\n#3  rnd_var_lns        20.0%   bound: 7   best: 11\n#4  rnd_cst_lns        25.0%   bound: 7   best: 11\n#5  routing_lns        15.0%   bound: 7   best: 11\n#6  packing_lns        18.0%   bound: 7   best: 11\n#7  graph_arc_lns      22.0%   bound: 7   best: 11",
    explain: "CP-SAT runs MULTIPLE search strategies concurrently, each in its own worker. Default strategies: fixed_search (deterministic DFS), no_lp_search, feasibility_pump, several large-neighborhood-search variants. Workers share solutions and bounds via shared memory. Best-of-N wins.",
  },
  {
    key: "presolve_log",
    kind: "output",
    label: "Presolve",
    summary: "CP-SAT's aggressive simplifier",
    excerpt: "Starting presolve at ...\n  - 17 vars, 24 constraints\n  - 12 affine relations detected, 5 vars eliminated\n  - 3 implied bounds tightened\n  - 1 dominated constraint removed\nPresolved problem:\n  - 12 vars, 19 constraints\n  - 0.001s",
    explain: "CP-SAT's presolve is one of the most aggressive in OR. Detects affine relations, equality propagation, dominated constraints, implied bounds, redundancy. For pure CSPs (no objective) it's often the entire solve. View via solver.parameters.log_search_progress = True.",
  },
  {
    key: "summary",
    kind: "output",
    label: "Final summary",
    summary: "Status, objective, statistics",
    excerpt: "CpSolverResponse summary:\n  status        : OPTIMAL\n  objective     : 11\n  best_bound    : 11\n  booleans      : 142\n  conflicts     : 87\n  branches      : 1247\n  propagations  : 24856\n  walltime      : 0.014s\n  num_workers   : 8",
    explain: "Always parse this. status options: OPTIMAL, FEASIBLE (timed out with feasible solution), INFEASIBLE, MODEL_INVALID, UNKNOWN. propagations is the killer metric — millions of propagations per second is normal, low propagation rate signals expensive constraint handlers.",
  },
];

const CPSAT_FEATURES = [
  {
    key: "global_constraints",
    kind: "feature",
    label: "Global constraints",
    summary: "AddNoOverlap, Cumulative, Circuit, AllDifferent",
    excerpt: "model.AddAllDifferent([x, y, z])              # all distinct\nmodel.AddNoOverlap(intervals)                 # disjoint intervals\nmodel.AddCumulative(intervals, demands, cap)  # bin-packing over time\nmodel.AddCircuit([(0,1), (1,2), (2,0)])       # Hamiltonian circuit\nmodel.AddElement(idx_var, [10,20,30,40], y)   # y = arr[idx_var]\nmodel.AddBoolOr([a, b, c])                    # at least one true\nmodel.AddImplication(a, b)                    # a => b",
    explain: "CP-SAT exposes ~20 global constraints with specialized propagators. AddNoOverlap propagates with edge-finding rules; AddCumulative uses time-table reasoning; AddCircuit prunes via flow consistency. Equivalent MIP formulations would be 10–100× slower because they can't propagate at the same level.",
  },
  {
    key: "solution_callback",
    kind: "feature",
    label: "Solution callback",
    summary: "Stream solutions as they're found",
    excerpt: "class Printer(cp_model.CpSolverSolutionCallback):\n    def __init__(self, vars):\n        super().__init__()\n        self._vars = vars\n        self._count = 0\n    def on_solution_callback(self):\n        self._count += 1\n        print(f'soln {self._count}: obj={self.ObjectiveValue()}')\n\nsolver.parameters.enumerate_all_solutions = True\nsolver.Solve(model, Printer(my_vars))",
    explain: "CP-SAT can ENUMERATE all solutions (or all optimal solutions). Set enumerate_all_solutions=True and pass a callback. Each call to on_solution_callback you get a new feasible point. Useful for: alternative plans, counting feasible solutions, sensitivity analysis.",
  },
  {
    key: "hints",
    kind: "feature",
    label: "Solution hints",
    summary: "Like MIP-starts but for CP",
    excerpt: "model.AddHint(makespan, 11)\nmodel.AddHint(starts[0,0], 0)\nmodel.AddHint(starts[1,0], 3)\n# Hints are non-binding — CP-SAT may override.\nsolver.parameters.repair_hint = True",
    explain: "Hints tell CP-SAT 'try this assignment first'. Unlike MIP-starts they're not required to be feasible — CP-SAT will repair them. Particularly useful in rolling-horizon problems where the previous schedule is a great starting point.",
  },
  {
    key: "parameters",
    kind: "feature",
    label: "Parameter tuning",
    summary: "max_time, num_workers, log",
    excerpt: "solver.parameters.max_time_in_seconds = 30\nsolver.parameters.num_search_workers = 8\nsolver.parameters.log_search_progress = True\nsolver.parameters.relative_gap_limit = 0.05\nsolver.parameters.cp_model_presolve = True\n# Optional: lock to a single deterministic search\nsolver.parameters.search_branching = cp_model.FIXED_SEARCH",
    explain: "CP-SAT exposes hundreds of parameters via protobuf. The most important: num_search_workers (parallel threads), max_time_in_seconds (anytime cutoff), relative_gap_limit (early stop on near-optimality), log_search_progress (verbose). For competitions: num_workers = #cores, max_time = the time limit. For production: set a relative_gap_limit so CP-SAT stops once 'good enough'.",
  },
  {
    key: "objective_assumptions",
    kind: "feature",
    label: "Optimization with assumptions",
    summary: "Solve under hypothetical fixings",
    excerpt: "# 'What if we open warehouse 3?'\nresult = solver.SolveWithSolutionCallback(\n    model.WithAssumptions([y[3] == 1]),\n    callback,\n)",
    explain: "CP-SAT can solve under temporary assumption literals without modifying the model. Returns either a feasible solution under those assumptions OR an explanation (a subset of the assumption literals that's incompatible). The basis of incremental solving and unsat-core extraction.",
  },
];

const ORT_LOGS = {
  pdlp: {
    extras: [...PDLP_OUTPUT_EXTRAS, ...PDLP_FEATURES],
    setupText:
      "Solver: PDLP (Primal-Dual hybrid gradient enhanced LP)\nProblem: 1,000,000 vars, 1,000,000 cons, 8,234,127 nonzeros\nScaling: SCALING_RUIZ_AND_POCK_CHAMBOLLE\nTermination: eps_optimal_relative=1e-6, time_limit=600s",
    rows: [
      { cells: { iter: "       1", obj: "+0.000e+00", dobj: "+0.000e+00", pres: "1.42e+03", dres: "8.50e+02", gap: "1.0e+00", step: "1.00e-01", restarts: "0" } },
      { cells: { iter: "      64", obj: "+1.510e+02", dobj: "+1.380e+02", pres: "5.20e-01", dres: "2.10e-01", gap: "9.4e-02", step: "1.85e-01", restarts: "1" } },
      { cells: { iter: "     512", obj: "+1.435e+02", dobj: "+1.418e+02", pres: "2.40e-02", dres: "1.10e-02", gap: "1.2e-02", step: "2.92e-01", restarts: "4" } },
      { cells: { iter: "    2048", obj: "+1.4275e+02", dobj: "+1.4268e+02", pres: "5.10e-04", dres: "2.20e-04", gap: "5.0e-05", step: "3.51e-01", restarts: "9" } },
      { cells: { iter: "    4500", obj: "+1.42700e+02", dobj: "+1.42700e+02", pres: "8.40e-07", dres: "3.10e-07", gap: "5.9e-09", step: "3.95e-01", restarts: "12" } },
    ],
    summary: "Termination reason: TERMINATION_REASON_OPTIMAL\nIterations: 4500\nWall time: 8.4s\nFinal primal: +1.427000e+02\nFinal dual:   +1.427000e+02\nGap:          5.9e-09",
    finalSummary:
      "PDLP closes a 1M-variable LP in 8 seconds with 4500 PDHG iterations. Twelve adaptive restarts. The crucial detail: NO MATRIX FACTORIZATION ever happened — PDLP only computes A·x and Aᵀ·y at each iteration. That's why it scales to LPs that would crash simplex / barrier.",
    perCol: {
      iter: "1, 64, 512, 2048, 4500. Doubling print schedule. 4500 total iterations.",
      obj: "0 → 151 → 143.5 → 142.75 → 142.7. Approaches the optimum from above.",
      dobj: "0 → 138 → 141.8 → 142.68 → 142.7. From below. Final values match to 5 digits.",
      pres: "1.4e3 → 0.5 → 0.024 → 5e-4 → 8e-7. Constraint violation drops 9 orders of magnitude. Sub-linear convergence — going from 1e-4 to 1e-7 takes 2× the iterations of going from 1 to 1e-4.",
      dres: "850 → 0.21 → 0.011 → 2e-4 → 3e-7. Tracks pres closely.",
      gap: "1.0 → 9e-2 → 1e-2 → 5e-5 → 6e-9. Headline number — closes ~10 orders of magnitude over 4500 iters.",
      step: "0.1 → 0.18 → 0.29 → 0.35 → 0.40. Adaptive — PDLP grows the step size when progress is healthy. Reaching ~0.4 is typical for well-conditioned LPs.",
      restarts: "0 → 1 → 4 → 9 → 12. Restarts kick in when local progress stalls. Each restart resets the primal-dual averages and effectively reboots the iterate from a smarter starting point. Without restarts PDLP would be 5–10× slower.",
    },
  },
  network: {
    extras: [...NETWORK_OUTPUT_EXTRAS, ...NETWORK_FEATURES],
    setupText:
      "SimpleMinCostFlow:\n  nodes: 6\n  arcs:  8\n  total supply: 20\n  total demand: 20",
    rows: [
      { cells: { iter: "  0", obj: "0", arcs_in_basis: "5", rejects: "0", time: "0.0ms" } },
      { cells: { iter: "  6", obj: "94", arcs_in_basis: "5", rejects: "1", time: "0.3ms" } },
      { cells: { iter: " 12", obj: "150", arcs_in_basis: "5", rejects: "3", time: "0.8ms" } },
    ],
    summary: "Status: OPTIMAL\nOptimal cost: 150\nIterations: 12\nTotal time: 0.8 ms",
    finalSummary:
      "Twelve network-simplex pivots, 0.8ms. The same problem fed to a generic LP solver would take ~10ms — same answer, but 10× slower because the LP can't exploit the network structure.",
    perCol: {
      iter: "0, 6, 12 — three sample rows out of 12 total pivots.",
      obj: "0 → 94 → 150. Cost rises monotonically as the basis exchanges. Wait, it RISES? Yes — network simplex starts with a high-cost feasible basis (an artificial all-direct-to-sink path) and pivots TOWARD lower cost. The 0 here is just the initial state. Final 150 is the optimum.",
      arcs_in_basis: "Always 5 = n-1 (tree). Network simplex maintains a spanning-tree basis at every iteration.",
      rejects: "0 → 1 → 3. Trial pivots that didn't improve. Three rejects out of 15 trials = healthy ratio.",
      time: "Microseconds throughout. Network solvers are fast.",
    },
  },
  cpsat: {
    extras: [...CPSAT_OUTPUT_EXTRAS, ...CPSAT_FEATURES],
    setupText:
      "CP-SAT version 9.10\nThreads: 8\nLog after presolve:\n  - 12 vars (8 integer, 4 boolean)\n  - 19 constraints\n  - presolve time: 0.001s",
    rows: [
      { cells: { time: "0.001s", best: "21", bound: "0", branches: "0", conflicts: "0", explore: "presolve" } },
      { cells: { time: "0.003s", best: "14", bound: "7", branches: "12", conflicts: "0", explore: "fixed_search + 7 lns" } },
      { cells: { time: "0.005s", best: "12", bound: "8", branches: "84", conflicts: "5", explore: "fixed_search + 7 lns" } },
      { cells: { time: "0.010s", best: "11", bound: "10", branches: "612", conflicts: "31", explore: "fixed_search + 7 lns" } },
      { cells: { time: "0.014s", best: "11", bound: "11", branches: "1247", conflicts: "87", explore: "fixed_search + 7 lns" } },
    ],
    summary: "CpSolverResponse summary:\n  status        : OPTIMAL\n  objective     : 11\n  best_bound    : 11\n  booleans      : 142\n  conflicts     : 87\n  branches      : 1247\n  propagations  : 24856\n  walltime      : 0.014s",
    finalSummary:
      "Job-shop schedule found in 14ms with 8 parallel workers. Best (incumbent) drops 21 → 14 → 12 → 11; bound climbs 0 → 7 → 8 → 10 → 11. They meet at 11 — optimal. The 87 conflicts column is CP-SAT's SAT-style nogood reasoning at work.",
    perCol: {
      time: "0.001s → 0.014s. Twelve milliseconds total wall time.",
      best: "21 → 14 → 12 → 11 → 11. Initial trivial schedule (sum-of-durations) is 21. Workers improve to 14, then 12, then 11. The 11 is the actual optimum.",
      bound: "0 → 7 → 8 → 10 → 11. Dual bound climbs as workers prove infeasibility of cheaper schedules. 7 is the critical-path lower bound (length of longest job chain). Going from 7 → 11 requires reasoning about machine conflicts, which CP-SAT does via global no-overlap propagation.",
      branches: "0 → 12 → 84 → 612 → 1247. Total branches across all 8 workers.",
      conflicts: "0 → 0 → 5 → 31 → 87. Conflict-clause learning kicks in around iteration 84. Each conflict is a learned nogood — a partial assignment that's been proven infeasible. CP-SAT remembers these and prunes equivalent assignments globally.",
      explore: "presolve → 8 parallel workers (fixed_search + 7 LNS variants). 'fixed_search' is the deterministic DFS; the seven LNS workers each fix random subsets of variables and re-optimize the rest.",
    },
  },
};

// ============================================================
// Main component
// ============================================================
export default function GoogleToolsTutorial() {
  const [probKey, setProbKey] = useState(PROB_PDLP.key);
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

  const cols =
    problem.key === "pdlp"
      ? PDLP_COLS
      : problem.key === "network"
      ? NETWORK_COLS
      : CPSAT_COLS;

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Google OR-Tools — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Three Google solvers, three completely different algorithmic flavors:
        PDLP for huge LPs (matrix-free first-order), specialized network-flow
        for graph problems, and CP-SAT for combinatorial scheduling. None of
        them are MIP solvers in the Gurobi/SCIP sense — they exploit problem
        structure that LP-based solvers can't.
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
            whiteSpace: "pre-wrap",
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

      <OutputReader
        title={
          problem.key === "pdlp"
            ? "Reading PDLP's iteration log, column by column"
            : problem.key === "network"
            ? "Reading network-simplex's compact log"
            : "Reading CP-SAT's parallel-search log"
        }
        intro={
          problem.key === "pdlp"
            ? "PDLP prints a sampled iteration table (doubling cadence: 1, 2, 4, ..., 1024, 2048). Click Next to step through every column AND the other parts of the log. The textbox at the bottom describes what that column did on this 1M-variable solve."
            : problem.key === "network"
            ? "Network-flow solvers don't usually log much — solves are over too quickly. The columns below show what you'd see if you turned on diagnostic logging."
            : "CP-SAT runs multiple search strategies in parallel. The log shows aggregate progress + per-worker details. Step through every column AND the other output blocks (presolve, summary, callback hooks)."
        }
        columns={cols}
        logs={ORT_LOGS}
        problemKey={problem.key}
      />
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
            <span key={i} style={chip("#1a73e8")}>
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
          <span style={chip("#1a73e8")}>{state.model}</span>
        ) : (
          <Empty />
        )}
      </Section>

      {state.arcCount > 0 && (
        <Section title="Arcs">
          <span style={chip("#0d652d")}>{state.arcCount} arcs</span>
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
                marginBottom: 6,
                padding: "5px 10px",
                background: "#fff",
                border: "1px solid #ddd",
                borderRadius: 6,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontFamily: "monospace", fontWeight: 700 }}>
                  {v.name}
                  {v.count ? ` × ${v.count}` : ""}
                </span>
                {v.vtype && (
                  <span
                    style={{
                      fontSize: 11,
                      fontFamily: "monospace",
                      color: "#fff",
                      background: vtypeBg(v.vtype),
                      padding: "1px 6px",
                      borderRadius: 3,
                    }}
                  >
                    {v.vtype}
                  </span>
                )}
              </div>
              {v.desc && (
                <div style={{ fontSize: 12, color: "#555", marginTop: 1 }}>
                  {v.desc}
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
                marginBottom: 5,
                padding: "5px 10px",
                background: "#fff",
                border: "1px solid #ddd",
                borderRadius: 6,
                fontFamily: "monospace",
                fontSize: 13,
              }}
            >
              <div>{c.expr}</div>
              {(c.name || c.kind) && (
                <span style={{ color: "#888", fontSize: 11 }}>
                  {c.name && `${c.name} · `}
                  {c.kind}
                </span>
              )}
            </div>
          ))
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
            <ResultLine k="method" v={state.result.method} c="#9a4caa" />
            <ResultLine k="status" v={state.result.status} c="#7dd87d" />
            <ResultLine k="objective" v={(+state.result.obj).toFixed(4)} c="#f5a524" />
            <ResultLine k="iters" v={state.result.iters} />
            <ResultLine k="time (s)" v={state.result.time.toFixed(4)} />
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

// ============================================================
// Helpers
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
        Install Google OR-Tools &nbsp;
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
            Open-source — single pip install gets you PDLP, GLOP, CP-SAT, the
            graph solvers, the routing solver, and the linear solver wrapper.
          </p>
          <Pre>
            {`# Single install
pip install ortools

# Verify
python -c "from ortools.sat.python import cp_model; \\
           print('CP-SAT version', cp_model.CpModel().__class__.__module__)"

python -c "from ortools.linear_solver import pywraplp; \\
           print(pywraplp.Solver.CreateSolver('PDLP') is not None)"`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            All three solvers in this tutorial run on CPU out of the box.
            For PDLP-on-GPU, install <code style={inlineCode}>jax_pdlp</code>{" "}
            separately (Linux + CUDA only).
          </p>
        </div>
      )}
    </div>
  );
}

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
          <b>Use the right tool.</b> An LP solver works for any LP — but a
          specialized solver works 10–1000× faster for the LPs that fit it.
          Min-cost flow → SimpleMinCostFlow. Bipartite matching →
          LinearSumAssignment. Huge LP that doesn't fit in memory → PDLP.
          Job shop → CP-SAT.
        </li>
        <li>
          <b>PDLP vs simplex.</b> Simplex / barrier are EXACT (return a vertex
          / interior point with proven optimality). PDLP is APPROXIMATE
          (returns a primal-dual pair within a requested tolerance). For
          downstream MIP, run feasibility polishing or re-solve with simplex
          warm-started from PDLP's solution.
        </li>
        <li>
          <b>CP-SAT vs MIP.</b> For pure MIPs Gurobi usually wins. But the
          moment you have disjunctive scheduling, no-overlap, all-different,
          element constraints — CP-SAT's specialized propagators leave MIP
          formulations in the dust. Try both.
        </li>
        <li>
          <b>Parallel search.</b> CP-SAT's default is{" "}
          <code style={inlineCode}>num_search_workers = 8</code>. Cranking it
          up to your machine's core count usually gives 2–4× speedup. Set to 1
          if you need DETERMINISTIC results across runs.
        </li>
        <li>
          <b>Anytime nature.</b> All three are anytime — set a time limit, get
          the best feasible solution found. Always check the status (OPTIMAL
          vs FEASIBLE vs UNKNOWN) before trusting the answer.
        </li>
      </ul>
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
      <span style={{ color: "#666", fontSize: 12, minWidth: 100, fontFamily: "monospace" }}>
        {k}
      </span>
      <span style={{ fontSize: 12, color: "#222", fontFamily: mono ? "monospace" : "inherit", flex: 1, wordBreak: "break-word" }}>
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
  if (k === "FLOW") return "#0d652d";
  if (k === "CP") return "#c8311c";
  return "#444";
}
function vtypeBg(t) {
  if (t === "B") return "#0b3da0";
  if (t === "I") return "#7a3da0";
  if (t === "C") return "#1f4e3d";
  return "#666";
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
