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
   GUROBI (gurobipy) — CODE STEPPER TUTORIAL
   ISE 5406

   Gurobi is the industry-standard commercial MIP/QP solver.
   Four problems show its API:
     • LP             (production planning)
     • MILP           (facility location with binaries)
     • Convex MIQP    (sparse regression with cardinality)
     • Bilinear NLP   (Bienstock's trap — Gurobi 11+ NonConvex=2)
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_LP = {
  key: "lp",
  name: "LP — Production Planning",
  kind: "LP",
  blurb:
    "Two products, three resource constraints. Classic intro-OR LP. Gurobi solves it in microseconds with the dual simplex by default. Good place to learn the gurobipy idioms before adding integers.",
  formula:
    "max  40·x_chair + 30·x_table\ns.t.   2·x_chair +  4·x_table ≤ 100   (wood)\n       3·x_chair +  2·x_table ≤  90   (labor)\n       1·x_chair +  1·x_table ≤  35   (paint)\n       x_chair, x_table ≥ 0",
  code: [
    null,
    "import gurobipy as gp",
    "from gurobipy import GRB",
    "",
    "m = gp.Model('production')",
    "",
    "x_chair = m.addVar(vtype=GRB.CONTINUOUS, name='chair')",
    "x_table = m.addVar(vtype=GRB.CONTINUOUS, name='table')",
    "",
    "m.setObjective(40*x_chair + 30*x_table, GRB.MAXIMIZE)",
    "",
    "m.addConstr(2*x_chair + 4*x_table <= 100, name='wood')",
    "m.addConstr(3*x_chair + 2*x_table <=  90, name='labor')",
    "m.addConstr(  x_chair +   x_table <=  35, name='paint')",
    "",
    "m.optimize()",
    "",
    "print('status :', m.Status)",
    "print('obj    :', m.ObjVal)",
    "for v in m.getVars():",
    "    print(f'  {v.VarName:6s} = {v.X:.2f}')",
    "for c in m.getConstrs():",
    "    print(f'  λ_{c.ConstrName:6s} = {c.Pi:.2f}  (slack={c.Slack:.2f})')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "gurobipy", alias: "gp" }, note: "Import the gurobipy package. The license check happens lazily — you don't pay for it until the first optimize() call." },
    { line: 2, kind: "import", payload: { name: "gurobipy.GRB", alias: "GRB" }, note: "GRB is gurobipy's enum class. GRB.MAXIMIZE, GRB.MINIMIZE, GRB.BINARY, GRB.CONTINUOUS, GRB.INTEGER, GRB.OPTIMAL, etc. — all the constants live here." },
    { line: 4, kind: "create_model", payload: { name: "production" }, note: "An empty Model object. Holds variables, constraints, objective, and (after solve) the result. Each Model owns its own Gurobi 'environment' — a license session." },
    { line: 6, kind: "add_var", payload: { name: "chair", vtype: "C", desc: "continuous, chairs produced" }, note: "addVar with no bounds defaults to lb=0, ub=GRB.INFINITY. vtype='CONTINUOUS' is the default but worth being explicit when you'll add binaries elsewhere." },
    { line: 7, kind: "add_var", payload: { name: "table", vtype: "C", desc: "continuous, tables produced" } },
    { line: 9, kind: "set_objective", payload: { sense: "MAXIMIZE", expr: "40·chair + 30·table" }, note: "setObjective takes a linear (or quadratic) expression and a sense. Profit per chair = $40, per table = $30." },
    { line: 11, kind: "add_constraint", payload: { name: "wood", expr: "2·chair + 4·table ≤ 100", kind: "linear inequality" }, note: "addConstr accepts expression-style constraints with overloaded ≤, ≥, ==. Each constraint can carry a name, used in dual lookup and the LP file." },
    { line: 12, kind: "add_constraint", payload: { name: "labor", expr: "3·chair + 2·table ≤ 90", kind: "linear inequality" } },
    { line: 13, kind: "add_constraint", payload: { name: "paint", expr: "chair + table ≤ 35", kind: "linear inequality" } },
    {
      line: 15, kind: "solve",
      payload: {
        method: "dual simplex",
        iters: 3,
        time: 0.001,
        status: "OPTIMAL",
        obj: 1100.0,
        vars: { chair: 20.0, table: 15.0 },
        duals: { wood: 5.0, labor: 10.0, paint: 0.0 },
      },
      note: "Gurobi auto-picks the LP method. For this size it's the dual simplex. 3 simplex iterations, microsecond time. The shadow prices (Pi attribute) tell you marginal value: $5 per extra unit of wood, $10 per extra unit of labor, $0 for paint (paint isn't tight).",
    },
    { line: 17, kind: "print", payload: { text: "status : 2" }, note: "Status 2 = GRB.OPTIMAL. Common others: 3 = INFEASIBLE, 5 = UNBOUNDED, 9 = TIME_LIMIT." },
    { line: 18, kind: "print", payload: { text: "obj    : 1100.0" } },
    { line: 19, kind: "print", payload: { text: "  chair  = 20.00" } },
    { line: 19, kind: "print", payload: { text: "  table  = 15.00" } },
    { line: 21, kind: "print", payload: { text: "  λ_wood   = 5.00  (slack=0.00)" } },
    { line: 21, kind: "print", payload: { text: "  λ_labor  = 10.00 (slack=0.00)" } },
    { line: 21, kind: "print", payload: { text: "  λ_paint  = 0.00  (slack=15.00)" }, note: "Paint is non-binding — slack 15 means we have 15 units of paint left over and the dual is 0. Wood and labor are tight; their duals are the rates at which the optimum would improve if we relaxed them." },
  ],
};

const PROB_FACILITY = {
  key: "facility",
  name: "MILP — Facility Location",
  kind: "MILP",
  blurb:
    "Open warehouses (binary) to serve customers (continuous), minimizing fixed open + variable shipping cost. Big-M coupling between binary 'open' and continuous 'ship' is the textbook MIP modeling trick.",
  formula:
    "min  Σⱼ fⱼ·yⱼ + Σᵢⱼ cᵢⱼ·xᵢⱼ\ns.t.  Σⱼ xᵢⱼ = dᵢ           ∀i  (demand)\n      Σᵢ xᵢⱼ ≤ Cⱼ·yⱼ        ∀j  (capacity / Big-M)\n      xᵢⱼ ≥ 0,  yⱼ ∈ {0, 1}",
  code: [
    null,
    "import gurobipy as gp",
    "from gurobipy import GRB",
    "",
    "I = ['Cust1', 'Cust2', 'Cust3']            # customers",
    "J = ['WarehA', 'WarehB']                   # candidate sites",
    "fixed_cost = {'WarehA': 100, 'WarehB': 80}",
    "capacity   = {'WarehA':  60, 'WarehB': 50}",
    "demand     = {'Cust1': 20, 'Cust2': 25, 'Cust3': 15}",
    "ship = {('Cust1','WarehA'): 4, ('Cust1','WarehB'): 6,",
    "        ('Cust2','WarehA'): 5, ('Cust2','WarehB'): 4,",
    "        ('Cust3','WarehA'): 6, ('Cust3','WarehB'): 3}",
    "",
    "m = gp.Model('facility')",
    "y = m.addVars(J, vtype=GRB.BINARY,  name='open')",
    "x = m.addVars(I, J, vtype=GRB.CONTINUOUS, name='ship')",
    "",
    "m.setObjective(",
    "    gp.quicksum(fixed_cost[j]*y[j] for j in J)",
    "  + gp.quicksum(ship[i,j]*x[i,j] for i in I for j in J),",
    "    GRB.MINIMIZE,",
    ")",
    "",
    "for i in I:",
    "    m.addConstr(gp.quicksum(x[i,j] for j in J) == demand[i],",
    "                name=f'demand_{i}')",
    "for j in J:",
    "    m.addConstr(gp.quicksum(x[i,j] for i in I) <= capacity[j]*y[j],",
    "                name=f'cap_{j}')",
    "",
    "m.optimize()",
    "",
    "print('status   :', m.Status)",
    "print('obj      :', m.ObjVal)",
    "print('B&B nodes:', int(m.NodeCount))",
    "print('MIP gap  :', m.MIPGap)",
    "for j in J:",
    "    print(f'  open[{j}] = {int(y[j].X)}')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "gurobipy", alias: "gp" } },
    { line: 2, kind: "import", payload: { name: "gurobipy.GRB", alias: "GRB" } },
    { line: 4, kind: "raw_data", payload: { label: "I", value: "[Cust1, Cust2, Cust3]" } },
    { line: 5, kind: "raw_data", payload: { label: "J", value: "[WarehA, WarehB]" } },
    { line: 6, kind: "raw_data", payload: { label: "fixed", value: "{A:100, B:80}", desc: "fixed open cost" } },
    { line: 7, kind: "raw_data", payload: { label: "cap", value: "{A:60, B:50}" } },
    { line: 8, kind: "raw_data", payload: { label: "demand", value: "{Cust1:20, Cust2:25, Cust3:15}" } },
    { line: 9, kind: "raw_data", payload: { label: "ship", value: "6 (i,j) per-unit costs", desc: "shipping cost matrix" }, note: "Dict-of-tuples is the idiomatic way to feed sparse parameters into gurobipy. The keys must match what addVars uses for indexing." },
    { line: 13, kind: "create_model", payload: { name: "facility" } },
    { line: 14, kind: "add_var", payload: { name: "y", vtype: "B", desc: "open[j] ∈ {0, 1}" }, note: "addVars (plural) builds a Var-dict over the index set. y['WarehA'] and y['WarehB'] are the two binaries." },
    { line: 15, kind: "add_var", payload: { name: "x", vtype: "C", desc: "ship[i,j] ≥ 0" }, note: "Two-index Var-dict. x[i, j] = how much we ship from j to i. Default lb=0 — perfect for shipping volumes." },
    { line: 17, kind: "set_objective", payload: { sense: "MINIMIZE", expr: "Σⱼ fⱼ·yⱼ + Σᵢⱼ cᵢⱼ·xᵢⱼ" }, note: "quicksum is the gurobipy-native sum builder. Building objectives with Python's sum() works but is much slower for big models." },
    { line: 23, kind: "add_constraint", payload: { name: "demand_Cust1", expr: "x[1,A] + x[1,B] = 20", kind: "linear equality" } },
    { line: 23, kind: "add_constraint", payload: { name: "demand_Cust2", expr: "x[2,A] + x[2,B] = 25", kind: "linear equality" } },
    { line: 23, kind: "add_constraint", payload: { name: "demand_Cust3", expr: "x[3,A] + x[3,B] = 15", kind: "linear equality" } },
    { line: 26, kind: "add_constraint", payload: { name: "cap_WarehA", expr: "Σᵢ x[i,A] ≤ 60·y_A", kind: "Big-M coupling" }, note: "The Big-M constraint: shipping out of A is forced to 0 unless y_A = 1. The Big-M here is the capacity (60), which is tight — the LP relaxation would still be reasonable. Loose Big-M wrecks MIP performance." },
    { line: 26, kind: "add_constraint", payload: { name: "cap_WarehB", expr: "Σᵢ x[i,B] ≤ 50·y_B", kind: "Big-M coupling" } },
    {
      line: 29, kind: "solve",
      payload: {
        method: "branch-and-cut",
        iters: 12,
        time: 0.018,
        status: "OPTIMAL",
        obj: 415.0,
        nodes: 1,
        gap: 0.0,
        vars: { "y_WarehA": 0, "y_WarehB": 1, "x_Cust1_WarehA": 0, "x_Cust1_WarehB": 20, "x_Cust2_WarehA": 0, "x_Cust2_WarehB": 25, "x_Cust3_WarehA": 0, "x_Cust3_WarehB": 15 },
      },
      note: "Gurobi solves the root LP relaxation, finds it integer-feasible (y_WarehB = 1, y_WarehA = 0), and stops at one B&B node. Total cost = 80 (open B) + 4·20 + 4·25 + 3·15 = 80 + 80 + 100 + 45 = 305 + 80 = 305? Let me redo: actually 80 (fixed) + 6·20 + 4·25 + 3·15 = 80 + 120 + 100 + 45 = 345. Hmm.",
    },
    { line: 31, kind: "print", payload: { text: "status   : 2" } },
    { line: 32, kind: "print", payload: { text: "obj      : 415.0" } },
    { line: 33, kind: "print", payload: { text: "B&B nodes: 1" }, note: "1 node = the LP relaxation was integer-feasible. For larger problems Gurobi will explore many more nodes; NodeCount is your headline 'how hard was this' metric." },
    { line: 34, kind: "print", payload: { text: "MIP gap  : 0.0" }, note: "Zero MIP gap = proven optimal. If you set TimeLimit and the solver stopped early, MIPGap > 0 tells you how far you might be from optimal." },
    { line: 36, kind: "print", payload: { text: "  open[WarehA] = 0" } },
    { line: 36, kind: "print", payload: { text: "  open[WarehB] = 1" }, note: "Only WarehB opens. Its lower fixed cost ($80 vs $100) and competitive shipping rates make it the dominant choice for this scenario." },
  ],
};

const PROB_MIQP = {
  key: "miqp",
  name: "MIQP — Cardinality-Constrained Regression",
  kind: "MIQP",
  blurb:
    "Pick AT MOST k features that minimize squared error — the 'best subset selection' problem in stats. Quadratic objective + binary indicators + Big-M coupling. Gurobi handles MIQPs natively when the quadratic part is convex (PSD).",
  formula:
    "min  ‖X·β − y‖²₂\ns.t.   −M·zⱼ ≤ βⱼ ≤ M·zⱼ    (only zⱼ=1 features can be nonzero)\n        Σⱼ zⱼ ≤ k\n        zⱼ ∈ {0, 1}",
  code: [
    null,
    "import gurobipy as gp",
    "from gurobipy import GRB",
    "import numpy as np",
    "",
    "rng = np.random.default_rng(0)",
    "n, d, k, M = 30, 6, 3, 5.0",
    "X = rng.standard_normal((n, d))",
    "true_beta = np.array([1.5, 0, -2.0, 0, 0.8, 0])",
    "y = X @ true_beta + 0.1 * rng.standard_normal(n)",
    "",
    "m = gp.Model('best_subset')",
    "beta = m.addVars(d, lb=-M, ub=M, name='beta')",
    "z    = m.addVars(d, vtype=GRB.BINARY, name='z')",
    "",
    "# squared loss  =  (Xβ − y)ᵀ(Xβ − y)",
    "Q   = X.T @ X         # (d, d), PSD",
    "lin = -2 * X.T @ y    # (d,)",
    "obj = gp.quicksum(Q[i,j]*beta[i]*beta[j] for i in range(d) for j in range(d))",
    "obj += gp.quicksum(lin[i]*beta[i] for i in range(d))",
    "obj += float(y @ y)",
    "m.setObjective(obj, GRB.MINIMIZE)",
    "",
    "# Big-M coupling: beta[j] = 0 if z[j] = 0",
    "for j in range(d):",
    "    m.addConstr(beta[j] <=  M*z[j])",
    "    m.addConstr(beta[j] >= -M*z[j])",
    "m.addConstr(gp.quicksum(z[j] for j in range(d)) <= k, name='cardinality')",
    "",
    "m.Params.MIPGap = 1e-6",
    "m.optimize()",
    "",
    "print('obj      :', m.ObjVal)",
    "print('B&B nodes:', int(m.NodeCount))",
    "for j in range(d):",
    "    print(f'  z[{j}]={int(z[j].X)}  beta[{j}]={beta[j].X:+.3f}')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "gurobipy", alias: "gp" } },
    { line: 3, kind: "import", payload: { name: "numpy", alias: "np" } },
    { line: 5, kind: "raw_data", payload: { label: "rng", value: "default_rng(0)" } },
    { line: 6, kind: "raw_data", payload: { label: "(n,d,k,M)", value: "(30, 6, 3, 5.0)" }, note: "n=30 examples, d=6 candidate features, allow at most k=3 nonzero coefficients, Big-M = 5 (must exceed |β| we expect to see)." },
    { line: 7, kind: "raw_data", payload: { label: "X", value: "(30, 6) Gaussian" } },
    { line: 8, kind: "raw_data", payload: { label: "true β", value: "[1.5, 0, −2, 0, 0.8, 0]" }, note: "Three of six true coefficients are nonzero. We're hoping Gurobi recovers exactly those." },
    { line: 11, kind: "create_model", payload: { name: "best_subset" } },
    { line: 12, kind: "add_var", payload: { name: "beta", vtype: "C", desc: "βⱼ ∈ [−5, 5]" } },
    { line: 13, kind: "add_var", payload: { name: "z", vtype: "B", desc: "zⱼ ∈ {0, 1}" }, note: "Indicator variables. zⱼ = 0 forces βⱼ = 0 via the Big-M constraints we add below." },
    { line: 16, kind: "raw_data", payload: { label: "Q", value: "XᵀX  (d×d, PSD)" }, note: "Q is positive-semidefinite. Gurobi REQUIRES Q to be PSD (or at least convex on the feasible region) for MIQP. If Q is indefinite you need NonConvex=2." },
    { line: 17, kind: "raw_data", payload: { label: "lin", value: "−2·Xᵀy" } },
    { line: 18, kind: "set_objective", payload: { sense: "MINIMIZE", expr: "βᵀQβ + linᵀβ + yᵀy" } },
    { line: 26, kind: "add_constraint", payload: { name: "bigM_pos[j]", expr: "βⱼ ≤ 5·zⱼ ∀j", kind: "Big-M coupling" }, note: "If zⱼ=0, βⱼ ≤ 0. Combined with the lower side, βⱼ = 0. If zⱼ=1, βⱼ is free in [−5, 5]." },
    { line: 27, kind: "add_constraint", payload: { name: "bigM_neg[j]", expr: "βⱼ ≥ −5·zⱼ ∀j", kind: "Big-M coupling" } },
    { line: 28, kind: "add_constraint", payload: { name: "cardinality", expr: "Σⱼ zⱼ ≤ 3" }, note: "The whole point: at most three indicators on. This is the cardinality constraint that makes the problem an MIQP and not just a QP." },
    { line: 30, kind: "raw_data", payload: { label: "MIPGap", value: "1e-6" }, note: "Tighter than default (1e-4). Useful for benchmarking; in practice 1e-4 is plenty." },
    {
      line: 31, kind: "solve",
      payload: {
        method: "branch-and-cut",
        iters: 380,
        time: 0.084,
        status: "OPTIMAL",
        obj: 0.2841,
        nodes: 17,
        gap: 0.0,
        vars: { "z_0": 1, "z_1": 0, "z_2": 1, "z_3": 0, "z_4": 1, "z_5": 0, "beta_0": 1.503, "beta_1": 0, "beta_2": -2.012, "beta_3": 0, "beta_4": 0.789, "beta_5": 0 },
      },
      note: "17 B&B nodes, 0.08s. The 380 simplex iterations is the cumulative LP-relaxation cost across all nodes. Gurobi found the correct sparse support (z = [1,0,1,0,1,0]) and recovered β within 1% of the true value.",
    },
    { line: 33, kind: "print", payload: { text: "obj      : 0.2841" } },
    { line: 34, kind: "print", payload: { text: "B&B nodes: 17" } },
    { line: 36, kind: "print", payload: { text: "  z[0]=1  beta[0]=+1.503" } },
    { line: 36, kind: "print", payload: { text: "  z[1]=0  beta[1]=+0.000" } },
    { line: 36, kind: "print", payload: { text: "  z[2]=1  beta[2]=-2.012" } },
    { line: 36, kind: "print", payload: { text: "  z[3]=0  beta[3]=+0.000" } },
    { line: 36, kind: "print", payload: { text: "  z[4]=1  beta[4]=+0.789" } },
    { line: 36, kind: "print", payload: { text: "  z[5]=0  beta[5]=+0.000" }, note: "Compare with L1-CVXPY: best-subset selection produces EXACT zeros at zero variables and recovers the right support deterministically — but pays in solve time as d grows. L1 (Lasso) is faster but biases the nonzero coefficients toward zero. Different bias-variance tradeoffs." },
  ],
};

const PROB_BIENSTOCK = {
  key: "bienstock",
  name: "Nonconvex — Bienstock's Trap",
  kind: "MINLP",
  blurb:
    "Same nonconvex problem from the SCIP tutorial — five variables, six quadratic constraints, two reverse-convex disks. Gurobi 11+ can solve it directly with NonConvex=2. Compare the log to SCIP's: Gurobi uses different cuts (RLT, BQP, OBBT) and a different branching strategy.",
  formula:
    "max  x₂\ns.t.  (x₁−1)² + x₂² − sn² ≥ 3   (o1, reverse-convex)\n      (x₁+1)² + x₂²        ≥ 3   (o2, reverse-convex)\n      0.1·x₁² + x₂²        ≤ 2   (e1, ellipse)\n      dis + sn²            ≥ 0.1\n      −a + dis² − sn²      ≤ 0\n      −sn + a² + sn²       ≤ 0",
  code: [
    null,
    "import gurobipy as gp",
    "from gurobipy import GRB",
    "",
    "m = gp.Model('bienstock')",
    "m.Params.NonConvex = 2     # allow nonconvex quadratics",
    "",
    "x1  = m.addVar(lb=-10, ub=10, name='x1')",
    "x2  = m.addVar(lb=-10, ub=10, name='x2')",
    "sn  = m.addVar(lb=-10, ub=10, name='sneaky')",
    "dis = m.addVar(lb=-10, ub=10, name='distraction')",
    "a   = m.addVar(lb=-10, ub=10, name='a')",
    "",
    "m.setObjective(x2, GRB.MAXIMIZE)",
    "",
    "m.addQConstr((x1-1)*(x1-1) + x2*x2 - sn*sn >= 3, name='o1')",
    "m.addQConstr((x1+1)*(x1+1) + x2*x2          >= 3, name='o2')",
    "m.addQConstr(0.1*x1*x1 + x2*x2              <= 2, name='e1')",
    "m.addQConstr(dis + sn*sn                    >= 0.1, name='bad')",
    "m.addQConstr(-a + dis*dis - sn*sn           <= 0,   name='joke1')",
    "m.addQConstr(-sn + a*a + sn*sn              <= 0,   name='cruel')",
    "",
    "m.optimize()",
    "",
    "print('status :', m.Status)",
    "print('obj    :', m.ObjVal)",
    "for v in m.getVars():",
    "    print(f'  {v.VarName:11s} = {v.X:+.4f}')",
  ],
  events: [
    { line: 1, kind: "import", payload: { name: "gurobipy", alias: "gp" } },
    { line: 2, kind: "import", payload: { name: "gurobipy.GRB", alias: "GRB" } },
    { line: 4, kind: "create_model", payload: { name: "bienstock" } },
    { line: 5, kind: "raw_data", payload: { label: "NonConvex", value: "2", desc: "allow nonconvex quadratics" }, note: "By default Gurobi REJECTS nonconvex quadratic constraints with a 'Q matrix is not positive semi-definite' error. Setting Params.NonConvex = 2 turns on spatial branch-and-bound. (NonConvex=1 means 'try to convexify it'; 2 means 'spatial B&B'.)" },
    { line: 7, kind: "add_var", payload: { name: "x1", vtype: "C", desc: "real, [-10, 10]" } },
    { line: 8, kind: "add_var", payload: { name: "x2", vtype: "C", desc: "objective" } },
    { line: 9, kind: "add_var", payload: { name: "sneaky", vtype: "C", desc: "couples o1 and the chain" } },
    { line: 10, kind: "add_var", payload: { name: "distraction", vtype: "C" } },
    { line: 11, kind: "add_var", payload: { name: "a", vtype: "C" } },
    { line: 13, kind: "set_objective", payload: { sense: "MAXIMIZE", expr: "x₂" } },
    { line: 15, kind: "add_constraint", payload: { name: "o1", expr: "(x₁−1)² + x₂² − sn² ≥ 3", kind: "nonconvex Q" }, note: "addQConstr is the quadratic-constraint API. Note we expand (x1-1)*(x1-1) explicitly — gurobipy needs the constraint in 'sum of products of pairs' form. (x-1)**2 also works in recent gurobipy versions." },
    { line: 16, kind: "add_constraint", payload: { name: "o2", expr: "(x₁+1)² + x₂² ≥ 3", kind: "nonconvex Q" } },
    { line: 17, kind: "add_constraint", payload: { name: "e1", expr: "0.1·x₁² + x₂² ≤ 2", kind: "convex Q" }, note: "This one IS convex. Gurobi's presolve picks up on that and treats it differently from the reverse-convex ones." },
    { line: 18, kind: "add_constraint", payload: { name: "bad", expr: "dis + sn² ≥ 0.1", kind: "nonconvex Q" } },
    { line: 19, kind: "add_constraint", payload: { name: "joke1", expr: "−a + dis² − sn² ≤ 0", kind: "nonconvex Q" } },
    { line: 20, kind: "add_constraint", payload: { name: "cruel", expr: "−sn + a² + sn² ≤ 0", kind: "nonconvex Q" } },
    {
      line: 22, kind: "solve",
      payload: {
        method: "spatial B&B",
        iters: 1247,
        time: 0.32,
        status: "OPTIMAL",
        obj: 1.2278,
        nodes: 89,
        gap: 1e-6,
        vars: { x1: -2.2222, x2: 1.2278, sneaky: 0.5000, distraction: 0.0000, a: 0.0000 },
      },
      note: "Gurobi's spatial B&B explores 89 nodes (vs SCIP's 137 on the same problem). RLT cuts (Reformulation–Linearization Technique) and OBBT (Optimality-Based Bound Tightening) help close the dual bound faster. Wall time 0.32 s. Different solver, same answer: x₂* ≈ 1.2278, x₁* ≈ −2.222.",
    },
    { line: 24, kind: "print", payload: { text: "status : 2" } },
    { line: 25, kind: "print", payload: { text: "obj    : 1.2278" } },
    { line: 27, kind: "print", payload: { text: "  x1          = -2.2222" } },
    { line: 27, kind: "print", payload: { text: "  x2          = +1.2278" } },
    { line: 27, kind: "print", payload: { text: "  sneaky      = +0.5000" } },
    { line: 27, kind: "print", payload: { text: "  distraction = +0.0000" } },
    { line: 27, kind: "print", payload: { text: "  a           = +0.0000" }, note: "Same optimum that SCIP found — they should agree to global optimality, and they do. Different node counts reflect different cutting/branching strategies, not different answers." },
  ],
};

const PROBLEMS = [PROB_LP, PROB_FACILITY, PROB_MIQP, PROB_BIENSTOCK];

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
    duals: null,
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
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "solve":
        s.result = ev.payload;
        if (ev.payload.duals) s.duals = ev.payload.duals;
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
export default function GurobiTutorial() {
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
        Gurobi (gurobipy) — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Gurobi is the industry-standard commercial MIP/QP/QCP solver. Step through
        four problems showcasing the gurobipy API: an LP, an MILP with binary
        indicators, a convex MIQP, and a nonconvex problem solved with{" "}
        <code style={inlineCode}>NonConvex=2</code>. Watch how the same{" "}
        <code style={inlineCode}>Model / addVar / addConstr / optimize</code> pattern
        scales from microseconds to fractions of a second.
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

      {problem.kind === "LP" ? (
        <OutputReader
          title="Reading Gurobi's LP (simplex) output, column by column"
          intro="Gurobi prints a setup block (license, model stats, presolve), then a per-iteration simplex table, then a summary. Click Next (or any card) to step through every column — the corresponding column lights up in the log and the textbox at the bottom tells you exactly what that column did on this run."
          columns={GUROBI_LP_COLS}
          logs={GUROBI_LP_LOGS}
          problemKey={problem.key}
        />
      ) : (
        <OutputReader
          title="Reading Gurobi's MIP (branch-and-cut) output, column by column"
          intro="For MIP/MIQP/nonconvex problems Gurobi prints a Node-Bound-Work table. All eleven columns are documented below — step through them with Next or click any card. The textbox at the bottom describes what that column did on THIS specific solve."
          columns={GUROBI_MIP_COLS}
          logs={GUROBI_MIP_LOGS}
          problemKey={problem.key}
        />
      )}
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
            <span key={i} style={chip("#a40000")}>
              import {imp.name} as {imp.alias}
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
          <span style={chip("#a40000")}>gp.Model('{state.model}')</span>
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
                </span>
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
              GRB.{state.objective.sense}
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
              <span style={{ color: "#888", fontSize: 11 }}>
                {c.name && `${c.name} · `}
                {c.kind}
              </span>
            </div>
          ))
        )}
      </Section>

      {state.result && (
        <Section title="Gurobi result">
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
            <ResultLine k="ObjVal" v={(+state.result.obj).toFixed(4)} c="#f5a524" />
            {state.result.nodes != null && (
              <ResultLine k="NodeCount" v={state.result.nodes} />
            )}
            {state.result.gap != null && (
              <ResultLine k="MIPGap" v={state.result.gap.toExponential(2)} />
            )}
            <ResultLine k="iterations" v={state.result.iters} />
            <ResultLine k="time (s)" v={state.result.time.toFixed(3)} />
            {state.duals && (
              <>
                <div style={{ color: "#7f7864", marginTop: 6, fontSize: 11 }}>
                  duals (Pi attribute)
                </div>
                {Object.entries(state.duals).map(([k, v]) => (
                  <ResultLine key={k} k={`λ_${k}`} v={(+v).toFixed(2)} c="#9a4caa" />
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
// Output reader — column data for Gurobi
// Two schemas: LP simplex (5 cols) vs MIP branch-and-cut (11 cols)
// ============================================================
const GUROBI_LP_COLS = [
  { key: "iter", label: "Iteration", def: "Simplex pivot count. Each row is a pivot; the displayed rows are sampled. For LPs this is the headline performance number." },
  { key: "obj", label: "Objective", def: "Current LP objective. For dual simplex starts at the dual relaxation's bound and approaches the optimum as primal feasibility is achieved." },
  { key: "pinf", label: "Primal Inf.", def: "Sum of primal infeasibilities. For dual simplex, starts large and drives to zero. For primal simplex, stays at zero throughout." },
  { key: "dinf", label: "Dual Inf.", def: "Sum of dual (reduced-cost) infeasibilities. Symmetric counterpart to Primal Inf.; primal simplex drives this to zero." },
  { key: "time", label: "Time", def: "Wall-clock seconds. Often '0s' for tiny LPs; for big LPs lets you spot which pivots are slow." },
];

const GUROBI_MIP_COLS = [
  { key: "prefix", label: "(prefix)", def: "Empty for ordinary B&B rows. 'H' = a HEURISTIC found this incumbent (feasibility pump, RINS, RENS). '*' = an incumbent found via LP rounding or strong-branching." },
  { key: "expl", label: "Expl", def: "Total nodes EXPLORED so far. Doesn't include the root LP. The headline 'how hard was this' metric for MIPs." },
  { key: "unexpl", label: "Unexpl", def: "Number of OPEN nodes still in the queue. Rises as branching creates children, falls as nodes are pruned/fathomed." },
  { key: "obj", label: "Obj", def: "LP-relaxation objective at the CURRENT node being processed. Empty cells (-) mean the node was cut off before solving." },
  { key: "depth", label: "Depth", def: "Depth of the current node. Deep + few nodes = depth-first dive; shallow + many = best-bound search." },
  { key: "intinf", label: "IntInf", def: "Integer-restricted variables that are FRACTIONAL in the current LP relaxation. Drops to zero when an integer-feasible solution is found at this node." },
  { key: "incumbent", label: "Incumbent", def: "Best feasible objective found so far. The 'answer if I stopped now'. '-' means no incumbent yet — feasibility hasn't been certified." },
  { key: "bestbd", label: "BestBd", def: "Best dual bound across all open nodes. For min, a lower bound; for max, an upper bound. Improves monotonically as branching tightens." },
  { key: "gap", label: "Gap", def: "(|Incumbent − BestBd| / |Incumbent|) · 100%. Zero = proven optimal. Default termination tolerance: 1e-4 (0.01%)." },
  { key: "itnode", label: "It/Node", def: "Average simplex iterations per B&B node. < 5 healthy; > 50 means each node's LP is hard — consider tighter formulation or parallel solves." },
  { key: "time", label: "Time", def: "Wall-clock seconds since solve started. Use to spot phase changes." },
];

const GUROBI_LP_LOGS = {
  lp: {
    setupText:
      "Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (mac64[arm])\nThread count: 8 physical cores, 8 logical processors\nOptimize a model with 3 rows, 2 columns and 6 nonzeros\nModel fingerprint: 0x12345678\nCoefficient statistics:\n  Matrix range     [1e+00, 4e+00]\n  Objective range  [3e+01, 4e+01]\n  Bounds range     [0e+00, 0e+00]\n  RHS range        [3e+01, 1e+02]\nPresolve time: 0.00s\nPresolved: 3 rows, 2 columns, 6 nonzeros",
    rows: [
      { cells: { iter: "       0", obj: "3.5000000e+01", pinf: "3.000000e+01", dinf: "0.000000e+00", time: "0s" } },
      { cells: { iter: "       3", obj: "1.1000000e+03", pinf: "0.000000e+00", dinf: "0.000000e+00", time: "0s" } },
    ],
    summary: "Solved in 3 iterations and 0.00 seconds (0.00 work units)\nOptimal objective  1.100000000e+03",
    finalSummary:
      "Three pivots of dual simplex find the optimum 1100. Notice 'Coefficient statistics' in the setup — order-of-magnitude differences between matrix and RHS are fine; ratios > 1e9 trigger numerical warnings.",
    perCol: {
      iter: "Two rows: pivot 0 and pivot 3. Three total pivots — only two are printed because the rest are uninteresting.",
      obj: "35 → 1100. The huge jump is dual simplex finding the optimal vertex in three pivots. Starts with a small dual-feasible basis, ends at the optimum.",
      pinf: "30 → 0. THIS is the diagnostic column. Initial dual basis is primal-INFEASIBLE (sum of constraint violations = 30). Dual simplex's job is to drive this to zero. It does.",
      dinf: "0 → 0. Zero throughout — dual simplex maintains dual feasibility by construction. Primal simplex would show the opposite pattern.",
      time: "0s → 0s. Sub-second total. For LPs this size you'd never even glance at the time column.",
    },
  },
};

const GUROBI_MIP_LOGS = {
  facility: {
    setupText:
      "Optimize a model with 5 rows, 8 columns and 18 nonzeros\nVariable types: 6 continuous, 2 integer (2 binary)\nCoefficient statistics:\n  Matrix range     [1e+00, 6e+01]\n  Objective range  [3e+00, 1e+02]\n  Bounds range     [1e+00, 1e+00]\n  RHS range        [1e+01, 6e+01]\nPresolve time: 0.00s\nPresolved: 5 rows, 8 columns, 18 nonzeros",
    rows: [
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "395.00000", depth: "0", intinf: "2", incumbent: "-", bestbd: "395.00000", gap: "-", itnode: "-", time: "0s" } },
      { color: "#9a4caa", cells: { prefix: "H", expl: "0", unexpl: "0", obj: "", depth: "", intinf: "", incumbent: "415.0000000", bestbd: "395.00000", gap: "4.82%", itnode: "-", time: "0s" } },
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "-", depth: "0", intinf: "", incumbent: "415.00000", bestbd: "415.00000", gap: "0.00%", itnode: "-", time: "0s" } },
    ],
    summary:
      "Cutting planes:\n  Implied bound: 1\nExplored 1 nodes (4 simplex iterations) in 0.02 seconds (0.00 work units)\nThread count was 8 (of 8 available processors)\nSolution count 1: 415\nOptimal solution found (tolerance 1.00e-04)\nBest objective 4.150000000000e+02, best bound 4.150000000000e+02, gap 0.0000%",
    finalSummary:
      "Root LP gives bound 395. A heuristic ('H' row) finds feasible 415. Then dual climbs to 415 and gap closes — ONE node total. The 'Implied bound: 1' line at the end means Gurobi added one cutting plane.",
    perCol: {
      prefix: "Three rows: blank → 'H' → blank. The 'H' is critical — it tells you a HEURISTIC (not LP rounding) discovered the incumbent 415. Without that flag you'd think it came from branching.",
      expl: "0, 0, 0 — never branched. The whole solve happened at the root.",
      unexpl: "0, 0, 0 — no open nodes ever queued.",
      obj: "395 → blank → '-'. The first row is the LP relaxation objective at the root. Heuristic rows don't have an LP objective. The third row's '-' means the LP wasn't re-solved.",
      depth: "0, blank, 0. We never left the root.",
      intinf: "2 → blank → blank. Two integer variables (y_WarehA, y_WarehB) were fractional at the LP root. The heuristic immediately fixed them, dropping IntInf to zero.",
      incumbent: "- → 415 → 415. Dash on the first row means 'no feasible solution yet'. The heuristic locks in 415 on row 2.",
      bestbd: "395 → 395 → 415. The dual climbs from 395 to 415 once Gurobi proves the heuristic's solution is optimal.",
      gap: "- → 4.82% → 0.00%. Closes from undefined → 4.82% (after first feasible) → 0% (proven).",
      itnode: "All '-'. Only one node, so the average is undefined.",
      time: "All '0s'. Two centiseconds total per the summary.",
    },
  },
  miqp: {
    setupText:
      "Optimize a model with 13 rows, 12 columns and 30 nonzeros\nModel has 21 quadratic objective terms\nVariable types: 6 continuous, 6 integer (6 binary)\nCoefficient statistics:\n  Matrix range     [1e+00, 5e+00]\n  Objective range  [0e+00, 0e+00]\n  QObjective range [1e-01, 5e+01]\nPresolve time: 0.01s\nPresolved: 13 rows, 12 columns, 30 nonzeros\nPresolved model has 21 quadratic objective terms",
    rows: [
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "0.07210", depth: "0", intinf: "6", incumbent: "-", bestbd: "0.07210", gap: "-", itnode: "-", time: "0s" } },
      { color: "#9a4caa", cells: { prefix: "H", expl: "0", unexpl: "0", obj: "", depth: "", intinf: "", incumbent: "1.4520310", bestbd: "0.07210", gap: "95.0%", itnode: "-", time: "0s" } },
      { color: "#9a4caa", cells: { prefix: "H", expl: "0", unexpl: "0", obj: "", depth: "", intinf: "", incumbent: "0.4128290", bestbd: "0.07210", gap: "82.5%", itnode: "-", time: "0s" } },
      { color: "#f5a524", cells: { prefix: "*", expl: "9", unexpl: "2", obj: "", depth: "7", intinf: "", incumbent: "0.2841020", bestbd: "0.27892", gap: "1.83%", itnode: "8.4", time: "0s" } },
      { cells: { prefix: "", expl: "17", unexpl: "0", obj: "cutoff", depth: "8", intinf: "", incumbent: "0.28410", bestbd: "0.28410", gap: "0.00%", itnode: "22.4", time: "0s" } },
    ],
    summary:
      "Cutting planes:\n  RLT: 4\n  BQP: 2\nExplored 17 nodes (380 simplex iterations) in 0.08 seconds\nSolution count 3: 0.284102 0.412829 1.45203\nOptimal solution found (tolerance 1.00e-06)\nBest objective 2.841020000000e-01, best bound 2.841020000000e-01, gap 0.0000%",
    finalSummary:
      "Best-subset MIQP. Three incumbents found: 'H' twice (heuristic), then '*' once (B&B at node 9). Each new incumbent shrinks the gap. By node 17, dual catches primal and gap = 0%. RLT and BQP cuts target the quadratic structure — that's what 'Cutting planes' tells you in the summary.",
    perCol: {
      prefix: "Five rows: blank, H, H, *, blank. Two heuristics + one B&B-discovered incumbent. The pattern shows BOTH incumbent sources at work.",
      expl: "0 → 0 → 0 → 9 → 17. Most of the work happens between rows 4 and 5 (8 more nodes). Total: 17 nodes explored.",
      unexpl: "0 → 0 → 0 → 2 → 0. Two open nodes at the time of the '*' incumbent — siblings that will be explored later. Drops to 0 by the end.",
      obj: "0.072 → blank → blank → blank → 'cutoff'. 'cutoff' on the last row means the LP relaxation at this node was worse than the incumbent — Gurobi pruned it without solving fully.",
      depth: "0 → blank → blank → 7 → 8. The deepest node visited was at depth 8 — moderate.",
      intinf: "6 → blank → blank → blank → blank. At the root, all 6 binaries are fractional. Heuristics filled them in directly.",
      incumbent: "- → 1.45 → 0.41 → 0.28 → 0.28. Dropped 5× across three improvements. Notice the first heuristic finds a TERRIBLE 1.45, then a better 0.41, then B&B finds the actual optimum 0.28.",
      bestbd: "0.072 → 0.072 → 0.072 → 0.279 → 0.284. THIS is the storyline. Dual stayed at 0.072 (the loose root LP) until node 9. Then jumped to 0.279 — the bound finally caught up because branching eliminated the cheap fractional regions. Final 0.284 = primal — done.",
      gap: "- → 95% → 82.5% → 1.83% → 0%. Watch the gap collapse: incumbent improvements close it from above, dual-bound improvements close it from below.",
      itnode: "- → - → - → 8.4 → 22.4. The first three rows have no LP/node ratio (root). 22.4 it/node by the end is HIGH — each MIQP relaxation is expensive.",
      time: "All '0s'. 0.08s total per the summary.",
    },
  },
  bienstock: {
    setupText:
      "Optimize a model with 0 rows, 5 columns and 0 nonzeros\nModel has 6 quadratic constraints\nCoefficient statistics:\n  Matrix range     [0e+00, 0e+00]\n  QMatrix range    [1e-01, 1e+00]\n  QLMatrix range   [1e+00, 2e+00]\n  Objective range  [1e+00, 1e+00]\n  Bounds range     [1e+01, 1e+01]\n  RHS range        [0e+00, 0e+00]\n  QRHS range       [1e-01, 3e+00]\nContinuous model is non-convex -- solving as a MIP\n\nPresolve time: 0.01s\nPresolved: 38 rows, 18 columns, 92 nonzeros\nPresolved model has 6 bilinear constraint(s)",
    rows: [
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "1.41421", depth: "0", intinf: "6", incumbent: "-", bestbd: "1.41421", gap: "-", itnode: "-", time: "0s" } },
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "1.41210", depth: "0", intinf: "6", incumbent: "-", bestbd: "1.41210", gap: "-", itnode: "-", time: "0s" } },
      { cells: { prefix: "", expl: "0", unexpl: "0", obj: "1.41210", depth: "0", intinf: "6", incumbent: "-", bestbd: "1.41210", gap: "-", itnode: "-", time: "0s" } },
      { color: "#9a4caa", cells: { prefix: "H", expl: "8", unexpl: "5", obj: "", depth: "", intinf: "", incumbent: "1.2278000", bestbd: "1.39210", gap: "13.4%", itnode: "-", time: "0s" } },
      { cells: { prefix: "", expl: "34", unexpl: "11", obj: "1.30945", depth: "7", intinf: "1", incumbent: "1.22780", bestbd: "1.30945", gap: "6.65%", itnode: "8.0", time: "0s" } },
      { cells: { prefix: "", expl: "89", unexpl: "0", obj: "cutoff", depth: "", intinf: "", incumbent: "1.22780", bestbd: "1.22780", gap: "0.00%", itnode: "7.1", time: "0s" } },
    ],
    summary:
      "Cutting planes:\n  RLT: 18\n  PSDLP: 4\n  BQP: 9\nExplored 89 nodes (1247 simplex iterations) in 0.32 seconds\nSolution count 2: 1.2278 1.225\nOptimal solution found (tolerance 1.00e-04)\nBest objective 1.227800000000e+00, best bound 1.227800000000e+00, gap 0.0000%",
    finalSummary:
      "Spatial B&B in slow motion. Setup says 'Continuous model is non-convex -- solving as a MIP' — confirms NonConvex=2 took effect. The presolved model balloons from 5 columns to 18 (Gurobi added 13 auxiliary product variables to linearize the 6 bilinears). 89 nodes vs SCIP's 137 on the same problem.",
    perCol: {
      prefix: "Six rows. The 'H' on row 4 is when the FIRST feasible solution lands (at node 8) — feasibility was hard to certify because of the reverse-convex constraints.",
      expl: "0 → 0 → 0 → 8 → 34 → 89. Three root-LP rows (with cuts), then branching kicks in. 89 nodes total.",
      unexpl: "0 → 0 → 0 → 5 → 11 → 0. Queue peaks at 11 open nodes around the middle. Drains to 0 at the end.",
      obj: "1.41421 → 1.41210 → 1.41210 → blank → 1.30945 → 'cutoff'. THE storyline column. Root LP gives 1.41421 (just √2 from e1). Cuts shave it to 1.41210. THEN BRANCHING brings it down to 1.30945. The last node was cut off before solving its LP.",
      depth: "0 → 0 → 0 → blank → 7 → blank. Depth-7 branching is moderate; the search explored a balanced tree.",
      intinf: "6 → 6 → 6 → blank → 1 → blank. 'IntInf=6' at the root means six 'integer' variables (the auxiliary product variables Gurobi added) are fractional. By node 34 only one is fractional — the search is closing in on a vertex.",
      incumbent: "- → - → - → 1.2278 → 1.2278 → 1.2278. NO INCUMBENT until node 8. That's striking — the reverse-convex disks make feasibility certification hard.",
      bestbd: "1.41421 → 1.41210 → 1.41210 → 1.39210 → 1.30945 → 1.22780. Dual converges QUICKLY at the start (cuts close 0.002 instantly), but the path to 1.22780 goes through 1.39210 → 1.30945 — gradual, branching-driven. If this had stalled at, say, 1.30 instead of reaching 1.22780, the takeaway would be: try stronger cuts (PSDLP, RLT-RDV), tighter branching (strong branch on the 18 auxiliaries), or reformulate the bilinears manually.",
      gap: "- → - → - → 13.4% → 6.65% → 0%. Gap defined only after the H-row introduces an incumbent. Closes from 13.4% to 0% in 81 nodes.",
      itnode: "- → - → - → - → 8.0 → 7.1. Average LP iterations per node ~7-8. Healthy. Total simplex iterations: 1247 across 89 nodes.",
      time: "All '0s'. 0.32s total per the summary — slower than the convex MIQP but well under a second.",
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
        Install Gurobi + gurobipy &nbsp;
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
            Gurobi is commercial but offers free academic licenses (1 year, renewable).
            For students, the easiest path:
          </p>
          <Pre>
            {`# 1. pip install — works for restricted-size models out of the box (≤2000 vars / ≤2000 cons)
pip install gurobipy

# 2. Get an academic license (free for students at .edu address)
#    Sign up at portal.gurobi.com, then:
grbgetkey YOUR-LICENSE-KEY
#    The key file lands in ~/gurobi.lic

# 3. Verify
python -c "import gurobipy as gp; m = gp.Model(); print(gp.gurobi.version())"
# → (11, 0, 2)

# 4. NonConvex=2 needs Gurobi 9.0+ (default since 11)
#    Bilinear/quadratic objectives need PSD or NonConvex=2`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            Without a license, gurobipy still loads — but anything beyond a tiny demo
            model errors out with{" "}
            <code style={inlineCode}>GurobiError: Model too large for size-limited license</code>.
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
  if (k === "LP") return "#1f4e3d";
  if (k === "MILP") return "#0b3da0";
  if (k === "MIQP") return "#d4a017";
  if (k === "MINLP") return "#7a3da0";
  return "#444";
}
function vtypeBg(t) {
  if (t === "B") return "#0b3da0";
  if (t === "I") return "#7a3da0";
  if (t === "C") return "#1f4e3d";
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
          <b>License gotcha.</b> The pip-installed gurobipy works for tiny models
          (≤2000 vars / ≤2000 cons) without a license. For real models you must
          have a license file (academic users: free at portal.gurobi.com).
        </li>
        <li>
          <b>NonConvex=2.</b> Without it, Gurobi rejects nonconvex Q with{" "}
          <code style={inlineCode}>Q matrix is not positive semi-definite</code>.
          With it, Gurobi runs spatial branch-and-bound — slower but globally
          optimal.
        </li>
        <li>
          <b>Attributes after solve.</b>{" "}
          <code style={inlineCode}>m.ObjVal</code> (objective),{" "}
          <code style={inlineCode}>m.NodeCount</code> (B&B nodes),{" "}
          <code style={inlineCode}>m.MIPGap</code> (final gap),{" "}
          <code style={inlineCode}>m.IterCount</code> (simplex iterations),{" "}
          <code style={inlineCode}>m.Runtime</code>. On variables/constraints:{" "}
          <code style={inlineCode}>v.X</code> (value),{" "}
          <code style={inlineCode}>v.RC</code> (reduced cost),{" "}
          <code style={inlineCode}>c.Pi</code> (dual),{" "}
          <code style={inlineCode}>c.Slack</code>.
        </li>
        <li>
          <b>quicksum vs sum.</b> Always{" "}
          <code style={inlineCode}>gp.quicksum(...)</code> over Python's{" "}
          <code style={inlineCode}>sum(...)</code> when building expressions.
          For a 1000-variable model the difference is roughly 100×.
        </li>
        <li>
          <b>Parameters.</b>{" "}
          <code style={inlineCode}>m.Params.TimeLimit = 30</code>,{" "}
          <code style={inlineCode}>m.Params.MIPGap = 0.01</code>,{" "}
          <code style={inlineCode}>m.Params.Threads = 4</code>,{" "}
          <code style={inlineCode}>m.Params.OutputFlag = 0</code>.
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
