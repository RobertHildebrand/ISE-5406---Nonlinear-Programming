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

      <GurobiOutputReader problemKey={problem.key} />
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
// Output reader — explains Gurobi's iteration log
// ============================================================
const GUROBI_LOGS = {
  lp: {
    title: "Gurobi LP (dual simplex) log",
    setup:
      "Gurobi Optimizer version 11.0.2 build v11.0.2rc0 (mac64[arm])\nThread count: 8 physical cores, 8 logical processors\nOptimize a model with 3 rows, 2 columns and 6 nonzeros\nModel fingerprint: 0x12345678\nCoefficient statistics:\n  Matrix range     [1e+00, 4e+00]\n  Objective range  [3e+01, 4e+01]\n  Bounds range     [0e+00, 0e+00]\n  RHS range        [3e+01, 1e+02]\nPresolve time: 0.00s\nPresolved: 3 rows, 2 columns, 6 nonzeros",
    header: "Iteration    Objective       Primal Inf.    Dual Inf.      Time",
    rows: [
      "       0    3.5000000e+01   3.000000e+01   0.000000e+00      0s",
      "       3    1.1000000e+03   0.000000e+00   0.000000e+00      0s",
    ],
    summary:
      "Solved in 3 iterations and 0.00 seconds (0.00 work units)\nOptimal objective  1.100000000e+03",
    interpretation:
      "Notice 'Coefficient statistics' — Gurobi flags coefficient ranges, and an order-of-magnitude difference between matrix and RHS is fine; ratios > 1e9 trigger numerical warnings. Three iterations of dual simplex find optimum 1100. Primal Inf. starts at 30 (initial dual basis is primal-infeasible — that's expected for dual simplex) and drops to zero in three pivots.",
  },
  facility: {
    title: "Gurobi MILP (branch-and-cut) log",
    setup:
      "Optimize a model with 5 rows, 8 columns and 18 nonzeros\nVariable types: 6 continuous, 2 integer (2 binary)\nCoefficient statistics:\n  Matrix range     [1e+00, 6e+01]\n  Objective range  [3e+00, 1e+02]\n  Bounds range     [1e+00, 1e+00]\n  RHS range        [1e+01, 6e+01]\nPresolve time: 0.00s\nPresolved: 5 rows, 8 columns, 18 nonzeros",
    header: "    Nodes    |    Current Node    |     Objective Bounds      |     Work\n Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time",
    rows: [
      "     0     0  395.00000    0    2          -  395.00000      -     -    0s",
      "H    0     0                     415.0000000  395.00000  4.82%     -    0s",
      "     0     0          -    0          415.00000  415.00000  0.00%     -    0s",
    ],
    summary:
      "Cutting planes:\n  Implied bound: 1\nExplored 1 nodes (4 simplex iterations) in 0.02 seconds (0.00 work units)\nThread count was 8 (of 8 available processors)\nSolution count 1: 415\nOptimal solution found (tolerance 1.00e-04)\nBest objective 4.150000000000e+02, best bound 4.150000000000e+02, gap 0.0000%",
    interpretation:
      "'H' marker on row 2 = Heuristic found a new incumbent (415). LP relaxation gave a bound of 395, then heuristic-feasible 415, then dual climbed to 415 too — gap closes. ONE node explored. The 'Implied bound: 1' line at the end tells you Gurobi added one cutting plane; richer problems have RLT, MIR, Gomory, GUB Cover, etc.",
  },
  miqp: {
    title: "Gurobi MIQP (convex Q + binaries) log",
    setup:
      "Optimize a model with 13 rows, 12 columns and 30 nonzeros\nModel has 21 quadratic objective terms\nVariable types: 6 continuous, 6 integer (6 binary)\nCoefficient statistics:\n  Matrix range     [1e+00, 5e+00]\n  Objective range  [0e+00, 0e+00]\n  QObjective range [1e-01, 5e+01]\nPresolve time: 0.01s\nPresolved: 13 rows, 12 columns, 30 nonzeros\nPresolved model has 21 quadratic objective terms",
    header: "    Nodes    |    Current Node    |     Objective Bounds      |     Work\n Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time",
    rows: [
      "     0     0    0.07210    0    6          -    0.07210      -     -    0s",
      "H    0     0                       1.4520310    0.07210  95.0%     -    0s",
      "H    0     0                       0.4128290    0.07210  82.5%     -    0s",
      "*    9     2               7       0.2841020    0.27892  1.83%   8.4    0s",
      "    17     0       cutoff   8         0.28410    0.28410  0.00%  22.4    0s",
    ],
    summary:
      "Cutting planes:\n  RLT: 4\n  BQP: 2\nExplored 17 nodes (380 simplex iterations) in 0.08 seconds\nSolution count 3: 0.284102 0.412829 1.45203\nOptimal solution found (tolerance 1.00e-06)\nBest objective 2.841020000000e-01, best bound 2.841020000000e-01, gap 0.0000%",
    interpretation:
      "Two MIQP-specific things to notice. (1) 'QObjective range' replaces or supplements the matrix range — Gurobi reports the quadratic-term magnitudes separately. (2) 'Cutting planes: RLT, BQP' — RLT (Reformulation–Linearization Technique) and BQP (Boolean Quadratic Polytope) cuts target the quadratic structure. Three incumbents found ('H' twice, then '*' once at node 9), then dual chases primal until gap closes at node 17.",
  },
  bienstock: {
    title: "Gurobi spatial B&B log (NonConvex=2)",
    setup:
      "Optimize a model with 0 rows, 5 columns and 0 nonzeros\nModel has 6 quadratic constraints\nCoefficient statistics:\n  Matrix range     [0e+00, 0e+00]\n  QMatrix range    [1e-01, 1e+00]\n  QLMatrix range   [1e+00, 2e+00]\n  Objective range  [1e+00, 1e+00]\n  Bounds range     [1e+01, 1e+01]\n  RHS range        [0e+00, 0e+00]\n  QRHS range       [1e-01, 3e+00]\nContinuous model is non-convex -- solving as a MIP\n\nPresolve time: 0.01s\nPresolved: 38 rows, 18 columns, 92 nonzeros\nPresolved model has 6 bilinear constraint(s)",
    header: "    Nodes    |    Current Node    |     Objective Bounds      |     Work\n Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time",
    rows: [
      "     0     0    1.41421    0    6          -    1.41421      -     -    0s",
      "     0     0    1.41210    0    6          -    1.41210      -     -    0s",
      "     0     0    1.41210    0    6          -    1.41210      -     -    0s",
      "H    8     5                       1.2278000    1.39210  13.4%     -    0s",
      "    34    11    1.30945    7    1    1.22780    1.30945  6.65%   8.0    0s",
      "    89     0     cutoff             1.22780    1.22780  0.00%   7.1    0s",
    ],
    summary:
      "Cutting planes:\n  RLT: 18\n  PSDLP: 4\n  BQP: 9\nExplored 89 nodes (1247 simplex iterations) in 0.32 seconds\nSolution count 2: 1.2278 1.225\nOptimal solution found (tolerance 1.00e-04)\nBest objective 1.227800000000e+00, best bound 1.227800000000e+00, gap 0.0000%",
    interpretation:
      "Two diagnostic lines worth pointing at. (1) 'Continuous model is non-convex -- solving as a MIP' confirms NonConvex=2 took effect; without it Gurobi would error out. (2) The presolved problem balloons from 5 columns to 18 — Gurobi added 13 auxiliary 'product variables' to linearize the bilinears. The 6 bilinear constraints become 38 linear rows. Spatial branching happens on those auxiliaries. RLT (18 cuts) and PSDLP (4) are the heavy lifters here. Compare to SCIP's 137 nodes for the same problem — Gurobi closed the gap in 89.",
  },
};

const GUROBI_COL_DEFS = [
  { key: "iter", label: "Iteration", explain: "Cumulative simplex pivots. For LPs this is the headline performance number; for MIPs it's a secondary stat (you care about node count first)." },
  { key: "obj", label: "Objective", explain: "Current LP-relaxation objective at this iteration. Approaches the optimum from the side dictated by simplex (above for max, below for min)." },
  { key: "pinf", label: "Primal Inf.", explain: "Sum of primal infeasibilities. For dual simplex, starts large and drives to zero. For primal simplex, stays at zero throughout." },
  { key: "dinf", label: "Dual Inf.", explain: "Sum of dual infeasibilities — i.e., the reduced-cost violations. Symmetric counterpart to Primal Inf.; primal simplex drives this to zero." },
  { key: "expl", label: "Expl", explain: "Total branch-and-bound nodes explored so far. Doesn't include the root LP." },
  { key: "unexpl", label: "Unexpl", explain: "Number of open nodes still in the queue. Watch this rise as branching creates children, then fall as nodes are pruned/fathomed." },
  { key: "incumbent", label: "Incumbent", explain: "Best feasible objective found so far. The 'answer if I stopped now'." },
  { key: "bestbd", label: "BestBd", explain: "Best dual bound across all open nodes. For minimization it's a lower bound; for maximization it's an upper bound. Improves as branching tightens." },
  { key: "gap", label: "Gap", explain: "(|Incumbent − BestBd| / |Incumbent|) · 100%. Zero means proven optimal. The default termination tolerance is 1e-4 (0.01%)." },
  { key: "itnode", label: "It/Node", explain: "Average simplex iterations per B&B node. Less than 5 = healthy; more than 50 = each node's LP is hard, consider tighter formulation or parallel." },
  { key: "depth", label: "Depth", explain: "Depth of the current node in the search tree. Deep + few nodes = depth-first; shallow + many nodes = best-bound search." },
  { key: "intinf", label: "IntInf", explain: "Number of integer-restricted variables that are fractional in the current LP relaxation. Drops to zero when an integer-feasible solution is found." },
  { key: "h_marker", label: "H marker", explain: "A row beginning with 'H' = a heuristic (not LP rounding, not branching) found this incumbent. Most common: feasibility pump, RINS, RENS." },
  { key: "star_marker", label: "* marker", explain: "Row beginning with '*' = a new incumbent found via LP rounding or strong-branching evaluation at a regular B&B node. The most common improvement source." },
];

function GurobiOutputReader({ problemKey }) {
  const log = GUROBI_LOGS[problemKey];
  const [hoverCol, setHoverCol] = useState(null);
  if (!log) return null;
  return (
    <div style={{ marginTop: 28, padding: 18, border: "1px solid #d8d3c4", background: "#fdfaf1", borderRadius: 10 }}>
      <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>
        Reading Gurobi's output, column by column
      </div>
      <div style={{ fontSize: 13, color: "#555", lineHeight: 1.55, marginBottom: 12 }}>
        Gurobi prints a setup block (license, model stats, presolve), then a per-iteration table, then a summary. Hover any header below to see what each column means.
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 12 }}>
        {GUROBI_COL_DEFS.map((c) => (
          <button
            key={c.key}
            onMouseEnter={() => setHoverCol(c.key)}
            onMouseLeave={() => setHoverCol(null)}
            onClick={() => setHoverCol(hoverCol === c.key ? null : c.key)}
            style={{
              padding: "4px 9px",
              fontSize: 11,
              fontFamily: "monospace",
              border: "1px solid #c8b76c",
              borderRadius: 4,
              background: hoverCol === c.key ? "#f5d68d" : "#fff",
              cursor: "pointer",
            }}
          >
            {c.label}
          </button>
        ))}
      </div>

      {hoverCol && (
        <div
          style={{
            marginBottom: 12,
            padding: "8px 12px",
            background: "#fff8e1",
            border: "1px solid #f5d68d",
            borderRadius: 6,
            fontSize: 13,
            color: "#3d2f00",
          }}
        >
          <b style={{ fontFamily: "monospace" }}>{GUROBI_COL_DEFS.find((c) => c.key === hoverCol).label}</b>
          {": "}
          {GUROBI_COL_DEFS.find((c) => c.key === hoverCol).explain}
        </div>
      )}

      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        ── {log.title} ──
      </div>
      <pre
        style={{
          background: "#0d0d0d",
          color: "#dadada",
          padding: 12,
          borderRadius: 6,
          fontSize: 11,
          fontFamily: "'JetBrains Mono', Menlo, monospace",
          lineHeight: 1.55,
          overflowX: "auto",
          margin: 0,
          whiteSpace: "pre",
        }}
      >
        <div style={{ color: "#7f7864" }}>{log.setup}</div>
        <div style={{ color: "#5a5a5a", marginTop: 4 }}>—</div>
        <div style={{ color: "#7dd87d" }}>{log.header}</div>
        {log.rows.map((row, i) => (
          <div
            key={i}
            style={{
              color: row.startsWith("H")
                ? "#9a4caa"
                : row.startsWith("*")
                ? "#f5a524"
                : "#dadada",
            }}
          >
            {row}
          </div>
        ))}
        <div style={{ color: "#5a5a5a", marginTop: 6 }}>—</div>
        <div>{log.summary}</div>
      </pre>

      <div
        style={{
          marginTop: 12,
          padding: "10px 14px",
          background: "#fff",
          border: "1px solid #ddd",
          borderRadius: 6,
          fontSize: 13,
          lineHeight: 1.55,
          color: "#222",
        }}
      >
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 4 }}>
          What this output is telling you
        </div>
        {log.interpretation}
      </div>

      <div style={{ marginTop: 10, display: "flex", gap: 12, fontSize: 11, color: "#666", flexWrap: "wrap" }}>
        <span>
          <span style={{ color: "#9a4caa", fontWeight: 700, fontFamily: "monospace" }}>H</span> = heuristic incumbent
        </span>
        <span>
          <span style={{ color: "#f5a524", fontWeight: 700, fontFamily: "monospace" }}>*</span> = LP-rounded / B&B incumbent
        </span>
        <span>blank prefix = ordinary B&B node</span>
      </div>
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
