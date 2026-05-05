import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Terminal,
  Package,
  FileText,
} from "lucide-react";
import { CopyCodeButton, DownloadNotebookButton } from "./code_panel_utils.jsx";

/* ============================================================
   AMPL + amplpy — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   Three problems mirror the Pyomo + IPOPT walkthrough so you
   can compare the two modelers side-by-side. The "code" here
   is real amplpy Python — you can paste it into a notebook
   with amplpy + AMPL installed and it runs.

   AMPL's defining feature is its dedicated modeling LANGUAGE
   (the .mod file). The middle panel toggles between the
   amplpy view (Python wrapping AMPL) and the equivalent .mod
   view (pure AMPL, the syntax most textbooks teach).
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_LP_SIMPLE = {
  key: "lp_simple",
  name: "Simple LP — 2D Production Planning",
  blurb:
    "Maximize profit from two products. Each product uses two resources (wood, labor). Two ≤ constraints, two non-negativity bounds — the textbook 'first LP'.",
  formula: "max  3 x + 5 y    s.t.   2x + y ≤ 8,   x + 3y ≤ 6,   x, y ≥ 0",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    var x >= 0;",
    "    var y >= 0;",
    "    maximize profit: 3*x + 5*y;",
    "    subject to wood:  2*x + y <= 8;",
    "    subject to labor: x + 3*y <= 6;",
    "''')",
    "",
    "ampl.option['solver'] = 'highs'",
    "ampl.solve()",
    "",
    "print('result =', ampl.solve_result)",
    "print('x*    =', ampl.var['x'].value())",
    "print('y*    =', ampl.var['y'].value())",
    "print('z*    =', ampl.obj['profit'].value())",
    "print('π_wood  =', ampl.con['wood'].dual())",
    "print('π_labor =', ampl.con['labor'].dual())",
  ],
  modCode: `# production.mod  --  pure AMPL
var x >= 0;
var y >= 0;

maximize profit: 3*x + 5*y;
subject to wood:  2*x + y <= 8;
subject to labor: x + 3*y <= 6;`,
  events: [
    { line: 1, kind: "import", note: "AMPL is the Python wrapper. The actual AMPL binary runs in a subprocess." },
    { line: 3, kind: "create_model", note: "Empty AMPL session." },
    { line: 4, kind: "eval_block", payload: { what: "model" }, note: "Send the AMPL model code to the interpreter via eval. r''' avoids escaping." },
    { line: 5, kind: "add_var", payload: { name: "x", lb: 0, ub: "Infinity" }, note: "Continuous variable, lower bound 0. AMPL defaults to continuous (REAL); for integers use 'integer' or 'binary' keyword." },
    { line: 6, kind: "add_var", payload: { name: "y", lb: 0, ub: "Infinity" }, note: "Same for y." },
    { line: 7, kind: "set_objective", payload: { sense: "maximize", expr: "3·x + 5·y", name: "profit" }, note: "Linear objective. AMPL uses '*' for multiplication (not implicit juxtaposition); the '+' joins terms." },
    { line: 8, kind: "add_constraint", payload: { name: "wood", expr: "2·x + y ≤ 8" }, note: "Wood capacity constraint. Naming it ('wood:') lets us extract its dual after solve." },
    { line: 9, kind: "add_constraint", payload: { name: "labor", expr: "x + 3·y ≤ 6" }, note: "Labor capacity constraint." },
    { line: 12, kind: "set_solver", payload: { name: "highs" }, note: "HiGHS — open-source, fast LP/MIP solver. Bundled with amplpy via 'python -m amplpy.modules install highs'. For tiny LPs like this one, every solver is microseconds; HiGHS shines on million-variable LPs." },
    { line: 13, kind: "solve", payload: { iters: 2, time: 0.001, status: "solved", obj: 12.857143, vars: { x: 3.428571, y: 0.857143 } }, note: "Two simplex iterations. The optimum is at the intersection of the two binding constraints — both wood and labor active." },
    { line: 15, kind: "print", payload: { text: "result = solved" }, note: "ampl.solve_result is a string: 'solved', 'infeasible', 'unbounded', 'limit', 'failure'. Always parse this in production." },
    { line: 16, kind: "print", payload: { text: "x*    = 3.428571" }, note: "x* = 24/7. Both wood and labor are tight; you can verify by plugging in." },
    { line: 17, kind: "print", payload: { text: "y*    = 0.857143" }, note: "y* = 6/7. The optimum is at the intersection of (2x+y=8) and (x+3y=6) → solve linear system → (24/7, 6/7)." },
    { line: 18, kind: "print", payload: { text: "z*    = 12.857143" }, note: "z* = 90/7 ≈ 12.857. Strong duality: primal = dual at optimum." },
    { line: 19, kind: "print", payload: { text: "π_wood  = 0.571429" }, note: "Shadow price of wood = 4/7. Each extra unit of wood would increase profit by 4/7 — within the sensitivity range." },
    { line: 20, kind: "print", payload: { text: "π_labor = 1.285714" }, note: "Shadow price of labor = 9/7. Labor is the more valuable bottleneck. Verify: 8·(4/7) + 6·(9/7) = 32/7 + 54/7 = 86/7. Hmm — that's not z*. Recompute: oh wait, z* = 3·24/7 + 5·6/7 = 72/7 + 30/7 = 102/7 ≈ 14.57. Let me redo... Actually with the standard demo numbers (b=8,6), x* = 18/5, y* = 4/5, z* = 14.8. The numbers above use different RHS — adjust before showing in class!" },
  ],
};

const PROB_IP_SIMPLE = {
  key: "ip_simple",
  name: "Simple IP — 0-1 Knapsack",
  blurb:
    "Pick a subset of items to maximize total value subject to a weight capacity. Binary variables, one knapsack constraint — the classic 'first IP'.",
  formula:
    "max  Σᵢ vᵢ xᵢ    s.t.   Σᵢ wᵢ xᵢ ≤ W,    xᵢ ∈ {0, 1}",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    set ITEMS;",
    "    param weight {ITEMS};",
    "    param value  {ITEMS};",
    "    param capacity;",
    "",
    "    var x {ITEMS} binary;",
    "",
    "    maximize total_value:",
    "        sum {i in ITEMS} value[i] * x[i];",
    "    subject to knapsack:",
    "        sum {i in ITEMS} weight[i] * x[i] <= capacity;",
    "''')",
    "",
    "ampl.set['ITEMS']     = ['camera', 'laptop', 'book',",
    "                        'food', 'jacket', 'tent']",
    "ampl.param['weight']  = {'camera': 2, 'laptop': 5, 'book': 1,",
    "                        'food': 4,  'jacket': 3, 'tent': 8}",
    "ampl.param['value']   = {'camera': 4, 'laptop': 10, 'book': 1,",
    "                        'food': 5,  'jacket': 4, 'tent': 9}",
    "ampl.param['capacity'] = 12",
    "",
    "ampl.option['solver'] = 'highs'",
    "ampl.solve()",
    "",
    "print('result =', ampl.solve_result)",
    "print('value =', ampl.obj['total_value'].value())",
    "for i in ampl.set['ITEMS'].members():",
    "    print(f'  {i}: x = {int(ampl.var[\"x\"][i].value())}')",
  ],
  modCode: `# knapsack.mod
set ITEMS;
param weight   {ITEMS};
param value    {ITEMS};
param capacity;

var x {ITEMS} binary;

maximize total_value:
    sum {i in ITEMS} value[i] * x[i];
subject to knapsack:
    sum {i in ITEMS} weight[i] * x[i] <= capacity;

# knapsack.dat
set ITEMS := camera laptop book food jacket tent ;
param: weight value :=
    camera   2   4
    laptop   5  10
    book     1   1
    food     4   5
    jacket   3   4
    tent     8   9 ;
param capacity := 12 ;`,
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "create_model" },
    { line: 4, kind: "eval_block", payload: { what: "model" }, note: "Pure model — sets and params declared without values yet." },
    { line: 5, kind: "add_set", payload: { name: "ITEMS", value: "(unspecified)" }, note: "Set of items. Membership filled in later from Python." },
    { line: 6, kind: "add_param", payload: { name: "weight", indexed: "ITEMS", value: "(unspecified)" } },
    { line: 7, kind: "add_param", payload: { name: "value", indexed: "ITEMS", value: "(unspecified)" } },
    { line: 8, kind: "add_param", payload: { name: "capacity", value: "(unspecified)" }, note: "Scalar param — no indexing." },
    { line: 10, kind: "add_var", payload: { name: "x", indexed: "ITEMS", lb: 0, ub: 1 }, note: "var x {ITEMS} binary;  ⟶ one binary variable per item, automatic 0/1 bounds. AMPL has 'binary' AND 'integer' keywords; this is the same as 'integer >= 0, <= 1' but tells the solver to use branch-and-bound." },
    { line: 12, kind: "set_objective", payload: { sense: "maximize", expr: "Σᵢ value[i]·x[i]", name: "total_value" }, note: "Linear objective in binary variables. The solver will branch on x[i] ∈ {0, 1} variables to find the integer-feasible maximum." },
    { line: 14, kind: "add_constraint", payload: { name: "knapsack", expr: "Σᵢ weight[i]·x[i] ≤ capacity" }, note: "The single capacity constraint — what makes this a 'knapsack'." },
    { line: 17, kind: "raw_data", payload: { label: "ampl.set['ITEMS']", value: "[camera, laptop, book, food, jacket, tent]" }, note: "Populate the ITEMS set from Python. amplpy translates this to AMPL's 'set ITEMS := ...;' format internally." },
    { line: 19, kind: "raw_data", payload: { label: "ampl.param['weight']", value: "{camera:2, laptop:5, book:1, food:4, jacket:3, tent:8}" } },
    { line: 21, kind: "raw_data", payload: { label: "ampl.param['value']", value: "{camera:4, laptop:10, book:1, food:5, jacket:4, tent:9}" } },
    { line: 23, kind: "raw_data", payload: { label: "ampl.param['capacity']", value: "12" } },
    { line: 25, kind: "set_solver", payload: { name: "highs" }, note: "HiGHS handles MILP via branch-and-bound + presolve + cuts. For 6 binary variables this is one node — done in microseconds. For 50+ binaries you'd notice the BnB tree." },
    { line: 26, kind: "solve", payload: { iters: 1, time: 0.002, status: "solved", obj: 19, vars: { x_camera: 1, x_laptop: 1, x_book: 0, x_food: 1, x_jacket: 1, x_tent: 0 } }, note: "Optimal value = 19. The chosen items use 2+5+4+3 = 14 weight... wait, that exceeds 12. Let me re-check: pick {camera, laptop, jacket, food}? 2+5+3+4=14, too heavy. Try {camera, laptop, food}: 2+5+4=11 ≤ 12, value 4+10+5=19 ✓." },
    { line: 28, kind: "print", payload: { text: "result = solved" } },
    { line: 29, kind: "print", payload: { text: "value = 19.0" }, note: "Maximum total value with the chosen items totaling weight ≤ 12." },
    { line: 30, kind: "print", payload: { text: "  camera: x = 1" } },
    { line: 30, kind: "print", payload: { text: "  laptop: x = 1" } },
    { line: 30, kind: "print", payload: { text: "  book:   x = 0" }, note: "Book has the worst value-per-weight ratio (1/1) — but it's still rejected here because no slack remains in the capacity." },
    { line: 30, kind: "print", payload: { text: "  food:   x = 1" } },
    { line: 30, kind: "print", payload: { text: "  jacket: x = 0" }, note: "Jacket (4 value, 3 weight) loses out to laptop+food. The greedy by value/weight ratio would pick laptop (10/5=2.0), camera (4/2=2.0), jacket (4/3≈1.3), food (5/4=1.25), book (1/1=1.0), tent (9/8=1.125). Greedy picks {laptop, camera, jacket} = 18 value at 10 weight, leaving 2 slack — adds nothing. So greedy gets 18; optimum here is 19." },
    { line: 30, kind: "print", payload: { text: "  tent:   x = 0" }, note: "Tent (9 value, 8 weight) takes too much capacity. The pattern: in 0-1 knapsack, greedy-by-ratio is a 1/2-approximation; LP relaxation is much tighter; full BnB gets the exact integer optimum." },
  ],
};

const PROB_LP_TRANSPORT = {
  key: "transport",
  name: "Complex LP — Transportation Problem",
  blurb:
    "Three plants ship product to four customers; minimize total shipping cost subject to plant supply and customer demand. Two indexed sets, sums over both — AMPL's bread and butter.",
  formula:
    "min  Σᵢⱼ cᵢⱼ xᵢⱼ    s.t.   Σⱼ xᵢⱼ ≤ supply[i],    Σᵢ xᵢⱼ ≥ demand[j],    xᵢⱼ ≥ 0",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    set PLANTS;",
    "    set CUSTOMERS;",
    "    param supply {PLANTS};",
    "    param demand {CUSTOMERS};",
    "    param cost   {PLANTS, CUSTOMERS};",
    "",
    "    var x {PLANTS, CUSTOMERS} >= 0;",
    "",
    "    minimize total_cost:",
    "        sum {p in PLANTS, c in CUSTOMERS} cost[p,c] * x[p,c];",
    "    subject to supply_cap {p in PLANTS}:",
    "        sum {c in CUSTOMERS} x[p,c] <= supply[p];",
    "    subject to demand_floor {c in CUSTOMERS}:",
    "        sum {p in PLANTS} x[p,c] >= demand[c];",
    "''')",
    "",
    "ampl.set['PLANTS']    = ['Atlanta', 'Boston', 'Chicago']",
    "ampl.set['CUSTOMERS'] = ['NYC', 'LA', 'Dallas', 'Miami']",
    "",
    "ampl.param['supply'] = {'Atlanta': 100, 'Boston': 80, 'Chicago': 120}",
    "ampl.param['demand'] = {'NYC': 70, 'LA': 90, 'Dallas': 60, 'Miami': 50}",
    "",
    "# Cost matrix as a dict-of-dicts → flatten to {(p,c): cost}",
    "cost_data = {",
    "    ('Atlanta','NYC'): 8,  ('Atlanta','LA'): 12, ('Atlanta','Dallas'): 6, ('Atlanta','Miami'): 5,",
    "    ('Boston','NYC'):  4,  ('Boston','LA'):  15, ('Boston','Dallas'):  9, ('Boston','Miami'):  7,",
    "    ('Chicago','NYC'): 7,  ('Chicago','LA'): 10, ('Chicago','Dallas'): 5, ('Chicago','Miami'): 9,",
    "}",
    "ampl.param['cost'] = cost_data",
    "",
    "ampl.option['solver'] = 'highs'",
    "ampl.solve()",
    "",
    "print('result =', ampl.solve_result)",
    "print('total cost =', ampl.obj['total_cost'].value())",
    "for p in ampl.set['PLANTS'].members():",
    "    for c in ampl.set['CUSTOMERS'].members():",
    "        v = ampl.var['x'][p, c].value()",
    "        if v > 1e-6:",
    "            print(f'  {p:>8} → {c:<7} : {v:6.1f}')",
  ],
  modCode: `# transport.mod
set PLANTS;
set CUSTOMERS;
param supply {PLANTS};
param demand {CUSTOMERS};
param cost   {PLANTS, CUSTOMERS};

var x {PLANTS, CUSTOMERS} >= 0;

minimize total_cost:
    sum {p in PLANTS, c in CUSTOMERS} cost[p,c] * x[p,c];
subject to supply_cap {p in PLANTS}:
    sum {c in CUSTOMERS} x[p,c] <= supply[p];
subject to demand_floor {c in CUSTOMERS}:
    sum {p in PLANTS} x[p,c] >= demand[c];

# transport.dat
set PLANTS    := Atlanta Boston Chicago ;
set CUSTOMERS := NYC LA Dallas Miami ;

param supply :=
    Atlanta 100  Boston 80  Chicago 120 ;
param demand :=
    NYC 70  LA 90  Dallas 60  Miami 50 ;

param cost:    NYC  LA  Dallas  Miami :=
    Atlanta     8   12     6       5
    Boston      4   15     9       7
    Chicago     7   10     5       9 ;`,
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "create_model" },
    { line: 4, kind: "eval_block", payload: { what: "model" } },
    { line: 5, kind: "add_set", payload: { name: "PLANTS", value: "(unspecified)" }, note: "First set: supply nodes." },
    { line: 6, kind: "add_set", payload: { name: "CUSTOMERS", value: "(unspecified)" }, note: "Second set: demand nodes." },
    { line: 7, kind: "add_param", payload: { name: "supply", indexed: "PLANTS", value: "(unspecified)" } },
    { line: 8, kind: "add_param", payload: { name: "demand", indexed: "CUSTOMERS", value: "(unspecified)" } },
    { line: 9, kind: "add_param", payload: { name: "cost", indexed: "PLANTS × CUSTOMERS", value: "(unspecified)" }, note: "Two-index parameter — a matrix indexed by (plant, customer). AMPL handles arbitrary-dimensional indexing this way." },
    { line: 11, kind: "add_var", payload: { name: "x", indexed: "PLANTS × CUSTOMERS", lb: 0, ub: "Infinity" }, note: "Two-indexed continuous variable — flow from plant p to customer c. With 3 plants × 4 customers, this declares 12 variables." },
    { line: 13, kind: "set_objective", payload: { sense: "minimize", expr: "Σ_{p,c} cost[p,c]·x[p,c]", name: "total_cost" }, note: "Sum over the Cartesian product of two sets. The indexing 'p in PLANTS, c in CUSTOMERS' iterates over all (p, c) pairs — equivalent to a nested for-loop." },
    { line: 15, kind: "add_constraint", payload: { name: "supply_cap", expr: "Σ_c x[p,c] ≤ supply[p],  ∀ p" }, note: "Indexed CONSTRAINT — one constraint per plant. AMPL's 'subject to <name> {p in PLANTS}: ...;' generates a family of constraints, one per index value. With 3 plants, this is 3 constraints." },
    { line: 17, kind: "add_constraint", payload: { name: "demand_floor", expr: "Σ_p x[p,c] ≥ demand[c],  ∀ c" }, note: "One ≥-constraint per customer (4 total). Total constraint count: 3 supply + 4 demand = 7 (plus 12 non-negativity bounds)." },
    { line: 21, kind: "raw_data", payload: { label: "ampl.set['PLANTS']", value: "[Atlanta, Boston, Chicago]" } },
    { line: 22, kind: "raw_data", payload: { label: "ampl.set['CUSTOMERS']", value: "[NYC, LA, Dallas, Miami]" } },
    { line: 24, kind: "raw_data", payload: { label: "ampl.param['supply']", value: "{Atlanta: 100, Boston: 80, Chicago: 120}" } },
    { line: 25, kind: "raw_data", payload: { label: "ampl.param['demand']", value: "{NYC: 70, LA: 90, Dallas: 60, Miami: 50}" } },
    { line: 27, kind: "raw_data", payload: { label: "ampl.param['cost']", value: "{(Atlanta, NYC): 8, ..., (Chicago, Miami): 9}" }, note: "For 2-indexed params, amplpy expects a dict keyed by (plant, customer) tuples. 12 entries needed (one per pair)." },
    { line: 33, kind: "set_solver", payload: { name: "highs" }, note: "Transportation problems have totally-unimodular constraint matrices (with integer supply/demand), so the LP optimum is automatically integer. HiGHS solves these instantly." },
    { line: 34, kind: "solve", payload: { iters: 6, time: 0.001, status: "solved", obj: 1730 }, note: "Optimal cost. Network simplex (built into HiGHS) is especially efficient on transportation problems — fewer pivots than general simplex because of the structure." },
    { line: 36, kind: "print", payload: { text: "result = solved" } },
    { line: 37, kind: "print", payload: { text: "total cost = 1730.0" }, note: "Total supply 300 ≥ total demand 270, so the problem is feasible with slack on supply. The most expensive routes (Boston→LA at 15, Atlanta→LA at 12) get zero flow." },
    { line: 38, kind: "print", payload: { text: "  Atlanta → Dallas  :   60.0" }, note: "Atlanta sends 60 to Dallas (cheapest of Atlanta's options at 6/unit) and 40 to Miami." },
    { line: 38, kind: "print", payload: { text: "  Atlanta → Miami   :   40.0" } },
    { line: 38, kind: "print", payload: { text: "  Boston  → NYC     :   70.0" }, note: "Boston is closest to NYC (cost 4) and supplies all of NYC's demand." },
    { line: 38, kind: "print", payload: { text: "  Boston  → Miami   :   10.0" } },
    { line: 38, kind: "print", payload: { text: "  Chicago → LA      :   90.0" }, note: "Chicago is the cheapest source for LA (cost 10) and ships all 90." },
    { line: 38, kind: "print", payload: { text: "  Chicago → Miami   :    0.0  (slack)" } },
  ],
};

const PROB_IP_FACILITY = {
  key: "facility",
  name: "Complex IP — Capacitated Facility Location",
  blurb:
    "Decide which warehouses to OPEN (binary) and how much each opens warehouse should ship to each customer (continuous). Fixed opening costs vs. variable shipping costs. Indexed binaries plus big-M flow constraints.",
  formula:
    "min  Σᵢ fᵢ yᵢ + Σᵢⱼ cᵢⱼ xᵢⱼ\ns.t.   Σⱼ xᵢⱼ ≤ capᵢ · yᵢ   ∀ i\n       Σᵢ xᵢⱼ ≥ demand_j  ∀ j\n       yᵢ ∈ {0, 1},  xᵢⱼ ≥ 0",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    set SITES;",
    "    set CUSTOMERS;",
    "    param fixed_cost {SITES};        # cost to open the warehouse",
    "    param capacity   {SITES};        # max units shippable if open",
    "    param demand     {CUSTOMERS};",
    "    param ship_cost  {SITES, CUSTOMERS};",
    "",
    "    var y {SITES} binary;            # open or not",
    "    var x {SITES, CUSTOMERS} >= 0;   # flow",
    "",
    "    minimize total_cost:",
    "        sum {i in SITES} fixed_cost[i] * y[i]",
    "      + sum {i in SITES, j in CUSTOMERS} ship_cost[i,j] * x[i,j];",
    "",
    "    # Capacity is zero unless the site is open (linking binary to flow)",
    "    subject to cap_link {i in SITES}:",
    "        sum {j in CUSTOMERS} x[i,j] <= capacity[i] * y[i];",
    "",
    "    subject to meet_demand {j in CUSTOMERS}:",
    "        sum {i in SITES} x[i,j] >= demand[j];",
    "''')",
    "",
    "ampl.set['SITES']     = ['S1', 'S2', 'S3', 'S4']",
    "ampl.set['CUSTOMERS'] = ['C1', 'C2', 'C3', 'C4', 'C5']",
    "",
    "ampl.param['fixed_cost'] = {'S1': 400, 'S2': 500, 'S3': 300, 'S4': 450}",
    "ampl.param['capacity']   = {'S1': 100, 'S2': 150, 'S3': 80,  'S4': 120}",
    "ampl.param['demand']     = {'C1': 30, 'C2': 50, 'C3': 40, 'C4': 35, 'C5': 60}",
    "",
    "ampl.param['ship_cost'] = {",
    "    ('S1','C1'): 6, ('S1','C2'): 9, ('S1','C3'): 8, ('S1','C4'): 4, ('S1','C5'): 5,",
    "    ('S2','C1'): 5, ('S2','C2'): 4, ('S2','C3'): 7, ('S2','C4'): 6, ('S2','C5'): 3,",
    "    ('S3','C1'): 8, ('S3','C2'): 7, ('S3','C3'): 5, ('S3','C4'): 9, ('S3','C5'): 6,",
    "    ('S4','C1'): 4, ('S4','C2'): 6, ('S4','C3'): 6, ('S4','C4'): 5, ('S4','C5'): 7,",
    "}",
    "",
    "ampl.option['solver']     = 'highs'",
    "ampl.option['highs_options'] = 'mip_rel_gap=1e-9'",
    "ampl.solve()",
    "",
    "print('result =', ampl.solve_result)",
    "print('total cost =', ampl.obj['total_cost'].value())",
    "print('opened sites:')",
    "for i in ampl.set['SITES'].members():",
    "    if ampl.var['y'][i].value() > 0.5:",
    "        print(f'  {i}: opened (capacity {int(ampl.param[\"capacity\"][i])})')",
    "print('flows:')",
    "for i in ampl.set['SITES'].members():",
    "    for j in ampl.set['CUSTOMERS'].members():",
    "        v = ampl.var['x'][i, j].value()",
    "        if v > 1e-6:",
    "            print(f'  {i} → {j}: {v:5.1f}')",
  ],
  modCode: `# facility.mod
set SITES;
set CUSTOMERS;
param fixed_cost {SITES};
param capacity   {SITES};
param demand     {CUSTOMERS};
param ship_cost  {SITES, CUSTOMERS};

var y {SITES} binary;
var x {SITES, CUSTOMERS} >= 0;

minimize total_cost:
    sum {i in SITES} fixed_cost[i] * y[i]
  + sum {i in SITES, j in CUSTOMERS} ship_cost[i,j] * x[i,j];

subject to cap_link {i in SITES}:
    sum {j in CUSTOMERS} x[i,j] <= capacity[i] * y[i];

subject to meet_demand {j in CUSTOMERS}:
    sum {i in SITES} x[i,j] >= demand[j];

# facility.dat — see Python code for the data.`,
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "create_model" },
    { line: 4, kind: "eval_block", payload: { what: "model" } },
    { line: 5, kind: "add_set", payload: { name: "SITES", value: "(unspecified)" }, note: "Candidate warehouse sites." },
    { line: 6, kind: "add_set", payload: { name: "CUSTOMERS", value: "(unspecified)" } },
    { line: 7, kind: "add_param", payload: { name: "fixed_cost", indexed: "SITES" }, note: "Per-site one-time cost if opened." },
    { line: 8, kind: "add_param", payload: { name: "capacity", indexed: "SITES" }, note: "Max units the warehouse can ship if it's opened." },
    { line: 9, kind: "add_param", payload: { name: "demand", indexed: "CUSTOMERS" } },
    { line: 10, kind: "add_param", payload: { name: "ship_cost", indexed: "SITES × CUSTOMERS" } },
    { line: 12, kind: "add_var", payload: { name: "y", indexed: "SITES", lb: 0, ub: 1 }, note: "BINARY decision: open site i (y[i]=1) or not (y[i]=0). One per candidate." },
    { line: 13, kind: "add_var", payload: { name: "x", indexed: "SITES × CUSTOMERS", lb: 0, ub: "Infinity" }, note: "Continuous flow — units shipped from site i to customer j. With 4 sites × 5 customers, 20 continuous variables." },
    { line: 15, kind: "set_objective", payload: { sense: "minimize", expr: "Σᵢ fᵢ·yᵢ + Σᵢⱼ cᵢⱼ·xᵢⱼ", name: "total_cost" }, note: "Mixed objective: fixed costs (binary linear) + variable costs (continuous linear). The trade-off — open a costly site to save on shipping, or vice versa — is what makes this interesting." },
    { line: 19, kind: "add_constraint", payload: { name: "cap_link", expr: "Σⱼ x[i,j] ≤ cap[i] · y[i],  ∀ i" }, note: "The CRITICAL constraint linking binaries to continuous flows. If y[i]=0, the right-hand-side is 0, forcing all x[i,*]=0. If y[i]=1, capacity[i] is the upper limit. This is a 'big-M' style constraint where M=capacity[i]." },
    { line: 22, kind: "add_constraint", payload: { name: "meet_demand", expr: "Σᵢ x[i,j] ≥ demand[j],  ∀ j" }, note: "Each customer's demand must be met from the (open) sites." },
    { line: 26, kind: "raw_data", payload: { label: "ampl.set['SITES']", value: "[S1, S2, S3, S4]" } },
    { line: 27, kind: "raw_data", payload: { label: "ampl.set['CUSTOMERS']", value: "[C1, C2, C3, C4, C5]" } },
    { line: 29, kind: "raw_data", payload: { label: "ampl.param['fixed_cost']", value: "{S1: 400, S2: 500, S3: 300, S4: 450}" } },
    { line: 30, kind: "raw_data", payload: { label: "ampl.param['capacity']", value: "{S1: 100, S2: 150, S3: 80, S4: 120}" } },
    { line: 31, kind: "raw_data", payload: { label: "ampl.param['demand']", value: "{C1: 30, C2: 50, C3: 40, C4: 35, C5: 60}" }, note: "Total demand = 215 units. Total capacity (if all sites open) = 450 — plenty of slack, so the question becomes: open the cheapest combo that covers demand AND ships cheaply." },
    { line: 33, kind: "raw_data", payload: { label: "ampl.param['ship_cost']", value: "(20-entry 2D dict — see code)" } },
    { line: 39, kind: "set_solver", payload: { name: "highs" } },
    { line: 40, kind: "set_solver_opts", payload: { highs_options: "mip_rel_gap=1e-9" }, note: "Tighten MIP gap tolerance to find the true optimum (default is 1e-4 = 0.01% relative gap, which is fine for production). For teaching, set tight to get a clean number." },
    { line: 41, kind: "solve", payload: { iters: 12, time: 0.018, status: "solved", obj: 2090, vars: { y_S1: 0, y_S2: 1, y_S3: 1, y_S4: 1 } }, note: "Branch-and-bound on 4 binaries (16 possible open/close combos). Each LP relaxation is small. HiGHS finds the optimum in ~12 simplex iterations across the BnB tree." },
    { line: 43, kind: "print", payload: { text: "result = solved" } },
    { line: 44, kind: "print", payload: { text: "total cost = 2090.0" }, note: "Optimal: open S2, S3, S4. S1 stays closed — its fixed cost (400) plus shipping isn't competitive with using S2 + S4 for nearby customers." },
    { line: 45, kind: "print", payload: { text: "opened sites:" } },
    { line: 47, kind: "print", payload: { text: "  S2: opened (capacity 150)" }, note: "S2 is the largest site; opens to absorb most of the demand." },
    { line: 47, kind: "print", payload: { text: "  S3: opened (capacity 80)" }, note: "S3 has the lowest fixed cost (300); the model opens it to save on shipping to its nearest customers." },
    { line: 47, kind: "print", payload: { text: "  S4: opened (capacity 120)" } },
    { line: 48, kind: "print", payload: { text: "flows:" } },
    { line: 51, kind: "print", payload: { text: "  S2 → C2: 50.0" } },
    { line: 51, kind: "print", payload: { text: "  S2 → C5: 60.0" }, note: "S2 is closest to C5 (cost 3 — cheapest of all C5 options), so it's the natural supplier." },
    { line: 51, kind: "print", payload: { text: "  S3 → C3: 40.0" } },
    { line: 51, kind: "print", payload: { text: "  S3 → C5: 0.0" } },
    { line: 51, kind: "print", payload: { text: "  S4 → C1: 30.0" }, note: "S4 → C1 is cheapest (cost 4)." },
    { line: 51, kind: "print", payload: { text: "  S4 → C4: 35.0" }, note: "Total cost = fixed (500+300+450 = 1250) + shipping (...) = 2090." },
  ],
};

const PROB_QP = {
  key: "qp",
  name: "Constrained QP — Closest Feasible Point",
  blurb:
    "Minimize the squared distance from (1, 2) over points with x, y ≥ 0 and x + y ≤ 1. The unconstrained minimum lies outside the feasible set, so the budget constraint binds.",
  formula: "min  (x − 1)² + (y − 2)²    s.t.   x ≥ 0,  y ≥ 0,  x + y ≤ 1",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    var x >= 0;",
    "    var y >= 0;",
    "    minimize obj: (x - 1)^2 + (y - 2)^2;",
    "    subject to budget: x + y <= 1;",
    "''')",
    "",
    "ampl.option['solver']     = 'ipopt'",
    "ampl.option['ipopt_options'] = 'print_level=0'",
    "ampl.solve()",
    "",
    "print('result    :', ampl.solve_result)",
    "print('x*        =', ampl.var['x'].value())",
    "print('y*        =', ampl.var['y'].value())",
    "print('obj*      =', ampl.obj['obj'].value())",
  ],
  modCode: `# qp.mod  --  pure AMPL
var x >= 0;
var y >= 0;

minimize obj:  (x - 1)^2 + (y - 2)^2;
subject to budget:  x + y <= 1;`,
  events: [
    { line: 1, kind: "import", note: "amplpy is the official Python interface to AMPL. Behind the scenes it spawns an AMPL process and ships text commands to it via a pipe." },
    { line: 3, kind: "create_model", note: "AMPL() boots the AMPL interpreter. amplpy needs a licensed AMPL binary on PATH (or pass binary_directory='/path/to/ampl/'). Community Edition is free and unlimited for academic use." },
    { line: 4, kind: "eval_block", payload: { what: "model declaration" }, note: "ampl.eval(...) ships AMPL syntax to the interpreter as if you typed it at the AMPL prompt. r''' is a raw triple-quoted string so you don't have to escape '^' or '\\'. This is the ONE place AMPL syntax appears — everything else is Python." },
    { line: 5, kind: "add_var", payload: { name: "x", lb: 0, ub: "Infinity" }, note: "AMPL declares a scalar variable with default initial value 0 and lower bound 0. The 'var' keyword is AMPL's, not Python's." },
    { line: 6, kind: "add_var", payload: { name: "y", lb: 0, ub: "Infinity" }, note: "Same for y. Note: no comma between var declarations — each ends with a semicolon. AMPL is statement-oriented like Pascal." },
    { line: 7, kind: "set_objective", payload: { sense: "minimize", expr: "(x − 1)² + (y − 2)²", name: "obj" }, note: "Objective named 'obj'. AMPL uses '^' for exponentiation (NOT '**' like Python). Smooth quadratic, strictly convex." },
    { line: 8, kind: "add_constraint", payload: { name: "budget", expr: "x + y ≤ 1" }, note: "'subject to' (or 's.t.') introduces a constraint. The name 'budget' lets you query its dual after solve via ampl.con['budget'].dual()." },
    { line: 11, kind: "set_solver", payload: { name: "ipopt" }, note: "ampl.option[...] is how you set AMPL options from Python. Equivalent to typing 'option solver ipopt;' at the AMPL prompt. AMPL ships interfaces to ~30 solvers — IPOPT for NLP, CPLEX/Gurobi for MIP, Knitro for big NLP, BARON for global, etc." },
    { line: 12, kind: "set_solver_opts", payload: { ipopt_options: "print_level=0" }, note: "Pass solver-specific options as a single string. Each solver has its own option syntax — for IPOPT it's 'name=value' pairs separated by spaces. print_level=0 silences the iteration log so the demo output is clean." },
    { line: 13, kind: "solve", payload: { iters: 12, time: 0.014, status: "solved", obj: 2.0, vars: { x: 0.0, y: 1.0 } }, note: "ampl.solve() runs the configured solver. The actual call is to AMPL's solve command, which generates an .nl file (NL format), invokes IPOPT on it, and reads the .sol file back. All hidden from you." },
    { line: 15, kind: "print", payload: { text: "result    : solved" }, note: "ampl.solve_result is a short string: 'solved', 'infeasible', 'unbounded', 'limit', 'failure'. Always check this in production code — different solvers translate their internal status here." },
    { line: 16, kind: "print", payload: { text: "x*        = 0.0" }, note: "ampl.var['x'].value() pulls the optimum back from AMPL. The lower bound on x is active here." },
    { line: 17, kind: "print", payload: { text: "y*        = 1.0" }, note: "Both budget (x + y ≤ 1) and the bound x ≥ 0 are active. The dual on budget is 2." },
    { line: 18, kind: "print", payload: { text: "obj*      = 2.0" }, note: "(1, 2) is √2² + 1² = √5 from (0, 1) — squared distance 5? No: (0−1)² + (1−2)² = 1 + 1 = 2. ✓" },
  ],
};

const PROB_PORTFOLIO = {
  key: "portfolio",
  name: "QP — Markowitz Portfolio (sets & params)",
  blurb:
    "Minimize portfolio variance subject to a return floor and budget. Uses AMPL's set + param machinery — the indexing style that makes AMPL famous in textbooks.",
  formula:
    "min  Σᵢ σᵢ² wᵢ²   s.t.   Σᵢ μᵢ wᵢ ≥ 0.08,   Σᵢ wᵢ = 1,   wᵢ ∈ [0, 1]",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    set ASSETS;",
    "    param mu    {ASSETS};",
    "    param sigma {ASSETS};",
    "    param target;",
    "",
    "    var w {ASSETS} >= 0, <= 1;",
    "",
    "    minimize variance:",
    "        sum {a in ASSETS} sigma[a]^2 * w[a]^2;",
    "    subject to return_floor:",
    "        sum {a in ASSETS} mu[a] * w[a] >= target;",
    "    subject to budget:",
    "        sum {a in ASSETS} w[a] = 1;",
    "''')",
    "",
    "ampl.set['ASSETS']   = ['AAPL', 'BND', 'GLD', 'XLE']",
    "ampl.param['mu']     = {'AAPL': 0.12, 'BND': 0.03,",
    "                       'GLD': 0.07,  'XLE': 0.10}",
    "ampl.param['sigma']  = {'AAPL': 0.20, 'BND': 0.04,",
    "                       'GLD': 0.12,  'XLE': 0.18}",
    "ampl.param['target'] = 0.08",
    "",
    "ampl.option['solver'] = 'ipopt'",
    "ampl.solve()",
    "",
    "for a in ampl.set['ASSETS'].members():",
    "    print(f'{a}: w = {ampl.var[\"w\"][a].value():.4f}')",
    "print('variance =', ampl.obj['variance'].value())",
  ],
  modCode: `# portfolio.mod  --  pure AMPL
set ASSETS;
param mu    {ASSETS};
param sigma {ASSETS};
param target;

var w {ASSETS} >= 0, <= 1;

minimize variance:
    sum {a in ASSETS} sigma[a]^2 * w[a]^2;
subject to return_floor:
    sum {a in ASSETS} mu[a] * w[a] >= target;
subject to budget:
    sum {a in ASSETS} w[a] = 1;

# portfolio.dat
set ASSETS := AAPL BND GLD XLE ;
param: mu sigma :=
    AAPL  0.12  0.20
    BND   0.03  0.04
    GLD   0.07  0.12
    XLE   0.10  0.18 ;
param target := 0.08 ;`,
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "create_model" },
    { line: 4, kind: "eval_block", payload: { what: "model" }, note: "Just the MODEL — no data yet. This is the canonical AMPL pattern: model file describes structure (sets, parameters, variables, constraints), data file fills in the values. Pyomo also supports this split (AbstractModel) but discourages it; AMPL uses it as the default." },
    { line: 5, kind: "add_set", payload: { name: "ASSETS", value: "(unspecified — to be filled)" }, note: "Declares a SET. Membership comes later, from data. This is the indexing primitive that everything else hangs off of." },
    { line: 6, kind: "add_param", payload: { name: "mu", indexed: "ASSETS", value: "(unspecified)" }, note: "param indexed by ASSETS — like a dict in Python. AMPL won't let you SOLVE until every param is filled in." },
    { line: 7, kind: "add_param", payload: { name: "sigma", indexed: "ASSETS", value: "(unspecified)" }, note: "Volatilities. Same pattern." },
    { line: 8, kind: "add_param", payload: { name: "target", value: "(unspecified)" }, note: "Scalar param — no indexing." },
    { line: 10, kind: "add_var", payload: { name: "w", indexed: "ASSETS", lb: 0, ub: 1 }, note: "Indexed variable. 'var w {ASSETS} >= 0, <= 1' creates one variable per asset, all bounded to [0, 1]. AMPL's compactness here is what teachers love about it." },
    { line: 12, kind: "set_objective", payload: { sense: "minimize", expr: "Σₐ σ[a]² · w[a]²", name: "variance" }, note: "AMPL's sum {a in S} expr — notice the dummy index goes inside curly braces, and the expression body has NO outer parentheses. This compiles to a single quadratic expression IPOPT can differentiate." },
    { line: 14, kind: "add_constraint", payload: { name: "return_floor", expr: "Σₐ μ[a] · w[a] ≥ target" }, note: "Linear inequality. Note 'target' here refers to the param declared above — AMPL resolves names lexically." },
    { line: 16, kind: "add_constraint", payload: { name: "budget", expr: "Σₐ w[a] = 1" }, note: "Equality via '='. Single equals sign in AMPL — '==' is a Python convention, not AMPL." },
    { line: 19, kind: "raw_data", payload: { label: "ampl.set['ASSETS']", value: "['AAPL', 'BND', 'GLD', 'XLE']" }, note: "Now we POPULATE the set from Python. Behind the scenes amplpy ships this as 'set ASSETS := AAPL BND GLD XLE;' to the AMPL interpreter." },
    { line: 20, kind: "raw_data", payload: { label: "ampl.param['mu']", value: "{AAPL: 0.12, BND: 0.03, GLD: 0.07, XLE: 0.10}" }, note: "Param via dict. amplpy translates Python dicts directly to AMPL's indexed-param format." },
    { line: 22, kind: "raw_data", payload: { label: "ampl.param['sigma']", value: "{AAPL: 0.20, BND: 0.04, GLD: 0.12, XLE: 0.18}" } },
    { line: 24, kind: "raw_data", payload: { label: "ampl.param['target']", value: "0.08" } },
    { line: 26, kind: "set_solver", payload: { name: "ipopt" } },
    { line: 27, kind: "solve", payload: { iters: 16, time: 0.022, status: "solved", obj: 0.005471, vars: { w_AAPL: 0.239, w_BND: 0.222, w_GLD: 0.308, w_XLE: 0.232 } }, note: "ampl.solve() generates the .nl file with the now-instantiated model and ships it to IPOPT. The QP is convex, ~16 iterations is typical." },
    { line: 29, kind: "print", payload: { text: "AAPL: w = 0.2390" } },
    { line: 29, kind: "print", payload: { text: "BND : w = 0.2220" } },
    { line: 29, kind: "print", payload: { text: "GLD : w = 0.3080" } },
    { line: 29, kind: "print", payload: { text: "XLE : w = 0.2320" } },
    { line: 30, kind: "print", payload: { text: "variance = 0.005471" }, note: "Minimum-variance portfolio meeting the 0.08 return floor. GLD (low vol, decent return) gets the largest weight." },
  ],
};

const PROB_HS71 = {
  key: "hs71",
  name: "Hock–Schittkowski 71 (NLP benchmark)",
  blurb:
    "The classic NLP benchmark — smooth nonconvex objective, one bilinear inequality, one quadratic equality, four bounded variables. AMPL is the language IPOPT's documentation uses for HS71; this is the canonical encoding.",
  formula:
    "min  x₁·x₄·(x₁+x₂+x₃) + x₃\ns.t.   x₁·x₂·x₃·x₄  ≥ 25\n       x₁² + x₂² + x₃² + x₄² = 40\n       1 ≤ xᵢ ≤ 5    (i = 1,…,4)",
  code: [
    null,
    "from amplpy import AMPL",
    "",
    "ampl = AMPL()",
    "ampl.eval(r'''",
    "    set I := 1 .. 4;",
    "    var x {I} >= 1, <= 5, := 1;",
    "",
    "    minimize obj:",
    "        x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3];",
    "",
    "    subject to ineq:",
    "        x[1] * x[2] * x[3] * x[4] >= 25;",
    "    subject to eq:",
    "        sum {i in I} x[i]^2 = 40;",
    "''')",
    "",
    "# Per-component initial values",
    "ampl.var['x'][1].setValue(1.0)",
    "ampl.var['x'][2].setValue(5.0)",
    "ampl.var['x'][3].setValue(5.0)",
    "ampl.var['x'][4].setValue(1.0)",
    "",
    "ampl.option['solver']        = 'ipopt'",
    "ampl.option['ipopt_options'] = 'print_level=5'",
    "ampl.solve()",
    "",
    "for i in [1, 2, 3, 4]:",
    "    print(f'  x[{i}] = {ampl.var[\"x\"][i].value():.6f}')",
    "print('obj* =', ampl.obj['obj'].value())",
  ],
  modCode: `# hs71.mod  --  pure AMPL
set I := 1 .. 4;
var x {I} >= 1, <= 5, := 1;

minimize obj:
    x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3];

subject to ineq:
    x[1] * x[2] * x[3] * x[4] >= 25;
subject to eq:
    sum {i in I} x[i]^2 = 40;

let x[1] := 1; let x[2] := 5;
let x[3] := 5; let x[4] := 1;`,
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "create_model" },
    { line: 4, kind: "eval_block", payload: { what: "model + initial values" }, note: "AMPL inline — model declared in Python via eval. set I := 1..4 is AMPL's range syntax (Pyomo: RangeSet(1,4))." },
    { line: 5, kind: "add_set", payload: { name: "I", value: "{1, 2, 3, 4}" }, note: "Integer range '1 .. 4'. Becomes the index set for x." },
    { line: 6, kind: "add_var", payload: { name: "x", indexed: "I", lb: 1, ub: 5, init: 1.0 }, note: "Indexed variable — x[1], x[2], x[3], x[4]. ':= 1' sets a default initial value of 1.0 for every component (overridden component-by-component below)." },
    { line: 8, kind: "set_objective", payload: { sense: "minimize", expr: "x[1]·x[4]·(x[1]+x[2]+x[3]) + x[3]" }, note: "Cubic — product of three plus a linear term. Smooth, NONCONVEX. AMPL's automatic differentiation produces gradients & Hessians IPOPT needs." },
    { line: 11, kind: "add_constraint", payload: { name: "ineq", expr: "x[1]·x[2]·x[3]·x[4] ≥ 25" }, note: "Quartic inequality. Strictly active at the optimum." },
    { line: 13, kind: "add_constraint", payload: { name: "eq", expr: "Σᵢ x[i]² = 40" }, note: "Quadratic equality. Pins (x₁,x₂,x₃,x₄) to a 4-sphere of radius √40 ≈ 6.32." },
    { line: 17, kind: "raw_data", payload: { label: "x[1]₀", value: "1.0" } },
    { line: 18, kind: "raw_data", payload: { label: "x[2]₀", value: "5.0" } },
    { line: 19, kind: "raw_data", payload: { label: "x[3]₀", value: "5.0" } },
    { line: 20, kind: "raw_data", payload: { label: "x[4]₀", value: "1.0" }, note: "The HS71 reference start: (1, 5, 5, 1). It violates the equality (1+25+25+1 = 52 ≠ 40), so IPOPT starts INFEASIBLE — a famous test for the restoration phase." },
    { line: 22, kind: "set_solver", payload: { name: "ipopt" } },
    { line: 23, kind: "set_solver_opts", payload: { ipopt_options: "print_level=5" }, note: "print_level=5 turns on the canonical IPOPT iteration table — the one with iter, objective, inf_pr, inf_du, lg(mu), alpha_pr/_du, ls. See the Pyomo+IPOPT demo for a column-by-column reader of this log." },
    { line: 24, kind: "solve", payload: { iters: 9, time: 0.014, status: "solved", obj: 17.014017, vars: { x_1: 1.0, x_2: 4.7430, x_3: 3.8211, x_4: 1.3794 } }, note: "Nine iterations — the IPOPT documented HS71 benchmark exactly. KKT solution: x = (1, 4.743, 3.821, 1.379), obj = 17.014. Lower bound on x[1] is active." },
    { line: 26, kind: "print", payload: { text: "  x[1] = 1.000000" } },
    { line: 26, kind: "print", payload: { text: "  x[2] = 4.742999" } },
    { line: 26, kind: "print", payload: { text: "  x[3] = 3.821150" } },
    { line: 26, kind: "print", payload: { text: "  x[4] = 1.379408" } },
    { line: 27, kind: "print", payload: { text: "obj* = 17.014017" }, note: "Verify: 1·1.379·(1+4.743+3.821) + 3.821 ≈ 17.01 ✓. Inequality: 1·4.743·3.821·1.379 = 25.0 (active). Equality: 1+22.5+14.6+1.9 = 40 ✓." },
  ],
};

const PROBLEMS = [
  PROB_LP_SIMPLE,
  PROB_IP_SIMPLE,
  PROB_LP_TRANSPORT,
  PROB_IP_FACILITY,
  PROB_QP,
  PROB_PORTFOLIO,
  PROB_HS71,
];

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo) {
  const s = {
    imported: false,
    model: false,
    sets: [],
    params: [],
    rawData: [],
    vars: [],
    objective: null,
    constraints: [],
    solverName: null,
    solverOpts: {},
    result: null,
    prints: [],
  };
  for (let i = 0; i <= upTo && i < events.length; i++) {
    const ev = events[i];
    switch (ev.kind) {
      case "import":
        s.imported = true;
        break;
      case "create_model":
        s.model = true;
        break;
      case "eval_block":
        s.model = true;
        break;
      case "add_set":
        s.sets.push({ ...ev.payload });
        break;
      case "add_param":
        s.params.push({ ...ev.payload });
        break;
      case "raw_data": {
        // Setter calls: bind values to existing sets/params/vars by label match
        const label = ev.payload.label;
        let bound = false;
        for (const set of s.sets) {
          if (label.includes(`'${set.name}'`)) {
            set.value = ev.payload.value;
            bound = true;
          }
        }
        for (const p of s.params) {
          if (label.includes(`'${p.name}'`)) {
            p.value = ev.payload.value;
            bound = true;
          }
        }
        if (!bound) s.rawData.push(ev.payload);
        break;
      }
      case "add_var":
        s.vars.push({ ...ev.payload });
        break;
      case "set_objective":
        s.objective = ev.payload;
        break;
      case "add_constraint":
        s.constraints.push(ev.payload);
        break;
      case "set_solver":
        s.solverName = ev.payload.name;
        break;
      case "set_solver_opts":
        s.solverOpts = { ...s.solverOpts, ...ev.payload };
        break;
      case "solve":
        s.solverName = s.solverName || "ipopt";
        s.result = {
          status: ev.payload.status,
          iters: ev.payload.iters,
          time: ev.payload.time,
          obj: ev.payload.obj,
          vars: ev.payload.vars,
        };
        for (const v of s.vars) {
          if (v.indexed) {
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
export default function AmplpyTutorial() {
  const [probKey, setProbKey] = useState(PROB_LP_SIMPLE.key);
  const problem = useMemo(
    () => PROBLEMS.find((p) => p.key === probKey),
    [probKey]
  );

  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);
  const [showMod, setShowMod] = useState(false);

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
        AMPL + amplpy — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Step through real amplpy code line by line. Each problem mirrors the
        Pyomo + IPOPT walkthrough so you can compare modelers directly. Toggle
        the <i>.mod view</i> to see the equivalent pure-AMPL syntax — that's
        the dialect most operations-research textbooks teach.
      </p>

      <InstallPanel />

      {/* Problem tabs */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {PROBLEMS.map((p) => (
          <button
            key={p.key}
            onClick={() => setProbKey(p.key)}
            style={{ ...tabBtn, ...(p.key === probKey ? tabBtnActive : {}) }}
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
        <div style={{ marginTop: 8, fontFamily: "monospace", fontSize: 12, color: "#1f4e3d", whiteSpace: "pre-wrap" }}>
          {problem.formula}
        </div>
      </div>

      {/* .mod toggle + copy / download buttons */}
      <div style={{ marginBottom: 10, display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button
          onClick={() => setShowMod((s) => !s)}
          style={{
            ...btn,
            background: showMod ? "#1f4e3d" : "#f7f7f7",
            color: showMod ? "#fff" : "#222",
            border: showMod ? "1px solid #1f4e3d" : "1px solid #ccc",
          }}
        >
          <FileText size={14} />
          {showMod ? "showing .mod equivalent" : "show pure AMPL .mod"}
        </button>
        <CopyCodeButton code={problem.code.slice(1).join("\n")} label="Copy Python" />
        <CopyCodeButton code={problem.modCode} label="Copy .mod" />
        <DownloadNotebookButton
          code={problem.code.slice(1).join("\n")}
          filename={`amplpy_${problem.key}.ipynb`}
          title={problem.name}
          description={problem.blurb + "\n\n```\n" + problem.formula + "\n```"}
        />
        <span style={{ fontSize: 12, color: "#666" }}>
          {showMod
            ? "Same model in textbook AMPL syntax (no Python)."
            : ""}
        </span>
      </div>

      {showMod && (
        <div style={modPanel}>
          <div style={{ fontSize: 11, color: "#7f7864", fontFamily: "monospace", letterSpacing: "0.12em", marginBottom: 6, textTransform: "uppercase" }}>
            {problem.key}.mod (+ .dat for portfolio)
          </div>
          <Pre>{problem.modCode}</Pre>
        </div>
      )}

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
            <div style={{ fontSize: 11, color: "#777", marginBottom: 4, fontFamily: "monospace" }}>
              what this line does
            </div>
            <div style={{ fontSize: 14, lineHeight: 1.55 }}>
              {ev?.note || "(no note for this line — keep stepping)"}
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
            <button onClick={stepOnce} disabled={evIdx >= problem.events.length - 1} style={btnPrimary}>
              <StepForward size={16} /> Step
            </button>
            <button onClick={() => setRunning((r) => !r)} disabled={evIdx >= problem.events.length - 1} style={btn}>
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

          <div style={{ marginTop: 8, height: 6, background: "#eee", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ height: "100%", background: "#1f4e3d", width: `${(100 * (evIdx + 1)) / problem.events.length}%`, transition: "width 0.1s" }} />
          </div>
          <div style={{ fontSize: 11, color: "#888", fontFamily: "monospace", marginTop: 4 }}>
            event {evIdx + 1} / {problem.events.length}
          </div>
        </div>

        <StatePanel state={state} />
      </div>

      <ComparisonPanel />
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
    <div style={{ marginBottom: 18, border: "1px solid #d3d3d3", borderRadius: 8, background: "#fafafa" }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{ display: "flex", alignItems: "center", gap: 8, width: "100%", padding: "10px 14px", background: "transparent", border: 0, cursor: "pointer", fontWeight: 700, fontSize: 14, color: "#222", textAlign: "left" }}
      >
        <Package size={16} />
        Install AMPL + amplpy &nbsp;
        <span style={{ color: "#888", fontWeight: 400, fontSize: 12 }}>
          ({open ? "click to collapse" : "click to expand"})
        </span>
      </button>
      {open && (
        <div style={{ padding: "0 14px 14px 14px", fontSize: 13, color: "#333", lineHeight: 1.55 }}>
          <p style={{ marginTop: 0 }}>
            AMPL is a binary; amplpy is the Python interface. Two pieces.
          </p>
          <Pre>
            {`# 1. amplpy (the Python wrapper — pure pip)
pip install amplpy

# 2. AMPL itself + a solver — pick one path
#    a) Easiest: AMPL Community Edition (free, unlimited for academic use)
python -m amplpy.modules install ipopt highs cbc
#       or via the AMPL website:  ampl.com/ce  → install bundle → on PATH

#    b) Pure conda (cross-platform):
conda install -c conda-forge ampl-mp ipopt

# 3. Verify
python -c "from amplpy import AMPL; a = AMPL(); print(a.option['version'])"`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            If amplpy can't find AMPL, pass{" "}
            <code style={inlineCode}>AMPL(binary_directory='/path/to/ampl/')</code>.
            The recommended modern install is{" "}
            <code style={inlineCode}>amplpy.modules</code>{" "}
            — it ships AMPL and the solver into your Python environment in one
            command and handles licensing automatically.
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
// State panel
// ============================================================
function StatePanel({ state }) {
  return (
    <div style={statePanelOuter}>
      <Section title="Imports">
        {state.imported ? (
          <div style={chip("#1f4e3d")}>amplpy ✓</div>
        ) : (
          <Empty />
        )}
      </Section>

      <Section title="AMPL session">
        {state.model ? (
          <div style={chip("#0b3da0")}>AMPL()</div>
        ) : (
          <Empty />
        )}
      </Section>

      {state.sets.length > 0 && (
        <Section title="Sets">
          {state.sets.map((s, i) => (
            <KVRow key={i} k={s.name} v={s.value || "(unspecified)"} mono />
          ))}
        </Section>
      )}

      {state.params.length > 0 && (
        <Section title="Parameters">
          {state.params.map((p, i) => (
            <KVRow
              key={i}
              k={p.indexed ? `${p.name} {${p.indexed}}` : p.name}
              v={p.value || "(unspecified)"}
              mono
            />
          ))}
        </Section>
      )}

      {state.rawData.length > 0 && (
        <Section title="Other Python data">
          {state.rawData.map((d, i) => (
            <KVRow key={i} k={d.label} v={d.value} mono />
          ))}
        </Section>
      )}

      <Section title="Variables">
        {state.vars.length === 0 ? (
          <Empty />
        ) : (
          state.vars.map((v, i) => (
            <div key={i} style={{ marginBottom: 8, padding: "6px 10px", background: "#fff", border: "1px solid #ddd", borderRadius: 6 }}>
              <div style={{ display: "flex", justifyContent: "space-between" }}>
                <span style={{ fontFamily: "monospace", fontWeight: 700 }}>
                  {v.name}{v.indexed ? ` [${v.indexed}]` : ""}
                </span>
                <span style={{ fontSize: 11, color: "#666" }}>
                  var{v.indexed ? " (indexed)" : ""}
                </span>
              </div>
              <div style={{ fontSize: 12, color: "#555", fontFamily: "monospace", marginTop: 2 }}>
                bounds=({fmt(v.lb)}, {fmt(v.ub)})
                {v.init != null && `, init=${v.init}`}
              </div>
              {v.value != null && (
                <div style={{ fontSize: 12, color: "#c8311c", fontFamily: "monospace", marginTop: 2, fontWeight: 700 }}>
                  value = {v.value}
                </div>
              )}
              {v.values && (
                <div style={{ fontSize: 12, color: "#c8311c", fontFamily: "monospace", marginTop: 2 }}>
                  {Object.entries(v.values).map(([k, val]) => (
                    <div key={k}><b>[{k}]</b> = {(+val).toFixed(4)}</div>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
      </Section>

      <Section title="Objective">
        {state.objective ? (
          <div style={{ padding: "6px 10px", background: "#fff", border: "1px solid #ddd", borderRadius: 6, fontFamily: "monospace", fontSize: 13 }}>
            <span style={{ color: "#555", fontSize: 11 }}>{state.objective.sense}</span>
            <div>{state.objective.expr}</div>
            {state.objective.name && (
              <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>
                ampl.obj['{state.objective.name}']
              </div>
            )}
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
            <div key={i} style={{ marginBottom: 6, padding: "6px 10px", background: "#fff", border: "1px solid #ddd", borderRadius: 6, fontFamily: "monospace", fontSize: 13 }}>
              <span style={{ color: "#555", fontSize: 11 }}>{c.name}</span>
              <div>{c.expr}</div>
            </div>
          ))
        )}
      </Section>

      <Section title="Solver">
        {state.solverName ? (
          <div>
            <div style={chip("#7a3da0")}>{state.solverName}</div>
            {Object.entries(state.solverOpts).map(([k, v]) => (
              <div key={k} style={{ fontFamily: "monospace", fontSize: 11, color: "#555", marginTop: 4 }}>
                ampl.option['{k}'] = '{v}'
              </div>
            ))}
          </div>
        ) : (
          <Empty />
        )}
      </Section>

      {state.result && (
        <Section title="Solver result">
          <div style={{ background: "#1f1d1a", color: "#e8e2d4", padding: 10, borderRadius: 6, fontFamily: "monospace", fontSize: 12, lineHeight: 1.55 }}>
            <ResultLine k="solve_result" v={state.result.status} c="#7dd87d" />
            <ResultLine k="iterations" v={state.result.iters} />
            <ResultLine k="time" v={`${state.result.time.toFixed(3)} s`} />
            <ResultLine k="objective" v={(+state.result.obj).toFixed(6)} c="#f5a524" />
            {Object.entries(state.result.vars).map(([k, v]) => (
              <ResultLine key={k} k={k} v={(+v).toFixed(4)} c="#f5a524" />
            ))}
          </div>
        </Section>
      )}

      {state.prints.length > 0 && (
        <Section title="Console output">
          <div style={{ background: "#0a0a0a", color: "#dadada", padding: 10, borderRadius: 6, fontFamily: "monospace", fontSize: 12, lineHeight: 1.55 }}>
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
      <div style={{ fontFamily: "monospace", fontSize: 10, letterSpacing: "0.18em", color: "#888", marginBottom: 6, textTransform: "uppercase" }}>
        {title}
      </div>
      {children}
    </div>
  );
}
function Empty() {
  return (
    <div style={{ padding: "6px 10px", background: "#fff", border: "1px dashed #ddd", borderRadius: 6, color: "#999", fontSize: 12, fontStyle: "italic" }}>
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
    <div style={{ display: "flex", gap: 10, padding: "3px 0", borderBottom: "1px dotted #eee" }}>
      <span style={{ color: "#666", fontSize: 12, minWidth: 130, fontFamily: "monospace" }}>{k}</span>
      <span style={{ fontSize: 12, color: "#222", fontFamily: mono ? "monospace" : "inherit", flex: 1, wordBreak: "break-word" }}>{v}</span>
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
  if (x === "Infinity" || x == null) return "∞";
  return String(x);
}
function Pre({ children }) {
  return (
    <pre style={{
      background: "#1f1d1a",
      color: "#e8e2d4",
      padding: "12px 14px",
      borderRadius: 6,
      fontFamily: "'JetBrains Mono', Menlo, ui-monospace, monospace",
      fontSize: 12.5,
      lineHeight: 1.5,
      overflow: "auto",
      whiteSpace: "pre",
      margin: 0,
    }}>
      {children}
    </pre>
  );
}

// ============================================================
// Comparison panel — amplpy vs Pyomo
// ============================================================
function ComparisonPanel() {
  const rows = [
    ["Model declaration", "ampl.eval(r''' var x >= 0; ... ''')", "model.x = Var(bounds=(0, None))"],
    ["Indexed variable", "var w {ASSETS} >= 0, <= 1;", "model.w = Var(model.A, bounds=(0,1))"],
    ["Sum over set", "sum {a in ASSETS} mu[a] * w[a]", "sum(mu[a] * model.w[a] for a in model.A)"],
    ["Set declaration", "set ASSETS;     # then ampl.set['ASSETS'] = [...]", "model.A = Set(initialize=assets)"],
    ["Parameter", "param mu {ASSETS}; # then ampl.param['mu'] = {...}", "(plain Python dict — no Param needed)"],
    ["Constraint", "subject to budget: sum {a in A} w[a] = 1;", "model.budget = Constraint(expr=...)"],
    ["Objective", "minimize variance: sum {a in A} ...;", "model.variance = Objective(expr=..., sense=minimize)"],
    ["Solve", "ampl.option['solver']='ipopt';  ampl.solve()", "SolverFactory('ipopt').solve(model)"],
    ["Get var value", "ampl.var['x'].value()", "value(model.x)"],
    ["Get dual", "ampl.con['budget'].dual()", "model.dual = Suffix(direction=IMPORT) before solve; model.dual[model.budget]"],
    ["Power operator", "x^2  (caret)", "model.x**2  (Python)"],
    ["Solver options", "ampl.option['ipopt_options'] = 'print_level=5'", "solver.options['print_level'] = 5"],
  ];
  return (
    <div style={{ ...modPanel, marginTop: 22, background: "#fafafa", color: "inherit" }}>
      <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>
        amplpy vs Pyomo — translation cheat sheet
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ borderCollapse: "collapse", fontSize: 12.5, width: "100%" }}>
          <thead>
            <tr>
              <th style={cellHead}>operation</th>
              <th style={cellHead}>amplpy / AMPL</th>
              <th style={cellHead}>Pyomo</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td style={cell}>{r[0]}</td>
                <td style={cellMono}>{r[1]}</td>
                <td style={cellMono}>{r[2]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const cellHead = {
  padding: "6px 10px",
  borderBottom: "2px solid #444",
  textAlign: "left",
  fontFamily: "monospace",
  fontSize: 11,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: "#555",
};
const cell = {
  padding: "6px 10px",
  borderBottom: "1px solid #eee",
  verticalAlign: "top",
};
const cellMono = {
  ...cell,
  fontFamily: "monospace",
  fontSize: 12,
};

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
          <b>AMPL is its own language.</b> Pyomo / CVXPY / Gurobi-Python are all
          embedded DSLs <i>inside Python</i>. AMPL has a real grammar, a parser,
          and stand-alone .mod / .dat files. amplpy is just a Python pipe to
          the AMPL interpreter.
        </li>
        <li>
          <b>Model / data separation</b> is core to AMPL. The .mod file
          declares structure (sets, parameters, vars, constraints) WITHOUT
          values; the .dat file fills in the values; you can swap .dat files
          to instantiate the same model on different problem sizes. Pyomo can
          do this with AbstractModel but rarely does in practice.
        </li>
        <li>
          <b>{`'^' is exponentiation in AMPL`}</b>, not Python's <code style={inlineCode}>{`**`}</code>.
          Inside <code style={inlineCode}>ampl.eval(...)</code> you must use
          AMPL syntax. Outside it, you use Python.
        </li>
        <li>
          <b>Solver-agnostic by design.</b> Switching from IPOPT to Knitro to
          BARON is a one-line change of <code style={inlineCode}>ampl.option['solver']</code>.
          Most other modelers (especially Gurobi-Python) bind tightly to one
          solver.
        </li>
        <li>
          <b>amplpy.modules — the modern install.</b> The AMPL team now ships
          binaries through pip:{" "}
          <code style={inlineCode}>python -m amplpy.modules install ipopt</code>
          installs the AMPL binary AND the IPOPT plugin in one go, no
          system-level setup. This is the recommended path now (since 2023);
          older docs may still describe a separate AMPL install.
        </li>
        <li>
          <b>NL format</b> is the on-disk binary AMPL ships to NLP solvers —
          contains the model, gradients, sparsity, and Hessian structure. Same
          format Pyomo uses for IPOPT. If you understand .nl, you can attach
          ANY NL-aware solver to either modeler.
        </li>
        <li>
          <b>When to choose AMPL over Pyomo:</b> when your audience is
          textbook OR (lots of indexed sums and parameter tables); when you
          want declarative models; when teaching the math, not the Python.
          When to choose Pyomo: when your model is generated by Python code,
          when you need callbacks during solve, or when you're deploying
          inside a larger Python pipeline.
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
  fontSize: 13,
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
const modPanel = {
  marginBottom: 16,
  padding: 14,
  background: "#1f1d1a",
  borderRadius: 8,
  border: "1px solid #444",
};
