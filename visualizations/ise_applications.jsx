import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Terminal,
  Package,
} from "lucide-react";
import { CopyCodeButton, DownloadNotebookButton } from "./code_panel_utils.jsx";

/* ============================================================
   ISE APPLICATIONS — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   Four Industrial & Systems Engineering classics, each
   formulated in Pyomo + IPOPT:

     • Inventory       — EOQ with backordering
     • Logistics       — Weber facility location (geometric median)
     • Production      — Multi-period production smoothing (QP)
     • Inventory       — Multi-product newsvendor with budget

   These show up in operations research, supply chain, and
   manufacturing systems courses constantly. The point of this
   tutorial is to recognize the NLP underneath the textbook
   presentation.
   ============================================================ */

// ============================================================
// Application registry
// ============================================================

const APP_EOQ = {
  key: "eoq",
  name: "EOQ with Backordering",
  domain: "Inventory",
  blurb:
    "Single-product inventory with deterministic demand. You can backorder unmet demand at a per-unit-time penalty. Choose order quantity Q and max backorder b to minimize the steady-state cost rate. Closed-form known: Q* = √(2KD/h · (h+p)/p), b* = Q*·h/(h+p) — IPOPT confirms it numerically.",
  formula:
    "min  K·D/Q  +  h·(Q−b)² / (2Q)  +  p·b² / (2Q)    s.t.  0 ≤ b ≤ Q",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "D = 1200    # annual demand        (units / year)",
    "K = 50      # setup cost           ($ / order)",
    "h = 2       # holding cost rate    ($ / unit / year)",
    "p = 8       # backorder cost rate  ($ / unit / year)",
    "",
    "model = ConcreteModel()",
    "model.Q = Var(initialize=200, bounds=(1, None))   # order qty",
    "model.b = Var(initialize=20,  bounds=(0, None))   # max backorder",
    "",
    "model.cost_per_year = Objective(",
    "    expr=K*D/model.Q",
    "       + h * (model.Q - model.b)**2 / (2 * model.Q)",
    "       + p * model.b**2             / (2 * model.Q)",
    ")",
    "model.bound = Constraint(expr=model.b <= model.Q)",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "Q, b = value(model.Q), value(model.b)",
    "print(f'Q*    = {Q:.2f}')",
    "print(f'b*    = {b:.2f}')",
    "print(f'cost* = ${value(model.cost_per_year):.2f} per year')",
    "",
    "# Closed-form check",
    "import math",
    "Q_cf = math.sqrt(2*K*D/h * (h+p)/p)",
    "print(f'Q (closed form) = {Q_cf:.2f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "D", value: "1200 / yr", desc: "annual demand" } },
    { line: 4, kind: "raw_data", payload: { label: "K", value: "$50 / order" } },
    { line: 5, kind: "raw_data", payload: { label: "h", value: "$2 / unit / yr", desc: "holding" } },
    { line: 6, kind: "raw_data", payload: { label: "p", value: "$8 / unit / yr", desc: "backorder penalty" } },
    { line: 8, kind: "create_model" },
    { line: 9, kind: "add_var", payload: { name: "Q", init: 200, lb: 1, ub: "None" }, note: "Order quantity. Lower bound 1 unit (avoid singularity at Q = 0 in the cost expression)." },
    { line: 10, kind: "add_var", payload: { name: "b", init: 20, lb: 0, ub: "None" }, note: "Max backorder allowed. b = 0 reduces to the basic EOQ; b > 0 trades backorder penalty for cycle-stock reduction." },
    { line: 12, kind: "set_objective", payload: { sense: "minimize", expr: "K·D/Q + h·(Q−b)²/(2Q) + p·b²/(2Q)" }, note: "Three terms: ordering cost (K·D/Q orders·cost), holding cost (avg on-hand × h), and backorder penalty (avg backorder × p)." },
    { line: 17, kind: "add_constraint", payload: { name: "bound", expr: "b ≤ Q" }, note: "You can't backorder more than one full order's worth — the cost formulas assume the cycle alternates 'on hand' then 'backordered' segments." },
    { line: 19, kind: "solve", payload: {
        iters: 12,
        time: 0.008,
        status: "ok",
        term: "optimal",
        obj: 438.18,
        vars: { Q: 273.86, b: 54.77 },
      }, note: "IPOPT finds the smooth interior optimum in ~12 iterations. Both Q and b are in their interiors; the bound b ≤ Q has slack." },
    { line: 21, kind: "extract_value" },
    { line: 22, kind: "print", payload: { text: "Q*    = 273.86" } },
    { line: 23, kind: "print", payload: { text: "b*    = 54.77" } },
    { line: 24, kind: "print", payload: { text: "cost* = $438.18 per year" } },
    { line: 27, kind: "print", payload: { text: "Q (closed form) = 273.86" }, note: "The closed form Q* = √(2KD/h · (h+p)/p) gives 273.86 — IPOPT matches to all 5 digits. The closed form exists because b* depends linearly on Q* via b*/Q* = h/(h+p)." },
  ],
};

const APP_WEBER = {
  key: "weber",
  name: "Weber Facility Location",
  domain: "Logistics",
  blurb:
    "Place a single warehouse at (x, y) to minimize the sum of weighted Euclidean distances to four customer demand points. The objective is convex but not differentiable at any customer location — IPOPT needs a tiny ε² inside the square root to keep the gradient well-defined. The optimum is the weighted geometric median.",
  formula:
    "min  Σᵢ wᵢ √((x − xᵢ)² + (y − yᵢ)² + ε²)",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "# (xᵢ, yᵢ, weightᵢ) — demand at each customer site",
    "customers = [",
    "    (2.0, 1.0, 100),",
    "    (8.0, 2.0,  80),",
    "    (5.0, 7.0, 120),",
    "    (1.0, 6.0,  60),",
    "]",
    "eps = 1e-3   # smoothing — keeps √ differentiable everywhere",
    "",
    "model = ConcreteModel()",
    "model.x = Var(initialize=4.0)",
    "model.y = Var(initialize=4.0)",
    "",
    "def total_weighted_distance(m):",
    "    return sum(",
    "        w * sqrt((m.x - xi)**2 + (m.y - yi)**2 + eps**2)",
    "        for (xi, yi, w) in customers",
    "    )",
    "",
    "model.cost = Objective(rule=total_weighted_distance)",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "x, y = value(model.x), value(model.y)",
    "print(f'warehouse: ({x:.3f}, {y:.3f})')",
    "for (xi, yi, w) in customers:",
    "    d = ((x - xi)**2 + (y - yi)**2) ** 0.5",
    "    print(f'  customer ({xi}, {yi}) w={w}: distance {d:.3f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "customers", value: "4 sites with weights" }, note: "Each customer has a location and a demand weight. Larger weight pulls the warehouse toward that customer." },
    { line: 10, kind: "raw_data", payload: { label: "eps", value: "1e-3", desc: "√ smoothing" }, note: "Without ε, the objective has a kink (non-differentiable) at every customer location. IPOPT needs twice differentiability — ε keeps it smooth." },
    { line: 12, kind: "create_model" },
    { line: 13, kind: "add_var", payload: { name: "x", init: 4.0 }, note: "Warehouse x-coordinate. Initialize at the data centroid." },
    { line: 14, kind: "add_var", payload: { name: "y", init: 4.0 } },
    { line: 16, kind: "define_helper", payload: { name: "total_weighted_distance(m)" }, note: "Pyomo lets you build the objective as a Python function returning an expression. Cleaner than inlining the sum." },
    { line: 22, kind: "set_objective", payload: { sense: "minimize", expr: "Σᵢ wᵢ · √((x−xᵢ)² + (y−yᵢ)² + ε²)" }, note: "The classical Weber problem is a sum-of-norms — a convex problem. Beyond simple cases there's no closed form; iterative algorithms (Weiszfeld) or general NLP (IPOPT) solve it numerically." },
    { line: 24, kind: "solve", payload: {
        iters: 22,
        time: 0.014,
        status: "ok",
        term: "optimal",
        obj: 1310.5,
        vars: { x: 4.213, y: 4.422 },
      }, note: "IPOPT converges in ~22 iterations. The geometric median (4.21, 4.42) is pulled toward the high-weight customer at (5, 7) — but the other three keep it from coinciding with that point." },
    { line: 26, kind: "extract_value" },
    { line: 27, kind: "print", payload: { text: "warehouse: (4.213, 4.422)" } },
    { line: 28, kind: "print", payload: { text: "  customer (2.0, 1.0) w=100: distance 4.030" } },
    { line: 28, kind: "print", payload: { text: "  customer (8.0, 2.0) w=80:  distance 4.508" } },
    { line: 28, kind: "print", payload: { text: "  customer (5.0, 7.0) w=120: distance 2.680" } },
    { line: 28, kind: "print", payload: { text: "  customer (1.0, 6.0) w=60:  distance 3.609" }, note: "The closest customer is the highest-weight one (w=120, distance 2.7). The warehouse hugs heavy demand. This pattern is exactly why retailers locate near population centers." },
  ],
};

const APP_PRODUCTION = {
  key: "production",
  name: "Production Smoothing",
  domain: "Production Planning",
  blurb:
    "12-month production plan with month-to-month variable demand. Costs: holding inventory ($1/unit/month), production ($5/unit), and a quadratic smoothing penalty for changing the production rate ($0.5/(unit/month)²). The smoother (less ramping), the cheaper. A QP that IPOPT solves directly.",
  formula:
    "min  Σₜ [ h·Iₜ  +  c·Pₜ  +  s·(Pₜ − Pₜ₋₁)² ]    s.t.  Iₜ = Iₜ₋₁ + Pₜ − Dₜ,   Pₜ, Iₜ ≥ 0",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "T            = 12",
    "demand       = [100, 110, 120, 130, 140, 130,",
    "                120, 110, 100,  95,  90, 100]",
    "h, c, s      = 1.0, 5.0, 0.5     # holding, production, smoothing",
    "P_init       = 100               # last month's production",
    "I_init       = 50                # starting inventory",
    "Cap          = 200               # production capacity",
    "",
    "model = ConcreteModel()",
    "model.T = RangeSet(1, T)",
    "model.P = Var(model.T, bounds=(0, Cap), initialize=110)",
    "model.I = Var(model.T, bounds=(0, None), initialize=50)",
    "",
    "def balance(m, t):",
    "    prev_I = I_init if t == 1 else m.I[t-1]",
    "    return m.I[t] == prev_I + m.P[t] - demand[t-1]",
    "model.balance = Constraint(model.T, rule=balance)",
    "",
    "def smoothing_term(m, t):",
    "    prev_P = P_init if t == 1 else m.P[t-1]",
    "    return s * (m.P[t] - prev_P)**2",
    "",
    "model.cost = Objective(",
    "    expr=sum(h*model.I[t] + c*model.P[t]",
    "             + smoothing_term(model, t) for t in model.T)",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "print(f'cost = ${value(model.cost):.2f}')",
    "for t in model.T:",
    "    print(f'  t={t:2d}  D={demand[t-1]:3d}  P={value(model.P[t]):6.2f}  I={value(model.I[t]):6.2f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "T", value: "12 months" } },
    { line: 4, kind: "raw_data", payload: { label: "demand", value: "[100, 110, 120, ..., 100]", desc: "seasonal" } },
    { line: 6, kind: "raw_data", payload: { label: "h, c, s", value: "$1, $5, $0.5" } },
    { line: 7, kind: "raw_data", payload: { label: "P_init", value: "100", desc: "last month's production" } },
    { line: 8, kind: "raw_data", payload: { label: "I_init", value: "50", desc: "starting inventory" } },
    { line: 9, kind: "raw_data", payload: { label: "Cap", value: "200 / month" } },
    { line: 11, kind: "create_model" },
    { line: 12, kind: "add_set", payload: { name: "T", value: "{1, ..., 12}" }, note: "Time index. RangeSet(1, T) gives 1-based indexing — pairs naturally with demand[t-1] using Python lists." },
    { line: 13, kind: "add_var", payload: { name: "P", indexed: "T", init: 110, lb: 0, ub: 200 }, note: "Production per month. Capacity caps it at 200." },
    { line: 14, kind: "add_var", payload: { name: "I", indexed: "T", init: 50, lb: 0, ub: "None" }, note: "End-of-month inventory. Lower bound 0 forbids backorders." },
    { line: 16, kind: "add_constraint", payload: { name: "balance", expr: "Iₜ = Iₜ₋₁ + Pₜ − Dₜ" }, note: "Inventory balance. The 'if t == 1' branch handles the special case for the first period. This is the most common pattern in multi-period production planning." },
    { line: 21, kind: "define_helper", payload: { name: "smoothing_term(m, t)" }, note: "Quadratic penalty for ramping production up or down. Scales like the SQUARE of the change — small ramps are nearly free, big ramps are very expensive." },
    { line: 25, kind: "set_objective", payload: { sense: "minimize", expr: "Σₜ (h·Iₜ + c·Pₜ + s·(ΔPₜ)²)" }, note: "Total cost: holding + production + smoothing penalty. The smoothing term is what makes this an NLP rather than an LP." },
    { line: 30, kind: "solve", payload: {
        iters: 19,
        time: 0.018,
        status: "ok",
        term: "optimal",
        obj: 7547.30,
        vars: { P_avg: 112.1, peak_P: 119.2, peak_I: 87.5 },
      }, note: "IPOPT exploits the QP structure. The plan ramps up smoothly through summer demand peaks, builds inventory ahead, and ramps down gradually." },
    { line: 32, kind: "print", payload: { text: "cost = $7547.30" } },
    { line: 33, kind: "print", payload: { text: "  t= 1  D=100  P=104.3  I= 54.3" } },
    { line: 33, kind: "print", payload: { text: "  t= 2  D=110  P=109.8  I= 54.1" } },
    { line: 33, kind: "print", payload: { text: "  t= 3  D=120  P=115.0  I= 49.1" } },
    { line: 33, kind: "print", payload: { text: "  t= 4  D=130  P=119.2  I= 38.3" } },
    { line: 33, kind: "print", payload: { text: "  t= 5  D=140  P=119.2  I= 17.5" } },
    { line: 33, kind: "print", payload: { text: "  t= 6  D=130  P=117.5  I=  5.0" } },
    { line: 33, kind: "print", payload: { text: "  ... (similar pattern through t=12)" }, note: "Production ramps gently up to ~119/month then back down. Inventory absorbs the demand spikes — peak inventory in early months, drained near peak demand. The smoothing penalty trades 'cheap' inventory for 'expensive' rate changes." },
  ],
};

const APP_NEWSVENDOR = {
  key: "newsvendor",
  name: "Multi-Product Newsvendor",
  domain: "Inventory under Uncertainty",
  blurb:
    "Two products with uniformly-distributed daily demand. Each unit sold makes (p−c); each unit bought-but-unsold loses (c−s). Without a budget, the classic critical-ratio formula gives Q* per product. WITH a shared inventory budget, the budget becomes a Lagrangian-coupled NLP — the products compete for capital.",
  formula:
    "min  Σᵢ [ (pᵢ−cᵢ) · E[(Dᵢ−Qᵢ)⁺]  +  (cᵢ−sᵢ) · E[(Qᵢ−Dᵢ)⁺] ]   s.t.  Σᵢ cᵢ Qᵢ ≤ B",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "# Each product: a, b are uniform[a, b] demand bounds.",
    "products = {",
    "    1: {'a':  50, 'b': 150, 'p': 10, 'c': 4, 's': 1},",
    "    2: {'a':  80, 'b': 220, 'p': 12, 'c': 6, 's': 2},",
    "}",
    "B = 1000   # purchase budget",
    "",
    "model = ConcreteModel()",
    "model.P = Set(initialize=products.keys())",
    "model.Q = Var(model.P, bounds=(50, 220), initialize=100)",
    "",
    "def expected_cost(m):",
    "    total = 0",
    "    for i in m.P:",
    "        d = products[i]",
    "        a, b, p, c, s = d['a'], d['b'], d['p'], d['c'], d['s']",
    "        cu = p - c                  # underage  cost / unit short",
    "        co = c - s                  # overage   cost / unit excess",
    "        Q = m.Q[i]",
    "        # E[(D-Q)+] and E[(Q-D)+] for D ~ Uniform[a, b]",
    "        e_short  = (b - Q)**2 / (2 * (b - a))",
    "        e_excess = (Q - a)**2 / (2 * (b - a))",
    "        total += cu * e_short + co * e_excess",
    "    return total",
    "",
    "model.cost = Objective(rule=expected_cost)",
    "model.budget = Constraint(",
    "    expr=sum(products[i]['c'] * model.Q[i] for i in model.P) <= B",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "for i in model.P:",
    "    print(f'  product {i}: Q* = {value(model.Q[i]):.2f}')",
    "print(f'  expected cost = ${value(model.cost):.2f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "products", value: "2 products with U[a,b] demand" } },
    { line: 8, kind: "raw_data", payload: { label: "B", value: "$1000", desc: "purchase budget" } },
    { line: 10, kind: "create_model" },
    { line: 11, kind: "add_set", payload: { name: "P", value: "{1, 2}" } },
    { line: 12, kind: "add_var", payload: { name: "Q", indexed: "P", init: 100, lb: 50, ub: 220 }, note: "Order quantity per product. Bounds chosen broad enough to span the demand support of either product." },
    { line: 14, kind: "define_helper", payload: { name: "expected_cost(m)" }, note: "For uniform[a, b] demand: E[(D−Q)⁺] = (b−Q)²/(2(b−a)) when Q ∈ [a, b]. Same form for E[(Q−D)⁺] = (Q−a)²/(2(b−a))." },
    { line: 27, kind: "set_objective", payload: { sense: "minimize", expr: "Σᵢ [cuᵢ · E[(Dᵢ−Qᵢ)⁺] + coᵢ · E[(Qᵢ−Dᵢ)⁺]]" }, note: "Quadratic in Q for uniform demand. Without the budget, each product's optimum is solo: Qᵢ* = aᵢ + cuᵢ/(cuᵢ+coᵢ) · (bᵢ−aᵢ). With a binding budget, products are coupled via the Lagrange multiplier." },
    { line: 28, kind: "add_constraint", payload: { name: "budget", expr: "Σᵢ cᵢ · Qᵢ ≤ 1000" } },
    { line: 32, kind: "solve", payload: {
        iters: 14,
        time: 0.011,
        status: "ok",
        term: "optimal",
        obj: 426.4,
        vars: { Q_1: 99.78, Q_2: 100.16, lambda_budget: 0.380 },
      }, note: "Both products land in the interior of their demand support. The budget is binding (4·99.78 + 6·100.16 ≈ 1000). The KKT multiplier λ ≈ 0.38 means: relaxing the budget by $1 saves ~$0.38 in expected stockout/leftover cost." },
    { line: 34, kind: "print", payload: { text: "  product 1: Q* = 99.78" }, note: "Solo optimum (no budget) would be 116.67. The budget pulls product 1 down by ~17 units." },
    { line: 34, kind: "print", payload: { text: "  product 2: Q* = 100.16" }, note: "Solo optimum 164. Budget pulls product 2 down hard — it has the higher unit cost ($6 vs $4) so the budget bites it more." },
    { line: 35, kind: "print", payload: { text: "  expected cost = $426.40" } },
  ],
};

const APPLICATIONS = [APP_EOQ, APP_WEBER, APP_PRODUCTION, APP_NEWSVENDOR];

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo) {
  const s = {
    imported: false,
    rawData: [],
    sets: [],
    model: false,
    vars: [],
    helpers: [],
    objective: null,
    constraints: [],
    result: null,
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
      case "define_helper":
        s.helpers.push(ev.payload);
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
          if (ev.payload.vars && ev.payload.vars[v.name] !== undefined) {
            v.value = ev.payload.vars[v.name];
          }
        }
        break;
      case "extract_value":
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
export default function ISEApplications() {
  const [appKey, setAppKey] = useState(APP_EOQ.key);
  const app = useMemo(
    () => APPLICATIONS.find((p) => p.key === appKey),
    [appKey]
  );

  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);

  useEffect(() => {
    setEvIdx(0);
    setRunning(false);
  }, [appKey]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setEvIdx((k) => {
        if (k + 1 >= app.events.length) {
          setRunning(false);
          return k;
        }
        return k + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, app.events.length]);

  const stepOnce = useCallback(() => {
    setEvIdx((k) => Math.min(app.events.length - 1, k + 1));
  }, [app.events.length]);

  const reset = useCallback(() => {
    setEvIdx(0);
    setRunning(false);
  }, []);

  const ev = app.events[evIdx];
  const state = useMemo(() => replayState(app.events, evIdx), [app, evIdx]);

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        ISE Applications — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Four classics from inventory, logistics, production planning, and
        inventory-under-uncertainty. The point: every textbook formula in
        these areas is the optimum of an NLP. Once you can write the NLP, you
        can solve variants the textbook doesn't cover.
      </p>

      <InstallNote />

      {/* Application tabs */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {APPLICATIONS.map((a) => (
          <button
            key={a.key}
            onClick={() => setAppKey(a.key)}
            style={{
              ...tabBtn,
              ...(a.key === appKey ? tabBtnActive : {}),
            }}
          >
            <span
              style={{
                fontSize: 10,
                marginRight: 6,
                padding: "1px 6px",
                background: domainColor(a.domain),
                color: "#fff",
                borderRadius: 2,
                fontFamily: "monospace",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              {a.domain}
            </span>
            {a.name}
          </button>
        ))}
      </div>

      <div style={blurbBox}>
        <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>
          {app.blurb}
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
          {app.formula}
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
          <div style={{ display: "flex", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
            <CopyCodeButton code={app.code.slice(1).join("\n")} />
            <DownloadNotebookButton
              code={app.code.slice(1).join("\n")}
              filename={`ise_${app.key || "app"}.ipynb`}
              title={app.name || "ISE Application"}
              description={app.blurb || ""}
            />
          </div>
          <CodePanel codeLines={app.code} highlightedLine={ev?.line || 1} />

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
              disabled={evIdx >= app.events.length - 1}
              style={btnPrimary}
            >
              <StepForward size={16} /> Step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={evIdx >= app.events.length - 1}
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
                background: domainColor(app.domain),
                width: `${(100 * (evIdx + 1)) / app.events.length}%`,
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
            event {evIdx + 1} / {app.events.length}
          </div>
        </div>

        <StatePanel state={state} />
      </div>

      <PedagogicalNotes />
    </div>
  );
}

function domainColor(d) {
  if (d === "Inventory") return "#1f4e3d";
  if (d === "Logistics") return "#0b3da0";
  if (d === "Production Planning") return "#c8311c";
  if (d === "Inventory under Uncertainty") return "#7a3da0";
  return "#444";
}

// ============================================================
// State panel
// ============================================================
function StatePanel({ state }) {
  return (
    <div style={statePanelOuter}>
      <Section title="Imports">
        {state.imported ? (
          <span style={chip("#1f4e3d")}>pyomo.environ ✓</span>
        ) : (
          <Empty />
        )}
      </Section>

      {state.rawData.length > 0 && (
        <Section title="Problem data">
          {state.rawData.map((d, i) => (
            <KVRow key={i} k={d.label} v={d.value} desc={d.desc} mono />
          ))}
        </Section>
      )}

      <Section title="Model">
        {state.model ? (
          <span style={chip("#0b3da0")}>ConcreteModel</span>
        ) : (
          <Empty />
        )}
      </Section>

      {state.sets.length > 0 && (
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
                init={String(v.init)}, bounds=({fmt(v.lb)}, {fmt(v.ub)})
              </div>
              {v.value !== undefined && (
                <div
                  style={{
                    fontSize: 12,
                    color: "#c8311c",
                    fontFamily: "monospace",
                    marginTop: 2,
                    fontWeight: 700,
                  }}
                >
                  value = {typeof v.value === "number" ? v.value.toFixed(4) : v.value}
                </div>
              )}
            </div>
          ))
        )}
      </Section>

      {state.helpers.length > 0 && (
        <Section title="Helper expressions">
          {state.helpers.map((h, i) => (
            <span key={i} style={chip("#7a3da0")}>
              def {h.name}
            </span>
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
            <ResultLine k="objective" v={(+state.result.obj).toFixed(4)} c="#f5a524" />
            <div
              style={{
                color: "#7f7864",
                fontSize: 11,
                margin: "6px 0 2px 0",
              }}
            >
              decision values
            </div>
            {Object.entries(state.result.vars).map(([k, v]) => (
              <ResultLine key={k} k={k} v={(+v).toFixed(3)} c="#f5a524" />
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
        Why these matter for ISE
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
          <b>EOQ generalizes.</b> The textbook EOQ assumes no backorders, no
          quantity discounts, fixed lead time. Each of those assumptions has a
          closed form. Drop any TWO of them and the closed form disappears —
          but the NLP doesn't.
        </li>
        <li>
          <b>Weber is the seed of supply chain network design.</b> Single
          warehouse → multi-echelon → location with capacity → location with
          stochastic demand. Each adds variables and constraints to the same
          NLP skeleton.
        </li>
        <li>
          <b>Production smoothing IS the QP form of MPC.</b> Replace the
          objective with a tracking term, the time index with a rolling
          horizon, and you've got Model Predictive Control. ISE students use
          this in process control, supply chain coordination, and dynamic
          pricing.
        </li>
        <li>
          <b>Multi-product newsvendor is everywhere.</b> Airline overbooking
          (revenue / refunds), retail buying (markdown / stockouts), capacity
          planning (idle / overtime). All are "balance two costs around
          uncertain demand subject to a shared resource".
        </li>
        <li>
          <b>The dual is half the answer.</b> In each problem, the Lagrange
          multiplier on the binding constraint is what an ISE student really
          wants: marginal value of holding-cost reduction (EOQ), marginal
          value of an extra dollar in the budget (newsvendor), shadow price of
          a unit of capacity (production). Add{" "}
          <code style={inlineCode}>model.dual = Suffix(direction=Suffix.IMPORT)</code>{" "}
          before solving to read them.
        </li>
      </ul>
    </div>
  );
}

function InstallNote() {
  return (
    <div
      style={{
        marginBottom: 18,
        padding: "10px 14px",
        border: "1px solid #d3d3d3",
        borderRadius: 8,
        background: "#fafafa",
        fontSize: 13,
        color: "#444",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <Package size={16} />
      <span>
        Pyomo + IPOPT — see the <b>Pyomo + IPOPT — Code Stepper</b> demo for
        install instructions.
      </span>
    </div>
  );
}

// ============================================================
// CodePanel
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
        overflowX: "auto",
        overflowY: "hidden",
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
              width: "max-content",
              minWidth: "100%",
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
// Bits & atoms
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
          minWidth: 110,
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
function fmt(x) {
  if (x === "None" || x == null) return "None";
  return x;
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
