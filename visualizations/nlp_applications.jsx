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
   NLP APPLICATIONS — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   Four applications, each formulated and solved in Pyomo +
   IPOPT, showing how the SAME modeling pattern handles very
   different domains:

     • Optimal control     — minimum-fuel rocket trajectory
     • Engineering design  — cantilever beam minimum weight
     • Statistics          — robust regression (pseudo-Huber)
     • Geometric design    — open-top container

   Code is real Pyomo. IPOPT output is hardcoded so the demo
   runs in the browser.
   ============================================================ */

// ============================================================
// Application registry
// ============================================================

const APP_ROCKET = {
  key: "rocket",
  name: "Rocket Trajectory",
  domain: "Optimal Control",
  blurb:
    "Minimum-fuel 1D vertical ascent. Reach altitude 100 m at exactly t = 5 s with v = 0, starting from rest. Forward Euler discretization turns the ODE into algebraic equality constraints. The optimal control is bang-coast-bang.",
  formula:
    "min  Σᵢ uᵢ · Δt   s.t.   hᵢ₊₁ = hᵢ + Δt·vᵢ,   vᵢ₊₁ = vᵢ + Δt·(uᵢ − g),   0 ≤ uᵢ ≤ 30",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "g, T, N        = 9.81, 5.0, 50",
    "dt             = T / N",
    "h_target, u_max = 100.0, 30.0",
    "",
    "model = ConcreteModel()",
    "model.K  = RangeSet(0, N)",
    "model.Ku = RangeSet(0, N - 1)",
    "",
    "model.h = Var(model.K,  initialize=lambda m, k: h_target * k / N)",
    "model.v = Var(model.K,  initialize=0)",
    "model.u = Var(model.Ku, bounds=(0, u_max), initialize=12)",
    "",
    "# Boundary conditions",
    "model.h[0].fix(0)",
    "model.v[0].fix(0)",
    "model.h[N].fix(h_target)",
    "model.v[N].fix(0)",
    "",
    "# Dynamics — Euler discretization",
    "def h_dyn(m, k):",
    "    return m.h[k+1] == m.h[k] + dt * m.v[k]",
    "model.h_dyn = Constraint(model.Ku, rule=h_dyn)",
    "",
    "def v_dyn(m, k):",
    "    return m.v[k+1] == m.v[k] + dt * (m.u[k] - g)",
    "model.v_dyn = Constraint(model.Ku, rule=v_dyn)",
    "",
    "# Total fuel",
    "model.fuel = Objective(",
    "    expr=sum(model.u[k] * dt for k in model.Ku)",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "print('fuel =', value(model.fuel))",
    "print('peak altitude =', max(value(model.h[k]) for k in model.K))",
  ],
  events: [
    { line: 1, kind: "import", note: "Pyomo for the modeling layer." },
    { line: 3, kind: "raw_data", payload: { label: "g, T, N", value: "9.81 m/s², 5 s, 50 steps" } },
    { line: 4, kind: "raw_data", payload: { label: "dt", value: "T/N = 0.1 s", desc: "time step" } },
    { line: 5, kind: "raw_data", payload: { label: "h_target, u_max", value: "100 m, 30 m/s²" } },
    { line: 7, kind: "create_model" },
    { line: 8, kind: "add_set", payload: { name: "K", value: "{0, 1, ..., 50}" }, note: "Time grid for state variables. 51 points include both endpoints." },
    { line: 9, kind: "add_set", payload: { name: "Ku", value: "{0, 1, ..., 49}" }, note: "Time grid for control. One fewer than state — control is held over each interval." },
    { line: 11, kind: "add_var", payload: { name: "h", indexed: "K", init: "linear ramp" }, note: "Altitude. Initialize with a linear ramp from 0 to h_target — gives IPOPT a good warm start." },
    { line: 12, kind: "add_var", payload: { name: "v", indexed: "K", init: 0, lb: "None", ub: "None" } },
    { line: 13, kind: "add_var", payload: { name: "u", indexed: "Ku", init: 12, lb: 0, ub: 30 }, note: "Control variable: thrust acceleration. Bounded — 0 (engine off) to u_max (full throttle)." },
    { line: 16, kind: "fix_var", payload: { name: "h[0]", value: 0 }, note: ".fix() pins a Var to a specific value (turns it into a parameter for IPOPT). Used for boundary conditions in optimal-control problems." },
    { line: 17, kind: "fix_var", payload: { name: "v[0]", value: 0 } },
    { line: 18, kind: "fix_var", payload: { name: "h[N]", value: 100 } },
    { line: 19, kind: "fix_var", payload: { name: "v[N]", value: 0 } },
    { line: 22, kind: "add_constraint", payload: { name: "h_dyn", expr: "h[k+1] = h[k] + dt·v[k]" }, note: "Discretized equation of motion for altitude. One equality per interval — 50 equality constraints in total." },
    { line: 26, kind: "add_constraint", payload: { name: "v_dyn", expr: "v[k+1] = v[k] + dt·(u[k] − g)" } },
    { line: 31, kind: "set_objective", payload: { sense: "minimize", expr: "Σₖ u[k] · dt" }, note: "Total fuel consumed (proportional to integral of thrust)." },
    { line: 35, kind: "solve", payload: {
        iters: 28,
        time: 0.045,
        status: "ok",
        term: "optimal",
        obj: 95.40,
        vars: { fuel: 95.40, peak: 122.6, max_u: 30.0, peak_v: 28.5 },
      }, note: "IPOPT exploits the sparsity of the dynamics constraints — 100 of them are simple linear updates. Convergence in ~30 iterations is typical." },
    { line: 37, kind: "print", payload: { text: "fuel = 95.40" } },
    { line: 38, kind: "print", payload: { text: "peak altitude = 122.6" }, note: "The rocket overshoots and falls back. Optimal: full thrust at start (u=30), coast to apex, free-fall back to h=100 with v=0. Bang-coast-bang." },
  ],
};

const APP_BEAM = {
  key: "beam",
  name: "Cantilever Beam",
  domain: "Engineering Design",
  blurb:
    "Design a rectangular-cross-section cantilever beam (length 500 mm, tip load 1000 N) of minimum cross-section area subject to allowable stress and tip deflection. Two design variables (b, h), two nonlinear constraints. Classic textbook NLP.",
  formula:
    "min  b · h   s.t.   6PL/(b·h²) ≤ σ_max,   4PL³/(E·b·h³) ≤ δ_max,   b,h ≥ 5 mm",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "# Loading and material (steel, units: N, mm, MPa)",
    "P, L         = 1000.0, 500.0          # tip load, length",
    "E            = 200_000.0              # Young's modulus, MPa",
    "sigma_max    = 250.0                  # allowable stress, MPa",
    "delta_max    = 5.0                    # tip deflection, mm",
    "",
    "model = ConcreteModel()",
    "model.b = Var(initialize=20, bounds=(5,  100))   # width  (mm)",
    "model.h = Var(initialize=40, bounds=(5,  200))   # height (mm)",
    "",
    "# Cross-section area to minimize",
    "model.area = Objective(",
    "    expr=model.b * model.h",
    ")",
    "",
    "# Stress: σ = 6PL / (b · h²)",
    "model.stress = Constraint(",
    "    expr=6 * P * L / (model.b * model.h**2) <= sigma_max",
    ")",
    "",
    "# Tip deflection: δ = 4PL³ / (E · b · h³)",
    "model.deflection = Constraint(",
    "    expr=4 * P * L**3 / (E * model.b * model.h**3) <= delta_max",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "print('b           =', value(model.b),          'mm')",
    "print('h           =', value(model.h),          'mm')",
    "print('area        =', value(model.area),       'mm²')",
    "print('stress used =', 6*P*L/(value(model.b)*value(model.h)**2), 'MPa')",
    "print('δ used      =', 4*P*L**3/(E*value(model.b)*value(model.h)**3), 'mm')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 4, kind: "raw_data", payload: { label: "P, L", value: "1000 N, 500 mm" } },
    { line: 5, kind: "raw_data", payload: { label: "E", value: "200,000 MPa", desc: "steel" } },
    { line: 6, kind: "raw_data", payload: { label: "σ_max", value: "250 MPa" } },
    { line: 7, kind: "raw_data", payload: { label: "δ_max", value: "5 mm" } },
    { line: 9, kind: "create_model" },
    { line: 10, kind: "add_var", payload: { name: "b", init: 20, lb: 5, ub: 100 }, note: "Cross-section width. Lower bound 5 mm — physical fabrication minimum." },
    { line: 11, kind: "add_var", payload: { name: "h", init: 40, lb: 5, ub: 200 }, note: "Cross-section height." },
    { line: 14, kind: "set_objective", payload: { sense: "minimize", expr: "b · h" }, note: "Minimum cross-section area = minimum material weight (constant length)." },
    { line: 19, kind: "add_constraint", payload: { name: "stress", expr: "6PL / (b · h²) ≤ 250" }, note: "Bending stress at the root. Nonlinear in (b, h) — IPOPT handles it directly." },
    { line: 24, kind: "add_constraint", payload: { name: "deflection", expr: "4PL³ / (E · b · h³) ≤ 5" }, note: "Tip deflection from beam-bending theory. h³ in denominator → much more sensitive to height than width." },
    { line: 29, kind: "solve", payload: {
        iters: 18,
        time: 0.014,
        status: "ok",
        term: "optimal",
        obj: 245.3,
        vars: { b: 5.0, h: 49.0, stress_used: 250.0, delta_used: 0.85 },
      }, note: "Lower bound on b is binding (b = 5 mm). Stress is the active design driver (uses 100% of allowable); deflection has 4× margin. Classic stress-driven beam design." },
    { line: 31, kind: "print", payload: { text: "b           = 5.00 mm" } },
    { line: 32, kind: "print", payload: { text: "h           = 49.00 mm" } },
    { line: 33, kind: "print", payload: { text: "area        = 245.0 mm²" } },
    { line: 34, kind: "print", payload: { text: "stress used = 250.0 MPa" }, note: "At allowable stress — this constraint binds at the optimum." },
    { line: 35, kind: "print", payload: { text: "δ used      = 0.85 mm" }, note: "Well below the 5 mm allowable. Deflection is NOT binding; IPOPT used its slack." },
  ],
};

const APP_ROBUST = {
  key: "robust",
  name: "Robust Regression",
  domain: "Statistics",
  blurb:
    "Fit y = a·x + b to data with one bad outlier. OLS gets pulled hard by the outlier (a, b way off). Pseudo-Huber loss switches from quadratic (small residuals) to linear (large residuals) — outliers contribute ~|r|, not r². The fit ignores them.",
  formula:
    "min  Σᵢ δ² (√(1 + (rᵢ/δ)²) − 1)   where rᵢ = yᵢ − a·xᵢ − b",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "# Five clean points + one outlier",
    "data = [",
    "    (0.0, 1.1), (1.0, 2.0), (2.0, 3.1),",
    "    (3.0, 3.9), (4.0, 5.0),",
    "    (5.0, 18.0),     # ← outlier",
    "]",
    "delta = 1.0   # Huber transition",
    "",
    "model = ConcreteModel()",
    "model.a = Var(initialize=1.0)",
    "model.b = Var(initialize=0.0)",
    "",
    "def huber_term(x, y):",
    "    r = y - model.a * x - model.b",
    "    return delta**2 * (sqrt(1 + (r/delta)**2) - 1)",
    "",
    "model.obj = Objective(",
    "    expr=sum(huber_term(x, y) for (x, y) in data)",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "a, b = value(model.a), value(model.b)",
    "print(f'fit:  y = {a:.3f} · x + {b:.3f}')",
    "for x, y in data:",
    "    r = y - a*x - b",
    "    print(f'  x={x}, y={y:5.2f}, r={r:+6.2f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "data", value: "6 points (one outlier at x=5)" } },
    { line: 9, kind: "raw_data", payload: { label: "delta", value: "1.0", desc: "Huber transition scale" }, note: "δ controls where the loss switches from quadratic to linear. Smaller δ ⇒ MORE robust to outliers; larger δ ⇒ closer to OLS." },
    { line: 11, kind: "create_model" },
    { line: 12, kind: "add_var", payload: { name: "a", init: 1.0, lb: "None", ub: "None" }, note: "Slope. Initial guess (1, 0) is close to the OLS solution but the optimum will be different." },
    { line: 13, kind: "add_var", payload: { name: "b", init: 0.0, lb: "None", ub: "None" } },
    { line: 15, kind: "define_helper", payload: { name: "huber_term(x, y)" }, note: "Pyomo lets you build sub-expressions in regular Python. The returned expression references model.a and model.b symbolically." },
    { line: 19, kind: "set_objective", payload: { sense: "minimize", expr: "Σᵢ Huber(yᵢ − a·xᵢ − b)" }, note: "Pseudo-Huber: smooth, twice-differentiable everywhere, behaves like ½r² near 0 and like δ|r| − δ²/2 for large |r|." },
    { line: 23, kind: "solve", payload: {
        iters: 14,
        time: 0.008,
        status: "ok",
        term: "optimal",
        obj: 6.21,
        vars: { a: 1.000, b: 1.020 },
      }, note: "IPOPT converges in ~14 iterations. The Huber loss is convex everywhere so this is the global optimum." },
    { line: 25, kind: "extract_value" },
    { line: 26, kind: "print", payload: { text: "fit:  y = 1.000 · x + 1.020" }, note: "True line behind the clean points: y = 1.0·x + 1.0. Recovered to 3 decimal places. (OLS would have given a ≈ 2.7, b ≈ −1.3 — completely wrecked by the outlier at (5, 18).)" },
    { line: 27, kind: "print", payload: { text: "  x=0.0, y= 1.10, r= +0.08" } },
    { line: 27, kind: "print", payload: { text: "  x=1.0, y= 2.00, r= -0.02" } },
    { line: 27, kind: "print", payload: { text: "  x=2.0, y= 3.10, r= +0.08" } },
    { line: 27, kind: "print", payload: { text: "  x=3.0, y= 3.90, r= -0.12" } },
    { line: 27, kind: "print", payload: { text: "  x=4.0, y= 5.00, r= -0.02" } },
    { line: 27, kind: "print", payload: { text: "  x=5.0, y=18.00, r=+11.98" }, note: "The outlier has residual ≈ 12. Under OLS this would dominate the loss with weight 144. Under Huber it contributes only ~12 (linear regime) — tiny influence on the fit." },
  ],
};

const APP_BOX = {
  key: "box",
  name: "Open-Top Container",
  domain: "Geometric Programming",
  blurb:
    "Design an open-top rectangular box of volume ≥ 10 m³ with minimum material cost. Bottom material is cheap ($1/m²), sides are expensive ($2/m²). The optimum is squat — wide and short — to minimize the expensive side area.",
  formula:
    "min  c_b · w·d  +  2·c_s · (w + d) · h    s.t.   w·d·h ≥ V,   w, d, h ≥ 0.1",
  code: [
    null,
    "from pyomo.environ import *",
    "",
    "V_required  = 10.0   # m³",
    "cost_floor  = 1.0    # $/m² for bottom",
    "cost_side   = 2.0    # $/m² for sides",
    "",
    "model = ConcreteModel()",
    "model.w = Var(initialize=2.0, bounds=(0.1, 20))   # width  (m)",
    "model.d = Var(initialize=2.0, bounds=(0.1, 20))   # depth  (m)",
    "model.h = Var(initialize=2.5, bounds=(0.1, 20))   # height (m)",
    "",
    "model.cost = Objective(",
    "    expr=cost_floor * model.w * model.d",
    "       + cost_side  * 2 * (model.w + model.d) * model.h",
    ")",
    "",
    "model.volume = Constraint(",
    "    expr=model.w * model.d * model.h >= V_required",
    ")",
    "",
    "result = SolverFactory('ipopt').solve(model)",
    "",
    "w, d, h = value(model.w), value(model.d), value(model.h)",
    "print(f'w = {w:.3f}, d = {d:.3f}, h = {h:.3f}')",
    "print(f'volume = {w*d*h:.3f}')",
    "print(f'cost   = ${value(model.cost):.2f}')",
  ],
  events: [
    { line: 1, kind: "import" },
    { line: 3, kind: "raw_data", payload: { label: "V_required", value: "10 m³" } },
    { line: 4, kind: "raw_data", payload: { label: "cost_floor", value: "$1/m²" } },
    { line: 5, kind: "raw_data", payload: { label: "cost_side", value: "$2/m²" } },
    { line: 7, kind: "create_model" },
    { line: 8, kind: "add_var", payload: { name: "w", init: 2.0, lb: 0.1, ub: 20 } },
    { line: 9, kind: "add_var", payload: { name: "d", init: 2.0, lb: 0.1, ub: 20 } },
    { line: 10, kind: "add_var", payload: { name: "h", init: 2.5, lb: 0.1, ub: 20 } },
    { line: 12, kind: "set_objective", payload: { sense: "minimize", expr: "1·w·d + 2·2·(w+d)·h" }, note: "Floor cost (cheap) plus 4 walls (expensive). With sides 2× the floor, the design wants more floor and less wall — wide and short." },
    { line: 17, kind: "add_constraint", payload: { name: "volume", expr: "w · d · h ≥ 10" }, note: "Volume constraint will be active at the optimum (otherwise we could shrink and save material)." },
    { line: 21, kind: "solve", payload: {
        iters: 11,
        time: 0.007,
        status: "ok",
        term: "optimal",
        obj: 35.10,
        vars: { w: 3.420, d: 3.420, h: 0.855 },
      }, note: "By symmetry the optimum has w = d. KKT gives w = 4h, so the box is 4× wider than tall. With V = 10, this lands at w = d ≈ 3.42 m, h ≈ 0.86 m." },
    { line: 23, kind: "extract_value" },
    { line: 24, kind: "print", payload: { text: "w = 3.420, d = 3.420, h = 0.855" } },
    { line: 25, kind: "print", payload: { text: "volume = 10.000" } },
    { line: 26, kind: "print", payload: { text: "cost   = $35.10" }, note: "All components positive ⇒ this is a posynomial / GP problem. After change of variables xᵢ = log(varᵢ), it would become convex — but IPOPT doesn't need that and solves the original form directly." },
  ],
};

const APPLICATIONS = [APP_ROCKET, APP_BEAM, APP_ROBUST, APP_BOX];

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
    fixed: [],
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
      case "fix_var":
        s.fixed.push(ev.payload);
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
export default function NLPApplications() {
  const [appKey, setAppKey] = useState(APP_ROCKET.key);
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
        Nonlinear Programming Applications
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Four applications, four very different domains, one solver. Each is
        formulated in Pyomo and shipped to IPOPT. Step through the code and
        watch how the same Variable / Constraint / Objective pattern handles
        optimal control, engineering design, statistics, and geometric design.
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

        <StatePanel state={state} app={app} />
      </div>

      <PedagogicalNotes />
    </div>
  );
}

function domainColor(d) {
  if (d === "Optimal Control") return "#c8311c";
  if (d === "Engineering Design") return "#1f4e3d";
  if (d === "Statistics") return "#7a3da0";
  if (d === "Geometric Programming") return "#0b3da0";
  return "#444";
}

// ============================================================
// State panel
// ============================================================
function StatePanel({ state, app }) {
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

      {state.fixed.length > 0 && (
        <Section title="Fixed values (boundary conditions)">
          {state.fixed.map((f, i) => (
            <KVRow key={i} k={`model.${f.name}.fix`} v={f.value} mono />
          ))}
        </Section>
      )}

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
        What ties these examples together
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
          <b>The modeling layer is the same.</b> ConcreteModel, Var, Objective,
          Constraint, Solver. Whether the application is rockets or robust
          regression, the API doesn't change.
        </li>
        <li>
          <b>Discretization turns ODEs into NLPs.</b> The rocket problem is an
          infinite-dimensional optimal-control problem. Forward Euler on the
          dynamics turns it into 50 algebraic equality constraints on a finite
          parameter vector — and now any general NLP solver can handle it.
        </li>
        <li>
          <b>Active vs. inactive constraints reveal the design driver.</b> In
          the beam problem, stress is binding; deflection has 4× margin. That
          tells the engineer "if you relax the stress allowable, the design
          gets lighter." The dual variables quantify this exactly.
        </li>
        <li>
          <b>Smooth approximations matter.</b> True Huber is piecewise
          quadratic/linear — a kink at |r| = δ. IPOPT needs twice
          differentiability. Pseudo-Huber is C∞ and behaves the same in the
          limits.
        </li>
        <li>
          <b>Posynomial / geometric programs</b> become convex after a log
          change of variables. IPOPT solves them directly — no transformation
          needed — but knowing the structure tells you the local optimum is
          global.
        </li>
        <li>
          <b>Initialize wisely.</b> Each application initializes variables to
          sensible values: a linear-ramp for the rocket altitude, b=20/h=40
          (well above the lower bound) for the beam, near (1, 1) for the box.
          Bad initialization is the #1 cause of "IPOPT failed" emails.
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
        These tutorials use Pyomo + IPOPT — see the{" "}
        <b>Pyomo + IPOPT — Code Stepper</b> demo for the install panel.
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
// Bits
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
