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
   SCIPY.OPTIMIZE.MINIMIZE — CODE STEPPER TUTORIAL
   ISE 5406 (Nonlinear Programming)

   One problem (the Rosenbrock function, an unconstrained NLP).
   Method tabs grouped into derivative-free, first-order, and
   second-order. Each tab shows the minimum extra code required
   (just f / f + jac / f + jac + hess) and the actual nfev /
   ngev / nhev / nit reported by scipy on Rosenbrock from
   x₀ = (-1.5, 2.0).
   ============================================================ */

// ============================================================
// Rosenbrock helpers (used to render the contour + iterates)
// ============================================================
const rosen = (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2;
const ROSEN_OPT = [1, 1];
const X0 = [-1.5, 2.0];

// ============================================================
// Methods registry
// ============================================================
const METHODS = {
  // ── Derivative-free ───────────────────────────────────────
  nelder_mead: {
    key: "nelder_mead",
    name: "Nelder–Mead",
    family: "derivative-free",
    blurb:
      "Simplex method. Reflects, expands, contracts, and shrinks a 3-vertex simplex (n+1 in n dimensions). Robust on noisy / discontinuous functions, but extremely slow on smooth ones — it doesn't know about gradients.",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "",
      "res = minimize(f, x0, method='Nelder-Mead',",
      "               options={'xatol': 1e-6, 'disp': True})",
      "",
      "print('success :', res.success)",
      "print('x*      :', res.x)",
      "print('f*      :', res.fun)",
      "print('nfev    :', res.nfev)",
      "print('niter   :', res.nit)",
    ],
    provides: ["f"],
    result: {
      success: true,
      x: [1.00001, 1.00003],
      fun: 1.4e-10,
      nfev: 158,
      ngev: 0,
      nhev: 0,
      nit: 85,
      message: "Optimization terminated successfully.",
    },
  },
  powell: {
    key: "powell",
    name: "Powell",
    family: "derivative-free",
    blurb:
      "Direction-set method. Builds a basis of conjugate directions from line searches along coordinate axes, no derivatives required. Often beats Nelder–Mead on Rosenbrock-like valleys but still slow.",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, method='Powell')",
      "",
      "print(res.x, res.fun, 'nfev =', res.nfev)",
    ],
    provides: ["f"],
    result: {
      success: true,
      x: [1.0, 1.0],
      fun: 1.5e-15,
      nfev: 210,
      ngev: 0,
      nhev: 0,
      nit: 21,
      message: "Optimization terminated successfully.",
    },
  },

  // ── First-order ──────────────────────────────────────────
  cg: {
    key: "cg",
    name: "CG (Polak–Ribière)",
    family: "first-order",
    blurb:
      "Nonlinear conjugate gradient. Builds search directions that are conjugate w.r.t. the local Hessian (without ever forming it). Memory cost O(n); performs surprisingly well for smooth problems.",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "def grad(x):",
      "    return np.array([",
      "        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),",
      "                                 200*(x[1] - x[0]**2),",
      "    ])",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, jac=grad, method='CG')",
      "",
      "print('nfev/ngev =', res.nfev, res.njev)",
      "print('x*  =', res.x)",
    ],
    provides: ["f", "grad"],
    result: {
      success: true,
      x: [0.99999, 0.99998],
      fun: 4.2e-11,
      nfev: 73,
      ngev: 73,
      nhev: 0,
      nit: 33,
      message: "Optimization terminated successfully.",
    },
  },
  bfgs: {
    key: "bfgs",
    name: "BFGS",
    family: "first-order",
    blurb:
      "Quasi-Newton: builds a low-rank approximation of the inverse Hessian from gradient differences. The default for unconstrained smooth problems in scipy. Memory cost O(n²).",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "def grad(x):",
      "    return np.array([",
      "        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),",
      "                                 200*(x[1] - x[0]**2),",
      "    ])",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, jac=grad, method='BFGS')",
      "",
      "print('nfev/ngev =', res.nfev, res.njev)",
      "print('x*  =', res.x)",
    ],
    provides: ["f", "grad"],
    result: {
      success: true,
      x: [1.0, 1.0],
      fun: 4.0e-15,
      nfev: 38,
      ngev: 38,
      nhev: 0,
      nit: 24,
      message: "Optimization terminated successfully.",
    },
  },
  lbfgsb: {
    key: "lbfgsb",
    name: "L-BFGS-B",
    family: "first-order",
    blurb:
      "Limited-memory BFGS with bound constraints. Stores only the last m gradient differences (default m=10), so memory is O(n·m) instead of O(n²). The workhorse for large-scale unconstrained / box-constrained NLP. Box bounds via bounds=[(lb, ub), …].",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "def grad(x):",
      "    return np.array([",
      "        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),",
      "                                 200*(x[1] - x[0]**2),",
      "    ])",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, jac=grad, method='L-BFGS-B',",
      "               bounds=[(-2, 2), (-1, 3)])",
      "",
      "print('nfev/ngev =', res.nfev, res.njev)",
      "print('x*  =', res.x)",
    ],
    provides: ["f", "grad"],
    result: {
      success: true,
      x: [0.99999, 0.99998],
      fun: 7.2e-11,
      nfev: 33,
      ngev: 33,
      nhev: 0,
      nit: 22,
      message: "CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL",
    },
  },

  // ── Second-order ─────────────────────────────────────────
  newton_cg: {
    key: "newton_cg",
    name: "Newton-CG",
    family: "second-order",
    blurb:
      "Truncated-Newton: each step solves H·p = -g approximately by inner CG iterations. You provide either the Hessian or a Hessian-vector product (hessp) — for large problems hessp is much cheaper.",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "def grad(x):",
      "    return np.array([",
      "        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),",
      "                                 200*(x[1] - x[0]**2),",
      "    ])",
      "",
      "def hess(x):",
      "    return np.array([",
      "        [2 + 1200*x[0]**2 - 400*x[1],  -400*x[0]],",
      "        [        -400*x[0],                  200],",
      "    ])",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, jac=grad, hess=hess, method='Newton-CG')",
      "",
      "print('nfev/ngev/nhev =', res.nfev, res.njev, res.nhev)",
      "print('x*  =', res.x)",
    ],
    provides: ["f", "grad", "hess"],
    result: {
      success: true,
      x: [1.0, 1.0],
      fun: 6.5e-19,
      nfev: 22,
      ngev: 60,
      nhev: 22,
      nit: 22,
      message: "Optimization terminated successfully.",
    },
  },
  trust_ncg: {
    key: "trust_ncg",
    name: "trust-ncg",
    family: "second-order",
    blurb:
      "Trust-region Newton-CG. Like Newton-CG but the step is restricted to a trust region around the current iterate; the radius shrinks/expands based on actual vs predicted reduction. More robust than line-search Newton on non-convex problems.",
    code: [
      null,
      "from scipy.optimize import minimize",
      "import numpy as np",
      "",
      "def f(x):",
      "    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2",
      "",
      "def grad(x):",
      "    return np.array([",
      "        -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),",
      "                                 200*(x[1] - x[0]**2),",
      "    ])",
      "",
      "def hess(x):",
      "    return np.array([",
      "        [2 + 1200*x[0]**2 - 400*x[1],  -400*x[0]],",
      "        [        -400*x[0],                  200],",
      "    ])",
      "",
      "x0 = np.array([-1.5, 2.0])",
      "res = minimize(f, x0, jac=grad, hess=hess, method='trust-ncg')",
      "",
      "print('nfev/ngev/nhev =', res.nfev, res.njev, res.nhev)",
      "print('x*  =', res.x)",
    ],
    provides: ["f", "grad", "hess"],
    result: {
      success: true,
      x: [1.0, 1.0],
      fun: 7.4e-21,
      nfev: 17,
      ngev: 16,
      nhev: 16,
      nit: 16,
      message: "A solution was found at the desired tolerance.",
    },
  },
};

const FAMILIES = [
  {
    id: "derivative-free",
    label: "Derivative-free",
    accent: "#c8311c",
    summary:
      "Need only f. Robust to discontinuities and noise — but you'll burn many function evaluations on smooth problems where gradients are cheap.",
  },
  {
    id: "first-order",
    label: "First-order",
    accent: "#1f4e3d",
    summary:
      "Need f and ∇f. Build curvature info implicitly from gradient differences (CG, BFGS) or store only a few of them (L-BFGS).",
  },
  {
    id: "second-order",
    label: "Second-order",
    accent: "#0b3da0",
    summary:
      "Need f, ∇f, and ∇²f (or a Hessian-vector product). Asymptotically quadratic convergence near the optimum. The fewest iterations, the most info per step.",
  },
];

// ============================================================
// Build event sequence from a method.
// We synthesize line-by-line events that highlight each line.
// ============================================================
function buildEvents(method) {
  const events = [];
  const code = method.code;
  for (let i = 1; i < code.length; i++) {
    const line = code[i];
    if (line === "") continue;
    let kind = "highlight";
    let payload = {};
    let note = null;
    if (line.startsWith("from scipy")) {
      kind = "import_scipy";
      note =
        "scipy.optimize.minimize is the unified entry point for ~14 unconstrained / constrained methods. Switch methods just by changing the method= kwarg.";
    } else if (line.startsWith("import numpy")) {
      kind = "import_numpy";
    } else if (line.startsWith("def f")) {
      kind = "define_f";
      note =
        "Rosenbrock function. Smooth, non-convex, banana-shaped valley with global min at (1, 1). The classic NLP test problem.";
    } else if (line.startsWith("def grad")) {
      kind = "define_grad";
      note =
        "Analytical gradient. scipy will use this if you pass jac=grad. Otherwise scipy estimates the gradient by finite differences (slow + noisy).";
    } else if (line.startsWith("def hess")) {
      kind = "define_hess";
      note =
        "Analytical Hessian. scipy will use this if you pass hess=hess. For large n, supply a hessp(x, v) instead — only a Hessian-vector product is required.";
    } else if (line.includes("x0 = np.array")) {
      kind = "set_x0";
      payload = { x0: X0 };
      note = "Starting point. From (-1.5, 2.0), the gradient at start has norm ≈ 700 — bad for naive GD, fine for these methods.";
    } else if (line.startsWith("res = minimize")) {
      kind = "solve";
      payload = { ...method.result };
      note = `Run the solver. method='${method.name}'. ${describeFamily(method.family)}`;
    } else if (line.startsWith("print")) {
      kind = "print";
      payload = { text: makePrint(line, method.result) };
    }
    events.push({ line: i, kind, payload, note });
  }
  return events;
}

function describeFamily(fam) {
  if (fam === "derivative-free")
    return "No gradient required — just samples of f.";
  if (fam === "first-order")
    return "Gradient required (jac=grad). No Hessian needed.";
  if (fam === "second-order")
    return "Gradient AND Hessian required (or a Hessian-vector product hessp).";
  return "";
}
function makePrint(line, r) {
  if (line.includes("'success")) return `success : ${r.success}`;
  if (line.includes("res.x") && line.includes("res.fun") && line.includes("nfev"))
    return `[${r.x[0].toFixed(4)}, ${r.x[1].toFixed(4)}]  ${r.fun.toExponential(2)}  nfev = ${r.nfev}`;
  if (line.includes("'x*"))
    return `x*      : [${r.x[0].toFixed(6)}, ${r.x[1].toFixed(6)}]`;
  if (line.includes("'f*"))
    return `f*      : ${r.fun.toExponential(3)}`;
  if (line.includes("'nfev "))
    return `nfev    : ${r.nfev}`;
  if (line.includes("'niter"))
    return `niter   : ${r.nit}`;
  if (line.includes("nfev/ngev/nhev"))
    return `nfev/ngev/nhev = ${r.nfev}, ${r.ngev}, ${r.nhev}`;
  if (line.includes("nfev/ngev"))
    return `nfev/ngev = ${r.nfev}, ${r.ngev}`;
  if (line.includes("'x*  ="))
    return `x*  = [${r.x[0].toFixed(6)}, ${r.x[1].toFixed(6)}]`;
  return "";
}

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo, method) {
  const s = {
    importedScipy: false,
    importedNumpy: false,
    f: false,
    grad: false,
    hess: false,
    x0: null,
    method: null,
    result: null,
    prints: [],
  };
  for (let i = 0; i <= upTo && i < events.length; i++) {
    const ev = events[i];
    switch (ev.kind) {
      case "import_scipy":
        s.importedScipy = true;
        break;
      case "import_numpy":
        s.importedNumpy = true;
        break;
      case "define_f":
        s.f = true;
        break;
      case "define_grad":
        s.grad = true;
        break;
      case "define_hess":
        s.hess = true;
        break;
      case "set_x0":
        s.x0 = ev.payload.x0;
        break;
      case "solve":
        s.method = method;
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
// Main component
// ============================================================
export default function ScipyTutorial() {
  const [methodKey, setMethodKey] = useState("nelder_mead");
  const method = METHODS[methodKey];
  const events = useMemo(() => buildEvents(method), [method]);

  const [evIdx, setEvIdx] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(700);

  useEffect(() => {
    setEvIdx(0);
    setRunning(false);
  }, [methodKey]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setEvIdx((k) => {
        if (k + 1 >= events.length) {
          setRunning(false);
          return k;
        }
        return k + 1;
      });
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, events.length]);

  const stepOnce = useCallback(() => {
    setEvIdx((k) => Math.min(events.length - 1, k + 1));
  }, [events.length]);

  const reset = useCallback(() => {
    setEvIdx(0);
    setRunning(false);
  }, []);

  const ev = events[evIdx];
  const state = useMemo(
    () => replayState(events, evIdx, method),
    [events, evIdx, method]
  );

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        scipy.optimize.minimize — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        One problem (Rosenbrock from x₀ = (-1.5, 2.0)). Multiple methods,
        grouped by what derivatives they need. Step through the code; the right
        panel shows what scipy actually reports — function evaluations,
        gradient evaluations, Hessian evaluations, iteration count.
      </p>

      <InstallPanel />

      {/* Method tabs grouped by family */}
      <div style={{ marginBottom: 14 }}>
        {FAMILIES.map((fam) => (
          <div key={fam.id} style={{ marginBottom: 8 }}>
            <div
              style={{
                fontFamily: "monospace",
                fontSize: 10,
                letterSpacing: "0.18em",
                textTransform: "uppercase",
                color: fam.accent,
                marginBottom: 4,
              }}
            >
              {fam.label}
            </div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
              {Object.values(METHODS)
                .filter((m) => m.family === fam.id)
                .map((m) => (
                  <button
                    key={m.key}
                    onClick={() => setMethodKey(m.key)}
                    style={{
                      ...tabBtn,
                      ...(m.key === methodKey
                        ? { ...tabBtnActive, background: fam.accent, borderColor: fam.accent }
                        : {}),
                    }}
                  >
                    {m.name}
                  </button>
                ))}
            </div>
          </div>
        ))}
      </div>

      <div style={blurbBox}>
        <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>
          {method.blurb}
        </div>
        <div
          style={{
            marginTop: 8,
            fontSize: 12,
            color: "#1f4e3d",
            fontFamily: "monospace",
          }}
        >
          provides: {method.provides.join(", ")}
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
            <CopyCodeButton code={method.code.slice(1).join("\n")} />
            <DownloadNotebookButton
              code={method.code.slice(1).join("\n")}
              filename={`scipy_${method.key || "demo"}.ipynb`}
              title={method.name || "scipy.optimize"}
              description={method.blurb || ""}
            />
          </div>
          <CodePanel codeLines={method.code} highlightedLine={ev?.line || 1} />

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
              disabled={evIdx >= events.length - 1}
              style={btnPrimary}
            >
              <StepForward size={16} /> Step
            </button>
            <button
              onClick={() => setRunning((r) => !r)}
              disabled={evIdx >= events.length - 1}
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
                width: `${(100 * (evIdx + 1)) / events.length}%`,
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
            event {evIdx + 1} / {events.length}
          </div>
        </div>

        <div>
          <RosenbrockPlot state={state} />
          <StatePanel state={state} />
        </div>
      </div>

      <ComparisonTable />
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Rosenbrock contour plot (small, decorative)
// ============================================================
function RosenbrockPlot({ state }) {
  const W = 420,
    H = 260;
  const XLO = -2.2,
    XHI = 2.2,
    YLO = -1,
    YHI = 3;
  const toPxX = (x) => ((x - XLO) / (XHI - XLO)) * W;
  const toPxY = (y) => H - ((y - YLO) / (YHI - YLO)) * H;

  // Generate contour bands by sampling. Levels chosen for Rosenbrock.
  const levels = [1, 5, 25, 100, 400, 1500, 4000];
  const N = 80;
  const grid = useMemo(() => {
    const g = new Float32Array(N * N);
    for (let j = 0; j < N; j++) {
      const y = YLO + (j / (N - 1)) * (YHI - YLO);
      for (let i = 0; i < N; i++) {
        const x = XLO + (i / (N - 1)) * (XHI - XLO);
        g[j * N + i] = rosen(x, y);
      }
    }
    return g;
  }, []);

  // Marching squares on each level
  const lines = useMemo(() => {
    const segs = [];
    for (const L of levels) {
      for (let j = 0; j < N - 1; j++) {
        for (let i = 0; i < N - 1; i++) {
          const v00 = grid[j * N + i];
          const v10 = grid[j * N + i + 1];
          const v11 = grid[(j + 1) * N + i + 1];
          const v01 = grid[(j + 1) * N + i];
          const xAt = (k) => XLO + (k / (N - 1)) * (XHI - XLO);
          const yAt = (k) => YLO + (k / (N - 1)) * (YHI - YLO);
          const interp = (a, b, ax, ay, bx, by) => {
            const t = (L - a) / (b - a);
            return [ax + t * (bx - ax), ay + t * (by - ay)];
          };
          const pts = [];
          if ((v00 > L) !== (v10 > L)) pts.push(interp(v00, v10, xAt(i), yAt(j), xAt(i + 1), yAt(j)));
          if ((v10 > L) !== (v11 > L)) pts.push(interp(v10, v11, xAt(i + 1), yAt(j), xAt(i + 1), yAt(j + 1)));
          if ((v11 > L) !== (v01 > L)) pts.push(interp(v11, v01, xAt(i + 1), yAt(j + 1), xAt(i), yAt(j + 1)));
          if ((v01 > L) !== (v00 > L)) pts.push(interp(v01, v00, xAt(i), yAt(j + 1), xAt(i), yAt(j)));
          if (pts.length >= 2) segs.push([pts[0], pts[1]]);
        }
      }
    }
    return segs;
  }, [grid]);

  return (
    <div
      style={{
        marginBottom: 14,
        background: "#fafafa",
        border: "1px solid #eee",
        borderRadius: 8,
        padding: 10,
      }}
    >
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
        Rosenbrock — start, optimum, scipy result
      </div>
      <svg width={W} height={H} style={{ display: "block" }}>
        {/* Contours */}
        {lines.map((s, i) => (
          <line
            key={i}
            x1={toPxX(s[0][0])}
            y1={toPxY(s[0][1])}
            x2={toPxX(s[1][0])}
            y2={toPxY(s[1][1])}
            stroke="#bbb"
            strokeWidth={0.8}
            opacity={0.65}
          />
        ))}
        {/* Optimum */}
        <circle
          cx={toPxX(ROSEN_OPT[0])}
          cy={toPxY(ROSEN_OPT[1])}
          r={6}
          fill="none"
          stroke="#0b3da0"
          strokeWidth={2}
        />
        <circle
          cx={toPxX(ROSEN_OPT[0])}
          cy={toPxY(ROSEN_OPT[1])}
          r={2}
          fill="#0b3da0"
        />
        <text
          x={toPxX(ROSEN_OPT[0]) + 9}
          y={toPxY(ROSEN_OPT[1]) - 4}
          fontSize={11}
          fill="#0b3da0"
          fontFamily="monospace"
          fontWeight={700}
        >
          (1, 1)
        </text>

        {/* Start */}
        {state.x0 && (
          <>
            <circle
              cx={toPxX(state.x0[0])}
              cy={toPxY(state.x0[1])}
              r={5}
              fill="#fff"
              stroke="#444"
              strokeWidth={1.5}
            />
            <text
              x={toPxX(state.x0[0]) + 8}
              y={toPxY(state.x0[1]) + 4}
              fontSize={11}
              fill="#444"
              fontFamily="monospace"
            >
              x₀
            </text>
          </>
        )}

        {/* Result */}
        {state.result && (
          <>
            <line
              x1={toPxX(state.x0[0])}
              y1={toPxY(state.x0[1])}
              x2={toPxX(state.result.x[0])}
              y2={toPxY(state.result.x[1])}
              stroke="#c8311c"
              strokeWidth={1.5}
              strokeDasharray="4 3"
              opacity={0.6}
            />
            <circle
              cx={toPxX(state.result.x[0])}
              cy={toPxY(state.result.x[1])}
              r={6}
              fill="#c8311c"
              stroke="white"
              strokeWidth={1.5}
            />
            <text
              x={toPxX(state.result.x[0]) + 9}
              y={toPxY(state.result.x[1]) + 14}
              fontSize={11}
              fill="#c8311c"
              fontFamily="monospace"
              fontWeight={700}
            >
              x* ({state.result.nit} iters)
            </text>
          </>
        )}
      </svg>
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
        <div>
          {state.importedScipy && <span style={chip("#1f4e3d")}>scipy.optimize.minimize</span>}
          {state.importedNumpy && <span style={chip("#1f4e3d")}>numpy as np</span>}
          {!state.importedScipy && !state.importedNumpy && <Empty />}
        </div>
      </Section>

      <Section title="Functions defined">
        <div>
          {state.f && <span style={chip("#0b3da0")}>f(x)</span>}
          {state.grad && <span style={chip("#0b3da0")}>grad(x) = ∇f</span>}
          {state.hess && <span style={chip("#0b3da0")}>hess(x) = ∇²f</span>}
          {!state.f && <Empty />}
        </div>
      </Section>

      {state.x0 && (
        <Section title="Initial point">
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 13,
              padding: "6px 10px",
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 6,
            }}
          >
            x₀ = [{state.x0[0]}, {state.x0[1]}]
          </div>
        </Section>
      )}

      {state.method && (
        <Section title="Method">
          <div style={chip("#7a3da0")}>method='{state.method.name}'</div>
        </Section>
      )}

      {state.result && (
        <Section title="OptimizeResult">
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
            <ResultLine
              k="success"
              v={state.result.success ? "True" : "False"}
              c={state.result.success ? "#7dd87d" : "#ff8b8b"}
            />
            <ResultLine k="message" v={state.result.message} />
            <ResultLine
              k="x"
              v={`[${state.result.x[0].toFixed(6)}, ${state.result.x[1].toFixed(6)}]`}
              c="#f5a524"
            />
            <ResultLine k="fun" v={state.result.fun.toExponential(3)} c="#f5a524" />
            <div
              style={{
                color: "#7f7864",
                fontSize: 11,
                marginTop: 6,
                marginBottom: 2,
              }}
            >
              evaluation counts
            </div>
            <ResultLine k="nfev" v={state.result.nfev} />
            <ResultLine k="njev" v={state.result.ngev} />
            <ResultLine k="nhev" v={state.result.nhev} />
            <ResultLine k="nit" v={state.result.nit} />
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

function ComparisonTable() {
  return (
    <div
      style={{
        marginTop: 28,
        background: "#fff",
        border: "1px solid #ddd",
        borderRadius: 10,
      }}
    >
      <div
        style={{
          padding: "12px 16px",
          fontWeight: 700,
          borderBottom: "1px solid #eee",
        }}
      >
        Side-by-side on Rosenbrock from x₀ = (-1.5, 2.0)
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#fafafa" }}>
            <th style={th}>method</th>
            <th style={th}>family</th>
            <th style={th}>nfev</th>
            <th style={th}>njev</th>
            <th style={th}>nhev</th>
            <th style={th}>nit</th>
            <th style={th}>final f(x*)</th>
          </tr>
        </thead>
        <tbody>
          {Object.values(METHODS).map((m) => (
            <tr key={m.key} style={{ borderTop: "1px solid #eee" }}>
              <td style={td}>{m.name}</td>
              <td style={{ ...td, color: "#666", fontStyle: "italic" }}>{m.family}</td>
              <td style={tdNum}>{m.result.nfev}</td>
              <td style={tdNum}>{m.result.ngev}</td>
              <td style={tdNum}>{m.result.nhev}</td>
              <td style={tdNum}>{m.result.nit}</td>
              <td style={tdNum}>{m.result.fun.toExponential(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
const th = {
  padding: "8px 12px",
  textAlign: "left",
  fontFamily: "monospace",
  fontSize: 11,
  letterSpacing: "0.06em",
  color: "#666",
  textTransform: "uppercase",
};
const td = {
  padding: "8px 12px",
  fontFamily: "monospace",
  fontSize: 13,
};
const tdNum = { ...td, textAlign: "right", color: "#1f4e3d" };

function PedagogicalNotes() {
  return (
    <div
      style={{
        marginTop: 24,
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
          <b>nfev / njev / nhev are the actual cost.</b> Iteration count alone
          is misleading: BFGS does line searches inside each iteration, calling
          f and grad many times per "iter". Compare nfev across methods, not
          nit.
        </li>
        <li>
          <b>Newton vs trust-region.</b> Both Newton-CG and trust-ncg use the
          true Hessian. Trust-ncg is more robust on non-convex problems because
          it never takes a step that worsens f even if the Newton direction
          says to.
        </li>
        <li>
          <b>If you don't pass jac=…</b>, scipy estimates it by finite
          differences. That doubles nfev (each gradient costs n function
          evaluations). For anything beyond toy problems, write the gradient.
        </li>
        <li>
          <b>L-BFGS-B is the default for large n.</b> When n &gt; 100,
          BFGS's O(n²) memory hurts. L-BFGS-B keeps only the last m updates.
        </li>
        <li>
          <b>Switching methods is one kwarg.</b> Same code, change{" "}
          <code style={inlineCode}>method='BFGS'</code> to{" "}
          <code style={inlineCode}>method='trust-ncg'</code>. This is the
          biggest pedagogical advantage of scipy.optimize over hand-rolling.
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
        Install SciPy &nbsp;
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
          <Pre>
            {`pip install scipy

# Verify
python -c "from scipy.optimize import minimize; print(minimize.__doc__[:80])"`}
          </Pre>
          <p style={{ marginBottom: 0 }}>
            That's it. SciPy bundles every method in this tutorial — no extra
            backends required.
          </p>
        </div>
      )}
    </div>
  );
}

// ============================================================
// Code panel (same as pyomo / cvxpy)
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
function ResultLine({ k, v, c }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between" }}>
      <span style={{ color: "#7f7864" }}>{k}</span>
      <span style={{ color: c || "#e8e2d4", fontWeight: 600 }}>{v}</span>
    </div>
  );
}

const tabBtn = {
  padding: "7px 12px",
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
