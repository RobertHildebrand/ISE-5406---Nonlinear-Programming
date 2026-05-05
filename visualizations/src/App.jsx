import React, { useState } from 'react';
import NeuralNetViz from '../nn_viz.jsx';
import OptimDemo from '../optim_demo.jsx';
import StepAnatomy from '../step_anatomy.jsx';
import SVMDemo from '../svm_demo.jsx';
import Optim3DDemo from '../optim3d_demo.jsx';
import KKTDemo from '../kkt_demo.jsx';
import NewtonDemo from '../newton_demo.jsx';
import LineSearchDemo from '../linesearch_demo.jsx';
import BarrierDemo from '../barrier_demo.jsx';
import PerceptronTutorial from '../perceptron_tutorial.jsx';
import SGDAdamTutorial from '../sgd_adam_tutorial.jsx';
import NNDemo from '../nn_demo.jsx';
import PyomoTutorial from '../pyomo_tutorial.jsx';
import CVXPYTutorial from '../cvxpy_tutorial.jsx';
import ScipyTutorial from '../scipy_tutorial.jsx';
import ScipTutorial from '../scip_tutorial.jsx';
import GurobiTutorial from '../gurobi_tutorial.jsx';
import GoogleToolsTutorial from '../google_tools_tutorial.jsx';
import FrankWolfeDemo from '../frank_wolfe_demo.jsx';
import SosDemo from '../sos_demo.jsx';
import AlgebraicOptDemo from '../algebraic_opt_demo.jsx';
import BranchBoundDemo from '../branch_bound_demo.jsx';
import FistaDemo from '../fista_demo.jsx';
import GanttDemo from '../gantt_demo.jsx';
import DualitySensitivityDemo from '../duality_sensitivity_demo.jsx';
import SimplexTableauDemo from '../simplex_tableau_demo.jsx';
import HeuristicsDemo from '../heuristics_demo.jsx';
import DecompositionDemo from '../decomposition_demo.jsx';
import PyTorchTutorial from '../pytorch_tutorial.jsx';
import MLCompare from '../ml_compare.jsx';
import NLPApplications from '../nlp_applications.jsx';
import ISEApplications from '../ise_applications.jsx';

// ---------- Categories (display order) ----------
const CATEGORIES = [
  {
    id: 'foundations',
    title: 'Foundations',
    blurb: 'Build intuition for what one optimization step actually does.',
    accent: '#1f4e3d',
  },
  {
    id: 'first-order',
    title: 'First-Order Methods',
    blurb: 'Gradient descent, momentum, Nesterov, RMSProp, Adam — and how to size their steps.',
    accent: '#1f4e3d',
  },
  {
    id: 'second-order',
    title: 'Second-Order Methods',
    blurb: "Newton's method and the local quadratic model.",
    accent: '#1f4e3d',
  },
  {
    id: 'constrained',
    title: 'Constrained Optimization',
    blurb: 'Feasible regions, active sets, KKT conditions, and interior-point methods.',
    accent: '#d4a017',
  },
  {
    id: 'classification',
    title: 'Classification & Kernels',
    blurb: 'Perceptron and SVM with linear, polynomial, and RBF kernels.',
    accent: '#1f4e3d',
  },
  {
    id: 'neural-nets',
    title: 'Neural Networks',
    blurb: 'Forward and backward pass, training, learned vs target.',
    accent: '#1f4e3d',
  },
  {
    id: 'advanced',
    title: 'Advanced Topics',
    blurb: 'Conditional gradient, polynomial / SOS optimization, algebraic methods.',
    accent: '#7a3da0',
  },
  {
    id: 'tutorials',
    title: 'In-Class Code Steppers',
    blurb: 'Walk through algorithm code line-by-line for live lecture demonstration.',
    accent: '#c8311c',
  },
];

const DEMOS = [
  // ── Foundations ────────────────────────────────────────────────
  {
    id: 'step',
    category: 'foundations',
    title: 'Anatomy of a Step',
    description:
      'One optimization iteration in slow motion: gradient, descent direction, step size, and the resulting move.',
    Component: StepAnatomy,
  },

  // ── First-order methods ────────────────────────────────────────
  {
    id: 'optim',
    category: 'first-order',
    title: 'First-Order Methods (2D)',
    description:
      'Compare GD, momentum, Nesterov, RMSProp, Adam on 2D test functions. Watch trajectories on the contour plot.',
    Component: OptimDemo,
  },
  {
    id: 'optim3d',
    category: 'first-order',
    title: 'First-Order Methods (3D)',
    description:
      'Same methods, but rotate the loss surface and watch trajectories descend in 3D. Tunable start point.',
    Component: Optim3DDemo,
  },
  {
    id: 'linesearch',
    category: 'first-order',
    title: 'Line Search (Armijo / Wolfe)',
    description:
      'Slide α along φ(α) = f(x + α·d). See which step sizes satisfy Armijo and strong-Wolfe; animate backtracking.',
    Component: LineSearchDemo,
  },

  // ── Second-order ───────────────────────────────────────────────
  {
    id: 'newton',
    category: 'second-order',
    title: 'Newton vs. First-Order',
    description:
      "Newton lands at the local quadratic model's minimum. Compare against gradient descent on the same surface.",
    Component: NewtonDemo,
  },

  // ── Constrained ────────────────────────────────────────────────
  {
    id: 'kkt',
    category: 'constrained',
    title: 'Constrained Optimization & KKT',
    description:
      'Feasible region, active constraints, Lagrange multipliers, and the KKT balance ∇f + Σ λᵢ ∇gᵢ = 0.',
    Component: KKTDemo,
  },
  {
    id: 'barrier',
    category: 'constrained',
    title: 'Interior Point / Log Barrier',
    description:
      'Sweep the barrier parameter t through LPs, SOCPs, and SDPs. Watch the central path bend through different feasible regions.',
    Component: BarrierDemo,
  },

  // ── Classification & kernels ───────────────────────────────────
  {
    id: 'svm',
    category: 'classification',
    title: 'Perceptron & Kernel SVM',
    description:
      'Step through the perceptron algorithm, then train kernel SVMs (linear / polynomial / RBF) with adjustable noise and label flips.',
    Component: SVMDemo,
  },

  // ── Neural networks ────────────────────────────────────────────
  {
    id: 'nn-demo',
    category: 'neural-nets',
    title: 'Neural Networks & Backpropagation',
    description:
      'Three-part walkthrough: forward pass with sliders, training with live loss curve and gradient halos, learned-vs-target heatmaps.',
    Component: NNDemo,
  },
  {
    id: 'nn',
    category: 'neural-nets',
    title: '8×8 Digit Classifier',
    description:
      'A small NN classifies hand-drawn 8×8 digit patterns. Watch every neuron activate during forward and backward propagation.',
    Component: NeuralNetViz,
  },

  // ── Advanced topics ────────────────────────────────────────────
  {
    id: 'simplex-tableau',
    category: 'advanced',
    title: 'Interactive Simplex Tableau',
    description:
      "Click-to-pivot tableau with practice mode. Click a column to enter, see the ratio test, click a row to pivot. Feasible-region plot tracks the current vertex. Sensitivity + duality panel appears when optimal.",
    Component: SimplexTableauDemo,
  },
  {
    id: 'duality-sensitivity',
    category: 'advanced',
    title: 'LP Duality & Sensitivity Analysis',
    description:
      "Slide the constraint right-hand-sides; watch the feasible region deform, the optimum vertex jump, and the shadow prices update live. Dual problem displayed alongside the primal. Sensitivity ranges shown as bars.",
    Component: DualitySensitivityDemo,
  },
  {
    id: 'decomposition',
    category: 'advanced',
    title: 'IP Decomposition: Benders, Dantzig-Wolfe, Lagrangian',
    description:
      "Three classical decompositions side-by-side. Master/subproblem split, bound progression chart, per-iteration table, and pseudocode for each method.",
    Component: DecompositionDemo,
  },
  {
    id: 'heuristics',
    category: 'advanced',
    title: 'Heuristics & Metaheuristics (TSP + Knapsack)',
    description:
      "Animated 2-opt local search and simulated annealing on a 15-city TSP, plus a genetic algorithm on 0-1 knapsack. Watch the tour rearrange itself in real time.",
    Component: HeuristicsDemo,
  },
  {
    id: 'branch-bound',
    category: 'advanced',
    title: 'Branch-and-Bound Tree Explorer',
    description:
      "Watch a small MILP get solved one node at a time. Click any node to see its LP relaxation; the tree panel shows branching decisions, fathoming, and the primal/dual gap closing.",
    Component: BranchBoundDemo,
  },
  {
    id: 'fista',
    category: 'advanced',
    title: 'Proximal Gradient & FISTA',
    description:
      "Lasso-style 2D problem. ISTA vs FISTA side-by-side, with the soft-thresholding step explicit and the Nesterov-momentum extrapolation drawn as a separate point on the contour plot.",
    Component: FistaDemo,
  },
  {
    id: 'frank-wolfe',
    category: 'advanced',
    title: 'Frank-Wolfe with Active Vertex Tracking',
    description:
      "Conditional gradient on a polytope. Watch the active vertex set grow and the iterate jump to the closest point in conv(S) at each step. Vanilla FW vs fully-corrective FW side-by-side. Drag the target.",
    Component: FrankWolfeDemo,
  },
  {
    id: 'sos',
    category: 'advanced',
    title: 'Polynomial Optimization & SOS',
    description:
      "Lasserre's hierarchy in action. Slide the SOS order to watch the lower bound λ_d tighten on a univariate polynomial. The moment-matrix structure plus a CVXPY code stepper.",
    Component: SosDemo,
  },
  {
    id: 'algebraic-opt',
    category: 'advanced',
    title: 'Algebraic Optimization',
    description:
      "Lagrange multipliers on an algebraic curve x⁴ + y⁴ = 1. Plot shows the objective contours, constraint curve, and all four real critical points. Three computational paths: sympy, homotopy continuation, SOS lifting.",
    Component: AlgebraicOptDemo,
  },
  {
    id: 'gantt',
    category: 'advanced',
    title: 'Job-Shop Gantt Chart (CP-SAT)',
    description:
      "The numerical answer from the CP-SAT job-shop demo turned into a picture. Three jobs, three machines, makespan 11. Toggle between machine view and job view; hover any block to highlight its job and machine neighbors.",
    Component: GanttDemo,
  },

  // ── In-class tutorials ─────────────────────────────────────────
  {
    id: 'perceptron-tutorial',
    category: 'tutorials',
    title: 'Perceptron — Code Walkthrough',
    description:
      'Python pseudocode on the left with the active line highlighted; scatter plot on the right updates each step. For live in-class demonstration.',
    Component: PerceptronTutorial,
  },
  {
    id: 'sgd-adam-tutorial',
    category: 'tutorials',
    title: 'SGD & Adam — Code Stepper',
    description:
      'Step through SGD, momentum, and Adam line-by-line on a logistic-regression problem. PyTorch comparison at the bottom.',
    Component: SGDAdamTutorial,
  },
  {
    id: 'pyomo-tutorial',
    category: 'tutorials',
    title: 'Pyomo + IPOPT — Code Stepper',
    description:
      'Build three NLPs (constrained QP, disk-projection, Markowitz portfolio) line-by-line in Pyomo and watch IPOPT solve them. Includes install instructions.',
    Component: PyomoTutorial,
  },
  {
    id: 'cvxpy-tutorial',
    category: 'tutorials',
    title: 'CVXPY — Code Stepper',
    description:
      'Step through LP, QP, SOCP, and SDP examples in CVXPY. Watch how the same Variable / Constraint / Problem pattern adapts as the problem class changes.',
    Component: CVXPYTutorial,
  },
  {
    id: 'scipy-tutorial',
    category: 'tutorials',
    title: 'scipy.optimize — Code Stepper',
    description:
      'Rosenbrock from x₀ = (-1.5, 2.0). Compare derivative-free, first-order, and second-order methods — Nelder-Mead, BFGS, Newton-CG, trust-ncg, and friends. Side-by-side nfev/njev/nhev/nit table.',
    Component: ScipyTutorial,
  },
  {
    id: 'scip-tutorial',
    category: 'tutorials',
    title: 'SCIP / PySCIPOpt — Code Stepper',
    description:
      'MILP and MINLP examples (knapsack, set cover, integer rectangle, Bienstock\'s nonconvex trap). Step through PySCIPOpt code, watch primal/dual bounds, and read SCIP\'s log column-by-column.',
    Component: ScipTutorial,
  },
  {
    id: 'gurobi-tutorial',
    category: 'tutorials',
    title: 'Gurobi (gurobipy) — Code Stepper',
    description:
      'Industrial MIP solver: production LP, facility-location MILP, cardinality-constrained MIQP, and Bienstock\'s nonconvex problem with NonConvex=2. Includes a column-by-column reader for Gurobi\'s log.',
    Component: GurobiTutorial,
  },
  {
    id: 'google-tools-tutorial',
    category: 'tutorials',
    title: 'Google OR-Tools — Code Stepper',
    description:
      'Three Google solvers: PDLP (matrix-free LP for million-variable problems), specialized network-flow (min-cost-flow / max-flow / assignment), and CP-SAT (job-shop scheduling). Each with column-by-column log reader.',
    Component: GoogleToolsTutorial,
  },
  {
    id: 'pytorch-tutorial',
    category: 'tutorials',
    title: 'PyTorch — Code Stepper',
    description:
      'Three classic ML problems with the same training-loop skeleton: linear regression, binary classification, and a small MLP on nonlinear data. State panel renders the model architecture and loss curve.',
    Component: PyTorchTutorial,
  },
  {
    id: 'ml-compare',
    category: 'tutorials',
    title: 'ML — Same Problem, Three Ways',
    description:
      'Same binary-classification problem solved three ways: scikit-learn (high-level), PyTorch (mid-level), and from scratch with NumPy. Side-by-side code, pros/cons, and a comparison table.',
    Component: MLCompare,
  },
  {
    id: 'nlp-applications',
    category: 'tutorials',
    title: 'NLP Applications — Code Stepper',
    description:
      'Four NLP applications across four domains: rocket trajectory (optimal control), cantilever beam (engineering), robust regression (statistics), open-top container (geometric programming). All in Pyomo + IPOPT.',
    Component: NLPApplications,
  },
  {
    id: 'ise-applications',
    category: 'tutorials',
    title: 'ISE Applications — Code Stepper',
    description:
      'Four classics ISE students should recognize: EOQ with backordering (inventory), Weber facility location (logistics), multi-period production smoothing (production), multi-product newsvendor with budget (inventory under uncertainty).',
    Component: ISEApplications,
  },
];

// ---------- Subcomponents ----------
function DemoCard({ demo, onClick }) {
  const isTutorial = demo.category === 'tutorials';
  return (
    <button onClick={onClick} style={cardStyle(isTutorial)}>
      <div style={{ fontSize: 17, fontWeight: 700, marginBottom: 6, color: '#111' }}>
        {demo.title}
      </div>
      <div style={{ fontSize: 13.5, color: '#555', lineHeight: 1.45 }}>
        {demo.description}
      </div>
      {isTutorial && (
        <div
          style={{
            marginTop: 10,
            display: 'inline-block',
            padding: '2px 8px',
            background: '#fdecea',
            color: '#c8311c',
            borderRadius: 4,
            fontFamily: 'monospace',
            fontSize: 10.5,
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            fontWeight: 600,
          }}
        >
          for instructor demo
        </div>
      )}
    </button>
  );
}

function cardStyle(isTutorial) {
  return {
    display: 'block',
    textAlign: 'left',
    padding: '18px 20px',
    border: '1px solid #e3e3e3',
    borderLeft: isTutorial ? '4px solid #c8311c' : '1px solid #e3e3e3',
    borderRadius: 10,
    background: '#fff',
    cursor: 'pointer',
    textDecoration: 'none',
    color: 'inherit',
    transition: 'all 0.12s',
    boxShadow: '0 1px 2px rgba(0,0,0,0.03)',
    width: '100%',
    fontFamily: 'inherit',
  };
}

function CategorySection({ category, demos, onLaunch }) {
  if (demos.length === 0) return null;
  return (
    <section style={{ marginBottom: 36 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: 12,
          paddingBottom: 8,
          marginBottom: 16,
          borderBottom: `2px solid ${category.accent}`,
        }}
      >
        <h2
          style={{
            fontSize: 20,
            fontWeight: 800,
            color: '#111',
            margin: 0,
          }}
        >
          {category.title}
        </h2>
        <span
          style={{
            fontFamily: 'monospace',
            fontSize: 11,
            color: '#888',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}
        >
          {demos.length} demo{demos.length === 1 ? '' : 's'}
        </span>
      </div>
      <p style={{ color: '#666', marginTop: -6, marginBottom: 14, fontSize: 14 }}>
        {category.blurb}
      </p>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 14,
        }}
      >
        {demos.map((d) => (
          <DemoCard key={d.id} demo={d} onClick={() => onLaunch(d.id)} />
        ))}
      </div>
    </section>
  );
}

// ---------- Top-level ----------
export default function App() {
  const [active, setActive] = useState(null);

  if (active) {
    const demo = DEMOS.find((d) => d.id === active);
    const Component = demo.Component;
    return (
      <div>
        <div
          style={{
            position: 'sticky',
            top: 0,
            zIndex: 50,
            background: '#111',
            color: '#fff',
            padding: '8px 16px',
            display: 'flex',
            alignItems: 'center',
            gap: 16,
            borderBottom: '1px solid #333',
          }}
        >
          <button
            onClick={() => setActive(null)}
            style={{
              background: '#fff',
              color: '#111',
              border: 'none',
              padding: '6px 12px',
              borderRadius: 6,
              cursor: 'pointer',
              fontWeight: 600,
            }}
          >
            ← Back
          </button>
          <span style={{ fontWeight: 600 }}>{demo.title}</span>
        </div>
        <Component />
      </div>
    );
  }

  // Group demos by category, preserving DEMOS order within each.
  const byCategory = Object.fromEntries(CATEGORIES.map((c) => [c.id, []]));
  for (const d of DEMOS) {
    if (byCategory[d.category]) byCategory[d.category].push(d);
  }

  return (
    <div style={{ minHeight: '100vh', background: '#f7f5f0' }}>
      <div
        style={{
          maxWidth: 1100,
          margin: '0 auto',
          padding: '56px 24px 80px',
        }}
      >
        {/* Header */}
        <header
          style={{
            paddingBottom: 22,
            marginBottom: 36,
            borderBottom: '1px solid #d4cfc4',
          }}
        >
          <div
            style={{
              fontFamily: 'monospace',
              fontSize: 11,
              letterSpacing: '0.18em',
              textTransform: 'uppercase',
              color: '#666',
              marginBottom: 10,
            }}
          >
            ISE 5406 · Nonlinear Programming · Spring 2026
          </div>
          <h1
            style={{
              fontSize: 38,
              fontWeight: 800,
              marginBottom: 8,
              color: '#111',
              lineHeight: 1.05,
            }}
          >
            Interactive Visualizations
          </h1>
          <p
            style={{
              color: '#555',
              fontSize: 16,
              maxWidth: 720,
              lineHeight: 1.5,
            }}
          >
            A collection of interactive demos for the course. Click any card
            to launch its demo — every one runs in the browser with no install.
            Demos marked <i>"for instructor demo"</i> are designed to walk
            through algorithm code line by line during lecture.
          </p>
          <div
            style={{
              marginTop: 14,
              fontFamily: 'monospace',
              fontSize: 12,
              color: '#888',
            }}
          >
            {DEMOS.length} demos · {CATEGORIES.length} sections
          </div>
        </header>

        {/* Sections */}
        {CATEGORIES.map((c) => (
          <CategorySection
            key={c.id}
            category={c}
            demos={byCategory[c.id]}
            onLaunch={setActive}
          />
        ))}

        {/* Footer */}
        <footer
          style={{
            marginTop: 60,
            paddingTop: 22,
            borderTop: '1px solid #d4cfc4',
            color: '#888',
            fontSize: 12,
            fontFamily: 'monospace',
          }}
        >
          source ·{' '}
          <a
            href="https://github.com/RobertHildebrand/ISE-5406---Nonlinear-Programming"
            style={{ color: '#666' }}
          >
            github.com/RobertHildebrand/ISE-5406---Nonlinear-Programming
          </a>
        </footer>
      </div>
    </div>
  );
}
