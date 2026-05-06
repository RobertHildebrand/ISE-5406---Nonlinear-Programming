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
import LpSolversDemo from '../lp_solvers_demo.jsx';
import NetworkFlowDemo from '../network_flow_demo.jsx';
import RlDemo from '../rl_demo.jsx';
import RlhfDemo from '../rlhf_demo.jsx';
import ExcelSolverDemo from '../excel_solver_demo.jsx';
import PyTorchTutorial from '../pytorch_tutorial.jsx';
import MLCompare from '../ml_compare.jsx';
import NLPApplications from '../nlp_applications.jsx';
import ISEApplications from '../ise_applications.jsx';
import AmplpyTutorial from '../amplpy_tutorial.jsx';
import TableauPivoterDemo from '../tableau_pivoter_demo.jsx';
import DualConstructionDemo from '../dual_construction_demo.jsx';
import SensitivityWalkthroughDemo from '../sensitivity_walkthrough_demo.jsx';
import PythonBasicsTutorial from '../python_basics_tutorial.jsx';
import ObjectiveSliderDemo from '../objective_slider_demo.jsx';
import TwoPhaseSimplexDemo from '../two_phase_simplex_demo.jsx';
import DualSimplexDemo from '../dual_simplex_demo.jsx';
import ModelingIntroduction from '../modeling_introduction.jsx';

// ---------- Categories (display order: LP → IP → NLP → ML → Tutorials) ----------
const CATEGORIES = [
  {
    id: 'lp',
    title: 'Linear Programming',
    blurb: 'Simplex, duality, sensitivity. The algorithmic foundation for most of OR — every IP and NLP solver leans on an LP at some point.',
    accent: '#0b3da0',
  },
  {
    id: 'ip',
    title: 'Integer Programming',
    blurb: 'Branch-and-bound, decomposition, scheduling. Combinatorial structure on top of LP — and the algorithms that exploit it.',
    accent: '#7a3da0',
  },
  {
    id: 'nlp',
    title: 'Nonlinear Programming',
    blurb: 'Continuous optimization with smooth nonlinear objectives or constraints. From a single gradient step to interior-point and conditional-gradient methods.',
    accent: '#1f4e3d',
  },
  {
    id: 'ml',
    title: 'Machine Learning, Networks, & RL',
    blurb: 'Supervised classification, neural networks, reinforcement learning. Optimization shows up everywhere here — these are the consumer applications.',
    accent: '#d4a017',
  },
  {
    id: 'tutorials',
    title: 'Getting Started',
    blurb: 'Setup and reference for students new to Python or rusty on the scientific stack.',
    accent: '#c8311c',
  },
];

const DEMOS = [
  // ── Foundations ────────────────────────────────────────────────
  {
    id: 'step',
    category: 'nlp',
    title: 'Anatomy of a Step',
    description:
      'One optimization iteration in slow motion: gradient, descent direction, step size, and the resulting move.',
    Component: StepAnatomy,
  },

  // ── First-order methods ────────────────────────────────────────
  {
    id: 'optim',
    category: 'nlp',
    title: 'First-Order Methods (2D)',
    description:
      'Compare GD, momentum, Nesterov, RMSProp, Adam on 2D test functions. Watch trajectories on the contour plot.',
    Component: OptimDemo,
  },
  {
    id: 'optim3d',
    category: 'nlp',
    title: 'First-Order Methods (3D)',
    description:
      'Same methods, but rotate the loss surface and watch trajectories descend in 3D. Tunable start point.',
    Component: Optim3DDemo,
  },
  {
    id: 'linesearch',
    category: 'nlp',
    title: 'Line Search (Armijo / Wolfe)',
    description:
      'Slide α along φ(α) = f(x + α·d). See which step sizes satisfy Armijo and strong-Wolfe; animate backtracking.',
    Component: LineSearchDemo,
  },

  // ── Second-order ───────────────────────────────────────────────
  {
    id: 'newton',
    category: 'nlp',
    title: 'Newton vs. First-Order',
    description:
      "Newton lands at the local quadratic model's minimum. Compare against gradient descent on the same surface.",
    Component: NewtonDemo,
  },

  // ── Constrained ────────────────────────────────────────────────
  {
    id: 'kkt',
    category: 'nlp',
    title: 'Constrained Optimization & KKT',
    description:
      'Feasible region, active constraints, Lagrange multipliers, and the KKT balance ∇f + Σ λᵢ ∇gᵢ = 0.',
    Component: KKTDemo,
  },
  {
    id: 'barrier',
    category: 'nlp',
    title: 'Interior Point / Log Barrier',
    description:
      'Sweep the barrier parameter t through LPs, SOCPs, and SDPs. Watch the central path bend through different feasible regions.',
    Component: BarrierDemo,
  },

  // ── Classification & kernels ───────────────────────────────────
  {
    id: 'svm',
    category: 'ml',
    title: 'Perceptron & Kernel SVM',
    description:
      'Step through the perceptron algorithm, then train kernel SVMs (linear / polynomial / RBF) with adjustable noise and label flips.',
    Component: SVMDemo,
  },

  // ── Neural networks ────────────────────────────────────────────
  {
    id: 'nn-demo',
    category: 'ml',
    title: 'Neural Networks & Backpropagation',
    description:
      'Three-part walkthrough: forward pass with sliders, training with live loss curve and gradient halos, learned-vs-target heatmaps.',
    Component: NNDemo,
  },
  {
    id: 'nn',
    category: 'ml',
    title: '8×8 Digit Classifier',
    description:
      'A small NN classifies hand-drawn 8×8 digit patterns. Watch every neuron activate during forward and backward propagation.',
    Component: NeuralNetViz,
  },

  // ── Advanced topics ────────────────────────────────────────────
  {
    id: 'objective-slider',
    category: 'lp',
    title: 'Objective-Level Slider — LP / IP / Convex / Nonconvex',
    description:
      'Slide the objective value z and watch the level set (line, lattice, circle, or multimodal contour) move across the feasible region. Four tabs compare LP, IP, convex NLP, and nonconvex NLP side by side. The optimum is the extremal z that still touches feasibility.',
    Component: ObjectiveSliderDemo,
  },
  {
    id: 'lp-solvers',
    category: 'lp',
    title: 'LP Modelers — PuLP / AMPL / Gurobi / CPLEX',
    description:
      "Same production LP, four languages. Tabs to switch; identical numerical answer. Includes API cheat sheet mapping common operations across the four modelers and per-language install notes.",
    Component: LpSolversDemo,
  },
  {
    id: 'excel-solver',
    category: 'lp',
    title: 'Excel Solver — Spreadsheet Optimization',
    description:
      "Three problem types laid out as Excel sheets: LP, IP (knapsack), and NLP (curve fit). Each with formula highlighting, Solver dialog mockup, results, plus a SUMPRODUCT/IF/INDEX cheat sheet.",
    Component: ExcelSolverDemo,
  },
  {
    id: 'network-flow',
    category: 'lp',
    title: 'Network Flow — NetworkX & OR-Tools',
    description:
      "One graph, three problems (shortest path, max flow, min-cost flow), two libraries side-by-side. Flow values overlaid on the graph in red; saturation indicated.",
    Component: NetworkFlowDemo,
  },
  {
    id: 'rl',
    category: 'ml',
    title: 'RL — Gridworld (Value / Policy / Q-Learning)',
    description:
      "5×5 gridworld with goal, lava, and step penalty. Toggle between value iteration, policy iteration, and tabular Q-learning. Watch V-values fill in as a heatmap and the greedy policy crystallize as arrows.",
    Component: RlDemo,
  },
  {
    id: 'rlhf',
    category: 'ml',
    title: 'RLHF — Reward Modeling + Policy Update',
    description:
      "1-D toy of the RLHF pipeline used to align modern LLMs. Sample preference pairs, fit a Bradley-Terry reward model, take a KL-regularized PPO step, watch the policy mean march toward the true x⋆.",
    Component: RlhfDemo,
  },
  {
    id: 'simplex-tableau',
    category: 'lp',
    title: 'Interactive Simplex Tableau',
    description:
      "Click-to-pivot tableau with practice mode. Click a column to enter, see the ratio test, click a row to pivot. Feasible-region plot tracks the current vertex. Each pivot now also writes out the elementary row operations in algebraic form.",
    Component: SimplexTableauDemo,
  },
  {
    id: 'two-phase-simplex',
    category: 'lp',
    title: 'Two-Phase Simplex',
    description:
      'When the LP has ≥ or = constraints, you can\'t just slap on slacks and start at the origin. Add artificial variables, minimize their sum (Phase 1), then switch to the original objective (Phase 2). Three example LPs including an infeasible one — watch artificials get driven out of the basis (or fail to).',
    Component: TwoPhaseSimplexDemo,
  },
  {
    id: 'dual-simplex',
    category: 'lp',
    title: 'Dual Simplex',
    description:
      'Maintains DUAL feasibility, chases primal feasibility — pick the leaving row first (negative RHS), then the entering column via dual ratio test. Used in MIP cuts, RHS perturbations, and post-modification re-optimization. Three preset tableaux including a Gomory-cut scenario, with side-by-side comparison of primal vs dual pivot rules.',
    Component: DualSimplexDemo,
  },
  {
    id: 'tableau-pivoter',
    category: 'lp',
    title: 'Tableau Pivoter — Step-by-Step Row Ops',
    description:
      "Type your own LP, pick any pivot row and column, watch the elementary row operations play out one at a time with the algebra written underneath. The Gauss-Jordan mechanics behind every simplex iteration, exposed.",
    Component: TableauPivoterDemo,
  },
  {
    id: 'dual-construction',
    category: 'lp',
    title: 'Primal → Dual Construction',
    description:
      "Build the dual of a general LP one stage at a time: transpose A, swap b ↔ c, flip objective sense, then apply per-constraint and per-variable sign rules. Toggle constraint types (≤ / = / ≥) and variable signs to see the dual reshape live.",
    Component: DualConstructionDemo,
  },
  {
    id: 'sensitivity-walkthrough',
    category: 'lp',
    title: 'Sensitivity Analysis — Step-by-Step Derivation',
    description:
      "Derive allowable ranges for c_j (basic & non-basic separately) and b_i from the optimal tableau. Each ratio test is written out in algebra; the binding bounds are highlighted. Mini-simplex runs internally on any LP you type.",
    Component: SensitivityWalkthroughDemo,
  },
  {
    id: 'duality-sensitivity',
    category: 'lp',
    title: 'LP Duality & Sensitivity Analysis',
    description:
      "Slide the constraint right-hand-sides; watch the feasible region deform, the optimum vertex jump, and the shadow prices update live. Dual problem displayed alongside the primal. Sensitivity ranges shown as bars.",
    Component: DualitySensitivityDemo,
  },
  {
    id: 'decomposition',
    category: 'ip',
    title: 'IP Decomposition: Benders, Dantzig-Wolfe, Lagrangian',
    description:
      "Three classical decompositions side-by-side. Master/subproblem split, bound progression chart, per-iteration table, and pseudocode for each method.",
    Component: DecompositionDemo,
  },
  {
    id: 'heuristics',
    category: 'ip',
    title: 'Heuristics & Metaheuristics (TSP + Knapsack)',
    description:
      "Animated 2-opt local search and simulated annealing on a 15-city TSP, plus a genetic algorithm on 0-1 knapsack. Watch the tour rearrange itself in real time.",
    Component: HeuristicsDemo,
  },
  {
    id: 'branch-bound',
    category: 'ip',
    title: 'Branch-and-Bound Tree Explorer',
    description:
      "Watch a small MILP get solved one node at a time. Click any node to see its LP relaxation; the tree panel shows branching decisions, fathoming, and the primal/dual gap closing.",
    Component: BranchBoundDemo,
  },
  {
    id: 'fista',
    category: 'nlp',
    title: 'Proximal Gradient & FISTA',
    description:
      "Lasso-style 2D problem. ISTA vs FISTA side-by-side, with the soft-thresholding step explicit and the Nesterov-momentum extrapolation drawn as a separate point on the contour plot.",
    Component: FistaDemo,
  },
  {
    id: 'frank-wolfe',
    category: 'nlp',
    title: 'Frank-Wolfe with Active Vertex Tracking',
    description:
      "Conditional gradient on a polytope. Watch the active vertex set grow and the iterate jump to the closest point in conv(S) at each step. Vanilla FW vs fully-corrective FW side-by-side. Drag the target.",
    Component: FrankWolfeDemo,
  },
  {
    id: 'sos',
    category: 'nlp',
    title: 'Polynomial Optimization & SOS',
    description:
      "Lasserre's hierarchy in action. Slide the SOS order to watch the lower bound λ_d tighten on a univariate polynomial. The moment-matrix structure plus a CVXPY code stepper.",
    Component: SosDemo,
  },
  {
    id: 'algebraic-opt',
    category: 'nlp',
    title: 'Algebraic Optimization',
    description:
      "Lagrange multipliers on an algebraic curve x⁴ + y⁴ = 1. Plot shows the objective contours, constraint curve, and all four real critical points. Three computational paths: sympy, homotopy continuation, SOS lifting.",
    Component: AlgebraicOptDemo,
  },
  {
    id: 'gantt',
    category: 'ip',
    title: 'Job-Shop Gantt Chart (CP-SAT)',
    description:
      "The numerical answer from the CP-SAT job-shop demo turned into a picture. Three jobs, three machines, makespan 11. Toggle between machine view and job view; hover any block to highlight its job and machine neighbors.",
    Component: GanttDemo,
  },

  // ── In-class tutorials ─────────────────────────────────────────
  {
    id: 'modeling-introduction',
    category: 'tutorials',
    title: 'Modeling Introduction — Translate a Problem into a Model',
    description:
      'Seven tabs walking through how to turn a real-world problem into an optimization model: the variables/constraints/objective trio, a fully worked bakery example, 12 common modeling patterns, variable types (incl. fixed-charge big-M), 10 pitfalls, the bridge to Pyomo / AMPL / gurobipy, and 4 click-to-reveal practice problems.',
    Component: ModelingIntroduction,
  },
  {
    id: 'python-basics',
    category: 'tutorials',
    title: 'Python Basics — Install, Packages, Syntax, Plotting',
    description:
      'New to Python or rusty? Walk through install paths (Anaconda, conda, pyenv, Colab), virtual environments, the scientific stack (numpy/pandas/matplotlib/scipy/geopandas), and core syntax (variables, loops, functions, comprehensions). Every snippet has a copy button.',
    Component: PythonBasicsTutorial,
  },
  {
    id: 'perceptron-tutorial',
    category: 'ml',
    title: 'Perceptron — Code Walkthrough',
    description:
      'Python pseudocode on the left with the active line highlighted; scatter plot on the right updates each step. For live in-class demonstration.',
    Component: PerceptronTutorial,
  },
  {
    id: 'sgd-adam-tutorial',
    category: 'nlp',
    title: 'SGD & Adam — Code Stepper',
    description:
      'Step through SGD, momentum, and Adam line-by-line on a logistic-regression problem. PyTorch comparison at the bottom.',
    Component: SGDAdamTutorial,
  },
  {
    id: 'pyomo-tutorial',
    category: 'nlp',
    title: 'Pyomo + IPOPT — Code Stepper',
    description:
      'Build three NLPs (constrained QP, disk-projection, Markowitz portfolio) line-by-line in Pyomo and watch IPOPT solve them. Includes install instructions.',
    Component: PyomoTutorial,
  },
  {
    id: 'amplpy-tutorial',
    category: 'lp',
    title: 'AMPL + amplpy — Code Stepper',
    description:
      'Mirror of the Pyomo + IPOPT walkthrough using amplpy. Three problems (QP, Markowitz portfolio, HS71). Toggle the .mod view to see the equivalent pure-AMPL syntax. Includes a Pyomo translation cheat sheet.',
    Component: AmplpyTutorial,
  },
  {
    id: 'cvxpy-tutorial',
    category: 'nlp',
    title: 'CVXPY — Code Stepper',
    description:
      'Step through LP, QP, SOCP, and SDP examples in CVXPY. Watch how the same Variable / Constraint / Problem pattern adapts as the problem class changes.',
    Component: CVXPYTutorial,
  },
  {
    id: 'scipy-tutorial',
    category: 'nlp',
    title: 'scipy.optimize — Code Stepper',
    description:
      'Rosenbrock from x₀ = (-1.5, 2.0). Compare derivative-free, first-order, and second-order methods — Nelder-Mead, BFGS, Newton-CG, trust-ncg, and friends. Side-by-side nfev/njev/nhev/nit table.',
    Component: ScipyTutorial,
  },
  {
    id: 'scip-tutorial',
    category: 'ip',
    title: 'SCIP / PySCIPOpt — Code Stepper',
    description:
      'MILP and MINLP examples (knapsack, set cover, integer rectangle, Bienstock\'s nonconvex trap). Step through PySCIPOpt code, watch primal/dual bounds, and read SCIP\'s log column-by-column.',
    Component: ScipTutorial,
  },
  {
    id: 'gurobi-tutorial',
    category: 'ip',
    title: 'Gurobi (gurobipy) — Code Stepper',
    description:
      'Industrial MIP solver: production LP, facility-location MILP, cardinality-constrained MIQP, and Bienstock\'s nonconvex problem with NonConvex=2. Includes a column-by-column reader for Gurobi\'s log.',
    Component: GurobiTutorial,
  },
  {
    id: 'google-tools-tutorial',
    category: 'ip',
    title: 'Google OR-Tools — Code Stepper',
    description:
      'Three Google solvers: PDLP (matrix-free LP for million-variable problems), specialized network-flow (min-cost-flow / max-flow / assignment), and CP-SAT (job-shop scheduling). Each with column-by-column log reader.',
    Component: GoogleToolsTutorial,
  },
  {
    id: 'pytorch-tutorial',
    category: 'ml',
    title: 'PyTorch — Code Stepper',
    description:
      'Three classic ML problems with the same training-loop skeleton: linear regression, binary classification, and a small MLP on nonlinear data. State panel renders the model architecture and loss curve.',
    Component: PyTorchTutorial,
  },
  {
    id: 'ml-compare',
    category: 'ml',
    title: 'ML — Same Problem, Three Ways',
    description:
      'Same binary-classification problem solved three ways: scikit-learn (high-level), PyTorch (mid-level), and from scratch with NumPy. Side-by-side code, pros/cons, and a comparison table.',
    Component: MLCompare,
  },
  {
    id: 'nlp-applications',
    category: 'nlp',
    title: 'NLP Applications — Code Stepper',
    description:
      'Four NLP applications across four domains: rocket trajectory (optimal control), cantilever beam (engineering), robust regression (statistics), open-top container (geometric programming). All in Pyomo + IPOPT.',
    Component: NLPApplications,
  },
  {
    id: 'ise-applications',
    category: 'nlp',
    title: 'ISE Applications — Code Stepper',
    description:
      'Four classics ISE students should recognize: EOQ with backordering (inventory), Weber facility location (logistics), multi-period production smoothing (production), multi-product newsvendor with budget (inventory under uncertainty).',
    Component: ISEApplications,
  },
];

// ---------- Subcomponents ----------
function DemoCard({ demo, onClick }) {
  return (
    <button onClick={onClick} style={cardStyle()}>
      <div style={{ fontSize: 17, fontWeight: 700, marginBottom: 6, color: '#111' }}>
        {demo.title}
      </div>
      <div style={{ fontSize: 13.5, color: '#555', lineHeight: 1.45 }}>
        {demo.description}
      </div>
    </button>
  );
}

function cardStyle() {
  return {
    display: 'block',
    textAlign: 'left',
    padding: '18px 20px',
    border: '1px solid #e3e3e3',
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
            Optimization for Operations Research · Spring 2026
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
