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

const DEMOS = [
  {
    id: 'nn',
    title: 'Neural Network Visualization',
    description: 'Interactive 8x8 digit classifier — watch forward/backprop in action.',
    Component: NeuralNetViz,
  },
  {
    id: 'optim',
    title: 'First-Order Optimization Methods (2D)',
    description: 'Compare gradient descent, momentum, Adam, etc. on 2D test functions.',
    Component: OptimDemo,
  },
  {
    id: 'optim3d',
    title: 'First-Order Methods in 3D',
    description: 'Same methods, but now you can rotate the loss surface and watch trajectories descend.',
    Component: Optim3DDemo,
  },
  {
    id: 'step',
    title: 'Anatomy of a Step',
    description: 'Step-by-step walkthrough of one optimization iteration.',
    Component: StepAnatomy,
  },
  {
    id: 'svm',
    title: 'Perceptron & Kernel SVM',
    description: 'Step through the perceptron algorithm, then train kernel SVMs (linear / polynomial / RBF).',
    Component: SVMDemo,
  },
  {
    id: 'perceptron-tutorial',
    title: 'Perceptron Code Walkthrough',
    description: 'Side-by-side: Python pseudocode with the active line highlighted, scatter plot updating each step.',
    Component: PerceptronTutorial,
  },
  {
    id: 'sgd-adam-tutorial',
    title: 'SGD & Adam — Code Stepper',
    description: 'In-class demo: step through SGD, momentum, and Adam line-by-line; live trajectory and value panel; PyTorch comparison at the bottom.',
    Component: SGDAdamTutorial,
  },
  {
    id: 'kkt',
    title: 'Constrained Optimization & KKT',
    description: 'Feasible region, active constraints, Lagrange multipliers, and the KKT balance ∇f = −Σ λᵢ ∇gᵢ.',
    Component: KKTDemo,
  },
  {
    id: 'newton',
    title: 'Newton vs. First-Order',
    description: 'Newton step lands at the local quadratic model’s minimum. Compare against gradient descent on the same surface.',
    Component: NewtonDemo,
  },
  {
    id: 'linesearch',
    title: 'Line Search (Armijo / Wolfe)',
    description: 'Slide α along φ(α) = f(x + αd); see which step sizes satisfy Armijo and strong-Wolfe conditions.',
    Component: LineSearchDemo,
  },
  {
    id: 'barrier',
    title: 'Interior Point / Log Barrier',
    description: 'Sweep the barrier parameter and watch the central path approach the LP optimum.',
    Component: BarrierDemo,
  },
  {
    id: 'nn-demo',
    title: 'Neural Networks & Backpropagation',
    description: 'Three-part walkthrough: forward pass with sliders, training with live loss/gradient visualization, and learned-vs-target heatmaps.',
    Component: NNDemo,
  },
];

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

  return (
    <div
      style={{
        maxWidth: 960,
        margin: '0 auto',
        padding: '60px 24px',
      }}
    >
      <h1 style={{ fontSize: 36, fontWeight: 800, marginBottom: 8 }}>
        ISE 5406 — Visualizations
      </h1>
      <p style={{ color: '#555', marginBottom: 36 }}>
        Click a card to launch a demo.
      </p>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 16,
        }}
      >
        {DEMOS.map((d) => {
          const cardStyle = {
            display: 'block',
            textAlign: 'left',
            padding: 20,
            border: '1px solid #ddd',
            borderRadius: 12,
            background: '#fff',
            cursor: 'pointer',
            textDecoration: 'none',
            color: 'inherit',
            transition: 'all 0.15s',
            boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
          };
          const inner = (
            <>
              <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 6 }}>
                {d.title}
              </div>
              <div style={{ fontSize: 14, color: '#666' }}>{d.description}</div>
            </>
          );
          if (d.href) {
            return (
              <a key={d.id} href={d.href} style={cardStyle}>
                {inner}
              </a>
            );
          }
          return (
            <button
              key={d.id}
              onClick={() => setActive(d.id)}
              style={cardStyle}
            >
              {inner}
            </button>
          );
        })}
      </div>
    </div>
  );
}
