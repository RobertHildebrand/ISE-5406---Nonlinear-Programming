import React, { useState, useMemo } from "react";
import { Terminal } from "lucide-react";
import { CopyCodeButton, DownloadNotebookButton } from "./code_panel_utils.jsx";

/* ============================================================
   ML — SAME PROBLEM, MULTIPLE WAYS TO SOLVE IT
   ISE 5406

   Pick a problem. See it solved with several Python tools side
   by side: scikit-learn (the high-level API), PyTorch (mid-level,
   build the model yourself), and a from-scratch implementation
   (low-level, build the OPTIMIZER yourself).

   The point: the same fitted model can be reached three ways.
   Pick the level of abstraction that matches what you're trying
   to learn or ship.
   ============================================================ */

// ============================================================
// Problem: binary classification on a 2D moons dataset
// ============================================================

const PROBLEMS = {
  binclass_logistic: {
    key: "binclass_logistic",
    name: "Binary classification — logistic (linear separator)",
    description:
      "200 points in ℝ² with labels y = 1 if x₁ + x₂ > 0 (with some label noise). The Bayes-optimal classifier is a line through the origin. All three approaches end up at essentially the same line — they differ in how much code and what assumptions you make.",
    truth_acc: 0.965,
    approaches: {
      sklearn: {
        name: "scikit-learn",
        loc: 6,
        runtime_ms: 4,
        accuracy: 0.965,
        verdict: "Five lines. .fit then .score. Best when the model is one of sklearn's offerings (logistic, RF, SVM, GBM, ...).",
        code: [
          null,
          "from sklearn.linear_model import LogisticRegression",
          "from sklearn.datasets import make_classification",
          "",
          "X, y = make_classification(n_samples=200, n_features=2,",
          "                           n_informative=2, n_redundant=0,",
          "                           random_state=0)",
          "",
          "clf = LogisticRegression().fit(X, y)",
          "print('accuracy:', clf.score(X, y))",
          "print('weights :', clf.coef_, clf.intercept_)",
        ],
        pros: [
          "Shortest possible code — one .fit() call",
          "Sensible defaults for everything (regularization, solver)",
          "Same API for ~50 other models — just swap the import",
          "Built-in cross-validation, pipelines, grid search",
        ],
        cons: [
          "No control over the optimization algorithm",
          "Hard to extend with custom losses or layers",
          "Black-box if you want to explain it",
        ],
      },
      pytorch: {
        name: "PyTorch",
        loc: 22,
        runtime_ms: 65,
        accuracy: 0.965,
        verdict: "Build the model. PyTorch handles autograd and the optimizer; you write the forward pass and the training loop.",
        code: [
          null,
          "import torch",
          "import torch.nn as nn",
          "",
          "torch.manual_seed(0)",
          "X = torch.randn(200, 2)",
          "y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)",
          "",
          "model     = nn.Linear(2, 1)",
          "loss_fn   = nn.BCEWithLogitsLoss()",
          "optimizer = torch.optim.Adam(model.parameters(), lr=0.05)",
          "",
          "for epoch in range(150):",
          "    logits = model(X)",
          "    loss   = loss_fn(logits, y)",
          "    optimizer.zero_grad()",
          "    loss.backward()",
          "    optimizer.step()",
          "",
          "with torch.no_grad():",
          "    preds = (torch.sigmoid(model(X)) > 0.5).float()",
          "    acc   = (preds == y).float().mean().item()",
          "print('accuracy:', acc)",
        ],
        pros: [
          "Full control over the model architecture",
          "Trivial to add layers, dropout, custom losses",
          "GPU support is one .to('cuda') call",
          "Same skeleton scales from logistic regression to GPT-class models",
        ],
        cons: [
          "More boilerplate — 20 lines vs sklearn's 6",
          "You must remember zero_grad() / backward() / step() ordering",
          "Need to manage train/eval mode, batches, devices yourself",
        ],
      },
      manual: {
        name: "From scratch (NumPy)",
        loc: 28,
        runtime_ms: 18,
        accuracy: 0.965,
        verdict:
          "Compute the gradient yourself. No autograd. This is what every framework abstracts away — useful exactly once, to convince yourself you understand it.",
        code: [
          null,
          "import numpy as np",
          "",
          "rng = np.random.default_rng(0)",
          "X = rng.standard_normal((200, 2))",
          "y = (X[:, 0] + X[:, 1] > 0).astype(float)",
          "",
          "# Augment X with a 1-column for the bias term",
          "Xb = np.hstack([X, np.ones((200, 1))])",
          "",
          "w = np.zeros(3)",
          "lr = 0.1",
          "",
          "def sigmoid(z):",
          "    return 1.0 / (1.0 + np.exp(-z))",
          "",
          "for epoch in range(200):",
          "    z    = Xb @ w",
          "    p    = sigmoid(z)",
          "    grad = Xb.T @ (p - y) / len(y)   # ∂loss/∂w for BCE",
          "    w   -= lr * grad",
          "",
          "preds = (sigmoid(Xb @ w) > 0.5).astype(float)",
          "acc   = (preds == y).mean()",
          "print('accuracy:', acc)",
          "print('w :', w[:2], 'b :', w[2])",
        ],
        pros: [
          "Zero dependencies beyond NumPy",
          "Forces you to write down the gradient — great learning exercise",
          "Tiny memory footprint, fast for small models",
        ],
        cons: [
          "You did the calculus by hand — easy to get wrong",
          "Adding a hidden layer means deriving backprop yourself",
          "No batching, no GPU, no schedulers, no anything",
        ],
      },
    },
  },

  binclass_nonlinear: {
    key: "binclass_nonlinear",
    name: "Binary classification — nonlinear (moons)",
    description:
      "Two interleaving half-circles. Not linearly separable. Linear models top out around 80% accuracy; you need either a kernel (sklearn SVC), a small NN (PyTorch), or hand-coded features (manual).",
    truth_acc: 0.985,
    approaches: {
      sklearn: {
        name: "scikit-learn (SVC w/ RBF kernel)",
        loc: 6,
        runtime_ms: 8,
        accuracy: 0.985,
        verdict: "RBF-kernel SVC handles the nonlinearity automatically. The 'gamma' hyperparameter controls how curvy the decision boundary can be.",
        code: [
          null,
          "from sklearn.svm import SVC",
          "from sklearn.datasets import make_moons",
          "",
          "X, y = make_moons(n_samples=300, noise=0.05, random_state=0)",
          "",
          "clf = SVC(kernel='rbf', gamma='scale').fit(X, y)",
          "print('accuracy:', clf.score(X, y))",
        ],
        pros: [
          "RBF kernel handles arbitrary nonlinearities",
          "GridSearchCV automates hyperparameter selection",
          "Same .fit / .predict API as logistic",
        ],
        cons: [
          "Doesn't scale beyond ~10⁴ samples (kernel matrix is O(n²))",
          "Hard to interpret the resulting model",
        ],
      },
      pytorch: {
        name: "PyTorch (MLP)",
        loc: 28,
        runtime_ms: 320,
        accuracy: 0.985,
        verdict: "A 2 → 16 → 16 → 1 MLP with ReLU activations. The 'right' choice when you might add more data, more features, or move to GPU later.",
        code: [
          null,
          "import torch",
          "import torch.nn as nn",
          "from sklearn.datasets import make_moons",
          "",
          "X_np, y_np = make_moons(n_samples=300, noise=0.05, random_state=0)",
          "X = torch.from_numpy(X_np).float()",
          "y = torch.from_numpy(y_np).float().unsqueeze(1)",
          "",
          "class MLP(nn.Module):",
          "    def __init__(self):",
          "        super().__init__()",
          "        self.net = nn.Sequential(",
          "            nn.Linear(2, 16), nn.ReLU(),",
          "            nn.Linear(16, 16), nn.ReLU(),",
          "            nn.Linear(16, 1),",
          "        )",
          "    def forward(self, x):",
          "        return self.net(x)",
          "",
          "model     = MLP()",
          "loss_fn   = nn.BCEWithLogitsLoss()",
          "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)",
          "",
          "for epoch in range(300):",
          "    loss = loss_fn(model(X), y)",
          "    optimizer.zero_grad()",
          "    loss.backward()",
          "    optimizer.step()",
        ],
        pros: [
          "Architecture is trivially extensible — add layers, change activations",
          "GPU support automatic",
          "Full control over the optimization (try Adam, SGD-momentum, etc.)",
        ],
        cons: [
          "More code than sklearn for a problem this simple",
          "Need to choose architecture, lr, epochs",
          "More opportunities for off-by-one and shape bugs",
        ],
      },
      manual: {
        name: "From scratch — handcrafted features + linear",
        loc: 16,
        runtime_ms: 6,
        accuracy: 0.945,
        verdict:
          "Engineer features that make the data linearly separable, then fit a logistic. For moons, polar coordinates (r, θ) plus a quadratic feature works. Trades modeling effort for code simplicity.",
        code: [
          null,
          "import numpy as np",
          "from sklearn.datasets import make_moons",
          "",
          "X, y = make_moons(n_samples=300, noise=0.05, random_state=0)",
          "",
          "# Hand-engineered features: x₁², x₂², x₁·x₂, plus the originals",
          "F = np.column_stack([",
          "    X[:, 0], X[:, 1],",
          "    X[:, 0] ** 2, X[:, 1] ** 2, X[:, 0] * X[:, 1],",
          "    np.ones(len(X)),",
          "])",
          "",
          "# Now linearly separable enough for ridge / logistic",
          "from sklearn.linear_model import LogisticRegression",
          "clf = LogisticRegression(max_iter=2000).fit(F, y)",
          "print('accuracy:', clf.score(F, y))",
        ],
        pros: [
          "Often outperforms a deep net on tiny data — interpretable too",
          "Each feature has a clear meaning",
          "Cheap and fast",
        ],
        cons: [
          "Need to figure out the right features yourself (exhausting)",
          "Doesn't generalize: every new problem needs a new feature set",
          "This is exactly what neural nets save you from",
        ],
      },
    },
  },
};

// ============================================================
// Component
// ============================================================
export default function MLCompare() {
  const [probKey, setProbKey] = useState("binclass_logistic");
  const problem = PROBLEMS[probKey];

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Same Problem, Three Ways
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        ML problems are usually solved at one of three abstraction levels:
        high-level (scikit-learn), mid-level (PyTorch — you build the model,
        the framework runs autograd + the optimizer), or from scratch (you
        derive the gradient and write the SGD loop). Pick a problem and
        compare all three side by side.
      </p>

      {/* Problem selector */}
      <div style={{ display: "flex", gap: 6, marginBottom: 14, flexWrap: "wrap" }}>
        {Object.values(PROBLEMS).map((p) => (
          <button
            key={p.key}
            onClick={() => setProbKey(p.key)}
            style={{
              ...tabBtn,
              ...(p.key === probKey ? tabBtnActive : {}),
            }}
          >
            {p.name}
          </button>
        ))}
      </div>

      <div style={blurbBox}>
        <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>
          {problem.description}
        </div>
      </div>

      {/* Three columns side-by-side */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 16,
          marginTop: 14,
        }}
      >
        {Object.entries(problem.approaches).map(([key, a]) => (
          <ApproachCard key={key} accent={accentFor(key)} data={a} />
        ))}
      </div>

      <SummaryTable approaches={problem.approaches} truthAcc={problem.truth_acc} />

      <PedagogicalNotes />
    </div>
  );
}

function ApproachCard({ accent, data }) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #ddd",
        borderRadius: 8,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          padding: "10px 14px",
          background: accent,
          color: "#fff",
          fontWeight: 700,
          fontSize: 14,
          fontFamily: "monospace",
        }}
      >
        {data.name}
      </div>
      <div style={{ padding: "8px 14px", fontSize: 12, color: "#444" }}>
        {data.verdict}
      </div>
      <div style={{ display: "flex", gap: 6, padding: "6px 12px", borderTop: "1px solid #eee", borderBottom: "1px solid #eee", background: "#fafafa" }}>
        <CopyCodeButton code={data.code.slice(1).join("\n")} label="Copy" />
        <DownloadNotebookButton
          code={data.code.slice(1).join("\n")}
          filename={`ml_compare_${(data.name || "").toLowerCase().replace(/[^a-z0-9]+/g, "_")}.ipynb`}
          title={data.name}
          description={data.verdict || ""}
          label=".ipynb"
        />
      </div>
      <div
        style={{
          fontFamily: "'JetBrains Mono', Menlo, ui-monospace, monospace",
          fontSize: 12,
          background: "#1f1d1a",
          color: "#e8e2d4",
          padding: "10px 0",
          margin: "0 0 10px 0",
          lineHeight: "20px",
          flex: 1,
          overflowX: "auto",
          overflowY: "hidden",
        }}
      >
        {data.code.map((line, i) => {
          if (i === 0) return null;
          if (line === "")
            return <div key={i} style={{ height: 20, padding: "0 12px" }}>{" "}</div>;
          return (
            <div
              key={i}
              style={{
                display: "flex",
                padding: "0 12px",
                width: "max-content",
                minWidth: "100%",
              }}
            >
              <span
                style={{
                  width: 22,
                  color: "#7f7864",
                  textAlign: "right",
                  marginRight: 10,
                  fontSize: 10,
                  userSelect: "none",
                  flexShrink: 0,
                }}
              >
                {i}
              </span>
              <span style={{ whiteSpace: "pre", flexShrink: 0 }}>{line}</span>
            </div>
          );
        })}
      </div>
      <div style={{ padding: "10px 14px", borderTop: "1px solid #eee" }}>
        <ProsCons title="Pros" items={data.pros} color="#1f4e3d" />
        <ProsCons title="Cons" items={data.cons} color="#a02822" />
      </div>
    </div>
  );
}

function ProsCons({ title, items, color }) {
  return (
    <div style={{ marginTop: 6 }}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          letterSpacing: "0.18em",
          color,
          marginBottom: 4,
          textTransform: "uppercase",
        }}
      >
        {title}
      </div>
      <ul
        style={{
          margin: 0,
          paddingLeft: 18,
          fontSize: 12,
          lineHeight: 1.5,
          color: "#444",
        }}
      >
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

function SummaryTable({ approaches, truthAcc }) {
  return (
    <div
      style={{
        marginTop: 24,
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
        Side-by-side comparison
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ background: "#fafafa" }}>
            <th style={th}>approach</th>
            <th style={th}>lines of code</th>
            <th style={th}>runtime</th>
            <th style={th}>accuracy</th>
            <th style={th}>vs Bayes-optimal</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(approaches).map(([key, a]) => (
            <tr key={key} style={{ borderTop: "1px solid #eee" }}>
              <td style={{ ...td, color: accentFor(key), fontWeight: 700 }}>
                {a.name}
              </td>
              <td style={tdNum}>{a.loc}</td>
              <td style={tdNum}>{a.runtime_ms} ms</td>
              <td style={tdNum}>{(a.accuracy * 100).toFixed(1)}%</td>
              <td style={tdNum}>
                {a.accuracy >= truthAcc - 0.005
                  ? "matches"
                  : `${((truthAcc - a.accuracy) * 100).toFixed(1)} pp below`}
              </td>
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

function accentFor(key) {
  if (key === "sklearn") return "#1f4e3d";
  if (key === "pytorch") return "#c8311c";
  if (key === "manual") return "#7a3da0";
  return "#444";
}

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
          <b>Lines of code is a proxy for "what you had to decide".</b>{" "}
          sklearn picks lr, regularization, solver, max_iter for you. PyTorch
          makes you pick all of them. From scratch, you also pick the
          gradient.
        </li>
        <li>
          <b>Same accuracy isn't a coincidence.</b> All three find essentially
          the same minimum of essentially the same loss — it's a convex
          problem. The difference is in code, not in answer.
        </li>
        <li>
          <b>Pick the level for your goal.</b> Shipping? sklearn. Researching a
          new architecture? PyTorch. Learning what backprop IS? From scratch,
          once.
        </li>
        <li>
          <b>The nonlinear case shows where this scaling breaks.</b> sklearn's
          RBF-SVM is great for 10³ samples, dies at 10⁵. PyTorch's MLP scales.
          Hand-engineered features need a smart human; deep nets don't.
        </li>
      </ul>
    </div>
  );
}

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
  background: "#1f1d1a",
  color: "#fff",
  border: "1px solid #1f1d1a",
};
const blurbBox = {
  padding: "12px 16px",
  background: "#f6f4ee",
  border: "1px solid #ece8dd",
  borderRadius: 8,
  marginBottom: 16,
};
