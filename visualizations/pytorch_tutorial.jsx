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
   PYTORCH — CODE STEPPER TUTORIAL
   ISE 5406

   Three classic ML problems, all using the same PyTorch
   training-loop skeleton (forward → loss → backward → step):
     • Linear regression (1 → 1)
     • Logistic / binary classification (2 → 1)
     • MLP on a moons-style nonlinear dataset (2 → 16 → 16 → 1)

   The right panel renders the model architecture, the optimizer
   state, and a hardcoded loss curve. The Python is real — runs
   as-is with torch installed.
   ============================================================ */

// ============================================================
// Problem registry
// ============================================================

const PROB_LINREG = {
  key: "linreg",
  name: "Linear Regression",
  task: "regression",
  blurb:
    "The 'Hello World' of PyTorch. One linear layer (y = wx + b), MSE loss, plain SGD. Watch how few moving parts it really has — model, loss, optimizer, and the four-line train loop.",
  code: [
    null,
    "import torch",
    "import torch.nn as nn",
    "",
    "# Synthetic data: y = 3x + 0.5 + noise",
    "torch.manual_seed(0)",
    "X = torch.randn(100, 1)",
    "y = 3.0 * X + 0.5 + 0.2 * torch.randn(100, 1)",
    "",
    "# Model — a single linear layer",
    "model = nn.Linear(in_features=1, out_features=1)",
    "",
    "loss_fn   = nn.MSELoss()",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.05)",
    "",
    "for epoch in range(200):",
    "    pred = model(X)",
    "    loss = loss_fn(pred, y)",
    "",
    "    optimizer.zero_grad()",
    "    loss.backward()",
    "    optimizer.step()",
    "",
    "w = model.weight.item()",
    "b = model.bias.item()",
    "print(f'learned: y = {w:.3f} * x + {b:.3f}')",
  ],
  events: [
    { line: 1, kind: "import_torch" },
    { line: 2, kind: "import_nn" },
    { line: 5, kind: "set_seed" },
    { line: 6, kind: "create_tensor", payload: { name: "X", shape: "(100, 1)", desc: "input features" }, note: "torch.randn samples from N(0, 1). Tensors are PyTorch's array type — like NumPy arrays but they track gradients and live on GPU on demand." },
    { line: 7, kind: "create_tensor", payload: { name: "y", shape: "(100, 1)", desc: "noisy targets, true model y = 3x + 0.5" } },
    { line: 10, kind: "create_model", payload: { kind: "Linear", layers: [{ name: "Linear", in: 1, out: 1, params: 2 }], total: 2 }, note: "nn.Linear(1, 1) creates a 1×1 weight + 1 bias = 2 parameters. They're initialized randomly and stored as model.weight and model.bias." },
    { line: 12, kind: "set_loss", payload: { name: "MSELoss", desc: "mean squared error" } },
    { line: 13, kind: "set_optimizer", payload: { name: "SGD", lr: 0.05, momentum: 0, desc: "vanilla stochastic gradient descent" }, note: "Pass model.parameters() so the optimizer knows what to update. lr=0.05 is the step size." },
    { line: 15, kind: "training_loop_start", payload: { epochs: 200 }, note: "Outer training loop. We do 200 full passes (epochs) over the small dataset. For larger data you'd add a DataLoader and an inner mini-batch loop." },
    { line: 16, kind: "forward", payload: { in: "X", out: "pred" }, note: "Forward pass. model(X) computes ŷ = X @ w.T + b. Building this expression also builds a computation graph that backward() will use." },
    { line: 17, kind: "compute_loss", payload: { value: 4.872 }, note: "Loss is a scalar tensor. It's the head of the autograd graph — backward() walks back from here." },
    { line: 19, kind: "zero_grad", note: "Gradients ACCUMULATE in PyTorch (good for some advanced use cases). For standard training you must zero them at the start of each iteration." },
    { line: 20, kind: "backward", note: "Compute ∂loss / ∂param for every parameter via reverse-mode autodiff. Each parameter's .grad attribute is now populated." },
    { line: 21, kind: "step", note: "The optimizer reads .grad on each parameter and updates the parameter in place. For SGD: param ← param − lr · param.grad." },
    { line: 23, kind: "training_loop_end", payload: { final_loss: 0.0387, final_w: 2.998, final_b: 0.501 } },
    { line: 24, kind: "extract_param", payload: { name: "w", value: 2.998 } },
    { line: 25, kind: "print", payload: { text: "learned: y = 2.998 * x + 0.501" }, note: "True coefficients were (3.0, 0.5). Recovered to 3 decimals after 200 epochs of SGD." },
  ],
  loss_curve: makeLossCurve(4.872, 0.0387, 200, "geometric"),
};

const PROB_LOGREG = {
  key: "logreg",
  name: "Binary Classification",
  task: "classification",
  blurb:
    "Same skeleton, swap MSE for BCEWithLogitsLoss and SGD for Adam. The model is still nn.Linear (logits in, sigmoid baked into the loss function for numerical stability). Watch the loss drop as the linear separator finds its place.",
  code: [
    null,
    "import torch",
    "import torch.nn as nn",
    "",
    "# 2D points, label = 1 if x₁ + x₂ > 0",
    "torch.manual_seed(0)",
    "N = 200",
    "X = torch.randn(N, 2)",
    "y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)",
    "",
    "model     = nn.Linear(2, 1)",
    "loss_fn   = nn.BCEWithLogitsLoss()",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.05)",
    "",
    "for epoch in range(150):",
    "    logits = model(X)",
    "    loss   = loss_fn(logits, y)",
    "",
    "    optimizer.zero_grad()",
    "    loss.backward()",
    "    optimizer.step()",
    "",
    "with torch.no_grad():",
    "    preds = (torch.sigmoid(model(X)) > 0.5).float()",
    "    acc   = (preds == y).float().mean().item()",
    "print(f'accuracy: {acc:.3f}')",
  ],
  events: [
    { line: 1, kind: "import_torch" },
    { line: 2, kind: "import_nn" },
    { line: 5, kind: "set_seed" },
    { line: 6, kind: "raw_data", payload: { label: "N", value: "200" } },
    { line: 7, kind: "create_tensor", payload: { name: "X", shape: "(200, 2)", desc: "2D point cloud" } },
    { line: 8, kind: "create_tensor", payload: { name: "y", shape: "(200, 1)", desc: "binary labels (0 or 1)" }, note: ".unsqueeze(1) adds a singleton dimension — BCE loss expects shape (N, 1) to match the model output." },
    { line: 10, kind: "create_model", payload: { kind: "Linear", layers: [{ name: "Linear", in: 2, out: 1, params: 3 }], total: 3 } },
    { line: 11, kind: "set_loss", payload: { name: "BCEWithLogitsLoss", desc: "binary cross-entropy with built-in sigmoid" }, note: "Better than computing sigmoid + BCELoss separately — it uses the log-sum-exp trick for numerical stability." },
    { line: 12, kind: "set_optimizer", payload: { name: "Adam", lr: 0.05, desc: "adaptive moment estimation" }, note: "Adam is the default optimizer for almost everything ML these days. lr=0.05 is aggressive but works fine for this tiny model." },
    { line: 14, kind: "training_loop_start", payload: { epochs: 150 } },
    { line: 15, kind: "forward", payload: { in: "X", out: "logits" }, note: "Output is RAW logits — no sigmoid here. The loss function applies it internally." },
    { line: 16, kind: "compute_loss", payload: { value: 0.713 } },
    { line: 18, kind: "zero_grad" },
    { line: 19, kind: "backward" },
    { line: 20, kind: "step" },
    { line: 22, kind: "training_loop_end", payload: { final_loss: 0.071, accuracy: 0.965 } },
    { line: 23, kind: "eval_block", note: "torch.no_grad() disables autograd graph construction. Saves memory and is much faster — always use it for evaluation." },
    { line: 24, kind: "extract_param", payload: { name: "preds", value: "0/1 predictions" }, note: "Apply sigmoid manually since we trained with BCEWithLogitsLoss. Threshold at 0.5." },
    { line: 25, kind: "extract_param", payload: { name: "acc", value: 0.965 } },
    { line: 26, kind: "print", payload: { text: "accuracy: 0.965" }, note: "96.5% accuracy on the training set. The decision boundary is a line through the origin since labels follow x₁ + x₂ > 0." },
  ],
  loss_curve: makeLossCurve(0.713, 0.071, 150, "geometric"),
};

const PROB_MLP = {
  key: "mlp",
  name: "MLP (Nonlinear)",
  task: "classification",
  blurb:
    "Two-moons dataset — two interleaving half-circles, NOT linearly separable. A 2 → 16 → 16 → 1 MLP with ReLU activations finds a curved decision boundary. This is where deep learning starts to actually matter.",
  code: [
    null,
    "import torch",
    "import torch.nn as nn",
    "",
    "torch.manual_seed(0)",
    "N = 300",
    "theta = torch.rand(N) * torch.pi",
    "X1 = torch.stack([torch.cos(theta),       torch.sin(theta)], dim=1)",
    "X2 = torch.stack([1 - torch.cos(theta),  -torch.sin(theta) + 0.5], dim=1)",
    "X  = torch.cat([X1, X2], dim=0) + 0.05 * torch.randn(2 * N, 2)",
    "y  = torch.cat([torch.zeros(N), torch.ones(N)]).unsqueeze(1)",
    "",
    "class MLP(nn.Module):",
    "    def __init__(self):",
    "        super().__init__()",
    "        self.fc1 = nn.Linear(2, 16)",
    "        self.fc2 = nn.Linear(16, 16)",
    "        self.fc3 = nn.Linear(16, 1)",
    "",
    "    def forward(self, x):",
    "        x = torch.relu(self.fc1(x))",
    "        x = torch.relu(self.fc2(x))",
    "        return self.fc3(x)",
    "",
    "model     = MLP()",
    "loss_fn   = nn.BCEWithLogitsLoss()",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)",
    "",
    "for epoch in range(300):",
    "    logits = model(X)",
    "    loss   = loss_fn(logits, y)",
    "    optimizer.zero_grad()",
    "    loss.backward()",
    "    optimizer.step()",
    "",
    "with torch.no_grad():",
    "    acc = ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean()",
    "print(f'accuracy: {acc:.3f}')",
  ],
  events: [
    { line: 1, kind: "import_torch" },
    { line: 2, kind: "import_nn" },
    { line: 4, kind: "set_seed" },
    { line: 5, kind: "raw_data", payload: { label: "N", value: "300" } },
    { line: 6, kind: "create_tensor", payload: { name: "theta", shape: "(300,)", desc: "angles in [0, π]" } },
    { line: 7, kind: "create_tensor", payload: { name: "X1", shape: "(300, 2)", desc: "upper half-circle" } },
    { line: 8, kind: "create_tensor", payload: { name: "X2", shape: "(300, 2)", desc: "lower half-circle, offset" } },
    { line: 9, kind: "create_tensor", payload: { name: "X", shape: "(600, 2)", desc: "concatenated + noise" } },
    { line: 10, kind: "create_tensor", payload: { name: "y", shape: "(600, 1)", desc: "labels" } },
    { line: 12, kind: "define_class", payload: { name: "MLP" }, note: "nn.Module is PyTorch's base class for any model with learnable parameters. You always do this dance: __init__ creates layers, forward implements the computation." },
    { line: 15, kind: "add_layer", payload: { name: "fc1", kind: "Linear", in: 2, out: 16, params: 48 }, note: "First hidden layer. Parameters: 2·16 weights + 16 biases = 48." },
    { line: 16, kind: "add_layer", payload: { name: "fc2", kind: "Linear", in: 16, out: 16, params: 272 } },
    { line: 17, kind: "add_layer", payload: { name: "fc3", kind: "Linear", in: 16, out: 1, params: 17 }, note: "Output is a single logit per input. Total parameters: 48 + 272 + 17 = 337." },
    { line: 19, kind: "define_forward", note: "Forward computation: linear → relu → linear → relu → linear. ReLU(x) = max(0, x). Each ReLU adds nonlinearity — without them the whole stack would collapse to one linear map." },
    { line: 24, kind: "create_model", payload: { kind: "MLP", layers: [
        { name: "fc1", kind: "Linear", in: 2, out: 16, params: 48, act: "ReLU" },
        { name: "fc2", kind: "Linear", in: 16, out: 16, params: 272, act: "ReLU" },
        { name: "fc3", kind: "Linear", in: 16, out: 1, params: 17 },
      ], total: 337 } },
    { line: 25, kind: "set_loss", payload: { name: "BCEWithLogitsLoss" } },
    { line: 26, kind: "set_optimizer", payload: { name: "Adam", lr: 0.01 }, note: "Lower lr (0.01) than the logistic case (0.05) because we have many more parameters and Adam is sensitive to large lr early on." },
    { line: 28, kind: "training_loop_start", payload: { epochs: 300 } },
    { line: 29, kind: "forward", payload: { in: "X", out: "logits" } },
    { line: 30, kind: "compute_loss", payload: { value: 0.694 } },
    { line: 31, kind: "zero_grad" },
    { line: 32, kind: "backward", note: "Backprop through all three linear layers AND the two ReLU activations. PyTorch handles all of this — you just call backward()." },
    { line: 33, kind: "step" },
    { line: 35, kind: "training_loop_end", payload: { final_loss: 0.0431, accuracy: 0.985 } },
    { line: 36, kind: "eval_block" },
    { line: 37, kind: "extract_param", payload: { name: "acc", value: 0.985 } },
    { line: 38, kind: "print", payload: { text: "accuracy: 0.985" }, note: "98.5% accuracy on a problem no linear classifier could exceed ~75% on. The hidden ReLU layers learned a curved decision boundary." },
  ],
  loss_curve: makeLossCurve(0.694, 0.0431, 300, "logistic"),
};

const PROBLEMS = [PROB_LINREG, PROB_LOGREG, PROB_MLP];

function makeLossCurve(start, end, epochs, shape) {
  const out = [];
  for (let e = 0; e < epochs; e++) {
    const t = e / (epochs - 1);
    let v;
    if (shape === "geometric") {
      // geometric decay
      v = end + (start - end) * Math.exp(-4 * t);
    } else {
      // S-curve plateau then drop
      v = end + (start - end) / (1 + Math.exp(8 * (t - 0.4)));
    }
    out.push({ epoch: e, loss: v });
  }
  return out;
}

// ============================================================
// State replay
// ============================================================
function replayState(events, upTo, problem) {
  const s = {
    importedTorch: false,
    importedNN: false,
    tensors: [],
    classDefined: null,
    layers: [],
    forwardDefined: false,
    model: null,
    loss: null,
    optimizer: null,
    inLoop: false,
    epochs: 0,
    currentForward: null,
    currentLoss: null,
    zeroGradDone: false,
    backwardDone: false,
    stepDone: false,
    lossHistory: [],
    finalLoss: null,
    finalAcc: null,
    extracted: [],
    prints: [],
  };
  for (let i = 0; i <= upTo && i < events.length; i++) {
    const ev = events[i];
    switch (ev.kind) {
      case "import_torch":
        s.importedTorch = true;
        break;
      case "import_nn":
        s.importedNN = true;
        break;
      case "set_seed":
        break;
      case "raw_data":
      case "create_tensor":
        s.tensors.push(ev.payload);
        break;
      case "define_class":
        s.classDefined = ev.payload.name;
        break;
      case "add_layer":
        s.layers.push(ev.payload);
        break;
      case "define_forward":
        s.forwardDefined = true;
        break;
      case "create_model":
        s.model = ev.payload;
        if (ev.payload.layers) s.layers = ev.payload.layers;
        break;
      case "set_loss":
        s.loss = ev.payload;
        break;
      case "set_optimizer":
        s.optimizer = ev.payload;
        break;
      case "training_loop_start":
        s.inLoop = true;
        s.epochs = ev.payload.epochs;
        // Snapshot loss curve up to start
        s.lossHistory = [problem.loss_curve[0]];
        break;
      case "forward":
        s.currentForward = ev.payload;
        s.zeroGradDone = false;
        s.backwardDone = false;
        s.stepDone = false;
        break;
      case "compute_loss":
        s.currentLoss = ev.payload.value;
        break;
      case "zero_grad":
        s.zeroGradDone = true;
        break;
      case "backward":
        s.backwardDone = true;
        break;
      case "step":
        s.stepDone = true;
        // After step, advance the loss curve to "end"
        s.lossHistory = problem.loss_curve;
        break;
      case "training_loop_end":
        s.inLoop = false;
        s.finalLoss = ev.payload.final_loss;
        if (ev.payload.accuracy !== undefined) s.finalAcc = ev.payload.accuracy;
        if (ev.payload.final_w !== undefined) {
          s.extracted.push({ name: "w", value: ev.payload.final_w });
          s.extracted.push({ name: "b", value: ev.payload.final_b });
        }
        s.lossHistory = problem.loss_curve;
        break;
      case "eval_block":
        break;
      case "extract_param":
        s.extracted.push(ev.payload);
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
// Main
// ============================================================
export default function PyTorchTutorial() {
  const [probKey, setProbKey] = useState(PROB_LINREG.key);
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
  const state = useMemo(
    () => replayState(problem.events, evIdx, problem),
    [problem, evIdx]
  );

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        PyTorch — Code Stepper
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Same training-loop skeleton, three problems. PyTorch's whole API for
        supervised learning fits on a postcard: build a model, build a loss,
        build an optimizer, call forward / backward / step in a loop. The right
        panel shows what's in memory at each step.
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
                background: p.task === "regression" ? "#d4a017" : "#0b3da0",
                color: "#fff",
                borderRadius: 2,
                fontFamily: "monospace",
                letterSpacing: "0.04em",
              }}
            >
              {p.task}
            </span>
            {p.name}
          </button>
        ))}
      </div>

      <div style={blurbBox}>
        <div style={{ fontSize: 13, color: "#444", lineHeight: 1.5 }}>
          {problem.blurb}
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
            <CopyCodeButton code={problem.code.slice(1).join("\n")} />
            <DownloadNotebookButton
              code={problem.code.slice(1).join("\n")}
              filename={`pytorch_${problem.key || "demo"}.ipynb`}
              title={problem.name}
              description={problem.blurb || ""}
            />
          </div>
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
              {ev?.note || stockNote(ev?.kind) || "(keep stepping)"}
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
                background: "#c8311c",
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

        <div>
          <LossCurve state={state} problem={problem} />
          <StatePanel state={state} />
        </div>
      </div>

      <PedagogicalNotes />
    </div>
  );
}

function stockNote(kind) {
  if (kind === "import_torch") return "PyTorch — tensor library + autograd + optimizers + neural-net building blocks.";
  if (kind === "import_nn") return "torch.nn provides Linear, Conv2d, ReLU, MSELoss, etc. — the building blocks of deep nets.";
  if (kind === "set_seed") return "Reproducibility. With the same seed, the same random tensors come out — useful for debugging and comparing runs.";
  if (kind === "create_tensor") return "Tensor created.";
  if (kind === "raw_data") return "Plain Python value — sample size, etc.";
  return null;
}

// ============================================================
// Loss curve
// ============================================================
function LossCurve({ state, problem }) {
  const W = 420,
    H = 160;
  const data = state.lossHistory;
  if (!data || data.length === 0) {
    return (
      <div
        style={{
          marginBottom: 14,
          background: "#fafafa",
          border: "1px solid #eee",
          borderRadius: 8,
          padding: 14,
          fontSize: 13,
          color: "#888",
          fontFamily: "monospace",
        }}
      >
        loss curve appears once training starts
      </div>
    );
  }
  const totalEpochs = problem.loss_curve.length;
  const maxLoss = Math.max(...problem.loss_curve.map((d) => d.loss));
  const minLoss = Math.min(...problem.loss_curve.map((d) => d.loss));
  const xPx = (e) => 30 + (e / (totalEpochs - 1)) * (W - 40);
  const yPx = (l) =>
    H - 30 - ((l - minLoss) / (maxLoss - minLoss + 1e-12)) * (H - 40);
  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${xPx(d.epoch).toFixed(1)} ${yPx(d.loss).toFixed(1)}`)
    .join(" ");
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
        Loss curve
      </div>
      <svg width={W} height={H} style={{ display: "block" }}>
        <line x1={30} y1={H - 30} x2={W - 10} y2={H - 30} stroke="#bbb" />
        <line x1={30} y1={10} x2={30} y2={H - 30} stroke="#bbb" />
        <text x={4} y={20} fontSize={10} fill="#666" fontFamily="monospace">
          {maxLoss.toFixed(2)}
        </text>
        <text x={4} y={H - 30} fontSize={10} fill="#666" fontFamily="monospace">
          {minLoss.toFixed(3)}
        </text>
        <text x={W - 30} y={H - 16} fontSize={10} fill="#666" fontFamily="monospace">
          epoch
        </text>
        <text x={32} y={H - 16} fontSize={10} fill="#666" fontFamily="monospace">
          0
        </text>
        <text x={W - 50} y={H - 16} fontSize={10} fill="#666" fontFamily="monospace">
          {totalEpochs}
        </text>
        <path d={path} fill="none" stroke="#c8311c" strokeWidth={2} />
        {data.length > 1 && (
          <circle
            cx={xPx(data[data.length - 1].epoch)}
            cy={yPx(data[data.length - 1].loss)}
            r={3.5}
            fill="#c8311c"
          />
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
        {!state.importedTorch && !state.importedNN ? (
          <Empty />
        ) : (
          <div>
            {state.importedTorch && <span style={chip("#c8311c")}>torch</span>}
            {state.importedNN && <span style={chip("#c8311c")}>torch.nn</span>}
          </div>
        )}
      </Section>

      {state.tensors.length > 0 && (
        <Section title="Tensors / data">
          {state.tensors.map((t, i) => (
            <KVRow
              key={i}
              k={t.name || t.label}
              v={t.shape || t.value}
              desc={t.desc}
              mono
            />
          ))}
        </Section>
      )}

      {(state.layers.length > 0 || state.classDefined || state.model) && (
        <Section title="Model architecture">
          {state.classDefined && !state.model && (
            <div style={{ fontSize: 12, color: "#888", marginBottom: 6, fontFamily: "monospace" }}>
              class {state.classDefined}(nn.Module): defined, not yet instantiated
            </div>
          )}
          {state.layers.length > 0 && (
            <div>
              {state.layers.map((l, i) => (
                <div
                  key={i}
                  style={{
                    padding: "5px 10px",
                    background: "#fff",
                    border: "1px solid #ddd",
                    borderRadius: 6,
                    marginBottom: 4,
                    fontFamily: "monospace",
                    fontSize: 12,
                    display: "flex",
                    justifyContent: "space-between",
                  }}
                >
                  <span>
                    <b>{l.name}</b>: {l.kind}({l.in} → {l.out})
                    {l.act && ` → ${l.act}`}
                  </span>
                  <span style={{ color: "#666" }}>{l.params} params</span>
                </div>
              ))}
              {state.model && state.model.total > 0 && (
                <div
                  style={{
                    padding: "5px 10px",
                    fontFamily: "monospace",
                    fontSize: 12,
                    color: "#1f4e3d",
                    fontWeight: 700,
                    textAlign: "right",
                  }}
                >
                  total: {state.model.total} parameters
                </div>
              )}
            </div>
          )}
        </Section>
      )}

      {state.loss && (
        <Section title="Loss function">
          <span style={chip("#1f4e3d")}>nn.{state.loss.name}</span>
          {state.loss.desc && (
            <span style={{ fontSize: 12, color: "#666", marginLeft: 8 }}>
              ({state.loss.desc})
            </span>
          )}
        </Section>
      )}

      {state.optimizer && (
        <Section title="Optimizer">
          <span style={chip("#7a3da0")}>
            torch.optim.{state.optimizer.name}(lr={state.optimizer.lr})
          </span>
        </Section>
      )}

      {state.inLoop && (
        <Section title="Training loop">
          <div
            style={{
              padding: 10,
              background: "#1f1d1a",
              color: "#e8e2d4",
              borderRadius: 6,
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: 1.55,
            }}
          >
            <ResultLine k="for epoch in range" v={state.epochs} />
            {state.currentForward && (
              <ResultLine
                k="forward"
                v={`model(${state.currentForward.in}) → ${state.currentForward.out}`}
                c="#7dd87d"
              />
            )}
            {state.currentLoss != null && (
              <ResultLine k="loss" v={state.currentLoss.toFixed(4)} c="#f5a524" />
            )}
            <ResultLine
              k="zero_grad()"
              v={state.zeroGradDone ? "✓" : "—"}
              c={state.zeroGradDone ? "#7dd87d" : "#7f7864"}
            />
            <ResultLine
              k="loss.backward()"
              v={state.backwardDone ? "✓" : "—"}
              c={state.backwardDone ? "#7dd87d" : "#7f7864"}
            />
            <ResultLine
              k="optimizer.step()"
              v={state.stepDone ? "✓" : "—"}
              c={state.stepDone ? "#7dd87d" : "#7f7864"}
            />
          </div>
        </Section>
      )}

      {(state.finalLoss != null || state.finalAcc != null || state.extracted.length > 0) && (
        <Section title="After training">
          <div
            style={{
              padding: 10,
              background: "#1f1d1a",
              color: "#e8e2d4",
              borderRadius: 6,
              fontFamily: "monospace",
              fontSize: 12,
              lineHeight: 1.55,
            }}
          >
            {state.finalLoss != null && (
              <ResultLine
                k="final loss"
                v={state.finalLoss.toExponential(3)}
                c="#f5a524"
              />
            )}
            {state.finalAcc != null && (
              <ResultLine
                k="accuracy"
                v={(state.finalAcc * 100).toFixed(1) + "%"}
                c="#7dd87d"
              />
            )}
            {state.extracted.map((e, i) => (
              <ResultLine
                key={i}
                k={e.name}
                v={typeof e.value === "number" ? (+e.value).toFixed(3) : e.value}
                c="#f5a524"
              />
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
          <b>The four-line ritual.</b>{" "}
          <code style={inlineCode}>optimizer.zero_grad()</code> →{" "}
          <code style={inlineCode}>loss.backward()</code> →{" "}
          <code style={inlineCode}>optimizer.step()</code>. Forget zero_grad
          and gradients accumulate across iterations — the most common PyTorch
          bug.
        </li>
        <li>
          <b>Autograd is a graph.</b> Every tensor with{" "}
          <code style={inlineCode}>requires_grad=True</code> tracks its history.
          Calling <code style={inlineCode}>backward()</code> walks back through
          that graph, computing gradients via reverse-mode automatic
          differentiation.
        </li>
        <li>
          <b>BCEWithLogitsLoss vs Sigmoid + BCELoss.</b> The "with logits"
          variant fuses sigmoid into the loss for numerical stability (avoids
          log(0) when sigmoid saturates). Always prefer it.
        </li>
        <li>
          <b>nn.Module.__init__ is the boilerplate.</b> Every model with
          learnable parameters subclasses{" "}
          <code style={inlineCode}>nn.Module</code>, calls{" "}
          <code style={inlineCode}>super().__init__()</code>, declares its
          layers as attributes (so PyTorch can find them via{" "}
          <code style={inlineCode}>.parameters()</code>), and implements{" "}
          <code style={inlineCode}>forward(x)</code>.
        </li>
        <li>
          <b>torch.no_grad() at eval time.</b> Disables graph construction —
          much faster, much less memory. Forgetting it is harmless but
          wasteful.
        </li>
        <li>
          <b>Adam vs SGD.</b> Adam works almost everywhere; SGD with momentum
          is what wins many image-classification benchmarks. Default to Adam,
          switch to SGD when you have time to tune.
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
        Install PyTorch &nbsp;
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
            CPU-only is fine for everything in this tutorial. For GPU you need
            a matching CUDA install — use the official selector at{" "}
            <code style={inlineCode}>pytorch.org/get-started</code>.
          </p>
          <Pre>
            {`# CPU-only (works everywhere)
pip install torch

# Apple Silicon Macs get MPS acceleration for free with the same install
# Linux + NVIDIA GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`}
          </Pre>
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
// Bits & pieces (same atoms as other tutorials)
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
