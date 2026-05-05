import React, { useState, useMemo, useEffect } from "react";
import { Play, Pause, RotateCcw, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   RLHF — Reinforcement Learning from Human Feedback
   ISE 5406

   A 1-D toy version of the RLHF pipeline:

     1. TRUE REWARD (unknown to the algorithm) — a smooth bump
        on x ∈ [0, 1] peaking at x⋆ = 0.7.
     2. PREFERENCE DATA — sample pairs (a, b), 'human' picks
        according to Bradley-Terry under the true reward.
     3. REWARD MODEL — fit r̂(x) (a small basis expansion) to
        the preferences via logistic regression on r̂(a) − r̂(b).
     4. POLICY UPDATE — KL-regularized PPO-style step:
            π_new = argmax_π  E_x∼π[r̂(x)]  −  β · KL(π ‖ π_ref)
        For a 1-D action space and Gaussian π = N(μ, σ²),
        the closed form is a soft shift of μ toward argmax r̂.

   The demo animates these three sub-problems in sequence and
   then iterates: more preference data → tighter r̂ → policy
   converges to x⋆.
   ============================================================ */

// ============================================================
// True reward, preference dataset
// ============================================================
function trueReward(x) {
  return Math.exp(-((x - 0.7) ** 2) * 18) - 0.4 * Math.exp(-((x - 0.2) ** 2) * 30);
}

function bradleyTerryPick(a, b, sigma = 0.0) {
  const ra = trueReward(a);
  const rb = trueReward(b);
  const noise = sigma > 0 ? sigma * (Math.random() - 0.5) * 2 : 0;
  const p = 1 / (1 + Math.exp(-(ra - rb + noise)));
  return Math.random() < p ? "a" : "b";
}

// ============================================================
// Reward model — small RBF basis expansion
// ============================================================
const N_BASIS = 12;
const BASIS_CENTERS = Array.from({ length: N_BASIS }, (_, i) => i / (N_BASIS - 1));
const BASIS_SIGMA = 0.08;
function basis(x) {
  return BASIS_CENTERS.map((c) => Math.exp(-((x - c) ** 2) / (2 * BASIS_SIGMA ** 2)));
}
function predictReward(theta, x) {
  const phi = basis(x);
  let s = 0;
  for (let i = 0; i < N_BASIS; i++) s += theta[i] * phi[i];
  return s;
}
function fitRewardModel(prefs, theta0, lr = 0.5, iters = 80, l2 = 0.05) {
  // Bradley-Terry log-likelihood. Each pref is (a, b, winner).
  // P(a > b) = sigma(r(a) - r(b))
  // grad = sum_pref (1{winner=a} - sigma(...)) * (phi(a) - phi(b))
  let theta = [...theta0];
  for (let it = 0; it < iters; it++) {
    const grad = new Array(N_BASIS).fill(0);
    for (const p of prefs) {
      const phia = basis(p.a);
      const phib = basis(p.b);
      const ra = phia.reduce((s, v, i) => s + v * theta[i], 0);
      const rb = phib.reduce((s, v, i) => s + v * theta[i], 0);
      const sig = 1 / (1 + Math.exp(-(ra - rb)));
      const y = p.winner === "a" ? 1 : 0;
      const factor = y - sig;
      for (let i = 0; i < N_BASIS; i++) grad[i] += factor * (phia[i] - phib[i]);
    }
    for (let i = 0; i < N_BASIS; i++) {
      grad[i] -= l2 * theta[i]; // ridge
      theta[i] += lr * grad[i] / Math.max(1, prefs.length);
    }
  }
  return theta;
}

// ============================================================
// Policy update — Gaussian policy, KL-regularized
//   π = N(μ, σ²)
//   π_ref = N(μ_ref, σ²)  (frozen)
//   maximize E_x∼π[r̂(x)] - β KL(π ‖ π_ref)
// Approximate via gradient ascent on μ.
// ============================================================
function gradientReward(theta, x) {
  // dr̂/dx via numerical differentiation
  const eps = 1e-3;
  return (predictReward(theta, x + eps) - predictReward(theta, x - eps)) / (2 * eps);
}
function policyUpdate(mu, muRef, theta, beta = 0.5, lr = 0.05, iters = 30) {
  let cur = mu;
  for (let it = 0; it < iters; it++) {
    const g = gradientReward(theta, cur) - beta * (cur - muRef);
    cur += lr * g;
    cur = Math.max(0, Math.min(1, cur));
  }
  return cur;
}

// ============================================================
// Main component
// ============================================================
const X_GRID = Array.from({ length: 200 }, (_, i) => i / 199);

export default function RlhfDemo() {
  const [nPrefs, setNPrefs] = useState(20);
  const [beta, setBeta] = useState(0.5);
  const [round, setRound] = useState(0);
  const [running, setRunning] = useState(false);
  const [history, setHistory] = useState([]);

  // RNG seed for reproducibility
  const seed = useRef(7);
  function rng() {
    seed.current = (seed.current * 1103515245 + 12345) & 0x7fffffff;
    return seed.current / 0x7fffffff;
  }

  function reset() {
    setRound(0);
    setHistory([]);
    setRunning(false);
    seed.current = 7;
  }

  function step() {
    // Run one RLHF round: collect preferences, fit reward model, update policy.
    const prevHist = history;
    const muRef = prevHist.length > 0 ? prevHist[prevHist.length - 1].muNew : 0.5;
    const sigma = 0.18;
    // Sample preferences from current policy (with mild exploration)
    const prefs = [];
    for (let i = 0; i < nPrefs; i++) {
      const a = clamp01(muRef + sigma * gauss());
      const b = clamp01(muRef + sigma * gauss());
      const winner = bradleyTerryPick(a, b, 0.1);
      prefs.push({ a, b, winner });
    }
    const allPrefs = prevHist.flatMap((h) => h.prefs).concat(prefs);
    const theta = fitRewardModel(allPrefs, new Array(N_BASIS).fill(0));
    const muNew = policyUpdate(muRef, muRef, theta, beta);
    setHistory((h) => [
      ...h,
      { round: prevHist.length + 1, prefs, allPrefs, theta, muRef, muNew, beta },
    ]);
    setRound((r) => r + 1);
  }

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      if (round >= 12) {
        setRunning(false);
        return;
      }
      step();
    }, 1000);
    return () => clearInterval(id);
  }, [running, round, history, nPrefs, beta]);

  const cur = history.length > 0 ? history[history.length - 1] : null;

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        RLHF — Reinforcement Learning from Human Feedback
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        A 1-D toy of the RLHF pipeline used to align modern LLMs.
        The 'human' compares pairs of action samples under an unknown
        reward; we fit a reward model from preferences with
        Bradley–Terry; then update a Gaussian policy by maximizing
        expected reward minus a KL penalty against the reference
        policy. Each Step button press runs one full round.
      </p>

      <div style={problemBox}>
        <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
          Three coupled optimization problems
        </div>
        <Tex block>
          {String.raw`\textbf{Bradley–Terry reward fit:} \quad \min_\theta \sum_i \log\!\Big(1 + e^{-(\hat r_\theta(a_i) - \hat r_\theta(b_i)) \cdot y_i}\Big) + \tfrac{\lambda}{2}\|\theta\|^2`}
        </Tex>
        <Tex block>
          {String.raw`\textbf{KL-regularized policy update (PPO objective):} \quad \max_\pi\;\; \mathbb{E}_{x \sim \pi}\big[\hat r(x)\big] \;-\; \beta\, \mathrm{KL}\!\big(\pi \,\|\, \pi_{\mathrm{ref}}\big)`}
        </Tex>
      </div>

      <div style={controlBox}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
          <Slider
            label="preferences per round"
            value={nPrefs}
            onChange={setNPrefs}
            min={5}
            max={80}
            step={5}
          />
          <Slider
            label={<>KL coefficient <Tex>{`\\beta`}</Tex></>}
            value={beta}
            onChange={setBeta}
            min={0.01}
            max={3}
            step={0.05}
          />
        </div>
        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <button onClick={step} style={btnPrimary}>
            Step (one RLHF round)
          </button>
          {!running ? (
            <button onClick={() => setRunning(true)} style={btn} disabled={round >= 12}>
              <Play size={14} /> Auto-run
            </button>
          ) : (
            <button onClick={() => setRunning(false)} style={btn}>
              <Pause size={14} /> Pause
            </button>
          )}
          <button onClick={reset} style={btn}>
            <RotateCcw size={14} /> Reset
          </button>
          <span style={{ alignSelf: "center", fontSize: 12, fontFamily: "monospace", color: "#666" }}>
            round {round}
          </span>
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
        <RewardPlot cur={cur} />
        <div>
          <PolicyPlot cur={cur} />
          <PreferencesPanel cur={cur} />
        </div>
      </div>

      <PolicyTrajectoryChart history={history} />
      <CodePanel />
      <PedagogicalNotes />
    </div>
  );
}

function gauss() {
  // Box-Muller
  const u = Math.random() || 1e-9;
  const v = Math.random() || 1e-9;
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function clamp01(x) {
  return Math.max(0.001, Math.min(0.999, x));
}

// useRef shim
function useRef(init) {
  return React.useRef(init);
}

// ============================================================
// Reward plot (true vs learned)
// ============================================================
function RewardPlot({ cur }) {
  const W = 480, H = 320, padL = 50, padR = 16, padT = 14, padB = 30;
  const cw = W - padL - padR, ch = H - padT - padB;
  const xs = (x) => padL + x * cw;
  const allTrue = X_GRID.map(trueReward);
  const yMin = -0.5;
  const yMax = 1.2;
  const ys = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * ch;
  const truePath = X_GRID.map((x, i) => `${xs(x)},${ys(allTrue[i])}`).join(" ");

  let learnedPath = "";
  if (cur) {
    const learned = X_GRID.map((x) => predictReward(cur.theta, x));
    // Normalize learned reward to match true scale roughly
    const lmax = Math.max(...learned);
    const tmax = Math.max(...allTrue);
    const scale = lmax > 0 ? tmax / lmax : 1;
    learnedPath = X_GRID.map((x, i) => `${xs(x)},${ys(learned[i] * scale)}`).join(" ");
  }

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Reward function
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={ys(0)} x2={padL + cw} y2={ys(0)} stroke="#ccc" strokeDasharray="2,3" />
        {/* True reward */}
        <polyline points={truePath} fill="none" stroke="#1f4e3d" strokeWidth={2.5} />
        {/* Learned reward */}
        {learnedPath && (
          <polyline
            points={learnedPath}
            fill="none"
            stroke="#c8311c"
            strokeWidth={2}
            strokeDasharray="5,3"
          />
        )}
        {/* x* marker */}
        <line x1={xs(0.7)} y1={padT} x2={xs(0.7)} y2={padT + ch} stroke="#0b3da0" strokeWidth={1} strokeDasharray="3,3" />
        <text x={xs(0.7) + 4} y={padT + 12} fontSize={11} fontFamily="monospace" fill="#0b3da0">
          x* = 0.7
        </text>
        {/* Axis labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={v} x={xs(v)} y={padT + ch + 16} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        {/* Legend */}
        <g transform={`translate(${padL + cw - 130}, ${padT + 4})`}>
          <rect width={120} height={36} fill="rgba(255,255,255,0.92)" stroke="#ccc" />
          <line x1={6} y1={12} x2={20} y2={12} stroke="#1f4e3d" strokeWidth={2.5} />
          <text x={26} y={16} fontSize={11}>true r(x)</text>
          <line x1={6} y1={28} x2={20} y2={28} stroke="#c8311c" strokeWidth={2} strokeDasharray="5,3" />
          <text x={26} y={32} fontSize={11}>learned r̂(x)</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Policy plot
// ============================================================
function PolicyPlot({ cur }) {
  const W = 480, H = 220, padL = 50, padR = 16, padT = 14, padB = 30;
  const cw = W - padL - padR, ch = H - padT - padB;
  const xs = (x) => padL + x * cw;
  const sigma = 0.18;
  const muRef = cur ? cur.muRef : 0.5;
  const muNew = cur ? cur.muNew : 0.5;
  const ys = (v) => padT + (1 - v / 3.0) * ch;
  const refPdf = X_GRID.map((x) => Math.exp(-((x - muRef) ** 2) / (2 * sigma ** 2)) / (sigma * Math.sqrt(2 * Math.PI)));
  const newPdf = X_GRID.map((x) => Math.exp(-((x - muNew) ** 2) / (2 * sigma ** 2)) / (sigma * Math.sqrt(2 * Math.PI)));
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Policy distribution
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={xs(0.7)} y1={padT} x2={xs(0.7)} y2={padT + ch} stroke="#0b3da0" strokeDasharray="3,3" />
        <polyline
          points={X_GRID.map((x, i) => `${xs(x)},${ys(refPdf[i])}`).join(" ")}
          fill="none"
          stroke="#888"
          strokeWidth={1.5}
        />
        <polyline
          points={X_GRID.map((x, i) => `${xs(x)},${ys(newPdf[i])}`).join(" ")}
          fill="rgba(11, 61, 160, 0.18)"
          stroke="#0b3da0"
          strokeWidth={2.5}
        />
        {/* mu_ref / mu_new arrow */}
        <line x1={xs(muRef)} y1={padT + ch + 10} x2={xs(muNew)} y2={padT + ch + 10} stroke="#c8311c" strokeWidth={2} markerEnd="url(#arrow)" />
        <defs>
          <marker id="arrow" markerWidth={8} markerHeight={8} refX={6} refY={4} orient="auto">
            <polygon points="0 0, 8 4, 0 8" fill="#c8311c" />
          </marker>
        </defs>
        <text x={xs(muRef) + 4} y={padT + ch + 26} fontSize={11} fontFamily="monospace" fill="#888">μ_ref = {muRef.toFixed(3)}</text>
        <text x={xs(muNew) - 50} y={padT + ch - 4} fontSize={11} fontFamily="monospace" fill="#0b3da0">μ_new = {muNew.toFixed(3)}</text>
      </svg>
    </div>
  );
}

// ============================================================
// Preferences panel
// ============================================================
function PreferencesPanel({ cur }) {
  if (!cur) return null;
  return (
    <div style={{ ...panel, marginTop: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Round {cur.round} — {cur.prefs.length} new preferences
      </div>
      <div style={{ fontSize: 12, fontFamily: "monospace", color: "#444" }}>
        Total preferences accumulated: <b>{cur.allPrefs.length}</b>
      </div>
      <div style={{ marginTop: 6, display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: 4, maxHeight: 140, overflowY: "auto", fontSize: 11, fontFamily: "monospace" }}>
        {cur.prefs.slice(0, 16).map((p, i) => (
          <div key={i} style={{ background: "#fafafa", padding: "2px 6px", borderRadius: 3 }}>
            a={p.a.toFixed(2)}, b={p.b.toFixed(2)} →{" "}
            <b style={{ color: p.winner === "a" ? "#0b3da0" : "#c8311c" }}>{p.winner}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================
// Trajectory chart
// ============================================================
function PolicyTrajectoryChart({ history }) {
  if (history.length === 0) return null;
  const W = 880, H = 140, padL = 50, padR = 16, padT = 12, padB = 26;
  const cw = W - padL - padR, ch = H - padT - padB;
  const ys = (v) => padT + (1 - v) * ch;
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        Policy mean μ over rounds (target = 0.7)
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={ys(0.7)} x2={padL + cw} y2={ys(0.7)} stroke="#0b3da0" strokeDasharray="3,3" />
        <text x={padL + cw - 6} y={ys(0.7) - 4} textAnchor="end" fontSize={10} fill="#0b3da0">
          true x* = 0.7
        </text>
        <polyline
          points={history.map((h, i) => `${xs(i)},${ys(h.muNew)}`).join(" ")}
          fill="none"
          stroke="#c8311c"
          strokeWidth={2}
        />
        {history.map((h, i) => (
          <circle key={i} cx={xs(i)} cy={ys(h.muNew)} r={4} fill="#c8311c" />
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <text key={v} x={padL - 6} y={ys(v) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
      </svg>
    </div>
  );
}

// ============================================================
// Slider
// ============================================================
function Slider({ label, value, onChange, min, max, step }) {
  return (
    <div>
      <div style={{ fontSize: 13, marginBottom: 4 }}>
        {label}: <b>{value.toFixed ? value.toFixed(2) : value}</b>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} style={{ width: "100%" }} />
    </div>
  );
}

// ============================================================
// Code panel
// ============================================================
function CodePanel() {
  return (
    <pre
      style={{
        marginTop: 18,
        background: "#1f1d1a",
        color: "#e8e2d4",
        padding: 14,
        borderRadius: 8,
        fontSize: 12,
        fontFamily: "'JetBrains Mono', Menlo, monospace",
        lineHeight: 1.55,
        whiteSpace: "pre",
        overflowX: "auto",
      }}
    >
{`# Pseudo-RLHF training loop
import torch, torch.nn as nn

reward_model = MLP(input_dim, hidden, 1)
policy = LMPolicy()         # the LLM
ref_policy = freeze(policy)

# 1. Reward modeling phase
for batch in preference_data:
    a, b, winner = batch
    r_a = reward_model(a)
    r_b = reward_model(b)
    # Bradley-Terry log-likelihood
    loss = -log_sigmoid(r_a - r_b) * (winner == "a") \\
           -log_sigmoid(r_b - r_a) * (winner == "b")
    loss.backward(); optimizer.step()

# 2. PPO-style policy update
for batch in prompt_dataset:
    samples = policy.sample(batch.prompt)
    rewards = reward_model(samples).detach()
    log_p   = policy.log_prob(samples, batch.prompt)
    log_p_ref = ref_policy.log_prob(samples, batch.prompt)
    kl      = (log_p - log_p_ref)
    advantage = rewards - rewards.mean()
    # Clipped policy gradient
    ratio = torch.exp(log_p - log_p.detach())
    ppo_loss = -torch.min(ratio * advantage,
                          ratio.clamp(1 - eps, 1 + eps) * advantage)
    loss = ppo_loss.mean() + beta * kl.mean()
    loss.backward(); optimizer.step()

# 3. Iterate: collect more preferences from updated policy, refit r̂, …`}
    </pre>
  );
}

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
          <b>Why preferences and not direct reward?</b> Humans are bad
          at producing scalar rewards but good at saying 'A is better
          than B'. Bradley-Terry turns those pairwise judgments into
          a learnable reward model.
        </li>
        <li>
          <b>The KL term keeps you near home.</b> Without it, the
          policy can chase reward-model artifacts (Goodhart's law) and
          drift to nonsensical outputs that score high on r̂ but
          collapse modeling capacity. <Tex>{`\\beta`}</Tex> trades off
          alignment vs preservation; tuning it is half the engineering.
        </li>
        <li>
          <b>Two optimizations alternating.</b> Reward modeling is a
          convex logistic regression. Policy update is non-convex
          (the policy is a deep net) but practically tractable with
          PPO/DPO/etc. The pipeline iterates between them.
        </li>
        <li>
          <b>DPO without explicit reward.</b> Direct Preference
          Optimization (Rafailov et al. 2023) shows the optimal
          KL-regularized policy implicitly defines a reward, so you
          can train the policy DIRECTLY on preferences with a closed-
          form gradient. No reward model, no PPO. Currently the
          dominant approach.
        </li>
        <li>
          <b>Slider experiments.</b> Decrease β toward 0 to watch the
          policy collapse onto wherever r̂ peaks (which may not be
          the true x* if the reward model is overfit). Decrease prefs
          per round to see how noisy r̂ harms convergence.
        </li>
        <li>
          <b>Where this slots into the LLM stack.</b> Pretrain on web
          text → supervised fine-tune on instructions → RLHF (this
          demo) for alignment / safety / preference steering. RLHF is
          how 'GPT-3.5' became 'ChatGPT'.
        </li>
      </ul>
    </div>
  );
}

const panel = { background: "#fafafa", border: "1px solid #ddd", borderRadius: 8, padding: 12 };
const problemBox = { marginBottom: 16, padding: "12px 16px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8 };
const controlBox = { marginBottom: 16, padding: "12px 16px", background: "#fff", border: "1px solid #e0d8c0", borderRadius: 8 };
const btn = { display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 6, border: "1px solid #ccc", background: "#f7f7f7", cursor: "pointer", fontWeight: 500, fontSize: 13 };
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
