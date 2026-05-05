import React, { useState, useEffect, useRef } from "react";
import { Play, Pause, RotateCcw, StepForward, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   REINFORCEMENT LEARNING — Gridworld
   ISE 5406

   A 5×5 gridworld with one terminal goal (top-right) and one
   "lava" trap (middle). Reward: -1 per step, +10 at goal,
   -10 at lava (terminal). Discount γ = 0.9.

   Three algorithms:
     • Value iteration (synchronous Bellman backups)
     • Policy iteration (alternates evaluation + improvement)
     • Q-learning (online, model-free, ε-greedy)

   Animation shows V-values as a heatmap and the greedy policy
   as arrows. Q-learning shows the agent's actual trajectory.
   ============================================================ */

// ============================================================
// Gridworld setup
// ============================================================
const GRID_SIZE = 5;
const ACTIONS = [
  { dx: 0, dy: -1, name: "↑" },
  { dx: 1, dy: 0, name: "→" },
  { dx: 0, dy: 1, name: "↓" },
  { dx: -1, dy: 0, name: "←" },
];
const GAMMA = 0.9;
const STEP_REWARD = -1;
const GOAL = { x: 4, y: 0, reward: 10 };
const LAVA = { x: 2, y: 2, reward: -10 };
const START = { x: 0, y: 4 };

function isTerminal(x, y) {
  return (x === GOAL.x && y === GOAL.y) || (x === LAVA.x && y === LAVA.y);
}
function rewardAt(x, y) {
  if (x === GOAL.x && y === GOAL.y) return GOAL.reward;
  if (x === LAVA.x && y === LAVA.y) return LAVA.reward;
  return STEP_REWARD;
}
function step(x, y, a) {
  // Deterministic; bumping a wall stays put.
  let nx = x + ACTIONS[a].dx;
  let ny = y + ACTIONS[a].dy;
  if (nx < 0 || nx >= GRID_SIZE || ny < 0 || ny >= GRID_SIZE) {
    nx = x;
    ny = y;
  }
  return { nx, ny, r: rewardAt(nx, ny), done: isTerminal(nx, ny) };
}

// ============================================================
// Value iteration generator
// ============================================================
function emptyV() {
  return Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
}
function copy2D(V) {
  return V.map((r) => [...r]);
}

function* valueIteration(maxSweeps) {
  let V = emptyV();
  for (let k = 0; k < maxSweeps; k++) {
    const newV = copy2D(V);
    let maxDelta = 0;
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (isTerminal(x, y)) {
          newV[y][x] = 0;
          continue;
        }
        let best = -Infinity;
        for (let a = 0; a < 4; a++) {
          const { nx, ny, r, done } = step(x, y, a);
          const val = r + (done ? 0 : GAMMA * V[ny][nx]);
          if (val > best) best = val;
        }
        newV[y][x] = best;
        maxDelta = Math.max(maxDelta, Math.abs(newV[y][x] - V[y][x]));
      }
    }
    V = newV;
    yield { V, sweep: k + 1, maxDelta, algo: "VI" };
    if (maxDelta < 1e-4) return;
  }
}

// ============================================================
// Policy iteration generator
// ============================================================
function* policyIteration(maxIters) {
  let policy = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(1));
  let V = emptyV();
  for (let k = 0; k < maxIters; k++) {
    // Policy evaluation (until convergence)
    for (let it = 0; it < 100; it++) {
      const newV = copy2D(V);
      for (let y = 0; y < GRID_SIZE; y++) {
        for (let x = 0; x < GRID_SIZE; x++) {
          if (isTerminal(x, y)) {
            newV[y][x] = 0;
            continue;
          }
          const a = policy[y][x];
          const { nx, ny, r, done } = step(x, y, a);
          newV[y][x] = r + (done ? 0 : GAMMA * V[ny][nx]);
        }
      }
      const delta = Math.max(
        ...newV.flat().map((v, i) => Math.abs(v - V.flat()[i]))
      );
      V = newV;
      if (delta < 1e-4) break;
    }
    // Policy improvement
    let stable = true;
    const newPolicy = policy.map((r) => [...r]);
    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        if (isTerminal(x, y)) continue;
        let bestA = 0, bestV = -Infinity;
        for (let a = 0; a < 4; a++) {
          const { nx, ny, r, done } = step(x, y, a);
          const val = r + (done ? 0 : GAMMA * V[ny][nx]);
          if (val > bestV) {
            bestV = val;
            bestA = a;
          }
        }
        if (bestA !== policy[y][x]) stable = false;
        newPolicy[y][x] = bestA;
      }
    }
    policy = newPolicy;
    yield { V, policy, sweep: k + 1, algo: "PI", stable };
    if (stable) return;
  }
}

// ============================================================
// Q-learning generator
// ============================================================
function emptyQ() {
  return Array.from({ length: GRID_SIZE }, () =>
    Array.from({ length: GRID_SIZE }, () => [0, 0, 0, 0])
  );
}
function* qLearning(nEpisodes, alpha, epsilon) {
  let Q = emptyQ();
  let trajectory = [];
  // Pseudo-random
  let seed = 17;
  function rng() {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }
  for (let ep = 0; ep < nEpisodes; ep++) {
    let { x, y } = START;
    trajectory = [{ x, y }];
    let returnVal = 0;
    for (let t = 0; t < 100; t++) {
      let a;
      if (rng() < epsilon) a = Math.floor(rng() * 4);
      else {
        let best = 0;
        for (let i = 1; i < 4; i++) if (Q[y][x][i] > Q[y][x][best]) best = i;
        a = best;
      }
      const { nx, ny, r, done } = step(x, y, a);
      let target = r;
      if (!done) {
        let bestNext = Q[ny][nx][0];
        for (let i = 1; i < 4; i++) if (Q[ny][nx][i] > bestNext) bestNext = Q[ny][nx][i];
        target += GAMMA * bestNext;
      }
      Q[y][x][a] += alpha * (target - Q[y][x][a]);
      x = nx;
      y = ny;
      trajectory.push({ x, y });
      returnVal += r;
      if (done) break;
    }
    // Compute V from Q (max over actions)
    const V = emptyV();
    const policy = Array.from({ length: GRID_SIZE }, () => Array(GRID_SIZE).fill(0));
    for (let y2 = 0; y2 < GRID_SIZE; y2++) {
      for (let x2 = 0; x2 < GRID_SIZE; x2++) {
        let best = 0;
        for (let i = 1; i < 4; i++) if (Q[y2][x2][i] > Q[y2][x2][best]) best = i;
        V[y2][x2] = Q[y2][x2][best];
        policy[y2][x2] = best;
      }
    }
    yield { V, policy, Q, sweep: ep + 1, returnVal, trajectory, algo: "Q" };
  }
}

// ============================================================
// Main component
// ============================================================
export default function RlDemo() {
  const [algo, setAlgo] = useState("vi");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(300);
  const [state, setState] = useState({ V: emptyV(), policy: null, sweep: 0 });
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);

  function reset() {
    setRunning(false);
    setState({ V: emptyV(), policy: null, sweep: 0 });
    setHistory([]);
    genRef.current = null;
  }

  function start() {
    if (algo === "vi") genRef.current = valueIteration(50);
    else if (algo === "pi") genRef.current = policyIteration(20);
    else genRef.current = qLearning(80, 0.1, 0.2);
    setHistory([]);
    setRunning(true);
  }

  function step1() {
    if (!genRef.current) start();
    const next = genRef.current.next();
    if (!next.done) {
      setState(next.value);
      setHistory((h) => [...h, next.value]);
    } else setRunning(false);
  }

  useEffect(() => {
    if (!running) return;
    if (!genRef.current) {
      if (algo === "vi") genRef.current = valueIteration(50);
      else if (algo === "pi") genRef.current = policyIteration(20);
      else genRef.current = qLearning(80, 0.1, 0.2);
      setHistory([]);
    }
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) {
        setRunning(false);
        return;
      }
      setState(next.value);
      setHistory((h) => [...h, next.value]);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed, algo]);

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Reinforcement Learning — Gridworld
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        A 5×5 gridworld with a goal (top-right, +10), lava (center, −10),
        and a step penalty of −1 everywhere else. Discount{" "}
        <Tex>{`\\gamma = 0.9`}</Tex>. Three algorithms compete: value
        iteration, policy iteration, and online Q-learning. Watch
        V-values fill in and the greedy policy crystallize.
      </p>

      <div style={problemBox}>
        <Tex block>
          {String.raw`V^\star(s) = \max_a \left[ r(s, a) + \gamma\, V^\star(s') \right] \quad\quad Q^\star(s, a) = r(s,a) + \gamma \max_{a'} Q^\star(s', a')`}
        </Tex>
        <div style={{ fontSize: 13, color: "#444", marginTop: 4 }}>
          Bellman optimality. Value iteration applies the max-Bellman
          backup synchronously; policy iteration alternates expectation +
          improvement; Q-learning learns Q(s, a) from transitions sampled
          online with ε-greedy exploration.
        </div>
      </div>

      <div style={{ marginBottom: 12, padding: "10px 14px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8 }}>
        <label style={{ marginRight: 14 }}>
          <input type="radio" checked={algo === "vi"} onChange={() => { setAlgo("vi"); reset(); }} />
          &nbsp;Value iteration
        </label>
        <label style={{ marginRight: 14 }}>
          <input type="radio" checked={algo === "pi"} onChange={() => { setAlgo("pi"); reset(); }} />
          &nbsp;Policy iteration
        </label>
        <label>
          <input type="radio" checked={algo === "ql"} onChange={() => { setAlgo("ql"); reset(); }} />
          &nbsp;Q-learning (ε = 0.2, α = 0.1)
        </label>
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
          <GridworldViz state={state} algo={algo} />
          <Controls running={running} start={start} stop={() => setRunning(false)} reset={reset} step={step1} speed={speed} setSpeed={setSpeed} />
        </div>
        <div>
          <StatePanel state={state} algo={algo} />
          {algo === "ql" && <ReturnsChart history={history} />}
          <CodeBlock algo={algo} />
        </div>
      </div>
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Gridworld SVG
// ============================================================
function GridworldViz({ state, algo }) {
  const cell = 80;
  const W = GRID_SIZE * cell + 4;
  const H = GRID_SIZE * cell + 4;
  const V = state.V;
  const policy = state.policy;
  const traj = state.trajectory;

  // Color scale for V: blue (negative) → white → green (positive)
  const flat = V.flat().filter(isFinite);
  const vmin = Math.min(...flat, -10);
  const vmax = Math.max(...flat, 10);
  function color(v) {
    if (vmin === vmax) return "#fff";
    const t = (v - vmin) / (vmax - vmin);
    if (t < 0.5) {
      const u = t * 2;
      const r = Math.round(11 + (255 - 11) * u);
      const g = Math.round(61 + (255 - 61) * u);
      const b = Math.round(160 + (255 - 160) * u);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      const u = (t - 0.5) * 2;
      const r = Math.round(255 - (255 - 31) * u);
      const g = Math.round(255 - (255 - 78) * u);
      const b = Math.round(255 - (255 - 61) * u);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }

  return (
    <div style={panel}>
      <svg width={W} height={H}>
        {V.map((row, y) =>
          row.map((v, x) => {
            const isGoal = x === GOAL.x && y === GOAL.y;
            const isLava = x === LAVA.x && y === LAVA.y;
            const fill = isGoal ? "#1f4e3d" : isLava ? "#c8311c" : color(v);
            return (
              <g key={`${x}_${y}`}>
                <rect
                  x={x * cell + 2}
                  y={y * cell + 2}
                  width={cell - 2}
                  height={cell - 2}
                  fill={fill}
                  stroke="#444"
                  strokeWidth={1}
                />
                {isGoal && (
                  <text x={x * cell + cell / 2} y={y * cell + cell / 2 + 5} textAnchor="middle" fontSize={14} fill="#fff" fontWeight={700}>
                    GOAL +10
                  </text>
                )}
                {isLava && (
                  <text x={x * cell + cell / 2} y={y * cell + cell / 2 + 5} textAnchor="middle" fontSize={14} fill="#fff" fontWeight={700}>
                    LAVA −10
                  </text>
                )}
                {!isGoal && !isLava && (
                  <text
                    x={x * cell + cell / 2}
                    y={y * cell + cell / 2 + 4}
                    textAnchor="middle"
                    fontSize={13}
                    fill={Math.abs(v) > (vmax - vmin) / 2 ? "#fff" : "#222"}
                    fontFamily="monospace"
                  >
                    {v.toFixed(2)}
                  </text>
                )}
                {/* Greedy arrow */}
                {policy && !isGoal && !isLava && (
                  <ArrowGlyph
                    cx={x * cell + cell / 2}
                    cy={y * cell + cell / 2 - 18}
                    dir={policy[y][x]}
                  />
                )}
              </g>
            );
          })
        )}
        {/* Trajectory (Q-learning) */}
        {traj && traj.length > 1 && (
          <polyline
            points={traj.map((p) => `${p.x * cell + cell / 2},${p.y * cell + cell / 2}`).join(" ")}
            fill="none"
            stroke="#f5a524"
            strokeWidth={3}
            strokeDasharray="4,3"
          />
        )}
        {/* Start marker */}
        <circle
          cx={START.x * cell + cell / 2}
          cy={START.y * cell + cell / 2 + 24}
          r={6}
          fill="#0b3da0"
          stroke="#fff"
          strokeWidth={2}
        />
        <text
          x={START.x * cell + cell / 2 + 12}
          y={START.y * cell + cell / 2 + 28}
          fontSize={11}
          fontFamily="monospace"
          fill="#0b3da0"
          fontWeight={700}
        >
          start
        </text>
      </svg>
    </div>
  );
}

function ArrowGlyph({ cx, cy, dir }) {
  const a = ACTIONS[dir];
  const len = 12;
  const dx = a.dx * len;
  const dy = a.dy * len;
  return (
    <g>
      <line x1={cx - dx / 2} y1={cy - dy / 2} x2={cx + dx / 2} y2={cy + dy / 2} stroke="#222" strokeWidth={2} />
      <circle cx={cx + dx / 2} cy={cy + dy / 2} r={3} fill="#222" />
    </g>
  );
}

// ============================================================
// Controls
// ============================================================
function Controls({ running, start, stop, reset, step, speed, setSpeed }) {
  return (
    <div style={{ ...panel, marginTop: 12 }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button onClick={step} style={btnPrimary}>
          <StepForward size={14} /> Step
        </button>
        {!running ? (
          <button onClick={start} style={btn}>
            <Play size={14} /> Run
          </button>
        ) : (
          <button onClick={stop} style={btn}>
            <Pause size={14} /> Pause
          </button>
        )}
        <button onClick={reset} style={btn}>
          <RotateCcw size={14} /> Reset
        </button>
      </div>
      <label style={{ fontSize: 12, fontFamily: "monospace", color: "#444" }}>
        speed: <b>{speed} ms/step</b>
      </label>
      <input type="range" min={50} max={1000} step={50} value={speed} onChange={(e) => setSpeed(+e.target.value)} style={{ width: "100%" }} />
    </div>
  );
}

// ============================================================
// State panel
// ============================================================
function StatePanel({ state, algo }) {
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        State
      </div>
      <KV k="algorithm" v={algo === "vi" ? "Value iteration" : algo === "pi" ? "Policy iteration" : "Q-learning"} />
      <KV k="iteration" v={state.sweep || 0} highlight />
      {state.maxDelta != null && <KV k="max ΔV" v={state.maxDelta.toExponential(3)} />}
      {state.stable === true && <KV k="policy stable" v="✓ converged" highlight />}
      {state.returnVal != null && <KV k="episode return" v={state.returnVal.toFixed(2)} />}
      <KV k="γ" v={GAMMA} />
    </div>
  );
}

function KV({ k, v, highlight }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px dotted #eee" }}>
      <span style={{ color: "#666", fontSize: 12, fontFamily: "monospace" }}>{k}</span>
      <span style={{ fontSize: 12, fontFamily: "monospace", color: highlight ? "#c8311c" : "#222", fontWeight: highlight ? 700 : 400 }}>{v}</span>
    </div>
  );
}

// ============================================================
// Returns chart (Q-learning)
// ============================================================
function ReturnsChart({ history }) {
  if (history.length === 0) return null;
  const W = 460, H = 140, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const returns = history.map((h) => h.returnVal);
  const yMin = Math.min(...returns, -50);
  const yMax = Math.max(...returns, 10);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  return (
    <div style={{ ...panel, marginTop: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        episode return over time
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={ys(0)} x2={padL + cw} y2={ys(0)} stroke="#ccc" strokeDasharray="3,3" />
        <polyline points={returns.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#0b3da0" strokeWidth={1.6} />
      </svg>
    </div>
  );
}

// ============================================================
// Code block (per-algo)
// ============================================================
function CodeBlock({ algo }) {
  const code =
    algo === "vi"
      ? `# Value iteration (synchronous Bellman backups)
import numpy as np
V = np.zeros((H, W))
gamma = 0.9
for sweep in range(100):
    V_new = V.copy()
    for y in range(H):
        for x in range(W):
            if is_terminal(x, y):
                continue
            best = -np.inf
            for a in range(4):
                nx, ny, r, done = step(x, y, a)
                val = r + (0 if done else gamma * V[ny, nx])
                best = max(best, val)
            V_new[y, x] = best
    if np.max(np.abs(V_new - V)) < 1e-4:
        break
    V = V_new`
      : algo === "pi"
      ? `# Policy iteration (eval + improvement)
policy = np.ones((H, W), dtype=int)   # action 1 = right
V = np.zeros((H, W))
gamma = 0.9
while True:
    # Policy evaluation
    for _ in range(100):
        V_new = V.copy()
        for y in range(H):
            for x in range(W):
                if is_terminal(x, y): continue
                a = policy[y, x]
                nx, ny, r, done = step(x, y, a)
                V_new[y, x] = r + (0 if done else gamma * V[ny, nx])
        if np.max(np.abs(V_new - V)) < 1e-4: break
        V = V_new
    # Policy improvement
    stable = True
    for y in range(H):
        for x in range(W):
            if is_terminal(x, y): continue
            best_a = max(range(4), key=lambda a:
                step(x, y, a)[2] +
                (0 if step(x, y, a)[3] else gamma * V[step(x, y, a)[1], step(x, y, a)[0]]))
            if best_a != policy[y, x]: stable = False
            policy[y, x] = best_a
    if stable: break`
      : `# Tabular Q-learning (off-policy, model-free)
import numpy as np
Q = np.zeros((H, W, 4))
alpha, gamma, epsilon = 0.1, 0.9, 0.2
for episode in range(800):
    x, y = START
    while not is_terminal(x, y):
        if np.random.rand() < epsilon:
            a = np.random.randint(4)
        else:
            a = np.argmax(Q[y, x])
        nx, ny, r, done = step(x, y, a)
        target = r + (0 if done else gamma * np.max(Q[ny, nx]))
        Q[y, x, a] += alpha * (target - Q[y, x, a])
        x, y = nx, ny

# Greedy policy
policy = Q.argmax(axis=2)`;
  return (
    <pre
      style={{
        background: "#1f1d1a",
        color: "#e8e2d4",
        padding: 12,
        borderRadius: 8,
        fontSize: 12,
        fontFamily: "'JetBrains Mono', Menlo, monospace",
        lineHeight: 1.55,
        whiteSpace: "pre",
        overflowX: "auto",
        marginTop: 12,
      }}
    >
      {code}
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
          <b>Bellman equation = fixed-point equation.</b> The optimal
          value function is the unique fixed point of the Bellman
          operator <Tex>{`T^\\star V = \\max_a (r + \\gamma V')`}</Tex>.
          Value iteration repeatedly applies it; policy iteration
          alternates evaluation and improvement.
        </li>
        <li>
          <b>Convergence.</b> Both VI and PI converge in a finite number
          of iterations under discounting. PI typically takes fewer
          OUTER iterations (often ≤ |S|·|A|) but each one solves a
          full LP system. VI is simpler but may take more sweeps.
        </li>
        <li>
          <b>Why model-free?</b> Q-learning never asks 'what's r(s, a)?'
          — it just observes transitions and updates. Crucial when the
          MDP is unknown (real-world agents) or too big to enumerate
          (Atari, MuJoCo).
        </li>
        <li>
          <b>ε-greedy.</b> Mix exploitation (greedy action) with random
          exploration to keep visiting all states. Modern variants
          replace ε with entropy regularization or upper-confidence
          bonuses.
        </li>
        <li>
          <b>Connection to LP.</b> The dual of the average-reward MDP
          is a linear program with one variable per state — that's
          where the LP heritage of OR meets RL.
        </li>
        <li>
          <b>Function approximation.</b> Replace the table V(s) with a
          neural net V_θ(s). Value iteration becomes 'fitted value
          iteration' (DQN), policy iteration becomes 'actor-critic'
          (A2C/A3C/PPO/SAC).
        </li>
      </ul>
    </div>
  );
}

const panel = { background: "#fafafa", border: "1px solid #ddd", borderRadius: 8, padding: 12 };
const problemBox = { marginBottom: 16, padding: "12px 16px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8 };
const btn = { display: "inline-flex", alignItems: "center", gap: 6, padding: "6px 12px", borderRadius: 6, border: "1px solid #ccc", background: "#f7f7f7", cursor: "pointer", fontWeight: 500, fontSize: 13 };
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
