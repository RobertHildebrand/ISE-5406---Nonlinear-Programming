import React, { useState, useEffect, useMemo, useRef } from "react";
import { Play, Pause, RotateCcw, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   HEURISTICS & METAHEURISTICS
   ISE 5406

   Three problem instances + three search strategies.

     Problems:
       • TSP (15 random cities)
       • 0-1 Knapsack (15 items)

     Strategies:
       • Hill-climbing / 2-opt local search
       • Simulated annealing (with temperature schedule)
       • Genetic algorithm (knapsack only — TSP variant left as
         exercise)

   Animation: every step a small SVG redraw. For TSP we redraw
   the tour; for knapsack we show items being included/excluded
   and the running best.
   ============================================================ */

// ============================================================
// TSP instance
// ============================================================
const N_CITIES = 15;
const RNG_SEED = 42;
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6d2b79f5) >>> 0;
    let r = t;
    r = Math.imul(r ^ (r >>> 15), r | 1);
    r ^= r + Math.imul(r ^ (r >>> 7), r | 61);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
const TSP_RNG = mulberry32(RNG_SEED);
const CITIES = Array.from({ length: N_CITIES }, () => ({
  x: 0.1 + 0.8 * TSP_RNG(),
  y: 0.1 + 0.8 * TSP_RNG(),
}));

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}
function tourLen(tour) {
  let s = 0;
  for (let i = 0; i < tour.length; i++) {
    s += dist(CITIES[tour[i]], CITIES[tour[(i + 1) % tour.length]]);
  }
  return s;
}

// 2-opt swap: reverse tour[i..j]
function twoOptSwap(tour, i, j) {
  const newTour = [...tour];
  while (i < j) {
    [newTour[i], newTour[j]] = [newTour[j], newTour[i]];
    i++;
    j--;
  }
  return newTour;
}

// ============================================================
// Knapsack instance
// ============================================================
const N_ITEMS = 15;
const KS_RNG = mulberry32(7);
const ITEMS = Array.from({ length: N_ITEMS }, (_, i) => ({
  id: i,
  weight: 5 + Math.floor(KS_RNG() * 30),
  value: 10 + Math.floor(KS_RNG() * 90),
}));
const CAPACITY = 100;

function ksValue(soln) {
  let v = 0, w = 0;
  for (let i = 0; i < soln.length; i++) {
    if (soln[i]) {
      v += ITEMS[i].value;
      w += ITEMS[i].weight;
    }
  }
  return { v, w, feasible: w <= CAPACITY };
}

// ============================================================
// Search strategies
// ============================================================
function* hillClimbTSP(tour, maxIters) {
  let cur = [...tour];
  let bestLen = tourLen(cur);
  for (let iter = 0; iter < maxIters; iter++) {
    let improved = false;
    let bestSwap = null;
    for (let i = 1; i < cur.length - 1; i++) {
      for (let j = i + 1; j < cur.length; j++) {
        const candidate = twoOptSwap(cur, i, j);
        const cl = tourLen(candidate);
        if (cl < bestLen - 1e-9 && (!bestSwap || cl < bestSwap.len)) {
          bestSwap = { i, j, len: cl, tour: candidate };
        }
      }
    }
    if (bestSwap) {
      cur = bestSwap.tour;
      bestLen = bestSwap.len;
      improved = true;
      yield { tour: cur, len: bestLen, accept: true, info: `2-opt swap (i=${bestSwap.i}, j=${bestSwap.j})` };
    }
    if (!improved) {
      yield { tour: cur, len: bestLen, accept: false, info: "local minimum reached", done: true };
      return;
    }
  }
}

function* simulatedAnnealingTSP(tour, maxIters, T0, alpha) {
  let cur = [...tour];
  let curLen = tourLen(cur);
  let best = [...cur];
  let bestLen = curLen;
  let T = T0;
  for (let iter = 0; iter < maxIters; iter++) {
    const i = 1 + Math.floor(TSP_RNG() * (cur.length - 2));
    const j = i + 1 + Math.floor(TSP_RNG() * (cur.length - 1 - i));
    const cand = twoOptSwap(cur, i, j);
    const candLen = tourLen(cand);
    const delta = candLen - curLen;
    let accept = false;
    if (delta < 0) accept = true;
    else if (TSP_RNG() < Math.exp(-delta / T)) accept = true;
    if (accept) {
      cur = cand;
      curLen = candLen;
      if (curLen < bestLen) {
        best = [...cur];
        bestLen = curLen;
      }
    }
    yield {
      tour: cur,
      len: curLen,
      best,
      bestLen,
      T,
      delta,
      accept,
      info: accept
        ? delta < 0
          ? `improvement (Δ=${delta.toFixed(3)})`
          : `accepted worse (P=e^(-Δ/T) = ${Math.exp(-delta / T).toFixed(3)})`
        : `rejected worse (Δ=${delta.toFixed(3)})`,
    };
    T *= alpha;
  }
}

function* geneticKS(popSize, maxIters) {
  // Random initial population
  const rng = mulberry32(13);
  let pop = Array.from({ length: popSize }, () =>
    Array.from({ length: N_ITEMS }, () => rng() < 0.5)
  );
  for (let iter = 0; iter < maxIters; iter++) {
    // Repair / fitness
    const scored = pop.map((s) => {
      const { v, w, feasible } = ksValue(s);
      return { s, v: feasible ? v : 0, w };
    });
    scored.sort((a, b) => b.v - a.v);
    const best = scored[0];
    yield {
      pop: scored,
      bestSoln: best.s,
      bestValue: best.v,
      bestWeight: best.w,
      info: `Generation ${iter + 1} | best = ${best.v}`,
    };
    // Tournament selection + uniform crossover + mutation
    const newPop = [];
    while (newPop.length < popSize) {
      const a = scored[Math.floor(rng() * scored.length / 2)];
      const b = scored[Math.floor(rng() * scored.length / 2)];
      const child = a.s.map((_, i) => (rng() < 0.5 ? a.s[i] : b.s[i]));
      // Mutation
      for (let i = 0; i < N_ITEMS; i++) {
        if (rng() < 0.05) child[i] = !child[i];
      }
      newPop.push(child);
    }
    pop = newPop;
  }
}

// ============================================================
// Main component
// ============================================================
export default function HeuristicsDemo() {
  const [problem, setProblem] = useState("tsp");
  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Heuristics &amp; Metaheuristics
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        When you can't afford an exact solver, search the solution space
        directly. Three strategies on two classic problems: 2-opt local
        search and simulated annealing on TSP, and a genetic algorithm on
        0-1 knapsack. Each step animates the current solution evolving.
      </p>

      <div style={{ marginBottom: 12 }}>
        <label style={{ marginRight: 14, fontSize: 13 }}>
          <input type="radio" checked={problem === "tsp"} onChange={() => setProblem("tsp")} />
          &nbsp;TSP (15 cities)
        </label>
        <label style={{ fontSize: 13 }}>
          <input type="radio" checked={problem === "ks"} onChange={() => setProblem("ks")} />
          &nbsp;0-1 Knapsack (15 items)
        </label>
      </div>

      {problem === "tsp" ? <TSPDemo /> : <KnapsackDemo />}

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// TSP demo
// ============================================================
function TSPDemo() {
  const [algo, setAlgo] = useState("sa");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(50);
  const [step, setStep] = useState(null);
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);
  const initialTour = useMemo(() => Array.from({ length: N_CITIES }, (_, i) => i), []);

  function reset() {
    setRunning(false);
    setStep(null);
    setHistory([]);
    genRef.current = null;
  }

  function start() {
    if (algo === "hill") {
      genRef.current = hillClimbTSP(initialTour, 1000);
    } else {
      genRef.current = simulatedAnnealingTSP(initialTour, 800, 0.5, 0.99);
    }
    setHistory([]);
    setStep(null);
    setRunning(true);
  }

  useEffect(() => {
    if (!running || !genRef.current) return;
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) {
        setRunning(false);
        return;
      }
      setStep(next.value);
      setHistory((h) => [...h, next.value]);
      if (next.value.done) setRunning(false);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed]);

  // Initial tour for display
  const displayed = step
    ? algo === "sa"
      ? { tour: step.best, len: step.bestLen }
      : { tour: step.tour, len: step.len }
    : { tour: initialTour, len: tourLen(initialTour) };

  return (
    <>
      <div style={{ marginBottom: 12, padding: "10px 12px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8 }}>
        <label style={{ marginRight: 14, fontSize: 13 }}>
          <input type="radio" checked={algo === "hill"} onChange={() => setAlgo("hill")} />
          &nbsp;Hill-climbing (2-opt local search)
        </label>
        <label style={{ fontSize: 13 }}>
          <input type="radio" checked={algo === "sa"} onChange={() => setAlgo("sa")} />
          &nbsp;Simulated annealing
        </label>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)", gap: 22 }}>
        <TourPlot tour={displayed.tour} step={step} algo={algo} />
        <div>
          <Controls running={running} onStart={start} onStop={() => setRunning(false)} onReset={reset} speed={speed} setSpeed={setSpeed} />
          <TSPState step={step} initialLen={tourLen(initialTour)} algo={algo} displayed={displayed} />
          <ProgressChart history={history} algo={algo} />
        </div>
      </div>

      <CodePanelTSP algo={algo} />
    </>
  );
}

function TourPlot({ tour, step, algo }) {
  const W = 480, H = 480;
  const xs = (x) => 30 + x * (W - 60);
  const ys = (y) => 30 + y * (H - 60);
  const path = tour
    .concat(tour[0])
    .map((i) => `${xs(CITIES[i].x)},${ys(CITIES[i].y)}`)
    .join(" ");
  return (
    <div style={panel}>
      <svg width={W} height={H}>
        {/* Tour */}
        <polyline points={path} fill="none" stroke="#0b3da0" strokeWidth={2} />
        {/* Best tour overlay (SA only) */}
        {step && algo === "sa" && step.tour !== step.best && (
          <polyline
            points={step.tour.concat(step.tour[0]).map((i) => `${xs(CITIES[i].x)},${ys(CITIES[i].y)}`).join(" ")}
            fill="none"
            stroke="#c8311c"
            strokeWidth={1.5}
            strokeDasharray="4,3"
            opacity={0.5}
          />
        )}
        {/* Cities */}
        {CITIES.map((c, i) => (
          <g key={i}>
            <circle cx={xs(c.x)} cy={ys(c.y)} r={6} fill="#fff" stroke="#1f4e3d" strokeWidth={2} />
            <text x={xs(c.x)} y={ys(c.y) + 3} textAnchor="middle" fontSize={9} fontFamily="monospace" fill="#1f4e3d">
              {i}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function TSPState({ step, initialLen, algo, displayed }) {
  return (
    <div style={{ ...panel, marginBottom: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        State
      </div>
      <KV k="initial tour length" v={initialLen.toFixed(4)} />
      {step && algo === "sa" && (
        <>
          <KV k="current tour length" v={step.len.toFixed(4)} />
          <KV k="best tour length" v={step.bestLen.toFixed(4)} highlight />
          <KV k="temperature T" v={step.T.toFixed(4)} />
          <KV k="last move" v={step.info} />
        </>
      )}
      {step && algo === "hill" && (
        <>
          <KV k="current tour length" v={step.len.toFixed(4)} highlight />
          <KV k="last move" v={step.info} />
        </>
      )}
      {!step && <KV k="status" v="press Start to begin search" />}
    </div>
  );
}

function ProgressChart({ history, algo }) {
  if (history.length === 0) return null;
  const W = 460, H = 160, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const lens = history.map((h) => (algo === "sa" ? h.len : h.len));
  const bests = history.map((h) => (algo === "sa" ? h.bestLen : h.len));
  const yMin = Math.min(...lens, ...bests);
  const yMax = Math.max(...lens, ...bests);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        tour length over iterations
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        {algo === "sa" && (
          <polyline
            points={lens.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")}
            fill="none"
            stroke="#888"
            strokeWidth={1}
          />
        )}
        <polyline
          points={bests.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")}
          fill="none"
          stroke="#c8311c"
          strokeWidth={2}
        />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMax.toFixed(2)}
        </text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMin.toFixed(2)}
        </text>
        {algo === "sa" && (
          <g transform={`translate(${padL + 10}, ${padT + 4})`}>
            <line x1={0} y1={6} x2={14} y2={6} stroke="#888" strokeWidth={1} />
            <text x={18} y={10} fontSize={10}>current</text>
            <line x1={60} y1={6} x2={74} y2={6} stroke="#c8311c" strokeWidth={2} />
            <text x={78} y={10} fontSize={10}>best</text>
          </g>
        )}
      </svg>
    </div>
  );
}

// ============================================================
// Knapsack demo
// ============================================================
function KnapsackDemo() {
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(150);
  const [step, setStep] = useState(null);
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);

  function reset() {
    setRunning(false);
    setStep(null);
    setHistory([]);
    genRef.current = null;
  }
  function start() {
    genRef.current = geneticKS(40, 50);
    setHistory([]);
    setStep(null);
    setRunning(true);
  }

  useEffect(() => {
    if (!running || !genRef.current) return;
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) {
        setRunning(false);
        return;
      }
      setStep(next.value);
      setHistory((h) => [...h, next.value]);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed]);

  return (
    <>
      <div style={{ marginBottom: 12, padding: "10px 12px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8, fontSize: 13 }}>
        Genetic algorithm: tournament selection + uniform crossover + 5%
        bit-flip mutation. Population of 40, 50 generations.
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)", gap: 22 }}>
        <ItemsView step={step} />
        <div>
          <Controls running={running} onStart={start} onStop={() => setRunning(false)} onReset={reset} speed={speed} setSpeed={setSpeed} />
          <KSStats step={step} />
          <KSProgress history={history} />
        </div>
      </div>
      <CodePanelKS />
    </>
  );
}

function ItemsView({ step }) {
  const sel = step ? step.bestSoln : null;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Items (best solution highlighted)
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #ccc" }}>
            <th style={{ textAlign: "left", padding: 4 }}>id</th>
            <th style={{ textAlign: "right", padding: 4 }}>weight</th>
            <th style={{ textAlign: "right", padding: 4 }}>value</th>
            <th style={{ textAlign: "right", padding: 4 }}>v/w</th>
            <th style={{ textAlign: "center", padding: 4 }}>picked?</th>
          </tr>
        </thead>
        <tbody>
          {ITEMS.map((it, i) => (
            <tr
              key={it.id}
              style={{
                borderBottom: "1px dotted #eee",
                background: sel && sel[i] ? "#fff4c8" : "transparent",
              }}
            >
              <td style={{ padding: 4 }}>{it.id}</td>
              <td style={{ padding: 4, textAlign: "right" }}>{it.weight}</td>
              <td style={{ padding: 4, textAlign: "right" }}>{it.value}</td>
              <td style={{ padding: 4, textAlign: "right" }}>
                {(it.value / it.weight).toFixed(2)}
              </td>
              <td style={{ padding: 4, textAlign: "center", fontWeight: 700, color: sel && sel[i] ? "#c8311c" : "#aaa" }}>
                {sel ? (sel[i] ? "✓" : "·") : "·"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function KSStats({ step }) {
  const W = 460, H = 30;
  if (!step) return null;
  const ratio = Math.min(1, step.bestWeight / CAPACITY);
  return (
    <div style={{ ...panel, marginBottom: 12 }}>
      <KV k="generation" v={step.info} />
      <KV k="best value" v={step.bestValue} highlight />
      <KV k="best weight" v={`${step.bestWeight} / ${CAPACITY}`} />
      <svg width={W} height={H} style={{ marginTop: 6 }}>
        <rect x={0} y={0} width={W} height={H} fill="#eee" />
        <rect
          x={0}
          y={0}
          width={W * ratio}
          height={H}
          fill={step.bestWeight > CAPACITY ? "#c8311c" : "#1f4e3d"}
        />
        <line x1={W * (CAPACITY / CAPACITY)} y1={0} x2={W * (CAPACITY / CAPACITY)} y2={H} stroke="#444" strokeDasharray="3,3" />
        <text x={W / 2} y={H / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#fff" fontWeight={700}>
          weight {step.bestWeight} / cap {CAPACITY}
        </text>
      </svg>
    </div>
  );
}

function KSProgress({ history }) {
  if (history.length === 0) return null;
  const W = 460, H = 160, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const bests = history.map((h) => h.bestValue);
  const yMin = Math.min(...bests);
  const yMax = Math.max(...bests);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        best fitness over generations
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <polyline
          points={bests.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")}
          fill="none"
          stroke="#c8311c"
          strokeWidth={2}
        />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMax}
        </text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
          {yMin}
        </text>
      </svg>
    </div>
  );
}

// ============================================================
// Controls (shared)
// ============================================================
function Controls({ running, onStart, onStop, onReset, speed, setSpeed }) {
  return (
    <div style={{ ...panel, marginBottom: 12 }}>
      <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
        {!running ? (
          <button onClick={onStart} style={btnPrimary}>
            <Play size={14} /> Start
          </button>
        ) : (
          <button onClick={onStop} style={btn}>
            <Pause size={14} /> Pause
          </button>
        )}
        <button onClick={onReset} style={btn}>
          <RotateCcw size={14} /> Reset
        </button>
      </div>
      <label style={{ fontSize: 12, fontFamily: "monospace", color: "#444" }}>
        speed: <b>{speed} ms/step</b>
      </label>
      <input
        type="range"
        min={20}
        max={500}
        step={10}
        value={speed}
        onChange={(e) => setSpeed(+e.target.value)}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function KV({ k, v, highlight }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px dotted #eee" }}>
      <span style={{ color: "#666", fontSize: 12, fontFamily: "monospace" }}>{k}</span>
      <span style={{ fontSize: 12, fontFamily: "monospace", color: highlight ? "#c8311c" : "#222", fontWeight: highlight ? 700 : 400 }}>
        {v}
      </span>
    </div>
  );
}

// ============================================================
// Code panels
// ============================================================
function CodePanelTSP({ algo }) {
  const code =
    algo === "hill"
      ? `# 2-opt local search
def hill_climb(tour):
    while True:
        best_swap = None
        for i in range(1, len(tour)-1):
            for j in range(i+1, len(tour)):
                cand = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                if length(cand) < length(tour):
                    if best_swap is None or length(cand) < best_swap[1]:
                        best_swap = (cand, length(cand))
        if best_swap is None:
            return tour       # local optimum
        tour = best_swap[0]
`
      : `# Simulated annealing
import math, random
def sa(tour, T0=0.5, alpha=0.99, n_iter=800):
    cur, cur_len = tour, length(tour)
    best, best_len = cur, cur_len
    T = T0
    for k in range(n_iter):
        i = random.randint(1, len(cur)-2)
        j = random.randint(i+1, len(cur)-1)
        cand = cur[:i] + cur[i:j+1][::-1] + cur[j+1:]
        delta = length(cand) - cur_len
        if delta < 0 or random.random() < math.exp(-delta / T):
            cur, cur_len = cand, length(cand)
            if cur_len < best_len:
                best, best_len = cur, cur_len
        T *= alpha
    return best
`;
  return <CodeBlock code={code} />;
}
function CodePanelKS() {
  const code = `# Genetic algorithm for 0-1 knapsack
import random
def ga(items, capacity, pop_size=40, generations=50):
    pop = [[random.random() < 0.5 for _ in items] for _ in range(pop_size)]
    def fitness(s):
        v = sum(it.value for it, b in zip(items, s) if b)
        w = sum(it.weight for it, b in zip(items, s) if b)
        return v if w <= capacity else 0
    for g in range(generations):
        pop.sort(key=fitness, reverse=True)
        elite = pop[:pop_size // 4]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            a = random.choice(elite)
            b = random.choice(elite)
            child = [a[i] if random.random() < 0.5 else b[i]
                     for i in range(len(a))]
            for i in range(len(child)):
                if random.random() < 0.05:
                    child[i] = not child[i]
            new_pop.append(child)
        pop = new_pop
    return max(pop, key=fitness)
`;
  return <CodeBlock code={code} />;
}

function CodeBlock({ code }) {
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
          <b>Local search vs metaheuristic.</b> Hill climbing converges to
          a local optimum and stops. Metaheuristics (SA, GA, tabu search,
          ant colony) add randomness or memory to escape local optima.
        </li>
        <li>
          <b>Simulated annealing.</b> Accepts worse moves with probability{" "}
          <Tex>{String.raw`P = \exp(-\Delta / T)`}</Tex>. As{" "}
          <Tex>{`T`}</Tex> cools, the algorithm becomes greedier.
          Provably converges to the global optimum if{" "}
          <Tex>{`T`}</Tex> cools logarithmically — but that's exponentially slow.
          In practice geometric cooling <Tex>{`T_{k+1} = \\alpha T_k`}</Tex>{" "}
          with <Tex>{String.raw`\alpha \in [0.95, 0.999]`}</Tex> works well.
        </li>
        <li>
          <b>Genetic algorithms.</b> Population-based: keep N candidate
          solutions, evaluate fitness, breed via selection + crossover +
          mutation. The selection pressure controls exploration vs
          exploitation. Elitism (keeping the best) prevents regression.
        </li>
        <li>
          <b>2-opt for TSP.</b> Reverses a tour segment between cities
          i and j. Equivalent to deleting two edges and reconnecting.{" "}
          <Tex>{`O(n^2)`}</Tex> moves per iteration; with{" "}
          <Tex>{`n=15`}</Tex> it's ~100 candidates per step.
        </li>
        <li>
          <b>When to use what.</b> Small problems (n ≤ 50 cities) → exact
          MIP via Gurobi/CPLEX is fast. Medium (n ≤ 500) → LKH heuristic,
          Concorde with cutting planes. Huge (n &gt; 1000) →
          metaheuristics dominate. ML-augmented variants (graph neural
          networks + RL) are state of the art.
        </li>
      </ul>
    </div>
  );
}

// ============================================================
// Style atoms
// ============================================================
const panel = {
  background: "#fafafa",
  border: "1px solid #ddd",
  borderRadius: 8,
  padding: 12,
};
const btn = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "6px 12px",
  borderRadius: 6,
  border: "1px solid #ccc",
  background: "#f7f7f7",
  cursor: "pointer",
  fontWeight: 500,
  fontSize: 13,
};
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
