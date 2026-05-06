import React, { useState, useEffect, useMemo, useRef } from "react";
import { Play, Pause, RotateCcw, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   HEURISTICS & METAHEURISTICS
   ISE 5406

   Three problem instances, each with multiple search strategies
   so students can compare them on the same data.

     Problems:
       • TSP (15 random cities)            — tour length
       • 0-1 Knapsack (15 items)           — value subject to capacity
       • Bin Packing (16 items)            — # bins to use

     Strategies (across the three problems):
       • Constructive heuristics:
           Nearest Neighbor (TSP)
           Greedy by value/weight (Knapsack)
           First-Fit / Best-Fit Decreasing (Bin Packing)
       • Local search:
           Hill-climbing / 2-opt (TSP)
           Bit-flip hill-climbing (Knapsack)
       • Metaheuristics:
           Simulated annealing (TSP, Knapsack, Bin Packing)
           Tabu search (TSP, Knapsack)
           Genetic algorithm (TSP, Knapsack)

   Each step animates the current solution + best-so-far. Side
   panels show iteration log, temperature schedule (SA), tabu list
   (tabu), and population diversity (GA).
   ============================================================ */

// ============================================================
// Shared RNG
// ============================================================
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

// ============================================================
// TSP instance
// ============================================================
const N_CITIES = 15;
const CITIES = (() => {
  const r = mulberry32(42);
  return Array.from({ length: N_CITIES }, () => ({
    x: 0.1 + 0.8 * r(),
    y: 0.1 + 0.8 * r(),
  }));
})();

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function tourLen(tour) {
  let s = 0;
  for (let i = 0; i < tour.length; i++) {
    s += dist(CITIES[tour[i]], CITIES[tour[(i + 1) % tour.length]]);
  }
  return s;
}
function twoOptSwap(tour, i, j) {
  const t = [...tour];
  while (i < j) {
    [t[i], t[j]] = [t[j], t[i]];
    i++; j--;
  }
  return t;
}

// ============================================================
// Knapsack instance
// ============================================================
const N_ITEMS = 15;
const ITEMS = (() => {
  const r = mulberry32(7);
  return Array.from({ length: N_ITEMS }, (_, i) => ({
    id: i,
    weight: 5 + Math.floor(r() * 30),
    value: 10 + Math.floor(r() * 90),
  }));
})();
const CAPACITY = 100;

function ksValue(soln) {
  let v = 0, w = 0;
  for (let i = 0; i < soln.length; i++) {
    if (soln[i]) { v += ITEMS[i].value; w += ITEMS[i].weight; }
  }
  return { v, w, feasible: w <= CAPACITY };
}

// Exact LP-relaxation upper bound (for comparison) — sort by ratio, fill greedy continuously
const KS_LP_UB = (() => {
  const sorted = [...ITEMS].sort((a, b) => b.value / b.weight - a.value / a.weight);
  let v = 0, w = 0;
  for (const it of sorted) {
    if (w + it.weight <= CAPACITY) { v += it.value; w += it.weight; }
    else { v += (it.value * (CAPACITY - w)) / it.weight; break; }
  }
  return v;
})();

// ============================================================
// Bin packing instance
// ============================================================
const N_BP_ITEMS = 16;
const BP_CAPACITY = 10;
const BP_ITEMS = (() => {
  const r = mulberry32(99);
  return Array.from({ length: N_BP_ITEMS }, (_, i) => ({
    id: i,
    size: 1 + Math.floor(r() * 6) + (r() < 0.3 ? 2 : 0), // 1..8 with skew
  }));
})();

// Lower bound: ceil(sum(sizes) / capacity)
const BP_LB = Math.ceil(BP_ITEMS.reduce((s, it) => s + it.size, 0) / BP_CAPACITY);

function bpBins(perm) {
  // Pack items in order into bins (next-fit-style), each bin closes when next item won't fit.
  // Used inside SA: a permutation determines the packing.
  const bins = [];
  for (const idx of perm) {
    const sz = BP_ITEMS[idx].size;
    let placed = false;
    for (const b of bins) {
      if (b.used + sz <= BP_CAPACITY) {
        b.items.push(idx); b.used += sz; placed = true; break;
      }
    }
    if (!placed) bins.push({ items: [idx], used: sz });
  }
  return bins;
}

// ============================================================
// TSP search strategies
// ============================================================
function* nearestNeighborTSP() {
  // Constructive — start from city 0, always move to nearest unvisited.
  const visited = new Set([0]);
  const tour = [0];
  yield { tour: [...tour], len: 0, info: "Start at city 0" };
  while (visited.size < N_CITIES) {
    let last = tour[tour.length - 1], best = -1, bd = Infinity;
    for (let i = 0; i < N_CITIES; i++) {
      if (visited.has(i)) continue;
      const d = dist(CITIES[last], CITIES[i]);
      if (d < bd) { bd = d; best = i; }
    }
    tour.push(best); visited.add(best);
    yield {
      tour: [...tour], len: tourLen([...tour, ...Array.from({ length: 0 })]),
      info: `Visit city ${best} (dist ${bd.toFixed(3)} from ${last})`,
    };
  }
  yield { tour: [...tour], len: tourLen(tour), info: "Tour complete (return to start)", done: true };
}

function* hillClimbTSP(tour) {
  let cur = [...tour];
  let bestLen = tourLen(cur);
  for (let iter = 0; iter < 1000; iter++) {
    let bestSwap = null;
    for (let i = 1; i < cur.length - 1; i++) {
      for (let j = i + 1; j < cur.length; j++) {
        const cand = twoOptSwap(cur, i, j);
        const cl = tourLen(cand);
        if (cl < bestLen - 1e-9 && (!bestSwap || cl < bestSwap.len)) {
          bestSwap = { i, j, len: cl, tour: cand };
        }
      }
    }
    if (!bestSwap) {
      yield { tour: cur, len: bestLen, accept: false, info: "local minimum reached", done: true };
      return;
    }
    cur = bestSwap.tour; bestLen = bestSwap.len;
    yield { tour: cur, len: bestLen, accept: true, info: `2-opt swap (i=${bestSwap.i}, j=${bestSwap.j})` };
  }
}

function* simulatedAnnealingTSP(tour, T0 = 0.5, alpha = 0.99, nIter = 800) {
  const rng = mulberry32(101);
  let cur = [...tour];
  let curLen = tourLen(cur);
  let best = [...cur], bestLen = curLen;
  let T = T0;
  let accCount = 0, rejCount = 0;
  for (let iter = 0; iter < nIter; iter++) {
    const i = 1 + Math.floor(rng() * (cur.length - 2));
    const j = i + 1 + Math.floor(rng() * (cur.length - 1 - i));
    const cand = twoOptSwap(cur, i, j);
    const candLen = tourLen(cand);
    const delta = candLen - curLen;
    let accept = false;
    let pAccept = 1;
    if (delta < 0) accept = true;
    else { pAccept = Math.exp(-delta / T); accept = rng() < pAccept; }
    if (accept) { cur = cand; curLen = candLen; accCount++; if (curLen < bestLen) { best = [...cur]; bestLen = curLen; } }
    else rejCount++;
    yield {
      tour: cur, len: curLen, best, bestLen, T, delta, accept, pAccept,
      accRate: accCount / (accCount + rejCount),
      info: accept
        ? delta < 0 ? `improvement (Δ=${delta.toFixed(3)})`
                   : `accepted worse (P=${pAccept.toFixed(3)})`
        : `rejected worse (Δ=${delta.toFixed(3)}, P=${pAccept.toFixed(3)})`,
    };
    T *= alpha;
  }
}

function* tabuSearchTSP(tour, tabuTenure = 12, nIter = 400) {
  let cur = [...tour];
  let curLen = tourLen(cur);
  let best = [...cur], bestLen = curLen;
  // Tabu list stores recently-used (i, j) pivots
  const tabu = []; // [{i,j,exp}]
  for (let iter = 0; iter < nIter; iter++) {
    let bestMove = null;
    for (let i = 1; i < cur.length - 1; i++) {
      for (let j = i + 1; j < cur.length; j++) {
        const isTabu = tabu.some((t) => t.i === i && t.j === j);
        const cand = twoOptSwap(cur, i, j);
        const cl = tourLen(cand);
        // Aspiration: accept if it beats the global best
        const aspires = cl < bestLen - 1e-9;
        if (isTabu && !aspires) continue;
        if (!bestMove || cl < bestMove.len) bestMove = { i, j, len: cl, tour: cand, aspires, wasTabu: isTabu };
      }
    }
    if (!bestMove) {
      yield { tour: cur, len: curLen, best, bestLen, tabu: [...tabu], info: "no non-tabu moves", done: true };
      return;
    }
    cur = bestMove.tour; curLen = bestMove.len;
    if (curLen < bestLen) { best = [...cur]; bestLen = curLen; }
    // Add (i,j) to tabu, decay older entries
    tabu.push({ i: bestMove.i, j: bestMove.j, exp: iter + tabuTenure });
    while (tabu.length && tabu[0].exp <= iter) tabu.shift();
    yield {
      tour: cur, len: curLen, best, bestLen, tabu: [...tabu],
      info: bestMove.aspires
        ? `aspiration override (i=${bestMove.i}, j=${bestMove.j}) — beats best`
        : bestMove.wasTabu
          ? "(should not reach)" : `move (i=${bestMove.i}, j=${bestMove.j})`,
    };
  }
}

function* geneticTSP(popSize = 30, generations = 60) {
  const rng = mulberry32(202);
  // Random initial population (random permutations starting at 0)
  const randPerm = () => {
    const arr = Array.from({ length: N_CITIES - 1 }, (_, i) => i + 1);
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return [0, ...arr];
  };
  let pop = Array.from({ length: popSize }, randPerm);
  // Order crossover (OX): pick a slice from parent A, fill remainder from B in order
  function ox(a, b) {
    const i = 1 + Math.floor(rng() * (N_CITIES - 2));
    const j = i + Math.floor(rng() * (N_CITIES - i));
    const child = Array(N_CITIES).fill(-1);
    for (let k = i; k <= j; k++) child[k] = a[k];
    let p = 0;
    for (let k = 0; k < N_CITIES; k++) {
      const idx = (j + 1 + k) % N_CITIES;
      while (p < N_CITIES && child.includes(b[p])) p++;
      if (child[idx] === -1) { child[idx] = b[p]; p++; }
    }
    // Rotate so 0 is first
    const z = child.indexOf(0);
    return [...child.slice(z), ...child.slice(0, z)];
  }
  // Swap mutation
  function mutate(t) {
    if (rng() > 0.3) return t;
    const r = [...t];
    const i = 1 + Math.floor(rng() * (N_CITIES - 1));
    const j = 1 + Math.floor(rng() * (N_CITIES - 1));
    [r[i], r[j]] = [r[j], r[i]];
    return r;
  }
  for (let gen = 0; gen < generations; gen++) {
    const scored = pop.map((t) => ({ t, len: tourLen(t) }));
    scored.sort((a, b) => a.len - b.len);
    const best = scored[0];
    const lenStats = {
      min: scored[0].len,
      mean: scored.reduce((s, x) => s + x.len, 0) / scored.length,
      max: scored[scored.length - 1].len,
    };
    yield {
      tour: best.t,
      len: best.len,
      lenStats,
      info: `Generation ${gen + 1} | best = ${best.len.toFixed(3)}, mean = ${lenStats.mean.toFixed(3)}`,
    };
    // Elitism
    const elite = scored.slice(0, Math.max(2, Math.floor(popSize / 5))).map((s) => s.t);
    const newPop = [...elite];
    while (newPop.length < popSize) {
      const a = scored[Math.floor(rng() * Math.floor(popSize / 2))].t;
      const b = scored[Math.floor(rng() * Math.floor(popSize / 2))].t;
      newPop.push(mutate(ox(a, b)));
    }
    pop = newPop;
  }
}

// ============================================================
// Knapsack search strategies
// ============================================================
function* greedyRatioKS() {
  // Sort items by value/weight, take while you can
  const sorted = [...ITEMS].sort((a, b) => b.value / b.weight - a.value / a.weight);
  const sel = Array(N_ITEMS).fill(false);
  let w = 0;
  yield { sel: [...sel], info: "Empty knapsack" };
  for (const it of sorted) {
    if (w + it.weight <= CAPACITY) {
      sel[it.id] = true; w += it.weight;
      const { v, w: ww } = ksValue(sel);
      yield { sel: [...sel], bestSoln: [...sel], bestValue: v, bestWeight: ww, info: `Take item ${it.id} (v=${it.value}, w=${it.weight}, ratio=${(it.value / it.weight).toFixed(2)})` };
    } else {
      yield { sel: [...sel], bestSoln: [...sel], bestValue: ksValue(sel).v, bestWeight: w, info: `Skip item ${it.id} (would overflow ${w + it.weight} > ${CAPACITY})` };
    }
  }
  const final = ksValue(sel);
  yield { sel: [...sel], bestSoln: [...sel], bestValue: final.v, bestWeight: final.w, info: "Done (greedy is not always optimal!)", done: true };
}

function* simulatedAnnealingKS(T0 = 30, alpha = 0.97, nIter = 200) {
  const rng = mulberry32(303);
  let cur = Array(N_ITEMS).fill(false);
  let curObj = ksObj(cur);
  let best = [...cur], bestObj = curObj;
  let T = T0;
  let accCount = 0, rejCount = 0;
  for (let iter = 0; iter < nIter; iter++) {
    const i = Math.floor(rng() * N_ITEMS);
    const cand = [...cur]; cand[i] = !cand[i];
    const candObj = ksObj(cand);
    const delta = curObj - candObj; // for max problem: delta > 0 means worse
    let accept = false, pAccept = 1;
    if (delta <= 0) accept = true;
    else { pAccept = Math.exp(-delta / T); accept = rng() < pAccept; }
    if (accept) { cur = cand; curObj = candObj; accCount++; if (curObj > bestObj) { best = [...cur]; bestObj = curObj; } }
    else rejCount++;
    const { v: bestValue, w: bestWeight } = ksValue(best);
    yield {
      sel: [...cur], curObj, bestSoln: [...best], bestValue, bestWeight,
      T, delta, accept, pAccept,
      accRate: accCount / (accCount + rejCount),
      info: accept
        ? delta <= 0 ? `flip ${i}: improvement (Δ_obj=${(-delta).toFixed(0)})`
                    : `flip ${i}: accepted worse (P=${pAccept.toFixed(3)})`
        : `flip ${i}: rejected (Δ_loss=${delta.toFixed(0)}, P=${pAccept.toFixed(3)})`,
    };
    T *= alpha;
  }
}
function ksObj(sel) {
  // Penalized objective: total_value if feasible, otherwise large negative penalty
  const { v, w, feasible } = ksValue(sel);
  if (feasible) return v;
  return v - 5 * (w - CAPACITY); // soft penalty
}

function* tabuSearchKS(tabuTenure = 7, nIter = 80) {
  let cur = Array(N_ITEMS).fill(false);
  let { v: curV, w: curW } = ksValue(cur);
  let best = [...cur], bestV = curV;
  const tabu = []; // {bit, exp}
  for (let iter = 0; iter < nIter; iter++) {
    let bestMove = null;
    for (let i = 0; i < N_ITEMS; i++) {
      const isTabu = tabu.some((t) => t.bit === i);
      const cand = [...cur]; cand[i] = !cand[i];
      const { v, w, feasible } = ksValue(cand);
      const score = feasible ? v : v - 1000;
      const aspires = feasible && v > bestV;
      if (isTabu && !aspires) continue;
      if (!bestMove || score > bestMove.score) bestMove = { i, score, sel: cand, v, w, feasible, aspires };
    }
    if (!bestMove) {
      yield { sel: cur, bestSoln: best, bestValue: bestV, tabu: [...tabu], info: "no non-tabu moves", done: true };
      return;
    }
    cur = bestMove.sel; curV = bestMove.v; curW = bestMove.w;
    if (bestMove.feasible && bestMove.v > bestV) { best = [...cur]; bestV = bestMove.v; }
    tabu.push({ bit: bestMove.i, exp: iter + tabuTenure });
    while (tabu.length && tabu[0].exp <= iter) tabu.shift();
    yield {
      sel: cur, curV, curW,
      bestSoln: best, bestValue: bestV, bestWeight: ksValue(best).w,
      tabu: [...tabu],
      info: bestMove.aspires
        ? `aspiration: flip ${bestMove.i} → new best ${bestMove.v}`
        : `flip ${bestMove.i} → ${bestMove.feasible ? `feasible v=${bestMove.v}` : "INFEASIBLE (penalized)"}`,
    };
  }
}

function* geneticKS(popSize = 40, generations = 50) {
  const rng = mulberry32(13);
  let pop = Array.from({ length: popSize }, () =>
    Array.from({ length: N_ITEMS }, () => rng() < 0.5)
  );
  for (let iter = 0; iter < generations; iter++) {
    const scored = pop.map((s) => {
      const { v, w, feasible } = ksValue(s);
      return { s, v: feasible ? v : 0, w };
    });
    scored.sort((a, b) => b.v - a.v);
    const best = scored[0];
    const fitnesses = scored.map((x) => x.v);
    yield {
      pop: scored, bestSoln: best.s, bestValue: best.v, bestWeight: best.w,
      diversity: {
        min: fitnesses[fitnesses.length - 1],
        mean: fitnesses.reduce((s, x) => s + x, 0) / fitnesses.length,
        max: fitnesses[0],
      },
      info: `Generation ${iter + 1} | best = ${best.v}, mean = ${(fitnesses.reduce((s, x) => s + x, 0) / fitnesses.length).toFixed(1)}`,
    };
    const newPop = [];
    while (newPop.length < popSize) {
      const a = scored[Math.floor(rng() * scored.length / 2)];
      const b = scored[Math.floor(rng() * scored.length / 2)];
      const child = a.s.map((_, i) => (rng() < 0.5 ? a.s[i] : b.s[i]));
      for (let i = 0; i < N_ITEMS; i++) if (rng() < 0.05) child[i] = !child[i];
      newPop.push(child);
    }
    pop = newPop;
  }
}

// ============================================================
// Bin packing search strategies
// ============================================================
function* firstFitDecreasing() {
  const sortedIdx = [...BP_ITEMS].sort((a, b) => b.size - a.size).map((it) => it.id);
  const bins = [];
  yield { bins: [], info: "Sorted items by descending size", remaining: sortedIdx };
  for (let k = 0; k < sortedIdx.length; k++) {
    const idx = sortedIdx[k];
    const sz = BP_ITEMS[idx].size;
    let placedIn = -1;
    for (let b = 0; b < bins.length; b++) {
      if (bins[b].used + sz <= BP_CAPACITY) { bins[b].items.push(idx); bins[b].used += sz; placedIn = b; break; }
    }
    if (placedIn < 0) { bins.push({ items: [idx], used: sz }); placedIn = bins.length - 1; }
    yield {
      bins: bins.map((b) => ({ items: [...b.items], used: b.used })),
      info: `Place item ${idx} (size ${sz}) in bin ${placedIn + 1}${placedIn === bins.length - 1 && bins[placedIn].items.length === 1 ? " (NEW)" : ""}`,
      remaining: sortedIdx.slice(k + 1),
    };
  }
  yield { bins, info: `Done — used ${bins.length} bins (lower bound ${BP_LB})`, done: true };
}

function* bestFitDecreasing() {
  const sortedIdx = [...BP_ITEMS].sort((a, b) => b.size - a.size).map((it) => it.id);
  const bins = [];
  yield { bins: [], info: "Sorted items by descending size", remaining: sortedIdx };
  for (let k = 0; k < sortedIdx.length; k++) {
    const idx = sortedIdx[k];
    const sz = BP_ITEMS[idx].size;
    // Find tightest-fitting bin
    let bestBin = -1, bestSlack = Infinity;
    for (let b = 0; b < bins.length; b++) {
      const slack = BP_CAPACITY - bins[b].used - sz;
      if (slack >= 0 && slack < bestSlack) { bestBin = b; bestSlack = slack; }
    }
    if (bestBin < 0) { bins.push({ items: [idx], used: sz }); bestBin = bins.length - 1; }
    else { bins[bestBin].items.push(idx); bins[bestBin].used += sz; }
    yield {
      bins: bins.map((b) => ({ items: [...b.items], used: b.used })),
      info: `Place item ${idx} (size ${sz}) in bin ${bestBin + 1} (slack ${BP_CAPACITY - bins[bestBin].used})`,
      remaining: sortedIdx.slice(k + 1),
    };
  }
  yield { bins, info: `Done — used ${bins.length} bins (lower bound ${BP_LB})`, done: true };
}

function* simulatedAnnealingBP(T0 = 4, alpha = 0.97, nIter = 200) {
  const rng = mulberry32(404);
  // Encode solution as a permutation; pack via next-fit-style.
  let perm = Array.from({ length: N_BP_ITEMS }, (_, i) => i).sort((a, b) => BP_ITEMS[b].size - BP_ITEMS[a].size);
  let curBins = bpBins(perm);
  let bestPerm = [...perm], bestBins = curBins.map((b) => ({ items: [...b.items], used: b.used }));
  let T = T0;
  for (let iter = 0; iter < nIter; iter++) {
    // Neighbor: swap two indices
    const i = Math.floor(rng() * N_BP_ITEMS);
    let j = Math.floor(rng() * N_BP_ITEMS);
    if (j === i) j = (j + 1) % N_BP_ITEMS;
    const cand = [...perm];
    [cand[i], cand[j]] = [cand[j], cand[i]];
    const candBins = bpBins(cand);
    const delta = candBins.length - curBins.length; // > 0 means worse
    let accept = false, pAccept = 1;
    if (delta <= 0) accept = true;
    else { pAccept = Math.exp(-delta / T); accept = rng() < pAccept; }
    if (accept) {
      perm = cand; curBins = candBins;
      if (curBins.length < bestBins.length) { bestPerm = [...perm]; bestBins = curBins.map((b) => ({ items: [...b.items], used: b.used })); }
    }
    yield {
      bins: curBins.map((b) => ({ items: [...b.items], used: b.used })),
      bestBins: bestBins.map((b) => ({ items: [...b.items], used: b.used })),
      curCount: curBins.length, bestCount: bestBins.length,
      T, delta, accept, pAccept,
      info: accept
        ? delta <= 0 ? `swap (${i},${j}): ${curBins.length} bins`
                    : `swap (${i},${j}): accepted worse (${candBins.length} bins, P=${pAccept.toFixed(3)})`
        : `swap (${i},${j}): rejected (${candBins.length} bins, P=${pAccept.toFixed(3)})`,
    };
    T *= alpha;
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
        When you can't afford an exact solver, search the solution space directly.
        Three classical combinatorial problems, each tackled by several heuristic
        families — constructive (greedy / nearest neighbor / first-fit-decreasing),
        local search (hill climbing), and metaheuristics (simulated annealing,
        tabu search, genetic algorithm). Switch problems and algorithms freely;
        every step is animated.
      </p>

      <div style={{ display: "flex", gap: 14, marginBottom: 12, flexWrap: "wrap" }}>
        {[
          { key: "tsp", label: "TSP — 15 cities" },
          { key: "ks", label: "0-1 Knapsack — 15 items" },
          { key: "bp", label: "Bin Packing — 16 items" },
        ].map((p) => (
          <label key={p.key} style={{ fontSize: 13, fontWeight: problem === p.key ? 700 : 400 }}>
            <input
              type="radio"
              checked={problem === p.key}
              onChange={() => setProblem(p.key)}
            />
            &nbsp;{p.label}
          </label>
        ))}
      </div>

      {problem === "tsp" && <TSPDemo />}
      {problem === "ks" && <KnapsackDemo />}
      {problem === "bp" && <BinPackingDemo />}

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// TSP demo
// ============================================================
const TSP_ALGOS = [
  { key: "nn", label: "Nearest Neighbor", kind: "constructive" },
  { key: "hill", label: "Hill Climbing (2-opt)", kind: "local" },
  { key: "sa", label: "Simulated Annealing", kind: "meta" },
  { key: "tabu", label: "Tabu Search", kind: "meta" },
  { key: "ga", label: "Genetic Algorithm", kind: "meta" },
];

function TSPDemo() {
  const [algo, setAlgo] = useState("sa");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(80);
  const [step, setStep] = useState(null);
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);
  const initialTour = useMemo(() => Array.from({ length: N_CITIES }, (_, i) => i), []);

  function reset() {
    setRunning(false); setStep(null); setHistory([]); genRef.current = null;
  }
  function start() {
    if (algo === "nn") genRef.current = nearestNeighborTSP();
    else if (algo === "hill") genRef.current = hillClimbTSP(initialTour);
    else if (algo === "sa") genRef.current = simulatedAnnealingTSP(initialTour);
    else if (algo === "tabu") genRef.current = tabuSearchTSP(initialTour);
    else if (algo === "ga") genRef.current = geneticTSP();
    setHistory([]); setStep(null); setRunning(true);
  }

  useEffect(() => {
    if (!running || !genRef.current) return;
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) { setRunning(false); return; }
      setStep(next.value);
      setHistory((h) => [...h, next.value]);
      if (next.value.done) setRunning(false);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed]);

  // Initial tour for display, plus tour overlay rules per algo
  const displayed = step
    ? algo === "sa" || algo === "tabu" || algo === "ga"
      ? { tour: step.best || step.tour, len: step.bestLen ?? step.len }
      : { tour: step.tour, len: step.len }
    : { tour: initialTour, len: tourLen(initialTour) };

  return (
    <>
      <AlgoSelector
        algos={TSP_ALGOS}
        algo={algo}
        setAlgo={(k) => { setAlgo(k); reset(); }}
      />

      <div style={{ display: "grid", gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)", gap: 22 }}>
        <TourPlot tour={displayed.tour} step={step} algo={algo} />
        <div>
          <Controls running={running} onStart={start} onStop={() => setRunning(false)} onReset={reset} speed={speed} setSpeed={setSpeed} />
          <TSPState step={step} initialLen={tourLen(initialTour)} algo={algo} displayed={displayed} />
          <ProgressChart history={history} algo={algo} kind="tsp" />
          {algo === "sa" && <SAExtras history={history} />}
          {algo === "tabu" && step && <TabuListView tabu={step.tabu || []} kind="tsp" />}
          {algo === "ga" && <GADiversityView history={history} kind="tsp" />}
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
  const path = tour.length === 0 ? "" : tour.concat(tour[0]).map((i) => `${xs(CITIES[i].x)},${ys(CITIES[i].y)}`).join(" ");
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        {algo === "nn" && step && !step.done ? "tour being built" : "tour"}
      </div>
      <svg width={W} height={H}>
        {path && <polyline points={path} fill="none" stroke="#0b3da0" strokeWidth={2} />}
        {step && (algo === "sa" || algo === "tabu") && step.tour !== step.best && (
          <polyline
            points={step.tour.concat(step.tour[0]).map((i) => `${xs(CITIES[i].x)},${ys(CITIES[i].y)}`).join(" ")}
            fill="none" stroke="#c8311c" strokeWidth={1.5} strokeDasharray="4,3" opacity={0.5}
          />
        )}
        {CITIES.map((c, i) => {
          const inTour = tour.includes(i);
          const isLast = tour.length > 0 && tour[tour.length - 1] === i;
          return (
            <g key={i}>
              <circle
                cx={xs(c.x)} cy={ys(c.y)}
                r={isLast ? 8 : 6}
                fill={isLast ? "#c8311c" : inTour ? "#fff" : "#eee"}
                stroke={inTour ? "#1f4e3d" : "#999"}
                strokeWidth={2}
              />
              <text x={xs(c.x)} y={ys(c.y) + 3} textAnchor="middle" fontSize={9} fontFamily="monospace" fill={inTour ? "#1f4e3d" : "#999"}>
                {i}
              </text>
            </g>
          );
        })}
        {step && step.info && (
          <text x={W - 12} y={H - 12} textAnchor="end" fontSize={11} fontFamily="monospace" fill="#444">
            {step.info}
          </text>
        )}
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
      {step && algo === "nn" && (
        <>
          <KV k="cities visited" v={`${step.tour.length} / ${N_CITIES}`} />
          <KV k="partial length" v={step.len.toFixed(4)} />
          <KV k="last action" v={step.info} />
        </>
      )}
      {step && algo === "hill" && (
        <>
          <KV k="current tour length" v={step.len.toFixed(4)} highlight />
          <KV k="last move" v={step.info} />
        </>
      )}
      {step && (algo === "sa" || algo === "tabu") && (
        <>
          <KV k="current tour length" v={step.len.toFixed(4)} />
          <KV k="best tour length" v={(step.bestLen ?? step.len).toFixed(4)} highlight />
          {algo === "sa" && (
            <>
              <KV k="temperature T" v={step.T.toFixed(4)} />
              <KV k="acceptance rate" v={`${(step.accRate * 100).toFixed(1)}%`} />
            </>
          )}
          <KV k="last move" v={step.info} />
        </>
      )}
      {step && algo === "ga" && (
        <>
          <KV k="best tour length" v={step.len.toFixed(4)} highlight />
          <KV k="population mean" v={step.lenStats?.mean.toFixed(4) || "—"} />
          <KV k="population worst" v={step.lenStats?.max.toFixed(4) || "—"} />
          <KV k="generation" v={step.info} />
        </>
      )}
      {!step && <KV k="status" v="press Start to begin search" />}
    </div>
  );
}

// ============================================================
// Knapsack demo
// ============================================================
const KS_ALGOS = [
  { key: "greedy", label: "Greedy by Ratio", kind: "constructive" },
  { key: "sa", label: "Simulated Annealing", kind: "meta" },
  { key: "tabu", label: "Tabu Search", kind: "meta" },
  { key: "ga", label: "Genetic Algorithm", kind: "meta" },
];

function KnapsackDemo() {
  const [algo, setAlgo] = useState("ga");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(120);
  const [step, setStep] = useState(null);
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);

  function reset() { setRunning(false); setStep(null); setHistory([]); genRef.current = null; }
  function start() {
    if (algo === "greedy") genRef.current = greedyRatioKS();
    else if (algo === "sa") genRef.current = simulatedAnnealingKS();
    else if (algo === "tabu") genRef.current = tabuSearchKS();
    else if (algo === "ga") genRef.current = geneticKS(40, 50);
    setHistory([]); setStep(null); setRunning(true);
  }
  useEffect(() => {
    if (!running || !genRef.current) return;
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) { setRunning(false); return; }
      setStep(next.value);
      setHistory((h) => [...h, next.value]);
      if (next.value.done) setRunning(false);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed]);

  return (
    <>
      <AlgoSelector algos={KS_ALGOS} algo={algo} setAlgo={(k) => { setAlgo(k); reset(); }} />

      <div style={{ display: "grid", gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)", gap: 22 }}>
        <ItemsView step={step} algo={algo} />
        <div>
          <Controls running={running} onStart={start} onStop={() => setRunning(false)} onReset={reset} speed={speed} setSpeed={setSpeed} />
          <KSStats step={step} algo={algo} />
          <KSProgress history={history} algo={algo} />
          {algo === "sa" && <SAExtras history={history} />}
          {algo === "tabu" && step && <TabuListView tabu={step.tabu || []} kind="ks" />}
          {algo === "ga" && <GADiversityView history={history} kind="ks" />}
        </div>
      </div>

      <CodePanelKS algo={algo} />
    </>
  );
}

function ItemsView({ step, algo }) {
  // For greedy, show items in greedy order (by ratio); otherwise by id.
  const order = algo === "greedy"
    ? [...ITEMS].sort((a, b) => b.value / b.weight - a.value / a.weight).map((it) => it.id)
    : ITEMS.map((it) => it.id);

  const sel = step ? (step.bestSoln || step.sel) : null;
  const cur = step ? step.sel : null;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Items {algo === "greedy" ? "(sorted by value / weight)" : ""}
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 12, borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #ccc" }}>
            <th style={{ textAlign: "left", padding: 4 }}>id</th>
            <th style={{ textAlign: "right", padding: 4 }}>weight</th>
            <th style={{ textAlign: "right", padding: 4 }}>value</th>
            <th style={{ textAlign: "right", padding: 4 }}>v/w</th>
            <th style={{ textAlign: "center", padding: 4 }}>best</th>
            {(algo === "sa" || algo === "tabu") && <th style={{ textAlign: "center", padding: 4 }}>cur</th>}
          </tr>
        </thead>
        <tbody>
          {order.map((idx) => {
            const it = ITEMS[idx];
            return (
              <tr key={it.id} style={{ borderBottom: "1px dotted #eee", background: sel && sel[it.id] ? "#fff4c8" : "transparent" }}>
                <td style={{ padding: 4 }}>{it.id}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{it.weight}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{it.value}</td>
                <td style={{ padding: 4, textAlign: "right" }}>{(it.value / it.weight).toFixed(2)}</td>
                <td style={{ padding: 4, textAlign: "center", fontWeight: 700, color: sel && sel[it.id] ? "#c8311c" : "#aaa" }}>
                  {sel ? (sel[it.id] ? "✓" : "·") : "·"}
                </td>
                {(algo === "sa" || algo === "tabu") && (
                  <td style={{ padding: 4, textAlign: "center", color: cur && cur[it.id] ? "#0b3da0" : "#aaa" }}>
                    {cur ? (cur[it.id] ? "●" : "·") : "·"}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function KSStats({ step, algo }) {
  const W = 460, H = 30;
  if (!step) return null;
  const bw = step.bestWeight ?? 0;
  const bv = step.bestValue ?? 0;
  const ratio = Math.min(1, bw / CAPACITY);
  return (
    <div style={{ ...panel, marginBottom: 12 }}>
      <KV k={algo === "ga" ? "generation" : "step"} v={step.info} />
      <KV k="best value" v={`${bv} / LP-UB ${KS_LP_UB.toFixed(1)}`} highlight />
      <KV k="best weight" v={`${bw} / ${CAPACITY}`} />
      {algo === "sa" && (
        <>
          <KV k="temperature T" v={step.T?.toFixed(3)} />
          <KV k="acceptance rate" v={`${((step.accRate || 0) * 100).toFixed(1)}%`} />
        </>
      )}
      <svg width={W} height={H} style={{ marginTop: 6 }}>
        <rect x={0} y={0} width={W} height={H} fill="#eee" />
        <rect x={0} y={0} width={W * ratio} height={H} fill={bw > CAPACITY ? "#c8311c" : "#1f4e3d"} />
        <text x={W / 2} y={H / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#fff" fontWeight={700}>
          weight {bw} / cap {CAPACITY}
        </text>
      </svg>
    </div>
  );
}

function KSProgress({ history, algo }) {
  if (history.length === 0) return null;
  const W = 460, H = 160, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const bests = history.map((h) => h.bestValue || 0);
  const yMin = 0, yMax = Math.max(...bests, KS_LP_UB);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        best fitness over {algo === "ga" ? "generations" : "iterations"}
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        {/* LP relaxation upper bound */}
        <line x1={padL} y1={ys(KS_LP_UB)} x2={padL + cw} y2={ys(KS_LP_UB)} stroke="#0b3da0" strokeDasharray="3,3" strokeWidth={1.5} />
        <text x={padL + cw - 4} y={ys(KS_LP_UB) - 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#0b3da0">
          LP UB ≈ {KS_LP_UB.toFixed(1)}
        </text>
        <polyline points={bests.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#c8311c" strokeWidth={2} />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMax.toFixed(0)}</text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMin.toFixed(0)}</text>
      </svg>
    </div>
  );
}

// ============================================================
// Bin packing demo
// ============================================================
const BP_ALGOS = [
  { key: "ffd", label: "First-Fit Decreasing", kind: "constructive" },
  { key: "bfd", label: "Best-Fit Decreasing", kind: "constructive" },
  { key: "sa", label: "Simulated Annealing", kind: "meta" },
];

function BinPackingDemo() {
  const [algo, setAlgo] = useState("ffd");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(180);
  const [step, setStep] = useState(null);
  const [history, setHistory] = useState([]);
  const genRef = useRef(null);

  function reset() { setRunning(false); setStep(null); setHistory([]); genRef.current = null; }
  function start() {
    if (algo === "ffd") genRef.current = firstFitDecreasing();
    else if (algo === "bfd") genRef.current = bestFitDecreasing();
    else if (algo === "sa") genRef.current = simulatedAnnealingBP();
    setHistory([]); setStep(null); setRunning(true);
  }
  useEffect(() => {
    if (!running || !genRef.current) return;
    const id = setInterval(() => {
      const next = genRef.current.next();
      if (next.done) { setRunning(false); return; }
      setStep(next.value);
      setHistory((h) => [...h, next.value]);
      if (next.value.done) setRunning(false);
    }, speed);
    return () => clearInterval(id);
  }, [running, speed]);

  return (
    <>
      <AlgoSelector algos={BP_ALGOS} algo={algo} setAlgo={(k) => { setAlgo(k); reset(); }} />
      <div style={{ marginBottom: 10, fontSize: 13, padding: "10px 14px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 6 }}>
        Pack {N_BP_ITEMS} items (sizes 1–8) into bins of capacity {BP_CAPACITY}.
        Lower bound: <b>⌈Σsize / capacity⌉ = {BP_LB} bins</b>. Total size = {BP_ITEMS.reduce((s, it) => s + it.size, 0)}.
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "minmax(540px, 1.4fr) minmax(380px, 1fr)", gap: 22 }}>
        <BinView step={step} algo={algo} />
        <div>
          <Controls running={running} onStart={start} onStop={() => setRunning(false)} onReset={reset} speed={speed} setSpeed={setSpeed} />
          <BPState step={step} algo={algo} />
          <BPProgress history={history} algo={algo} />
          {algo === "sa" && <SAExtras history={history} />}
        </div>
      </div>
      <CodePanelBP algo={algo} />
    </>
  );
}

function BinView({ step, algo }) {
  const bins = step
    ? algo === "sa" ? step.bestBins : step.bins
    : [];
  const remaining = step?.remaining || [];
  const cellSize = 30;
  const W = 720, H = 360;

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        Bins
      </div>
      <svg width={W} height={H}>
        {bins.map((b, bi) => {
          const x = 20 + (bi % 6) * (cellSize * 2 + 16);
          const y = 16 + Math.floor(bi / 6) * (cellSize * BP_CAPACITY * 0.5 + 30);
          const binH = cellSize * BP_CAPACITY * 0.5;
          // Stack items
          let curY = y + binH;
          return (
            <g key={bi}>
              <rect x={x} y={y} width={cellSize * 2} height={binH} fill="#fafafa" stroke="#1f4e3d" strokeWidth={1.5} />
              {b.items.map((idx, ii) => {
                const sz = BP_ITEMS[idx].size;
                const h = sz * (binH / BP_CAPACITY);
                curY -= h;
                const colors = ["#c8311c", "#0b3da0", "#7a3da0", "#1f4e3d", "#d4a017", "#3d8a8a"];
                return (
                  <g key={ii}>
                    <rect x={x} y={curY} width={cellSize * 2} height={h - 1} fill={colors[idx % colors.length]} opacity={0.85} stroke="#fff" strokeWidth={1} />
                    <text x={x + cellSize} y={curY + h / 2 + 4} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#fff" fontWeight={700}>
                      {idx} ({sz})
                    </text>
                  </g>
                );
              })}
              <text x={x + cellSize} y={y + binH + 14} textAnchor="middle" fontSize={11} fontFamily="monospace" fill="#444">
                bin {bi + 1} ({b.used}/{BP_CAPACITY})
              </text>
            </g>
          );
        })}
      </svg>
      {remaining.length > 0 && algo !== "sa" && (
        <div style={{ marginTop: 8, fontSize: 12, fontFamily: "monospace", color: "#666" }}>
          remaining: {remaining.map((i) => `${i}(${BP_ITEMS[i].size})`).join(", ")}
        </div>
      )}
    </div>
  );
}

function BPState({ step, algo }) {
  if (!step) return <div style={{ ...panel, marginBottom: 12 }}><KV k="status" v="press Start to begin" /></div>;
  return (
    <div style={{ ...panel, marginBottom: 12 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 6 }}>
        State
      </div>
      <KV k="lower bound" v={`${BP_LB} bins`} />
      {algo !== "sa" && step.bins && <KV k="bins used so far" v={step.bins.length} highlight />}
      {algo === "sa" && (
        <>
          <KV k="current bins" v={step.curCount} />
          <KV k="best bins" v={step.bestCount} highlight />
          <KV k="temperature T" v={step.T?.toFixed(3)} />
        </>
      )}
      <KV k="last action" v={step.info} />
    </div>
  );
}

function BPProgress({ history, algo }) {
  if (history.length === 0) return null;
  const W = 460, H = 160, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const bests = history.map((h) =>
    algo === "sa" ? h.bestCount : (h.bins?.length ?? 0)
  );
  const yMin = BP_LB - 0.5;
  const yMax = Math.max(...bests, BP_LB + 1) + 0.5;
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / (yMax - yMin)) * ch;
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        best bin count over iterations
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={ys(BP_LB)} x2={padL + cw} y2={ys(BP_LB)} stroke="#0b3da0" strokeDasharray="3,3" strokeWidth={1.5} />
        <text x={padL + cw - 4} y={ys(BP_LB) - 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#0b3da0">
          LB = {BP_LB}
        </text>
        <polyline points={bests.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#c8311c" strokeWidth={2} />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{Math.round(yMax)}</text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{Math.round(yMin + 0.5)}</text>
      </svg>
    </div>
  );
}

// ============================================================
// Reusable view components
// ============================================================
function ProgressChart({ history, algo, kind }) {
  if (history.length === 0) return null;
  const W = 460, H = 160, padL = 50, padT = 12, padB = 26, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const lens = history.map((h) => h.len ?? 0);
  const bests = history.map((h) => h.bestLen ?? h.len ?? 0);
  const yMin = Math.min(...lens, ...bests);
  const yMax = Math.max(...lens, ...bests);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  const showCur = algo === "sa" || algo === "tabu";
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        tour length over iterations
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        {showCur && (
          <polyline points={lens.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#888" strokeWidth={1} />
        )}
        <polyline points={bests.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#c8311c" strokeWidth={2} />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMax.toFixed(2)}</text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMin.toFixed(2)}</text>
        {showCur && (
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

function SAExtras({ history }) {
  if (!history || history.length === 0) return null;
  const last = history[history.length - 1];
  const W = 460, H = 100, padL = 50, padT = 8, padB = 22, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const Ts = history.map((h) => h.T || 0);
  const accs = history.map((h) => h.accRate || 0);
  if (Ts.length === 0) return null;
  const tmax = Math.max(...Ts, 1e-9);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const yT = (v) => padT + (1 - v / tmax) * ch;
  const yA = (v) => padT + (1 - v) * ch;

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        SA: temperature schedule + acceptance rate
      </div>
      <svg width={W} height={H}>
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <polyline points={Ts.map((v, i) => `${xs(i)},${yT(v)}`).join(" ")} fill="none" stroke="#7a3da0" strokeWidth={1.5} />
        <polyline points={accs.map((v, i) => `${xs(i)},${yA(v)}`).join(" ")} fill="none" stroke="#0b3da0" strokeWidth={1.5} />
        <g transform={`translate(${padL + 10}, ${padT})`}>
          <line x1={0} y1={6} x2={14} y2={6} stroke="#7a3da0" strokeWidth={1.5} />
          <text x={18} y={10} fontSize={10}>T (cooling)</text>
          <line x1={90} y1={6} x2={104} y2={6} stroke="#0b3da0" strokeWidth={1.5} />
          <text x={108} y={10} fontSize={10}>accept rate</text>
        </g>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">0</text>
        <text x={padL - 4} y={padT + 8} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">1</text>
      </svg>
      <div style={{ fontSize: 11.5, color: "#555", marginTop: 4, fontFamily: "monospace" }}>
        last: T = {last.T?.toFixed(4)}, accept rate = {((last.accRate || 0) * 100).toFixed(1)}%
        {typeof last.pAccept === "number" ? `, last P(accept) = ${last.pAccept.toFixed(3)}` : ""}
      </div>
    </div>
  );
}

function TabuListView({ tabu, kind }) {
  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        Tabu list ({tabu.length} entries)
      </div>
      {tabu.length === 0 ? (
        <div style={{ fontSize: 12, color: "#888", fontFamily: "monospace" }}>(empty — no recent moves)</div>
      ) : (
        <div style={{ fontFamily: "monospace", fontSize: 12, color: "#444", lineHeight: 1.5 }}>
          {tabu.map((t, i) => (
            <span key={i} style={{
              display: "inline-block",
              padding: "2px 8px",
              margin: "2px 4px 2px 0",
              borderRadius: 4,
              background: "#fde8a0",
              color: "#7a5a00",
              border: "1px solid #f5d68d",
            }}>
              {kind === "tsp" ? `(${t.i},${t.j})` : `flip ${t.bit}`}
              <span style={{ color: "#a07700", marginLeft: 4, fontSize: 10 }}>(exp@{t.exp})</span>
            </span>
          ))}
        </div>
      )}
      <div style={{ fontSize: 11, color: "#777", marginTop: 6, lineHeight: 1.5 }}>
        Reversing one of these moves is currently TABU — except by aspiration (if it would beat the global best).
      </div>
    </div>
  );
}

function GADiversityView({ history, kind }) {
  if (history.length === 0) return null;
  const last = history[history.length - 1];
  const stats = last.lenStats || last.diversity;
  if (!stats) return null;
  const W = 460, H = 100, padL = 50, padT = 8, padB = 22, padR = 8;
  const cw = W - padL - padR, ch = H - padT - padB;
  const mins = history.map((h) => (h.lenStats?.min ?? h.diversity?.min) || 0);
  const means = history.map((h) => (h.lenStats?.mean ?? h.diversity?.mean) || 0);
  const maxs = history.map((h) => (h.lenStats?.max ?? h.diversity?.max) || 0);
  const yMin = Math.min(...mins, ...means, ...maxs);
  const yMax = Math.max(...mins, ...means, ...maxs);
  const xs = (i) => padL + (i / Math.max(1, history.length - 1)) * cw;
  const ys = (v) => padT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ch;
  const isMax = kind === "ks"; // higher = better for knapsack

  return (
    <div style={panel}>
      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#888", marginBottom: 4 }}>
        GA: population fitness (best / mean / worst)
      </div>
      <svg width={W} height={H}>
        <polygon
          points={[
            ...mins.map((_, i) => `${xs(i)},${ys(mins[i])}`),
            ...maxs.map((_, i) => `${xs(history.length - 1 - i)},${ys(maxs[history.length - 1 - i])}`),
          ].join(" ")}
          fill="rgba(122, 61, 160, 0.10)"
          stroke="none"
        />
        <polyline points={maxs.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke={isMax ? "#888" : "#c8311c"} strokeWidth={1.5} />
        <polyline points={means.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke="#7a3da0" strokeWidth={1.5} />
        <polyline points={mins.map((v, i) => `${xs(i)},${ys(v)}`).join(" ")} fill="none" stroke={isMax ? "#c8311c" : "#888"} strokeWidth={1.5} />
        <line x1={padL} y1={padT + ch} x2={padL + cw} y2={padT + ch} stroke="#bbb" />
        <line x1={padL} y1={padT} x2={padL} y2={padT + ch} stroke="#bbb" />
        <text x={padL - 4} y={padT + 4} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMax.toFixed(1)}</text>
        <text x={padL - 4} y={padT + ch} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">{yMin.toFixed(1)}</text>
      </svg>
      <div style={{ fontSize: 11, color: "#777", marginTop: 4, lineHeight: 1.5 }}>
        Diversity collapses as generations advance — the population converges. If it collapses too fast, increase mutation rate.
      </div>
    </div>
  );
}

// ============================================================
// Algo selector (radio strip)
// ============================================================
function AlgoSelector({ algos, algo, setAlgo }) {
  return (
    <div style={{ marginBottom: 12, padding: "10px 12px", background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8, display: "flex", flexWrap: "wrap", gap: 14, alignItems: "center" }}>
      <span style={{ fontSize: 11, fontFamily: "monospace", letterSpacing: "0.12em", color: "#666", textTransform: "uppercase" }}>algorithm:</span>
      {algos.map((a) => (
        <label key={a.key} style={{ fontSize: 13, fontWeight: algo === a.key ? 700 : 400 }}>
          <input type="radio" checked={algo === a.key} onChange={() => setAlgo(a.key)} />
          &nbsp;{a.label}
          <span style={{ fontSize: 10.5, color: "#888", marginLeft: 4, fontFamily: "monospace" }}>({a.kind})</span>
        </label>
      ))}
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
          <button onClick={onStart} style={btnPrimary}><Play size={14} /> Start</button>
        ) : (
          <button onClick={onStop} style={btn}><Pause size={14} /> Pause</button>
        )}
        <button onClick={onReset} style={btn}><RotateCcw size={14} /> Reset</button>
      </div>
      <label style={{ fontSize: 12, fontFamily: "monospace", color: "#444" }}>
        speed: <b>{speed} ms/step</b>
      </label>
      <input type="range" min={20} max={500} step={10} value={speed} onChange={(e) => setSpeed(+e.target.value)} style={{ width: "100%" }} />
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
// Code panels — algorithm-specific source
// ============================================================
function CodePanelTSP({ algo }) {
  const codes = {
    nn: `# Nearest neighbor heuristic for TSP
def nearest_neighbor(cities, start=0):
    n = len(cities)
    visited = {start}
    tour = [start]
    while len(tour) < n:
        last = tour[-1]
        nxt = min(
            (i for i in range(n) if i not in visited),
            key=lambda i: dist(cities[last], cities[i]),
        )
        tour.append(nxt); visited.add(nxt)
    return tour
`,
    hill: `# 2-opt local search
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
`,
    sa: `# Simulated annealing
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
`,
    tabu: `# Tabu search with 2-opt neighborhood
def tabu(tour, tenure=12, n_iter=400):
    cur, cur_len = tour, length(tour)
    best, best_len = cur, cur_len
    tabu = {}                             # (i,j) -> expiry iter
    for it in range(n_iter):
        best_move = None
        for i in range(1, len(cur)-1):
            for j in range(i+1, len(cur)):
                cand = cur[:i] + cur[i:j+1][::-1] + cur[j+1:]
                cl = length(cand)
                aspires = cl < best_len   # beats global best
                if (i, j) in tabu and not aspires:
                    continue
                if best_move is None or cl < best_move[1]:
                    best_move = ((i, j), cl, cand)
        if best_move is None: break
        (i, j), cur_len, cur = best_move
        tabu[(i, j)] = it + tenure
        tabu = {k: v for k, v in tabu.items() if v > it}
        if cur_len < best_len: best, best_len = cur, cur_len
    return best
`,
    ga: `# Genetic algorithm with order crossover (OX)
import random
def ga(n_cities, pop_size=30, generations=60):
    def random_perm():
        p = list(range(1, n_cities)); random.shuffle(p); return [0] + p
    def ox(a, b):
        i, j = sorted(random.sample(range(1, n_cities), 2))
        child = [-1] * n_cities
        child[i:j+1] = a[i:j+1]
        p = 0
        for k in [(j+1+m) % n_cities for m in range(n_cities)]:
            while p < n_cities and b[p] in child: p += 1
            if child[k] == -1: child[k] = b[p]; p += 1
        z = child.index(0); return child[z:] + child[:z]
    pop = [random_perm() for _ in range(pop_size)]
    for g in range(generations):
        pop.sort(key=length)
        elite = pop[:max(2, pop_size // 5)]
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            a = random.choice(pop[:pop_size//2])
            b = random.choice(pop[:pop_size//2])
            child = ox(a, b)
            if random.random() < 0.3:                  # swap mutation
                i, j = random.sample(range(1, n_cities), 2)
                child[i], child[j] = child[j], child[i]
            new_pop.append(child)
        pop = new_pop
    return min(pop, key=length)
`,
  };
  return <CodeBlock code={codes[algo]} />;
}
function CodePanelKS({ algo }) {
  const codes = {
    greedy: `# Greedy by value/weight ratio (NOT always optimal)
def greedy_ratio(items, capacity):
    sel = [False] * len(items)
    w = 0
    for it in sorted(items, key=lambda it: -it.value/it.weight):
        if w + it.weight <= capacity:
            sel[it.id] = True; w += it.weight
    return sel
`,
    sa: `# Simulated annealing with bit-flip neighborhood
import math, random
def sa_ks(items, capacity, T0=30, alpha=0.97, n_iter=200):
    n = len(items)
    cur = [False] * n
    def obj(s):
        v = sum(items[i].value for i in range(n) if s[i])
        w = sum(items[i].weight for i in range(n) if s[i])
        return v if w <= capacity else v - 5*(w - capacity)
    best, best_v = cur, obj(cur); T = T0
    for k in range(n_iter):
        i = random.randrange(n)
        cand = cur.copy(); cand[i] = not cand[i]
        delta = obj(cur) - obj(cand)            # max problem
        if delta <= 0 or random.random() < math.exp(-delta / T):
            cur = cand
            if obj(cur) > best_v:
                best, best_v = cur, obj(cur)
        T *= alpha
    return best
`,
    tabu: `# Tabu search on knapsack — neighborhood = single bit-flips
def tabu_ks(items, capacity, tenure=7, n_iter=80):
    n = len(items); cur = [False] * n
    def feas(s): return sum(items[i].weight for i in range(n) if s[i]) <= capacity
    def val(s): return sum(items[i].value for i in range(n) if s[i])
    best, best_v = cur, val(cur); tabu = {}
    for it in range(n_iter):
        best_move = None
        for i in range(n):
            cand = cur.copy(); cand[i] = not cand[i]
            score = val(cand) if feas(cand) else val(cand) - 1000
            aspires = feas(cand) and val(cand) > best_v
            if i in tabu and not aspires: continue
            if best_move is None or score > best_move[1]:
                best_move = (i, score, cand)
        if best_move is None: break
        i, _, cur = best_move
        tabu[i] = it + tenure
        tabu = {k: v for k, v in tabu.items() if v > it}
        if feas(cur) and val(cur) > best_v:
            best, best_v = cur, val(cur)
    return best
`,
    ga: `# Genetic algorithm for 0-1 knapsack
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
                if random.random() < 0.05: child[i] = not child[i]
            new_pop.append(child)
        pop = new_pop
    return max(pop, key=fitness)
`,
  };
  return <CodeBlock code={codes[algo]} />;
}
function CodePanelBP({ algo }) {
  const codes = {
    ffd: `# First-Fit Decreasing
def ffd(items, capacity):
    bins = []
    for it in sorted(items, key=lambda x: -x.size):
        for b in bins:
            if b.used + it.size <= capacity:
                b.add(it); break
        else:
            bins.append(Bin([it]))         # NEW bin
    return bins
`,
    bfd: `# Best-Fit Decreasing — pick the TIGHTEST bin
def bfd(items, capacity):
    bins = []
    for it in sorted(items, key=lambda x: -x.size):
        best = None
        for b in bins:
            slack = capacity - b.used - it.size
            if slack >= 0 and (best is None or slack < best[1]):
                best = (b, slack)
        if best is None:
            bins.append(Bin([it]))
        else:
            best[0].add(it)
    return bins
`,
    sa: `# Simulated annealing on a permutation encoding
import math, random
def sa_bp(items, capacity, T0=4, alpha=0.97, n_iter=200):
    perm = sorted(range(len(items)), key=lambda i: -items[i].size)
    def pack(p):
        bins = []
        for idx in p:
            for b in bins:
                if b.used + items[idx].size <= capacity:
                    b.add(items[idx]); break
            else:
                bins.append(Bin([items[idx]]))
        return bins
    cur = pack(perm); best = cur; T = T0
    for k in range(n_iter):
        i, j = random.sample(range(len(perm)), 2)
        cand = perm.copy(); cand[i], cand[j] = cand[j], cand[i]
        cand_b = pack(cand)
        delta = len(cand_b) - len(cur)
        if delta <= 0 or random.random() < math.exp(-delta / T):
            perm, cur = cand, cand_b
            if len(cur) < len(best): best = cur
        T *= alpha
    return best
`,
  };
  return <CodeBlock code={codes[algo]} />;
}

function CodeBlock({ code }) {
  return (
    <pre style={{
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
    }}>
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
          <b>Constructive vs improvement vs metaheuristic.</b> A constructive
          heuristic builds a solution in one pass (nearest neighbor, greedy,
          first-fit). An improvement heuristic starts with any solution and
          locally refines it (2-opt). A metaheuristic adds randomness or memory
          to escape local optima (SA, tabu, GA, ant colony).
        </li>
        <li>
          <b>Simulated annealing.</b> Accepts worse moves with probability{" "}
          <Tex>{String.raw`P = \exp(-\Delta / T)`}</Tex>. As <Tex>{`T`}</Tex>{" "}
          cools, the algorithm becomes greedier. Provably converges to the
          global optimum if <Tex>{`T`}</Tex> cools logarithmically — but that's
          exponentially slow. Geometric cooling{" "}
          <Tex>{`T_{k+1} = \\alpha T_k`}</Tex> with{" "}
          <Tex>{String.raw`\alpha \in [0.95, 0.999]`}</Tex> works in practice.
        </li>
        <li>
          <b>Tabu search.</b> Forbids reversing recent moves for a fixed{" "}
          <i>tenure</i> — escaping local optima by FORCING uphill moves once
          the neighborhood is exhausted. Aspiration overrides the tabu list
          when a tabu move would beat the global best. Strongly outperforms
          plain hill-climbing on TSP-like problems.
        </li>
        <li>
          <b>Genetic algorithms.</b> Population-based: keep N candidate
          solutions, evaluate fitness, breed via selection + crossover +
          mutation. Selection pressure controls exploration vs exploitation.
          Elitism (keeping the best) prevents regression. For TSP, plain
          uniform crossover doesn't preserve permutations — use <i>order
          crossover</i> (OX) or <i>partially-mapped crossover</i> (PMX) instead.
        </li>
        <li>
          <b>Greedy by ratio for knapsack.</b> Optimal for the FRACTIONAL
          knapsack (LP relaxation), but not for the 0/1 problem — see
          textbook counter-examples like <Tex>{`(w, v) = (10, 60), (20, 100), (30, 120)`}</Tex>{" "}
          with capacity 50. The dashed blue line in the progress chart shows
          the LP-relaxation upper bound.
        </li>
        <li>
          <b>FFD &amp; BFD for bin packing.</b> First-fit-decreasing has a
          worst-case ratio of 11/9 (Johnson, 1973) — never uses more than
          11/9 + 6/9 of OPT bins. BFD has the same worst-case bound but is
          often tighter on random instances. Both run in <Tex>{`O(n \\log n)`}</Tex>.
        </li>
        <li>
          <b>When to use what.</b> Small problems (n ≤ 50 cities, n ≤ 30
          knapsack items) → exact MIP via Gurobi/CPLEX. Medium → LKH (TSP),
          DP / branch-and-bound (knapsack), MIP relaxations. Large
          (n &gt; 1000) → metaheuristics dominate. ML-augmented variants
          (graph neural networks + RL, learned heuristics) are state of the
          art on combinatorial benchmarks since ~2020.
        </li>
        <li>
          <b>How to compare two heuristics fairly.</b> Run both with the same
          time budget (not the same iteration count — different algorithms
          have different per-iteration cost). Average over 30+ random seeds.
          Report the gap to the best-known or LP-relaxation upper bound, not
          the raw objective.
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
