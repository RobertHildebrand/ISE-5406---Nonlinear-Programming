import React, { useState, useMemo } from "react";
import { StepForward, RotateCcw, Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   BRANCH-AND-BOUND TREE EXPLORER
   ISE 5406

   A 2-variable mixed-integer linear program small enough to
   visualize. We pre-compute the entire B&B tree (best-first,
   cheapest-bound branching) and let the user step through it
   one node at a time. The plot shows the LP relaxation
   feasible region for each node and the LP optimum; the tree
   panel shows the search structure with primal/dual bounds
   and pruning status.
   ============================================================ */

// ============================================================
// Problem
//   max  3 x + 4 y
//   s.t. 2 x + y <=  6
//        x + 2 y <=  6
//        x, y >= 0,  x, y integer
//
// LP optimum: (2, 2), obj = 14
// IP optimum: (2, 2),  obj = 14   (this LP is integral — boring)
//
// Try a less-integral example:
//   max  5.5 x + 4 y
//   s.t. 2 x + y <= 7
//        x + 2 y <= 6
//        x, y >= 0, integer
//
// LP: (8/3, 5/3) ≈ (2.667, 1.667), obj ≈ 21.333
// IP: (3, 1), obj = 19.5  (or (1, 2.5) → 19.5 if y were continuous)
//
// Computed tree below.
// ============================================================
const C = [5.5, 4]; // objective coefficients (maximize)
const A = [
  [2, 1],
  [1, 2],
];
const B = [7, 6];

function solveLP(extraBounds) {
  // Solve the LP relaxation:
  //   max c·x s.t. A x <= b, ext_lo <= x <= ext_hi
  // 2-D LP — enumerate the candidate vertices (intersections of constraint
  // pairs) and pick the maximum feasible one. Constraints:
  //   2x+y <= 7, x+2y <= 6, x >= lo[0], x <= hi[0], y >= lo[1], y <= hi[1]
  const { lo, hi } = extraBounds;
  const lines = [
    [2, 1, 7],   // 2x+y = 7
    [1, 2, 6],   // x+2y = 6
    [1, 0, lo[0]], // x = lo[0]
    [1, 0, hi[0]], // x = hi[0]
    [0, 1, lo[1]], // y = lo[1]
    [0, 1, hi[1]], // y = hi[1]
  ];
  const vertices = [];
  for (let i = 0; i < lines.length; i++) {
    for (let j = i + 1; j < lines.length; j++) {
      const [a1, b1, c1] = lines[i];
      const [a2, b2, c2] = lines[j];
      const det = a1 * b2 - a2 * b1;
      if (Math.abs(det) < 1e-9) continue;
      const x = (c1 * b2 - c2 * b1) / det;
      const y = (a1 * c2 - a2 * c1) / det;
      if (
        2 * x + y <= 7 + 1e-9 &&
        x + 2 * y <= 6 + 1e-9 &&
        x >= lo[0] - 1e-9 &&
        x <= hi[0] + 1e-9 &&
        y >= lo[1] - 1e-9 &&
        y <= hi[1] + 1e-9
      ) {
        vertices.push({ x, y });
      }
    }
  }
  if (vertices.length === 0) return { feasible: false };
  let best = vertices[0];
  let bestVal = C[0] * best.x + C[1] * best.y;
  for (const v of vertices) {
    const val = C[0] * v.x + C[1] * v.y;
    if (val > bestVal + 1e-9) {
      bestVal = val;
      best = v;
    }
  }
  return { feasible: true, x: best.x, y: best.y, obj: bestVal, vertices };
}

function isInteger(v) {
  return Math.abs(v - Math.round(v)) < 1e-6;
}

// Build the B&B tree (best-first by parent's LP value, branch on most-fractional).
function buildTree() {
  let nextId = 0;
  const ROOT = {
    id: nextId++,
    parent: null,
    branchInfo: null,
    lo: [0, 0],
    hi: [Infinity, Infinity],
    depth: 0,
  };
  const nodes = [ROOT];
  const queue = [0];
  let incumbent = { obj: -Infinity, x: null };
  const events = []; // step-by-step: [{action, nodeId, ...}]

  events.push({ action: "init", nodeId: 0 });

  while (queue.length > 0) {
    // Best-first: pick node with highest parent LP value (or just FIFO for first)
    const idx = 0;
    const id = queue.splice(idx, 1)[0];
    const node = nodes[id];
    const lp = solveLP({ lo: node.lo, hi: node.hi });
    node.lp = lp;
    if (!lp.feasible) {
      node.status = "infeasible";
      events.push({ action: "infeasible", nodeId: id });
      continue;
    }
    node.lpObj = lp.obj;
    node.lpX = { x: lp.x, y: lp.y };
    if (lp.obj <= incumbent.obj + 1e-9) {
      node.status = "fathomed_bound";
      events.push({ action: "fathom_bound", nodeId: id });
      continue;
    }
    if (isInteger(lp.x) && isInteger(lp.y)) {
      node.status = "integer";
      events.push({ action: "integer_found", nodeId: id, obj: lp.obj });
      if (lp.obj > incumbent.obj) {
        incumbent = { obj: lp.obj, x: { x: Math.round(lp.x), y: Math.round(lp.y) } };
        node.becameIncumbent = true;
      }
      continue;
    }
    // Branch on the most-fractional variable
    const fracX = Math.abs(lp.x - Math.round(lp.x));
    const fracY = Math.abs(lp.y - Math.round(lp.y));
    const branchVar = fracX > fracY ? 0 : 1;
    const branchVal = branchVar === 0 ? lp.x : lp.y;
    const lo = Math.floor(branchVal);
    const hi = Math.ceil(branchVal);
    node.branchVar = branchVar;
    node.branchVal = branchVal;
    node.status = "branched";

    const left = {
      id: nextId++,
      parent: id,
      branchInfo: { var: branchVar, op: "<=", val: lo },
      lo: [...node.lo],
      hi: [...node.hi],
      depth: node.depth + 1,
    };
    left.hi[branchVar] = lo;
    const right = {
      id: nextId++,
      parent: id,
      branchInfo: { var: branchVar, op: ">=", val: hi },
      lo: [...node.lo],
      hi: [...node.hi],
      depth: node.depth + 1,
    };
    right.lo[branchVar] = hi;
    nodes.push(left, right);
    node.children = [left.id, right.id];
    events.push({ action: "branched", nodeId: id, leftId: left.id, rightId: right.id });
    queue.push(left.id, right.id);
  }
  return { nodes, events, incumbent };
}

const TREE = buildTree();

// ============================================================
// Main component
// ============================================================
export default function BranchBoundDemo() {
  const [step, setStep] = useState(0);
  const [selectedId, setSelectedId] = useState(0);

  const visibleNodes = useMemo(() => {
    const ids = new Set();
    for (let i = 0; i <= step && i < TREE.events.length; i++) {
      const ev = TREE.events[i];
      ids.add(ev.nodeId);
      if (ev.leftId != null) ids.add(ev.leftId);
      if (ev.rightId != null) ids.add(ev.rightId);
    }
    return ids;
  }, [step]);

  const incumbent = useMemo(() => {
    let best = { obj: -Infinity, x: null, fromNode: null };
    for (let i = 0; i <= step && i < TREE.events.length; i++) {
      const ev = TREE.events[i];
      if (ev.action === "integer_found" && ev.obj > best.obj) {
        const node = TREE.nodes[ev.nodeId];
        best = {
          obj: ev.obj,
          x: { x: Math.round(node.lpX.x), y: Math.round(node.lpX.y) },
          fromNode: ev.nodeId,
        };
      }
    }
    return best;
  }, [step]);

  const dualBound = useMemo(() => {
    let best = -Infinity;
    for (let i = 0; i <= step && i < TREE.events.length; i++) {
      const ev = TREE.events[i];
      const node = TREE.nodes[ev.nodeId];
      if (node.lpObj != null && node.status !== "fathomed_bound" && node.status !== "infeasible" && node.status !== "integer") {
        // Open node — its LP value contributes to the dual bound
        if (i === step || (ev.action === "branched" && step > i)) {
          // Skip — already split
        }
      }
    }
    // Simpler computation: max over all CURRENTLY OPEN nodes of their LP value
    // For the visualization, just use the max LP obj among visible UNFATHOMED leaf nodes
    const visibleIds = [];
    for (let i = 0; i <= step && i < TREE.events.length; i++) {
      const ev = TREE.events[i];
      if (ev.action === "init" || ev.action === "branched") {
        // leaves are children of branched nodes
      }
    }
    // Just return the root LP for simplicity, or the max LP of any leaf still undecided
    let maxOpen = -Infinity;
    TREE.nodes.forEach((n, idx) => {
      if (!visibleNodes.has(idx)) return;
      // Is it a leaf in the visible tree?
      const hasVisibleChild =
        n.children && n.children.some((cid) => visibleNodes.has(cid));
      if (hasVisibleChild) return;
      // Its status determines whether it contributes
      if (n.lpObj != null && n.status !== "fathomed_bound" && n.status !== "infeasible") {
        if (n.lpObj > maxOpen) maxOpen = n.lpObj;
      }
    });
    return maxOpen;
  }, [step, visibleNodes]);

  const totalSteps = TREE.events.length - 1;
  const stepEvent = TREE.events[Math.min(step, totalSteps)];
  const selectedNode = TREE.nodes[selectedId];

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Branch-and-Bound Tree Explorer
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        Watch a small MILP get solved by branch-and-bound, one node at a
        time. The LP relaxation at each node is shown on the plot; the tree
        panel shows search structure, branching decisions, and pruning. Click
        any node in the tree to inspect its LP relaxation.
      </p>

      <div style={problemBox}>
        <div style={{ marginBottom: 6, fontSize: 13, fontWeight: 600 }}>
          Problem
        </div>
        <Tex block>
          {String.raw`\max\;\; 5.5\,x + 4\,y \quad \text{s.t.} \quad 2x + y \le 7,\;\; x + 2y \le 6,\;\; x, y \ge 0,\;\; x, y \in \mathbb{Z}`}
        </Tex>
        <div style={{ fontSize: 13, color: "#555", marginTop: 4 }}>
          Continuous LP optimum:{" "}
          <Tex>{String.raw`(8/3,\; 5/3) \approx (2.667,\; 1.667),\;\; \text{obj} \approx 21.333`}</Tex>
          . Integer optimum: <Tex>{String.raw`(3, 1),\;\; \text{obj} = 19.5`}</Tex>.
        </div>
      </div>

      <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
        <button
          onClick={() => setStep((s) => Math.min(totalSteps, s + 1))}
          disabled={step >= totalSteps}
          style={btnPrimary}
        >
          <StepForward size={16} /> Next event
        </button>
        <button onClick={() => setStep(0)} style={btn}>
          <RotateCcw size={16} /> Reset
        </button>
        <span style={{ alignSelf: "center", fontSize: 12, fontFamily: "monospace", color: "#666" }}>
          event {step} / {totalSteps}
        </span>
      </div>

      <div style={eventBox}>
        <EventDescription ev={stepEvent} />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(440px, 1fr) minmax(420px, 1fr)",
          gap: 22,
          alignItems: "flex-start",
        }}
      >
        <FeasibleRegionPlot
          node={selectedNode}
          incumbent={incumbent}
        />
        <TreeView
          visibleNodes={visibleNodes}
          selectedId={selectedId}
          setSelectedId={setSelectedId}
          incumbent={incumbent}
          dualBound={dualBound}
        />
      </div>

      <NodeDetail node={selectedNode} incumbent={incumbent} />
      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Event description
// ============================================================
function EventDescription({ ev }) {
  if (!ev) return null;
  if (ev.action === "init") {
    return <span>Initialize the search at the <b>root node (id 0)</b> with the original LP relaxation. No bounds on integers yet.</span>;
  }
  if (ev.action === "branched") {
    const node = TREE.nodes[ev.nodeId];
    const v = node.branchVar === 0 ? "x" : "y";
    const val = node.branchVal.toFixed(3);
    return (
      <span>
        Node <b>{ev.nodeId}</b>: LP optimum has{" "}
        <Tex>{`${v} = ${val}`}</Tex> (fractional). Branch on{" "}
        <Tex>{`${v}`}</Tex> → create children <Tex>{`${v} \\le ${Math.floor(node.branchVal)}`}</Tex> (left) and <Tex>{`${v} \\ge ${Math.ceil(node.branchVal)}`}</Tex> (right).
      </span>
    );
  }
  if (ev.action === "integer_found") {
    return (
      <span>
        Node <b>{ev.nodeId}</b>: LP relaxation is integer-feasible.{" "}
        <Tex>{`\\text{obj} = ${ev.obj.toFixed(2)}`}</Tex>. Update incumbent if it improved.
      </span>
    );
  }
  if (ev.action === "infeasible") {
    return <span>Node <b>{ev.nodeId}</b>: LP is infeasible. Fathom this branch.</span>;
  }
  if (ev.action === "fathom_bound") {
    return (
      <span>
        Node <b>{ev.nodeId}</b>: LP value <Tex>{`\\le \\text{incumbent}`}</Tex> — fathom by bound.
      </span>
    );
  }
  return null;
}

// ============================================================
// Feasible region plot
// ============================================================
function FeasibleRegionPlot({ node, incumbent }) {
  const W = 480, H = 480;
  const padL = 50, padR = 16, padT = 18, padB = 30;
  const chartW = W - padL - padR;
  const chartH = H - padT - padB;
  const xmin = -0.3, xmax = 4.0;
  const ymin = -0.3, ymax = 4.0;
  const xs = (x) => padL + ((x - xmin) / (xmax - xmin)) * chartW;
  const ys = (y) => padT + (1 - (y - ymin) / (ymax - ymin)) * chartH;

  // Compute polygon for this node's LP
  const lp = node.lp || solveLP({ lo: node.lo, hi: node.hi });
  let polyPts = [];
  if (lp.feasible) {
    // Order vertices CCW around centroid
    const cx = lp.vertices.reduce((s, v) => s + v.x, 0) / lp.vertices.length;
    const cy = lp.vertices.reduce((s, v) => s + v.y, 0) / lp.vertices.length;
    polyPts = [...lp.vertices].sort(
      (a, b) => Math.atan2(a.y - cy, a.x - cx) - Math.atan2(b.y - cy, b.x - cx)
    );
  }

  // Integer lattice points in the visible range
  const lattice = [];
  for (let i = 0; i <= 4; i++) {
    for (let j = 0; j <= 4; j++) {
      lattice.push({ x: i, y: j });
    }
  }

  return (
    <div style={panel}>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>
        Node {node.id}: LP relaxation
      </div>
      <svg width={W} height={H}>
        {/* axes */}
        <line x1={padL} y1={ys(0)} x2={padL + chartW} y2={ys(0)} stroke="#bbb" />
        <line x1={xs(0)} y1={padT} x2={xs(0)} y2={padT + chartH} stroke="#bbb" />
        {[1, 2, 3].map((v) => (
          <React.Fragment key={v}>
            <line x1={xs(v)} y1={padT} x2={xs(v)} y2={padT + chartH} stroke="#eee" strokeDasharray="2,3" />
            <line x1={padL} y1={ys(v)} x2={padL + chartW} y2={ys(v)} stroke="#eee" strokeDasharray="2,3" />
          </React.Fragment>
        ))}

        {/* feasible polygon */}
        {polyPts.length >= 3 && (
          <polygon
            points={polyPts.map((p) => `${xs(p.x)},${ys(p.y)}`).join(" ")}
            fill="rgba(31, 78, 61, 0.12)"
            stroke="#1f4e3d"
            strokeWidth={2}
          />
        )}

        {/* node bound lines (from branching) */}
        {Number.isFinite(node.lo[0]) && node.lo[0] > 0 && (
          <line x1={xs(node.lo[0])} y1={padT} x2={xs(node.lo[0])} y2={padT + chartH} stroke="#c8311c" strokeWidth={2} strokeDasharray="6,3" />
        )}
        {Number.isFinite(node.hi[0]) && node.hi[0] < 4 && (
          <line x1={xs(node.hi[0])} y1={padT} x2={xs(node.hi[0])} y2={padT + chartH} stroke="#c8311c" strokeWidth={2} strokeDasharray="6,3" />
        )}
        {Number.isFinite(node.lo[1]) && node.lo[1] > 0 && (
          <line x1={padL} y1={ys(node.lo[1])} x2={padL + chartW} y2={ys(node.lo[1])} stroke="#c8311c" strokeWidth={2} strokeDasharray="6,3" />
        )}
        {Number.isFinite(node.hi[1]) && node.hi[1] < 4 && (
          <line x1={padL} y1={ys(node.hi[1])} x2={padL + chartW} y2={ys(node.hi[1])} stroke="#c8311c" strokeWidth={2} strokeDasharray="6,3" />
        )}

        {/* integer lattice */}
        {lattice.map((p, i) => {
          const inside =
            p.x >= node.lo[0] - 1e-9 &&
            p.x <= node.hi[0] + 1e-9 &&
            p.y >= node.lo[1] - 1e-9 &&
            p.y <= node.hi[1] + 1e-9 &&
            2 * p.x + p.y <= 7 + 1e-9 &&
            p.x + 2 * p.y <= 6 + 1e-9;
          return (
            <circle
              key={i}
              cx={xs(p.x)}
              cy={ys(p.y)}
              r={inside ? 3 : 2}
              fill={inside ? "#444" : "#ccc"}
            />
          );
        })}

        {/* LP optimum at this node */}
        {lp.feasible && (
          <>
            <circle cx={xs(lp.x)} cy={ys(lp.y)} r={6} fill="#0b3da0" stroke="#fff" strokeWidth={2} />
            <text x={xs(lp.x) + 8} y={ys(lp.y) - 8} fontSize={11} fontFamily="monospace" fill="#0b3da0">
              ({lp.x.toFixed(2)}, {lp.y.toFixed(2)})
            </text>
          </>
        )}

        {/* incumbent */}
        {incumbent.x && (
          <>
            <rect
              x={xs(incumbent.x.x) - 5}
              y={ys(incumbent.x.y) - 5}
              width={10}
              height={10}
              fill="#f5a524"
              stroke="#fff"
              strokeWidth={2}
            />
            <text x={xs(incumbent.x.x) + 8} y={ys(incumbent.x.y) + 12} fontSize={10} fontFamily="monospace" fill="#a37300">
              incumbent ({incumbent.x.x}, {incumbent.x.y})
            </text>
          </>
        )}

        {/* axis labels */}
        {[0, 1, 2, 3, 4].map((v) => (
          <text key={`xl${v}`} x={xs(v)} y={padT + chartH + 14} textAnchor="middle" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        {[0, 1, 2, 3, 4].map((v) => (
          <text key={`yl${v}`} x={padL - 6} y={ys(v) + 3} textAnchor="end" fontSize={10} fontFamily="monospace" fill="#666">
            {v}
          </text>
        ))}
        <text x={padL + chartW - 6} y={padT + chartH - 6} textAnchor="end" fontSize={11} fontFamily="monospace" fill="#666">
          x
        </text>
        <text x={padL + 6} y={padT + 12} fontSize={11} fontFamily="monospace" fill="#666">
          y
        </text>
      </svg>
    </div>
  );
}

// ============================================================
// Tree view
// ============================================================
function TreeView({ visibleNodes, selectedId, setSelectedId, incumbent, dualBound }) {
  const W = 480, H = 480;
  // Layout: position each node by depth (y) and an in-order x.
  // Compute positions by traversing tree.
  const positions = {};
  const layoutSubtree = (id, xL, xR, depth) => {
    const cx = (xL + xR) / 2;
    positions[id] = { x: cx, y: 30 + depth * 70 };
    const node = TREE.nodes[id];
    if (node.children) {
      const mid = (xL + xR) / 2;
      layoutSubtree(node.children[0], xL, mid, depth + 1);
      layoutSubtree(node.children[1], mid, xR, depth + 1);
    }
  };
  layoutSubtree(0, 20, W - 20, 0);

  return (
    <div style={panel}>
      <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>
        B&amp;B tree
      </div>
      <div style={{ fontSize: 12, color: "#444", fontFamily: "monospace", marginBottom: 8 }}>
        primal (incumbent) ={" "}
        <span style={{ color: "#a37300", fontWeight: 700 }}>
          {incumbent.obj === -Infinity ? "−∞" : incumbent.obj.toFixed(2)}
        </span>
        {"   "}
        dual (best LP) ={" "}
        <span style={{ color: "#0b3da0", fontWeight: 700 }}>
          {dualBound === -Infinity ? "+∞" : dualBound.toFixed(2)}
        </span>
        {"   "}
        gap ={" "}
        <span style={{ fontWeight: 700 }}>
          {incumbent.obj === -Infinity || dualBound === -Infinity
            ? "—"
            : `${(((dualBound - incumbent.obj) / Math.max(0.01, Math.abs(incumbent.obj))) * 100).toFixed(2)}%`}
        </span>
      </div>
      <svg width={W} height={H}>
        {/* edges */}
        {TREE.nodes.map((node) => {
          if (!node.children) return null;
          if (!visibleNodes.has(node.id)) return null;
          return node.children.map((cid) => {
            if (!visibleNodes.has(cid)) return null;
            const a = positions[node.id];
            const b = positions[cid];
            const child = TREE.nodes[cid];
            const v = child.branchInfo.var === 0 ? "x" : "y";
            const op = child.branchInfo.op;
            const val = child.branchInfo.val;
            return (
              <g key={cid}>
                <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#888" strokeWidth={1.5} />
                <rect
                  x={(a.x + b.x) / 2 - 22}
                  y={(a.y + b.y) / 2 - 9}
                  width={44}
                  height={18}
                  rx={3}
                  fill="#fff"
                  stroke="#bbb"
                />
                <text
                  x={(a.x + b.x) / 2}
                  y={(a.y + b.y) / 2 + 4}
                  textAnchor="middle"
                  fontSize={11}
                  fontFamily="monospace"
                  fill="#222"
                >
                  {v} {op === "<=" ? "≤" : "≥"} {val}
                </text>
              </g>
            );
          });
        })}

        {/* nodes */}
        {TREE.nodes.map((node) => {
          if (!visibleNodes.has(node.id)) return null;
          const p = positions[node.id];
          const isSelected = node.id === selectedId;
          let color = "#aaa";
          if (node.status === "branched") color = "#0b3da0";
          else if (node.status === "integer") color = "#1f4e3d";
          else if (node.status === "infeasible") color = "#888";
          else if (node.status === "fathomed_bound") color = "#c8311c";
          if (node.becameIncumbent) color = "#f5a524";
          return (
            <g key={node.id} style={{ cursor: "pointer" }} onClick={() => setSelectedId(node.id)}>
              <circle
                cx={p.x}
                cy={p.y}
                r={isSelected ? 22 : 18}
                fill={color}
                stroke={isSelected ? "#000" : "#fff"}
                strokeWidth={isSelected ? 3 : 2}
              />
              <text
                x={p.x}
                y={p.y - 3}
                textAnchor="middle"
                fontSize={10}
                fontFamily="monospace"
                fill="#fff"
                fontWeight={700}
              >
                #{node.id}
              </text>
              <text x={p.x} y={p.y + 9} textAnchor="middle" fontSize={9} fontFamily="monospace" fill="#fff">
                {node.lpObj != null ? node.lpObj.toFixed(2) : "—"}
              </text>
            </g>
          );
        })}

        {/* legend */}
        <g transform={`translate(10, ${H - 100})`}>
          <rect x={0} y={0} width={170} height={92} fill="rgba(255,255,255,0.94)" stroke="#ccc" />
          <circle cx={12} cy={14} r={6} fill="#0b3da0" /><text x={24} y={18} fontSize={11}>branched</text>
          <circle cx={12} cy={32} r={6} fill="#1f4e3d" /><text x={24} y={36} fontSize={11}>integer-feasible</text>
          <circle cx={12} cy={50} r={6} fill="#f5a524" /><text x={24} y={54} fontSize={11}>incumbent</text>
          <circle cx={12} cy={68} r={6} fill="#c8311c" /><text x={24} y={72} fontSize={11}>fathomed (bound)</text>
          <circle cx={12} cy={86} r={6} fill="#888" /><text x={24} y={90} fontSize={11}>infeasible</text>
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Node detail panel
// ============================================================
function NodeDetail({ node, incumbent }) {
  if (!node || !node.lp) return null;
  return (
    <div style={{ marginTop: 18, padding: 14, background: "#f6f4ee", border: "1px solid #ece8dd", borderRadius: 8 }}>
      <div style={{ fontFamily: "monospace", fontSize: 10, color: "#888", letterSpacing: "0.18em", textTransform: "uppercase", marginBottom: 8 }}>
        Selected node detail (id {node.id})
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.6 }}>
        <div>
          Bounds added by branching:{" "}
          <Tex>{`${node.lo[0]} \\le x \\le ${Number.isFinite(node.hi[0]) ? node.hi[0] : "\\infty"}, \\;\\; ${node.lo[1]} \\le y \\le ${Number.isFinite(node.hi[1]) ? node.hi[1] : "\\infty"}`}</Tex>
        </div>
        <div style={{ marginTop: 4 }}>
          LP relaxation:{" "}
          {node.lp.feasible ? (
            <Tex>{`(x, y) = (${node.lp.x.toFixed(3)},\\; ${node.lp.y.toFixed(3)}),\\;\\; \\text{obj} = ${node.lp.obj.toFixed(3)}`}</Tex>
          ) : (
            <span>infeasible</span>
          )}
        </div>
        <div style={{ marginTop: 4 }}>
          Status: <b style={{ color: nodeColor(node.status) }}>{node.status || "open"}</b>
          {node.becameIncumbent && (
            <span style={{ marginLeft: 10, color: "#a37300" }}>★ became incumbent</span>
          )}
        </div>
        {node.status === "branched" && (
          <div style={{ marginTop: 4 }}>
            Branched on{" "}
            <Tex>{`${node.branchVar === 0 ? "x" : "y"} = ${node.branchVal.toFixed(3)}`}</Tex>{" "}
            (most-fractional rule).
          </div>
        )}
      </div>
    </div>
  );
}

function nodeColor(status) {
  if (status === "branched") return "#0b3da0";
  if (status === "integer") return "#1f4e3d";
  if (status === "fathomed_bound") return "#c8311c";
  if (status === "infeasible") return "#888";
  return "#444";
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
          <b>Three ways to fathom.</b> A node is closed when (1) its LP is
          infeasible, (2) the LP optimum is integer-feasible, or (3) the LP
          optimum's objective is no better than the current incumbent
          (fathom by bound).
        </li>
        <li>
          <b>Most-fractional branching.</b> We branch on the variable whose
          LP value is closest to <Tex>{`0.5`}</Tex> away from an integer.
          Other rules: pseudo-cost, strong branching, reliability branching.
        </li>
        <li>
          <b>Best-first vs depth-first.</b> Best-first explores the open
          node with the best (largest, for max) LP value — that's the one
          most likely to improve the dual bound. Depth-first dives deep
          quickly to find feasible solutions. Most solvers blend both.
        </li>
        <li>
          <b>Why the gap matters.</b> The gap{" "}
          <Tex>{`(\\text{dual} - \\text{primal}) / |\\text{primal}|`}</Tex>{" "}
          is the official 'how close are we' metric. When gap = 0, the
          incumbent is provably optimal.
        </li>
        <li>
          <b>What real solvers add.</b> Cutting planes (Gomory, MIR,
          covers) tighten the LP at every node. Strong branching evaluates
          children's LPs before committing. Heuristics (RINS, RENS,
          feasibility pump) find good incumbents fast.
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
const problemBox = {
  marginBottom: 16,
  padding: "12px 16px",
  background: "#f6f4ee",
  border: "1px solid #ece8dd",
  borderRadius: 8,
};
const eventBox = {
  marginBottom: 14,
  padding: "10px 14px",
  background: "#fff4c8",
  border: "1px solid #f5d68d",
  borderRadius: 8,
  fontSize: 14,
  lineHeight: 1.5,
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
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
