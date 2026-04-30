import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward, Shuffle } from "lucide-react";

/* ============================================================
   NEURAL NETWORK & BACKPROPAGATION — INTERACTIVE
   ISE 5406 (Nonlinear Programming)

   Three sections, all sharing one small 2 → 5 → 4 → 1 MLP:
     1. Forward pass: drag inputs, watch each neuron compute.
     2. Backprop training: train on y = sin(πx₁)·cos(πx₂),
        watch the loss descend and gradients flow on edges.
     3. Heatmaps: predicted function vs target, side by side.
   ============================================================ */

// ---------------- Network core ----------------
const ARCH = [2, 5, 4, 1];
const LAYER_NAMES = ["Input", "Hidden 1", "Hidden 2", "Output"];

const ACTS = {
  tanh: { f: (x) => Math.tanh(x), df: (y) => 1 - y * y },
  relu: { f: (x) => Math.max(0, x), df: (y) => (y > 0 ? 1 : 0) },
  sigmoid: { f: (x) => 1 / (1 + Math.exp(-x)), df: (y) => y * (1 - y) },
};

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function randn(rng) {
  const u = Math.max(rng(), 1e-9);
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function createNet(seed = 42) {
  const rng = mulberry32(seed);
  const W = [];
  const b = [];
  for (let l = 1; l < ARCH.length; l++) {
    const nIn = ARCH[l - 1];
    const nOut = ARCH[l];
    const scale = Math.sqrt(2 / nIn);
    const wMat = [];
    for (let j = 0; j < nOut; j++) {
      const row = [];
      for (let i = 0; i < nIn; i++) row.push(randn(rng) * scale);
      wMat.push(row);
    }
    W.push(wMat);
    b.push(Array.from({ length: nOut }, () => randn(rng) * 0.1));
  }
  return { W, b };
}

function forward(net, x, actName) {
  const act = ACTS[actName];
  const a = [x.slice()];
  const z = [null];
  for (let l = 0; l < net.W.length; l++) {
    const Wl = net.W[l];
    const bl = net.b[l];
    const aPrev = a[l];
    const zL = [];
    const aL = [];
    for (let j = 0; j < Wl.length; j++) {
      let s = bl[j];
      for (let i = 0; i < Wl[j].length; i++) s += Wl[j][i] * aPrev[i];
      zL.push(s);
      aL.push(l === net.W.length - 1 ? s : act.f(s));
    }
    z.push(zL);
    a.push(aL);
  }
  return { a, z };
}

function backward(net, fwd, yTrue, actName) {
  const act = ACTS[actName];
  const { a } = fwd;
  const L = net.W.length;
  const dW = net.W.map((W) => W.map((row) => row.map(() => 0)));
  const db = net.b.map((b) => b.map(() => 0));
  let delta = a[L].map((yh, j) => yh - yTrue[j]);
  for (let l = L - 1; l >= 0; l--) {
    const aPrev = a[l];
    for (let j = 0; j < net.W[l].length; j++) {
      db[l][j] = delta[j];
      for (let i = 0; i < net.W[l][j].length; i++) {
        dW[l][j][i] = delta[j] * aPrev[i];
      }
    }
    if (l > 0) {
      const newDelta = new Array(aPrev.length).fill(0);
      for (let i = 0; i < aPrev.length; i++) {
        let s = 0;
        for (let j = 0; j < net.W[l].length; j++) s += net.W[l][j][i] * delta[j];
        newDelta[i] = s * act.df(aPrev[i]);
      }
      delta = newDelta;
    }
  }
  return { dW, db };
}

function applyGrads(net, grads, lr) {
  for (let l = 0; l < net.W.length; l++) {
    for (let j = 0; j < net.W[l].length; j++) {
      net.b[l][j] -= lr * grads.db[l][j];
      for (let i = 0; i < net.W[l][j].length; i++) {
        net.W[l][j][i] -= lr * grads.dW[l][j][i];
      }
    }
  }
}

function gradNorm(grads) {
  let s = 0;
  for (const M of grads.dW) for (const r of M) for (const v of r) s += v * v;
  for (const v of grads.db.flat()) s += v * v;
  return Math.sqrt(s);
}

// ---------------- Visualization helpers ----------------
const COLOR_POS = "#1f4e3d";
const COLOR_NEG = "#c8311c";
const COLOR_INK = "#1a1815";

function nodePositions(arch, w, h, padX, padY) {
  const positions = [];
  for (let l = 0; l < arch.length; l++) {
    const layer = [];
    const x = padX + (w - 2 * padX) * (l / Math.max(1, arch.length - 1));
    const n = arch[l];
    for (let j = 0; j < n; j++) {
      const y = padY + (h - 2 * padY) * ((j + 1) / (n + 1));
      layer.push({ x, y });
    }
    positions.push(layer);
  }
  return positions;
}

function activationColor(v) {
  const m = Math.min(1, Math.abs(v));
  return v >= 0
    ? `rgba(31, 78, 61, ${0.15 + 0.75 * m})`
    : `rgba(200, 49, 28, ${0.15 + 0.75 * m})`;
}
function weightColor(w) {
  const m = Math.min(1, Math.abs(w) / 1.5);
  if (w >= 0) return `rgba(26, 24, 21, ${0.15 + 0.7 * m})`;
  return `rgba(200, 49, 28, ${0.2 + 0.7 * m})`;
}

// ---------------- Network SVG ----------------
function NetSVG({
  net,
  fwd,
  width = 920,
  height = 340,
  gradMags = null,
  gradMax = null,
  hovered = null,
  setHovered = null,
}) {
  const padX = 50;
  const padY = 36;
  const positions = useMemo(
    () => nodePositions(ARCH, width, height, padX, padY),
    [width, height]
  );

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ background: "#fff", borderRadius: 6 }}
    >
      {ARCH.map((_, l) => (
        <text
          key={`ln-${l}`}
          x={positions[l][0].x}
          y={18}
          textAnchor="middle"
          fontFamily="monospace"
          fontSize="10"
          fill="#888"
          style={{ letterSpacing: "0.12em" }}
        >
          {LAYER_NAMES[l].toUpperCase()}
        </text>
      ))}

      {/* edges */}
      {net.W.map((Wl, l) =>
        Wl.map((row, j) =>
          row.map((w, i) => {
            const a = positions[l][i];
            const b = positions[l + 1][j];
            let strokeWidth = 0.5 + Math.min(4, Math.abs(w) * 1.5);
            let stroke = weightColor(w);
            if (gradMags) {
              const gm = gradMags[l][j][i];
              const intensity = Math.min(1, gm / (gradMax || 0.3));
              if (intensity > 0.05) {
                stroke = `rgba(31, 78, 61, ${0.3 + 0.7 * intensity})`;
                strokeWidth = 1 + 4 * intensity;
              }
            }
            return (
              <line
                key={`e-${l}-${j}-${i}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={stroke}
                strokeWidth={strokeWidth}
              >
                <title>
                  {`W[${l}][${j}][${i}] = ${w.toFixed(3)}` +
                    (gradMags ? `   ∂L/∂W = ${gradMags[l][j][i].toFixed(4)}` : "")}
                </title>
              </line>
            );
          })
        )
      )}

      {/* nodes */}
      {ARCH.map((nL, l) =>
        Array.from({ length: nL }, (_, j) => {
          const p = positions[l][j];
          const a = fwd.a[l][j];
          const isHover = hovered && hovered.l === l && hovered.j === j;
          return (
            <g key={`n-${l}-${j}`}>
              <circle
                cx={p.x}
                cy={p.y}
                r={isHover ? 21 : 18}
                fill={activationColor(a)}
                stroke={isHover ? "#111" : "#fff"}
                strokeWidth={isHover ? 2.5 : 1.5}
                onMouseEnter={setHovered ? () => setHovered({ l, j }) : undefined}
                onMouseLeave={setHovered ? () => setHovered(null) : undefined}
                style={{ cursor: setHovered ? "pointer" : "default" }}
              />
              <text
                x={p.x}
                y={p.y + 4}
                textAnchor="middle"
                fontFamily="monospace"
                fontSize="11"
                fontWeight="600"
                fill={Math.abs(a) > 0.5 ? "#f4efe6" : "#1a1815"}
                style={{ pointerEvents: "none" }}
              >
                {a.toFixed(2)}
              </text>
            </g>
          );
        })
      )}
    </svg>
  );
}

// ---------------- Loss curve SVG ----------------
function LossCurve({ history, step }) {
  const W = 460;
  const H = 320;
  const pad = 38;

  if (history.length < 2) {
    return (
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ width: "100%", maxHeight: 360, background: "#fff", borderRadius: 6 }}
      >
        <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="#fff" stroke="#ddd" />
        <text
          x={W / 2}
          y={H / 2}
          textAnchor="middle"
          fontFamily="monospace"
          fontSize={12}
          fill="#999"
        >
          press Train to begin
        </text>
      </svg>
    );
  }

  const eps = 1e-6;
  const logs = history.map((v) => Math.log10(Math.max(eps, v)));
  const yMin = Math.min(-3, ...logs);
  const yMax = Math.max(0.5, ...logs);
  const xMax = history.length;
  const path = logs
    .map((v, i) => {
      const x = pad + (W - 2 * pad) * (i / Math.max(1, xMax - 1));
      const y = pad + ((H - 2 * pad) * (yMax - v)) / (yMax - yMin);
      return `${i === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid meet"
      style={{ width: "100%", maxHeight: 360, background: "#fff", borderRadius: 6 }}
    >
      <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="#fafafa" stroke="#ddd" />

      {/* y-axis label */}
      <text
        x={12}
        y={H / 2}
        transform={`rotate(-90 12 ${H / 2})`}
        textAnchor="middle"
        fontFamily="monospace"
        fontSize={10}
        fill="#666"
      >
        log₁₀(ℒ)
      </text>
      {/* x-axis label */}
      <text
        x={W / 2}
        y={H - 8}
        textAnchor="middle"
        fontFamily="monospace"
        fontSize={10}
        fill="#666"
      >
        step
      </text>

      {[0, 1, 2, 3, 4].map((i) => {
        const v = yMax - ((yMax - yMin) * i) / 4;
        const y = pad + ((H - 2 * pad) * i) / 4;
        return (
          <g key={`grid-${i}`}>
            <line
              x1={pad}
              y1={y}
              x2={W - pad}
              y2={y}
              stroke="#ccc"
              strokeWidth={0.5}
              strokeDasharray="2 4"
            />
            <text
              x={pad - 6}
              y={y + 3}
              textAnchor="end"
              fontFamily="monospace"
              fontSize={9}
              fill="#888"
            >
              {v.toFixed(1)}
            </text>
          </g>
        );
      })}

      <path d={path} fill="none" stroke={COLOR_NEG} strokeWidth={1.8} />

      <text
        x={W - pad - 4}
        y={pad + 14}
        textAnchor="end"
        fontFamily="monospace"
        fontSize={11}
        fontWeight={700}
        fill={COLOR_NEG}
      >
        step {step} · ℒ = {history[history.length - 1].toFixed(4)}
      </text>
    </svg>
  );
}

// ---------------- Heatmap canvas ----------------
function colormap(v) {
  const c = Math.max(-1, Math.min(1, v));
  if (c >= 0) {
    const t = c;
    return [
      Math.round(244 + (31 - 244) * t),
      Math.round(239 + (78 - 239) * t),
      Math.round(230 + (61 - 230) * t),
    ];
  } else {
    const t = -c;
    return [
      Math.round(244 + (200 - 244) * t),
      Math.round(239 + (49 - 239) * t),
      Math.round(230 + (28 - 230) * t),
    ];
  }
}

function HeatmapCanvas({ fn, version, label }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width;
    const H = cv.height;
    const img = ctx.createImageData(W, H);
    for (let py = 0; py < H; py++) {
      for (let px = 0; px < W; px++) {
        const x1 = (px / W) * 2 - 1;
        const x2 = 1 - (py / H) * 2;
        const v = fn(x1, x2);
        const [r, g, b] = colormap(v);
        const idx = (py * W + px) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [version, fn]);
  return (
    <div>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 11,
          color: "#666",
          letterSpacing: "0.12em",
          marginBottom: 6,
          textTransform: "uppercase",
        }}
      >
        {label}
      </div>
      <canvas
        ref={ref}
        width={240}
        height={240}
        style={{
          width: "100%",
          maxWidth: 320,
          imageRendering: "pixelated",
          border: "1px solid #ddd",
          borderRadius: 4,
        }}
      />
    </div>
  );
}

// ---------------- Section 1: Forward pass ----------------
function ForwardPassSection({ act, setAct }) {
  const [x1, setX1] = useState(0.6);
  const [x2, setX2] = useState(-0.4);
  const [hovered, setHovered] = useState(null);
  const net = useMemo(() => createNet(7), []);
  const fwd = useMemo(() => forward(net, [x1, x2], act), [net, x1, x2, act]);

  const inspector = useMemo(() => {
    if (!hovered) return null;
    const { l, j } = hovered;
    if (l === 0) {
      return {
        title: `Input x${j + 1}`,
        body: `value = ${fwd.a[0][j].toFixed(4)}\nrole: raw input feature, no computation.`,
      };
    }
    const isOut = l === ARCH.length - 1;
    const Wrow = net.W[l - 1][j];
    const aPrev = fwd.a[l - 1];
    const z = fwd.z[l][j];
    const a = fwd.a[l][j];
    const b = net.b[l - 1][j];
    let dot = `${b.toFixed(2)}`;
    for (let i = 0; i < Wrow.length; i++) {
      const w = Wrow[i];
      const x = aPrev[i];
      const sgn = w * x >= 0 ? " + " : " − ";
      dot += `${sgn}(${Math.abs(w).toFixed(2)}·${x.toFixed(2)})`;
    }
    return {
      title: isOut ? "Output ŷ" : `Hidden h[${l}][${j}]`,
      body: `z = ${dot} = ${z.toFixed(4)}\na = ${isOut ? "identity" : act}(z) = ${a.toFixed(4)}`,
    };
  }, [hovered, fwd, net, act]);

  return (
    <section style={section}>
      <h2 style={h2}>1. Forward pass</h2>
      <p style={pTxt}>
        A neural network is a stack of weighted sums passed through nonlinear
        squashing functions. Drag the input sliders below; each circle's fill
        encodes its activation — green for positive, red for negative, opacity
        for magnitude. Edge thickness encodes weight magnitude.{" "}
        <b>Hover any neuron</b> to see its computation.
      </p>

      <NetSVG net={net} fwd={fwd} hovered={hovered} setHovered={setHovered} />

      <div style={legendRow}>
        <span style={legendItem}>
          <span style={{ ...swatch, background: COLOR_POS }} />
          positive activation
        </span>
        <span style={legendItem}>
          <span style={{ ...swatch, background: COLOR_NEG }} />
          negative activation
        </span>
        <span style={{ ...legendItem, color: "#888" }}>
          edge thickness ∝ |weight|
        </span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 14,
          marginTop: 14,
        }}
      >
        <div>
          <label style={label}>
            x₁: <b>{x1.toFixed(2)}</b>
          </label>
          <input
            type="range"
            min={-2}
            max={2}
            step={0.05}
            value={x1}
            onChange={(e) => setX1(+e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <label style={label}>
            x₂: <b>{x2.toFixed(2)}</b>
          </label>
          <input
            type="range"
            min={-2}
            max={2}
            step={0.05}
            value={x2}
            onChange={(e) => setX2(+e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <label style={label}>activation</label>
          <select
            value={act}
            onChange={(e) => setAct(e.target.value)}
            style={selectStyle}
          >
            <option value="tanh">tanh</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">sigmoid</option>
          </select>
        </div>
        <div style={{ display: "flex", alignItems: "flex-end" }}>
          <button
            onClick={() => {
              setX1(+(Math.random() * 4 - 2).toFixed(2));
              setX2(+(Math.random() * 4 - 2).toFixed(2));
            }}
            style={btn}
          >
            <Shuffle size={14} /> randomize
          </button>
        </div>
      </div>

      <div style={inspectorBox}>
        {inspector ? (
          <>
            <div style={inspectorTitle}>{inspector.title}</div>
            <pre style={inspectorBody}>{inspector.body}</pre>
          </>
        ) : (
          <div style={{ color: "#888", fontFamily: "monospace", fontSize: 13 }}>
            hover a neuron above • each unit computes h = σ(W·x + b)
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------- Section 2: Backprop training ----------------
function targetFn(x1, x2) {
  return Math.sin(Math.PI * x1) * Math.cos(Math.PI * x2);
}

function BackpropSection({ act, sharedNetRef, setNetVersion, netVersion }) {
  const stepRef = useRef(0);
  const lossHistRef = useRef([]);
  const lastGradsRef = useRef(null);
  const [training, setTraining] = useState(false);
  const [lr, setLr] = useState(0.05);
  const [speed, setSpeed] = useState(20);
  // tickCounter triggers re-renders after mutation.
  const [tick, setTick] = useState(0);

  const trainStep = useCallback(() => {
    const x1 = Math.random() * 2 - 1;
    const x2 = Math.random() * 2 - 1;
    const yTrue = targetFn(x1, x2);
    const fwd = forward(sharedNetRef.current, [x1, x2], act);
    const grads = backward(sharedNetRef.current, fwd, [yTrue], act);
    applyGrads(sharedNetRef.current, grads, lr);
    stepRef.current += 1;
    const yhat = fwd.a[fwd.a.length - 1][0];
    const loss = 0.5 * (yhat - yTrue) ** 2;
    if (lossHistRef.current.length === 0) {
      lossHistRef.current.push(loss);
    } else {
      const last = lossHistRef.current[lossHistRef.current.length - 1];
      lossHistRef.current.push(0.95 * last + 0.05 * loss);
    }
    if (lossHistRef.current.length > 800) lossHistRef.current.shift();
    lastGradsRef.current = {
      grads,
      yhat,
      yTrue,
      gn: gradNorm(grads),
    };
  }, [act, lr, sharedNetRef]);

  useEffect(() => {
    if (!training) return;
    let raf = 0;
    let acc = 0;
    let last = 0;
    const loop = (t) => {
      if (last === 0) last = t;
      const dt = t - last;
      last = t;
      acc += (dt * speed) / 1000;
      let count = 0;
      while (acc >= 1 && count < 50) {
        trainStep();
        acc -= 1;
        count += 1;
      }
      if (count > 0) {
        setNetVersion((v) => v + 1);
        setTick((n) => n + 1);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [training, speed, trainStep, setNetVersion]);

  const reset = () => {
    sharedNetRef.current = createNet(Math.floor(Math.random() * 1000));
    stepRef.current = 0;
    lossHistRef.current = [];
    lastGradsRef.current = null;
    setTraining(false);
    setNetVersion((v) => v + 1);
    setTick((n) => n + 1);
  };

  // Snapshot rendering
  const fwdSnap = useMemo(
    () => forward(sharedNetRef.current, [0.3, 0.3], act),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [act, tick, netVersion, sharedNetRef]
  );
  const gradMags = lastGradsRef.current
    ? lastGradsRef.current.grads.dW.map((M) =>
        M.map((r) => r.map((v) => Math.abs(v)))
      )
    : null;
  const gradMax = lastGradsRef.current
    ? Math.max(
        0.05,
        ...lastGradsRef.current.grads.dW.flat(2).map((v) => Math.abs(v))
      )
    : null;

  return (
    <section style={section}>
      <h2 style={h2}>2. Backpropagation in motion</h2>
      <p style={pTxt}>
        The same network now trains on the function{" "}
        <code style={codeInline}>y = sin(πx₁) · cos(πx₂)</code>. Each step, a
        random point is sampled, the network computes its prediction, and
        gradients flow backward by the chain rule. The green halos on edges
        show <code style={codeInline}>‖∂ℒ/∂W‖</code> — the weights getting the
        strongest update signal. Learning rate <code style={codeInline}>η</code>{" "}
        and speed are knobs below.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 1.5fr) minmax(0, 1fr)",
          gap: 18,
          alignItems: "stretch",
        }}
      >
        <div>
          <div style={subLabel}>NETWORK · gradient halos in green</div>
          <NetSVG
            net={sharedNetRef.current}
            fwd={fwdSnap}
            gradMags={gradMags}
            gradMax={gradMax}
          />
        </div>
        <div>
          <div style={subLabel}>LOSS ℒ OVER STEPS (LOG SCALE)</div>
          <LossCurve history={lossHistRef.current} step={stepRef.current} />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 8,
          marginTop: 14,
          flexWrap: "wrap",
          alignItems: "flex-end",
        }}
      >
        <button onClick={() => setTraining((t) => !t)} style={btnPrimary}>
          {training ? <Pause size={16} /> : <Play size={16} />}
          {training ? "Pause" : "Train"}
        </button>
        <button
          onClick={() => {
            trainStep();
            setNetVersion((v) => v + 1);
            setTick((n) => n + 1);
          }}
          style={btn}
        >
          <StepForward size={16} /> Step Once
        </button>
        <button onClick={reset} style={btn}>
          <RotateCcw size={16} /> Reset Weights
        </button>

        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={label}>
            η (lr): <b>{lr.toFixed(3)}</b>
          </label>
          <input
            type="range"
            min={0.001}
            max={0.3}
            step={0.001}
            value={lr}
            onChange={(e) => setLr(+e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
        <div style={{ flex: 1, minWidth: 140 }}>
          <label style={label}>
            speed: <b>{speed}</b>
          </label>
          <input
            type="range"
            min={1}
            max={50}
            step={1}
            value={speed}
            onChange={(e) => setSpeed(+e.target.value)}
            style={{ width: "100%" }}
          />
        </div>
      </div>

      <div
        style={{
          ...statBox,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
          gap: 8,
          marginTop: 14,
        }}
      >
        <Stat label="step" value={stepRef.current} />
        <Stat
          label="loss (EMA)"
          value={
            lossHistRef.current.length
              ? lossHistRef.current[lossHistRef.current.length - 1].toFixed(4)
              : "—"
          }
        />
        <Stat
          label="‖∇W‖"
          value={lastGradsRef.current ? lastGradsRef.current.gn.toFixed(3) : "—"}
        />
        <Stat
          label="ŷ"
          value={lastGradsRef.current ? lastGradsRef.current.yhat.toFixed(3) : "—"}
        />
        <Stat
          label="y target"
          value={lastGradsRef.current ? lastGradsRef.current.yTrue.toFixed(3) : "—"}
        />
      </div>
    </section>
  );
}

// ---------------- Section 3: Heatmaps ----------------
function HeatmapsSection({ act, sharedNetRef, netVersion }) {
  // Throttle: only re-draw the prediction heatmap every N net-version bumps.
  const lastDrawnRef = useRef(-1);
  const [drawVersion, setDrawVersion] = useState(0);
  useEffect(() => {
    if (netVersion - lastDrawnRef.current >= 5 || lastDrawnRef.current === -1) {
      lastDrawnRef.current = netVersion;
      setDrawVersion((v) => v + 1);
    }
  }, [netVersion]);

  const predFn = useCallback(
    (x1, x2) => {
      const fwd = forward(sharedNetRef.current, [x1, x2], act);
      return fwd.a[fwd.a.length - 1][0];
    },
    [act, sharedNetRef, drawVersion] // eslint-disable-line react-hooks/exhaustive-deps
  );
  const targFn = useCallback((x1, x2) => targetFn(x1, x2), []);

  return (
    <section style={section}>
      <h2 style={h2}>3. Learned vs target</h2>
      <p style={pTxt}>
        The heatmap below shows what the network has learned to compute over the
        entire input plane (x₁, x₂) ∈ [−1, 1]². As training proceeds, the left
        panel converges toward the right.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: 18,
          marginTop: 8,
        }}
      >
        <HeatmapCanvas
          fn={predFn}
          version={drawVersion}
          label="prediction ŷ(x₁, x₂)"
        />
        <HeatmapCanvas
          fn={targFn}
          version={0}
          label="target y = sin(πx₁) · cos(πx₂)"
        />
      </div>

      <div
        style={{
          marginTop: 12,
          fontSize: 12,
          color: "#666",
          fontFamily: "monospace",
        }}
      >
        green = positive · red = negative · cream = near 0
      </div>
    </section>
  );
}

// ---------------- Top-level ----------------
export default function NNDemo() {
  const [act, setAct] = useState("tanh");
  // Net shared between Section 2 (training) and Section 3 (heatmaps).
  const sharedNetRef = useRef(createNet(13));
  const [netVersion, setNetVersion] = useState(0);

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        Neural Networks &amp; Backpropagation
      </h1>
      <p style={{ color: "#666", marginBottom: 22, maxWidth: 880 }}>
        A small 2 → 5 → 4 → 1 multilayer perceptron, drawn in full. First the
        forward pass with sliders. Then training by backprop on a 2D target
        function — watch the loss descend and gradients glow on the edges.
        Finally, side-by-side heatmaps of what the network has learned vs the
        target.
      </p>

      <ForwardPassSection act={act} setAct={setAct} />
      <BackpropSection
        act={act}
        sharedNetRef={sharedNetRef}
        setNetVersion={setNetVersion}
        netVersion={netVersion}
      />
      <HeatmapsSection
        act={act}
        sharedNetRef={sharedNetRef}
        netVersion={netVersion}
      />
    </div>
  );
}

// ---------------- UI bits ----------------
function Stat({ label, value }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "3px 0" }}>
      <span style={{ color: "#666", fontFamily: "monospace", fontSize: 12 }}>
        {label}
      </span>
      <span style={{ fontFamily: "monospace", fontSize: 12, fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}

const section = {
  padding: "28px 28px 24px",
  marginBottom: 20,
  background: "#fff",
  borderRadius: 12,
  border: "1px solid #e7e7e7",
  boxShadow: "0 1px 2px rgba(0,0,0,0.03)",
};
const h2 = { fontSize: 20, fontWeight: 800, marginBottom: 8 };
const pTxt = { color: "#444", lineHeight: 1.55, marginBottom: 14, maxWidth: 800 };
const subLabel = {
  fontFamily: "monospace",
  fontSize: 10,
  color: "#666",
  letterSpacing: "0.12em",
  marginBottom: 6,
  textTransform: "uppercase",
};
const codeInline = {
  background: "#f0eee9",
  padding: "1px 6px",
  borderRadius: 4,
  fontFamily: "monospace",
  fontSize: 13,
};
const legendRow = {
  display: "flex",
  gap: 18,
  flexWrap: "wrap",
  marginTop: 12,
  fontSize: 12,
  fontFamily: "monospace",
  color: "#444",
};
const legendItem = { display: "flex", alignItems: "center", gap: 6 };
const swatch = {
  display: "inline-block",
  width: 12,
  height: 12,
  borderRadius: 3,
  border: "1px solid rgba(0,0,0,0.15)",
};
const label = {
  display: "block",
  fontSize: 12,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
};
const selectStyle = {
  width: "100%",
  padding: "6px 8px",
  borderRadius: 6,
  border: "1px solid #ccc",
  background: "#fff",
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
const statBox = {
  padding: "10px 14px",
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 8,
};
const inspectorBox = {
  marginTop: 14,
  padding: "12px 16px",
  background: "#f4efe6",
  border: "1px solid #ddd",
  borderRadius: 8,
};
const inspectorTitle = {
  fontFamily: "monospace",
  fontSize: 12,
  fontWeight: 700,
  color: "#444",
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  marginBottom: 6,
};
const inspectorBody = {
  fontFamily: "monospace",
  fontSize: 13,
  color: "#222",
  margin: 0,
  whiteSpace: "pre-wrap",
  lineHeight: 1.5,
};
