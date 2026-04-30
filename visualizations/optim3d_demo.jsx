import React, { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { Play, Pause, RotateCcw, StepForward } from "lucide-react";

/* ============================================================
   FIRST-ORDER METHODS — 3D VISUALIZATION
   For ISE 5405/5406. Same test functions as the 2D demo, drawn
   as a height-mapped surface. Trajectory of the optimizer is
   plotted as a polyline floating just above the surface, with
   a "shadow" projected onto the xy-plane below.
   Mouse-drag the canvas to rotate the camera.
   ============================================================ */

// ---------------- Test functions ----------------
const FUNCTIONS = {
  bowl: {
    name: "Bowl  f = ½(x² + y²)",
    f: (x, y) => 0.5 * (x * x + y * y),
    g: (x, y) => [x, y],
    box: [-3, 3, -3, 3],
    start: [-2.5, 2.5],
    zClip: 9,
  },
  ill: {
    name: "Ill-Conditioned Bowl  ½(x² + 25y²)",
    f: (x, y) => 0.5 * (x * x + 25 * y * y),
    g: (x, y) => [x, 25 * y],
    box: [-3, 3, -1.4, 1.4],
    start: [-2.5, 1.0],
    zClip: 25,
  },
  rosenbrock: {
    name: "Rosenbrock  (1−x)² + 100(y−x²)²",
    f: (x, y) => (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x),
    g: (x, y) => [
      -2 * (1 - x) - 400 * x * (y - x * x),
      200 * (y - x * x),
    ],
    box: [-2, 2, -1, 3],
    start: [-1.5, 2.5],
    zClip: 400,
  },
  beale: {
    name: "Beale",
    f: (x, y) => {
      const A = 1.5 - x + x * y;
      const B = 2.25 - x + x * y * y;
      const C = 2.625 - x + x * y * y * y;
      return A * A + B * B + C * C;
    },
    g: (x, y) => {
      const A = 1.5 - x + x * y;
      const B = 2.25 - x + x * y * y;
      const C = 2.625 - x + x * y * y * y;
      return [
        2 * A * (-1 + y) + 2 * B * (-1 + y * y) + 2 * C * (-1 + y * y * y),
        2 * A * x + 2 * B * (2 * x * y) + 2 * C * (3 * x * y * y),
      ];
    },
    box: [-1, 4, -2, 2],
    start: [3, 1.6],
    zClip: 200,
  },
  himmelblau: {
    name: "Himmelblau (multi-modal)",
    f: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return a * a + b * b;
    },
    g: (x, y) => {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
    },
    box: [-5, 5, -5, 5],
    start: [-3, 3.5],
    zClip: 400,
  },
};

// ---------------- Optimizers ----------------
// state object carries whatever the method needs.
const METHODS = {
  gd: {
    name: "Gradient Descent",
    init: () => ({}),
    step: (state, fn, hp) => {
      const [gx, gy] = fn.g(state.x, state.y);
      return { ...state, x: state.x - hp.lr * gx, y: state.y - hp.lr * gy };
    },
    hp: [{ key: "lr", label: "lr (η)", min: 0.001, max: 0.5, step: 0.001, def: 0.05 }],
  },
  momentum: {
    name: "Heavy-Ball Momentum",
    init: () => ({ vx: 0, vy: 0 }),
    step: (state, fn, hp) => {
      const [gx, gy] = fn.g(state.x, state.y);
      const vx = hp.beta * state.vx + gx;
      const vy = hp.beta * state.vy + gy;
      return {
        ...state,
        x: state.x - hp.lr * vx,
        y: state.y - hp.lr * vy,
        vx,
        vy,
      };
    },
    hp: [
      { key: "lr", label: "lr (η)", min: 0.001, max: 0.5, step: 0.001, def: 0.02 },
      { key: "beta", label: "β (momentum)", min: 0, max: 0.99, step: 0.01, def: 0.9 },
    ],
  },
  nesterov: {
    name: "Nesterov AG",
    init: () => ({ vx: 0, vy: 0 }),
    step: (state, fn, hp) => {
      const lookx = state.x - hp.beta * state.vx;
      const looky = state.y - hp.beta * state.vy;
      const [gx, gy] = fn.g(lookx, looky);
      const vx = hp.beta * state.vx + gx;
      const vy = hp.beta * state.vy + gy;
      return {
        ...state,
        x: state.x - hp.lr * vx,
        y: state.y - hp.lr * vy,
        vx,
        vy,
      };
    },
    hp: [
      { key: "lr", label: "lr (η)", min: 0.001, max: 0.5, step: 0.001, def: 0.02 },
      { key: "beta", label: "β", min: 0, max: 0.99, step: 0.01, def: 0.9 },
    ],
  },
  rmsprop: {
    name: "RMSProp",
    init: () => ({ sx: 0, sy: 0 }),
    step: (state, fn, hp) => {
      const [gx, gy] = fn.g(state.x, state.y);
      const sx = hp.rho * state.sx + (1 - hp.rho) * gx * gx;
      const sy = hp.rho * state.sy + (1 - hp.rho) * gy * gy;
      return {
        ...state,
        x: state.x - (hp.lr * gx) / (Math.sqrt(sx) + 1e-8),
        y: state.y - (hp.lr * gy) / (Math.sqrt(sy) + 1e-8),
        sx,
        sy,
      };
    },
    hp: [
      { key: "lr", label: "lr (η)", min: 0.001, max: 0.5, step: 0.001, def: 0.05 },
      { key: "rho", label: "ρ", min: 0, max: 0.999, step: 0.001, def: 0.9 },
    ],
  },
  adam: {
    name: "Adam",
    init: () => ({ mx: 0, my: 0, sx: 0, sy: 0, t: 0 }),
    step: (state, fn, hp) => {
      const [gx, gy] = fn.g(state.x, state.y);
      const t = state.t + 1;
      const mx = hp.b1 * state.mx + (1 - hp.b1) * gx;
      const my = hp.b1 * state.my + (1 - hp.b1) * gy;
      const sx = hp.b2 * state.sx + (1 - hp.b2) * gx * gx;
      const sy = hp.b2 * state.sy + (1 - hp.b2) * gy * gy;
      const mhx = mx / (1 - Math.pow(hp.b1, t));
      const mhy = my / (1 - Math.pow(hp.b1, t));
      const shx = sx / (1 - Math.pow(hp.b2, t));
      const shy = sy / (1 - Math.pow(hp.b2, t));
      return {
        ...state,
        x: state.x - (hp.lr * mhx) / (Math.sqrt(shx) + 1e-8),
        y: state.y - (hp.lr * mhy) / (Math.sqrt(shy) + 1e-8),
        mx,
        my,
        sx,
        sy,
        t,
      };
    },
    hp: [
      { key: "lr", label: "lr (η)", min: 0.001, max: 0.5, step: 0.001, def: 0.1 },
      { key: "b1", label: "β₁", min: 0, max: 0.999, step: 0.001, def: 0.9 },
      { key: "b2", label: "β₂", min: 0, max: 0.9999, step: 0.0001, def: 0.999 },
    ],
  },
};

// ---------------- 3D projection ----------------
// Camera: spherical (theta = azimuth around z, phi = elevation tilt).
// Orthographic projection.
function project([x, y, z], camera) {
  const ct = Math.cos(camera.theta);
  const st = Math.sin(camera.theta);
  // rotate around z by theta
  const x1 = x * ct - y * st;
  const y1 = x * st + y * ct;
  const z1 = z;
  // rotate around x by phi (elevation)
  const cp = Math.cos(camera.phi);
  const sp = Math.sin(camera.phi);
  const x2 = x1;
  const y2 = y1 * cp - z1 * sp;
  const z2 = y1 * sp + z1 * cp;
  return {
    sx: camera.cx + camera.scale * x2,
    sy: camera.cy - camera.scale * y2,
    depth: z2,
  };
}

// 3-stop colormap: indigo -> teal -> amber.
function heightColor(t) {
  t = Math.max(0, Math.min(1, t));
  let r, g, b;
  if (t < 0.5) {
    const u = t * 2;
    r = 40 + u * (40 - 40);
    g = 60 + u * (140 - 60);
    b = 110 + u * (130 - 110);
  } else {
    const u = (t - 0.5) * 2;
    r = 40 + u * (220 - 40);
    g = 140 + u * (170 - 140);
    b = 130 + u * (60 - 130);
  }
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

// ---------------- Surface canvas ----------------
function Surface3D({ fn, trajectory, camera, showGrid }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#fafafa";
    ctx.fillRect(0, 0, W, H);

    // --- Sample the surface ---
    const N = 48;
    const [xLo, xHi, yLo, yHi] = fn.box;
    const xs = new Float64Array(N + 1);
    const ys = new Float64Array(N + 1);
    for (let i = 0; i <= N; i++) xs[i] = xLo + ((xHi - xLo) * i) / N;
    for (let j = 0; j <= N; j++) ys[j] = yLo + ((yHi - yLo) * j) / N;

    // raw f values
    const F = new Float64Array((N + 1) * (N + 1));
    let fmin = Infinity,
      fmax = -Infinity;
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        const v = Math.min(fn.zClip, fn.f(xs[i], ys[j]));
        F[i * (N + 1) + j] = v;
        if (v < fmin) fmin = v;
        if (v > fmax) fmax = v;
      }
    }
    const fSpan = fmax - fmin || 1;
    // visual z height (in data units) — scale so that the surface has
    // a roughly cube-ish aspect ratio.
    const dataDx = xHi - xLo;
    const zVis = (v) => 1.6 * ((v - fmin) / fSpan) - 0.0;

    // Pre-project all grid vertices.
    const proj = new Array((N + 1) * (N + 1));
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        proj[i * (N + 1) + j] = project(
          [
            ((xs[i] - (xLo + xHi) / 2) / dataDx) * 3,
            ((ys[j] - (yLo + yHi) / 2) / (yHi - yLo)) * 3,
            zVis(F[i * (N + 1) + j]),
          ],
          camera
        );
      }
    }
    const projGround = new Array((N + 1) * (N + 1));
    for (let i = 0; i <= N; i++) {
      for (let j = 0; j <= N; j++) {
        projGround[i * (N + 1) + j] = project(
          [
            ((xs[i] - (xLo + xHi) / 2) / dataDx) * 3,
            ((ys[j] - (yLo + yHi) / 2) / (yHi - yLo)) * 3,
            -0.05,
          ],
          camera
        );
      }
    }

    // --- Build quads with depth for painter's algorithm ---
    const quads = [];
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const a = proj[i * (N + 1) + j];
        const b = proj[(i + 1) * (N + 1) + j];
        const c = proj[(i + 1) * (N + 1) + (j + 1)];
        const d = proj[i * (N + 1) + (j + 1)];
        const depth = (a.depth + b.depth + c.depth + d.depth) / 4;
        const fAvg =
          (F[i * (N + 1) + j] +
            F[(i + 1) * (N + 1) + j] +
            F[(i + 1) * (N + 1) + (j + 1)] +
            F[i * (N + 1) + (j + 1)]) /
          4;
        quads.push({ a, b, c, d, depth, t: (fAvg - fmin) / fSpan });
      }
    }
    quads.sort((a, b) => a.depth - b.depth);

    // --- Draw a faint xy ground grid (sparse) ---
    ctx.strokeStyle = "rgba(80,80,80,0.18)";
    ctx.lineWidth = 1;
    const groundEvery = 6;
    for (let i = 0; i <= N; i += groundEvery) {
      ctx.beginPath();
      for (let j = 0; j <= N; j += 4) {
        const p = projGround[i * (N + 1) + j];
        if (j === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.stroke();
    }
    for (let j = 0; j <= N; j += groundEvery) {
      ctx.beginPath();
      for (let i = 0; i <= N; i += 4) {
        const p = projGround[i * (N + 1) + j];
        if (i === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      }
      ctx.stroke();
    }

    // --- Draw quads ---
    for (const q of quads) {
      ctx.fillStyle = heightColor(q.t);
      ctx.beginPath();
      ctx.moveTo(q.a.sx, q.a.sy);
      ctx.lineTo(q.b.sx, q.b.sy);
      ctx.lineTo(q.c.sx, q.c.sy);
      ctx.lineTo(q.d.sx, q.d.sy);
      ctx.closePath();
      ctx.fill();
      if (showGrid) {
        ctx.strokeStyle = "rgba(0,0,0,0.15)";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }

    // --- Trajectory ---
    if (trajectory && trajectory.length > 0) {
      // Project trajectory points (and shadows on ground)
      const traj = trajectory.map((p) =>
        project(
          [
            ((p.x - (xLo + xHi) / 2) / dataDx) * 3,
            ((p.y - (yLo + yHi) / 2) / (yHi - yLo)) * 3,
            zVis(Math.min(fn.zClip, fn.f(p.x, p.y))) + 0.04,
          ],
          camera
        )
      );
      const shadow = trajectory.map((p) =>
        project(
          [
            ((p.x - (xLo + xHi) / 2) / dataDx) * 3,
            ((p.y - (yLo + yHi) / 2) / (yHi - yLo)) * 3,
            -0.04,
          ],
          camera
        )
      );

      // shadow line on ground
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      shadow.forEach((p, k) => {
        if (k === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      });
      ctx.stroke();

      // vertical "drop" lines from trajectory to its shadow at every Kth step
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.lineWidth = 1;
      for (let k = 0; k < traj.length; k += 4) {
        ctx.beginPath();
        ctx.moveTo(traj[k].sx, traj[k].sy);
        ctx.lineTo(shadow[k].sx, shadow[k].sy);
        ctx.stroke();
      }

      // trajectory polyline, on top of surface
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      traj.forEach((p, k) => {
        if (k === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      });
      ctx.stroke();
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      traj.forEach((p, k) => {
        if (k === 0) ctx.moveTo(p.sx, p.sy);
        else ctx.lineTo(p.sx, p.sy);
      });
      ctx.stroke();

      // start marker (open circle)
      const start = traj[0];
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(start.sx, start.sy, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      // current point (filled)
      const cur = traj[traj.length - 1];
      ctx.fillStyle = "#c8311c";
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cur.sx, cur.sy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }, [fn, trajectory, camera, showGrid]);

  return (
    <canvas
      ref={canvasRef}
      width={620}
      height={500}
      style={{
        background: "#fafafa",
        borderRadius: 6,
        cursor: "grab",
        touchAction: "none",
      }}
    />
  );
}

// ---------------- Top-level component ----------------
export default function Optim3DDemo() {
  const [fnKey, setFnKey] = useState("rosenbrock");
  const [methodKey, setMethodKey] = useState("gd");
  const fn = FUNCTIONS[fnKey];
  const method = METHODS[methodKey];

  const [hp, setHp] = useState(() => Object.fromEntries(method.hp.map((p) => [p.key, p.def])));
  const [trajectory, setTrajectory] = useState(() => [
    { x: fn.start[0], y: fn.start[1] },
  ]);
  const stateRef = useRef({ x: fn.start[0], y: fn.start[1], ...method.init() });
  const [running, setRunning] = useState(false);
  const [showGrid, setShowGrid] = useState(false);

  // camera state
  const [camera, setCamera] = useState({
    theta: -Math.PI / 4,
    phi: Math.PI / 3.2,
    scale: 110,
    cx: 310,
    cy: 280,
  });
  const dragRef = useRef(null);

  // Reset trajectory whenever function or method changes.
  useEffect(() => {
    stateRef.current = { x: fn.start[0], y: fn.start[1], ...method.init() };
    setTrajectory([{ x: fn.start[0], y: fn.start[1] }]);
    setHp(Object.fromEntries(method.hp.map((p) => [p.key, p.def])));
    setRunning(false);
  }, [fnKey, methodKey]);

  const step = useCallback(() => {
    const next = method.step(stateRef.current, fn, hp);
    if (
      !isFinite(next.x) ||
      !isFinite(next.y) ||
      Math.abs(next.x) > 1000 ||
      Math.abs(next.y) > 1000
    ) {
      // diverged
      setRunning(false);
      return;
    }
    stateRef.current = next;
    setTrajectory((prev) => [...prev, { x: next.x, y: next.y }]);
  }, [fn, hp, method]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(step, 60);
    return () => clearInterval(id);
  }, [running, step]);

  const reset = () => {
    stateRef.current = { x: fn.start[0], y: fn.start[1], ...method.init() };
    setTrajectory([{ x: fn.start[0], y: fn.start[1] }]);
    setRunning(false);
  };

  // mouse drag → rotate
  const onMouseDown = (e) => {
    dragRef.current = {
      x: e.clientX,
      y: e.clientY,
      theta: camera.theta,
      phi: camera.phi,
    };
  };
  const onMouseMove = (e) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    const newTheta = dragRef.current.theta - dx * 0.01;
    const newPhi = Math.max(
      0.05,
      Math.min(Math.PI / 2 - 0.05, dragRef.current.phi - dy * 0.01)
    );
    setCamera((c) => ({ ...c, theta: newTheta, phi: newPhi }));
  };
  const onMouseUp = () => {
    dragRef.current = null;
  };

  const cur = trajectory[trajectory.length - 1];
  const fcur = fn.f(cur.x, cur.y);

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 30, fontWeight: 800, marginBottom: 4 }}>
        First-Order Methods — 3D
      </h1>
      <p style={{ color: "#666", marginBottom: 22 }}>
        Watch the trajectory walk down the loss surface. Drag the canvas to
        rotate; the dashed line on the floor is the projected (xy) trajectory.
      </p>

      <div
        style={{
          display: "flex",
          gap: 24,
          alignItems: "flex-start",
          flexWrap: "wrap",
        }}
      >
        <div
          onMouseDown={onMouseDown}
          onMouseMove={onMouseMove}
          onMouseUp={onMouseUp}
          onMouseLeave={onMouseUp}
        >
          <Surface3D
            fn={fn}
            trajectory={trajectory}
            camera={camera}
            showGrid={showGrid}
          />
        </div>

        <div style={{ minWidth: 280, flex: 1 }}>
          <div style={controlGroup}>
            <label style={label}>Function</label>
            <select
              value={fnKey}
              onChange={(e) => setFnKey(e.target.value)}
              style={select}
            >
              {Object.entries(FUNCTIONS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          <div style={controlGroup}>
            <label style={label}>Method</label>
            <select
              value={methodKey}
              onChange={(e) => setMethodKey(e.target.value)}
              style={select}
            >
              {Object.entries(METHODS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          {method.hp.map((pp) => (
            <div key={pp.key} style={controlGroup}>
              <label style={label}>
                {pp.label}: <b>{(+hp[pp.key]).toFixed(pp.step < 0.01 ? 4 : pp.step < 1 ? 3 : 0)}</b>
              </label>
              <input
                type="range"
                min={pp.min}
                max={pp.max}
                step={pp.step}
                value={hp[pp.key]}
                onChange={(e) =>
                  setHp((prev) => ({ ...prev, [pp.key]: +e.target.value }))
                }
                style={{ width: "100%" }}
              />
            </div>
          ))}

          <div style={controlGroup}>
            <label style={{ ...label, display: "flex", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                checked={showGrid}
                onChange={(e) => setShowGrid(e.target.checked)}
              />
              show mesh wireframe
            </label>
          </div>

          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            <button onClick={step} style={btnPrimary}>
              <StepForward size={16} /> Step
            </button>
            <button onClick={() => setRunning((r) => !r)} style={btn}>
              {running ? <Pause size={16} /> : <Play size={16} />}
              {running ? "Pause" : "Run"}
            </button>
            <button onClick={reset} style={btn}>
              <RotateCcw size={16} /> Reset
            </button>
          </div>

          <div style={statBox}>
            <Stat label="iter" value={trajectory.length - 1} />
            <Stat label="x, y" value={`(${cur.x.toFixed(3)}, ${cur.y.toFixed(3)})`} />
            <Stat label="f(x,y)" value={fcur.toExponential(3)} />
          </div>

          <p style={{ ...pTxt, marginTop: 14, fontSize: 13 }}>
            <b>Try this.</b> Pick <i>Ill-Conditioned Bowl</i> + plain GD with a
            largish step — you'll see the classic zig-zag in the long valley.
            Switch to <i>Heavy-Ball Momentum</i> with the same step and watch
            the trajectory smooth out. On <i>Rosenbrock</i>, GD crawls along
            the banana floor while Adam glides through.
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------- Shared UI ----------------
function Stat({ label, value }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        padding: "4px 0",
      }}
    >
      <span
        style={{
          color: "#666",
          fontFamily: "monospace",
          fontSize: 13,
        }}
      >
        {label}
      </span>
      <span style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 600 }}>
        {value}
      </span>
    </div>
  );
}

const controlGroup = { marginTop: 10 };
const label = {
  display: "block",
  fontSize: 13,
  color: "#444",
  marginBottom: 4,
  fontFamily: "monospace",
};
const select = {
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
const btnPrimary = { ...btn, background: "#111", color: "#fff", border: "1px solid #111" };
const statBox = {
  marginTop: 14,
  padding: "8px 12px",
  background: "#fafafa",
  border: "1px solid #eee",
  borderRadius: 6,
};
const pTxt = { color: "#444", lineHeight: 1.55, marginBottom: 14, maxWidth: 720 };
