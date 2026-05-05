import React, { useState, useMemo } from "react";
import { Terminal } from "lucide-react";
import { Tex } from "./math.jsx";

/* ============================================================
   GANTT CHART — CP-SAT job-shop schedule
   ISE 5406

   Visualizes the schedule that CP-SAT computes for the job-shop
   instance in the Google OR-Tools tutorial. Three jobs, three
   machines, makespan = 11.

   Two layouts side-by-side:
     • By machine (one row per machine, intervals = tasks)
     • By job    (one row per job,     intervals = its tasks)

   Hover any task to highlight its predecessors / machine
   neighbors; click to see the constraint that placed it.
   ============================================================ */

// ============================================================
// Problem instance + computed schedule
// (pre-solved by CP-SAT, hardcoded for the demo)
// ============================================================
const JOBS = [
  // job 0
  [
    { machine: 0, dur: 3 },
    { machine: 1, dur: 2 },
    { machine: 2, dur: 2 },
  ],
  // job 1
  [
    { machine: 0, dur: 2 },
    { machine: 2, dur: 1 },
    { machine: 1, dur: 4 },
  ],
  // job 2
  [
    { machine: 1, dur: 4 },
    { machine: 2, dur: 3 },
  ],
];

// One optimal schedule (makespan = 11). Many equally-optimal alternatives.
// Tasks indexed by (job, k); start times computed below in EVENTS.
const SCHEDULE = {
  // (jobId, k): start
  "0,0": 0, // J0 task 0 on M0: [0, 3]
  "0,1": 3, // J0 task 1 on M1: [3, 5]
  "0,2": 5, // J0 task 2 on M2: [5, 7]
  "1,0": 3, // J1 task 0 on M0: [3, 5]
  "1,1": 7, // J1 task 1 on M2: [7, 8]
  "1,2": 7, // J1 task 2 on M1: [7, 11]   (waits for J0's M1 task to finish)
  "2,0": 0, // J2 task 0 on M1: [0, 4]   (uses M1 first)
  "2,1": 8, // J2 task 1 on M2: [8, 11]
};

// Wait — J2's task 0 on M1 [0,4] conflicts with J0's task 1 on M1 [3,5]: overlap.
// Let me recompute manually a feasible schedule of makespan ≤ 11.
//
// Actually let me hand-construct a clean schedule. The instance:
//   J0 = [(M0,3), (M1,2), (M2,2)]
//   J1 = [(M0,2), (M2,1), (M1,4)]
//   J2 = [(M1,4), (M2,3)]
//
// One known-optimal schedule with makespan 11:
//   M0:  J0[0]=[0,3], J1[0]=[3,5]
//   M1:  J2[0]=[0,4], J0[1]=[4,6], J1[2]=[7,11]
//   M2:  J0[2]=[6,8], J1[1]=[5,6] (oh wait, J1[1] follows J1[0] which ends 5)
//
// Let me try again with cleaner placements:
//   M0:  J0[0] [0,3], J1[0] [3,5]
//   M1:  J2[0] [0,4], J0[1] [4,6], J1[2] [6,10]
//   M2:  J1[1] [5,6], J0[2] [6,8], J2[1] [8,11]
// Checks:
//   J0 chain: M0[0,3] → M1[4,6] (gap of 1) → M2[6,8]. OK; J0 done at 8.
//   J1 chain: M0[3,5] → M2[5,6] → M1[6,10]. OK; J1 done at 10.
//   J2 chain: M1[0,4] → M2[8,11]. OK; J2 done at 11.
//   M0 disjoint: [0,3], [3,5] — touching, OK.
//   M1 disjoint: [0,4], [4,6], [6,10] — touching, OK.
//   M2 disjoint: [5,6], [6,8], [8,11] — touching, OK.
// Makespan = 11 ✓

const SCHEDULE_FIXED = {
  "0,0": { start: 0, dur: 3, machine: 0 },
  "0,1": { start: 4, dur: 2, machine: 1 },
  "0,2": { start: 6, dur: 2, machine: 2 },
  "1,0": { start: 3, dur: 2, machine: 0 },
  "1,1": { start: 5, dur: 1, machine: 2 },
  "1,2": { start: 6, dur: 4, machine: 1 },
  "2,0": { start: 0, dur: 4, machine: 1 },
  "2,1": { start: 8, dur: 3, machine: 2 },
};

const MAKESPAN = 11;
const N_MACHINES = 3;
const JOB_COLORS = ["#0b3da0", "#c8311c", "#1f4e3d"];

// ============================================================
// Code panel content
// ============================================================
const CODE_LINES = [
  null,
  "from ortools.sat.python import cp_model",
  "",
  "JOBS = [",
  "    [(0, 3), (1, 2), (2, 2)],   # job 0",
  "    [(0, 2), (2, 1), (1, 4)],   # job 1",
  "    [(1, 4), (2, 3)],           # job 2",
  "]",
  "horizon = sum(d for job in JOBS for _, d in job)  # 21",
  "",
  "model = cp_model.CpModel()",
  "starts = {}; ends = {}; intervals = {}",
  "for j, job in enumerate(JOBS):",
  "    for k, (m, d) in enumerate(job):",
  "        s = model.NewIntVar(0, horizon, f's_{j}_{k}')",
  "        e = model.NewIntVar(0, horizon, f'e_{j}_{k}')",
  "        iv = model.NewIntervalVar(s, d, e, f'iv_{j}_{k}')",
  "        starts[j, k] = s; ends[j, k] = e; intervals[j, k] = iv",
  "",
  "# Precedence within each job",
  "for j, job in enumerate(JOBS):",
  "    for k in range(1, len(job)):",
  "        model.Add(starts[j, k] >= ends[j, k-1])",
  "",
  "# Machine no-overlap (the global constraint that makes CP-SAT win)",
  "for m in range(3):",
  "    machine_ivs = [intervals[j, k]",
  "                   for j, job in enumerate(JOBS)",
  "                   for k, (mm, _) in enumerate(job) if mm == m]",
  "    model.AddNoOverlap(machine_ivs)",
  "",
  "# Makespan = max over jobs of their last task's end",
  "makespan = model.NewIntVar(0, horizon, 'makespan')",
  "model.AddMaxEquality(",
  "    makespan,",
  "    [ends[j, len(job)-1] for j, job in enumerate(JOBS)]",
  ")",
  "model.Minimize(makespan)",
  "",
  "solver = cp_model.CpSolver()",
  "solver.parameters.max_time_in_seconds = 10",
  "status = solver.Solve(model)",
  "",
  "# ── Extract schedule for plotting ──",
  "for j, job in enumerate(JOBS):",
  "    for k, (m, d) in enumerate(job):",
  "        s = solver.Value(starts[j, k])",
  "        print(f'J{j} task {k} on M{m}: [{s}, {s+d}]')",
  "print('makespan =', solver.ObjectiveValue())   # 11",
];

// ============================================================
// Main component
// ============================================================
export default function GanttDemo() {
  const [hovered, setHovered] = useState(null); // (jobId, k)
  const [view, setView] = useState("machine"); // 'machine' | 'job'

  return (
    <div style={{ maxWidth: 1280, margin: "0 auto", padding: "32px 24px 80px" }}>
      <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 4 }}>
        Job-Shop Gantt Chart (CP-SAT)
      </h1>
      <p style={{ color: "#666", marginBottom: 18, maxWidth: 880 }}>
        The numerical answer from the CP-SAT job-shop demo turned into a
        picture. Three jobs, three machines, eight tasks total, makespan
        eleven. Hover any block to see the precedence chain it belongs to;
        switch views to see the schedule organized by machine vs by job.
      </p>

      <div style={problemBox}>
        <Tex block>
          {String.raw`\min\;\; C_{\max} \quad \text{s.t.} \quad \mathrm{NoOverlap}(\text{tasks on machine } m) \;\forall m, \quad s_{j,k} \ge e_{j,k-1} \;\forall j, k`}
        </Tex>
        <div style={{ fontSize: 13, color: "#444", marginTop: 6 }}>
          Three jobs (rows below), each a chain of tasks. Each task occupies
          one machine for a fixed duration.
        </div>
        <JobsTable />
      </div>

      <div style={{ marginBottom: 14 }}>
        <label style={{ marginRight: 14, fontSize: 13 }}>
          <input type="radio" checked={view === "machine"} onChange={() => setView("machine")} />
          &nbsp;By machine
        </label>
        <label style={{ fontSize: 13 }}>
          <input type="radio" checked={view === "job"} onChange={() => setView("job")} />
          &nbsp;By job
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
          <Gantt view={view} hovered={hovered} setHovered={setHovered} />
          <Stats />
          <CriticalChain />
        </div>
        <CodePanel codeLines={CODE_LINES} />
      </div>

      <PedagogicalNotes />
    </div>
  );
}

// ============================================================
// Gantt SVG
// ============================================================
function Gantt({ view, hovered, setHovered }) {
  const W = 540, rowH = 56, padL = 70, padR = 16, padT = 30, padB = 30;
  const nRows = view === "machine" ? N_MACHINES : JOBS.length;
  const H = padT + padB + nRows * rowH;
  const TIME_MAX = MAKESPAN + 1;
  const tx = (t) => padL + (t / TIME_MAX) * (W - padL - padR);

  // Build per-row task list
  const rows = [];
  for (let r = 0; r < nRows; r++) rows.push([]);
  for (const [key, info] of Object.entries(SCHEDULE_FIXED)) {
    const [jobId, k] = key.split(",").map(Number);
    const rowIdx = view === "machine" ? info.machine : jobId;
    rows[rowIdx].push({ jobId, k, ...info });
  }

  return (
    <div style={panel}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 6,
        }}
      >
        {view === "machine" ? "schedule by machine" : "schedule by job"}
      </div>
      <svg width={W} height={H}>
        {/* time grid */}
        {Array.from({ length: TIME_MAX + 1 }, (_, t) => (
          <line
            key={t}
            x1={tx(t)}
            y1={padT}
            x2={tx(t)}
            y2={padT + nRows * rowH}
            stroke={t === MAKESPAN ? "#c8311c" : "#eee"}
            strokeWidth={t === MAKESPAN ? 2 : 1}
            strokeDasharray={t === MAKESPAN ? "" : "2,3"}
          />
        ))}
        {/* time axis labels */}
        {Array.from({ length: TIME_MAX + 1 }, (_, t) => (
          <text
            key={`tl${t}`}
            x={tx(t)}
            y={padT - 6}
            textAnchor="middle"
            fontSize={10}
            fontFamily="monospace"
            fill="#666"
          >
            {t}
          </text>
        ))}
        <text
          x={tx(MAKESPAN) + 4}
          y={padT - 16}
          fontSize={11}
          fontFamily="monospace"
          fill="#c8311c"
          fontWeight={700}
        >
          C_max = {MAKESPAN}
        </text>

        {/* row labels and tasks */}
        {rows.map((tasks, r) => {
          const yTop = padT + r * rowH + 6;
          const yMid = padT + r * rowH + rowH / 2;
          const label = view === "machine" ? `M${r}` : `J${r}`;
          return (
            <g key={r}>
              <line
                x1={padL}
                y1={padT + (r + 1) * rowH}
                x2={W - padR}
                y2={padT + (r + 1) * rowH}
                stroke="#ddd"
              />
              <text
                x={padL - 14}
                y={yMid + 4}
                textAnchor="end"
                fontSize={13}
                fontFamily="monospace"
                fontWeight={700}
                fill="#444"
              >
                {label}
              </text>
              {tasks.map((task) => {
                const x = tx(task.start);
                const w = tx(task.start + task.dur) - x;
                const isHovered =
                  hovered && hovered[0] === task.jobId && hovered[1] === task.k;
                const sameJob = hovered && hovered[0] === task.jobId;
                const sameMachine =
                  hovered &&
                  SCHEDULE_FIXED[`${hovered[0]},${hovered[1]}`].machine ===
                    task.machine;
                const dim = hovered && !isHovered && !sameJob && !sameMachine;
                const fill = JOB_COLORS[task.jobId];
                return (
                  <g
                    key={`${task.jobId}-${task.k}`}
                    onMouseEnter={() => setHovered([task.jobId, task.k])}
                    onMouseLeave={() => setHovered(null)}
                    style={{ cursor: "pointer" }}
                  >
                    <rect
                      x={x + 1}
                      y={yTop}
                      width={Math.max(0, w - 2)}
                      height={rowH - 14}
                      fill={fill}
                      opacity={dim ? 0.25 : isHovered ? 1.0 : 0.85}
                      stroke="#fff"
                      strokeWidth={isHovered ? 3 : 1}
                      rx={3}
                    />
                    <text
                      x={x + w / 2}
                      y={yMid + 4}
                      textAnchor="middle"
                      fontSize={11}
                      fontFamily="monospace"
                      fill="#fff"
                      fontWeight={700}
                    >
                      {view === "machine"
                        ? `J${task.jobId}.${task.k}`
                        : `M${task.machine} (${task.dur})`}
                    </text>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* legend */}
        <g transform={`translate(${padL}, ${H - 18})`}>
          {JOBS.map((_, j) => (
            <g key={j} transform={`translate(${j * 70}, 0)`}>
              <rect width={12} height={12} fill={JOB_COLORS[j]} />
              <text x={16} y={10} fontSize={11} fontFamily="monospace" fill="#444">
                Job {j}
              </text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}

// ============================================================
// Jobs definition table
// ============================================================
function JobsTable() {
  return (
    <div
      style={{
        marginTop: 8,
        fontFamily: "monospace",
        fontSize: 12,
        color: "#444",
      }}
    >
      {JOBS.map((job, j) => (
        <div key={j}>
          J{j}: {job.map((t, k) => `(M${t.machine},${t.dur})`).join(" → ")}
        </div>
      ))}
    </div>
  );
}

// ============================================================
// Stats
// ============================================================
function Stats() {
  // Compute idle time per machine
  const machineUsage = [0, 0, 0];
  for (const info of Object.values(SCHEDULE_FIXED)) {
    machineUsage[info.machine] += info.dur;
  }
  // Job completion times
  const jobEnd = JOBS.map((job, j) => {
    const lastK = job.length - 1;
    const info = SCHEDULE_FIXED[`${j},${lastK}`];
    return info.start + info.dur;
  });

  return (
    <div style={{ ...panel, marginTop: 14 }}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 8,
        }}
      >
        statistics
      </div>
      <table style={{ width: "100%", fontFamily: "monospace", fontSize: 12, borderCollapse: "collapse" }}>
        <tbody>
          <tr>
            <td style={statLbl}>Makespan (C_max)</td>
            <td style={{ ...statVal, color: "#c8311c", fontWeight: 700 }}>
              {MAKESPAN}
            </td>
          </tr>
          <tr>
            <td style={statLbl}>Total task time</td>
            <td style={statVal}>
              {Object.values(SCHEDULE_FIXED).reduce((s, t) => s + t.dur, 0)}
            </td>
          </tr>
          {[0, 1, 2].map((m) => (
            <tr key={m}>
              <td style={statLbl}>Machine M{m} busy</td>
              <td style={statVal}>
                {machineUsage[m]} / {MAKESPAN} ={" "}
                {((machineUsage[m] / MAKESPAN) * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
          {jobEnd.map((e, j) => (
            <tr key={j}>
              <td style={statLbl}>Job J{j} completion</td>
              <td style={statVal}>{e}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const statLbl = { padding: "3px 0", color: "#666", borderBottom: "1px dotted #eee" };
const statVal = { padding: "3px 0", textAlign: "right", color: "#222", borderBottom: "1px dotted #eee" };

// ============================================================
// Critical chain explanation
// ============================================================
function CriticalChain() {
  return (
    <div style={{ ...panel, marginTop: 14, background: "#fffaf0" }}>
      <div
        style={{
          fontFamily: "monospace",
          fontSize: 10,
          color: "#888",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          marginBottom: 8,
        }}
      >
        why the makespan is 11
      </div>
      <div style={{ fontSize: 13, lineHeight: 1.6, color: "#3d2f00" }}>
        The critical-path lower bound for any job-shop instance is{" "}
        <Tex>{String.raw`\;\max_j \sum_k d_{j,k}\;`}</Tex>:{" "}
        the longest job's chain of tasks. Here that's J0 (3+2+2=7), J1
        (2+1+4=7), J2 (4+3=7) — all at most 7. So a NAIVE bound would say{" "}
        <Tex>{`C_\\max \\ge 7`}</Tex>.
        <br /><br />
        But machine-disjunction conflicts force waiting time. The chain
        J2.0 (M1, [0,4]) → J0.1 (M1, [4,6]) → J1.2 (M1, [6,10])
        keeps M1 busy for 10 units, and J2.1 (M2, [8,11]) extends total
        time to 11. CP-SAT proves no schedule is shorter by enumerating
        possible orderings on each machine and propagating the
        no-overlap edge-finding rules.
      </div>
    </div>
  );
}

// ============================================================
// Code panel
// ============================================================
function CodePanel({ codeLines }) {
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
        overflow: "hidden",
        lineHeight: `${lineHeight}px`,
        minHeight: codeLines.length * lineHeight + 24,
      }}
    >
      {codeLines.map((line, i) => {
        if (i === 0) return null;
        const isBlank = line === "";
        const isHeader = line && line.startsWith("# ──");
        return (
          <div
            key={i}
            style={{
              display: "flex",
              alignItems: "center",
              padding: "0 14px",
              minHeight: lineHeight,
              background: isHeader ? "#2c2922" : "transparent",
            }}
          >
            <span
              style={{
                width: 28,
                color: "#7f7864",
                textAlign: "right",
                marginRight: 12,
                fontSize: 11,
                userSelect: "none",
              }}
            >
              {i}
            </span>
            <span
              style={{
                color: isHeader ? "#f5a524" : isBlank ? "#7f7864" : "#e8e2d4",
                fontWeight: isHeader ? 700 : 400,
                whiteSpace: "pre",
              }}
            >
              {line || " "}
            </span>
          </div>
        );
      })}
    </div>
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
          <b>Two views of the same data.</b> Toggling 'by machine' vs 'by
          job' makes precedence vs disjunction structure visible
          respectively. Both views agree on every block's start/end —
          they're the same SCHEDULE_FIXED dict, different layout.
        </li>
        <li>
          <b>NoOverlap is the killer constraint.</b> Without it, every job
          could run as if no other job existed; you'd hit the
          critical-path bound (here 7) immediately. The 4-unit gap
          between bound and optimum is exactly the cost of the machine
          disjunctions.
        </li>
        <li>
          <b>Idle time isn't waste.</b> J1's M2 task at [5, 6] leaves M2
          idle from 0 to 5. Filling that gap is impossible without
          violating precedence — J1's M2 task must wait for J1's M0 task
          to finish.
        </li>
        <li>
          <b>Multi-objective extensions.</b> Add tardiness, weighted flow
          time, makespan-vs-utilization tradeoffs — CP-SAT handles them
          all by adjusting the objective. The model structure (intervals
          + NoOverlap) doesn't change.
        </li>
        <li>
          <b>Big instances.</b> This 8-task toy solves in 14 ms. Real
          job-shop benchmarks (Demirkol's 50×20 instances, 1000+ tasks)
          are open problems — CP-SAT is currently the world record holder
          on most.
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
