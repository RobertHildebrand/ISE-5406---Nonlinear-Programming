import React, { useState, useEffect } from "react";
import { Play, Pause, RotateCcw, ChevronLeft, ChevronRight } from "lucide-react";

/* ============================================================
   SHARED OUTPUT READER COMPONENT
   ISE 5406 — drives all four solver-tutorial output panels.

   Props:
     title         — header string
     intro         — short paragraph under the header
     columns       — [{ key, label, def }, ...] in left-to-right log order
     logs          — { [problemKey]: {
                        setupText?: string,
                        rows: [{ prefix?, color?, cells: { [colKey]: value } }],
                        summary?: string,
                        legend?: ReactNode,
                        perCol: { [colKey]: string },  // problem-specific
                        finalSummary?: string,
                      }}
     problemKey    — which problem to render
   ============================================================ */

export function OutputReader({ title, intro, columns, logs, problemKey }) {
  const log = logs[problemKey];
  const extras = log?.extras || [];
  const [step, setStep] = useState(0); // 0 = none, 1..N column, N+1..N+M extra
  const [playing, setPlaying] = useState(false);
  const totalSteps = columns.length + extras.length;

  // Reset when problem changes
  useEffect(() => {
    setStep(0);
    setPlaying(false);
  }, [problemKey]);

  // Auto-advance when playing
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setStep((s) => {
        if (s >= totalSteps) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 1800);
    return () => clearInterval(id);
  }, [playing, totalSteps]);

  if (!log) return null;
  const onColStep = step >= 1 && step <= columns.length;
  const onExtraStep = step > columns.length;
  const activeCol = onColStep ? columns[step - 1] : null;
  const activeExtra = onExtraStep ? extras[step - columns.length - 1] : null;
  const interpretation = activeCol
    ? log.perCol?.[activeCol.key] || activeCol.def
    : null;

  return (
    <div
      style={{
        marginTop: 28,
        padding: 18,
        border: "1px solid #d8d3c4",
        background: "#fdfaf1",
        borderRadius: 10,
      }}
    >
      <div style={{ fontWeight: 700, fontSize: 16, marginBottom: 6 }}>{title}</div>
      <div
        style={{
          fontSize: 13,
          color: "#555",
          lineHeight: 1.55,
          marginBottom: 14,
        }}
      >
        {intro}
      </div>

      {/* Always-visible column definitions */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 8,
          marginBottom: 14,
        }}
      >
        {columns.map((c, i) => {
          const isActive = activeCol?.key === c.key;
          return (
            <button
              key={c.key}
              onClick={() => setStep(i + 1)}
              style={{
                textAlign: "left",
                padding: "8px 10px",
                background: isActive ? "#fff4c8" : "#fff",
                border: isActive ? "2px solid #f5a524" : "1px solid #e0d8c0",
                borderRadius: 6,
                cursor: "pointer",
                transition: "all 0.15s",
                fontFamily: "inherit",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: isActive ? "#7d5a00" : "#888",
                  fontFamily: "monospace",
                  letterSpacing: "0.1em",
                }}
              >
                COL {String(i + 1).padStart(2, "0")}
              </div>
              <div
                style={{
                  fontWeight: 700,
                  fontFamily: "monospace",
                  fontSize: 12,
                  color: isActive ? "#3d2f00" : "#222",
                  marginTop: 1,
                }}
              >
                {c.label}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "#555",
                  marginTop: 4,
                  lineHeight: 1.45,
                }}
              >
                {c.def}
              </div>
            </button>
          );
        })}
      </div>

      {/* Extras grid — output sections + power-user knobs */}
      {extras.length > 0 && (
        <>
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 10,
              color: "#888",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              marginBottom: 6,
              marginTop: 4,
            }}
          >
            Other parts of the output · advanced features
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
              gap: 8,
              marginBottom: 14,
            }}
          >
            {extras.map((ex, i) => {
              const idx = columns.length + i + 1;
              const isActive = activeExtra?.key === ex.key;
              const isAdvanced = ex.kind === "feature";
              return (
                <button
                  key={ex.key}
                  onClick={() => setStep(idx)}
                  style={{
                    textAlign: "left",
                    padding: "8px 10px",
                    background: isActive
                      ? "#fff4c8"
                      : isAdvanced
                      ? "#f3edf7"
                      : "#f4f6ee",
                    border: isActive
                      ? "2px solid #f5a524"
                      : isAdvanced
                      ? "1px solid #c4b8d0"
                      : "1px solid #c0cfa0",
                    borderRadius: 6,
                    cursor: "pointer",
                    transition: "all 0.15s",
                    fontFamily: "inherit",
                  }}
                >
                  <div
                    style={{
                      fontSize: 10,
                      color: isActive
                        ? "#7d5a00"
                        : isAdvanced
                        ? "#6e4c8a"
                        : "#4a6a2a",
                      fontFamily: "monospace",
                      letterSpacing: "0.1em",
                    }}
                  >
                    {isAdvanced ? "ADVANCED" : "OUTPUT"} ·{" "}
                    {String(idx).padStart(2, "0")}
                  </div>
                  <div
                    style={{
                      fontWeight: 700,
                      fontSize: 13,
                      color: isActive ? "#3d2f00" : "#222",
                      marginTop: 1,
                    }}
                  >
                    {ex.label}
                  </div>
                  <div
                    style={{
                      fontSize: 12,
                      color: "#555",
                      marginTop: 4,
                      lineHeight: 1.45,
                    }}
                  >
                    {ex.summary}
                  </div>
                </button>
              );
            })}
          </div>
        </>
      )}

      {/* Stepper controls */}
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          marginBottom: 12,
          flexWrap: "wrap",
        }}
      >
        <button
          onClick={() => setStep((s) => Math.max(0, s - 1))}
          disabled={step === 0}
          style={btn(step === 0)}
        >
          <ChevronLeft size={14} /> Prev
        </button>
        <button
          onClick={() => setStep((s) => Math.min(totalSteps, s + 1))}
          disabled={step === totalSteps}
          style={btnPrimary(step === totalSteps)}
        >
          Next <ChevronRight size={14} />
        </button>
        <button
          onClick={() => setPlaying((p) => !p)}
          disabled={step === totalSteps}
          style={btn(step === totalSteps)}
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}{" "}
          {playing ? "Pause" : "Play all"}
        </button>
        <button
          onClick={() => {
            setStep(0);
            setPlaying(false);
          }}
          style={btn(false)}
        >
          <RotateCcw size={14} /> Reset
        </button>
        <span
          style={{
            marginLeft: 8,
            fontFamily: "monospace",
            fontSize: 12,
            color: "#888",
          }}
        >
          step {step} / {totalSteps}
        </span>
      </div>

      {/* Setup block */}
      {log.setupText && (
        <pre
          style={{
            background: "#0d0d0d",
            color: "#7f7864",
            padding: 12,
            borderRadius: "6px 6px 0 0",
            fontSize: 11,
            fontFamily: "'JetBrains Mono', Menlo, monospace",
            lineHeight: 1.55,
            margin: 0,
            whiteSpace: "pre",
            borderBottom: "1px dashed #333",
          }}
        >
          {log.setupText}
        </pre>
      )}

      {/* Log table with active column highlighting */}
      <div
        style={{
          overflowX: "auto",
          background: "#0d0d0d",
          padding: 0,
          borderRadius: log.setupText ? 0 : "6px 6px 0 0",
        }}
      >
        <table
          style={{
            borderCollapse: "collapse",
            fontFamily: "'JetBrains Mono', Menlo, monospace",
            fontSize: 11,
            color: "#dadada",
            whiteSpace: "nowrap",
            width: "100%",
          }}
        >
          <thead>
            <tr>
              {columns.map((c) => {
                const isActive = activeCol?.key === c.key;
                return (
                  <th
                    key={c.key}
                    style={{
                      padding: "6px 10px",
                      textAlign: "left",
                      background: isActive ? "#f5a524" : "transparent",
                      color: isActive ? "#1f1d1a" : "#7dd87d",
                      borderBottom: "1px solid #444",
                      fontWeight: isActive ? 700 : 600,
                    }}
                  >
                    {c.label}
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {log.rows.map((row, i) => (
              <tr key={i}>
                {columns.map((c) => {
                  const isActive = activeCol?.key === c.key;
                  const baseColor = row.color || "#dadada";
                  return (
                    <td
                      key={c.key}
                      style={{
                        padding: "3px 10px",
                        background: isActive ? "#3b3526" : "transparent",
                        color: isActive ? "#fff8e1" : baseColor,
                        fontWeight: isActive ? 700 : 400,
                        borderRight: isActive
                          ? "1px solid #f5a524"
                          : "none",
                        borderLeft: isActive
                          ? "1px solid #f5a524"
                          : "none",
                      }}
                    >
                      {row.cells[c.key] ?? ""}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary block */}
      {log.summary && (
        <pre
          style={{
            background: "#0d0d0d",
            color: "#dadada",
            padding: 12,
            borderRadius: "0 0 6px 6px",
            fontSize: 11,
            fontFamily: "'JetBrains Mono', Menlo, monospace",
            lineHeight: 1.55,
            margin: 0,
            whiteSpace: "pre",
            borderTop: "1px dashed #333",
          }}
        >
          {log.summary}
        </pre>
      )}

      {/* Active column interpretation */}
      {activeCol && (
        <div
          style={{
            marginTop: 14,
            padding: "12px 16px",
            background: "#fff4c8",
            border: "2px solid #f5a524",
            borderRadius: 8,
          }}
        >
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 11,
              color: "#7d5a00",
              marginBottom: 6,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
            }}
          >
            What &quot;{activeCol.label}&quot; is telling you on THIS run
          </div>
          <div style={{ fontSize: 14, color: "#3d2f00", lineHeight: 1.6 }}>
            {interpretation}
          </div>
        </div>
      )}

      {/* Active extra interpretation (output section OR advanced feature) */}
      {activeExtra && (
        <div
          style={{
            marginTop: 14,
            padding: "12px 16px",
            background:
              activeExtra.kind === "feature" ? "#f3edf7" : "#fff4c8",
            border:
              activeExtra.kind === "feature"
                ? "2px solid #6e4c8a"
                : "2px solid #f5a524",
            borderRadius: 8,
          }}
        >
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 11,
              color:
                activeExtra.kind === "feature" ? "#5a3a78" : "#7d5a00",
              marginBottom: 6,
              textTransform: "uppercase",
              letterSpacing: "0.12em",
            }}
          >
            {activeExtra.kind === "feature" ? "Power-user feature" : "Output section"}{" "}
            · {activeExtra.label}
          </div>
          {activeExtra.excerpt && (
            <pre
              style={{
                background: "#0d0d0d",
                color: "#dadada",
                padding: "10px 14px",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "'JetBrains Mono', Menlo, monospace",
                lineHeight: 1.55,
                margin: "0 0 10px 0",
                whiteSpace: "pre",
                overflowX: "auto",
              }}
            >
              {activeExtra.excerpt}
            </pre>
          )}
          <div
            style={{
              fontSize: 14,
              color:
                activeExtra.kind === "feature" ? "#2d1f3d" : "#3d2f00",
              lineHeight: 1.6,
            }}
          >
            {activeExtra.explain}
          </div>
        </div>
      )}

      {!activeCol && !activeExtra && (
        <div
          style={{
            marginTop: 14,
            padding: "12px 16px",
            background: "#f4efe6",
            border: "1px dashed #c8b76c",
            borderRadius: 8,
            fontSize: 13,
            color: "#666",
            textAlign: "center",
          }}
        >
          Press <b>Next</b> (or click any card above) to step through every
          column AND the other parts of the output →
        </div>
      )}

      {/* Final overall summary */}
      {log.finalSummary && step === totalSteps && (
        <div
          style={{
            marginTop: 12,
            padding: "12px 16px",
            background: "#fff",
            border: "1px solid #ddd",
            borderRadius: 8,
            fontSize: 13,
            color: "#222",
            lineHeight: 1.55,
          }}
        >
          <div
            style={{
              fontFamily: "monospace",
              fontSize: 10,
              color: "#888",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              marginBottom: 4,
            }}
          >
            overall takeaway
          </div>
          {log.finalSummary}
        </div>
      )}

      {log.legend && (
        <div
          style={{
            marginTop: 10,
            display: "flex",
            gap: 12,
            fontSize: 11,
            color: "#666",
            flexWrap: "wrap",
          }}
        >
          {log.legend}
        </div>
      )}
    </div>
  );
}

function btn(disabled) {
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "5px 10px",
    borderRadius: 5,
    border: "1px solid #ccc",
    background: disabled ? "#e9e9e9" : "#f7f7f7",
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 12,
    fontWeight: 500,
    color: disabled ? "#999" : "#222",
  };
}
function btnPrimary(disabled) {
  return {
    ...btn(disabled),
    background: disabled ? "#888" : "#111",
    color: "#fff",
    border: disabled ? "1px solid #888" : "1px solid #111",
  };
}
