import React from "react";

export default function DetectionResults({ result }) {
  if (!result) return null;
  const { detections, inference_time_ms, engine, is_real_model, warning } = result;

  return (
    <>
      {warning && <div className="warning-banner">⚠️ {warning}</div>}

      <div className="stats-row">
        <div className="stat-card">
          <div className="label">Inference time</div>
          <div className="value">{inference_time_ms} ms</div>
        </div>
        <div className="stat-card">
          <div className="label">Objects found</div>
          <div className="value">{detections.length}</div>
        </div>
        <div className="stat-card">
          <div className="label">Engine</div>
          <div className="value" style={{ fontSize: 14 }}>
            {engine} {is_real_model ? "✅" : "⚠️ fallback"}
          </div>
        </div>
      </div>

      <div className="results-card">
        <h3>Detections</h3>
        {detections.length === 0 ? (
          <div className="empty-state">No objects detected above the confidence threshold.</div>
        ) : (
          detections
            .slice()
            .sort((a, b) => b.confidence - a.confidence)
            .map((d, i) => (
              <div className="detection-row" key={i}>
                <span className="class-name">{d.class_name}</span>
                <div className="confidence-bar-track">
                  <div className="confidence-bar-fill" style={{ width: `${d.confidence * 100}%` }} />
                </div>
                <span className="confidence-value">{(d.confidence * 100).toFixed(1)}%</span>
              </div>
            ))
        )}
      </div>
    </>
  );
}
