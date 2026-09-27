import React from "react";

export default function ImageViewer({ originalUrl, annotatedBase64 }) {
  if (!originalUrl) return null;
  return (
    <div className="viewer-grid">
      <div className="viewer-card">
        <h3>Original</h3>
        <img src={originalUrl} alt="Original upload" />
      </div>
      <div className="viewer-card">
        <h3>Detections</h3>
        {annotatedBase64 ? (
          <img src={`data:image/jpeg;base64,${annotatedBase64}`} alt="Annotated result" />
        ) : (
          <div className="empty-state">Run a prediction to see annotated results here.</div>
        )}
      </div>
    </div>
  );
}
