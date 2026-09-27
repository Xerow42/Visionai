import React, { useState } from "react";
import { api } from "./api";
import UploadZone from "./components/UploadZone";
import WebcamCapture from "./components/WebcamCapture";
import ImageViewer from "./components/ImageViewer";
import DetectionResults from "./components/DetectionResults";

export default function App() {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("upload"); // "upload" | "webcam"

  const runPrediction = async (file) => {
    setError(null);
    setResult(null);
    setOriginalUrl(URL.createObjectURL(file));
    setLoading(true);
    try {
      const data = await api.predict(file);
      setResult(data);
    } catch (err) {
      setError(err?.response?.data?.detail || "Prediction failed. Please try a different image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <div className="header">
        <h1>🔍 VisionAI — Object Detection</h1>
        <p className="subtitle">
          Pretrained YOLO26n (COCO, 80 classes) — upload an image or use your webcam.
        </p>
      </div>

      <div className="controls">
        <button className={mode === "upload" ? "primary" : ""} onClick={() => setMode("upload")}>
          📁 Upload image
        </button>
        <button className={mode === "webcam" ? "primary" : ""} onClick={() => setMode("webcam")}>
          🎥 Use webcam
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {mode === "upload" ? (
        <UploadZone onFileSelected={runPrediction} onValidationError={setError} />
      ) : (
        <WebcamCapture onCapture={runPrediction} onError={setError} />
      )}

      {loading && (
        <div style={{ textAlign: "center", padding: 24 }}>
          <div className="spinner" />
          <p>Running inference…</p>
        </div>
      )}

      <ImageViewer originalUrl={originalUrl} annotatedBase64={result?.annotated_image_base64} />
      <DetectionResults result={result} />
    </div>
  );
}
