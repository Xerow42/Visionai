import React, { useEffect, useRef, useState } from "react";

export default function WebcamCapture({ onCapture, onError }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [active, setActive] = useState(false);

  const start = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setActive(true);
    } catch (err) {
      onError("Could not access the webcam. Check browser permissions.");
    }
  };

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    setActive(false);
  };

  useEffect(() => () => stop(), []);

  const capture = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      const file = new File([blob], "webcam-capture.jpg", { type: "image/jpeg" });
      onCapture(file);
    }, "image/jpeg", 0.92);
  };

  return (
    <div>
      {!active ? (
        <button onClick={start}>🎥 Start webcam</button>
      ) : (
        <>
          <video ref={videoRef} className="webcam-preview" autoPlay playsInline muted />
          <div className="controls" style={{ marginTop: 10 }}>
            <button className="primary" onClick={capture}>📸 Capture & analyze</button>
            <button onClick={stop}>Stop webcam</button>
          </div>
        </>
      )}
    </div>
  );
}
