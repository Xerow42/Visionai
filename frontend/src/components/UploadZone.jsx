import React, { useRef, useState } from "react";

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE_BYTES = 10 * 1024 * 1024;

export default function UploadZone({ onFileSelected, onValidationError }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const validateAndEmit = (file) => {
    if (!file) return;
    if (!ACCEPTED_TYPES.includes(file.type)) {
      onValidationError(`Unsupported file type "${file.type}". Use JPEG, PNG, or WEBP.`);
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      onValidationError(`File is ${(file.size / 1_000_000).toFixed(1)} MB — the limit is 10 MB.`);
      return;
    }
    onFileSelected(file);
  };

  return (
    <div
      className={`dropzone ${dragOver ? "dragover" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        validateAndEmit(e.dataTransfer.files?.[0]);
      }}
    >
      <p>📁 Drag & drop an image here, or click to browse</p>
      <p style={{ fontSize: 12, color: "#666c7e" }}>JPEG, PNG, or WEBP — up to 10 MB</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(e) => validateAndEmit(e.target.files?.[0])}
      />
    </div>
  );
}
