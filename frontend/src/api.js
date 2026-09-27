import axios from "axios";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({ baseURL: API_BASE_URL, timeout: 30000 });

export const api = {
  predict: (file) => {
    const form = new FormData();
    form.append("file", file);
    return client.post("/predict", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },
  analyze: (file) => {
    const form = new FormData();
    form.append("file", file);
    return client.post("/analyze", form, {
      headers: { "Content-Type": "multipart/form-data" },
    }).then((r) => r.data);
  },
  health: () => client.get("/health").then((r) => r.data),
};
