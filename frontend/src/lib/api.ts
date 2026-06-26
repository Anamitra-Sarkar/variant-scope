const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://variantscope-api.onrender.com";

export async function predictVariant(
  sequence: string,
  modelType: string = "esm2",
  position?: number
) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence, model_type: modelType, position }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  if (!res.ok) throw new Error("API unreachable");
  return res.json();
}
