"use client";
import { useCallback, useMemo, useRef, useState } from "react";
import { analyzeCtVsMri, drawGradCAMLikeHeatmap } from "./utils/analyze";

export default function Page() {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<{
    modality: "CT" | "MRI";
    confidence: number;
    features: Record<string, number>;
  } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const heatmapRef = useRef<HTMLCanvasElement | null>(null);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    setImageUrl(url);
    setResult(null);
    setError(null);
  }, []);

  const canAnalyze = useMemo(() => !!imageUrl && !isProcessing, [imageUrl, isProcessing]);

  const onAnalyze = useCallback(async () => {
    if (!imageRef.current || !canvasRef.current || !heatmapRef.current) return;
    setIsProcessing(true);
    setError(null);
    try {
      const res = await analyzeCtVsMri(imageRef.current, canvasRef.current);
      setResult(res);
      await drawGradCAMLikeHeatmap(canvasRef.current, heatmapRef.current);
    } catch (err: any) {
      setError(err?.message ?? "Analysis failed");
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return (
    <main style={{
      minHeight: "100vh",
      display: "grid",
      gridTemplateRows: "auto 1fr",
      background: "#0b1220",
      color: "#e6edf3",
      fontFamily: "ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, Apple Color Emoji, Segoe UI Emoji"
    }}>
      <header style={{
        padding: "16px 24px",
        borderBottom: "1px solid #1f2a44",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between"
      }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>CT vs MRI Analyzer</h1>
        <a href="https://github.com" target="_blank" rel="noreferrer" style={{ color: "#8ab4ff" }}>Source</a>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "420px 1fr", gap: 24, padding: 24 }}>
        <section style={{ background: "#0f172a", border: "1px solid #1f2a44", borderRadius: 12, padding: 16 }}>
          <h2 style={{ marginTop: 0, fontSize: 16 }}>Upload</h2>
          <input type="file" accept="image/*" onChange={onFileChange} />
          <p style={{ fontSize: 12, color: "#98a2b3" }}>Runs fully in your browser. Not medical advice.</p>

          <button onClick={onAnalyze} disabled={!canAnalyze} style={{
            marginTop: 12,
            padding: "10px 14px",
            borderRadius: 8,
            background: canAnalyze ? "#2563eb" : "#334155",
            color: "white",
            border: 0,
            cursor: canAnalyze ? "pointer" : "not-allowed"
          }}>
            {isProcessing ? "Analyzing..." : "Analyze Image"}
          </button>

          {error && <div style={{ marginTop: 12, color: "#fda29b" }}>{error}</div>}

          {result && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 14 }}>Predicted: <strong>{result.modality}</strong></div>
              <div style={{ fontSize: 12, color: "#98a2b3" }}>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
              <details style={{ marginTop: 8 }}>
                <summary style={{ cursor: "pointer" }}>Feature scores</summary>
                <ul style={{ marginTop: 6 }}>
                  {Object.entries(result.features).map(([k, v]) => (
                    <li key={k}>
                      <code>{k}</code>: {v.toFixed(3)}
                    </li>
                  ))}
                </ul>
              </details>
            </div>
          )}
        </section>

        <section style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ background: "#0f172a", border: "1px solid #1f2a44", borderRadius: 12, padding: 12 }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Original</h2>
            <div style={{ position: "relative", width: "100%", aspectRatio: "1 / 1", background: "#111827", borderRadius: 8, overflow: "hidden" }}>
              {imageUrl ? (
                <img ref={imageRef} alt="uploaded" src={imageUrl} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              ) : (
                <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", color: "#64748b", fontSize: 12 }}>No image</div>
              )}
            </div>
          </div>

          <div style={{ background: "#0f172a", border: "1px solid #1f2a44", borderRadius: 12, padding: 12 }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Preprocessed</h2>
            <div style={{ position: "relative", width: "100%", aspectRatio: "1 / 1", background: "#111827", borderRadius: 8, overflow: "hidden" }}>
              <canvas ref={canvasRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>

          <div style={{ background: "#0f172a", border: "1px solid #1f2a44", borderRadius: 12, padding: 12, gridColumn: "1 / span 2" }}>
            <h2 style={{ marginTop: 0, fontSize: 16 }}>Heatmap</h2>
            <div style={{ position: "relative", width: "100%", aspectRatio: "2 / 1", background: "#111827", borderRadius: 8, overflow: "hidden" }}>
              <canvas ref={heatmapRef} style={{ width: "100%", height: "100%" }} />
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
