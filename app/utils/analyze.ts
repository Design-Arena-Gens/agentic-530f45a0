export type AnalysisResult = {
  modality: "CT" | "MRI";
  confidence: number; // 0..1
  features: Record<string, number>;
};

function ensureCanvasContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D not supported");
  return ctx;
}

function toGrayscale(data: Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const a = data[i + 3];
    const y = Math.round(0.2126 * r + 0.7152 * g + 0.0722 * b);
    out[i] = out[i + 1] = out[i + 2] = y;
    out[i + 3] = a;
  }
  return out;
}

function normalizeIntensity(data: Uint8ClampedArray): Uint8ClampedArray {
  let min = 255, max = 0;
  for (let i = 0; i < data.length; i += 4) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = Math.max(1, max - min);
  const out = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) {
    const y = data[i];
    const n = Math.round(((y - min) / range) * 255);
    out[i] = out[i + 1] = out[i + 2] = n;
    out[i + 3] = data[i + 3];
  }
  return out;
}

function computeHistogram(data: Uint8ClampedArray): number[] {
  const hist = new Array(256).fill(0);
  let count = 0;
  for (let i = 0; i < data.length; i += 4) {
    hist[data[i]]++;
    count++;
  }
  for (let i = 0; i < 256; i++) hist[i] /= count;
  return hist;
}

function computeSobelEdges(gray: Uint8ClampedArray, width: number, height: number) {
  const gxKernel = [
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
  ];
  const gyKernel = [
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
  ];
  const magnitude = new Float32Array(width * height);
  const clamp = (x: number, min: number, max: number) => Math.max(min, Math.min(max, x));
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      let idx = 0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = clamp(x + kx, 0, width - 1);
          const py = clamp(y + ky, 0, height - 1);
          const i = (py * width + px) * 4;
          const v = gray[i];
          gx += gxKernel[idx] * v;
          gy += gyKernel[idx] * v;
          idx++;
        }
      }
      const m = Math.hypot(gx, gy);
      magnitude[y * width + x] = m;
    }
  }
  return magnitude;
}

function mean(values: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i];
  return s / values.length;
}

function std(values: ArrayLike<number>, mu?: number): number {
  const m = mu ?? mean(values);
  let s2 = 0;
  for (let i = 0; i < values.length; i++) {
    const d = values[i] - m;
    s2 += d * d;
  }
  return Math.sqrt(s2 / Math.max(1, values.length - 1));
}

function percentiles(values: Float32Array, ps: number[]): number[] {
  const arr = Array.from(values).sort((a, b) => a - b);
  return ps.map(p => {
    const idx = Math.min(arr.length - 1, Math.max(0, Math.round((p / 100) * (arr.length - 1))));
    return arr[idx];
  });
}

function computeTextureEnergy(gray: Uint8ClampedArray, width: number, height: number): number {
  let energy = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const i = (y * width + x) * 4;
      const dx = gray[i] - gray[i + 4];
      const dy = gray[i] - gray[i + width * 4];
      energy += Math.abs(dx) + Math.abs(dy);
    }
  }
  return energy / (width * height);
}

function logistic(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export async function analyzeCtVsMri(img: HTMLImageElement, outCanvas: HTMLCanvasElement): Promise<AnalysisResult> {
  const size = 256; // normalize to 256x256 for deterministic features
  outCanvas.width = size;
  outCanvas.height = size;
  const ctx = ensureCanvasContext(outCanvas);

  // draw and get data
  ctx.clearRect(0, 0, size, size);
  ctx.drawImage(img, 0, 0, size, size);
  const imageData = ctx.getImageData(0, 0, size, size);

  // grayscale + normalize
  const gray = toGrayscale(imageData.data);
  const norm = normalizeIntensity(gray);
  const normImage = new ImageData(norm, size, size);
  ctx.putImageData(normImage, 0, 0);

  // histogram features
  const hist = computeHistogram(norm);
  const histMean = hist.reduce((s, p, i) => s + p * i, 0);
  const histVar = hist.reduce((s, p, i) => s + p * (i - histMean) * (i - histMean), 0);
  const histEntropy = -hist.reduce((s, p) => (p > 0 ? s + p * Math.log2(p) : s), 0);

  // edges
  const edges = computeSobelEdges(norm, size, size);
  const edgeMean = mean(edges);
  const edgeStd = std(edges, edgeMean);
  const [edgeP50, edgeP90, edgeP99] = percentiles(edges, [50, 90, 99]);

  // texture energy
  const textureE = computeTextureEnergy(norm, size, size);

  // simple linear classifier (hand-tuned heuristics for demo)
  // Assumptions: CT often has higher contrast edges to bone/air; MRI has softer gradients and higher entropy after normalization
  const features: Record<string, number> = {
    histMean,
    histVar,
    histEntropy,
    edgeMean,
    edgeStd,
    edgeP50,
    edgeP90,
    edgeP99,
    textureE
  };

  const weights: Record<keyof typeof features, number> = {
    histMean: -0.01,
    histVar: -0.002,
    histEntropy: 0.8,
    edgeMean: -0.015,
    edgeStd: -0.02,
    edgeP50: -0.001,
    edgeP90: -0.0008,
    edgeP99: -0.0005,
    textureE: -0.004
  };
  const bias = -1.0; // shifts toward CT by default

  let score = bias;
  for (const k of Object.keys(features) as Array<keyof typeof features>) {
    score += features[k] * weights[k];
  }
  const probMRI = logistic(score);
  const modality = probMRI > 0.5 ? "MRI" : "CT";
  const confidence = modality === "MRI" ? probMRI : 1 - probMRI;

  return { modality, confidence, features };
}

export async function drawGradCAMLikeHeatmap(srcCanvas: HTMLCanvasElement, heatmapCanvas: HTMLCanvasElement) {
  // Fake CAM: use edge magnitude and local variance to indicate "attention"
  const width = srcCanvas.width;
  const height = srcCanvas.height;
  heatmapCanvas.width = width;
  heatmapCanvas.height = height;
  const srcCtx = ensureCanvasContext(srcCanvas);
  const dstCtx = ensureCanvasContext(heatmapCanvas);
  const { data } = srcCtx.getImageData(0, 0, width, height);

  // compute local variance on grayscale
  const gray = new Uint8ClampedArray(data.length);
  for (let i = 0; i < data.length; i += 4) gray[i] = gray[i + 1] = gray[i + 2] = data[i];

  const window = 5;
  const radius = Math.floor(window / 2);
  const variance = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0, sum2 = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = Math.min(width - 1, Math.max(0, x + dx));
          const yy = Math.min(height - 1, Math.max(0, y + dy));
          const v = gray[(yy * width + xx) * 4];
          sum += v;
          sum2 += v * v;
          count++;
        }
      }
      const mu = sum / count;
      const varpx = sum2 / count - mu * mu;
      variance[y * width + x] = varpx;
    }
  }

  // normalize to 0..1
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < variance.length; i++) {
    const v = variance[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const norm = new Float32Array(variance.length);
  const range = Math.max(1e-6, max - min);
  for (let i = 0; i < variance.length; i++) norm[i] = (variance[i] - min) / range;

  // apply colormap (inferno-ish)
  const out = dstCtx.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const v = norm[i];
      const r = Math.min(255, Math.max(0, Math.round(255 * Math.pow(v, 0.35))));
      const g = Math.min(255, Math.max(0, Math.round(255 * Math.pow(v, 1.2) * (1 - v))));
      const b = Math.min(255, Math.max(0, Math.round(255 * (1 - v))));
      const j = i * 4;
      out.data[j] = r;
      out.data[j + 1] = g;
      out.data[j + 2] = b;
      out.data[j + 3] = 180;
    }
  }
  dstCtx.putImageData(out, 0, 0);
}
