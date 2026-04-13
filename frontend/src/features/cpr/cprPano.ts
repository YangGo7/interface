import * as cornerstone from '@cornerstonejs/core';
import { initCornerstone, registerNativeDicomFileWithMetadata } from '../../viewer/cornerstone/init';

export type ArchPoint = { x: number; y: number };

type GenerateCprOptions = {
  files: File[];
  archPoints: ArchPoint[];
  slabHalfWidthPx?: number;
  samplesAlongArch?: number;
};

type GenerateCprResult = {
  dataUrl: string;
  width: number;
  height: number;
  sampleCount: number;
  sliceCount: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function catmullRom(points: ArchPoint[], samplesPerSegment: number) {
  if (points.length < 2) return points;
  const extended = [points[0], ...points, points[points.length - 1]];
  const dense: ArchPoint[] = [];
  for (let i = 0; i < extended.length - 3; i += 1) {
    const p0 = extended[i];
    const p1 = extended[i + 1];
    const p2 = extended[i + 2];
    const p3 = extended[i + 3];
    for (let step = 0; step < samplesPerSegment; step += 1) {
      const t = step / samplesPerSegment;
      const t2 = t * t;
      const t3 = t2 * t;
      dense.push({
        x:
          0.5 *
          ((2 * p1.x) +
            (-p0.x + p2.x) * t +
            (2 * p0.x - 5 * p1.x + 4 * p2.x - p3.x) * t2 +
            (-p0.x + 3 * p1.x - 3 * p2.x + p3.x) * t3),
        y:
          0.5 *
          ((2 * p1.y) +
            (-p0.y + p2.y) * t +
            (2 * p0.y - 5 * p1.y + 4 * p2.y - p3.y) * t2 +
            (-p0.y + 3 * p1.y - 3 * p2.y + p3.y) * t3),
      });
    }
  }
  dense.push(points[points.length - 1]);
  return dense;
}

function resamplePolyline(points: ArchPoint[], targetCount: number) {
  if (points.length < 2) return points;
  const lengths = [0];
  for (let i = 1; i < points.length; i += 1) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    lengths.push(lengths[i - 1] + Math.hypot(dx, dy));
  }
  const total = lengths[lengths.length - 1];
  if (total <= 0) return points;
  const result: ArchPoint[] = [];
  for (let i = 0; i < targetCount; i += 1) {
    const target = (i / Math.max(1, targetCount - 1)) * total;
    let segment = 1;
    while (segment < lengths.length && lengths[segment] < target) segment += 1;
    const prevIndex = Math.max(0, segment - 1);
    const nextIndex = Math.min(points.length - 1, segment);
    const segmentLength = lengths[nextIndex] - lengths[prevIndex] || 1;
    const alpha = (target - lengths[prevIndex]) / segmentLength;
    result.push({
      x: points[prevIndex].x + (points[nextIndex].x - points[prevIndex].x) * alpha,
      y: points[prevIndex].y + (points[nextIndex].y - points[prevIndex].y) * alpha,
    });
  }
  return result;
}

function computeNormals(points: ArchPoint[]) {
  return points.map((point, index) => {
    const prev = points[Math.max(0, index - 1)];
    const next = points[Math.min(points.length - 1, index + 1)];
    const tx = next.x - prev.x;
    const ty = next.y - prev.y;
    const length = Math.hypot(tx, ty) || 1;
    return {
      x: -ty / length,
      y: tx / length,
      point,
    };
  });
}

function sampleNearest(pixelData: ArrayLike<number>, width: number, height: number, x: number, y: number) {
  const px = clamp(Math.round(x), 0, width - 1);
  const py = clamp(Math.round(y), 0, height - 1);
  return Number(pixelData[py * width + px] ?? 0);
}

function percentile(values: Float32Array, ratio: number) {
  const sorted = Array.from(values).sort((a, b) => a - b);
  const index = clamp(Math.round((sorted.length - 1) * ratio), 0, Math.max(0, sorted.length - 1));
  return sorted[index] ?? 0;
}

export async function generateDentalArchCprPano({
  files,
  archPoints,
  slabHalfWidthPx = 6,
  samplesAlongArch = 240,
}: GenerateCprOptions): Promise<GenerateCprResult> {
  if (!files.length) throw new Error('No DICOM files provided.');
  if (archPoints.length < 3) throw new Error('At least 3 arch points are required.');

  await initCornerstone();

  const smooth = catmullRom(archPoints, 20);
  const sampledArch = resamplePolyline(smooth, samplesAlongArch);
  const normals = computeNormals(sampledArch);

  const registeredImageIds = await Promise.all(files.map((file) => registerNativeDicomFileWithMetadata(file)));
  const sampledRows: Float32Array[] = [];
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (const imageId of registeredImageIds) {
    const image = await cornerstone.imageLoader.loadAndCacheImage(imageId);
    const width = image.columns || image.width;
    const height = image.rows || image.height;
    const pixelData = image.getPixelData();
    const slope = Number(image.slope ?? 1) || 1;
    const intercept = Number(image.intercept ?? 0) || 0;

    const row = new Float32Array(sampledArch.length);
    for (let i = 0; i < normals.length; i += 1) {
      const { point, x: nx, y: ny } = normals[i];
      let accum = 0;
      let count = 0;
      for (let offset = -slabHalfWidthPx; offset <= slabHalfWidthPx; offset += 1) {
        const sx = point.x + nx * offset;
        const sy = point.y + ny * offset;
        const value = sampleNearest(pixelData, width, height, sx, sy) * slope + intercept;
        accum += value;
        count += 1;
      }
      const averaged = accum / Math.max(1, count);
      row[i] = averaged;
      if (averaged < min) min = averaged;
      if (averaged > max) max = averaged;
    }
    sampledRows.push(row);
  }

  const outputWidth = sampledArch.length;
  const outputHeight = sampledRows.length;
  const canvas = document.createElement('canvas');
  canvas.width = outputWidth;
  canvas.height = outputHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Could not create pano canvas.');

  const imageData = context.createImageData(outputWidth, outputHeight);
  const range = Math.max(1, max - min);
  for (let y = 0; y < outputHeight; y += 1) {
    const row = sampledRows[y];
    for (let x = 0; x < outputWidth; x += 1) {
      const normalized = clamp(Math.round(((row[x] - min) / range) * 255), 0, 255);
      const flippedY = outputHeight - 1 - y;
      const offset = (flippedY * outputWidth + x) * 4;
      imageData.data[offset] = normalized;
      imageData.data[offset + 1] = normalized;
      imageData.data[offset + 2] = normalized;
      imageData.data[offset + 3] = 255;
    }
  }
  context.putImageData(imageData, 0, 0);

  return {
    dataUrl: canvas.toDataURL('image/png'),
    width: outputWidth,
    height: outputHeight,
    sampleCount: sampledArch.length,
    sliceCount: sampledRows.length,
  };
}

export async function buildDicomPreviewDataUrl(file: File) {
  await initCornerstone();
  const imageId = await registerNativeDicomFileWithMetadata(file);
  const image = await cornerstone.imageLoader.loadAndCacheImage(imageId);
  const width = image.columns || image.width;
  const height = image.rows || image.height;
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Could not create preview context.');
  const pixelData = image.getPixelData();
  const slope = Number(image.slope ?? 1) || 1;
  const intercept = Number(image.intercept ?? 0) || 0;
  const linear = new Float32Array(width * height);

  for (let i = 0; i < linear.length; i += 1) {
    linear[i] = Number(pixelData[i] ?? 0) * slope + intercept;
  }

  const low = percentile(linear, 0.02);
  const high = percentile(linear, 0.98);
  const range = Math.max(1, high - low);
  const imageData = context.createImageData(width, height);

  for (let i = 0; i < linear.length; i += 1) {
    const value = clamp(Math.round(((linear[i] - low) / range) * 255), 0, 255);
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }

  context.putImageData(imageData, 0, 0);

  return {
    imageId,
    rows: image.rows,
    columns: image.columns,
    dataUrl: canvas.toDataURL('image/png'),
  };
}
