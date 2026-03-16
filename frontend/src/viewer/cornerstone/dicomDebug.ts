import dicomParser from 'dicom-parser';

export type LocalDicomInspection = {
  fileName: string;
  transferSyntaxUid: string;
  rows: number;
  columns: number;
  samplesPerPixel: number;
  photometricInterpretation: string;
  bitsAllocated: number;
  bitsStored: number;
  pixelRepresentation: number;
  pixelBytes: number;
  expectedBytes: number;
  numberOfFrames: number;
  windowCenter: number | null;
  windowWidth: number | null;
  canRenderPreview: boolean;
  previewDataUrl: string | null;
  issues: string[];
};

const parseNumber = (value: string | undefined): number | null => {
  if (!value) {
    return null;
  }

  const parsed = Number(value.split('\\')[0]);
  return Number.isFinite(parsed) ? parsed : null;
};

const getMinMax = (data: ArrayLike<number>) => {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < data.length; index += 1) {
    const value = Number(data[index]);
    if (value < min) {
      min = value;
    }
    if (value > max) {
      max = value;
    }
  }

  return {
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
  };
};

export async function inspectLocalDicomFile(file: File): Promise<LocalDicomInspection> {
  const byteArray = new Uint8Array(await file.arrayBuffer());
  const dataSet = dicomParser.parseDicom(byteArray);
  const transferSyntaxUid = dataSet.string('x00020010') || '1.2.840.10008.1.2';
  const rows = dataSet.uint16('x00280010') || 0;
  const columns = dataSet.uint16('x00280011') || 0;
  const samplesPerPixel = dataSet.uint16('x00280002') || 1;
  const photometricInterpretation = dataSet.string('x00280004') || 'UNKNOWN';
  const bitsAllocated = dataSet.uint16('x00280100') || 0;
  const bitsStored = dataSet.uint16('x00280101') || bitsAllocated;
  const pixelRepresentation = dataSet.uint16('x00280103') || 0;
  const pixelElement = dataSet.elements.x7fe00010;
  const pixelBytes = pixelElement?.length || 0;
  const singleFrameBytes = rows * columns * samplesPerPixel * Math.max(1, bitsAllocated / 8);
  const expectedBytes = singleFrameBytes;

  // Detect multi-frame: either from tag (0028,0008) or computed from pixel data size
  const numberOfFramesTag = dataSet.string('x00280008');
  let numberOfFrames = numberOfFramesTag ? parseInt(numberOfFramesTag, 10) : 1;
  if ((!numberOfFrames || numberOfFrames <= 1) && singleFrameBytes > 0 && pixelBytes > singleFrameBytes) {
    numberOfFrames = Math.round(pixelBytes / singleFrameBytes);
  }
  if (isNaN(numberOfFrames) || numberOfFrames < 1) numberOfFrames = 1;

  const issues: string[] = [];
  let previewDataUrl: string | null = null;
  let canRenderPreview = false;

  if (!pixelElement) {
    issues.push('PixelData tag is missing.');
  }
  if (!rows || !columns) {
    issues.push('Rows/Columns tag is missing.');
  }
  if (samplesPerPixel !== 1) {
    issues.push(`Only monochrome preview is supported here. samplesPerPixel=${samplesPerPixel}`);
  }
  if (![8, 16].includes(bitsAllocated)) {
    issues.push(`Unsupported bitsAllocated=${bitsAllocated}`);
  }
  if (!['1.2.840.10008.1.2', '1.2.840.10008.1.2.1'].includes(transferSyntaxUid)) {
    issues.push(`Compressed or unsupported transfer syntax: ${transferSyntaxUid}`);
  }
  if (numberOfFrames > 1) {
    issues.push(`Multi-frame DICOM: ${numberOfFrames} frames detected`);
  }
  if (expectedBytes && pixelBytes && pixelBytes < expectedBytes && numberOfFrames <= 1) {
    issues.push(`PixelData length looks short. expected=${expectedBytes}, actual=${pixelBytes}`);
  }

  if (issues.length === 0 && pixelElement) {
    const pixelBytesView = dataSet.byteArray.slice(pixelElement.dataOffset, pixelElement.dataOffset + pixelElement.length);
    let scalarData: Uint8Array | Uint16Array | Int16Array;

    if (bitsAllocated === 8) {
      scalarData = pixelBytesView;
    } else {
      const pixelBuffer = pixelBytesView.buffer.slice(pixelBytesView.byteOffset, pixelBytesView.byteOffset + pixelBytesView.byteLength);
      scalarData = pixelRepresentation === 1 ? new Int16Array(pixelBuffer) : new Uint16Array(pixelBuffer);
    }

    const { min, max } = getMinMax(scalarData);
    const range = Math.max(1, max - min);
    const canvas = document.createElement('canvas');
    canvas.width = columns;
    canvas.height = rows;
    const context = canvas.getContext('2d');

    if (context) {
      const imageData = context.createImageData(columns, rows);
      for (let index = 0; index < scalarData.length; index += 1) {
        const normalized = Math.round(((Number(scalarData[index]) - min) / range) * 255);
        const gray = photometricInterpretation === 'MONOCHROME1' ? 255 - normalized : normalized;
        const rgbaOffset = index * 4;
        imageData.data[rgbaOffset] = gray;
        imageData.data[rgbaOffset + 1] = gray;
        imageData.data[rgbaOffset + 2] = gray;
        imageData.data[rgbaOffset + 3] = 255;
      }
      context.putImageData(imageData, 0, 0);
      previewDataUrl = canvas.toDataURL('image/png');
      canRenderPreview = true;
    }
  }

  return {
    fileName: file.name,
    transferSyntaxUid,
    rows,
    columns,
    samplesPerPixel,
    photometricInterpretation,
    bitsAllocated,
    bitsStored,
    pixelRepresentation,
    pixelBytes,
    expectedBytes,
    numberOfFrames,
    windowCenter: parseNumber(dataSet.string('x00281050')),
    windowWidth: parseNumber(dataSet.string('x00281051')),
    canRenderPreview,
    previewDataUrl,
    issues,
  };
}
