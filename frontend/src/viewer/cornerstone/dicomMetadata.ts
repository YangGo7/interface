import dicomParser from 'dicom-parser';

export type DicomOverlayMetadata = {
  fileName: string;
  patientName: string;
  patientId: string;
  patientSex: string;
  patientBirthDate: string;
  studyDate: string;
  studyTime: string;
  studyDescription: string;
  studyInstanceUID: string;
  studyId: string;
  accessionNumber: string;
  seriesDescription: string;
  seriesInstanceUID: string;
  seriesNumber: number;
  instanceNumber: number;
  modality: string;
  institutionName: string;
  manufacturer: string;
  bodyPartExamined: string;
  sopClassUID: string;
  transferSyntaxUid: string;
  imageType: string[];
  rows: number;
  columns: number;
  samplesPerPixel: number;
  photometricInterpretation: string;
  bitsAllocated: number;
  bitsStored: number;
  highBit: number;
  pixelRepresentation: number;
  windowCenter: number;
  windowWidth: number;
  numberOfFrames: number;
  pixelSpacing: [number, number];
  rowPixelSpacing: number;
  columnPixelSpacing: number;
  imageOrientationPatient: number[];
  imagePositionPatient: number[];
  sliceThickness: number;
  spacingBetweenSlices: number;
  frameOfReferenceUID: string;
};

export const parseDicomNumber = (value: string | undefined, fallback: number) => {
  if (!value) {
    return fallback;
  }

  const parsed = Number(value.split('\\')[0]);
  return Number.isFinite(parsed) ? parsed : fallback;
};

export const parseDicomNumberList = (value: string | undefined, fallback: number[]) => {
  if (!value) {
    return fallback;
  }

  const parsed = value
    .split('\\')
    .map((entry) => Number(entry))
    .filter((entry) => Number.isFinite(entry));

  return parsed.length > 0 ? parsed : fallback;
};

const parseDicomStringList = (value: string | undefined) =>
  (value || '')
    .split('\\')
    .map((entry) => entry.trim())
    .filter(Boolean);

const normalizePersonName = (value: string | undefined) =>
  (value || '')
    .replace(/\^+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

export const parseDicomMetadataFromDataSet = (
  dataSet: dicomParser.DataSet,
  fileName = ''
): DicomOverlayMetadata => {
  const rows = dataSet.uint16('x00280010') || 0;
  const columns = dataSet.uint16('x00280011') || 0;
  const samplesPerPixel = dataSet.uint16('x00280002') || 1;
  const photometricInterpretation = dataSet.string('x00280004') || 'MONOCHROME2';
  const bitsAllocated = dataSet.uint16('x00280100') || 16;
  const bitsStored = dataSet.uint16('x00280101') || bitsAllocated;
  const highBit = dataSet.uint16('x00280102') || bitsStored - 1;
  const pixelRepresentation = dataSet.uint16('x00280103') || 0;
  const modality = dataSet.string('x00080060') || 'OT';
  const studyInstanceUID = dataSet.string('x0020000d') || '';
  const seriesInstanceUID = dataSet.string('x0020000e') || '';
  const frameOfReferenceUID = dataSet.string('x00200052') || '';
  const pixelSpacing = parseDicomNumberList(dataSet.string('x00280030'), [1, 1]);
  const rowPixelSpacing = pixelSpacing[0] || 1;
  const columnPixelSpacing = pixelSpacing[1] || 1;
  const imageOrientationPatient = parseDicomNumberList(dataSet.string('x00200037'), [1, 0, 0, 0, 1, 0]);
  const imagePositionPatient = parseDicomNumberList(dataSet.string('x00200032'), [0, 0, 0]);
  const sliceThickness = parseDicomNumber(dataSet.string('x00180050'), 1);
  const spacingBetweenSlices = parseDicomNumber(dataSet.string('x00180088'), sliceThickness);
  const derivedWindowWidth = Math.max(1, (dataSet.uint16('x00280107') || 4095) - (dataSet.uint16('x00280106') || 0));
  const derivedWindowCenter = (dataSet.uint16('x00280106') || 0) + derivedWindowWidth / 2;
  const windowCenter = parseDicomNumber(dataSet.string('x00281050'), derivedWindowCenter);
  const windowWidth = parseDicomNumber(dataSet.string('x00281051'), derivedWindowWidth);
  const numberOfFrames = Math.max(1, parseInt(dataSet.string('x00280008') || '1', 10) || 1);

  return {
    fileName,
    patientName: normalizePersonName(dataSet.string('x00100010')),
    patientId: dataSet.string('x00100020') || '',
    patientSex: dataSet.string('x00100040') || '',
    patientBirthDate: dataSet.string('x00100030') || '',
    studyDate: dataSet.string('x00080020') || '',
    studyTime: dataSet.string('x00080030') || '',
    studyDescription: dataSet.string('x00081030') || '',
    studyInstanceUID,
    studyId: dataSet.string('x00200010') || '',
    accessionNumber: dataSet.string('x00080050') || '',
    seriesDescription: dataSet.string('x0008103e') || '',
    seriesInstanceUID,
    seriesNumber: parseDicomNumber(dataSet.string('x00200011'), 0),
    instanceNumber: parseDicomNumber(dataSet.string('x00200013'), 0),
    modality,
    institutionName: dataSet.string('x00080080') || '',
    manufacturer: dataSet.string('x00080070') || '',
    bodyPartExamined: dataSet.string('x00180015') || '',
    sopClassUID: dataSet.string('x00080016') || '',
    transferSyntaxUid: dataSet.string('x00020010') || '',
    imageType: parseDicomStringList(dataSet.string('x00080008')),
    rows,
    columns,
    samplesPerPixel,
    photometricInterpretation,
    bitsAllocated,
    bitsStored,
    highBit,
    pixelRepresentation,
    windowCenter,
    windowWidth,
    numberOfFrames,
    pixelSpacing: [rowPixelSpacing, columnPixelSpacing],
    rowPixelSpacing,
    columnPixelSpacing,
    imageOrientationPatient,
    imagePositionPatient,
    sliceThickness,
    spacingBetweenSlices,
    frameOfReferenceUID,
  };
};
