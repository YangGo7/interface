import dicomParser from 'dicom-parser';

export type FolderSeries = {
  id: string;
  studyId: string;
  label: string;
  description: string;
  modality: string;
  files: File[];
  orientation: 'Axial' | 'Sagittal' | 'Coronal' | 'Oblique' | 'Unknown';
  sliceCount: number;
  spacingLabel: string;
  compression: string;
  isCompressed: boolean;
  volumeEligible: boolean;
};

export type FolderStudy = {
  id: string;
  label: string;
  description: string;
  patientId: string;
  studyDate: string;
  modalities: string[];
  totalFiles: number;
  totalSeries: number;
  series: FolderSeries[];
};

type SeriesFileEntry = {
  file: File;
  instanceNumber: number;
  zPosition: number;
};

type MutableSeries = {
  label: string;
  description: string;
  modality: string;
  files: SeriesFileEntry[];
  orientation: FolderSeries['orientation'];
  pixelSpacing: [number, number] | null;
  sliceSpacing: number | null;
  transferSyntaxUid: string;
};

const parseNum = (value?: string, fallback = 0) => {
  const parsed = Number((value || '').split('\\')[0]);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const parseNumList = (value?: string) =>
  (value || '').split('\\').map(Number).filter((entry) => Number.isFinite(entry));

const formatStudyDate = (raw?: string) => {
  if (!raw || raw.length !== 8) return raw || '';
  return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
};

const classifyOrientation = (iop?: string): FolderSeries['orientation'] => {
  const values = parseNumList(iop);
  if (values.length < 6) return 'Unknown';
  const row = values.slice(0, 3);
  const col = values.slice(3, 6);
  const normal = [
    row[1] * col[2] - row[2] * col[1],
    row[2] * col[0] - row[0] * col[2],
    row[0] * col[1] - row[1] * col[0],
  ].map((value) => Math.abs(value));
  const [x, y, z] = normal;
  const dominant = Math.max(x, y, z);
  if (!Number.isFinite(dominant) || dominant < 0.7) return 'Oblique';
  if (dominant === z) return 'Axial';
  if (dominant === y) return 'Coronal';
  if (dominant === x) return 'Sagittal';
  return 'Unknown';
};

const describeCompression = (uid?: string) => {
  switch (uid) {
    case '1.2.840.10008.1.2':
      return 'Implicit VR Little Endian';
    case '1.2.840.10008.1.2.1':
      return 'Explicit VR Little Endian';
    case '1.2.840.10008.1.2.2':
      return 'Explicit VR Big Endian';
    case '1.2.840.10008.1.2.4.50':
      return 'JPEG Baseline';
    case '1.2.840.10008.1.2.4.57':
    case '1.2.840.10008.1.2.4.70':
      return 'JPEG Lossless';
    case '1.2.840.10008.1.2.4.90':
      return 'JPEG2000 Lossless';
    case '1.2.840.10008.1.2.4.91':
      return 'JPEG2000';
    case '1.2.840.10008.1.2.5':
      return 'RLE Lossless';
    default:
      return uid ? `UID ${uid}` : 'Unknown';
  }
};

const formatSpacingLabel = (pixelSpacing: [number, number] | null, sliceSpacing: number | null) => {
  const inPlane = pixelSpacing
    ? `${pixelSpacing[0].toFixed(2)} x ${pixelSpacing[1].toFixed(2)} mm`
    : 'n/a';
  const throughPlane = Number.isFinite(sliceSpacing) && sliceSpacing !== null
    ? `${sliceSpacing.toFixed(2)} mm`
    : 'n/a';
  return `${inPlane} / ${throughPlane}`;
};

export async function buildDicomFolderStudies(files: File[]): Promise<FolderStudy[]> {
  const studies = new Map<
    string,
    {
      label: string;
      description: string;
      patientId: string;
      studyDate: string;
      series: Map<string, MutableSeries>;
    }
  >();

  for (const file of files) {
    if (!/\.(dcm|dicom)$/i.test(file.name)) continue;
    const dataSet = dicomParser.parseDicom(new Uint8Array(await file.arrayBuffer()), { untilTag: 'x7fe00010' });
    const studyId = dataSet.string('x0020000d') || 'study-unknown';
    const studyDescription = dataSet.string('x00081030') || 'Unnamed Study';
    const studyDateRaw = dataSet.string('x00080020') || '';
    const studyDate = formatStudyDate(studyDateRaw);
    const studyLabel = studyDate || studyDescription;
    const patientId = dataSet.string('x00100020') || 'Unknown';
    const seriesId = dataSet.string('x0020000e') || `series-${file.name}`;
    const seriesDescription = dataSet.string('x0008103e') || 'Unnamed Series';
    const modality = dataSet.string('x00080060') || 'OT';
    const instanceNumber = parseNum(dataSet.string('x00200013'), Number.MAX_SAFE_INTEGER);
    const ipp = parseNumList(dataSet.string('x00200032'));
    const zPosition = ipp[2] ?? Number.MAX_SAFE_INTEGER;
    const orientation = classifyOrientation(dataSet.string('x00200037'));
    const spacingValues = parseNumList(dataSet.string('x00280030'));
    const pixelSpacing = spacingValues.length >= 2
      ? [spacingValues[0], spacingValues[1]] as [number, number]
      : null;
    const sliceSpacing = parseNum(dataSet.string('x00180088') || dataSet.string('x00180050'), NaN);
    const transferSyntaxUid = dataSet.string('x00020010') || '';

    if (!studies.has(studyId)) {
      studies.set(studyId, {
        label: studyLabel,
        description: studyDescription,
        patientId,
        studyDate,
        series: new Map(),
      });
    }
    const study = studies.get(studyId)!;
    if (!study.series.has(seriesId)) {
      study.series.set(seriesId, {
        label: seriesDescription,
        description: seriesDescription,
        modality,
        files: [],
        orientation,
        pixelSpacing,
        sliceSpacing: Number.isFinite(sliceSpacing) ? sliceSpacing : null,
        transferSyntaxUid,
      });
    }
    study.series.get(seriesId)!.files.push({ file, instanceNumber, zPosition });
  }

  return Array.from(studies.entries()).map(([studyId, study]) => {
    const series = Array.from(study.series.entries()).map(([seriesId, item]) => {
      const sortedFiles = [...item.files]
        .sort((a, b) => (a.instanceNumber - b.instanceNumber) || (a.zPosition - b.zPosition) || a.file.name.localeCompare(b.file.name))
        .map((entry) => entry.file);
      const compression = describeCompression(item.transferSyntaxUid);
      const isCompressed = !['1.2.840.10008.1.2', '1.2.840.10008.1.2.1', '1.2.840.10008.1.2.2'].includes(item.transferSyntaxUid);
      const sliceCount = sortedFiles.length;
      const volumeEligible = item.modality.toUpperCase().includes('CT') && sliceCount > 8;
      return {
        id: seriesId,
        studyId,
        label: item.label,
        description: item.description,
        modality: item.modality,
        files: sortedFiles,
        orientation: item.orientation,
        sliceCount,
        spacingLabel: formatSpacingLabel(item.pixelSpacing, item.sliceSpacing),
        compression,
        isCompressed,
        volumeEligible,
      };
    });

    return {
      id: studyId,
      label: study.label,
      description: study.description,
      patientId: study.patientId,
      studyDate: study.studyDate,
      modalities: Array.from(new Set(series.map((item) => item.modality))).sort(),
      totalFiles: series.reduce((sum, item) => sum + item.files.length, 0),
      totalSeries: series.length,
      series,
    };
  });
}
