import dicomParser from 'dicom-parser';

export type FolderSeries = {
  id: string;
  studyId: string;
  label: string;
  description: string;
  modality: string;
  files: File[];
};

export type FolderStudy = {
  id: string;
  label: string;
  description: string;
  series: FolderSeries[];
};

type SeriesFileEntry = {
  file: File;
  instanceNumber: number;
  zPosition: number;
};

const parseNum = (value?: string, fallback = 0) => {
  const parsed = Number((value || '').split('\\')[0]);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const parseNumList = (value?: string) =>
  (value || '').split('\\').map(Number).filter((entry) => Number.isFinite(entry));

export async function buildDicomFolderStudies(files: File[]): Promise<FolderStudy[]> {
  const studies = new Map<string, { label: string; description: string; series: Map<string, { label: string; description: string; modality: string; files: SeriesFileEntry[] }> }>();

  for (const file of files) {
    if (!/\.(dcm|dicom)$/i.test(file.name)) continue;
    const dataSet = dicomParser.parseDicom(new Uint8Array(await file.arrayBuffer()), { untilTag: 'x7fe00010' });
    const studyId = dataSet.string('x0020000d') || 'study-unknown';
    const studyDescription = dataSet.string('x00081030') || 'Unnamed Study';
    const studyLabel = dataSet.string('x00080020') || studyDescription;
    const seriesId = dataSet.string('x0020000e') || `series-${file.name}`;
    const seriesDescription = dataSet.string('x0008103e') || 'Unnamed Series';
    const modality = dataSet.string('x00080060') || 'OT';
    const instanceNumber = parseNum(dataSet.string('x00200013'), Number.MAX_SAFE_INTEGER);
    const ipp = parseNumList(dataSet.string('x00200032'));
    const zPosition = ipp[2] ?? Number.MAX_SAFE_INTEGER;

    if (!studies.has(studyId)) {
      studies.set(studyId, { label: studyLabel, description: studyDescription, series: new Map() });
    }
    const study = studies.get(studyId)!;
    if (!study.series.has(seriesId)) {
      study.series.set(seriesId, { label: seriesDescription, description: seriesDescription, modality, files: [] });
    }
    study.series.get(seriesId)!.files.push({ file, instanceNumber, zPosition });
  }

  return Array.from(studies.entries()).map(([studyId, study]) => ({
    id: studyId,
    label: study.label,
    description: study.description,
    series: Array.from(study.series.entries()).map(([seriesId, series]) => ({
      id: seriesId,
      studyId,
      label: series.label,
      description: series.description,
      modality: series.modality,
      files: [...series.files]
        .sort((a, b) => (a.instanceNumber - b.instanceNumber) || (a.zPosition - b.zPosition) || a.file.name.localeCompare(b.file.name))
        .map((entry) => entry.file),
    })),
  }));
}
