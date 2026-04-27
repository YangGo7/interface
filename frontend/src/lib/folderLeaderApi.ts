import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import type { DicomOverlayMetadata } from '../viewer/cornerstone/dicomMetadata';

const resolveDirectApiBase = () => {
  const configured = ((import.meta as any)?.env?.VITE_API_BASE_URL as string | undefined)?.trim();
  if (configured) return configured;
  if (typeof window === 'undefined') return 'http://localhost:5000';

  const { protocol, hostname, port, origin } = window.location;
  const isLocalDevHost = hostname === 'localhost' || hostname === '127.0.0.1';
  if (isLocalDevHost && (port === '3000' || port === '5173')) {
    return `${protocol}//${hostname}:5000`;
  }
  return origin;
};

const DIRECT_API_BASE = resolveDirectApiBase();

export type ServerFolderFileEntry = {
  name: string;
  relativePath: string;
  downloadUrl: string;
  size?: number;
};

export type ServerFolderSeries = {
  id: string;
  studyId: string;
  label: string;
  description: string;
  modality: string;
  orientation: 'Axial' | 'Sagittal' | 'Coronal' | 'Oblique' | 'Unknown';
  sliceCount: number;
  spacingLabel: string;
  compression: string;
  isCompressed: boolean;
  volumeEligible: boolean;
  files: ServerFolderFileEntry[];
};

export type ServerFolderStudy = {
  id: string;
  label: string;
  description: string;
  patientId: string;
  patientName: string;
  patientAge: string;
  patientSex: string;
  studyDate: string;
  modalities: string[];
  totalFiles: number;
  totalSeries: number;
  previewUrl?: string | null;
  series: ServerFolderSeries[];
};

export type ServerFolderIndexResponse = {
  success?: boolean;
  root_path?: string;
  root_exists?: boolean;
  studies?: ServerFolderStudy[];
  images?: ServerFolderImage[];
  message?: string;
};

export type ServerFolderImage = {
  name: string;
  relativePath: string;
  downloadUrl: string;
  previewUrl?: string | null;
  folderLabel: string;
  width: number;
  height: number;
  format: string;
  size?: number;
  isDicom?: boolean;
  linkedStudyId?: string | null;
  patientId?: string;
  patientName?: string;
  patientAge?: string;
  patientSex?: string;
  studyDate?: string;
  modalities?: string[];
  description?: string;
  hasSidecarJson?: boolean;
  dicomInfo?: DicomOverlayMetadata | null;
};

export type ServerFolderRootPathResponse = {
  success?: boolean;
  root_path?: string;
  root_exists?: boolean;
  message?: string;
};

export function resolveServerAssetUrl(path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${DIRECT_API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

async function fetchApi(path: string, init?: RequestInit) {
  const response = await fetch(path, init);
  const contentType = response.headers.get('content-type') || '';
  const looksLikeHtml = !contentType.includes('application/json');
  const requestedRelativeApi = !/^https?:\/\//i.test(path) && path.startsWith('/api/');

  if (requestedRelativeApi && looksLikeHtml) {
    return fetch(resolveServerAssetUrl(path), init);
  }

  return response;
}

async function readJsonOrThrow<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  const raw = await response.text();
  const trimmed = raw.trim();

  if (!contentType.includes('application/json') && trimmed.startsWith('<')) {
    throw new Error(`Expected JSON but received HTML from ${response.url || 'request'}.`);
  }

  try {
    return (raw ? JSON.parse(raw) : {}) as T;
  } catch {
    throw new Error(
      `Failed to parse JSON from ${response.url || 'request'} (${contentType || 'unknown content-type'}).`
    );
  }
}

export async function fetchServerFolderIndex() {
  const response = await fetchApi(`/api/dicom-server/studies?t=${Date.now()}`, {
    cache: 'no-store',
  });
  const data = await readJsonOrThrow<ServerFolderIndexResponse>(response);

  if (!response.ok || !data.success) {
    throw new Error(data.message || 'Failed to load DICOM server studies.');
  }

  return data;
}

export async function fetchServerFolderRootPath() {
  const response = await fetchApi('/api/dicom-server/root-path');
  const data = await readJsonOrThrow<ServerFolderRootPathResponse>(response);

  if (!response.ok || !data.success) {
    throw new Error(data.message || 'Failed to load DICOM server root path.');
  }

  return data;
}

export async function updateServerFolderRootPath(rootPath: string) {
  const response = await fetchApi('/api/dicom-server/root-path', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ root_path: rootPath }),
  });
  const data = await readJsonOrThrow<ServerFolderRootPathResponse>(response);

  if (!response.ok || !data.success) {
    throw new Error(data.message || 'Failed to update DICOM server root path.');
  }

  return data;
}

export async function pickServerFolderRootPath() {
  const response = await fetchApi('/api/dicom-server/root-path/pick', {
    method: 'POST',
  });
  const data = await readJsonOrThrow<ServerFolderRootPathResponse>(response);

  if (!response.ok || !data.success) {
    throw new Error(data.message || 'Failed to pick DICOM server root path.');
  }

  return data;
}

async function fetchServerDicomFile(entry: ServerFolderFileEntry) {
  const response = await fetchApi(entry.downloadUrl);
  if (!response.ok) {
    throw new Error(`Failed to load ${entry.name}.`);
  }

  const blob = await response.blob();
  return new File([blob], entry.name, {
    type: blob.type || 'application/dicom',
    lastModified: Date.now(),
  });
}

export async function materializeServerStudy(study: ServerFolderStudy): Promise<FolderStudy> {
  const series = await Promise.all(
    study.series.map(async (seriesEntry) => {
      const files = await Promise.all(seriesEntry.files.map((fileEntry) => fetchServerDicomFile(fileEntry)));
      return {
        id: seriesEntry.id,
        studyId: seriesEntry.studyId,
        label: seriesEntry.label,
        description: seriesEntry.description,
        modality: seriesEntry.modality,
        files,
        orientation: seriesEntry.orientation,
        sliceCount: seriesEntry.sliceCount,
        spacingLabel: seriesEntry.spacingLabel,
        compression: seriesEntry.compression,
        isCompressed: seriesEntry.isCompressed,
        volumeEligible: seriesEntry.volumeEligible,
      };
    })
  );

  return {
    id: study.id,
    label: study.label,
    description: study.description,
    patientId: study.patientId,
    studyDate: study.studyDate,
    modalities: study.modalities,
    totalFiles: study.totalFiles,
    totalSeries: study.totalSeries,
    series,
  };
}
