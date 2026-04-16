import { isDicomFile } from './uploadSelection';

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

async function readJsonOrThrow<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  const raw = await response.text();
  const trimmed = raw.trim();

  if (!contentType.includes('application/json') && trimmed.startsWith('<')) {
    throw new Error(
      `Expected JSON but received HTML from ${response.url || 'request'}. Check the API route/proxy.`
    );
  }

  try {
    return (raw ? JSON.parse(raw) : {}) as T;
  } catch {
    throw new Error(
      `Failed to parse JSON from ${response.url || 'request'} (${contentType || 'unknown content-type'}).`
    );
  }
}

function withDirectApiBase(path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${DIRECT_API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

async function fetchApi(path: string, init?: RequestInit) {
  const response = await fetch(path, init);
  const contentType = response.headers.get('content-type') || '';
  const looksLikeHtml = !contentType.includes('application/json');
  const requestedRelativeApi = !/^https?:\/\//i.test(path) && path.startsWith('/api/');

  if (requestedRelativeApi && looksLikeHtml) {
    return fetch(withDirectApiBase(path), init);
  }

  return response;
}

export async function requestAsyncDetection(primaryFile: File, folderFiles: File[]) {
  const form = new FormData();
  form.append('image', primaryFile);

  const res = await fetchApi('/api/detect_async', { method: 'POST', body: form });
  const data = await readJsonOrThrow<{ success?: boolean; job_id?: string; preview_url?: string; message?: string }>(res);

  if (!res.ok || !data.success || !data.job_id) {
    throw new Error(data.message || 'Detection request failed');
  }

  const dicom = isDicomFile(primaryFile);
  return {
    jobId: data.job_id as string,
    previewUrl: data.preview_url || (!dicom ? URL.createObjectURL(primaryFile) : undefined),
    originalFile: dicom ? primaryFile : undefined,
    originalFolderFiles: folderFiles.length > 0 ? folderFiles : undefined,
    originalFolderMode: folderFiles.length > 0,
    originalIsDicom: dicom,
    originalFileName: primaryFile.name,
  };
}

export async function requestPatientReport(primaryFile: File, userName: string, language: string) {
  const form = new FormData();
  form.append('image', primaryFile);
  form.append('user_name', userName);
  form.append('language', language);

  const res = await fetchApi('/api/v2/analyze', { method: 'POST', body: form });
  const data = await readJsonOrThrow<{ report_url?: string; error?: string } & Record<string, unknown>>(res);

  if (!data.report_url) {
    throw new Error(data.error || 'Report generation failed');
  }

  return data;
}
