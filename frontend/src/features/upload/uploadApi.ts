import { isDicomFile } from './uploadSelection';

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

export async function requestAsyncDetection(primaryFile: File, folderFiles: File[]) {
  const form = new FormData();
  form.append('image', primaryFile);

  const res = await fetch('/api/detect_async', { method: 'POST', body: form });
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

  const res = await fetch('/api/v2/analyze', { method: 'POST', body: form });
  const data = await readJsonOrThrow<{ report_url?: string; error?: string } & Record<string, unknown>>(res);

  if (!data.report_url) {
    throw new Error(data.error || 'Report generation failed');
  }

  return data;
}
