import { isDicomFile } from './uploadSelection';

export async function requestAsyncDetection(primaryFile: File, folderFiles: File[]) {
  const form = new FormData();
  form.append('image', primaryFile);

  const res = await fetch('/api/detect_async', { method: 'POST', body: form });
  const data = await res.json();

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
  const data = await res.json();

  if (!data.report_url) {
    throw new Error(data.error || 'Report generation failed');
  }

  return data;
}
