export type WebReportSessionResponse = {
  success: boolean;
  session?: {
    id: string;
    status: string;
    error?: string | null;
    language: string;
    created_at: string;
    updated_at: string;
    finalized_at?: string | null;
    is_finalized: boolean;
    assets: Record<string, string | null>;
    ai_result?: any;
    doctor_overrides?: any;
    effective_result?: any;
    report?: {
      version?: number;
      status?: string;
      html_path?: string | null;
      pdf_path?: string | null;
      page_url?: string | null;
      html_url?: string | null;
      pdf_url?: string | null;
    } | null;
  };
  error?: string;
};

const WEB_REPORT_DIRECT_API_BASE = 'http://localhost:5000';

async function readJson<T>(response: Response): Promise<T> {
  const contentType = response.headers.get('content-type') || '';
  const raw = await response.text();
  const trimmed = raw.trim();

  if (!contentType.includes('application/json') && trimmed.startsWith('<')) {
    throw new Error(
      `Expected JSON but received HTML from ${response.url || 'request'}. Check the API route/proxy.`
    );
  }

  let data: unknown;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    throw new Error(
      `Failed to parse JSON from ${response.url || 'request'} (${contentType || 'unknown content-type'}).`
    );
  }

  if (!response.ok) {
    throw new Error((data as { error?: string }).error || 'Request failed');
  }
  return data as T;
}

function withDirectApiBase(path: string) {
  if (/^https?:\/\//i.test(path)) return path;
  return `${WEB_REPORT_DIRECT_API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

async function fetchWebReportApi(input: string, init?: RequestInit) {
  const response = await fetch(input, init);
  const contentType = response.headers.get('content-type') || '';
  const looksLikeHtml = !contentType.includes('application/json');
  const requestedRelativeApi = !/^https?:\/\//i.test(input) && input.startsWith('/api/web_report/');

  if (requestedRelativeApi && response.ok && looksLikeHtml && response.url.includes('localhost:5173')) {
    return fetch(withDirectApiBase(input), init);
  }

  return response;
}

export async function createWebReportSession(language: string) {
  const response = await fetchWebReportApi('/api/web_report/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language }),
  });
  return readJson<{ success: boolean; session_id: string }>(response);
}

export async function uploadWebReportSession(sessionId: string, file: File) {
  const form = new FormData();
  form.append('image', file);
  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}/upload`, {
    method: 'POST',
    body: form,
  });
  return readJson<{ success: boolean; session_id: string; status: string }>(response);
}

export async function createWebReportFromChart(payload: {
  result: any;
  source_url?: string | null;
  overlay_url?: string | null;
  language?: string;
}) {
  const response = await fetchWebReportApi('/api/web_report/from-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson<{ success: boolean; session_id: string; report_url: string; html_url: string }>(response);
}

export async function fetchWebReportSession(sessionId: string) {
  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}`);
  return readJson<WebReportSessionResponse>(response);
}

export async function patchWebReportOverrides(
  sessionId: string,
  payload: {
    tooth_overrides?: Record<string, Record<string, unknown>>;
    report_note?: string;
    attached_captures?: Array<Record<string, unknown>>;
    reset_tooth_ids?: string[];
  }
) {
  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}/overrides`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson<{ success: boolean; doctor_overrides: any; effective_result: any }>(response);
}

export async function regenerateWebReport(sessionId: string) {
  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}/report/regenerate`, {
    method: 'POST',
  });
  return readJson<{ success: boolean; version: number; report_url: string; page_url: string }>(response);
}

export async function finalizeWebReport(sessionId: string) {
  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}/report/finalize`, {
    method: 'POST',
  });
  return readJson<{ success: boolean; version: number; report_url: string; pdf_url: string }>(response);
}

export async function transcribeWebReportDictation(
  sessionId: string,
  audioBlob: Blob,
  mimeType: string
) {
  const form = new FormData();
  const extension = mimeType.includes('ogg') ? 'ogg' : mimeType.includes('mp4') ? 'm4a' : 'webm';
  form.append('audio', audioBlob, `dictation.${extension}`);

  const response = await fetchWebReportApi(`/api/web_report/session/${sessionId}/dictation`, {
    method: 'POST',
    body: form,
  });
  return readJson<{
    success: boolean;
    transcript: string;
    soap_note: {
      subjective: string;
      objective: string;
      assessment: string;
      plan: string;
    };
    tooth_findings: Array<{ tooth: string; keywords: string[] }>;
    keywords: string[];
    report_note_text: string;
  }>(response);
}
