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

async function readJson<T>(response: Response): Promise<T> {
  const data = await response.json();
  if (!response.ok) {
    throw new Error((data as { error?: string }).error || 'Request failed');
  }
  return data as T;
}

export async function createWebReportSession(language: string) {
  const response = await fetch('/api/web_report/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language }),
  });
  return readJson<{ success: boolean; session_id: string }>(response);
}

export async function uploadWebReportSession(sessionId: string, file: File) {
  const form = new FormData();
  form.append('image', file);
  const response = await fetch(`/api/web_report/session/${sessionId}/upload`, {
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
  const response = await fetch('/api/web_report/from-chart', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson<{ success: boolean; session_id: string; report_url: string; html_url: string }>(response);
}

export async function fetchWebReportSession(sessionId: string) {
  const response = await fetch(`/api/web_report/session/${sessionId}`);
  return readJson<WebReportSessionResponse>(response);
}

export async function patchWebReportOverrides(
  sessionId: string,
  payload: {
    tooth_overrides?: Record<string, Record<string, unknown>>;
    report_note?: string;
    reset_tooth_ids?: string[];
  }
) {
  const response = await fetch(`/api/web_report/session/${sessionId}/overrides`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return readJson<{ success: boolean; doctor_overrides: any; effective_result: any }>(response);
}

export async function regenerateWebReport(sessionId: string) {
  const response = await fetch(`/api/web_report/session/${sessionId}/report/regenerate`, {
    method: 'POST',
  });
  return readJson<{ success: boolean; version: number; report_url: string; page_url: string }>(response);
}

export async function finalizeWebReport(sessionId: string) {
  const response = await fetch(`/api/web_report/session/${sessionId}/report/finalize`, {
    method: 'POST',
  });
  return readJson<{ success: boolean; version: number; report_url: string; pdf_url: string }>(response);
}
