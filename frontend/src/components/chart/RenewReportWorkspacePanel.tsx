import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchWebReportSession,
  listWebReportVersions,
  patchWebReportOverrides,
  regenerateWebReport,
  type WebReportSessionResponse,
  type WebReportVersionSummary,
} from '../../lib/webReportApi';

type ReportWorkspaceCaptureItem = {
  id: string;
  dataUrl: string;
  createdAt: number | string;
  label?: string;
  size?: string;
};

type RenewReportWorkspacePanelProps = {
  visible: boolean;
  left: string;
  top: string;
  width: string;
  height: string;
  sessionId: string | null;
  isLoading: boolean;
  error: string | null;
  availableCaptures?: ReportWorkspaceCaptureItem[];
  selectedCaptureIds?: string[];
  onToggleCaptureSelection?: (captureId: string) => void;
  previewRefreshToken?: number;
  onClose: () => void;
};

const panelBorder = '#4C4C4C';
const shellBg = '#111111';
const sectionBg = '#2D2D2D';
const cardBg = '#333333';
const cardInnerBg = '#414141';

const assetPath = (relativePath: string) => encodeURI(`/imgs/${relativePath}`);
const headerMarkerIcon = assetPath('7 7.png');

function normalizeToothValue(value: any) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') {
    return String(value.tooth_label || value.tooth || value.label || '').trim();
  }
  return String(value).trim();
}

function uniqueSortedTeeth(values: Array<string | number>) {
  return [...new Set(values.map((value) => normalizeToothValue(value)).filter(Boolean))].sort((a, b) => Number(a) - Number(b));
}

function teethFromMap(map?: Record<string, any>) {
  return uniqueSortedTeeth(Object.keys(map || {}));
}

function teethFromList(list?: Array<string | number>) {
  return uniqueSortedTeeth(Array.isArray(list) ? (list as any[]) : []);
}

function formatSummaryLine(label: string, teeth: string[], maxItems = 4) {
  if (!teeth.length) return '';
  const visibleTeeth = teeth.slice(0, maxItems).map((tooth) => `#${tooth}`);
  const hiddenCount = teeth.length - visibleTeeth.length;
  const suffix = hiddenCount > 0 ? ` +${hiddenCount}` : '';
  return `${label}: ${visibleTeeth.join(', ')}${suffix}`;
}

function buildSummaryRows(result: any) {
  const rows = [
    formatSummaryLine('Caries', teethFromMap(result?.caries_by_tooth_best || result?.caries_by_tooth)),
    formatSummaryLine('Periapical', teethFromMap(result?.periapical_by_tooth_best || result?.periapical_by_tooth)),
    formatSummaryLine('Missing', teethFromList(result?.missing_teeth || result?.teeth_missing)),
    formatSummaryLine('Implant', teethFromMap(result?.implant_by_tooth_best || result?.implant_by_tooth)),
    formatSummaryLine('Crown', teethFromMap(result?.crown_by_tooth_best || result?.crown_by_tooth)),
    formatSummaryLine('Filling', teethFromMap(result?.filling_by_tooth_best || result?.filling_by_tooth)),
    formatSummaryLine(
      'Bone loss',
      Object.entries(result?.pbl || {})
        .filter(([, value]) => Number(value || 0) > 0)
        .sort((a, b) => Number(b[1] || 0) - Number(a[1] || 0))
        .map(([tooth]) => String(tooth))
    ),
  ].filter(Boolean);

  return rows.length ? rows : ['No major findings detected.'];
}

function formatTimestamp(value?: string | number | null) {
  if (value === null || value === undefined || value === '') return '-';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
}

function buildReportHtmlUrl(sessionId: string | null, version: number | null, nonce?: number, draftPreview = false) {
  if (!sessionId) return '';
  const params = new URLSearchParams();
  if (version) params.set('version', String(version));
  if (typeof nonce === 'number') params.set('r', String(nonce));
  const query = params.toString();
  const basePath = `/api/web_report/session/${sessionId}/report${draftPreview ? '/preview' : ''}`;
  return `${basePath}${query ? `?${query}` : ''}`;
}

function buildReportPdfUrl(sessionId: string | null, version: number | null) {
  if (!sessionId || !version) return '';
  return `/api/web_report/session/${sessionId}/report/pdf?version=${version}`;
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ color: '#9E9E9E', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#FFFFFF', fontSize: 12, textAlign: 'right' }}>{value}</span>
    </div>
  );
}

export function RenewReportWorkspacePanel({
  visible,
  left,
  top,
  width,
  height,
  sessionId,
  isLoading,
  error,
  availableCaptures = [],
  selectedCaptureIds = [],
  onToggleCaptureSelection,
  previewRefreshToken = 0,
  onClose,
}: RenewReportWorkspacePanelProps) {
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [actionState, setActionState] = useState<'idle' | 'regenerating'>('idle');
  const [reportNoteDraft, setReportNoteDraft] = useState('');
  const [reportNoteFocused, setReportNoteFocused] = useState(false);
  const [reportNoteSaveState, setReportNoteSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [summaryExpanded, setSummaryExpanded] = useState(false);
  const [versions, setVersions] = useState<WebReportVersionSummary[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const latestReportVersionRef = useRef<number | null>(null);

  useEffect(() => {
    if (!visible || !sessionId) {
      setSession(null);
      setSessionError(null);
      setVersions([]);
      return;
    }

    let cancelled = false;

    const load = async () => {
      try {
        const [sessionData, versionsData] = await Promise.all([
          fetchWebReportSession(sessionId),
          listWebReportVersions(sessionId),
        ]);
        if (cancelled) return;
        setSession(sessionData.session || null);
        setSessionError(sessionData.error || null);
        setVersions(versionsData.versions || []);
      } catch (nextError: any) {
        if (cancelled) return;
        setSessionError(nextError?.message || 'Failed to load report session');
      }
    };

    void load();
    const timer = window.setInterval(load, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId, visible]);

  useEffect(() => {
    setSummaryExpanded(false);
    setSelectedVersion(null);
    setReportNoteDraft('');
    setReportNoteSaveState('idle');
  }, [sessionId, visible]);

  useEffect(() => {
    if (reportNoteFocused) return;
    setReportNoteDraft((session?.doctor_overrides?.report_note as string | undefined) || '');
  }, [reportNoteFocused, session?.doctor_overrides?.report_note]);

  useEffect(() => {
    if (selectedVersion !== null) return;
    const fallbackVersion = session?.report?.version ?? session?.current_report_version ?? versions[0]?.version ?? null;
    if (fallbackVersion !== null) {
      setSelectedVersion(fallbackVersion);
    }
  }, [selectedVersion, session?.report?.version, session?.current_report_version, versions]);

  useEffect(() => {
    const latestVersion = session?.report?.version ?? session?.current_report_version ?? null;
    if (latestVersion !== null && (selectedVersion === null || selectedVersion === latestReportVersionRef.current)) {
      setSelectedVersion(latestVersion);
    }
    latestReportVersionRef.current = latestVersion;
  }, [selectedVersion, session?.current_report_version, session?.report?.version]);

  const effectiveResult = session?.effective_result || session?.ai_result || {};
  const summaryRows = useMemo(() => buildSummaryRows(effectiveResult), [effectiveResult]);
  const visibleSummaryRows = summaryExpanded ? summaryRows : summaryRows.slice(0, 4);
  const currentVersion = selectedVersion ?? session?.report?.version ?? session?.current_report_version ?? null;
  const selectedVersionInfo = versions.find((item) => item.version === currentVersion) || null;
  const liveDraftVersion = session?.current_report_version ?? session?.report?.version ?? null;
  const useDraftPreview = Boolean(!session?.is_finalized && currentVersion && liveDraftVersion && currentVersion === liveDraftVersion);
  const previewUrl = buildReportHtmlUrl(sessionId, currentVersion, previewNonce, useDraftPreview);
  const savedPrintUrl = buildReportHtmlUrl(sessionId, currentVersion, undefined, false);
  const pdfUrl = buildReportPdfUrl(sessionId, currentVersion);
  const combinedError = error || sessionError;
  const reportNote = (session?.doctor_overrides?.report_note as string | undefined)?.trim() || '';

  const saveReportNote = async () => {
    if (!sessionId || !session) return;
    const currentServerNote = (session?.doctor_overrides?.report_note as string | undefined) || '';
    if (reportNoteDraft === currentServerNote) return;

    setReportNoteSaveState('saving');
    setSessionError(null);
    try {
      const response = await patchWebReportOverrides(sessionId, {
        report_note: reportNoteDraft,
      });
      setSession((prev) =>
        prev
          ? {
              ...prev,
              doctor_overrides: response.doctor_overrides,
              effective_result: response.effective_result,
              updated_at: new Date().toISOString(),
            }
          : prev
      );
      setReportNoteSaveState('saved');
      setPreviewNonce((current) => current + 1);
    } catch (nextError: any) {
      setSessionError(nextError?.message || 'Failed to save report note');
      setReportNoteSaveState('error');
    }
  };
  const attachedCaptures = useMemo(
    () =>
      ((session?.doctor_overrides?.attached_captures as Array<Record<string, unknown>> | undefined) || [])
        .map((capture, index) => ({
          id: String(capture.id || `attached-${index}`),
          dataUrl: String(capture.reportDataUrl || capture.dataUrl || ''),
          createdAt: String(capture.createdAt || ''),
          label: String(capture.label || 'Capture'),
          size: capture.size ? String(capture.size) : undefined,
        }))
        .filter((capture) => Boolean(capture.dataUrl)),
    [session?.doctor_overrides]
  );
  const galleryCaptures = availableCaptures.length
    ? availableCaptures.map((capture, index) => ({
        id: capture.id || `capture-${index}`,
        dataUrl: capture.dataUrl,
        createdAt: capture.createdAt,
        label: capture.label || `Capture ${index + 1}`,
        size: capture.size,
      }))
    : attachedCaptures;

  const handleRegeneratePreview = async () => {
    if (!sessionId || actionState !== 'idle') return;

    setActionState('regenerating');
    setSessionError(null);
    try {
      const response = await regenerateWebReport(sessionId);
      const [sessionData, versionsData] = await Promise.all([
        fetchWebReportSession(sessionId),
        listWebReportVersions(sessionId),
      ]);
      setSession(sessionData.session || null);
      setSessionError(sessionData.error || null);
      setVersions(versionsData.versions || []);
      setSelectedVersion(response.version || sessionData.session?.report?.version || null);
      setPreviewNonce((current) => current + 1);
    } catch (nextError: any) {
      setSessionError(nextError?.message || 'Failed to regenerate report');
    } finally {
      setActionState('idle');
    }
  };

  const sanitizePrintHtml = (html: string) => {
    const cleanedTitle = 'Dental Analysis Report';
    const printStyle = `
      <style>
        @page { size: auto; margin: 10mm; }
        html, body {
          background: #1E1E1E !important;
        }
        body {
          max-width: none !important;
          margin: 0 !important;
          padding: 18px !important;
        }
        * {
          -webkit-print-color-adjust: exact !important;
          print-color-adjust: exact !important;
        }
        .page-shell,
        .renew-section,
        .renew-section-body,
        .odontogram,
        .odonto-shell,
        .odonto-main,
        .odonto-canvas {
          overflow: hidden !important;
        }
        .odonto-shell {
          grid-template-columns: minmax(0, 1fr) !important;
        }
        .odonto-legend-panel {
          display: none !important;
        }
        .odonto-main-top {
          padding: 10px 12px 0 !important;
        }
        .odonto-canvas {
          padding: 8px 10px 14px !important;
        }
        .odonto-body-frame {
          grid-template-columns: 16px minmax(0, 1fr) 16px !important;
          gap: 2px !important;
        }
        .odonto-row,
        .odonto-quadrant {
          gap: 1px !important;
        }
        .odonto-separator {
          width: 4px !important;
        }
        .odonto-tooth {
          width: 24px !important;
          height: 68px !important;
        }
        .odonto-tooth img {
          width: 22px !important;
          height: 56px !important;
        }
        .odonto-label {
          font-size: 0.62rem !important;
        }
        .odonto-side {
          width: 16px !important;
          font-size: 0.72rem !important;
        }
        .renew-section {
          break-inside: auto !important;
          page-break-inside: auto !important;
        }
        .renew-section-header {
          break-after: avoid-page !important;
          page-break-after: avoid !important;
        }
      </style>
    `;
    try {
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');

      if (doc.querySelector('title')) {
        doc.title = cleanedTitle;
      }

      doc.querySelectorAll('label').forEach((label) => {
        if (label.querySelector('input[data-tooth-toggle]') || /include/i.test(label.textContent || '')) {
          label.remove();
        }
      });

      doc.querySelectorAll<HTMLTextAreaElement>('[data-tooth-note], [data-capture-note]').forEach((textarea) => {
        const value = (textarea.value || textarea.textContent || '').trim();
        const wrapper = textarea.parentElement;
        if (!wrapper) return;

        wrapper.style.breakInside = 'avoid';
        wrapper.style.pageBreakInside = 'avoid';

        if (!value) {
          wrapper.remove();
          return;
        }

        const title = textarea.hasAttribute('data-tooth-note') ? 'NOTE' : 'MEMO';
        const titleBlock = doc.createElement('div');
        titleBlock.style.fontSize = '0.75rem';
        titleBlock.style.fontWeight = '800';
        titleBlock.style.letterSpacing = '0.1em';
        titleBlock.style.textTransform = 'uppercase';
        titleBlock.style.color = '#94A3B8';
        titleBlock.style.marginBottom = '8px';
        titleBlock.textContent = title;

        const noteBlock = doc.createElement('div');
        noteBlock.style.marginTop = '10px';
        noteBlock.style.padding = '12px';
        noteBlock.style.border = '1px solid #4C4C4C';
        noteBlock.style.background = '#353535';
        noteBlock.style.color = '#E2E8F0';
        noteBlock.style.whiteSpace = 'pre-wrap';
        noteBlock.textContent = value;
        textarea.replaceWith(noteBlock);
        if (!wrapper.contains(titleBlock)) {
          wrapper.insertBefore(titleBlock, noteBlock);
        }
      });

      doc.querySelectorAll('.renew-section').forEach((section) => {
        const headerText = section.querySelector('.renew-section-header span:last-child')?.textContent?.trim().toLowerCase();
        if (headerText !== 'capture') return;

        const body = section.querySelector<HTMLElement>('.renew-section-body');
        if (body) {
          body.style.display = 'block';
          body.style.gap = '0';
          body.style.breakInside = 'auto';
          body.style.pageBreakInside = 'auto';
        }

        Array.from(section.querySelectorAll<HTMLElement>('.renew-section-body > div')).forEach((card, index) => {
          card.style.breakInside = 'avoid';
          card.style.pageBreakInside = 'avoid';
          card.style.marginBottom = index === 0 ? '20px' : '0';
        });
      });

      let nextHtml = `<!DOCTYPE html>\n${doc.documentElement.outerHTML}`;
      nextHtml = nextHtml.replace(/WebReport_[0-9a-fA-F]+(?:_live_preview)?/g, 'Patient Report');
      if (/<\/head>/i.test(nextHtml)) {
        nextHtml = nextHtml.replace(/<\/head>/i, `${printStyle}</head>`);
      }
      return nextHtml;
    } catch {
      let nextHtml = html.replace(/<title>.*?<\/title>/i, `<title>${cleanedTitle}</title>`);
      nextHtml = nextHtml.replace(/WebReport_[0-9a-fA-F]+(?:_live_preview)?/g, 'Patient Report');
      if (/<\/head>/i.test(nextHtml)) {
        nextHtml = nextHtml.replace(/<\/head>/i, `${printStyle}</head>`);
      }
      return nextHtml;
    }
  };

  const handleSavePdf = async () => {
    if (!savedPrintUrl) return;
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;

    try {
      const response = await fetch(savedPrintUrl, { credentials: 'same-origin' });
      if (!response.ok) {
        throw new Error('Failed to load saved report');
      }

      const html = await response.text();
      const printableHtml = sanitizePrintHtml(html);

      printWindow.document.open();
      printWindow.document.write(printableHtml);
      printWindow.document.close();

      let hasPrinted = false;
      const runPrint = () => {
        if (hasPrinted) return;
        hasPrinted = true;
        try {
          printWindow.focus();
          printWindow.print();
        } catch {
          // Ignore popup print timing failures.
        }
      };

      printWindow.addEventListener('load', runPrint, { once: true });
      window.setTimeout(runPrint, 1200);
    } catch {
      printWindow.location.href = savedPrintUrl;
    }
  };

  useEffect(() => {
    if (!previewRefreshToken) return;
    setPreviewNonce((current) => current + 1);
  }, [previewRefreshToken]);

  if (!visible) return null;

  return (
    <div
      style={{
        position: 'absolute',
        left,
        top,
        width,
        height,
        zIndex: 40,
        background: shellBg,
        border: `1px solid ${panelBorder}`,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          height: 18,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: '#5C5C5C',
          borderBottom: `1px solid ${panelBorder}`,
          padding: '0 8px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
          <img src={headerMarkerIcon} alt="" draggable={false} style={{ width: 7, height: 7, display: 'block' }} />
          <span>Report</span>
        </div>
        <button
          type="button"
          onClick={onClose}
          style={{
            width: 18,
            height: 14,
            border: '1px solid #2C2C2C',
            background: '#8D8D8D',
            color: '#111111',
            fontSize: 10,
            fontWeight: 700,
            lineHeight: '12px',
            padding: 0,
            cursor: 'pointer',
          }}
          aria-label="Close report workspace"
        >
          -
        </button>
      </div>

      <div
        style={{
          height: 'calc(100% - 18px)',
          display: 'grid',
          gridTemplateColumns: '316px minmax(0, 1fr)',
          background: shellBg,
        }}
      >
        <div
          style={{
            borderRight: `1px solid ${panelBorder}`,
            background: sectionBg,
            display: 'grid',
            gridTemplateRows: 'auto 1fr',
            minHeight: 0,
          }}
        >
          <div style={{ padding: '20px 18px 16px', borderBottom: `1px solid ${panelBorder}` }}>
            <div style={{ color: '#FFFFFF', fontSize: 24, fontWeight: 700, lineHeight: 1.05 }}>Report Workspace</div>
            <div style={{ marginTop: 8, color: '#A7A7A7', fontSize: 12, lineHeight: 1.5 }}>
              Keep the current shell. Review summary, version, and capture references here.
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 14, flexWrap: 'wrap' }}>
              {sessionId ? (
                <button
                  type="button"
                  onClick={handleSavePdf}
                  style={{
                    color: '#FFFFFF',
                    fontSize: 12,
                    fontWeight: 700,
                    padding: '8px 12px',
                    background: '#00C0F3',
                    border: '1px solid #008DB1',
                    cursor: 'pointer',
                  }}
                >
                  Save PDF
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => {
                  void handleRegeneratePreview();
                }}
                disabled={!sessionId || actionState !== 'idle'}
                style={{
                  color: '#F1F1F1',
                  fontSize: 12,
                  fontWeight: 700,
                  padding: '8px 12px',
                  background: '#4A4A4A',
                  border: `1px solid ${panelBorder}`,
                  cursor: sessionId && actionState === 'idle' ? 'pointer' : 'default',
                  opacity: sessionId ? (actionState === 'idle' ? 1 : 0.6) : 0.45,
                }}
              >
                {actionState === 'regenerating' ? 'Generating...' : 'New Draft'}
              </button>
            </div>
          </div>

          <div style={{ overflowY: 'auto', padding: 18, display: 'grid', gap: 18, alignContent: 'start' }}>
            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div
                style={{
                  padding: '12px 14px',
                  borderBottom: `1px solid ${panelBorder}`,
                  color: '#FFFFFF',
                  fontSize: 13,
                  fontWeight: 700,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: 12,
                }}
              >
                <span>Summary</span>
                {summaryRows.length > 4 ? (
                  <button
                    type="button"
                    onClick={() => setSummaryExpanded((current) => !current)}
                    style={{
                      appearance: 'none',
                      border: `1px solid ${panelBorder}`,
                      background: cardInnerBg,
                      color: '#D6D6D6',
                      width: 24,
                      height: 24,
                      lineHeight: 1,
                      fontSize: 13,
                      cursor: 'pointer',
                    }}
                    aria-label={summaryExpanded ? 'Collapse summary' : 'Expand summary'}
                  >
                    {summaryExpanded ? '▲' : '▼'}
                  </button>
                ) : null}
              </div>
              <div style={{ padding: 14, display: 'grid', gap: 8 }}>
                {visibleSummaryRows.map((row) => (
                  <div
                    key={row}
                    style={{
                      background: cardInnerBg,
                      border: `1px solid ${panelBorder}`,
                      padding: '10px 12px',
                      color: '#E5E5E5',
                      fontSize: 12,
                      lineHeight: 1.5,
                    }}
                  >
                    {row}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                Captures
              </div>
              <div style={{ padding: 14 }}>
                {galleryCaptures.length ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                    {galleryCaptures.slice(0, 4).map((capture) => {
                      const isSelected = selectedCaptureIds.includes(capture.id);
                      return (
                      <button
                        key={capture.id}
                        type="button"
                        onClick={() => onToggleCaptureSelection?.(capture.id)}
                        style={{
                          appearance: 'none',
                          cursor: onToggleCaptureSelection ? 'pointer' : 'default',
                          background: cardInnerBg,
                          border: `1px solid ${isSelected ? '#00C0F3' : panelBorder}`,
                          padding: 8,
                          display: 'grid',
                          gap: 8,
                          textAlign: 'left',
                          boxShadow: isSelected ? 'inset 0 0 0 1px rgba(0, 192, 243, 0.3)' : 'none',
                        }}
                      >
                        <img
                          src={capture.dataUrl}
                          alt={capture.label || 'Capture'}
                          draggable={false}
                          style={{
                            display: 'block',
                            width: '100%',
                            height: 78,
                            objectFit: 'cover',
                            background: '#111111',
                            border: `1px solid ${panelBorder}`,
                          }}
                        />
                        <div style={{ color: '#D7D7D7', fontSize: 10, lineHeight: 1.4 }}>
                          <div>{formatTimestamp(capture.createdAt)}</div>
                          <div style={{ color: '#9E9E9E' }}>{capture.size || capture.label || 'Capture reference'}</div>
                          <div style={{ marginTop: 4, color: isSelected ? '#00C0F3' : '#7E7E7E' }}>
                            {isSelected ? 'Selected for document preview' : 'Click to add below detail'}
                          </div>
                        </div>
                      </button>
                    )})}
                  </div>
                ) : (
                  <div
                    style={{
                      minHeight: 120,
                      border: `1px dashed ${panelBorder}`,
                      background: cardInnerBg,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: '#8E8E8E',
                      fontSize: 11,
                      textAlign: 'center',
                      lineHeight: 1.5,
                      padding: 14,
                    }}
                  >
                    Captured images will appear here.
                  </div>
                )}
                {galleryCaptures.length ? (
                  <div style={{ marginTop: 10, color: '#9E9E9E', fontSize: 11, lineHeight: 1.5 }}>
                    Only selected captures are synced into the document preview.
                  </div>
                ) : null}
                {galleryCaptures.length > 4 ? (
                  <div style={{ marginTop: 8, color: '#9E9E9E', fontSize: 11 }}>{`${galleryCaptures.length - 4} more capture(s) available.`}</div>
                ) : null}
              </div>
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                Report Note
              </div>
              <div style={{ padding: 14, display: 'grid', gap: 10 }}>
                <textarea
                  value={reportNoteDraft}
                  onChange={(event) => {
                    setReportNoteDraft(event.target.value);
                    setReportNoteSaveState('idle');
                  }}
                  onFocus={() => setReportNoteFocused(true)}
                  onBlur={() => {
                    setReportNoteFocused(false);
                    void saveReportNote();
                  }}
                  placeholder="Write clinician note for this report"
                  style={{
                    width: '100%',
                    minHeight: 110,
                    resize: 'vertical',
                    border: `1px solid ${panelBorder}`,
                    background: cardInnerBg,
                    color: '#E5E5E5',
                    fontSize: 12,
                    lineHeight: 1.6,
                    padding: '12px 14px',
                    outline: 'none',
                  }}
                />
                <div style={{ color: reportNoteSaveState === 'error' ? '#FF7B7B' : '#9E9E9E', fontSize: 11 }}>
                  {reportNoteSaveState === 'saving'
                    ? 'Saving note...'
                    : reportNoteSaveState === 'saved'
                      ? 'Saved to report Draft.'
                      : reportNoteSaveState === 'error'
                        ? 'Failed to save note.'
                        : 'Blur the field to sync this note with the report draft.'}
                </div>
              </div>
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                Draft
              </div>
              <div style={{ padding: 14, display: 'grid', gap: 12 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <div
                    style={{
                      background: cardInnerBg,
                      border: `1px solid ${panelBorder}`,
                      padding: '10px 12px',
                    }}
                  >
                    <div style={{ color: '#9E9E9E', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Current</div>
                    <div style={{ marginTop: 8, color: '#FFFFFF', fontSize: 24, fontWeight: 700, lineHeight: 1 }}>
                      {currentVersion ?? '-'}
                    </div>
                  </div>
                  <div
                    style={{
                      background: cardInnerBg,
                      border: `1px solid ${panelBorder}`,
                      padding: '10px 12px',
                    }}
                  >
                    <div style={{ color: '#9E9E9E', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>Status</div>
                    <div style={{ marginTop: 10, color: '#FFFFFF', fontSize: 14, fontWeight: 700 }}>
                      {selectedVersionInfo?.status || session?.report?.status || (session?.is_finalized ? 'final' : 'draft')}
                    </div>
                  </div>
                </div>

                <label style={{ display: 'grid', gap: 6 }}>
                  <span style={{ color: '#9E9E9E', fontSize: 11 }}>Draft Select</span>
                  <select
                    value={currentVersion ?? ''}
                    onChange={(event) => setSelectedVersion(Number(event.target.value || 0) || null)}
                    style={{
                      width: '100%',
                      height: 34,
                      border: `1px solid ${panelBorder}`,
                      background: cardInnerBg,
                      color: '#FFFFFF',
                      fontSize: 12,
                      padding: '0 10px',
                      outline: 'none',
                    }}
                  >
                    {(versions.length ? versions : currentVersion ? [{ version: currentVersion, status: session?.report?.status || 'draft' }] : []).map((item) => (
                      <option key={item.version} value={item.version}>
                        {`draft ${item.version}${item.status ? ` (${item.status})` : ''}`}
                      </option>
                    ))}
                  </select>
                </label>

                <InfoRow label="Generated" value={formatTimestamp(selectedVersionInfo?.created_at || session?.report?.created_at)} />
                <InfoRow label="Draft" value={String(versions.length || (currentVersion ? 1 : 0))} />
                <InfoRow label="Updated" value={formatTimestamp(session?.updated_at)} />

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <a
                    href={previewUrl}
                    target="_blank"
                    rel="noreferrer"
                    style={{
                      color: '#F3F3F3',
                      textDecoration: 'none',
                      fontSize: 11,
                      fontWeight: 700,
                      padding: '7px 10px',
                      background: cardInnerBg,
                      border: `1px solid ${panelBorder}`,
                    }}
                  >
                    HTML
                  </a>
                  {selectedVersionInfo?.pdf_path || session?.report?.pdf_path ? (
                    <a
                      href={pdfUrl}
                      target="_blank"
                      rel="noreferrer"
                      style={{
                        color: '#F3F3F3',
                        textDecoration: 'none',
                        fontSize: 11,
                        fontWeight: 700,
                        padding: '7px 10px',
                        background: cardInnerBg,
                        border: `1px solid ${panelBorder}`,
                      }}
                    >
                      PDF
                    </a>
                  ) : null}
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateRows: 'auto 1fr', minWidth: 0, minHeight: 0, background: '#171717' }}>
          <div
            style={{
              padding: '18px 20px 14px',
              borderBottom: `1px solid ${panelBorder}`,
              display: 'flex',
              alignItems: 'end',
              justifyContent: 'space-between',
              gap: 16,
              background: sectionBg,
            }}
          >
            <div>
              <div style={{ color: '#FFFFFF', fontSize: 18, fontWeight: 700 }}>Document Preview</div>
              <div style={{ marginTop: 6, color: '#9E9E9E', fontSize: 12 }}>
                Preview follows the selected report version.
              </div>
            </div>
            <div style={{ color: '#9E9E9E', fontSize: 12, textAlign: 'right' }}>
              {session?.report?.html_path ? 'HTML Ready' : isLoading ? 'Preparing...' : 'Waiting for document'}
            </div>
          </div>

          <div style={{ minHeight: 0, padding: 20, background: '#161616' }}>
            {isLoading ? (
              <div
                style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: sectionBg,
                  border: `1px solid ${panelBorder}`,
                  color: '#D9D9D9',
                  fontSize: 15,
                  fontWeight: 700,
                }}
              >
                Preparing report...
              </div>
            ) : combinedError ? (
              <div
                style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: sectionBg,
                  border: `1px solid ${panelBorder}`,
                  color: '#FFB4B4',
                  fontSize: 15,
                  fontWeight: 700,
                  padding: 28,
                  textAlign: 'center',
                }}
              >
                {combinedError}
              </div>
            ) : sessionId ? (
              <div
                style={{
                  height: '100%',
                  background: '#202020',
                  border: `1px solid ${panelBorder}`,
                  padding: 18,
                  boxSizing: 'border-box',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    background: '#FFFFFF',
                    border: '1px solid #D2D2D2',
                    overflow: 'hidden',
                  }}
                >
                  <iframe
                    title="Web report"
                    src={previewUrl}
                    style={{
                      width: '100%',
                      height: '100%',
                      border: 'none',
                      background: '#FFFFFF',
                    }}
                  />
                </div>
              </div>
            ) : (
              <div
                style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: sectionBg,
                  border: `1px solid ${panelBorder}`,
                  color: '#D9D9D9',
                  fontSize: 15,
                  fontWeight: 700,
                }}
              >
                Report session is not ready.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
