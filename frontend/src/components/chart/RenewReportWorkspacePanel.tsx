import { useEffect, useMemo, useState } from 'react';
import {
  fetchWebReportSession,
  regenerateWebReport,
  type WebReportSessionResponse,
} from '../../lib/webReportApi';
import { buildWebReportKeywords, countWebReportFindingTeeth } from '../../lib/webReportKeywords';

type RenewReportWorkspacePanelProps = {
  visible: boolean;
  left: string;
  top: string;
  width: string;
  height: string;
  sessionId: string | null;
  isLoading: boolean;
  error: string | null;
  onClose: () => void;
};

const panelBorder = '#4C4C4C';
const shellBg = '#111111';
const sectionBg = '#2D2D2D';
const cardBg = '#333333';
const cardInnerBg = '#414141';

const assetPath = (relativePath: string) => encodeURI(`/imgs/${relativePath}`);
const headerMarkerIcon = assetPath('7 7.png');

function countFindingEntries(result: any, type: 'caries' | 'periapical' | 'implant' | 'crown' | 'filling') {
  const best = result?.[`${type}_by_tooth_best`];
  const fallback = result?.[`${type}_by_tooth`];
  return Object.keys(best || fallback || {}).length;
}

function countMissingEntries(result: any) {
  const values = Array.isArray(result?.missing_teeth)
    ? result.missing_teeth
    : Array.isArray(result?.teeth_missing)
      ? result.teeth_missing
      : [];
  return values.length;
}

function countBoneLossEntries(result: any) {
  return Object.values(result?.pbl || {}).filter((value) => Number(value || 0) > 0).length;
}

function formatTimestamp(value?: string | null) {
  if (!value) return '-';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
      <span style={{ color: '#9E9E9E', fontSize: 12 }}>{label}</span>
      <span style={{ color: '#FFFFFF', fontSize: 12, textAlign: 'right' }}>{value}</span>
    </div>
  );
}

function SummaryTile({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent: string;
}) {
  return (
    <div
      style={{
        background: cardInnerBg,
        border: `1px solid ${panelBorder}`,
        padding: '12px 12px 10px',
        display: 'grid',
        gap: 8,
      }}
    >
      <div style={{ width: 22, height: 4, background: accent }} />
      <div style={{ color: '#AFAFAF', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em' }}>{label}</div>
      <div style={{ color: '#FFFFFF', fontSize: 22, fontWeight: 700, lineHeight: 1 }}>{value}</div>
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
  onClose,
}: RenewReportWorkspacePanelProps) {
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const [actionState, setActionState] = useState<'idle' | 'regenerating'>('idle');

  useEffect(() => {
    if (!visible || !sessionId) {
      setSession(null);
      setSessionError(null);
      return;
    }

    let cancelled = false;

    const load = async () => {
      try {
        const data = await fetchWebReportSession(sessionId);
        if (cancelled) return;
        setSession(data.session || null);
        setSessionError(data.error || null);
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

  const reportUrl = sessionId ? `/api/web_report/session/${sessionId}/report` : '';
  const effectiveResult = session?.effective_result || session?.ai_result || {};
  const keywords = useMemo(() => buildWebReportKeywords(effectiveResult), [effectiveResult]);
  const totalFindingTeeth = useMemo(() => countWebReportFindingTeeth(effectiveResult), [effectiveResult]);
  const summaryTiles = useMemo(
    () => [
      { label: 'Findings', value: String(totalFindingTeeth), accent: '#00C0F3' },
      { label: 'Caries', value: String(countFindingEntries(effectiveResult, 'caries')), accent: '#FCFF2A' },
      { label: 'Missing', value: String(countMissingEntries(effectiveResult)), accent: '#8B8B8B' },
      { label: 'Bone Loss', value: String(countBoneLossEntries(effectiveResult)), accent: '#FF9D00' },
    ],
    [effectiveResult, totalFindingTeeth]
  );
  const previewUrl = reportUrl ? `${reportUrl}${reportUrl.includes('?') ? '&' : '?'}v=${session?.report?.version ?? 1}&r=${previewNonce}` : '';
  const combinedError = error || sessionError;
  const reportNote = (session?.doctor_overrides?.report_note as string | undefined)?.trim() || '';

  const handleRegeneratePreview = async () => {
    if (!sessionId || actionState !== 'idle') return;

    setActionState('regenerating');
    setSessionError(null);
    try {
      await regenerateWebReport(sessionId);
      const data = await fetchWebReportSession(sessionId);
      setSession(data.session || null);
      setSessionError(data.error || null);
      setPreviewNonce((current) => current + 1);
    } catch (nextError: any) {
      setSessionError(nextError?.message || 'Failed to regenerate report');
    } finally {
      setActionState('idle');
    }
  };

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
          <div style={{ padding: '22px 18px 18px', borderBottom: `1px solid ${panelBorder}` }}>
            <div style={{ color: '#FFFFFF', fontSize: 26, fontWeight: 700, lineHeight: 1.05 }}>Report Workspace</div>
            <div style={{ marginTop: 8, color: '#A7A7A7', fontSize: 13, lineHeight: 1.5 }}>
              Renew UI style summary on the left, generated document preview on the right.
            </div>

            <div style={{ display: 'flex', gap: 8, marginTop: 16, flexWrap: 'wrap' }}>
              {sessionId ? (
                <a
                  href={reportUrl}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    color: '#FFFFFF',
                    textDecoration: 'none',
                    fontSize: 12,
                    fontWeight: 700,
                    padding: '8px 12px',
                    background: '#00C0F3',
                    border: '1px solid #008DB1',
                  }}
                >
                  Open Full
                </a>
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
                {actionState === 'regenerating' ? 'Regenerating...' : 'Regenerate Preview'}
              </button>
            </div>
          </div>

          <div style={{ overflowY: 'auto', padding: 18, display: 'grid', gap: 18, alignContent: 'start' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              {summaryTiles.map((tile) => (
                <SummaryTile key={tile.label} label={tile.label} value={tile.value} accent={tile.accent} />
              ))}
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                Session
              </div>
              <div style={{ padding: 14, display: 'grid', gap: 10 }}>
                <InfoRow label="Session" value={sessionId ? sessionId.slice(0, 8) : '-'} />
                <InfoRow label="Language" value={session?.language || '-'} />
                <InfoRow label="Status" value={session?.is_finalized ? 'Final' : sessionId ? 'Draft' : '-'} />
                <InfoRow label="Version" value={String(session?.report?.version ?? 1)} />
                <InfoRow label="Updated" value={formatTimestamp(session?.updated_at)} />
              </div>
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                AI Findings
              </div>
              <div style={{ padding: 14, display: 'grid', gap: 8 }}>
                {keywords.length ? (
                  keywords.slice(0, 6).map((keyword) => (
                    <div
                      key={keyword}
                      style={{
                        background: cardInnerBg,
                        border: `1px solid ${panelBorder}`,
                        padding: '10px 12px',
                        color: '#E5E5E5',
                        fontSize: 12,
                        whiteSpace: 'pre-line',
                        lineHeight: 1.5,
                      }}
                    >
                      {keyword}
                    </div>
                  ))
                ) : (
                  <div style={{ color: '#9E9E9E', fontSize: 12 }}>No keywords available.</div>
                )}
              </div>
            </div>

            <div style={{ background: cardBg, border: `1px solid ${panelBorder}` }}>
              <div style={{ padding: '12px 14px', borderBottom: `1px solid ${panelBorder}`, color: '#FFFFFF', fontSize: 13, fontWeight: 700 }}>
                Report Note
              </div>
              <div
                style={{
                  padding: 14,
                  color: reportNote ? '#E5E5E5' : '#8F8F8F',
                  fontSize: 12,
                  lineHeight: 1.6,
                  whiteSpace: 'pre-line',
                }}
              >
                {reportNote || 'No clinician note has been added yet.'}
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
                Generated report document inside the Renew workspace shell.
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
