import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchWebReportSession,
  finalizeWebReport,
  patchWebReportOverrides,
  regenerateWebReport,
  type WebReportSessionResponse,
} from '../lib/webReportApi';
import { buildWebReportKeywords } from '../lib/webReportKeywords';

type ToothReviewForm = {
  caries: boolean;
  periapical: boolean;
  missing: boolean;
  implant: boolean;
  crown: boolean;
  filling: boolean;
  bone_loss_level: number;
  bone_loss_pct: number;
  note: string;
};

const emptyForm: ToothReviewForm = {
  caries: false,
  periapical: false,
  missing: false,
  implant: false,
  crown: false,
  filling: false,
  bone_loss_level: 0,
  bone_loss_pct: 0,
  note: '',
};

export function WebReportDrawer({
  sessionId,
  selectedToothId: linkedToothId,
  onClose,
}: {
  sessionId: string;
  selectedToothId?: string | null;
  onClose: () => void;
}) {
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionState, setActionState] = useState<'idle' | 'regenerating' | 'finalizing'>('idle');
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [activeTab, setActiveTab] = useState<'keywords' | 'review' | 'report'>('keywords');
  const [selectedToothId, setSelectedToothId] = useState('');
  const [reviewForm, setReviewForm] = useState<ToothReviewForm>(emptyForm);
  const [reportNoteDraft, setReportNoteDraft] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const lastSavedPayloadRef = useRef('');

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      try {
        const data = await fetchWebReportSession(sessionId);
        if (!cancelled) {
          setSession(data.session || null);
          setError(data.error || null);
        }
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || 'Failed to load report session');
        }
      }
    };

    load();
    const timer = window.setInterval(load, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  const effectiveResult = session?.effective_result;
  const keywords = useMemo(
    () => buildWebReportKeywords(session?.effective_result || session?.ai_result || {}),
    [session]
  );
  const teeth = useMemo(
    () =>
      [...(((effectiveResult?.teeth as any[]) || []).filter((tooth) => tooth?.tooth_label))]
        .sort((a, b) => Number(a.tooth_label) - Number(b.tooth_label)),
    [effectiveResult]
  );

  useEffect(() => {
    if (!teeth.length) {
      if (!editorDirty) {
        setSelectedToothId('');
      }
      return;
    }
    if (!selectedToothId || !teeth.some((tooth) => String(tooth.tooth_label) === selectedToothId)) {
      setSelectedToothId(String(teeth[0].tooth_label));
    }
  }, [teeth, selectedToothId, editorDirty]);

  useEffect(() => {
    if (editorDirty || !linkedToothId) return;
    if (teeth.some((tooth) => String(tooth.tooth_label) === linkedToothId)) {
      setSelectedToothId(linkedToothId);
    }
  }, [editorDirty, linkedToothId, teeth]);

  const selectedTooth = useMemo(
    () => teeth.find((tooth) => String(tooth.tooth_label) === selectedToothId) || null,
    [teeth, selectedToothId]
  );

  useEffect(() => {
    if (!session || !selectedTooth || editorDirty) return;
    const note = (session.doctor_overrides?.teeth?.[selectedToothId]?.note as string | undefined) || selectedTooth.note || '';
    setReviewForm({
      caries: Boolean(selectedTooth.caries),
      periapical: Boolean(selectedTooth.periapical),
      missing: Boolean(selectedTooth.missing),
      implant: Boolean(selectedTooth.implant),
      crown: Boolean(selectedTooth.crown),
      filling: Boolean(selectedTooth.filling),
      bone_loss_level: Number(selectedTooth.bone_loss_level || 0),
      bone_loss_pct: Number(selectedTooth.bone_loss_pct || 0),
      note,
    });
    setReportNoteDraft((session.doctor_overrides?.report_note as string | undefined) || '');
  }, [session, selectedTooth, selectedToothId, editorDirty]);

  const currentPayload = useMemo(() => {
    if (!selectedToothId) return '';
    return JSON.stringify({
      tooth_overrides: { [selectedToothId]: reviewForm },
      report_note: reportNoteDraft,
    });
  }, [selectedToothId, reviewForm, reportNoteDraft]);

  useEffect(() => {
    if (!sessionId || !session || !selectedToothId || !editorDirty || session.is_finalized) return;
    if (currentPayload === lastSavedPayloadRef.current) {
      setEditorDirty(false);
      setSaveState('saved');
      return;
    }

    const timer = window.setTimeout(async () => {
      setSaveState('saving');
      try {
        const response = await patchWebReportOverrides(sessionId, {
          tooth_overrides: { [selectedToothId]: reviewForm },
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
        lastSavedPayloadRef.current = currentPayload;
        setEditorDirty(false);
        setSaveState('saved');
      } catch (err: any) {
        setError(err?.message || 'Autosave failed');
        setSaveState('error');
      }
    }, 800);

    return () => {
      window.clearTimeout(timer);
    };
  }, [currentPayload, editorDirty, reviewForm, reportNoteDraft, selectedToothId, session, sessionId]);

  const handleRegenerate = async () => {
    setActionState('regenerating');
    setError(null);
    try {
      await regenerateWebReport(sessionId);
      const refreshed = await fetchWebReportSession(sessionId);
      setSession(refreshed.session || null);
    } catch (err: any) {
      setError(err?.message || 'Report regeneration failed');
    } finally {
      setActionState('idle');
    }
  };

  const handleFinalize = async () => {
    setActionState('finalizing');
    setError(null);
    try {
      await finalizeWebReport(sessionId);
      const refreshed = await fetchWebReportSession(sessionId);
      setSession(refreshed.session || null);
    } catch (err: any) {
      setError(err?.message || 'Report finalization failed');
    } finally {
      setActionState('idle');
    }
  };

  const handleResetToAi = async () => {
    if (!selectedToothId || session?.is_finalized) return;
    setSaveState('saving');
    try {
      const response = await patchWebReportOverrides(sessionId, {
        reset_tooth_ids: [selectedToothId],
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
      lastSavedPayloadRef.current = '';
      setEditorDirty(false);
      setSaveState('saved');
    } catch (err: any) {
      setError(err?.message || 'Reset failed');
      setSaveState('error');
    }
  };

  const reportUrl = `/api/web_report/session/${sessionId}/report`;
  const reportPageUrl = reportUrl;
  const hasPdf = Boolean(session?.report?.pdf_path);
  const pdfUrl = `/api/web_report/session/${sessionId}/report/pdf`;
  const statusLabel = session?.is_finalized ? 'Final' : 'Draft';

  return (
    <div className="fixed bottom-16 left-1/2 z-[220] flex h-[74vh] max-h-[800px] w-[960px] max-w-[calc(100vw-2rem)] -translate-x-1/2 flex-col overflow-hidden rounded-[34px] border border-white/10 bg-[#07101F]/98 text-slate-100 shadow-[0_36px_120px_rgba(3,8,20,0.62)] backdrop-blur-xl">
      <div className="pointer-events-none absolute -top-5 left-1/2 -translate-x-1/2">
        <div className="h-12 w-44 rounded-full bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-500 opacity-75 blur-[10px]" />
      </div>
      <div className="absolute -top-3 left-1/2 -translate-x-1/2 rounded-full border border-cyan-300/30 bg-[#09172f] px-5 py-1 text-[10px] font-semibold uppercase tracking-[0.3em] text-cyan-200 shadow-[0_10px_30px_rgba(34,211,238,0.22)]">
        Report Sheet
      </div>
      <div className="absolute left-1/2 top-4 h-1.5 w-24 -translate-x-1/2 rounded-full bg-white/25" />

      <div className="flex items-center justify-between border-b border-white/10 bg-[linear-gradient(180deg,rgba(10,25,47,0.96),rgba(7,16,31,0.86))] px-6 pb-4 pt-8">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-cyan-500/15 text-cyan-200 shadow-[inset_0_0_0_1px_rgba(103,232,249,0.15)]">
            <span className="text-sm font-bold tracking-[0.18em]">AI</span>
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.26em] text-cyan-300">AI Note</p>
            <p className="mt-1 text-sm text-slate-200">Session {sessionId.slice(0, 8)} · {statusLabel}</p>
            <p className="mt-1 text-xs text-slate-400">Bottom-sheet workspace for report actions, notes, and clinical review.</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <a
            href={reportPageUrl}
            target="_blank"
            rel="noreferrer"
            className="rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/10"
          >
            Open Full
          </a>
          <button
            onClick={onClose}
            className="rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-xs text-slate-100 hover:bg-white/10"
          >
            Minimize
          </button>
        </div>
      </div>

      <div className="flex items-center gap-2 border-b border-white/10 bg-black/10 px-6 py-3">
        {(['keywords', 'review', 'report'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`rounded-full px-4 py-2 text-xs font-medium transition ${
              activeTab === tab
                ? 'bg-cyan-400 text-slate-950 shadow-[0_10px_25px_rgba(34,211,238,0.22)]'
                : 'bg-white/5 text-slate-200 hover:bg-white/10'
            }`}
          >
            {tab === 'keywords' ? 'Keywords' : tab === 'review' ? 'Review' : 'Report'}
          </button>
        ))}
        <div className="ml-auto flex gap-2">
          <button
            onClick={handleRegenerate}
            disabled={actionState !== 'idle'}
            className="rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-xs text-slate-100 disabled:opacity-50"
          >
            {actionState === 'regenerating' ? 'Regenerating...' : 'Regenerate'}
          </button>
          <button
            onClick={handleFinalize}
            disabled={actionState !== 'idle' || Boolean(session?.is_finalized)}
            className="rounded-full bg-emerald-500 px-3 py-1.5 text-xs font-medium text-slate-950 disabled:opacity-50"
          >
            {session?.is_finalized ? 'Finalized' : actionState === 'finalizing' ? 'Finalizing...' : 'Finalize'}
          </button>
        </div>
      </div>

      {error && (
        <div className="mx-6 mt-4 rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {activeTab === 'report' ? (
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="rounded-[28px] border border-cyan-400/15 bg-[linear-gradient(135deg,rgba(34,211,238,0.12),rgba(14,165,233,0.04))] p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Report Workspace</p>
            <p className="mt-2 text-sm leading-6 text-slate-200">
              The report is no longer embedded in the bottom sheet. Review findings here, then open the full HTML document in its own page.
            </p>
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Current Version</p>
              <div className="mt-3 flex items-end gap-3">
                <span className="text-4xl font-semibold tracking-[-0.04em] text-white">{session?.report?.version ?? 1}</span>
                <span className="mb-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-200">
                  {statusLabel}
                </span>
              </div>
              <p className="mt-3 text-sm text-slate-300">
                Regenerate after review changes, then open the full report page to inspect the actual document layout.
              </p>
            </div>

            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Open Document</p>
              <div className="mt-4 flex flex-wrap gap-2">
                <a
                  href={reportPageUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full bg-cyan-400 px-4 py-2 text-xs font-semibold text-slate-950 shadow-[0_10px_25px_rgba(34,211,238,0.22)]"
                >
                  Open HTML Report
                </a>
                <a
                  href={reportUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full border border-slate-700 bg-white/5 px-4 py-2 text-xs text-slate-100 hover:bg-white/10"
                >
                  Raw HTML
                </a>
                {hasPdf ? (
                  <a
                    href={pdfUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-slate-700 bg-white/5 px-4 py-2 text-xs text-slate-100 hover:bg-white/10"
                  >
                    Open PDF
                  </a>
                ) : (
                  <span className="rounded-full border border-slate-800 bg-black/20 px-4 py-2 text-xs text-slate-500">
                    PDF unavailable
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-[24px] border border-white/10 bg-white/5 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Workflow</p>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              {[
                '1. Review AI keywords and tooth findings.',
                '2. Edit labels and report note in Review.',
                '3. Regenerate and open the full report page.',
              ].map((step) => (
                <div key={step} className="rounded-2xl border border-white/10 bg-black/10 px-4 py-4 text-sm text-slate-200">
                  {step}
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : activeTab === 'keywords' ? (
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="rounded-[26px] border border-cyan-400/15 bg-[linear-gradient(135deg,rgba(34,211,238,0.12),rgba(14,165,233,0.04))] p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Model Findings</p>
            <p className="mt-2 text-sm leading-6 text-slate-300">
              Use these quick keywords to review findings before regenerating and opening the full report page.
            </p>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            {keywords.map((keyword) => (
              <span
                key={keyword}
                className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-slate-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]"
              >
                {keyword}
              </span>
            ))}
          </div>

          <div className="mt-5 rounded-[24px] border border-white/10 bg-white/5 p-4">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Report Note</p>
            <textarea
              rows={4}
              value={reportNoteDraft}
              disabled={session?.is_finalized}
              onChange={(event) => {
                setReportNoteDraft(event.target.value);
                setEditorDirty(true);
                setSaveState('idle');
              }}
              className="mt-3 w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
              placeholder="Add a clinician note for the report draft"
            />
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-y-auto px-6 py-5">
          <div className="mb-4 flex items-center justify-between">
            <span
              className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${
                saveState === 'saving'
                  ? 'bg-amber-500/15 text-amber-200'
                  : saveState === 'saved'
                    ? 'bg-emerald-500/15 text-emerald-200'
                    : saveState === 'error'
                      ? 'bg-red-500/15 text-red-200'
                      : 'bg-slate-800 text-slate-300'
              }`}
            >
              {saveState === 'saving' ? 'Saving' : saveState === 'saved' ? 'Saved' : saveState === 'error' ? 'Error' : 'Idle'}
            </span>
            <div className="text-xs text-slate-400">Review updates apply to the effective result and next report regeneration.</div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Tooth</label>
              <select
                value={selectedToothId}
                onChange={(event) => {
                  setSelectedToothId(event.target.value);
                  setEditorDirty(false);
                  setSaveState('idle');
                }}
                className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
              >
                {teeth.map((tooth) => (
                  <option key={tooth.tooth_label} value={String(tooth.tooth_label)}>
                    Tooth {tooth.tooth_label}
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-2">
              {(
                [
                  ['caries', 'Caries'],
                  ['periapical', 'Periapical'],
                  ['missing', 'Missing'],
                  ['implant', 'Implant'],
                  ['crown', 'Crown'],
                  ['filling', 'Filling'],
                ] as const
              ).map(([field, label]) => (
                <label key={field} className="flex items-center gap-2 rounded-2xl border border-slate-800 bg-slate-900 px-3 py-2 text-sm">
                  <input
                    type="checkbox"
                    checked={reviewForm[field]}
                    disabled={session?.is_finalized}
                    onChange={(event) => {
                      setReviewForm((prev) => ({ ...prev, [field]: event.target.checked }));
                      setEditorDirty(true);
                      setSaveState('idle');
                    }}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </div>

            <div className="grid grid-cols-2 gap-2">
              <label className="block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Bone Level</span>
                <input
                  type="number"
                  min={0}
                  max={4}
                  value={reviewForm.bone_loss_level}
                  disabled={session?.is_finalized}
                  onChange={(event) => {
                    setReviewForm((prev) => ({ ...prev, bone_loss_level: Number(event.target.value || 0) }));
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                  className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Bone %</span>
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={0.1}
                  value={reviewForm.bone_loss_pct}
                  disabled={session?.is_finalized}
                  onChange={(event) => {
                    setReviewForm((prev) => ({ ...prev, bone_loss_pct: Number(event.target.value || 0) }));
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                  className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
                />
              </label>
            </div>

            <label className="block">
              <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Tooth Note</span>
              <textarea
                rows={3}
                value={reviewForm.note}
                disabled={session?.is_finalized}
                onChange={(event) => {
                  setReviewForm((prev) => ({ ...prev, note: event.target.value }));
                  setEditorDirty(true);
                  setSaveState('idle');
                }}
                className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
              />
            </label>

            <label className="block">
              <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Report Note</span>
              <textarea
                rows={3}
                value={reportNoteDraft}
                disabled={session?.is_finalized}
                onChange={(event) => {
                  setReportNoteDraft(event.target.value);
                  setEditorDirty(true);
                  setSaveState('idle');
                }}
                className="w-full rounded-2xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm outline-none focus:border-cyan-400"
              />
            </label>

            <div className="flex gap-2">
              <button
                onClick={handleResetToAi}
                disabled={!selectedToothId || Boolean(session?.is_finalized)}
                className="flex-1 rounded-2xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-100 disabled:opacity-50"
              >
                Reset Tooth
              </button>
              <button
                onClick={() => setActiveTab('report')}
                className="flex-1 rounded-2xl bg-cyan-500 px-3 py-2 text-sm font-medium text-slate-950"
              >
                Open Report Actions
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
