import { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { ChartPage } from './ChartPage';
import {
  fetchWebReportSession,
  finalizeWebReport,
  patchWebReportOverrides,
  regenerateWebReport,
  type WebReportSessionResponse,
} from '../lib/webReportApi';

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

export function WebChartPage() {
  const { sessionId = '' } = useParams();
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionState, setActionState] = useState<'idle' | 'regenerating' | 'finalizing'>('idle');
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [selectedToothId, setSelectedToothId] = useState('');
  const [reviewForm, setReviewForm] = useState<ToothReviewForm>(emptyForm);
  const [reportNoteDraft, setReportNoteDraft] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const lastSavedPayloadRef = useRef('');

  useEffect(() => {
    if (!sessionId) return;
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
          setError(err?.message || 'Failed to load session');
        }
      }
    };

    load();
    const timer = window.setInterval(() => {
      load();
    }, 1500);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  const effectiveResult = session?.effective_result;
  const previewUrl = session?.assets?.preview_url || session?.assets?.source_url || null;
  const reportHref = session?.report?.page_url || `/report/${sessionId}`;
  const chartKey = useMemo(
    () => `${sessionId}:${session?.updated_at || 'pending'}`,
    [sessionId, session?.updated_at]
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
  }, [currentPayload, editorDirty, reportNoteDraft, reviewForm, selectedToothId, session, sessionId]);

  const handleRegenerate = async () => {
    if (!sessionId) return;
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
    if (!sessionId) return;
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
    if (!sessionId || !selectedToothId || session?.is_finalized) return;
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
      const refreshedTooth =
        ((response.effective_result?.teeth as any[]) || []).find(
          (tooth) => String(tooth.tooth_label) === selectedToothId
        ) || null;
      if (refreshedTooth) {
        setReviewForm({
          caries: Boolean(refreshedTooth.caries),
          periapical: Boolean(refreshedTooth.periapical),
          missing: Boolean(refreshedTooth.missing),
          implant: Boolean(refreshedTooth.implant),
          crown: Boolean(refreshedTooth.crown),
          filling: Boolean(refreshedTooth.filling),
          bone_loss_level: Number(refreshedTooth.bone_loss_level || 0),
          bone_loss_pct: Number(refreshedTooth.bone_loss_pct || 0),
          note: refreshedTooth.note || '',
        });
      }
      lastSavedPayloadRef.current = '';
      setEditorDirty(false);
      setSaveState('saved');
    } catch (err: any) {
      setError(err?.message || 'Reset failed');
      setSaveState('error');
    }
  };

  if (!sessionId) {
    return <CenteredMessage title="Invalid session" detail="No session id was provided." />;
  }

  if (!session) {
    return <CenteredMessage title="Loading session" detail="Connecting to the web report workflow..." />;
  }

  if (session.status !== 'completed' && session.status !== 'finalized') {
    return (
      <div className="min-h-screen bg-slate-950 px-6 py-10 text-slate-100">
        <div className="mx-auto max-w-4xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
          <div className="flex items-center justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.24em] text-cyan-300">Web Report Session</p>
              <h1 className="mt-3 text-3xl font-semibold">Chart is preparing</h1>
              <p className="mt-2 text-sm text-slate-400">Session: {sessionId}</p>
            </div>
            <Link to="/" className="rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
              Home
            </Link>
          </div>

          <div className="mt-8 rounded-2xl border border-slate-800 bg-slate-950 p-6">
            <p className="text-sm text-slate-300">Status: <span className="font-semibold text-cyan-300">{session.status}</span></p>
            {session.error && <p className="mt-3 text-sm text-red-300">{session.error}</p>}
            {previewUrl && (
              <div className="mt-6 overflow-hidden rounded-2xl border border-slate-800 bg-black">
                <img src={previewUrl} alt="Preview" className="max-h-[480px] w-full object-contain" />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!effectiveResult) {
    return <CenteredMessage title="No analysis result" detail="The session completed without an effective result payload." />;
  }

  return (
    <div className="relative min-h-screen bg-slate-950">
      <div className="fixed right-4 top-4 z-[200] flex gap-2">
        <Link
          to={reportHref}
          className="rounded-full border border-cyan-400/40 bg-slate-950/90 px-4 py-2 text-sm font-medium text-cyan-200 shadow-lg backdrop-blur"
        >
          Report Draft
        </Link>
        <button
          onClick={handleRegenerate}
          disabled={actionState !== 'idle'}
          className="rounded-full border border-slate-700 bg-slate-950/90 px-4 py-2 text-sm font-medium text-slate-100 shadow-lg backdrop-blur disabled:opacity-50"
        >
          {actionState === 'regenerating' ? 'Regenerating...' : 'Regenerate Report'}
        </button>
        <button
          onClick={handleFinalize}
          disabled={actionState !== 'idle' || Boolean(session.is_finalized)}
          className="rounded-full bg-emerald-600/90 px-4 py-2 text-sm font-medium text-white shadow-lg backdrop-blur disabled:opacity-50"
        >
          {session.is_finalized ? 'Finalized' : actionState === 'finalizing' ? 'Finalizing...' : 'Finalize Report'}
        </button>
      </div>

      {error && (
        <div className="fixed left-4 top-4 z-[200] rounded-2xl border border-red-500/40 bg-black/80 px-4 py-3 text-sm text-red-200 shadow-lg backdrop-blur">
          {error}
        </div>
      )}

      <div className="fixed bottom-4 left-4 z-[180] w-[360px] max-w-[calc(100vw-2rem)] rounded-3xl border border-white/10 bg-slate-950/92 p-4 text-slate-100 shadow-2xl backdrop-blur">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Doctor Review</p>
            <p className="mt-1 text-xs text-slate-400">Session {sessionId.slice(0, 8)}</p>
          </div>
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
        </div>

        <div className="mt-4 space-y-4">
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
                  disabled={session.is_finalized}
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
                disabled={session.is_finalized}
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
                disabled={session.is_finalized}
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
              disabled={session.is_finalized}
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
              rows={2}
              value={reportNoteDraft}
              disabled={session.is_finalized}
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
              disabled={!selectedToothId || session.is_finalized}
              className="flex-1 rounded-2xl border border-slate-700 px-3 py-2 text-sm font-medium text-slate-100 disabled:opacity-50"
            >
              Reset Tooth
            </button>
            <Link
              to={reportHref}
              className="flex-1 rounded-2xl bg-cyan-600 px-3 py-2 text-center text-sm font-medium text-white"
            >
              Open Draft
            </Link>
          </div>
        </div>
      </div>

      <ChartPage key={chartKey} result={effectiveResult} />
    </div>
  );
}

function CenteredMessage({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
        <h1 className="text-2xl font-semibold">{title}</h1>
        <p className="mt-3 text-sm text-slate-300">{detail}</p>
        <Link to="/" className="mt-6 inline-flex rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
          Return home
        </Link>
      </div>
    </div>
  );
}
