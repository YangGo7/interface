import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  fetchWebReportSession,
  finalizeWebReport,
  listWebReportVersions,
  patchWebReportOverrides,
  regenerateWebReport,
  rollbackWebReportVersion,
  type WebReportSessionResponse,
  type WebReportVersionSummary,
} from '../lib/webReportApi';
import logo from '../assets/O3_logo_only.png';

type ToothReviewForm = {
  included: boolean;
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

type ToothConditionField =
  | 'caries'
  | 'periapical'
  | 'missing'
  | 'implant'
  | 'crown'
  | 'filling'
  | 'boneLoss';

type AttachedCaptureDraft = {
  id: string;
  label: string;
  dataUrl: string;
  reportDataUrl?: string;
  size?: string;
  createdAt: string;
  note?: string;
  toothId?: string;
};

const emptyForm: ToothReviewForm = {
  included: true,
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

const toothConditionOptions = [
  ['caries', 'Caries'],
  ['periapical', 'Periapical'],
  ['missing', 'Missing'],
  ['implant', 'Implant'],
  ['crown', 'Crown'],
  ['filling', 'Filling'],
] as const;

function applyConditionToggle(form: ToothReviewForm, field: ToothConditionField, checked: boolean): ToothReviewForm {
  if (field === 'boneLoss') {
    return {
      ...form,
      bone_loss_level: checked ? Math.max(1, Number(form.bone_loss_level || 0)) : 0,
      bone_loss_pct: checked ? Math.max(10, Number(form.bone_loss_pct || 0)) : 0,
    };
  }
  return {
    ...form,
    [field]: checked,
  };
}

function buildReviewForm(tooth: any, toothOverride: any): ToothReviewForm {
  const hasOverrideField = (field: string) => Object.prototype.hasOwnProperty.call(toothOverride || {}, field);
  return {
    included: hasOverrideField('included') ? Boolean(toothOverride.included) : tooth?.included !== false,
    caries: hasOverrideField('caries') ? Boolean(toothOverride.caries) : Boolean(tooth?.caries),
    periapical: hasOverrideField('periapical') ? Boolean(toothOverride.periapical) : Boolean(tooth?.periapical),
    missing: hasOverrideField('missing') ? Boolean(toothOverride.missing) : Boolean(tooth?.missing),
    implant: hasOverrideField('implant') ? Boolean(toothOverride.implant) : Boolean(tooth?.implant),
    crown: hasOverrideField('crown') ? Boolean(toothOverride.crown) : Boolean(tooth?.crown),
    filling: hasOverrideField('filling') ? Boolean(toothOverride.filling) : Boolean(tooth?.filling),
    bone_loss_level: hasOverrideField('bone_loss_level')
      ? Number(toothOverride.bone_loss_level || 0)
      : Number(tooth?.bone_loss_level || 0),
    bone_loss_pct: hasOverrideField('bone_loss_pct')
      ? Number(toothOverride.bone_loss_pct || 0)
      : Number(tooth?.bone_loss_pct || 0),
    note: hasOverrideField('note') ? String(toothOverride.note || '') : String(tooth?.note || ''),
  };
}

function buildReportHtmlUrl(sessionId: string, version?: number | null, nonce?: number) {
  const params = new URLSearchParams();
  if (version) params.set('version', String(version));
  if (typeof nonce === 'number') params.set('r', String(nonce));
  const query = params.toString();
  return `/api/web_report/session/${sessionId}/report${query ? `?${query}` : ''}`;
}

function hasToothSignal(tooth: any) {
  return Boolean(
    tooth?.caries ||
      tooth?.periapical ||
      tooth?.missing ||
      tooth?.implant ||
      tooth?.crown ||
      tooth?.filling ||
      Number(tooth?.bone_loss_level || 0) > 0 ||
      Number(tooth?.bone_loss_pct || 0) > 0 ||
      String(tooth?.note || '').trim()
  );
}

function formatTimestamp(value?: string | null) {
  if (!value) return '-';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

export function WebReportPage() {
  const { sessionId = '' } = useParams();
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [versions, setVersions] = useState<WebReportVersionSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [actionState, setActionState] = useState<'idle' | 'saving' | 'regenerating' | 'rolling-back' | 'finalizing'>('idle');
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [selectedToothId, setSelectedToothId] = useState('');
  const [reviewForm, setReviewForm] = useState<ToothReviewForm>(emptyForm);
  const [reportNoteDraft, setReportNoteDraft] = useState('');
  const [attachedCapturesDraft, setAttachedCapturesDraft] = useState<AttachedCaptureDraft[]>([]);
  const [selectedCaptureId, setSelectedCaptureId] = useState('');
  const [editorDirty, setEditorDirty] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [previewNonce, setPreviewNonce] = useState(0);
  const lastSavedPayloadRef = useRef('');
  const captureInputRef = useRef<HTMLInputElement | null>(null);
  const isKorean = session?.language === 'Korean';

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    const load = async () => {
      try {
        const [sessionData, versionsData] = await Promise.all([
          fetchWebReportSession(sessionId),
          listWebReportVersions(sessionId),
        ]);
        if (cancelled) return;
        setSession(sessionData.session || null);
        setVersions(versionsData.versions || []);
        setError(sessionData.error || null);
      } catch (err: any) {
        if (!cancelled) {
          setError(err?.message || 'Failed to load report session');
        }
      }
    };

    void load();
    const timer = window.setInterval(load, 2000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  useEffect(() => {
    if (selectedVersion !== null) return;
    const currentVersion = session?.report?.version ?? versions[0]?.version ?? null;
    if (currentVersion !== null) {
      setSelectedVersion(currentVersion);
    }
  }, [selectedVersion, session?.report?.version, versions]);

  const effectiveResult = session?.effective_result;
  const teeth = useMemo(
    () =>
      [...(((effectiveResult?.teeth as any[]) || []).filter((tooth) => tooth?.tooth_label))].sort(
        (a, b) => Number(a.tooth_label) - Number(b.tooth_label)
      ),
    [effectiveResult]
  );
  const draftTeeth = useMemo(
    () =>
      teeth.filter((tooth) => {
        const override = session?.doctor_overrides?.teeth?.[String(tooth.tooth_label)] || {};
        return hasToothSignal(tooth) || Object.prototype.hasOwnProperty.call(override, 'included');
      }),
    [session?.doctor_overrides, teeth]
  );
  const selectedTooth = useMemo(
    () => teeth.find((tooth) => String(tooth.tooth_label) === selectedToothId) || null,
    [teeth, selectedToothId]
  );
  const currentVersion = selectedVersion ?? session?.report?.version ?? versions[0]?.version ?? null;
  const previewUrl = sessionId ? buildReportHtmlUrl(sessionId, currentVersion, previewNonce) : '';
  const selectedVersionInfo = versions.find((item) => item.version === currentVersion) || null;
  const selectedToothCaptures = useMemo(
    () => attachedCapturesDraft.filter((capture) => (capture.toothId || '') === selectedToothId),
    [attachedCapturesDraft, selectedToothId]
  );
  const allCaptures = useMemo(() => attachedCapturesDraft, [attachedCapturesDraft]);
  const activeCapture = useMemo(
    () =>
      allCaptures.find((capture) => capture.id === selectedCaptureId) ||
      allCaptures[allCaptures.length - 1] ||
      null,
    [allCaptures, selectedCaptureId]
  );
  const currentPayload = useMemo(
    () =>
      JSON.stringify({
        tooth_overrides: selectedToothId ? { [selectedToothId]: reviewForm } : {},
        report_note: reportNoteDraft,
        attached_captures: attachedCapturesDraft,
      }),
    [attachedCapturesDraft, reportNoteDraft, reviewForm, selectedToothId]
  );

  useEffect(() => {
    if (!teeth.length) {
      if (!editorDirty) setSelectedToothId('');
      return;
    }
    if (!selectedToothId || !teeth.some((tooth) => String(tooth.tooth_label) === selectedToothId)) {
      const next = draftTeeth[0] || teeth[0];
      if (next) setSelectedToothId(String(next.tooth_label));
    }
  }, [draftTeeth, editorDirty, selectedToothId, teeth]);

  useEffect(() => {
    if (!allCaptures.length) {
      setSelectedCaptureId('');
      return;
    }
    if (!allCaptures.some((capture) => capture.id === selectedCaptureId)) {
      setSelectedCaptureId(allCaptures[allCaptures.length - 1].id);
    }
  }, [allCaptures, selectedCaptureId]);

  useEffect(() => {
    if (!session || !selectedTooth || editorDirty) return;
    const toothOverride = session.doctor_overrides?.teeth?.[selectedToothId] || {};
    setReviewForm(buildReviewForm(selectedTooth, toothOverride));
    setReportNoteDraft((session.doctor_overrides?.report_note as string | undefined) || '');
      setAttachedCapturesDraft(
      ((session.doctor_overrides?.attached_captures as AttachedCaptureDraft[] | undefined) || []).map((capture) => ({
        ...capture,
        id: capture.id || `${capture.createdAt}-${capture.label}`,
        toothId: capture.toothId ? String(capture.toothId) : undefined,
      }))
    );
  }, [editorDirty, selectedTooth, selectedToothId, session]);

  useEffect(() => {
    if (!sessionId || !session || !editorDirty || session.is_finalized) return;
    if (currentPayload === lastSavedPayloadRef.current) {
      setEditorDirty(false);
      setSaveState('saved');
      return;
    }

    const timer = window.setTimeout(async () => {
      await persistOverrides();
    }, 700);

    return () => {
      window.clearTimeout(timer);
    };
  }, [currentPayload, editorDirty, session, sessionId]);

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data || data.source !== 'web-report-preview') return;
      const nextTooth = String(data.toothLabel || '');
      if (!nextTooth) return;
      setSelectedToothId(nextTooth);
      if (data.action === 'toggle-tooth') {
        const tooth = teeth.find((item) => String(item.tooth_label) === nextTooth);
        const toothOverride = session?.doctor_overrides?.teeth?.[nextTooth] || {};
        if (tooth) {
          setReviewForm({
            ...buildReviewForm(tooth, toothOverride),
            included: Boolean(data.checked),
          });
          setEditorDirty(true);
          setSaveState('idle');
        }
        return;
      }
      if (data.action === 'toggle-condition') {
        const field = String(data.field || '') as ToothConditionField;
        const checked = Boolean(data.checked);
        const tooth = teeth.find((item) => String(item.tooth_label) === nextTooth);
        const toothOverride = session?.doctor_overrides?.teeth?.[nextTooth] || {};
        if (
          tooth &&
          ['caries', 'periapical', 'missing', 'implant', 'crown', 'filling', 'boneLoss'].includes(field)
        ) {
          setReviewForm(applyConditionToggle(buildReviewForm(tooth, toothOverride), field, checked));
          setEditorDirty(true);
          setSaveState('idle');
        }
        return;
      }
      if (data.action === 'note-change') {
        const tooth = teeth.find((item) => String(item.tooth_label) === nextTooth);
        const toothOverride = session?.doctor_overrides?.teeth?.[nextTooth] || {};
        if (tooth) {
          setReviewForm({
            ...buildReviewForm(tooth, toothOverride),
            note: String(data.note || ''),
          });
          setEditorDirty(true);
          setSaveState('idle');
        }
        return;
      }
      if (data.action === 'open-capture') {
        if (!session?.is_finalized) {
          window.setTimeout(() => captureInputRef.current?.click(), 0);
        }
        return;
      }
      if (data.action === 'context-tooth') {
        const tooth = teeth.find((item) => String(item.tooth_label) === nextTooth);
        const toothOverride = session?.doctor_overrides?.teeth?.[nextTooth] || {};
        if (tooth) {
          setReviewForm({
            ...buildReviewForm(tooth, toothOverride),
            included: true,
          });
        }
        setEditorDirty(true);
        setSaveState('idle');
      }
    };

    window.addEventListener('message', handleMessage);
    return () => {
      window.removeEventListener('message', handleMessage);
    };
  }, [session?.doctor_overrides, teeth]);

  const persistOverrides = async () => {
    if (!sessionId || !session || session.is_finalized) return true;

    setActionState('saving');
    setSaveState('saving');
    try {
      const response = await patchWebReportOverrides(sessionId, {
        tooth_overrides: selectedToothId ? { [selectedToothId]: reviewForm } : {},
        report_note: reportNoteDraft,
        attached_captures: attachedCapturesDraft,
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
      return true;
    } catch (err: any) {
      setError(err?.message || 'Draft save failed');
      setSaveState('error');
      return false;
    } finally {
      setActionState('idle');
    }
  };

  const reloadSession = async () => {
    if (!sessionId) return;
    const [sessionData, versionsData] = await Promise.all([
      fetchWebReportSession(sessionId),
      listWebReportVersions(sessionId),
    ]);
    setSession(sessionData.session || null);
    setVersions(versionsData.versions || []);
    setError(sessionData.error || null);
  };

  const handleGenerateDraft = async () => {
    if (!sessionId) return;
    setActionState('regenerating');
    setError(null);
    try {
      if (editorDirty) {
        const saved = await persistOverrides();
        if (!saved) return;
      }
      const response = await regenerateWebReport(sessionId);
      await reloadSession();
      setSelectedVersion(response.version);
      setPreviewNonce((current) => current + 1);
    } catch (err: any) {
      setError(err?.message || 'Draft generation failed');
    } finally {
      setActionState('idle');
    }
  };

  const handleRestoreVersion = async () => {
    if (!sessionId || !currentVersion) return;
    setActionState('rolling-back');
    setError(null);
    try {
      setEditorDirty(false);
      const rollback = await rollbackWebReportVersion(sessionId, currentVersion);
      setSession((prev) =>
        prev
          ? {
              ...prev,
              doctor_overrides: rollback.doctor_overrides,
              effective_result: rollback.effective_result,
              updated_at: new Date().toISOString(),
            }
          : prev
      );
      const response = await regenerateWebReport(sessionId);
      await reloadSession();
      setSelectedVersion(response.version);
      setPreviewNonce((current) => current + 1);
      setSaveState('saved');
    } catch (err: any) {
      setError(err?.message || 'Rollback failed');
      setSaveState('error');
    } finally {
      setActionState('idle');
    }
  };

  const handleFinalize = async () => {
    if (!sessionId) return;
    setActionState('finalizing');
    setError(null);
    try {
      if (editorDirty) {
        const saved = await persistOverrides();
        if (!saved) return;
      }
      await finalizeWebReport(sessionId);
      await reloadSession();
      setPreviewNonce((current) => current + 1);
    } catch (err: any) {
      setError(err?.message || 'Finalization failed');
    } finally {
      setActionState('idle');
    }
  };

  const handleAddCaptureFiles = async (files: FileList | File[] | null) => {
    if (!files?.length) return;

    try {
      const nextCaptures = await Promise.all(
        Array.from(files)
          .filter((file) => file.type.startsWith('image/'))
          .map(async (file, index) => ({
            id: `${selectedToothId || 'capture'}-${Date.now()}-${index}-${file.name}`,
            label: file.name,
            dataUrl: await readFileAsDataUrl(file),
            size: `${Math.round(file.size / 1024)} KB`,
            createdAt: new Date().toISOString(),
            note: '',
            toothId: selectedToothId || undefined,
          }))
      );

      if (!nextCaptures.length) return;
      setAttachedCapturesDraft((prev) => [...prev, ...nextCaptures]);
      setSelectedCaptureId(nextCaptures[nextCaptures.length - 1].id);
      setEditorDirty(true);
      setSaveState('idle');
    } catch (nextError: any) {
      setError(nextError?.message || 'Capture upload failed');
    }
  };

  const handleDropCapture = async (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (session.is_finalized) return;
    await handleAddCaptureFiles(event.dataTransfer.files);
  };

  const handleRemoveCapture = (captureId: string) => {
    setAttachedCapturesDraft((prev) => prev.filter((capture) => capture.id !== captureId));
    setEditorDirty(true);
    setSaveState('idle');
  };

  if (!sessionId) {
    return <SimpleState title="Invalid session" detail="No session id was provided." />;
  }

  if (!session) {
    return <SimpleState title="Loading report" detail="Connecting to the report session..." />;
  }

  if ((session.status !== 'completed' && session.status !== 'finalized') || !session.report?.html_path) {
    return (
      <SimpleState
        title={isKorean ? '리포트 준비 중' : 'Report not ready'}
        detail={error || (isKorean ? `현재 상태: ${session.status}` : `Current status: ${session.status}`)}
        action={
          <Link to={`/chart/${sessionId}`} className="inline-flex rounded-full border border-slate-700 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800">
            {isKorean ? '차트로 돌아가기' : 'Back to chart'}
          </Link>
        }
      />
    );
  }

  return (
    <div className="min-h-screen bg-[#171717] text-slate-100">
      <div className="border-b border-[#4C4C4C] bg-[#2D2D2D] px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src={logo} alt="Logo" className="h-7 w-auto object-contain" />
            <div>
              <div className="text-lg font-semibold text-white">Report Editor</div>
              <div className="text-xs text-slate-400">Session {sessionId.slice(0, 8)}</div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill state={saveState} />
            <button
              onClick={() => {
                void persistOverrides();
              }}
              disabled={actionState !== 'idle' || session.is_finalized}
              className="rounded-lg border border-[#5C5C5C] bg-[#3A3A3A] px-3 py-2 text-sm text-white disabled:opacity-50"
            >
              Save Draft
            </button>
            <button
              onClick={() => {
                void handleGenerateDraft();
              }}
              disabled={actionState !== 'idle' || session.is_finalized}
              className="rounded-lg border border-[#008DB1] bg-[#00C0F3] px-3 py-2 text-sm font-semibold text-[#08131A] disabled:opacity-50"
            >
              {actionState === 'regenerating' ? 'Generating...' : 'Generate Draft Version'}
            </button>
            <button
              onClick={() => {
                void handleFinalize();
              }}
              disabled={actionState !== 'idle' || session.is_finalized}
              className="rounded-lg border border-emerald-500/50 bg-emerald-600 px-3 py-2 text-sm font-semibold text-white disabled:opacity-50"
            >
              {session.is_finalized ? 'Finalized' : actionState === 'finalizing' ? 'Finalizing...' : 'Finalize'}
            </button>
            <Link to={`/chart/${sessionId}`} className="rounded-lg border border-[#5C5C5C] bg-transparent px-3 py-2 text-sm text-slate-200">
              Back to chart
            </Link>
          </div>
        </div>
        {error ? <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-200">{error}</div> : null}
      </div>

      <div className="grid min-h-[calc(100vh-73px)] grid-cols-[380px_minmax(0,1fr)]">
        <aside className="border-r border-[#4C4C4C] bg-[#2D2D2D]">
          <div className="h-full overflow-y-auto p-4">
            <SectionCard title="Version">
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Current" value={String(session.report?.version ?? '-')} />
                <MetricCard label="Viewing" value={String(currentVersion ?? '-')} />
              </div>
              <label className="mt-3 block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Version History</span>
                <select
                  value={currentVersion ?? ''}
                  onChange={(event) => setSelectedVersion(Number(event.target.value || 0) || null)}
                  className="w-full rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-sm text-white outline-none"
                >
                  {versions.map((item) => (
                    <option key={item.version} value={item.version}>
                      {`Version ${item.version} (${item.status})`}
                    </option>
                  ))}
                </select>
              </label>
              <div className="mt-3 grid gap-1 text-xs text-slate-400">
                <div>{`Saved: ${formatTimestamp(selectedVersionInfo?.created_at || session.report?.created_at)}`}</div>
                <div>{`Updated: ${formatTimestamp(session.updated_at)}`}</div>
              </div>
              <button
                onClick={() => {
                  void handleRestoreVersion();
                }}
                disabled={!currentVersion || actionState !== 'idle' || session.is_finalized}
                className="mt-3 w-full rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm font-medium text-amber-100 disabled:opacity-50"
              >
                {actionState === 'rolling-back' ? 'Restoring...' : 'Restore This Version As Draft'}
              </button>
            </SectionCard>

            <SectionCard title="AI Draft">
              <div className="max-h-52 space-y-2 overflow-y-auto pr-1">
                {draftTeeth.map((tooth) => {
                  const toothId = String(tooth.tooth_label);
                  const override = session.doctor_overrides?.teeth?.[toothId] || {};
                  const included = Object.prototype.hasOwnProperty.call(override, 'included')
                    ? Boolean(override.included)
                    : tooth.included !== false;
                  return (
                    <button
                      key={toothId}
                      type="button"
                      onClick={() => {
                        setSelectedToothId(toothId);
                        setSaveState('idle');
                      }}
                      className={`flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left ${
                        selectedToothId === toothId ? 'border-cyan-400 bg-cyan-400/10' : 'border-[#4C4C4C] bg-[#3A3A3A]'
                      }`}
                    >
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-white">{`Tooth ${toothId}`}</div>
                        <div className="truncate text-[11px] text-slate-400">
                          {String(tooth.note || override.note || '').trim() ? 'Note added' : hasToothSignal(tooth) ? 'AI finding' : 'Manual draft'}
                        </div>
                      </div>
                      <input
                        type="checkbox"
                        checked={included}
                        onChange={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                          setSelectedToothId(toothId);
                          setReviewForm({
                            ...buildReviewForm(tooth, override),
                            included: event.target.checked,
                          });
                          setEditorDirty(true);
                          setSaveState('idle');
                        }}
                      />
                    </button>
                  );
                })}
              </div>
              <div className="mt-2 text-[11px] text-slate-500">Uncheck and save to remove the tooth from the report draft.</div>
            </SectionCard>

            <SectionCard title="Selected Tooth">
              <div className="mb-3 text-sm text-slate-300">{selectedToothId ? `Tooth ${selectedToothId}` : 'Select a tooth from preview or draft list.'}</div>
              <label className="mb-3 flex items-center gap-2 rounded-lg border border-[#4C4C4C] bg-[#3A3A3A] px-3 py-2 text-sm">
                <input
                  type="checkbox"
                  checked={reviewForm.included}
                  disabled={!selectedToothId || session.is_finalized}
                  onChange={(event) => {
                    setReviewForm((prev) => ({ ...prev, included: event.target.checked }));
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                />
                <span>Include in report draft</span>
              </label>

              <div className="grid grid-cols-2 gap-2">
                {toothConditionOptions.map(([field, label]) => (
                  <label key={field} className="flex items-center gap-2 rounded-lg border border-[#4C4C4C] bg-[#3A3A3A] px-3 py-2 text-sm">
                    <input
                      type="checkbox"
                      checked={reviewForm[field]}
                      disabled={!selectedToothId || session.is_finalized}
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

              <div className="mt-3 grid grid-cols-2 gap-2">
                <label className="block">
                  <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Bone Level</span>
                  <input
                    type="number"
                    min={0}
                    max={4}
                    value={reviewForm.bone_loss_level}
                    disabled={!selectedToothId || session.is_finalized}
                    onChange={(event) => {
                      setReviewForm((prev) => ({ ...prev, bone_loss_level: Number(event.target.value || 0) }));
                      setEditorDirty(true);
                      setSaveState('idle');
                    }}
                    className="w-full rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-sm outline-none"
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
                    disabled={!selectedToothId || session.is_finalized}
                    onChange={(event) => {
                      setReviewForm((prev) => ({ ...prev, bone_loss_pct: Number(event.target.value || 0) }));
                      setEditorDirty(true);
                      setSaveState('idle');
                    }}
                    className="w-full rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-sm outline-none"
                  />
                </label>
              </div>

              <label className="mt-3 block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Tooth Note</span>
                <textarea
                  rows={4}
                  value={reviewForm.note}
                  disabled={!selectedToothId || session.is_finalized}
                  onChange={(event) => {
                    setReviewForm((prev) => ({ ...prev, note: event.target.value }));
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                  className="w-full rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-sm outline-none"
                />
              </label>

                <div className="mt-2 text-[11px] text-slate-500">여기에 넣은 이미지는 현재 선택된 치아 detail 아래에 연결됩니다.</div>
              <div className="mt-2 text-[11px] text-slate-500">Right click a tooth in preview to open it here and create a manual draft note.</div>
            </SectionCard>

            <SectionCard title="Independent Capture">
              <div className="mb-2 text-sm text-slate-300">Detailed Findings와 별개로 관리되는 캡처 이미지 영역입니다.</div>
              <div
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  void handleDropCapture(event);
                }}
                className="rounded-lg border border-[#4C4C4C] bg-[#3A3A3A] p-2"
              >
                <input
                  ref={captureInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={(event) => {
                    void handleAddCaptureFiles(event.target.files);
                    event.currentTarget.value = '';
                  }}
                />
                {activeCapture ? (
                  <div className="rounded-md border border-[#4C4C4C] bg-[#E5E7EB] p-2">
                    <img src={activeCapture.dataUrl} alt={activeCapture.label} className="h-56 w-full bg-white object-contain" />
                  </div>
                ) : (
                  <div className="flex h-56 items-center justify-center rounded-md border border-dashed border-[#6A6A6A] bg-[#E5E7EB] px-4 text-center text-sm text-slate-600">
                    {'Drop image here or use Add Image.'}
                  </div>
                )}
                <div className="mt-2 flex items-center justify-between gap-3">
                  <div className="min-w-0 text-xs text-slate-400">
                    <div className="truncate">{activeCapture?.label || 'No capture image'}</div>
                    <div>
                      {activeCapture
                        ? `${activeCapture.size || formatTimestamp(activeCapture.createdAt)}${activeCapture.toothId ? ` · Tooth ${activeCapture.toothId}` : ''}`
                        : 'No capture selected'}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={session.is_finalized}
                      onClick={() => captureInputRef.current?.click()}
                      className="rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-xs text-white disabled:opacity-50"
                    >
                      Add Image
                    </button>
                    <button
                      type="button"
                      disabled={!activeCapture || session.is_finalized}
                      onClick={() => activeCapture && handleRemoveCapture(activeCapture.id)}
                      className="rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-xs text-white disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
              <label className="mt-3 block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Memo</span>
                <textarea
                  rows={4}
                  value={activeCapture?.note || ''}
                  disabled={!activeCapture || session.is_finalized}
                  onChange={(event) => {
                    if (!activeCapture) return;
                    setAttachedCapturesDraft((prev) =>
                      prev.map((item) => (item.id === activeCapture.id ? { ...item, note: event.target.value } : item))
                    );
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                  placeholder={activeCapture ? 'Capture memo' : 'Add image first'}
                  className="w-full rounded-lg border border-[#4C4C4C] bg-[#E5E7EB] px-3 py-3 text-sm text-slate-900 outline-none disabled:bg-[#D1D5DB]"
                />
              </label>
              {allCaptures.length > 1 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {allCaptures.map((capture) => (
                    <button
                      key={capture.id}
                      type="button"
                      onClick={() => setSelectedCaptureId(capture.id)}
                      className={`rounded-lg border px-2 py-1 text-xs ${
                        activeCapture?.id === capture.id ? 'border-cyan-400 bg-cyan-400/10 text-cyan-200' : 'border-[#4C4C4C] bg-[#3A3A3A] text-slate-300'
                      }`}
                    >
                      {capture.label || 'Capture'}
                    </button>
                  ))}
                </div>
              ) : null}
            </SectionCard>

            <SectionCard title="Report Note">
              <textarea
                rows={4}
                value={reportNoteDraft}
                disabled={session.is_finalized}
                onChange={(event) => {
                  setReportNoteDraft(event.target.value);
                  setEditorDirty(true);
                  setSaveState('idle');
                }}
                className="w-full rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-sm outline-none"
              />
            </SectionCard>
          </div>
        </aside>

        <main className="min-w-0 overflow-y-auto bg-[#161616] p-5">
          <div className="mb-3 flex items-center justify-between gap-4">
            <div>
              <div className="text-lg font-semibold text-white">Preview</div>
              <div className="text-xs text-slate-400">Click or right click a tooth in preview to edit that draft item.</div>
            </div>
            <div className="text-right text-xs text-slate-400">
              <div>{`Viewing version ${currentVersion ?? '-'}`}</div>
              <div>{selectedVersionInfo?.status || session.report?.status || 'draft'}</div>
            </div>
          </div>
          <div className="h-[calc(100vh-420px)] min-h-[420px] overflow-hidden rounded-lg border border-[#4C4C4C] bg-white">
            <iframe
              title="Editable report preview"
              src={previewUrl}
              className="h-full w-full border-none"
              sandbox="allow-scripts allow-same-origin allow-popups allow-forms"
            />
          </div>

          <section className="mt-5 rounded-lg border border-[#4C4C4C] bg-[#2D2D2D]">
            <div className="border-b border-[#4C4C4C] px-4 py-3 text-sm font-semibold text-white">Capture For Report</div>
            <div className="p-4">
              <div className="mb-3 text-sm text-slate-300">프리뷰 아래에서 바로 추가되고, draft 저장 후 리포트에 포함되는 캡처 이미지 영역입니다.</div>
              <div
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  void handleDropCapture(event);
                }}
                className="rounded-lg border border-[#4C4C4C] bg-[#3A3A3A] p-2"
              >
                <input
                  ref={captureInputRef}
                  type="file"
                  accept="image/*"
                  multiple
                  className="hidden"
                  onChange={(event) => {
                    void handleAddCaptureFiles(event.target.files);
                    event.currentTarget.value = '';
                  }}
                />
                {activeCapture ? (
                  <div className="rounded-md border border-[#4C4C4C] bg-[#E5E7EB] p-2">
                    <img src={activeCapture.dataUrl} alt={activeCapture.label} className="h-64 w-full bg-white object-contain" />
                  </div>
                ) : (
                  <div className="flex h-64 items-center justify-center rounded-md border border-dashed border-[#6A6A6A] bg-[#E5E7EB] px-4 text-center text-sm text-slate-600">
                    Drop image here or use Add Image.
                  </div>
                )}
                <div className="mt-2 flex items-center justify-between gap-3">
                  <div className="min-w-0 text-xs text-slate-400">
                    <div className="truncate">{activeCapture?.label || 'No capture image'}</div>
                    <div>
                      {activeCapture
                        ? `${activeCapture.size || formatTimestamp(activeCapture.createdAt)}${activeCapture.toothId ? ` · Tooth ${activeCapture.toothId}` : ''}`
                        : 'No capture selected'}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      disabled={session.is_finalized}
                      onClick={() => captureInputRef.current?.click()}
                      className="rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-xs text-white disabled:opacity-50"
                    >
                      Add Image
                    </button>
                    <button
                      type="button"
                      disabled={!activeCapture || session.is_finalized}
                      onClick={() => activeCapture && handleRemoveCapture(activeCapture.id)}
                      className="rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-2 text-xs text-white disabled:opacity-50"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              </div>
              <label className="mt-3 block">
                <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Memo</span>
                <textarea
                  rows={4}
                  value={activeCapture?.note || ''}
                  disabled={!activeCapture || session.is_finalized}
                  onChange={(event) => {
                    if (!activeCapture) return;
                    setAttachedCapturesDraft((prev) =>
                      prev.map((item) => (item.id === activeCapture.id ? { ...item, note: event.target.value } : item))
                    );
                    setEditorDirty(true);
                    setSaveState('idle');
                  }}
                  placeholder={activeCapture ? 'Capture memo' : 'Add image first'}
                  className="w-full rounded-lg border border-[#4C4C4C] bg-[#E5E7EB] px-3 py-3 text-sm text-slate-900 outline-none disabled:bg-[#D1D5DB]"
                />
              </label>
              {allCaptures.length > 1 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {allCaptures.map((capture) => (
                    <button
                      key={capture.id}
                      type="button"
                      onClick={() => setSelectedCaptureId(capture.id)}
                      className={`rounded-lg border px-2 py-1 text-xs ${
                        activeCapture?.id === capture.id ? 'border-cyan-400 bg-cyan-400/10 text-cyan-200' : 'border-[#4C4C4C] bg-[#3A3A3A] text-slate-300'
                      }`}
                    >
                      {capture.label || 'Capture'}
                    </button>
                  ))}
                </div>
              ) : null}
            </div>
          </section>
        </main>
      </div>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[#4C4C4C] bg-[#414141] px-3 py-3">
      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">{label}</div>
      <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
    </div>
  );
}

function StatusPill({ state }: { state: 'idle' | 'saving' | 'saved' | 'error' }) {
  const className =
    state === 'saving'
      ? 'bg-amber-500/15 text-amber-200'
      : state === 'saved'
        ? 'bg-emerald-500/15 text-emerald-200'
        : state === 'error'
          ? 'bg-red-500/15 text-red-200'
          : 'bg-slate-800 text-slate-300';
  const label = state === 'saving' ? 'Saving' : state === 'saved' ? 'Saved' : state === 'error' ? 'Error' : 'Idle';
  return <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] ${className}`}>{label}</span>;
}

function SectionCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="mb-4 rounded-lg border border-[#4C4C4C] bg-[#333333]">
      <div className="border-b border-[#4C4C4C] px-4 py-3 text-sm font-semibold text-white">{title}</div>
      <div className="p-4">{children}</div>
    </section>
  );
}

function SimpleState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="min-h-screen bg-slate-950 px-6 py-12 text-slate-100">
      <div className="mx-auto max-w-2xl rounded-3xl border border-slate-800 bg-slate-900 p-8 shadow-2xl">
        <h1 className="text-2xl font-semibold">{title}</h1>
        <p className="mt-3 text-sm text-slate-300">{detail}</p>
        {action ? <div className="mt-6">{action}</div> : null}
      </div>
    </div>
  );
}
