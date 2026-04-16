import { useEffect, useMemo, useRef, useState } from 'react';
import {
  fetchWebReportSession,
  finalizeWebReport,
  patchWebReportOverrides,
  regenerateWebReport,
  transcribeWebReportDictation,
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

type ReportCaptureItem = {
  id: string;
  label: string;
  dataUrl: string;
  reportDataUrl?: string;
  size?: string;
  createdAt: string;
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

type DictationState = 'idle' | 'recording' | 'processing';

function getPreferredAudioMimeType() {
  if (typeof MediaRecorder === 'undefined') return '';
  const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus'];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || '';
}

function mergeReportNote(current: string, incoming: string) {
  const trimmedCurrent = current.trim();
  const trimmedIncoming = incoming.trim();
  if (!trimmedIncoming) return trimmedCurrent;
  if (!trimmedCurrent) return trimmedIncoming;
  return `${trimmedCurrent}\n\n${trimmedIncoming}`;
}

function roundToTwoDecimals(value: number) {
  return Number(Number(value || 0).toFixed(2));
}

function buildCaptureReferenceLine(capture: ReportCaptureItem) {
  return `Capture reference: ${capture.createdAt}${capture.size ? ` (${capture.size})` : ''}`;
}

function buildToothNoteSuggestions(toothId: string, form: ToothReviewForm) {
  if (!toothId) return [];

  const suggestions: string[] = [];

  if (form.caries) suggestions.push('Caries');
  if (form.periapical) suggestions.push('Periapical');
  if (form.missing) suggestions.push('Missing');
  if (form.implant) suggestions.push('Implant');
  if (form.crown) suggestions.push('Crown');
  if (form.filling) suggestions.push('Filling');
  if (form.bone_loss_level >= 3) suggestions.push('Bone loss');

  return suggestions;
}

function buildToothNoteAutocompletePhrases(form: ToothReviewForm) {
  const phrases = [
    'periapical lesion',
    'periodontal bone loss',
    'caries lesion',
    'missing tooth',
    'implant fixture',
    'crown restoration',
    'filling restoration',
    'bone loss',
    'furcation involvement',
    'widened periodontal ligament space',
    'lamina dura disruption',
    'pericoronal radiolucency',
    'root canal treated tooth',
    'calculus deposition',
    'alveolar bone resorption',
    'radiolucent lesion',
    'radiopaque lesion',
  ];

  if (form.periapical) phrases.unshift('periapical lesion');
  if (form.caries) phrases.unshift('caries lesion');
  if (form.missing) phrases.unshift('missing tooth');
  if (form.implant) phrases.unshift('implant fixture');
  if (form.crown) phrases.unshift('crown restoration');
  if (form.filling) phrases.unshift('filling restoration');
  if (form.bone_loss_level > 0 || form.bone_loss_pct > 0) phrases.unshift('bone loss');

  return [...new Set(phrases)];
}

function getInlinePhraseAutocomplete(current: string, suggestions: string[], caretPosition: number) {
  const beforeCaret = current.slice(0, caretPosition);
  const afterCaret = current.slice(caretPosition);
  const currentLine = beforeCaret.split('\n').pop() || '';
  const trailingOnLine = afterCaret.split('\n')[0] || '';
  if (trailingOnLine.length > 0) return null;
  const fragmentMatch = currentLine.match(/([a-zA-Z][a-zA-Z ]*)$/);
  const fragment = fragmentMatch?.[1]?.trimStart() || '';
  if (fragment.length < 2) return null;

  const match = suggestions.find((suggestion) => {
    const lowerSuggestion = suggestion.toLowerCase();
    const lowerFragment = fragment.toLowerCase();
    return lowerSuggestion.startsWith(lowerFragment) && lowerSuggestion !== lowerFragment;
  });
  if (!match) return null;

  const remainder = match.slice(fragment.length);
  if (!remainder) return null;

  return { ghostText: remainder, applyText: remainder, prefixText: beforeCaret, fullText: match };
}

export function WebReportDrawer({
  sessionId,
  selectedToothId: linkedToothId,
  availableCaptures = [],
  onClose,
  open = true,
  layout = 'modal',
  isInactive = false,
  positionMode = 'fixed',
}: {
  sessionId: string;
  selectedToothId?: string | null;
  availableCaptures?: ReportCaptureItem[];
  onClose: () => void;
  open?: boolean;
  layout?: 'modal' | 'dock';
  isInactive?: boolean;
  positionMode?: 'fixed' | 'absolute';
}) {
  const [session, setSession] = useState<WebReportSessionResponse['session'] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [actionState, setActionState] = useState<'idle' | 'regenerating' | 'finalizing'>('idle');
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');
  const [activeTab, setActiveTab] = useState<'keywords' | 'review' | 'report' | 'guide'>('keywords');
  const [selectedToothId, setSelectedToothId] = useState('');
  const [reviewForm, setReviewForm] = useState<ToothReviewForm>(emptyForm);
  const [reportNoteDraft, setReportNoteDraft] = useState('');
  const [attachedCapturesDraft, setAttachedCapturesDraft] = useState<ReportCaptureItem[]>([]);
  const [editorDirty, setEditorDirty] = useState(false);
  const lastSavedPayloadRef = useRef('');
  const toothNoteTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [toothNoteCaret, setToothNoteCaret] = useState(0);
  const [dictationState, setDictationState] = useState<DictationState>('idle');
  const [dictationError, setDictationError] = useState<string | null>(null);
  const [dictationTranscript, setDictationTranscript] = useState('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

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
  const keywordGroups = useMemo(() => {
    const groups: Record<'Caries' | 'Missing' | 'Bone loss', string[]> = {
      Caries: [],
      Missing: [],
      'Bone loss': [],
    };

    keywords.forEach((keyword) => {
      const normalized = String(keyword || '').replace(/\r\n/g, '\n');
      const colonIndex = normalized.indexOf(':');
      const rawLabel = colonIndex >= 0 ? normalized.slice(0, colonIndex).trim() : normalized.trim();
      const values =
        colonIndex >= 0
          ? normalized.slice(colonIndex + 1).match(/#\d+/g) || []
          : normalized.match(/#\d+/g) || [];

      if (/caries/i.test(rawLabel)) groups.Caries = values;
      if (/missing/i.test(rawLabel)) groups.Missing = values;
      if (/bone/i.test(rawLabel)) groups['Bone loss'] = values;
    });

    return groups;
  }, [keywords]);
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
    const toothOverride = session.doctor_overrides?.teeth?.[selectedToothId] || {};
    const hasOverrideField = (field: string) => Object.prototype.hasOwnProperty.call(toothOverride, field);
    const baseForm = {
      caries: hasOverrideField('caries') ? Boolean(toothOverride.caries) : Boolean(selectedTooth.caries),
      periapical: hasOverrideField('periapical') ? Boolean(toothOverride.periapical) : Boolean(selectedTooth.periapical),
      missing: hasOverrideField('missing') ? Boolean(toothOverride.missing) : Boolean(selectedTooth.missing),
      implant: hasOverrideField('implant') ? Boolean(toothOverride.implant) : Boolean(selectedTooth.implant),
      crown: hasOverrideField('crown') ? Boolean(toothOverride.crown) : Boolean(selectedTooth.crown),
      filling: hasOverrideField('filling') ? Boolean(toothOverride.filling) : Boolean(selectedTooth.filling),
      bone_loss_level: hasOverrideField('bone_loss_level')
        ? Number(toothOverride.bone_loss_level || 0)
        : Number(selectedTooth.bone_loss_level || 0),
      bone_loss_pct: hasOverrideField('bone_loss_pct')
        ? roundToTwoDecimals(Number(toothOverride.bone_loss_pct || 0))
        : roundToTwoDecimals(Number(selectedTooth.bone_loss_pct || 0)),
      note: '',
    };
    const note = (toothOverride.note as string | undefined) ?? selectedTooth.note ?? '';
    const suggestion = buildToothNoteSuggestions(selectedToothId, baseForm)[0] || '';
    const nextNote = note || suggestion;
    setReviewForm({ ...baseForm, note: nextNote });
    setToothNoteCaret(nextNote.length);
    setReportNoteDraft((session.doctor_overrides?.report_note as string | undefined) || '');
    setAttachedCapturesDraft((session.doctor_overrides?.attached_captures as ReportCaptureItem[] | undefined) || []);
  }, [session, selectedTooth, selectedToothId, editorDirty]);

  const currentPayload = useMemo(() => {
    return JSON.stringify({
      tooth_overrides: selectedToothId ? { [selectedToothId]: reviewForm } : {},
      report_note: reportNoteDraft,
      attached_captures: attachedCapturesDraft,
    });
  }, [selectedToothId, reviewForm, reportNoteDraft, attachedCapturesDraft]);

  const updateReportNoteDraft = (value: string) => {
    setReportNoteDraft(value);
    setEditorDirty(true);
    setSaveState('idle');
  };

  const attachCaptureToReport = (capture: ReportCaptureItem) => {
    setAttachedCapturesDraft((prev) => {
      if (prev.some((item) => item.id === capture.id)) return prev;
      return [
        ...prev,
        {
          ...capture,
          dataUrl: capture.reportDataUrl || capture.dataUrl,
        },
      ];
    });
    const referenceLine = buildCaptureReferenceLine(capture);
    setReportNoteDraft((prev) => {
      if (prev.includes(referenceLine)) return prev;
      return mergeReportNote(prev, referenceLine);
    });
    setEditorDirty(true);
    setSaveState('idle');
  };

  const removeAttachedCapture = (captureId: string) => {
    setAttachedCapturesDraft((prev) => prev.filter((item) => item.id !== captureId));
    setEditorDirty(true);
    setSaveState('idle');
  };

  const updateToothNoteDraft = (value: string) => {
    setReviewForm((prev) => ({ ...prev, note: value }));
    setEditorDirty(true);
    setSaveState('idle');
  };

  const toothNoteAutocomplete = useMemo(
    () => getInlinePhraseAutocomplete(reviewForm.note, buildToothNoteAutocompletePhrases(reviewForm), toothNoteCaret),
    [
      reviewForm.note,
      toothNoteCaret,
      reviewForm.caries,
      reviewForm.periapical,
      reviewForm.missing,
      reviewForm.implant,
      reviewForm.crown,
      reviewForm.filling,
      reviewForm.bone_loss_level,
      reviewForm.bone_loss_pct,
    ]
  );

  const acceptToothNoteAutocomplete = () => {
    if (!toothNoteAutocomplete) return;
    const nextValue =
      reviewForm.note.slice(0, toothNoteCaret) +
      toothNoteAutocomplete.applyText +
      reviewForm.note.slice(toothNoteCaret);
    const nextCaret = toothNoteCaret + toothNoteAutocomplete.applyText.length;
    updateToothNoteDraft(nextValue);
    setToothNoteCaret(nextCaret);
    requestAnimationFrame(() => {
      toothNoteTextareaRef.current?.focus();
      toothNoteTextareaRef.current?.setSelectionRange(nextCaret, nextCaret);
    });
  };

  const persistOverrides = async () => {
    if (!sessionId || !session || session.is_finalized) return true;
    if (currentPayload === lastSavedPayloadRef.current) {
      setEditorDirty(false);
      setSaveState('saved');
      return true;
    }

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
      setError(err?.message || 'Autosave failed');
      setSaveState('error');
      return false;
    }
  };

  useEffect(() => {
    if (!sessionId || !session || !editorDirty || session.is_finalized) return;
    if (currentPayload === lastSavedPayloadRef.current) {
      setEditorDirty(false);
      setSaveState('saved');
      return;
    }

    const timer = window.setTimeout(async () => {
      await persistOverrides();
    }, 800);

    return () => {
      window.clearTimeout(timer);
    };
  }, [attachedCapturesDraft, currentPayload, editorDirty, selectedToothId, session, sessionId]);

  const handleRegenerate = async () => {
    setActionState('regenerating');
    setError(null);
    try {
      if (editorDirty) {
        const saved = await persistOverrides();
        if (!saved) return;
      }
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
    const finalWindow = typeof window !== 'undefined' ? window.open('about:blank', '_blank') : null;
    setActionState('finalizing');
    setError(null);
    try {
      if (editorDirty) {
        const saved = await persistOverrides();
        if (!saved) {
          finalWindow?.close();
          return;
        }
      }
      await finalizeWebReport(sessionId);
      const refreshed = await fetchWebReportSession(sessionId);
      setSession(refreshed.session || null);
      if (finalWindow) {
        finalWindow.location.href = reportPageUrl;
      } else if (typeof window !== 'undefined') {
        window.open(reportPageUrl, '_blank');
      }
    } catch (err: any) {
      finalWindow?.close();
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

  const handleOpenReportActions = async () => {
    setActiveTab('report');
    await handleRegenerate();
  };

  const stopActiveStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  useEffect(() => stopActiveStream, []);

  /*
  Voice dictation is intentionally disabled in the current Renew report modal flow.
  Keep the previous implementation here so it can be restored later if needed.

  const handleStartDictation = async () => {
    if (dictationState !== 'idle' || session?.is_finalized) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      setDictationError('Microphone recording is not available in this browser.');
      return;
    }

    try {
      setDictationError(null);
      setDictationTranscript('');
      recordedChunksRef.current = [];

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const mimeType = getPreferredAudioMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

      recorder.addEventListener('dataavailable', (event) => {
        if (event.data && event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      });

      recorder.addEventListener('stop', async () => {
        const blobType = recorder.mimeType || mimeType || 'audio/webm';
        const audioBlob = new Blob(recordedChunksRef.current, { type: blobType });
        stopActiveStream();

        if (!audioBlob.size) {
          setDictationState('idle');
          setDictationError('No audio was recorded.');
          return;
        }

        setDictationState('processing');
        try {
          const response = await transcribeWebReportDictation(sessionId, audioBlob, blobType);
          setDictationTranscript(response.transcript || '');
          setReportNoteDraft((prev) => mergeReportNote(prev, response.report_note_text || ''));
          setEditorDirty(true);
          setSaveState('idle');
          setActiveTab('review');
        } catch (err: any) {
          setDictationError(err?.message || 'Dictation processing failed.');
        } finally {
          setDictationState('idle');
          mediaRecorderRef.current = null;
          recordedChunksRef.current = [];
        }
      });

      mediaRecorderRef.current = recorder;
      recorder.start();
      setDictationState('recording');
    } catch (err: any) {
      stopActiveStream();
      setDictationState('idle');
      setDictationError(err?.message || 'Microphone permission failed.');
    }
  };

  const handleStopDictation = () => {
    if (dictationState !== 'recording') return;
    mediaRecorderRef.current?.stop();
  };

  const renderDictationControl = () => {
    if (session?.is_finalized) return null;

    const isRecording = dictationState === 'recording';
    const isProcessing = dictationState === 'processing';

    return (
      <button
        type="button"
        onClick={isRecording ? handleStopDictation : handleStartDictation}
        disabled={isProcessing}
        className="inline-flex items-center justify-center"
        style={{
          width: 42,
          height: 42,
          border: '1px solid #7A7A7A',
          background: '#6E6E6E',
          color: '#FFFFFF',
          borderRadius: 0,
          opacity: isProcessing ? 0.55 : 1,
        }}
        title={isRecording ? 'Stop voice input' : 'Start voice input'}
      >
        {isRecording ? <Square size={16} /> : <Mic size={18} />}
      </button>
    );
  };
  */

  const handleStartDictation = async () => {};
  const handleStopDictation = () => {};
  const renderDictationControl = () => null;


  const reportUrl = `/api/web_report/session/${sessionId}/report`;
  const reportPageUrl = reportUrl;
  const hasPdf = Boolean(session?.report?.pdf_path);
  const pdfUrl = `/api/web_report/session/${sessionId}/report/pdf`;
  const statusLabel = session?.is_finalized ? 'Final' : 'Draft';
  const isDock = layout === 'dock';
  const dockBodyShellStyle = isDock
    ? { margin: '0 6px 6px', background: '#2B2B2B', border: '1px solid #3A3A3A', borderTop: 'none' }
    : undefined;
  const dockSectionTitleStyle = { color: '#FFFFFF', fontSize: 14, marginBottom: 8 } as const;
  const dockMutedLabelStyle = { color: '#BFBFBF', fontSize: 12, marginBottom: 6 } as const;
  const dockFieldStyle = {
    width: '100%',
    background: '#1E1E1E',
    border: '1px solid #CFCFCF',
    color: '#FFFFFF',
    padding: '10px 12px',
    fontSize: 14,
    outline: 'none',
  } as const;
  const dockCardStyle = {
    borderTop: '1px solid #5A5A5A',
    borderBottom: '1px solid #5A5A5A',
    padding: '14px 14px 16px',
  } as const;
  const rootClassName = isDock
    ? `${positionMode === 'absolute' ? 'absolute' : 'fixed'} flex flex-col overflow-hidden border text-slate-100`
    : 'fixed bottom-3 left-3 right-3 top-20 z-[220] flex flex-col overflow-hidden rounded-[30px] border text-slate-100 backdrop-blur-xl md:bottom-5 md:left-auto md:right-5 md:top-24 md:w-[540px]';
  const rootStyle = isDock
      ? {
        position: positionMode,
        right: 24,
        bottom: 112,
        top: 'auto',
        left: 'auto',
        zIndex: positionMode === 'absolute' ? 40 : 2147482900,
        width: 420,
        maxWidth: positionMode === 'absolute' ? 'calc(100% - 24px)' : 'calc(100vw - 2rem)',
        height: 498,
        background: '#4A4F55',
        borderColor: '#5A5A5A',
        boxShadow: '0 8px 24px rgba(0,0,0,0.22)',
        color: '#FFFFFF',
        opacity: open ? 1 : 0,
        transform: open ? 'translateY(0) scale(1)' : 'translateY(12px) scale(0.95)',
        pointerEvents: open ? ('auto' as const) : ('none' as const),
        transition: 'opacity 200ms ease, transform 200ms ease',
        colorScheme: 'dark' as const,
      }
    : {
        background: 'linear-gradient(180deg, rgba(12, 28, 52, 0.98), rgba(8, 17, 34, 0.98))',
        borderColor: 'rgba(103, 232, 249, 0.35)',
        boxShadow: '0 36px 120px rgba(3, 8, 20, 0.62)',
        color: '#E2E8F0',
        colorScheme: 'dark' as const,
      };


  return (
    <div className={`${rootClassName} ${isInactive ? 'opacity-40 pointer-events-none' : ''}`} style={rootStyle}>
      {!isDock && (
        <>
          <div className={`pointer-events-none absolute -top-5 left-1/2 -translate-x-1/2 ${isDock ? 'right-14 md:left-auto md:translate-x-0' : 'md:left-auto md:right-14 md:translate-x-0'}`}>
            <div className="h-12 w-44 rounded-full bg-gradient-to-r from-cyan-400 via-sky-400 to-blue-500 opacity-75 blur-[10px]" />
          </div>
          <div className="absolute left-5 top-5 rounded-full border border-cyan-300/30 bg-[#09172f] px-4 py-1 text-[10px] font-semibold uppercase tracking-[0.28em] text-cyan-200 shadow-[0_10px_30px_rgba(34,211,238,0.22)]">
            Report Modal
          </div>
        </>
      )}
      {isDock && (
        <div
          style={{
            position: 'absolute',
            inset: '0 0 auto 0',
            height: 30,
            background: '#4A4F55',
            borderBottom: '1px solid #5A5A5A',
            zIndex: 2,
          }}
        >
          <div
            style={{
              position: 'absolute',
              left: 8,
              top: 5,
              color: '#FFFFFF',
              fontSize: 18,
              fontWeight: 400,
            }}
          >
            Report
          </div>
        </div>
      )}
      <button
        type="button"
        onClick={onClose}
        className={isDock ? 'absolute right-6 top-6 inline-flex items-center justify-center border text-white' : 'absolute right-5 top-5 inline-flex h-8 w-auto shrink-0 items-center justify-center rounded-full border-2 border-white/90 bg-cyan-400 px-3 py-1.5 text-[13px] font-semibold leading-none text-slate-950 shadow-[0_10px_24px_rgba(0,0,0,0.22)] hover:bg-cyan-300'}
        style={{
          position: 'absolute',
          top: isDock ? 4 : 20,
          right: isDock ? 8 : 20,
          width: isDock ? 22 : 'fit-content',
          height: isDock ? 22 : 32,
          minWidth: isDock ? 22 : 0,
          margin: 0,
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          whiteSpace: 'nowrap',
          zIndex: 2,
          borderRadius: isDock ? 0 : 999,
          padding: isDock ? 0 : undefined,
          background: isDock ? '#163D73' : undefined,
          borderColor: isDock ? '#FFFFFF' : undefined,
          fontSize: isDock ? 12 : undefined,
          fontWeight: isDock ? 400 : undefined,
        }}
      >
        {isDock ? 'X' : 'Hide'}
      </button>
      <div className="hidden" />

      <div
        className={`flex items-center justify-between ${isDock ? 'bg-transparent px-6 pb-0 pt-20' : 'border-b border-cyan-300/15 bg-[linear-gradient(180deg,rgba(27,78,113,0.34),rgba(7,16,31,0.12))] px-5 pb-4 pt-14 md:px-6'}`}
        style={isDock ? {
          margin: '40px 6px 0',
          padding: '16px 14px 12px',
          background: '#2B2B2B',
          border: '1px solid #3A3A3A',
          borderBottom: 'none',
        } : undefined}
      >
        <div className="flex min-w-0 items-center gap-4">
          <div className={`flex items-center justify-center rounded-2xl ${isDock ? 'hidden' : 'bg-cyan-500/15 text-cyan-200 shadow-[inset_0_0_0_1px_rgba(103,232,249,0.15)] h-12 w-12'}`}>
            <span className="text-sm font-bold tracking-[0.18em]">AI</span>
          </div>
          <div className="min-w-0">
            <p className={`${isDock ? 'mb-2 text-[16px] font-medium text-white' : 'text-[11px] font-semibold uppercase tracking-[0.26em] text-cyan-300'}`}>AI Note</p>
            <div>
              <div>
                <p className={`mt-1 truncate ${isDock ? 'text-[13px] text-[#BFBFBF]' : 'text-[15px] text-slate-200'}`}>Session {sessionId.slice(0, 8)} - {statusLabel}</p>
                <p className={`mt-1 ${isDock ? 'text-[12px] leading-6 text-[#9A9A9A]' : 'text-[13px] text-slate-400'}`}>
              
                </p>
              </div>
            </div>
          </div>
        </div>
        <div className="ml-3 flex items-center gap-2">
          <a
            href={reportPageUrl}
            target="_blank"
            rel="noreferrer"
            className={isDock ? 'hidden' : 'rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-[13px] text-slate-100 hover:bg-white/10'}
          >
            Open Full
          </a>
        </div>
      </div>

      <div
        className={`flex items-center gap-0 border-b border-white/10 ${isDock ? 'bg-[#2F2E2D] px-6 py-4' : 'bg-black/10 px-6 py-3'}`}
        style={isDock ? {
          margin: '0 6px',
          padding: '0 14px 12px',
          background: '#2B2B2B',
          borderLeft: '1px solid #3A3A3A',
          borderRight: '1px solid #3A3A3A',
          borderBottom: '1px solid #5A5A5A',
        } : undefined}
      >
        {(['keywords', 'review', 'report', 'guide'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`${isDock ? 'flex-1 border px-3 py-1 text-[12px] font-light' : 'rounded-full px-4 py-2 text-[13px] font-medium transition'} ${
              activeTab === tab
                ? isDock ? 'text-white' : 'bg-cyan-400 text-slate-950 shadow-[0_10px_25px_rgba(34,211,238,0.22)]'
                : isDock ? 'text-[#F1F1F1]' : 'bg-white/5 text-slate-200 hover:bg-white/10'
            }`}
            style={
              isDock
                ? {
                    borderRadius: 0,
                    borderColor: '#6F6F6F',
                    background: activeTab === tab ? '#8E8E8E' : '#9A9A9A',
                    boxShadow: activeTab === tab ? 'inset 0 3px 0 #00C0F3' : 'none',
                    transform: activeTab === tab ? 'translateY(-1px)' : 'none',
                    transition: 'background 140ms ease, box-shadow 140ms ease, transform 140ms ease',
                  }
                : undefined
            }
          >
            {tab === 'keywords' ? 'Keywords' : tab === 'review' ? 'Review' : tab === 'report' ? 'Report' : 'Guide'}
          </button>
        ))}
      </div>

      {error && (
        <div className={`${isDock ? 'mx-4 mt-3' : 'mx-6 mt-4'} rounded-2xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-[15px] text-red-200`}>
          {error}
        </div>
      )}

      {dictationError && (
        <div className={`${isDock ? 'mx-4 mt-3' : 'mx-6 mt-4'} rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-[15px] text-amber-100`}>
          {dictationError}
        </div>
      )}

      {activeTab === 'report' ? (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className={`flex-1 overflow-y-auto ${isDock ? 'px-0 py-0' : 'px-5 py-5 md:px-6'}`} style={dockBodyShellStyle}>
          {isDock ? (
            <div style={{ padding: '18px 18px 20px' }}>
              <div style={dockSectionTitleStyle}>Report</div>
              <div style={{ ...dockCardStyle, display: 'grid', gap: 12 }}>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                  <div>
                    <div style={dockMutedLabelStyle}>Current Version</div>
                    <div style={{ color: '#E5E5E5', fontSize: 13 }}>{session?.report?.version ?? 1}</div>
                  </div>
                  <div>
                    <div style={dockMutedLabelStyle}>Status</div>
                    <div style={{ color: '#E5E5E5', fontSize: 13 }}>{statusLabel}</div>
                  </div>
                </div>
                <div>
                  <div style={dockMutedLabelStyle}>Document</div>
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    <a
                      href={reportPageUrl}
                      target="_blank"
                      rel="noreferrer"
                      style={{ ...dockFieldStyle, width: 'auto', textDecoration: 'none', padding: '6px 12px', background: '#8E8E8E', borderColor: '#6F6F6F' }}
                    >
                      Open Report
                    </a>
                    <a
                      href={reportUrl}
                      target="_blank"
                      rel="noreferrer"
                      style={{ ...dockFieldStyle, width: 'auto', textDecoration: 'none', padding: '6px 12px' }}
                    >
                      HTML
                    </a>
                    {hasPdf ? (
                      <a
                        href={pdfUrl}
                        target="_blank"
                        rel="noreferrer"
                        style={{ ...dockFieldStyle, width: 'auto', textDecoration: 'none', padding: '6px 12px' }}
                      >
                        PDF
                      </a>
                    ) : null}
                  </div>
                </div>
                <div>
                  <div style={dockMutedLabelStyle}>Attached Captures</div>
                  <div style={{ color: '#9A9A9A', fontSize: 12 }}>
                    {attachedCapturesDraft.length ? `${attachedCapturesDraft.length} capture(s) attached` : 'No captures attached'}
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 18, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
                <div>
                  <div style={{ color: '#FFFFFF', fontSize: 14 }}>Report Actions</div>
                  <div style={{ color: '#9A9A9A', fontSize: 11, marginTop: 4 }}>Finalize after report check.</div>
                </div>
                <button
                  onClick={handleFinalize}
                  disabled={actionState !== 'idle' || Boolean(session?.is_finalized)}
                  style={{
                    ...dockFieldStyle,
                    width: 'auto',
                    padding: '8px 16px',
                    background: '#8E8E8E',
                    borderColor: '#6F6F6F',
                    color: '#FFFFFF',
                  }}
                >
                  {session?.is_finalized ? 'Finalized' : actionState === 'finalizing' ? 'Finalizing...' : 'Finalize'}
                </button>
              </div>
            </div>
          ) : (
          <>
          <div className="rounded-[28px] border border-cyan-400/15 bg-[linear-gradient(135deg,rgba(34,211,238,0.12),rgba(14,165,233,0.04))] p-5">
            {/* <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Report Workspace</p>
            <p className="mt-2 text-[15px] leading-6 text-slate-200">
              Review findings in this floating panel, then open the full HTML document in its own page when you need the final layout.
            </p>
            <br /> */}
          </div>

          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Current Version</p>
              <div className="mt-3 flex items-end gap-3">
                <span className="text=[2rem] font-semibold tracking-[-0.04em] text-white">{session?.report?.version ?? 1}</span>
                <span className="mb-1 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-cyan-200">
                  {statusLabel}
                </span>
                <br/>
              </div>
              <p className="mt-3 text-[13px] text-slate-300">
                <br/>
                {/* Regenerate after review changes, then open the full report page to inspect the actual document layout. */}
              </p>
              <br/>
            </div>

            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Open Document</p>
              <br/>
              <div className="mt-4 flex flex-wrap gap-2">
                <br/>
                <a
                  href={reportPageUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full bg-cyan-400 px-3 py-1.5 text-[8px] font-semibold text-slate-950 shadow-[0_10px_25px_rgba(34,211,238,0.22)]"
                >
                  Open Report
                </a>
                <a
                  href={reportUrl}
                  target="_blank"
                  rel="noreferrer"
                  className="rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-[8px] text-slate-100 hover:bg-white/10"
                >
                  HTML
                </a>
                {hasPdf ? (
                  <a
                    href={pdfUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-slate-700 bg-white/5 px-3 py-1.5 text-[8px] text-slate-100 hover:bg-white/10"
                  >
                    PDF
                  </a>
                ) : null}
              </div>
              <br/>
            </div>
          </div>

          <div className="mt-4 rounded-[24px] border border-white/10 bg-white/5 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Attached Captures</p>
            {attachedCapturesDraft.length ? (
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {attachedCapturesDraft.map((capture) => (
                  <div key={capture.id} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                    <img
                      src={capture.dataUrl}
                      alt={capture.label}
                      className="h-[110px] w-full rounded-xl border border-white/10 bg-black object-contain"
                    />
                    <div className="mt-2 flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <p className="truncate text-[13px] font-medium text-slate-100">{capture.createdAt}</p>
                        <p className="text-[11px] text-slate-500">{capture.size || capture.label}</p>
                      </div>
                      <button
                        type="button"
                        onClick={() => removeAttachedCapture(capture.id)}
                        className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-slate-300 hover:bg-white/10"
                      >
                        Remove
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-[13px] text-slate-500">No captures attached to this report yet.</p>
            )}
          </div>

          {/* 
          <div className="mt-4 rounded-[24px] border border-white/10 bg-white/5 p-5">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Available Captures</p>
            {availableCaptures.length ? (
              <div className="mt-4 grid gap-3 md:grid-cols-2">
                {availableCaptures.map((capture) => {
                  const alreadyAttached = attachedCapturesDraft.some((item) => item.id === capture.id);
                  return (
                    <div key={capture.id} className="rounded-2xl border border-white/10 bg-black/20 p-3">
                      <img
                        src={capture.dataUrl}
                        alt={capture.label}
                        className="h-[96px] w-full rounded-xl border border-white/10 bg-black object-contain"
                      />
                      <div className="mt-2 flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-[13px] font-medium text-slate-100">{capture.createdAt}</p>
                          <p className="text-[11px] text-slate-500">{capture.size || capture.label}</p>
                        </div>
                        <button
                          type="button"
                          disabled={alreadyAttached}
                          onClick={() => attachCaptureToReport(capture)}
                          className="rounded-full border border-white/10 bg-cyan-400 px-2.5 py-1 text-[11px] font-semibold text-slate-950 disabled:bg-white/5 disabled:text-slate-500"
                        >
                          {alreadyAttached ? 'Added' : 'Add'}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="mt-3 text-[13px] text-slate-500">No captures available from the current chart.</p>
            )}
          </div>
          */}
          </>
          )}
          </div>
          {!isDock && (
          <div className={`border-t border-white/10 bg-black/10 px-5 py-2.5 md:px-6`}>
            <div className="flex items-center justify-between gap-3 rounded-[18px] border border-white/10 bg-white/5 px-4 py-2.5">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Report Actions</p>
                <p className="mt-0.5 text-[11px] leading-4 text-slate-400">Finalize after report check.</p>
              </div>
              <button
                onClick={handleFinalize}
                disabled={actionState !== 'idle' || Boolean(session?.is_finalized)}
                className="rounded-full bg-emerald-500 px-4 py-1.5 text-[13px] font-semibold text-slate-950 shadow-[0_10px_24px_rgba(0,0,0,0.22)] disabled:opacity-100"
                style={{ border: '2px solid rgba(255,255,255,0.95)' }}
              >
                {session?.is_finalized ? 'Finalized' : actionState === 'finalizing' ? 'Finalizing...' : 'Finalize'}
              </button>
            </div>
          </div>
          )}
        </div>
      ) : activeTab === 'keywords' ? (
        <div className={`flex-1 overflow-y-auto ${isDock ? 'px-0 py-0' : 'px-5 py-5 md:px-6'}`} style={isDock ? { margin: '0 6px 6px', background: '#2B2B2B', border: '1px solid #3A3A3A', borderTop: 'none' } : undefined}>
          {isDock ? (
            <div style={{ padding: '18px 18px 20px' }}>
              <div style={{ color: '#FFFFFF', fontSize: 14, marginBottom: 16 }}>Model Findings</div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr 1fr',
                  gap: 0,
                  borderTop: '1px solid #5A5A5A',
                  borderBottom: '1px solid #5A5A5A',
                }}
              >
                {(['Caries', 'Missing', 'Bone loss'] as const).map((title, index) => (
                  <div
                    key={title}
                    style={{
                      padding: '16px 14px 18px',
                      borderRight: index < 2 ? '1px solid #5A5A5A' : 'none',
                      minHeight: 168,
                    }}
                  >
                    <div style={{ color: '#BFBFBF', fontSize: 12, marginBottom: 14 }}>{title}:</div>
                    <div style={{ display: 'grid', gap: 10 }}>
                      {(keywordGroups[title].length ? keywordGroups[title] : ['-']).map((value) => (
                        <div key={`${title}-${value}`} style={{ color: '#E5E5E5', fontSize: 12 }}>
                          {value}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 18, color: '#FFFFFF', fontSize: 14, marginBottom: 8 }}>Report Note</div>
              <div
                style={{
                  border: '1px solid #CFCFCF',
                  background: '#1E1E1E',
                  padding: 12,
                }}
              >
                <div className="flex items-end gap-3">
                  <textarea
                    rows={5}
                    value={reportNoteDraft}
                    disabled={session?.is_finalized}
                    onChange={(event) => updateReportNoteDraft(event.target.value)}
                    className="flex-1 bg-transparent text-[14px] leading-6 text-white outline-none placeholder:text-[#9A9A9A]"
                    style={{ caretColor: '#ffffff', resize: 'none' }}
                    placeholder="Add a clinician note for the report draft"
                  />
                  {renderDictationControl()}
                </div>
              </div>
              {dictationTranscript && (
                <p style={{ marginTop: 8, color: '#9A9A9A', fontSize: 11 }}>
                  Transcript captured and summarized into SOAP format.
                </p>
              )}
            </div>
          ) : (
            <>
              <div className="rounded-[26px] border border-cyan-400/15 bg-[linear-gradient(135deg,rgba(34,211,238,0.12),rgba(14,165,233,0.04))] p-5">
                <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Model Findings</p>
                <p className="mt-2 text-sm leading-6 text-slate-300" />
              </div>

              <div className="mt-4 flex flex-wrap gap-2">
                {keywords.map((keyword) => {
                  const normalizedKeyword = String(keyword || '').replace(/\r\n/g, '\n');
                  const newlineParts = normalizedKeyword.split('\n').filter(Boolean);
                  const colonIndex = normalizedKeyword.indexOf(':');
                  const extractedLabel = colonIndex >= 0 ? normalizedKeyword.slice(0, colonIndex + 1).trim() : newlineParts[0] || normalizedKeyword.trim();
                  const extractedValues = colonIndex >= 0
                    ? (normalizedKeyword.slice(colonIndex + 1).match(/#\d+/g) || [])
                    : newlineParts.slice(1).flatMap((line) => line.match(/#\d+/g) || []);
                  const lines = [extractedLabel, ...extractedValues].filter(Boolean);

                  return (
                    <div
                      key={keyword}
                      className="rounded-2xl border border-white/10 bg-white/5 px-3 py-2 text-[13px] font-medium text-slate-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] text-center"
                    >
                      <div className="flex flex-col items-center">
                        {lines.map((line, index) => (
                          <div key={`${keyword}-${line}-${index}`} className={index === 0 ? 'block' : 'mt-0.5 block'}>
                            {line}
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              <div className="mt-5 rounded-[24px] border border-white/10 bg-white/5 p-4">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Report Note</p>
                  {renderDictationControl()}
                </div>
                <textarea
                  rows={4}
                  value={reportNoteDraft}
                  disabled={session?.is_finalized}
                  onChange={(event) => updateReportNoteDraft(event.target.value)}
                  className="mt-3 w-full rounded-2xl bg-slate-950 px-3 py-2 text-[15px] leading-6 text-white outline-none placeholder:text-slate-400"
                  style={{ border: '1px solid rgba(255,255,255,0.22)', caretColor: '#ffffff' }}
                  placeholder="Add a clinician note for the report draft"
                />
                {dictationTranscript && (
                  <p className="mt-2 text-[12px] leading-5 text-slate-500">
                    Transcript captured and summarized into SOAP format.
                  </p>
                )}
              </div>
            </>
          )}
        </div>
      ) : activeTab === 'review' ? (
        <div className="flex flex-1 flex-col overflow-hidden">
          <div className={`flex-1 overflow-y-auto ${isDock ? 'px-0 py-0' : 'px-5 py-5 md:px-6'}`} style={dockBodyShellStyle}>
            {isDock ? (
              <div style={{ padding: '18px 18px 20px' }}>
                <div style={dockSectionTitleStyle}>Tooth</div>
                <select
                  value={selectedToothId}
                  onChange={(event) => {
                    setSelectedToothId(event.target.value);
                    setEditorDirty(false);
                    setSaveState('idle');
                  }}
                  style={{ ...dockFieldStyle, marginBottom: 14 }}
                >
                  {teeth.map((tooth) => (
                    <option key={tooth.tooth_label} value={String(tooth.tooth_label)}>
                      Tooth {tooth.tooth_label}
                    </option>
                  ))}
                </select>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
                  <label>
                    <div style={dockMutedLabelStyle}>Bone Level</div>
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
                      style={dockFieldStyle}
                    />
                  </label>
                  <label>
                    <div style={dockMutedLabelStyle}>Bone %</div>
                    <input
                      type="number"
                      min={0}
                      max={100}
                      step={0.01}
                      value={reviewForm.bone_loss_pct}
                      disabled={session?.is_finalized}
                      onChange={(event) => {
                        const roundedValue = roundToTwoDecimals(Number(event.target.value || 0));
                        setReviewForm((prev) => ({ ...prev, bone_loss_pct: roundedValue }));
                        setEditorDirty(true);
                        setSaveState('idle');
                      }}
                      style={dockFieldStyle}
                    />
                  </label>
                </div>

                <div style={dockSectionTitleStyle}>Tooth Note</div>
                <textarea
                  ref={toothNoteTextareaRef}
                  rows={4}
                  value={reviewForm.note}
                  disabled={session?.is_finalized}
                  onChange={(event) => {
                    updateToothNoteDraft(event.target.value);
                    setToothNoteCaret(event.target.selectionStart ?? event.target.value.length);
                  }}
                  onClick={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                  onKeyUp={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                  onSelect={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                  onKeyDown={(event) => {
                    if (event.key === 'Tab' && toothNoteAutocomplete?.ghostText) {
                      event.preventDefault();
                      event.stopPropagation();
                      acceptToothNoteAutocomplete();
                    }
                  }}
                  style={{ ...dockFieldStyle, resize: 'none', minHeight: 98, marginBottom: 14 }}
                  placeholder="Add a note for the selected tooth"
                />

                <div style={dockSectionTitleStyle}>Report Note</div>
                <textarea
                  rows={4}
                  value={reportNoteDraft}
                  disabled={session?.is_finalized}
                  onChange={(event) => updateReportNoteDraft(event.target.value)}
                  style={{ ...dockFieldStyle, resize: 'none', minHeight: 98 }}
                  placeholder="Add a clinician note for the report draft"
                />
              </div>
            ) : (
              <>
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
                    {/* {saveState === 'saving' ? 'Saving' : saveState === 'saved' ? 'Saved' : saveState === 'error' ? 'Error' : 'Idle'} */}
                  </span>
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
                      className="w-full rounded-2xl bg-slate-950 px-3 py-2 text-[15px] text-white outline-none"
                      style={{ backgroundColor: '#020617', color: '#f8fafc', colorScheme: 'dark', border: '1px solid rgba(255,255,255,0.22)', WebkitTextFillColor: '#f8fafc' }}
                    >
                      {teeth.map((tooth) => (
                        <option
                          key={tooth.tooth_label}
                          value={String(tooth.tooth_label)}
                          style={{ backgroundColor: '#020617', color: '#f8fafc' }}
                        >
                          Tooth {tooth.tooth_label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                    <label className="block" style={{ flex: 1, minWidth: 0 }}>
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
                        className="w-full rounded-2xl bg-slate-900 px-3 py-2 text-[15px] outline-none"
                        style={{ border: '1px solid #ffffff' }}
                      />
                    </label>
                    <label className="block" style={{ flex: 1, minWidth: 0 }}>
                      <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Bone %</span>
                      <input
                        type="number"
                        min={0}
                        max={100}
                        step={0.01}
                        value={reviewForm.bone_loss_pct}
                        disabled={session?.is_finalized}
                        onChange={(event) => {
                          const roundedValue = roundToTwoDecimals(Number(event.target.value || 0));
                          setReviewForm((prev) => ({ ...prev, bone_loss_pct: roundedValue }));
                          setEditorDirty(true);
                          setSaveState('idle');
                        }}
                        className="w-full rounded-2xl bg-slate-900 px-3 py-2 text-[15px] outline-none"
                        style={{ border: '1px solid #ffffff' }}
                      />
                    </label>
                  </div>

                  <label className="block">
                    <span className="mb-1 block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Tooth Note</span>
                    <textarea
                      ref={toothNoteTextareaRef}
                      rows={3}
                      value={reviewForm.note}
                      disabled={session?.is_finalized}
                      onChange={(event) => {
                        updateToothNoteDraft(event.target.value);
                        setToothNoteCaret(event.target.selectionStart ?? event.target.value.length);
                      }}
                      onClick={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                      onKeyUp={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                      onSelect={(event) => setToothNoteCaret(event.currentTarget.selectionStart ?? 0)}
                      onKeyDown={(event) => {
                        if (event.key === 'Tab' && toothNoteAutocomplete?.ghostText) {
                          event.preventDefault();
                          event.stopPropagation();
                          acceptToothNoteAutocomplete();
                        }
                      }}
                      className="mt-3 w-full rounded-2xl bg-slate-950 px-3 py-2 text-[15px] leading-6 text-white outline-none placeholder:text-slate-400"
                      style={{ border: '1px solid rgba(255,255,255,0.22)', caretColor: '#ffffff' }}
                      placeholder="Add a note for the selected tooth"
                    />
                    {toothNoteAutocomplete?.ghostText ? (
                      <button
                        type="button"
                        onClick={acceptToothNoteAutocomplete}
                        className="mt-2 block text-left text-[13px] text-slate-500 transition-colors hover:text-slate-300"
                      >
                        Suggestion: <span className="text-slate-400">{toothNoteAutocomplete.fullText}</span>
                      </button>
                    ) : null}
                  </label>

                  <label className="block">
                    <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
                      <span className="block text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Report Note</span>
                      {renderDictationControl()}
                    </div>
                    <textarea
                      rows={3}
                      value={reportNoteDraft}
                      disabled={session?.is_finalized}
                      onChange={(event) => updateReportNoteDraft(event.target.value)}
                      className="w-full rounded-2xl bg-slate-950 px-3 py-2 text-[15px] leading-6 text-white outline-none placeholder:text-slate-400"
                      style={{ border: '1px solid rgba(255,255,255,0.22)', caretColor: '#ffffff' }}
                      placeholder="Add a clinician note for the report draft"
                    />
                  </label>
                </div>
              </>
            )}
          </div>
          {!isDock && (
            <div className="border-t border-white/10 bg-black/10 px-5 py-4 md:px-6">
              <div className="flex gap-2">
                <button
                  onClick={handleResetToAi}
                  disabled={!selectedToothId || Boolean(session?.is_finalized)}
                  className="flex-1 rounded-2xl border border-slate-700 px-3 py-2 text-[15px] font-medium text-slate-100 disabled:opacity-50"
                >
                  Reset Tooth
                </button>
                <button
                  onClick={handleOpenReportActions}
                  disabled={actionState !== 'idle'}
                  className="flex-1 rounded-2xl bg-cyan-500 px-3 py-2 text-[15px] font-medium text-slate-950 disabled:opacity-50"
                >
                  {actionState === 'regenerating' ? 'Generating...' : 'Open Report Actions'}
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className={`flex-1 overflow-y-auto ${isDock ? 'px-0 py-0' : 'px-5 py-5 md:px-6'}`} style={dockBodyShellStyle}>
          {isDock ? (
            <div style={{ padding: '18px 18px 20px' }}>
              <div style={dockSectionTitleStyle}>Guide</div>
              <div style={{ ...dockCardStyle, display: 'grid', gap: 10 }}>
                {[
                  '1. Review AI keywords and tooth findings.',
                  '2. Edit labels and report note in Review.',
                  '3. Open or generate the report from Report.',
                  '4. Finalize after confirmation.',
                ].map((step) => (
                  <div key={step} style={{ color: '#E5E5E5', fontSize: 12 }}>
                    {step}
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 18, ...dockSectionTitleStyle }}>Session Info</div>
              <div style={{ ...dockCardStyle, display: 'grid', gap: 10 }}>
                <div style={{ color: '#BFBFBF', fontSize: 12 }}>Session: <span style={{ color: '#E5E5E5' }}>{sessionId.slice(0, 8)}</span></div>
                <div style={{ color: '#BFBFBF', fontSize: 12 }}>Language: <span style={{ color: '#E5E5E5' }}>{session?.language || '-'}</span></div>
                <div style={{ color: '#BFBFBF', fontSize: 12 }}>Status: <span style={{ color: '#E5E5E5' }}>{statusLabel}</span></div>
                <div style={{ color: '#BFBFBF', fontSize: 12 }}>Reviewed Tooth: <span style={{ color: '#E5E5E5' }}>{selectedToothId || '-'}</span></div>
              </div>
            </div>
          ) : (
          <>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Session Info</p>
              <div className="mt-4 space-y-3 text-[15px] text-slate-200">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Session</span>
                  <span className="font-medium text-white">{sessionId.slice(0, 8)}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Language</span>
                  <span>{session?.language || '-'}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Status</span>
                  <span>{statusLabel}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Updated</span>
                  <span>{session?.updated_at ? new Date(session.updated_at).toLocaleString() : '-'}</span>
                </div>
              </div>
            </div>

            <div className="rounded-[24px] border border-white/10 bg-white/5 p-5">
              <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-400">Document Info</p>
              <div className="mt-4 space-y-3 text-[15px] text-slate-200">
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Version</span>
                  <span>{session?.report?.version ?? 1}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">HTML</span>
                  <span>{session?.report?.html_path ? 'Ready' : 'Pending'}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">PDF</span>
                  <span>{hasPdf ? 'Ready' : 'Unavailable'}</span>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <span className="text-slate-400">Reviewed Tooth</span>
                  <span>{selectedToothId || '-'}</span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-[26px] border border-cyan-400/15 bg-[linear-gradient(135deg,rgba(34,211,238,0.12),rgba(14,165,233,0.04))] p-5">
            <br/>
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-cyan-300">Workflow</p>
            <div className="mt-4 grid gap-3">
              {[
                '1. Review AI keywords and tooth findings.',
                '2. Edit labels and report note in Review.',
                '3. Generate the report from Review.',
                '4. Inspect the result in Report and finalize.',
              ].map((step) => (
                <div key={step} className="rounded-2xl border border-white/10 bg-black/10 px-4 py-4 text-[15px] text-slate-200">
                  {step}
                </div>
              ))}
            </div>
          </div>
          </>
          )}
        </div>
      )}
    </div>
  );
}
