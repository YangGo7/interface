import { useEffect, useMemo, useState } from 'react';
import * as cornerstone from '@cornerstonejs/core';
import { ChevronDown, Search } from 'lucide-react';
import type { FolderStudy } from '../../features/upload/dicomFolderStudies';
import { initCornerstone, registerNativeDicomFileWithMetadata } from '../../viewer/cornerstone/init';

type StudiesWorkspacePanelProps = {
  studies: FolderStudy[];
  selectedSeriesId: string | null;
  onSelectSeries: (seriesId: string) => void;
};

export function StudiesWorkspacePanel({ studies, selectedSeriesId, onSelectSeries }: StudiesWorkspacePanelProps) {
  const [expandedStudyIds, setExpandedStudyIds] = useState<string[]>(() => (studies[0] ? [studies[0].id] : []));
  const [searchTerm, setSearchTerm] = useState('');
  const [studyThumbs, setStudyThumbs] = useState<Record<string, string>>({});

  const normalizedStudies = useMemo(
    () =>
      studies.map((study) => {
        const normalizedSeries = (study.series || []).map((series) => ({
          ...series,
          modality: series.modality || 'OT',
          sliceCount: series.sliceCount ?? series.files?.length ?? 0,
        }));

        return {
          ...study,
          patientId: study.patientId || '',
          studyDate: study.studyDate || '',
          modalities: study.modalities?.length
            ? study.modalities
            : Array.from(new Set(normalizedSeries.map((series) => series.modality))).filter(Boolean),
          totalFiles: study.totalFiles ?? normalizedSeries.reduce((sum, series) => sum + (series.files?.length || 0), 0),
          totalSeries: study.totalSeries ?? normalizedSeries.length,
          series: normalizedSeries,
        };
      }),
    [studies],
  );

  const filteredStudies = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    return normalizedStudies
      .map((study) => {
        const filteredSeries = study.series.filter((series) => {
          if (!query) return true;
          const haystack = [
            study.description,
            study.label,
            study.patientId,
            study.studyDate,
            ...(study.modalities || []),
            series.label,
            series.modality,
          ]
            .filter(Boolean)
            .join(' ')
            .toLowerCase();
          return haystack.includes(query);
        });

        if (filteredSeries.length === 0) return null;
        return { ...study, series: filteredSeries };
      })
      .filter(Boolean) as FolderStudy[];
  }, [normalizedStudies, searchTerm]);

  const toggleStudy = (studyId: string) => {
    setExpandedStudyIds((prev) => (prev.includes(studyId) ? prev.filter((id) => id !== studyId) : [...prev, studyId]));
  };

  useEffect(() => {
    let cancelled = false;

    const generateThumbs = async () => {
      await initCornerstone();
      const nextThumbs: Record<string, string> = {};

      for (const study of normalizedStudies) {
        const firstFile = study.series[0]?.files?.[0];
        if (!firstFile) continue;

        try {
          const imageId = await registerNativeDicomFileWithMetadata(firstFile);
          const image = await cornerstone.imageLoader.loadAndCacheImage(imageId);
          const canvas = document.createElement('canvas');
          canvas.width = 64;
          canvas.height = 44;

          if (typeof image.getCanvas === 'function') {
            const sourceCanvas = image.getCanvas();
            const context = canvas.getContext('2d');
            if (!context) continue;
            context.fillStyle = '#000';
            context.fillRect(0, 0, canvas.width, canvas.height);

            const scale = Math.min(canvas.width / sourceCanvas.width, canvas.height / sourceCanvas.height);
            const width = sourceCanvas.width * scale;
            const height = sourceCanvas.height * scale;
            const x = (canvas.width - width) / 2;
            const y = (canvas.height - height) / 2;
            context.drawImage(sourceCanvas, x, y, width, height);
          } else {
            const renderCanvas = document.createElement('canvas');
            renderCanvas.width = image.columns;
            renderCanvas.height = image.rows;
            await cornerstone.utilities.renderToCanvasCPU(renderCanvas, image, 'OT');
            const context = canvas.getContext('2d');
            if (!context) continue;
            context.fillStyle = '#000';
            context.fillRect(0, 0, canvas.width, canvas.height);
            const scale = Math.min(canvas.width / renderCanvas.width, canvas.height / renderCanvas.height);
            const width = renderCanvas.width * scale;
            const height = renderCanvas.height * scale;
            const x = (canvas.width - width) / 2;
            const y = (canvas.height - height) / 2;
            context.drawImage(renderCanvas, x, y, width, height);
          }

          nextThumbs[study.id] = canvas.toDataURL('image/png');
        } catch (error) {
          console.error('Failed to build study thumbnail', study.id, error);
        }
      }

      if (!cancelled) {
        setStudyThumbs(nextThumbs);
      }
    };

    generateThumbs();

    return () => {
      cancelled = true;
    };
  }, [normalizedStudies]);

  if (normalizedStudies.length === 0) {
    return (
      <div className="flex h-full items-center justify-center px-2 text-center text-[8px] leading-4 text-slate-500">
        No local study list is loaded.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="border-b border-white/8 px-2 pb-2 pt-1">
        <div className="flex items-center gap-1 rounded-xl border border-white/10 bg-black/25 px-1.5 py-1">
          <Search className="h-2.5 w-2.5 text-slate-500" />
          <input
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            placeholder="Search"
            className="min-w-0 flex-1 bg-transparent text-[8px] text-slate-200 outline-none placeholder:text-slate-500"
          />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {filteredStudies.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-white/10 bg-black/15 px-3 py-4 text-[8px] text-slate-500">
            No studies match the current search.
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {filteredStudies.map((study) => {
            const expanded = expandedStudyIds.includes(study.id);
            const primaryModality = study.modalities[0] || 'ST';

            return (
              <section key={study.id} className="flex flex-col gap-1">
                <button
                  type="button"
                  onClick={() => toggleStudy(study.id)}
                  className={`overflow-hidden rounded-[8px] border transition ${
                    expanded ? 'border-cyan-500/50 bg-[#0a1d47]' : 'border-white/8 bg-black/20 hover:bg-white/5'
                  } p-1 text-left shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]`}
                >
                  <div className="flex gap-2">
                    <div className="relative h-[34px] w-[50px] shrink-0 rounded-[4px] border border-white/10 bg-black overflow-hidden">
                      {studyThumbs[study.id] ? (
                        <img
                          src={studyThumbs[study.id]}
                          alt={study.description || study.label || 'Study thumbnail'}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <div className="h-full w-full bg-[radial-gradient(circle_at_center,rgba(160,160,160,0.5)_0_4px,transparent_5px)]" />
                      )}
                      <div className="absolute bottom-0.5 left-0.5 rounded-[2px] bg-black/60 px-0.5 py-0 text-[5px] font-bold text-white">
                        {primaryModality}
                      </div>
                    </div>
                    
                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-1">
                        <div className="truncate text-[8px] font-semibold text-white">
                          {study.description || study.label || 'Study'}
                        </div>
                        <ChevronDown className={`h-2.5 w-2.5 shrink-0 text-white/50 transition ${expanded ? 'rotate-180' : ''}`} />
                      </div>
                      <div className="mt-0.5 flex items-center gap-2 text-[7px] text-cyan-100/60 font-medium">
                        <span>Series {study.totalSeries}</span>
                        <span className="h-2 w-px bg-white/10" />
                        <span>Files {study.totalFiles}</span>
                        {study.patientId && (
                          <>
                            <span className="h-2 w-px bg-white/10" />
                            <span className="truncate">{study.patientId}</span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                </button>

                {expanded && (
                  <div className="ml-2 mt-0.5 space-y-1 rounded-[8px] border-l-2 border-cyan-500/30 bg-black/10 pl-2 py-1">
                    {study.series.map((series) => {
                      const isActive = selectedSeriesId === series.id;
                      const hasFiles = series.files && series.files.length > 0;

                      return (
                        <button
                          key={series.id}
                          type="button"
                          onClick={() => onSelectSeries(series.id)}
                          className={`w-full rounded-[6px] border px-2 py-1.5 text-left transition ${
                            isActive
                              ? 'border-cyan-400/40 bg-cyan-400/15 text-white shadow-sm'
                              : 'border-white/5 bg-transparent text-slate-400 hover:border-white/10 hover:bg-white/5 hover:text-white'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2">
                            <div className="min-w-0 flex-1">
                              <div className="truncate text-[8px] font-medium">
                                {series.label || series.description || 'Unnamed Series'}
                              </div>
                              <div className="mt-0.5 flex items-center gap-1.5 text-[6px] text-slate-500">
                                <span className="rounded-sm bg-white/5 px-1 py-0 uppercase tracking-wider">{series.modality || 'OT'}</span>
                                <span>{series.sliceCount} imgs</span>
                                {series.orientation && (
                                  <>
                                    <span className="opacity-30">•</span>
                                    <span>{series.orientation}</span>
                                  </>
                                )}
                              </div>
                            </div>
                            {isActive && (
                              <div className="h-1 w-1 rounded-full bg-cyan-400 shadow-[0_0_4px_rgba(34,211,238,0.8)]" />
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </section>
            );
          })}
          </div>
        )}
      </div>
    </div>
  );
}
