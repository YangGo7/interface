import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { FolderOpen, Image as ImageIcon, RefreshCw, Search } from 'lucide-react';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import {
  fetchServerFolderIndex,
  materializeServerStudy,
  resolveServerAssetUrl,
  type ServerFolderImage,
  type ServerFolderStudy,
} from '../lib/folderLeaderApi';
import { TopHeader } from '../components/TopHeader';

const formatCount = (value: number, label: string) => `${value} ${label}${value === 1 ? '' : 's'}`;

export default function FolderLeaderPage() {
  const navigate = useNavigate();
  const [studies, setStudies] = useState<ServerFolderStudy[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [openingKey, setOpeningKey] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<'studies' | 'images'>('studies');
  const [searchTerm, setSearchTerm] = useState('');
  const [rootPath, setRootPath] = useState('');
  const [rootExists, setRootExists] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);
  const [images, setImages] = useState<ServerFolderImage[]>([]);

  const loadStudies = async (isManualRefresh = false) => {
    setError(null);
    if (isManualRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      const data = await fetchServerFolderIndex();
      setStudies(data.studies || []);
      setImages(data.images || []);
      setRootPath(data.root_path || '');
      setRootExists(data.root_exists !== false);
      setLastUpdatedAt(new Date());
    } catch (nextError: any) {
      setError(nextError?.message || 'Failed to load studies.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    void loadStudies(false);
  }, []);

  const filteredStudies = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return studies;

    return studies.filter((study) => {
      const haystack = [
        study.label,
        study.description,
        study.patientId,
        study.studyDate,
        ...(study.modalities || []),
        ...(study.series || []).flatMap((series) => [
          series.label,
          series.description,
          series.modality,
          series.spacingLabel,
          series.compression,
        ]),
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      return haystack.includes(query);
    });
  }, [searchTerm, studies]);

  const filteredImages = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return images;

    return images.filter((image) => {
      const haystack = [
        image.name,
        image.folderLabel,
        image.relativePath,
        image.format,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();
      return haystack.includes(query);
    });
  }, [images, searchTerm]);

  const openStudy = async (study: ServerFolderStudy, selectedSeriesId?: string) => {
    const targetSeriesId = selectedSeriesId || study.series[0]?.id || null;
    const key = `${study.id}:${targetSeriesId || 'first'}`;
    setOpeningKey(key);
    setError(null);

    try {
      const folderStudy: FolderStudy = await materializeServerStudy(study);
      navigate('/chart-legacy', {
        state: {
          originalFolderMode: true,
          originalFolderStudies: [folderStudy],
          folderSelectedSeriesId: targetSeriesId,
          previewUrl: study.previewUrl ? resolveServerAssetUrl(study.previewUrl) : undefined,
          originalIsDicom: true,
          originalFileName:
            folderStudy.series.find((series) => series.id === targetSeriesId)?.label ||
            folderStudy.label ||
            'DICOM Study',
          folderSource: 'server',
        },
      });
    } catch (nextError: any) {
      setError(nextError?.message || 'Failed to open the selected study.');
    } finally {
      setOpeningKey(null);
    }
  };

  const openImage = (image: ServerFolderImage) => {
    const isDicomImage = Boolean(image.isDicom || /\.(dcm|dicom)$/i.test(image.name));
    const imageUrl = resolveServerAssetUrl(image.downloadUrl);
    const previewUrl = image.previewUrl ? resolveServerAssetUrl(image.previewUrl) : imageUrl;
    navigate('/chart-legacy', {
      state: {
        previewUrl,
        imageUrl,
        originalFileName: image.name,
        originalIsDicom: isDicomImage,
        folderSource: 'server-image',
        result: {
          image_url: imageUrl,
          overlay_url: previewUrl,
          preview_url: previewUrl,
          det_counts: {},
          is_volume: false,
        },
      },
    });
  };

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(34,211,238,0.16),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(14,165,233,0.12),transparent_30%),linear-gradient(180deg,#050816_0%,#07111f_55%,#04070d_100%)]" />

      <div className="relative z-10">
        <TopHeader actions={<div className="text-sm font-medium text-gray-500">Folder Leader Original</div>} />
      </div>

      <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-4 px-4 py-4 sm:px-6 lg:px-8">
        <header className="rounded-[28px] border border-white/8 bg-white/5 px-5 py-5 shadow-[0_24px_80px_rgba(2,6,23,0.45)] backdrop-blur-xl">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-200/80">
                Folder Leader
              </div>
              <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white">
                Local DICOM folder browser
              </h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-300">
                The backend scans the configured DICOM server folder, groups studies and series, and opens the
                selected series directly in the chart page.
              </p>
            </div>

            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => loadStudies(true)}
                disabled={loading || refreshing}
                className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/12 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:border-cyan-300/50 hover:bg-cyan-400/18 disabled:cursor-not-allowed disabled:opacity-60"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <Link
                to="/folder-leader-ver-2"
                className="inline-flex items-center gap-2 rounded-full border border-cyan-400/30 bg-cyan-400/12 px-4 py-2 text-sm font-semibold text-cyan-100 transition hover:border-cyan-300/50 hover:bg-cyan-400/18"
              >
                Version 2 (DB)
              </Link>
              <Link
                to="/upload"
                className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-white/20 hover:bg-white/10"
              >
                Upload mode
              </Link>
            </div>
          </div>

          <div className="mt-5 grid gap-3 md:grid-cols-[minmax(0,1fr)_auto]">
            <label className="flex items-center gap-3 rounded-2xl border border-white/8 bg-black/25 px-4 py-3">
              <Search className="h-4 w-4 text-slate-500" />
              <input
                value={searchTerm}
                onChange={(event) => setSearchTerm(event.target.value)}
                placeholder="Search by patient, study, series, modality, or spacing"
                className="min-w-0 flex-1 bg-transparent text-sm text-slate-100 outline-none placeholder:text-slate-500"
              />
            </label>

            <div className="flex flex-wrap items-center gap-2 rounded-2xl border border-white/8 bg-black/20 px-4 py-3 text-xs text-slate-300">
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                {formatCount(studies.length, 'study')}
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                {formatCount(studies.reduce((sum, study) => sum + study.totalSeries, 0), 'series')}
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                {formatCount(images.length, 'image')}
              </span>
              <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1">
                {rootExists ? 'Root available' : 'Root missing'}
              </span>
              {rootPath ? <span className="text-slate-500">Root: {rootPath}</span> : null}
              {lastUpdatedAt ? <span className="text-slate-500">Updated: {lastUpdatedAt.toLocaleTimeString()}</span> : null}
            </div>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={() => setActiveSection('studies')}
              className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition ${
                activeSection === 'studies'
                  ? 'bg-cyan-400 text-slate-950'
                  : 'border border-white/10 bg-white/5 text-slate-200 hover:bg-white/10'
              }`}
            >
              <FolderOpen className="h-4 w-4" />
              DICOM Studies
            </button>
            <button
              type="button"
              onClick={() => setActiveSection('images')}
              className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-semibold transition ${
                activeSection === 'images'
                  ? 'bg-cyan-400 text-slate-950'
                  : 'border border-white/10 bg-white/5 text-slate-200 hover:bg-white/10'
              }`}
            >
              <ImageIcon className="h-4 w-4" />
              Images
            </button>
          </div>
        </header>

        {error ? (
          <div className="rounded-3xl border border-rose-400/20 bg-rose-400/10 px-5 py-4 text-sm text-rose-100">
            {error}
          </div>
        ) : null}

        <section className="min-h-0 flex-1 overflow-hidden rounded-[32px] border border-white/8 bg-[linear-gradient(180deg,rgba(8,15,28,0.92),rgba(3,6,14,0.95))] shadow-[0_24px_90px_rgba(2,6,23,0.52)]">
          {loading ? (
            <div className="flex h-[65vh] items-center justify-center px-6 text-center">
              <div>
                <div className="mx-auto h-12 w-12 animate-spin rounded-full border-2 border-cyan-300/30 border-t-cyan-300" />
                <p className="mt-4 text-sm text-slate-400">Scanning the configured DICOM folder...</p>
              </div>
            </div>
          ) : activeSection === 'studies' && filteredStudies.length === 0 ? (
            <div className="flex h-[65vh] items-center justify-center px-6 text-center">
              <div className="max-w-md rounded-3xl border border-dashed border-white/10 bg-white/4 px-6 py-8">
                <FolderOpen className="mx-auto h-8 w-8 text-slate-500" />
                <h2 className="mt-4 text-lg font-semibold text-white">
                  {searchTerm ? 'No matching studies' : 'No DICOM studies found'}
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  {searchTerm
                    ? 'Try a different keyword.'
                    : 'Check the DICOM_SERVER_ROOT environment variable and make sure the folder contains readable DICOM files.'}
                </p>
              </div>
            </div>
          ) : activeSection === 'images' && filteredImages.length === 0 ? (
            <div className="flex h-[65vh] items-center justify-center px-6 text-center">
              <div className="max-w-md rounded-3xl border border-dashed border-white/10 bg-white/4 px-6 py-8">
                <ImageIcon className="mx-auto h-8 w-8 text-slate-500" />
                <h2 className="mt-4 text-lg font-semibold text-white">
                  {searchTerm ? 'No matching images' : 'No image files found'}
                </h2>
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  {searchTerm
                    ? 'Try a different keyword.'
                    : 'Supported image files are listed separately from DICOM studies.'}
                </p>
              </div>
            </div>
          ) : (
            <div className="h-full overflow-y-auto p-4 sm:p-5">
              {activeSection === 'studies' ? (
                <div className="grid gap-4 xl:grid-cols-2">
                  {filteredStudies.map((study) => {
                  const firstSeriesId = study.series[0]?.id || null;
                  const isOpeningStudy = openingKey === `${study.id}:${firstSeriesId || 'first'}`;

                  return (
                    <article
                      key={study.id}
                      className="overflow-hidden rounded-[28px] border border-white/8 bg-white/5 shadow-[0_20px_60px_rgba(2,6,23,0.28)]"
                    >
                      <button
                        type="button"
                        onClick={() => openStudy(study, firstSeriesId || undefined)}
                        disabled={Boolean(openingKey)}
                        className="flex w-full items-start justify-between gap-4 border-b border-white/8 px-5 py-4 text-left transition hover:bg-white/5 disabled:cursor-not-allowed disabled:opacity-70"
                      >
                        <div className="min-w-0">
                          <div className="text-[11px] font-semibold uppercase tracking-[0.28em] text-cyan-200/80">
                            Study
                          </div>
                          <h2 className="mt-2 truncate text-xl font-semibold text-white">
                            {study.label || study.description || study.id}
                          </h2>
                          <p className="mt-1 text-sm text-slate-400">
                            {study.description || 'Unnamed study'}
                          </p>
                        </div>

                        <div className="text-right text-xs text-slate-400">
                          <div>{study.studyDate || 'No date'}</div>
                          <div className="mt-1">{study.patientId || 'Unknown patient'}</div>
                        </div>
                      </button>

                      <div className="grid gap-3 px-5 py-4">
                        <div className="flex flex-wrap gap-2 text-xs text-slate-300">
                          <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1">
                            {formatCount(study.totalSeries, 'series')}
                          </span>
                          <span className="rounded-full border border-white/10 bg-black/20 px-3 py-1">
                            {formatCount(study.totalFiles, 'file')}
                          </span>
                          {study.modalities.map((modality) => (
                            <span key={modality} className="rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-cyan-100">
                              {modality}
                            </span>
                          ))}
                        </div>

                        <div className="grid gap-2">
                          {study.series.map((series) => {
                            const isActive = openingKey === `${study.id}:${series.id}`;
                            return (
                              <button
                                key={series.id}
                                type="button"
                                onClick={() => openStudy(study, series.id)}
                                disabled={Boolean(openingKey)}
                                className={`rounded-2xl border px-4 py-3 text-left transition ${
                                  isActive
                                    ? 'border-cyan-300/40 bg-cyan-400/12'
                                    : 'border-white/8 bg-black/15 hover:border-cyan-300/25 hover:bg-white/5'
                                } disabled:cursor-not-allowed disabled:opacity-70`}
                              >
                                <div className="flex items-start justify-between gap-4">
                                  <div className="min-w-0">
                                    <div className="flex flex-wrap items-center gap-2 text-xs text-slate-400">
                                      <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-[11px] uppercase tracking-[0.12em]">
                                        {series.modality || 'OT'}
                                      </span>
                                      {series.volumeEligible ? (
                                        <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 text-[11px] text-emerald-100">
                                          Volume ready
                                        </span>
                                      ) : null}
                                    </div>
                                    <div className="mt-2 truncate text-sm font-semibold text-white">
                                      {series.label || series.description || 'Unnamed series'}
                                    </div>
                                    <div className="mt-1 text-xs text-slate-400">
                                      {series.spacingLabel} - {series.compression}
                                    </div>
                                  </div>

                                  <div className="text-right text-xs text-slate-400">
                                    <div>{formatCount(series.sliceCount, 'slice')}</div>
                                    <div className="mt-1">{series.orientation}</div>
                                  </div>
                                </div>
                              </button>
                            );
                          })}
                        </div>
                      </div>

                      {isOpeningStudy ? (
                        <div className="border-t border-white/8 px-5 py-3 text-sm text-cyan-100">
                          Loading study files for chart page...
                        </div>
                      ) : null}
                    </article>
                  );
                  })}
                </div>
              ) : (
                <div className="folder-leader-image-grid">
                  {filteredImages.map((image) => {
                    const isOpeningImage = openingKey === `image:${image.relativePath}`;
                    return (
                      <button
                        key={image.relativePath}
                        type="button"
                        onClick={() => openImage(image)}
                        disabled={Boolean(openingKey)}
                        className={`folder-leader-image-card group overflow-hidden rounded-[12px] border text-left transition ${
                          isOpeningImage
                            ? 'border-cyan-300/40 bg-cyan-400/12'
                            : 'border-white/8 bg-white/5 hover:border-cyan-300/25 hover:bg-white/8'
                        } disabled:cursor-not-allowed disabled:opacity-70`}
                        title={image.name}
                      >
                        <div className="px-2 pt-2">
                          <div className="folder-leader-image-thumb relative w-full overflow-hidden rounded-[8px] border border-white/8 bg-[linear-gradient(135deg,rgba(15,23,42,0.88),rgba(8,15,28,0.98))]">
                            <div className="flex h-full w-full items-center justify-center overflow-hidden bg-black/20">
                              <img
                                src={image.downloadUrl}
                                alt={image.name}
                                loading="lazy"
                                className="h-full w-full object-contain px-1 transition duration-300 group-hover:scale-[1.02]"
                                onError={(event) => {
                                  const target = event.currentTarget;
                                  const fallbackSvg =
                                    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480"><rect width="640" height="480" rx="36" fill="#0f172a"/><rect x="64" y="64" width="512" height="352" rx="24" fill="#111827" stroke="#334155"/><path d="M180 308l72-92 88 108 56-64 96 108" fill="none" stroke="#64748b" stroke-width="18" stroke-linecap="round" stroke-linejoin="round"/><circle cx="226" cy="188" r="24" fill="#64748b"/></svg>';
                                  target.onerror = null;
                                  target.src = `data:image/svg+xml,${encodeURIComponent(fallbackSvg)}`;
                                }}
                              />
                            </div>
                          </div>
                        </div>
                        <div className="mt-1.5 border-t border-white/8 px-2.5 pt-1.5">
                          <div className="truncate text-[9px] font-medium uppercase tracking-[0.06em] text-slate-400">
                            {image.format || 'IMAGE'} / {image.width} x {image.height}
                          </div>
                        </div>
                        <div className="border-t border-white/8 bg-slate-950/55 px-2.5 py-1.5">
                          <div className="truncate text-[11px] font-semibold text-slate-50">{image.name}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
