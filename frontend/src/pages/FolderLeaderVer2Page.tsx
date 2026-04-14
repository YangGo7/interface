import { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, Database, RefreshCw, Search, X } from 'lucide-react';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import {
  fetchServerFolderIndex,
  materializeServerStudy,
  resolveServerAssetUrl,
  type ServerFolderStudy,
} from '../lib/folderLeaderApi';
import { TopHeader } from '../components/TopHeader';

const formatCount = (value: number, label: string) => `${value} ${label}${value === 1 ? '' : 's'}`;

function buildPatientLabel(study: ServerFolderStudy) {
  const parts = [study.patientName, study.patientAge, study.patientSex].filter(Boolean);
  return parts.join(' / ') || study.patientId || 'Unknown patient';
}

function getStudyModality(study: ServerFolderStudy) {
  return study.modalities.length > 0 ? study.modalities.join(', ') : 'OT';
}

export default function FolderLeaderVer2Page() {
  const navigate = useNavigate();
  const [studies, setStudies] = useState<ServerFolderStudy[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [joiningStudyId, setJoiningStudyId] = useState<string | null>(null);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [selectedSeriesId, setSelectedSeriesId] = useState<string | null>(null);
  const [patientIdQuery, setPatientIdQuery] = useState('');
  const [patientNameQuery, setPatientNameQuery] = useState('');
  const [studyDateQuery, setStudyDateQuery] = useState('');
  const [modalityQuery, setModalityQuery] = useState('');
  const [sexQuery, setSexQuery] = useState('');
  const [descriptionQuery, setDescriptionQuery] = useState('');
  const [rootPath, setRootPath] = useState('');
  const [rootExists, setRootExists] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<Date | null>(null);

  const loadStudies = async (isManualRefresh = false) => {
    setError(null);
    if (isManualRefresh) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      const data = await fetchServerFolderIndex();
      const nextStudies = data.studies || [];
      setStudies(nextStudies);
      setRootPath(data.root_path || '');
      setRootExists(data.root_exists !== false);
      setLastUpdatedAt(new Date());

      if (nextStudies.length > 0) {
        setSelectedStudyId((currentId) => {
          if (currentId && nextStudies.some((study) => study.id === currentId)) {
            return currentId;
          }
          return null; // Don't auto-select first study
        });
      } else {
        setSelectedStudyId(null);
        setSelectedSeriesId(null);
      }
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
    const patientId = patientIdQuery.trim().toLowerCase();
    const patientName = patientNameQuery.trim().toLowerCase();
    const studyDate = studyDateQuery.trim().toLowerCase();
    const modality = modalityQuery.trim().toLowerCase();
    const sex = sexQuery.trim().toLowerCase();
    const description = descriptionQuery.trim().toLowerCase();

    return studies.filter((study) => {
      if (patientId && !study.patientId.toLowerCase().includes(patientId)) return false;
      if (patientName && !buildPatientLabel(study).toLowerCase().includes(patientName)) return false;
      if (studyDate && !study.studyDate.toLowerCase().includes(studyDate)) return false;
      if (modality && !getStudyModality(study).toLowerCase().includes(modality)) return false;
      if (sex && !study.patientSex.toLowerCase().includes(sex)) return false;
      if (description) {
        const descriptionHaystack = [study.description, study.label, ...study.series.map((series) => series.description)]
          .filter(Boolean)
          .join(' ')
          .toLowerCase();
        if (!descriptionHaystack.includes(description)) return false;
      }
      return true;
    });
  }, [descriptionQuery, modalityQuery, patientIdQuery, patientNameQuery, sexQuery, studies, studyDateQuery]);

  useEffect(() => {
    if (filteredStudies.length === 0) {
      setSelectedStudyId(null);
      setSelectedSeriesId(null);
      return;
    }

    const nextSelectedStudy = filteredStudies.find((study) => study.id === selectedStudyId) || null;
    if (nextSelectedStudy && nextSelectedStudy.id !== selectedStudyId) {
      setSelectedStudyId(nextSelectedStudy.id);
    }

    if (nextSelectedStudy) {
      const hasSelectedSeries = nextSelectedStudy.series.some((series) => series.id === selectedSeriesId);
      if (!hasSelectedSeries) {
        setSelectedSeriesId(nextSelectedStudy.series[0]?.id || null);
      }
    } else {
      setSelectedSeriesId(null);
    }
  }, [filteredStudies, selectedSeriesId, selectedStudyId]);

  const selectedStudy = useMemo(
    () => filteredStudies.find((study) => study.id === selectedStudyId) || null,
    [filteredStudies, selectedStudyId]
  );

  const selectedSeries =
    selectedStudy?.series.find((series) => series.id === selectedSeriesId) || selectedStudy?.series[0] || null;

  const previewUrl = selectedStudy?.previewUrl ? resolveServerAssetUrl(selectedStudy.previewUrl) : null;

  const clearFilters = () => {
    setPatientIdQuery('');
    setPatientNameQuery('');
    setStudyDateQuery('');
    setModalityQuery('');
    setSexQuery('');
    setDescriptionQuery('');
  };

  const openStudy = async () => {
    if (!selectedStudy) return;
    const targetSeriesId = selectedSeries?.id || selectedStudy.series[0]?.id || null;
    setJoiningStudyId(selectedStudy.id);
    setError(null);

    try {
      const folderStudy: FolderStudy = await materializeServerStudy(selectedStudy);
      navigate('/chart', {
        state: {
          originalFolderMode: true,
          originalFolderStudies: [folderStudy],
          folderSelectedSeriesId: targetSeriesId,
          previewUrl: selectedStudy.previewUrl ? resolveServerAssetUrl(selectedStudy.previewUrl) : undefined,
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
      setJoiningStudyId(null);
    }
  };

  const closeSelection = () => {
    setSelectedStudyId(null);
    setSelectedSeriesId(null);
  };

  return (
    <div className="folder-leader-v2-page">
      <TopHeader actions={<div className="text-sm font-medium text-slate-500"></div>} />
      <main className="folder-leader-v2-shell">
        <section className="folder-leader-v2-panel">
          <div className="folder-leader-v2-panel-title">
            <Database className="h-4 w-4" />
            Master Database
          </div>
          <div className="folder-leader-v2-db-box">
            <div className="folder-leader-v2-db-icon">
              <Database className="h-7 w-7" />
            </div>
            <div className="folder-leader-v2-db-copy">
              <div className="folder-leader-v2-db-name">Master Database</div>
              <div className="folder-leader-v2-db-rule" />
              <div className="folder-leader-v2-db-meta">Type : OnDemand3DWeb</div>
              <div className="folder-leader-v2-db-meta">Root : {rootPath || 'Not configured'}</div>
              <div className="folder-leader-v2-db-meta">
                Status : {rootExists ? 'Available' : 'Missing'}
                {lastUpdatedAt ? ` / Updated ${lastUpdatedAt.toLocaleTimeString()}` : ''}
              </div>
            </div>
            <div className="folder-leader-v2-db-actions">
              <button type="button" onClick={() => loadStudies(true)} disabled={loading || refreshing}>
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <Link to="/folder-leader">Grid</Link>
              <Link to="/upload">Upload</Link>
            </div>
          </div>
        </section>

        <section className="folder-leader-v2-panel">
          <div className="folder-leader-v2-panel-title">
            <Search className="h-4 w-4" />
            Search
          </div>
          <div className="folder-leader-v2-search-box">
            <div className="folder-leader-v2-search-grid">
              <label>
                <span>Patient ID</span>
                <input value={patientIdQuery} onChange={(event) => setPatientIdQuery(event.target.value)} />
              </label>
              <label>
                <span>Patient Name</span>
                <input value={patientNameQuery} onChange={(event) => setPatientNameQuery(event.target.value)} />
              </label>
              <label>
                <span>Date Created</span>
                <input value={studyDateQuery} onChange={(event) => setStudyDateQuery(event.target.value)} placeholder="YYYY-MM-DD" />
              </label>
              <label>
                <span>Modality</span>
                <input value={modalityQuery} onChange={(event) => setModalityQuery(event.target.value)} placeholder="CT, CBCT" />
              </label>
              <label>
                <span>Sex</span>
                <input value={sexQuery} onChange={(event) => setSexQuery(event.target.value)} placeholder="M / F" />
              </label>
              <label className="folder-leader-v2-search-wide">
                <span>Description</span>
                <input value={descriptionQuery} onChange={(event) => setDescriptionQuery(event.target.value)} />
              </label>
            </div>

            <div className="folder-leader-v2-search-actions">
              <div className="folder-leader-v2-search-status">
                <span>{formatCount(filteredStudies.length, 'result')}</span>
                <span>{formatCount(studies.reduce((sum, study) => sum + study.totalSeries, 0), 'series')}</span>
              </div>
              <div className="folder-leader-v2-search-buttons">
                <button type="button" onClick={() => loadStudies(true)} disabled={loading || refreshing}>
                  Search
                </button>
                <button type="button" onClick={clearFilters}>
                  <X className="h-4 w-4" />
                  Clear
                </button>
              </div>
            </div>
          </div>
        </section>

        {error ? <div className="folder-leader-v2-error">{error}</div> : null}

        <section className={`folder-leader-v2-workspace ${selectedStudy ? 'has-selection' : ''}`}>
          <div className="folder-leader-v2-table-panel">
            <div className="folder-leader-v2-table-header">
              <div>ID</div>
              <div>Patient Name</div>
              <div>Date Created</div>
              <div>Description</div>
              <div>#Imgs</div>
              <div>Modality</div>
              <div>Comment</div>
            </div>

            <div className="folder-leader-v2-table-body">
              {loading ? (
                <div className="folder-leader-v2-empty">Scanning the configured DICOM folder...</div>
              ) : filteredStudies.length === 0 ? (
                <div className="folder-leader-v2-empty">No matching DICOM studies.</div>
              ) : (
                filteredStudies.map((study) => {
                  const isSelected = study.id === selectedStudy?.id;
                  return (
                    <button
                      key={study.id}
                      type="button"
                      onClick={() => {
                        if (selectedStudy?.id === study.id) {
                          closeSelection();
                          return;
                        }
                        setSelectedStudyId(study.id);
                        setSelectedSeriesId(study.series[0]?.id || null);
                      }}
                      className={`folder-leader-v2-table-row ${isSelected ? 'is-selected' : ''}`}
                    >
                      <div className="folder-leader-v2-cell-id">{study.patientId || '-'}</div>
                      <div className="folder-leader-v2-cell-name">{buildPatientLabel(study)}</div>
                      <div>{study.studyDate || '-'}</div>
                      <div className="folder-leader-v2-cell-description">{study.description || study.label}</div>
                      <div>{study.totalFiles}</div>
                      <div>{getStudyModality(study)}</div>
                      <div>{study.series[0]?.orientation || '-'}</div>
                    </button>
                  );
                })
              )}
            </div>
          </div>

          {selectedStudy ? (
            <div className="folder-leader-v2-detail-panel">
              <div className="folder-leader-v2-detail-head">
                <div>
                  <div className="folder-leader-v2-detail-title">Selected Study</div>
                  <div className="folder-leader-v2-detail-name">{selectedStudy.description || selectedStudy.label}</div>
                </div>
                <div className="folder-leader-v2-detail-actions">

                  <button
                    type="button"
                    onClick={openStudy}
                    disabled={joiningStudyId === selectedStudy.id}
                    className="folder-leader-v2-join"
                  >
                    {joiningStudyId === selectedStudy.id ? 'Opening...' : 'Join'}
                    <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="folder-leader-v2-detail-body">
                <div className="folder-leader-v2-preview-box">
                  {previewUrl ? (
                    <img
                      src={previewUrl}
                      alt={selectedStudy.description || selectedStudy.label}
                      className="folder-leader-v2-preview-image"
                      onError={(event) => {
                        const target = event.currentTarget;
                        const fallbackSvg =
                          '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="160" viewBox="0 0 400 160"><rect width="400" height="160" fill="#0d0d0d"/><rect x="40" y="28" width="320" height="104" fill="#2c2c2c" stroke="#8a8a8a"/><path d="M80 108h240M110 72h180" stroke="#bdbdbd" stroke-width="10" stroke-linecap="round"/></svg>';
                        target.onerror = null;
                        target.src = `data:image/svg+xml,${encodeURIComponent(fallbackSvg)}`;
                      }}
                    />
                  ) : (
                    <div className="folder-leader-v2-preview-placeholder">Preview unavailable</div>
                  )}
                </div>

                <div className="folder-leader-v2-detail-grid">
                  <div>
                    <span>Patient ID</span>
                    <strong>{selectedStudy.patientId || '-'}</strong>
                  </div>
                  <div>
                    <span>Patient Name</span>
                    <strong>{buildPatientLabel(selectedStudy)}</strong>
                  </div>
                  <div>
                    <span>Date Created</span>
                    <strong>{selectedStudy.studyDate || '-'}</strong>
                  </div>
                  <div>
                    <span>Modality</span>
                    <strong>{getStudyModality(selectedStudy)}</strong>
                  </div>
                  <div>
                    <span>Series</span>
                    <strong>{selectedStudy.totalSeries}</strong>
                  </div>
                  <div>
                    <span>Files</span>
                    <strong>{selectedStudy.totalFiles}</strong>
                  </div>
                </div>

                <div className="folder-leader-v2-series-list">
                  {selectedStudy.series.map((series) => {
                    const isSelected = series.id === selectedSeries?.id;
                    return (
                      <button
                        key={series.id}
                        type="button"
                        onClick={() => setSelectedSeriesId(series.id)}
                        className={`folder-leader-v2-series-row ${isSelected ? 'is-selected' : ''}`}
                      >
                        <div className="folder-leader-v2-series-main">
                          <strong>{series.label || series.description}</strong>
                          <span>
                            {series.modality} / {series.orientation}
                          </span>
                        </div>
                        <div className="folder-leader-v2-series-side">
                          <span>{series.sliceCount} imgs</span>
                          <span>{series.spacingLabel}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          ) : null}
        </section>
      </main>
    </div>
  );
}
