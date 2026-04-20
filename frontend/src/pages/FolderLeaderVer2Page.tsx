import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Database, Image as ImageIcon, RefreshCw, Search, X } from 'lucide-react';
import { SettingsModal } from '../components/SettingsModal';
import { readStoredNumberingSystem, writeStoredNumberingSystem, type AppNumberingSystem } from '../lib/appSettings';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import { buildDicomFolderStudies } from '../features/upload/dicomFolderStudies';
import {
  fetchServerFolderIndex,
  pickServerFolderRootPath,
  fetchServerFolderRootPath,
  materializeServerStudy,
  resolveServerAssetUrl,
  updateServerFolderRootPath,
  type ServerFolderImage,
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

type EnrichedServerImage = ServerFolderImage & {
  linkedStudy: ServerFolderStudy | null;
  patientId: string;
  patientName: string;
  patientSex: string;
  studyDate: string;
  modality: string;
  description: string;
  linkedSeriesId: string | null;
};

type StudyListEntry =
  | {
      kind: 'study';
      key: string;
      patientId: string;
      patientName: string;
      studyDate: string;
      description: string;
      imageCount: number;
      modality: string;
      comment: string;
      study: ServerFolderStudy;
    }
  | {
      kind: 'image';
      key: string;
      patientId: string;
      patientName: string;
      studyDate: string;
      description: string;
      imageCount: number;
      modality: string;
      comment: string;
      image: EnrichedServerImage;
    };

const normalizeRelativePath = (value: string) => value.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');

const getParentDirectory = (value: string) => {
  const normalized = normalizeRelativePath(value);
  const lastSlash = normalized.lastIndexOf('/');
  return lastSlash >= 0 ? normalized.slice(0, lastSlash) : '';
};

const splitSegments = (value: string) => normalizeRelativePath(value).split('/').filter(Boolean);

const countSharedSegments = (left: string, right: string) => {
  const leftParts = splitSegments(left);
  const rightParts = splitSegments(right);
  let count = 0;
  while (count < leftParts.length && count < rightParts.length && leftParts[count] === rightParts[count]) {
    count += 1;
  }
  return count;
};

const buildStudyDirectoryCandidates = (study: ServerFolderStudy) => {
  const candidates = new Set<string>();
  study.series.forEach((series) => {
    series.files.forEach((file) => {
      const folder = getParentDirectory(file.relativePath);
      if (folder) candidates.add(folder);
    });
  });
  return Array.from(candidates);
};

const findBestStudyForImage = (image: ServerFolderImage, studies: ServerFolderStudy[]) => {
  const imageDirectory = getParentDirectory(image.relativePath);
  let bestStudy: ServerFolderStudy | null = null;
  let bestSeriesId: string | null = null;
  let bestScore = -1;

  studies.forEach((study) => {
    const candidateDirectories = buildStudyDirectoryCandidates(study);
    let studyBestScore = -1;
    let studyBestSeriesId: string | null = null;

    study.series.forEach((series) => {
      series.files.forEach((file) => {
        const fileDirectory = getParentDirectory(file.relativePath);
        if (!fileDirectory) return;

        let score = countSharedSegments(imageDirectory, fileDirectory);
        if (imageDirectory && fileDirectory && (imageDirectory.startsWith(fileDirectory) || fileDirectory.startsWith(imageDirectory))) {
          score += 100;
        }
        if (image.folderLabel && fileDirectory.toLowerCase().includes(image.folderLabel.toLowerCase())) {
          score += 10;
        }
        if (score > studyBestScore) {
          studyBestScore = score;
          studyBestSeriesId = series.id;
        }
      });
    });

    if (studyBestScore < 0 && candidateDirectories.length === 0) return;
    if (studyBestScore > bestScore) {
      bestScore = studyBestScore;
      bestStudy = study;
      bestSeriesId = studyBestSeriesId || study.series[0]?.id || null;
    }
  });

  return {
    study: bestScore > 0 ? bestStudy : null,
    seriesId: bestScore > 0 ? bestSeriesId : null,
  };
};

const buildImagePatientName = (fileName: string) => fileName.replace(/\.[^.]+$/, '').trim() || fileName;

const buildImagePseudoPatientId = (relativePath: string) => {
  let hash = 0;
  for (const char of relativePath) {
    hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  }
  return `IMG-${String(hash % 100000000).padStart(8, '0')}`;
};

const isDicomFile = (file: File) => /\.(dcm|dicom)$/i.test(file.name);
const settingIcons = {
  inactive: encodeURI('/imgs/botton/setting.png'),
  active: encodeURI('/imgs/botton/setting click.png'),
};

export default function FolderLeaderVer2Page() {
  const navigate = useNavigate();
  const uploadFileInputRef = useRef<HTMLInputElement | null>(null);
  const uploadFolderInputRef = useRef<HTMLInputElement | null>(null);
  const uploadMenuRef = useRef<HTMLDivElement | null>(null);
  const [studies, setStudies] = useState<ServerFolderStudy[]>([]);
  const [images, setImages] = useState<ServerFolderImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [joiningStudyId, setJoiningStudyId] = useState<string | null>(null);
  const [openingImageKey, setOpeningImageKey] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<'studies' | 'images'>('studies');
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [selectedSeriesId, setSelectedSeriesId] = useState<string | null>(null);
  const [selectedImageKey, setSelectedImageKey] = useState<string | null>(null);
  const [patientIdQuery, setPatientIdQuery] = useState('');
  const [patientNameQuery, setPatientNameQuery] = useState('');
  const [studyDateQuery, setStudyDateQuery] = useState('');
  const [modalityQuery, setModalityQuery] = useState('');
  const [sexQuery, setSexQuery] = useState('');
  const [descriptionQuery, setDescriptionQuery] = useState('');
  const [rootPath, setRootPath] = useState('');
  const [rootExists, setRootExists] = useState(true);
  const [uploadMenuOpen, setUploadMenuOpen] = useState(false);
  const [uploadingLocal, setUploadingLocal] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settingsRootDraft, setSettingsRootDraft] = useState('');
  const [settingsNumberingDraft, setSettingsNumberingDraft] = useState<AppNumberingSystem>(() => readStoredNumberingSystem());
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [settingsSaving, setSettingsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
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
      setImages(data.images || []);
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

  useEffect(() => {
    if (!uploadMenuOpen) return;

    const handlePointerDown = (event: MouseEvent) => {
      if (uploadMenuRef.current?.contains(event.target as Node)) return;
      setUploadMenuOpen(false);
    };

    window.addEventListener('mousedown', handlePointerDown);
    return () => window.removeEventListener('mousedown', handlePointerDown);
  }, [uploadMenuOpen]);

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

  const mergedImages = useMemo<EnrichedServerImage[]>(() => {
    return images.map((image) => {
      const directStudy = studies.find((study) => study.id === image.linkedStudyId) || null;
      const match = directStudy ? { study: directStudy, seriesId: directStudy.series[0]?.id || null } : findBestStudyForImage(image, studies);
      const linkedStudy = match.study;
      const imagePatientName = buildImagePatientName(image.name);
      const imagePatientId = buildImagePseudoPatientId(image.relativePath);
      return {
        ...image,
        linkedStudy,
        linkedSeriesId: match.seriesId,
        patientId: imagePatientId,
        patientName: imagePatientName,
        patientSex: image.patientSex || linkedStudy?.patientSex || '',
        studyDate: image.studyDate || linkedStudy?.studyDate || '',
        modality: (image.modalities && image.modalities.length ? image.modalities.join(', ') : '') || (linkedStudy ? getStudyModality(linkedStudy) : ''),
        description: image.description || linkedStudy?.description || linkedStudy?.label || '',
      };
    });
  }, [images, studies]);

  const filteredImages = useMemo(() => {
    const patientId = patientIdQuery.trim().toLowerCase();
    const patientName = patientNameQuery.trim().toLowerCase();
    const description = descriptionQuery.trim().toLowerCase();
    const studyDate = studyDateQuery.trim().toLowerCase();
    const modality = modalityQuery.trim().toLowerCase();
    const sex = sexQuery.trim().toLowerCase();

    return mergedImages.filter((image) => {
      const haystack = [
        image.name,
        image.folderLabel,
        image.relativePath,
        image.format,
        image.patientId,
        image.patientName,
        image.patientSex,
        image.studyDate,
        image.modality,
        image.description,
      ]
        .filter(Boolean)
        .join(' ')
        .toLowerCase();

      if (patientId && !haystack.includes(patientId)) return false;
      if (patientName && !haystack.includes(patientName)) return false;
      if (description && !haystack.includes(description)) return false;
      if (studyDate && !haystack.includes(studyDate)) return false;
      if (modality && !haystack.includes(modality)) return false;
      if (sex && !haystack.includes(sex)) return false;
      return true;
    });
  }, [descriptionQuery, mergedImages, modalityQuery, patientIdQuery, patientNameQuery, sexQuery, studyDateQuery]);

  const filteredStudyRows = useMemo<StudyListEntry[]>(() => {
    const studyRows: StudyListEntry[] = filteredStudies.map((study) => ({
      kind: 'study',
      key: `study:${study.id}`,
      patientId: study.patientId || '-',
      patientName: buildPatientLabel(study),
      studyDate: study.studyDate || '-',
      description: study.description || study.label || '-',
      imageCount: study.totalFiles,
      modality: getStudyModality(study),
      comment: study.series[0]?.orientation || '-',
      study,
    }));

    const imageRows: StudyListEntry[] = filteredImages.map((image) => ({
      kind: 'image',
      key: `image:${image.relativePath}`,
      patientId: image.patientId || '-',
      patientName: image.patientName || image.patientId || image.folderLabel || '-',
      studyDate: image.studyDate || '-',
      description: image.description || image.folderLabel || image.name || '-',
      imageCount: 1,
      modality: image.modality || image.format || '-',
      comment: image.folderLabel || image.format || '-',
      image,
    }));

    return [...studyRows, ...imageRows];
  }, [filteredImages, filteredStudies]);

  const previewUrl = selectedStudy?.previewUrl ? resolveServerAssetUrl(selectedStudy.previewUrl) : null;
  const selectedImage = useMemo(
    () => mergedImages.find((image) => image.relativePath === selectedImageKey) || null,
    [mergedImages, selectedImageKey]
  );
  const selectedImagePreviewUrl = selectedImage ? resolveServerAssetUrl(selectedImage.downloadUrl) : null;
  const hasSelection = activeSection === 'images' ? Boolean(selectedImage) : Boolean(selectedStudy || selectedImage);

  useEffect(() => {
    if (!selectedImageKey) return;
    if (!filteredImages.some((image) => image.relativePath === selectedImageKey)) {
      setSelectedImageKey(null);
    }
  }, [filteredImages, selectedImageKey]);

  const selectStudy = (study: ServerFolderStudy) => {
    setSelectedImageKey(null);
    setSelectedStudyId((currentId) => (currentId === study.id ? null : study.id));
    setSelectedSeriesId((currentId) => {
      if (selectedStudy?.id === study.id) return null;
      if (study.series.some((series) => series.id === currentId)) return currentId;
      return study.series[0]?.id || null;
    });
  };

  const selectImage = (image: EnrichedServerImage) => {
    setSelectedStudyId(null);
    setSelectedSeriesId(null);
    setSelectedImageKey((currentKey) => (currentKey === image.relativePath ? null : image.relativePath));
  };

  const clearFilters = () => {
    setPatientIdQuery('');
    setPatientNameQuery('');
    setStudyDateQuery('');
    setModalityQuery('');
    setSexQuery('');
    setDescriptionQuery('');
  };

  const openStudyEntry = async (study: ServerFolderStudy, targetSeriesId?: string | null) => {
    const resolvedSeriesId = targetSeriesId || study.series[0]?.id || null;
    setJoiningStudyId(study.id);
    setError(null);

    try {
      const folderStudy: FolderStudy = await materializeServerStudy(study);
      navigate('/renew', {
        state: {
          originalFolderMode: true,
          originalFolderStudies: [folderStudy],
          folderSelectedSeriesId: resolvedSeriesId,
          previewUrl: study.previewUrl ? resolveServerAssetUrl(study.previewUrl) : undefined,
          originalIsDicom: true,
          originalFileName:
            folderStudy.series.find((series) => series.id === resolvedSeriesId)?.label ||
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

  const openStudy = async () => {
    if (!selectedStudy) return;
    await openStudyEntry(selectedStudy, selectedSeries?.id || selectedStudy.series[0]?.id || null);
  };

  const openImage = async (image: EnrichedServerImage) => {
    setOpeningImageKey(image.relativePath);
    setError(null);
    try {
      const imageUrl = resolveServerAssetUrl(image.downloadUrl);
      const response = await fetch(imageUrl);
      if (!response.ok) {
        throw new Error(`Failed to load ${image.name}.`);
      }
      const blob = await response.blob();
      const originalFile = new File([blob], image.name, {
        type: blob.type || `image/${(image.format || 'png').toLowerCase()}`,
        lastModified: Date.now(),
      });

      navigate('/renew', {
        state: {
          previewUrl: imageUrl,
          imageUrl,
          originalFile,
          originalFileName: image.name,
          originalIsDicom: false,
          folderSource: 'server-image',
          userName: image.patientName || buildImagePatientName(image.name) || 'Patient',
          linkedStudyId: image.linkedStudy?.id || null,
          linkedSeriesId: image.linkedSeriesId,
        },
      });
    } catch (nextError: any) {
      setError(nextError?.message || 'Failed to open the selected image.');
    } finally {
      setOpeningImageKey(null);
    }
  };

  const closeSelection = () => {
    setSelectedStudyId(null);
    setSelectedSeriesId(null);
    setSelectedImageKey(null);
  };

  const handleOpenSettings = async () => {
    setIsSettingsOpen(true);
    setSettingsError(null);
    setSettingsRootDraft(rootPath);
    setSettingsNumberingDraft(readStoredNumberingSystem());
    setSettingsLoading(true);
    try {
      const data = await fetchServerFolderRootPath();
      setSettingsRootDraft(data.root_path || '');
    } catch (nextError: any) {
      setSettingsError(nextError?.message || 'Failed to load current settings.');
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleCloseSettings = () => {
    setIsSettingsOpen(false);
    setSettingsError(null);
    setSettingsRootDraft(rootPath);
    setSettingsNumberingDraft(readStoredNumberingSystem());
  };

  const handleSaveSettings = async () => {
    const nextRootPath = settingsRootDraft.trim();
    if (!nextRootPath) {
      setSettingsError('Root folder path is required.');
      return;
    }

    setSettingsSaving(true);
    setSettingsError(null);
    try {
      const data = await updateServerFolderRootPath(nextRootPath);
      setRootPath(data.root_path || nextRootPath);
      setRootExists(data.root_exists !== false);
      writeStoredNumberingSystem(settingsNumberingDraft);
      setIsSettingsOpen(false);
      void loadStudies(false);
    } catch (nextError: any) {
      setSettingsError(nextError?.message || 'Failed to save settings.');
    } finally {
      setSettingsSaving(false);
    }
  };

  const handleBrowseRootFolder = async () => {
    setSettingsError(null);
    setSettingsLoading(true);
    try {
      const data = await pickServerFolderRootPath();
      setSettingsRootDraft(data.root_path || '');
    } catch (nextError: any) {
      setSettingsError(nextError?.message || 'Failed to pick a folder.');
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleLocalFilePick = (file: File | null) => {
    if (!file) return;
    setUploadMenuOpen(false);
    navigate('/renew', {
      state: {
        originalFile: file,
        originalFileName: file.name,
        originalIsDicom: isDicomFile(file),
      },
    });
  };

  const handleLocalFolderPick = async (files: File[]) => {
    if (!files.length) return;
    setUploadMenuOpen(false);
    setUploadingLocal(true);
    setError(null);
    try {
      const folderStudies = await buildDicomFolderStudies(files);
      const firstSeries = folderStudies.flatMap((study) => study.series)[0] || null;
      navigate('/renew', {
        state: {
          originalFolderMode: true,
          originalFolderStudies: folderStudies,
          folderSelectedSeriesId: firstSeries?.id || null,
          originalIsDicom: true,
          originalFileName: firstSeries?.label || files[0]?.name || 'DICOM Folder',
        },
      });
    } catch (nextError: any) {
      setError(nextError?.message || 'Failed to read the selected folder.');
    } finally {
      setUploadingLocal(false);
      if (uploadFolderInputRef.current) uploadFolderInputRef.current.value = '';
    }
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
          <div className="folder-leader-v2-db-box relative">
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
            <div className="folder-leader-v2-db-actions relative" ref={uploadMenuRef}>
              <button type="button" onClick={() => loadStudies(true)} disabled={loading || refreshing || uploadingLocal}>
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </button>
              <button
                type="button"
                onClick={() => setUploadMenuOpen((current) => !current)}
                disabled={uploadingLocal}
              >
                Upload
              </button>
              {uploadMenuOpen && (
                <div className="absolute right-0 top-full z-20 mt-2 flex min-w-[140px] flex-col overflow-hidden rounded border border-slate-300 bg-white shadow-lg">
                  <button
                    type="button"
                    className="border-b border-slate-200 px-3 py-2 text-left text-sm font-medium text-slate-700 hover:bg-slate-100"
                    onClick={() => uploadFileInputRef.current?.click()}
                  >
                    Upload File
                  </button>
                  <button
                    type="button"
                    className="px-3 py-2 text-left text-sm font-medium text-slate-700 hover:bg-slate-100"
                    onClick={() => uploadFolderInputRef.current?.click()}
                  >
                    Upload Folder
                  </button>
                </div>
              )}
              <input
                ref={uploadFileInputRef}
                type="file"
                accept="image/*,.dcm,.dicom,application/dicom"
                hidden
                onChange={(event) => {
                  handleLocalFilePick(event.target.files?.[0] || null);
                  event.currentTarget.value = '';
                }}
              />
              <input
                type="file"
                hidden
                multiple
                ref={(node) => {
                  uploadFolderInputRef.current = node;
                  if (!node) return;
                  node.setAttribute('webkitdirectory', '');
                  node.setAttribute('directory', '');
                }}
                onChange={(event) => {
                  void handleLocalFolderPick(Array.from(event.target.files || []));
                }}
              />
            </div>
            <button
              type="button"
              onClick={() => { void handleOpenSettings(); }}
              aria-pressed={isSettingsOpen}
              aria-label="Open settings"
              title="Settings"
              style={{
                position: 'absolute',
                right: '10px',
                bottom: '10px',
                width: '38px',
                height: '38px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                border: '1px solid #6D6D6D',
                background: '#3E3E3E',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.08)',
                padding: 0,
                cursor: 'pointer',
              }}
            >
              <img
                src={isSettingsOpen ? settingIcons.active : settingIcons.inactive}
                alt=""
                draggable={false}
                style={{ width: '24px', height: '24px', objectFit: 'contain', display: 'block' }}
              />
            </button>
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
                <span>{formatCount(filteredStudyRows.length, 'result')}</span>
                <span>{formatCount(studies.reduce((sum, study) => sum + study.totalSeries, 0), 'series')}</span>
                <span>{formatCount(filteredImages.length, 'image')}</span>
              </div>
              <div className="folder-leader-v2-search-buttons">
                <button type="button" onClick={() => setActiveSection('studies')} className={activeSection === 'studies' ? 'is-active' : ''}>
                  Studies
                </button>
                <button type="button" onClick={() => setActiveSection('images')} className={activeSection === 'images' ? 'is-active' : ''}>
                  Images
                </button>
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

        <section className={`folder-leader-v2-workspace ${hasSelection ? 'has-selection' : ''}`}>
          {activeSection === 'images' ? (
            <>
              <div className="folder-leader-v2-table-panel">
                <div className="folder-leader-v2-table-header" style={{ gridTemplateColumns: '1.2fr 1.1fr 0.8fr 1fr 0.7fr 0.6fr 0.5fr' }}>
                  <div>Name</div>
                  <div>Patient</div>
                  <div>Date</div>
                  <div>Folder</div>
                  <div>Format</div>
                  <div>Size</div>
                  <div>Open</div>
                </div>
                <div className="folder-leader-v2-table-body">
                  {loading ? (
                    <div className="folder-leader-v2-empty">Scanning image files...</div>
                  ) : filteredImages.length === 0 ? (
                    <div className="folder-leader-v2-empty">No matching image files.</div>
                  ) : (
                    filteredImages.map((image) => {
                      const isSelected = image.relativePath === selectedImage?.relativePath;
                      return (
                        <div
                          key={image.relativePath}
                          className={`folder-leader-v2-table-row ${isSelected ? 'is-selected' : ''}`}
                          style={{ gridTemplateColumns: '1.2fr 1.1fr 0.8fr 1fr 0.7fr 0.6fr 0.5fr', alignItems: 'center' }}
                          onClick={() => selectImage(image)}
                          onDoubleClick={() => void openImage(image)}
                        >
                          <div className="folder-leader-v2-cell-name" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            <ImageIcon className="h-4 w-4" />
                            {image.name}
                          </div>
                          <div>{image.patientName || image.patientId || '-'}</div>
                          <div>{image.studyDate || '-'}</div>
                          <div>{image.folderLabel || '-'}</div>
                          <div>{image.format || '-'}</div>
                          <div>{image.size ? `${Math.max(1, Math.round(image.size / 1024))} KB` : '-'}</div>
                          <div>
                            <button
                              type="button"
                              onClick={(event) => {
                                event.stopPropagation();
                                void openImage(image);
                              }}
                              disabled={openingImageKey === image.relativePath}
                              className="folder-leader-v2-join"
                            >
                              {openingImageKey === image.relativePath ? 'Opening...' : 'Open'}
                            </button>
                          </div>
                        </div>
                      );
                    })
                  )}
                </div>
              </div>
              {selectedImage ? (
                <div className="folder-leader-v2-detail-panel">
                  <div className="folder-leader-v2-detail-head">
                    <div>
                      <div className="folder-leader-v2-detail-title">Selected Image</div>
                      <div className="folder-leader-v2-detail-name">{selectedImage.name}</div>
                    </div>
                    <div className="folder-leader-v2-detail-actions">
                      <button
                        type="button"
                        onClick={() => void openImage(selectedImage)}
                        disabled={openingImageKey === selectedImage.relativePath}
                        className="folder-leader-v2-join"
                      >
                        {openingImageKey === selectedImage.relativePath ? 'Opening...' : 'Join'}
                        <ArrowRight className="h-4 w-4" />
                      </button>
                    </div>
                  </div>

                  <div className="folder-leader-v2-detail-body">
                    <div className="folder-leader-v2-preview-box">
                      {selectedImagePreviewUrl ? (
                        <img
                          src={selectedImagePreviewUrl}
                          alt={selectedImage.name}
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
                        <span>Patient</span>
                        <strong>{selectedImage.patientName || selectedImage.patientId || '-'}</strong>
                      </div>
                      <div>
                        <span>Date Created</span>
                        <strong>{selectedImage.studyDate || '-'}</strong>
                      </div>
                      <div>
                        <span>Format</span>
                        <strong>{selectedImage.format || '-'}</strong>
                      </div>
                      <div>
                        <span>Size</span>
                        <strong>{selectedImage.size ? `${Math.max(1, Math.round(selectedImage.size / 1024))} KB` : '-'}</strong>
                      </div>
                      <div>
                        <span>Dimensions</span>
                        <strong>{selectedImage.width && selectedImage.height ? `${selectedImage.width} x ${selectedImage.height}` : '-'}</strong>
                      </div>
                      <div>
                        <span>Folder</span>
                        <strong>{selectedImage.folderLabel || '-'}</strong>
                      </div>
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          ) : (
            <>
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
              ) : filteredStudyRows.length === 0 ? (
                <div className="folder-leader-v2-empty">No matching studies or images.</div>
              ) : (
                filteredStudyRows.map((entry) => {
                  if (entry.kind === 'image') {
                    const image = entry.image;
                    return (
                      <button
                        key={entry.key}
                        type="button"
                        onClick={() => selectImage(image)}
                        onDoubleClick={() => void openImage(image)}
                        className={`folder-leader-v2-table-row ${selectedImage?.relativePath === image.relativePath ? 'is-selected' : ''}`}
                      >
                        <div className="folder-leader-v2-cell-id">{entry.patientId}</div>
                        <div className="folder-leader-v2-cell-name">{entry.patientName}</div>
                        <div>{entry.studyDate}</div>
                        <div className="folder-leader-v2-cell-description">{entry.description}</div>
                        <div>{entry.imageCount}</div>
                        <div>{entry.modality}</div>
                        <div>{entry.comment}</div>
                      </button>
                    );
                  }

                  const study = entry.study;
                  const isSelected = study.id === selectedStudy?.id;
                  return (
                    <button
                      key={entry.key}
                      type="button"
                      onClick={() => selectStudy(study)}
                      onDoubleClick={() => void openStudyEntry(study, study.series[0]?.id || null)}
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

          {selectedImage ? (
            <div className="folder-leader-v2-detail-panel">
              <div className="folder-leader-v2-detail-head">
                <div>
                  <div className="folder-leader-v2-detail-title">Selected Image</div>
                  <div className="folder-leader-v2-detail-name">{selectedImage.name}</div>
                </div>
                <div className="folder-leader-v2-detail-actions">
                  <button
                    type="button"
                    onClick={() => void openImage(selectedImage)}
                    disabled={openingImageKey === selectedImage.relativePath}
                    className="folder-leader-v2-join"
                  >
                    {openingImageKey === selectedImage.relativePath ? 'Opening...' : 'Join'}
                    <ArrowRight className="h-4 w-4" />
                  </button>
                </div>
              </div>

              <div className="folder-leader-v2-detail-body">
                <div className="folder-leader-v2-preview-box">
                  {selectedImagePreviewUrl ? (
                    <img
                      src={selectedImagePreviewUrl}
                      alt={selectedImage.name}
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
                    <span>Patient</span>
                    <strong>{selectedImage.patientName || selectedImage.patientId || '-'}</strong>
                  </div>
                  <div>
                    <span>Date Created</span>
                    <strong>{selectedImage.studyDate || '-'}</strong>
                  </div>
                  <div>
                    <span>Format</span>
                    <strong>{selectedImage.format || '-'}</strong>
                  </div>
                  <div>
                    <span>Size</span>
                    <strong>{selectedImage.size ? `${Math.max(1, Math.round(selectedImage.size / 1024))} KB` : '-'}</strong>
                  </div>
                  <div>
                    <span>Dimensions</span>
                    <strong>{selectedImage.width && selectedImage.height ? `${selectedImage.width} x ${selectedImage.height}` : '-'}</strong>
                  </div>
                  <div>
                    <span>Folder</span>
                    <strong>{selectedImage.folderLabel || '-'}</strong>
                  </div>
                </div>
              </div>
            </div>
          ) : selectedStudy ? (
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
                        onDoubleClick={() => void openStudyEntry(selectedStudy, series.id)}
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
            </>
          )}
        </section>
      </main>

      <SettingsModal
        visible={isSettingsOpen}
        rootFolderPath={settingsRootDraft}
        onRootFolderPathChange={setSettingsRootDraft}
        onBrowseRootFolder={() => { void handleBrowseRootFolder(); }}
        numberingSystem={settingsNumberingDraft}
        onNumberingSystemChange={setSettingsNumberingDraft}
        onClose={handleCloseSettings}
        onSave={() => { void handleSaveSettings(); }}
        saving={settingsSaving}
        loading={settingsLoading}
        error={settingsError}
      />
    </div>
  );
}
