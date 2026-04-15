import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLocation } from 'react-router-dom';
import O3Logo from '../assets/O3_logo_only.png';
import { StudiesWorkspacePanel } from '../components/chart/StudiesWorkspacePanel';
import { WebReportDrawer } from '../components/WebReportDrawer';
import { createWebReportFromChart } from '../lib/webReportApi';
import { fetchServerFolderIndex, materializeServerStudy } from '../lib/folderLeaderApi';
import type { FolderStudy } from '../features/upload/dicomFolderStudies';
import { clearAllAnnotations, setActiveTool as setCornerstoneActiveTool } from '../viewer/cornerstone/tools';

const DESIGN_WIDTH = 1920;
const DESIGN_HEIGHT = 1080;

const wp = (value: number) => `${value}px`;
const hp = (value: number) => `${value}px`;
const scalePx = (value: number) => `${value}px`;
const relativePercent = (value: number, total: number) => `${(value / total) * 100}%`;

const assetPath = (relativePath: string) => encodeURI(`/imgs/${relativePath}`);
const headerMarkerIcon = assetPath('7 7.png');

const reportButtonIcons = {
  inactive: assetPath('메인-비활성화 아이콘/report버튼 (94 94).png'),
  active: assetPath('메인-클릭/report버튼 (94 94).png'),
};

const railIcons = {
  studies: assetPath('메인-클릭/좌측 bar 아이콘 (51 57)/자산 425@4x.png'),
  captures: assetPath('메인-비활성화 아이콘/좌측 bar 아이콘 (51 57)/자산 584.png'),
  report: assetPath('메인-비활성화 아이콘/report버튼 (94 94).png'),
};

const toolbarIcons = [
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 20@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 21@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 22@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 23@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 24@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 26@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 31@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 30@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 25@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 29@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 28@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 27@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 35@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 34@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 33@4x.png'),
  assetPath('메인-비활성화 아이콘/tool 아이콘 (36 36)/자산 32@4x.png'),
];

const displayRailIcons = {
  studies: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 427@4x.png'),
  captures: assetPath('mian_deactive/left_bar_deactive (51 57)/자산 584.png'),
};

const activeRailIcons = {
  studies: assetPath('main_active/left_bar_active (51 57)/자산 425@4x.png'),
  captures: assetPath('main_active/left_bar_active (51 57)/자산 587.png'),
};

const displayToolbarIcons = [
  assetPath('mian_deactive/tools_deactive(36 36)/자산 20@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 21@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 22@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 23@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 24@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 26@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 31@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 30@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 25@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 29@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 28@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 27@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 35@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 34@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 33@4x.png'),
  assetPath('mian_deactive/tools_deactive(36 36)/자산 32@4x.png'),
];

const activeToolbarIcons = [
  assetPath('main_active/tools_active (36 36)/자산 399@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 401@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 400@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 398@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 397@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 391@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 396@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 395@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 386@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 394@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 393@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 392@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 390@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 389@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 388@4x.png'),
  assetPath('main_active/tools_active (36 36)/자산 387@4x.png'),
];

const displayReportButtonIcons = {
  inactive: assetPath('mian_deactive/report_deactive.png'),
  active: assetPath('main_active/report_active (94 94).png'),
};

const legendItems = [
  { label: 'Urgent Priority', color: '#FF0037', top: 834 },
  { label: 'Treatment Required', color: '#FCFF2A', top: 867 },
  { label: 'Implant', color: '#003DFF', top: 900 },
  { label: 'Missing Tooth', color: '#3F3F3F', top: 933 },
  { label: 'Healthy Tooth', color: '#FFFFFF', top: 966 },
] as const;

const upperSizes = ['23 79', '20 75', '22 82', '25 74', '25 67', '38 58', '36 54', '35 49'];
const lowerSizes = ['18 71', '17 70', '22 78', '25 75', '26 76', '39 68', '38 65', '41 63'];
const upperLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const upperRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const lowerLeftOrders = [8, 7, 6, 5, 4, 3, 2, 1] as const;
const lowerRightOrders = [1, 2, 3, 4, 5, 6, 7, 8] as const;
const upperTeeth = [18, 17, 16, 15, 14, 13, 12, 11] as const;
const upperRightTeeth = [21, 22, 23, 24, 25, 26, 27, 28] as const;
const lowerTeeth = [48, 47, 46, 45, 44, 43, 42, 41] as const;
const lowerRightTeeth = [31, 32, 33, 34, 35, 36, 37, 38] as const;
const outerToInnerOffsets = [390.5, 332.5, 274.5, 216.5, 166.5, 116.5, 66.5, 22.5] as const;
const innerToOuterOffsets = [22.5, 66.5, 116.5, 166.5, 216.5, 274.5, 332.5, 390.5] as const;
const upperBaseline = 935;
const lowerTop = 953;
const chartSectionTop = 804;
const chartHeaderHeight = 18;
const chartContentTop = chartSectionTop + chartHeaderHeight;
const chartContentHeight = 246;
const chartLegendLeft = 222;
const chartLegendTop = 844;
const chartLegendHeight = 151;
const RAIL_ICON_WIDTH = 61;
const RAIL_ICON_HEIGHT = 68;
const TOOL_ICON_SIZE = 43;

type ToolbarKey =
  | 'pointer'
  | 'pan'
  | 'wlww'
  | 'invert'
  | 'magnifier'
  | 'flip'
  | 'measure-length'
  | 'measure-draw'
  | 'measure-eraser'
  | 'measure-clear'
  | 'measure-rotate'
  | 'measure-reset'
  | 'output-capture'
  | 'output-save'
  | 'task-original'
  | 'task-heatmap';

function getToothAsset(tooth: number) {
  const index = (tooth % 10) - 1;
  const upper = tooth >= 11 && tooth <= 28;
  const prefix = upper ? 'U' : 'L';
  const size = upper ? upperSizes[index] : lowerSizes[index];
  return assetPath(`치아/건강치(ffffff)/${prefix}-${index + 1} (${size})_4.png`);
}

function ToolIcon({
  icon,
  activeIcon,
  left,
  top,
  active = false,
  onClick,
  label,
}: {
  icon: string;
  activeIcon?: string;
  left: number;
  top: number;
  active?: boolean;
  onClick?: () => void;
  label?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      aria-label={label}
      style={{
        width: wp(TOOL_ICON_SIZE),
        height: hp(TOOL_ICON_SIZE),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        border: `${scalePx(1)} solid ${active ? '#00C0F3' : 'black'}`,
        background: active ? '#A8A8A8' : '#8D8D8D',
        boxShadow: active ? `0 0 ${scalePx(8)} rgba(0, 192, 243, 0.35)` : 'none',
        cursor: 'pointer',
        padding: 0,
      }}
    >
      <img
        src={active && activeIcon ? activeIcon : icon}
        alt=""
        draggable={false}
        style={{ width: '100%', height: '100%', position: 'absolute', left: 0, top: 0 }}
      />
    </button>
  );
}

function ToothImage({
  tooth,
  left,
  top,
  width,
  height,
  rotate = false,
}: {
  tooth: number;
  left: number;
  top: number;
  width: number;
  height: number;
  rotate?: boolean;
}) {
  return (
    <img
      src={getToothAsset(tooth)}
      alt=""
      draggable={false}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        transform: rotate ? `translate(${width}px, ${height}px) rotate(180deg)` : undefined,
        transformOrigin: rotate ? 'top left' : undefined,
      }}
    />
  );
}

function getHealthToothAsset(arch: 'U' | 'L', order: number) {
  const size = arch === 'U' ? upperSizes[order - 1] : lowerSizes[order - 1];
  return assetPath(`teeth/health(ffffff)/${arch}-${order} (${size})_4.png`);
}

function ToothSlotImage({
  arch,
  order,
  left,
  top,
  width,
  height,
  flipX = false,
}: {
  arch: 'U' | 'L';
  order: number;
  left: number;
  top: number;
  width: number;
  height: number;
  flipX?: boolean;
}) {
  return (
    <img
      src={getHealthToothAsset(arch, order)}
      alt=""
      draggable={false}
      style={{
        width: wp(width),
        height: hp(height),
        left: wp(left),
        top: hp(top),
        position: 'absolute',
        transform: flipX ? `translate(${width}px, 0) scaleX(-1)` : undefined,
        transformOrigin: flipX ? 'top left' : undefined,
      }}
    />
  );
}

function useViewportSize() {
  const [viewport, setViewport] = useState(() => ({
    width: typeof window === 'undefined' ? DESIGN_WIDTH : window.innerWidth,
    height: typeof window === 'undefined' ? DESIGN_HEIGHT : window.innerHeight,
  }));

  useEffect(() => {
    const updateViewport = () => {
      setViewport({
        width: window.innerWidth,
        height: window.innerHeight,
      });
    };

    updateViewport();
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);

  return viewport;
}

export function RenewPage() {
  const location = useLocation();
  const locationState = (location.state as any) || {};
  const result = locationState?.result;
  const [activeFolderStudies, setActiveFolderStudies] = useState<FolderStudy[]>(() => {
    const raw = (locationState.originalFolderStudies as FolderStudy[] | undefined) || [];
    const seen = new Set<string>();
    return raw.filter((study) => {
      if (seen.has(study.id)) return false;
      seen.add(study.id);
      return true;
    });
  });
  const [serverStudies, setServerStudies] = useState<any[]>([]);
  const [isReportActive, setIsReportActive] = useState(false);
  const [isChartVisible, setIsChartVisible] = useState(true);
  const [workspaceSection, setWorkspaceSection] = useState<'studies' | 'report' | 'none'>('studies');
  const [selectedToolbarButton, setSelectedToolbarButton] = useState<ToolbarKey>('pointer');
  const [flashToolbarButton, setFlashToolbarButton] = useState<ToolbarKey | null>(null);
  const [reportSessionId, setReportSessionId] = useState<string | null>(locationState?.reportSessionId || null);
  const [reportDrawerOpen, setReportDrawerOpen] = useState(false);
  const [reportStartState, setReportStartState] = useState<'idle' | 'creating'>('idle');
  const [reportError, setReportError] = useState<string | null>(null);
  const [inverted, setInverted] = useState(false);
  const [flipped, setFlipped] = useState(false);
  const [viewMode, setViewMode] = useState<'original' | 'heatmap'>('original');
  const [selectedFolderSeriesId, setSelectedFolderSeriesId] = useState<string | null>(
    locationState.folderSelectedSeriesId || activeFolderStudies.flatMap((study) => study.series)[0]?.id || null
  );
  const viewport = useViewportSize();
  const cacheBuster = useMemo(() => Date.now(), []);
  const scale = Math.min(viewport.width / DESIGN_WIDTH, viewport.height / DESIGN_HEIGHT);
  const stageWidth = viewport.width;
  const stageHeight = DESIGN_HEIGHT * scale;
  const designCanvasWidth = stageWidth / scale;
  const viewerExtraWidth = Math.max(0, designCanvasWidth - DESIGN_WIDTH);
  const viewerLeft = 237;
  const topBarLeft = viewerLeft;
  const viewerWidth = 1676 + viewerExtraWidth;
  const topBarWidth = viewerWidth;
  const rightEdge = viewerLeft + viewerWidth;
  const chartSectionLeft = 239;
  const chartSectionWidth = rightEdge - chartSectionLeft;
  const chartLegendDividerX = chartSectionLeft + chartSectionWidth * 0.15;
  const chartLegendWidth = chartLegendDividerX - chartLegendLeft - 28;
  const reportLeft = rightEdge - 106;
  const odontoFrameLeft = chartLegendDividerX + 36;
  const odontoFrameRight = reportLeft - 58;
  const chartOdontoCenterX = (odontoFrameLeft + odontoFrameRight) / 2;
  const toothSlotCentersLeft = outerToInnerOffsets.map((offset) => chartOdontoCenterX - offset);
  const toothSlotCentersRight = innerToOuterOffsets.map((offset) => chartOdontoCenterX + offset);
  const chartOdontoFramePaddingY = 18;
  const chartOdontoFrameTop = chartContentTop + chartOdontoFramePaddingY;
  const chartOdontoFrameBottom = chartContentTop + chartContentHeight - chartOdontoFramePaddingY;
  const chartOdontoLineLeft = toothSlotCentersLeft[0] - 24;
  const chartOdontoLineRight = toothSlotCentersRight[toothSlotCentersRight.length - 1] + 24;
  const chartOdontoLineWidth = chartOdontoLineRight - chartOdontoLineLeft;
  const chartOdontoLineY = Math.round(chartContentTop + chartContentHeight / 2);
  const chartOdontoVerticalTop = chartOdontoFrameTop;
  const chartOdontoVerticalHeight = chartOdontoFrameBottom - chartOdontoFrameTop;
  const viewLabelTop = 75;
  const measureLabelTop = 195;
  const outputLabelTop = 315;
  const sectionArrowOffsetY = 6;
  const panoFrameHeight = isChartVisible ? 755 : 1019;
  const reportTop = chartContentTop + 130;
  const fdiLeft = rightEdge - 82;
  const lLabelLeft = rightEdge - 29;
  const chartHeaderHideLeft = rightEdge - 28;
  const panoChartToggleLeft = viewerLeft + 8;
  const panoChartToggleTop = 1044;
  const panoLabelTop = Math.round(49 + panoFrameHeight / 2 - 7);
  const isChartBodyVisible = isChartVisible;
  const panoBodyTop = 68;
  const panoBodyHeight = panoFrameHeight - 19;
  const panoBodyWidth = viewerWidth - 2;
  const studiesPanelLeft = viewerLeft + 16;
  const studiesPanelTop = 74;
  const studiesPanelWidth = 252;
  const studiesPanelHeight = 356;
  const getUrlWithCacheBuster = (url?: string | null) => {
    if (!url) return null;
    if (url.startsWith('blob:') || url.startsWith('data:')) return url;
    return `${url}${url.includes('?') ? '&' : '?'}t=${cacheBuster}`;
  };
  const originalPanoUrl = getUrlWithCacheBuster(
    result?.preview_url || locationState.previewUrl || result?.image_url || locationState.imageUrl || null
  );
  const heatmapPanoUrl = getUrlWithCacheBuster(
    result?.heatmap_overlay_url || result?.overlay_url || result?.preview_url || locationState.previewUrl || null
  );
  const panoViewerUrl = viewMode === 'heatmap' ? heatmapPanoUrl || originalPanoUrl : originalPanoUrl;
  const combinedStudies = useMemo(() => {
    const activeIds = new Set(activeFolderStudies.map((study) => study.id));
    const activeFingerprints = new Set(activeFolderStudies.map((study) => `${study.label}::${study.description}::${study.patientId}`));
    const additional = serverStudies.filter((study) => {
      if (activeIds.has(study.id)) return false;
      const fingerprint = `${study.label}::${study.description}::${study.patientId}`;
      if (activeFingerprints.has(fingerprint)) return false;
      return true;
    });
    const merged = [...activeFolderStudies, ...additional];
    const seen = new Set<string>();
    return merged.filter((study) => {
      if (seen.has(study.id)) return false;
      seen.add(study.id);
      return true;
    });
  }, [activeFolderStudies, serverStudies]);

  const handleOpenStudies = () => {
    setWorkspaceSection((current) => (current === 'studies' ? 'none' : 'studies'));
    setReportDrawerOpen(false);
    setIsReportActive(false);
    setReportError(null);
  };

  const handleSelectSeries = async (seriesId: string) => {
    const existingSeries = activeFolderStudies.flatMap((study) => study.series).find((series) => series.id === seriesId);
    if (existingSeries) {
      setSelectedFolderSeriesId(seriesId);
      handleOpenStudies();
      return;
    }

    const targetStudy = serverStudies.find((study) => study.series.some((series: any) => series.id === seriesId));
    if (!targetStudy) return;

    try {
      const materialized = await materializeServerStudy(targetStudy);
      setActiveFolderStudies((current) => {
        if (current.some((study) => study.id === materialized.id)) return current;
        return [...current, materialized];
      });
      setSelectedFolderSeriesId(seriesId);
      handleOpenStudies();
    } catch (error) {
      console.error('Failed to materialize study inside RenewPage', error);
    }
  };

  const handleChartToggle = () => {
    setIsChartVisible((current) => !current);
  };

  useEffect(() => {
    if (locationState.folderSource === 'server') {
      fetchServerFolderIndex()
        .then((data) => setServerStudies(data.studies || []))
        .catch(console.error);
    }
  }, [locationState.folderSource]);

  const flashToolbarActive = (button: ToolbarKey) => {
    setFlashToolbarButton(button);
    window.setTimeout(() => {
      setFlashToolbarButton((current) => (current === button ? null : current));
    }, 180);
  };

  const handleToolSelection = (button: ToolbarKey) => {
    switch (button) {
      case 'pointer':
        setCornerstoneActiveTool('Pan');
        setSelectedToolbarButton('pointer');
        return;
      case 'pan':
        setCornerstoneActiveTool('Pan');
        setSelectedToolbarButton('pan');
        return;
      case 'wlww':
        setCornerstoneActiveTool('WindowLevel');
        setSelectedToolbarButton('wlww');
        return;
      case 'invert':
        setInverted((current) => !current);
        setSelectedToolbarButton((current) => (current === 'invert' ? 'pointer' : 'invert'));
        return;
      case 'magnifier':
        setCornerstoneActiveTool('Pan');
        setSelectedToolbarButton((current) => (current === 'magnifier' ? 'pointer' : 'magnifier'));
        return;
      case 'flip':
        setFlipped((current) => !current);
        setSelectedToolbarButton((current) => (current === 'flip' ? 'pointer' : 'flip'));
        return;
      case 'measure-length':
        setCornerstoneActiveTool('Length');
        setSelectedToolbarButton('measure-length');
        return;
      case 'measure-draw':
        setCornerstoneActiveTool('ArrowAnnotate');
        setSelectedToolbarButton('measure-draw');
        return;
      case 'measure-eraser':
        setCornerstoneActiveTool('Eraser');
        setSelectedToolbarButton('measure-eraser');
        return;
      case 'measure-clear':
        clearAllAnnotations();
        flashToolbarActive('measure-clear');
        setSelectedToolbarButton('pointer');
        return;
      case 'measure-rotate':
        setCornerstoneActiveTool('TrackballRotate');
        setSelectedToolbarButton('measure-rotate');
        return;
      case 'measure-reset':
        clearAllAnnotations();
        setInverted(false);
        setFlipped(false);
        setViewMode('original');
        flashToolbarActive('measure-reset');
        setSelectedToolbarButton('pointer');
        setReportError(null);
        setCornerstoneActiveTool('Pan');
        return;
      case 'output-capture':
      case 'output-save':
        setSelectedToolbarButton(button);
        return;
      case 'task-original':
        setViewMode('original');
        setSelectedToolbarButton('task-original');
        return;
      case 'task-heatmap':
        setViewMode('heatmap');
        setSelectedToolbarButton('task-heatmap');
        return;
      default:
        return;
    }
  };

  const handleStartReport = async () => {
    setWorkspaceSection('report');

    if (reportStartState === 'creating') return;

    if (reportSessionId) {
      setReportDrawerOpen((current) => {
        const next = !current;
        setIsReportActive(next);
        setWorkspaceSection(next ? 'report' : 'none');
        return next;
      });
      setReportError(null);
      return;
    }

    if (!result) {
      setReportError('Analysis result is not ready yet.');
      return;
    }

    setReportStartState('creating');
    setReportError(null);
    try {
      const response = await createWebReportFromChart({
        result,
        source_url: result?.image_url,
        overlay_url: result?.overlay_url,
        language: 'English',
      });
      setReportSessionId(response.session_id);
      setReportDrawerOpen(true);
      setIsReportActive(true);
    } catch (error: any) {
      setReportError(error?.message || 'Failed to start report workspace');
      setIsReportActive(false);
    } finally {
      setReportStartState('idle');
    }
  };

  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#000000',
        overflow: 'hidden',
        fontFamily: '"Noto Sans", "Noto Sans KR", sans-serif',
      }}
    >
      <div
        style={{
          width: `${stageWidth}px`,
          height: `${stageHeight}px`,
          position: 'relative',
          overflow: 'hidden',
          background: '#414950',
          boxShadow: 'inset 1px 0 0 rgba(0, 0, 0, 0.18), inset -1px 0 0 rgba(0, 0, 0, 0.18)',
        }}
      >
        <div
          style={{
            width: wp(designCanvasWidth),
            height: hp(DESIGN_HEIGHT),
            left: 0,
            top: 0,
            position: 'absolute',
            background: '#414950',
            transform: `scale(${scale})`,
            transformOrigin: 'top left',
          }}
        >
          <div style={{ width: wp(designCanvasWidth), height: hp(DESIGN_HEIGHT), left: 0, top: 0, position: 'absolute', background: '#414950' }} />

          <img
            src={O3Logo}
            alt="O3"
            draggable={false}
            style={{
              width: wp(144),
              height: hp(38),
              left: wp(-20),
              top: hp(10),
              position: 'absolute',
              objectFit: 'contain',
              zIndex: 12,
            }}
          />
          <div
            style={{
              left: wp(88),
              top: hp(12),
              position: 'absolute',
              color: 'white',
              fontSize: scalePx(20),
              fontWeight: 700,
              letterSpacing: '0.08em',
              zIndex: 12,
            }}
          >
            SATURN
          </div>

          <div style={{ width: wp(topBarWidth), height: hp(18), left: wp(topBarLeft), top: hp(49), position: 'absolute', background: '#5C5C5C', zIndex: 1 }} />
          <div style={{ width: wp(viewerWidth), height: hp(1018), left: wp(viewerLeft), top: hp(50), position: 'absolute', background: 'black' }} />
          {isChartVisible && (
            <div style={{ width: wp(viewerWidth), height: hp(18), left: wp(viewerLeft), top: hp(804), position: 'absolute', background: '#5C5C5C', zIndex: 1 }} />
          )}
          <div style={{ width: wp(viewerWidth), height: hp(panoFrameHeight), left: wp(viewerLeft), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
          {isChartVisible && (
            <div
              style={{
                width: wp(viewerWidth),
                height: hp(264),
                left: wp(viewerLeft),
                top: hp(804),
                position: 'absolute',
                border: `${scalePx(1)} solid #4C4C4C`,
                pointerEvents: 'none',
              }}
            />
          )}
          <div style={{ width: scalePx(1), height: hp(1019), left: wp(viewerLeft), top: hp(49), position: 'absolute', background: '#4C4C4C' }} />
          <div
            style={{
              width: wp(panoBodyWidth),
              height: hp(panoBodyHeight),
              left: wp(viewerLeft + 1),
              top: hp(panoBodyTop),
              position: 'absolute',
              overflow: 'hidden',
              background: '#000000',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {panoViewerUrl ? (
              <img
                src={panoViewerUrl}
                alt="Panorama"
                draggable={false}
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'contain',
                  filter: `invert(${inverted ? 1 : 0})`,
                  transform: flipped ? 'scaleX(-1)' : 'none',
                  transformOrigin: 'center',
                  opacity: viewMode === 'heatmap' ? 0.96 : 1,
                }}
              />
            ) : (
              <div
                style={{
                  color: '#7A7A7A',
                  fontSize: scalePx(14),
                  fontWeight: 700,
                  letterSpacing: '0.04em',
                }}
              >
                No panorama source
              </div>
            )}
          </div>

        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(70), height: hp(1019), left: wp(12), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        {/* 먼 세로선이지 */} 
        {/*<div style={{ width: scalePx(1), height: hp(1019), left: wp(48), top: hp(49), position: 'absolute', background: '#3F3F3F' }} />  
        <div style={{ width: scalePx(1), height: hp(1019), left: wp(68), top: hp(49), position: 'absolute', background: '#5C5C5C' }} /> */}
        <button
          type="button"
          onClick={handleOpenStudies}
          aria-pressed={workspaceSection === 'studies'}
          style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(52), position: 'absolute', background: '#2D2D2D', padding: 0, cursor: 'pointer' }}
        >
          <img style={{ width: '100%', height: '100%', display: 'block' }} src={workspaceSection === 'studies' ? activeRailIcons.studies : displayRailIcons.studies} alt="" />
        </button>
        <button
          type="button"
          onClick={() => { void handleStartReport(); }}
          aria-pressed={workspaceSection === 'report' || reportDrawerOpen}
          style={{ width: wp(RAIL_ICON_WIDTH), height: hp(RAIL_ICON_HEIGHT), left: wp(16), top: hp(122), position: 'absolute', background: '#2D2D2D', padding: 0, cursor: 'pointer' }}
        >
          <img style={{ width: '100%', height: '100%', display: 'block' }} src={workspaceSection === 'report' || reportDrawerOpen ? activeRailIcons.captures : displayRailIcons.captures} alt="" />
        </button>

        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', background: '#2D2D2D' }} />
        <div style={{ width: wp(154), height: hp(1019), left: wp(82), top: hp(49), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
        <div style={{ width: wp(148), height: hp(336), left: wp(86), top: hp(49), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(148), height: hp(84), left: wp(86), top: hp(389), position: 'absolute', background: '#333333' }} />
        <div style={{ width: wp(140), height: hp(116), left: wp(88), top: hp(69), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(118), left: wp(88), top: hp(189), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(72), left: wp(88), top: hp(309), position: 'absolute', background: '#414141' }} />
        <div style={{ width: wp(140), height: hp(52), left: wp(88), top: hp(411), position: 'absolute', background: '#414141' }} />

        <div style={{ left: wp(109), top: hp(50), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>General</div>
        <div style={{ left: wp(109), top: hp(viewLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(10), fontWeight: 700 }}>View</div>
        <div style={{ left: wp(109), top: hp(measureLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Measure</div>
        <div style={{ left: wp(109), top: hp(outputLabelTop), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700 }}>Output</div>
        <div style={{ left: wp(109), top: hp(390), position: 'absolute', color: 'white', fontSize: scalePx(14), fontWeight: 700 }}>Task</div>

        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(57), position: 'absolute', background: '#D9D9D9' }} />
        <div style={{ width: scalePx(7), height: scalePx(7), left: wp(96), top: hp(397), position: 'absolute', background: '#D9D9D9' }} />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(viewLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(measureLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />
        <div
          style={{
            width: 0,
            height: 0,
            left: wp(96),
            top: hp(outputLabelTop + sectionArrowOffsetY),
            position: 'absolute',
            borderLeft: `${scalePx(4)} solid transparent`,
            borderRight: `${scalePx(4)} solid transparent`,
            borderTop: `${scalePx(8)} solid #D9D9D9`,
          }}
        />

        {[
          { key: 'pointer' as ToolbarKey, x: 91.5, y: 96, label: 'Mouse', active: selectedToolbarButton === 'pointer' },
          { key: 'pan' as ToolbarKey, x: 136.5, y: 96, label: 'Zoom and drag', active: selectedToolbarButton === 'pan' },
          { key: 'wlww' as ToolbarKey, x: 181.5, y: 96, label: 'Window level', active: selectedToolbarButton === 'wlww' },
          { key: 'invert' as ToolbarKey, x: 91.5, y: 141, label: 'Invert', active: inverted },
          { key: 'magnifier' as ToolbarKey, x: 136.5, y: 141, label: 'Magnification', active: selectedToolbarButton === 'magnifier' },
          { key: 'flip' as ToolbarKey, x: 181.5, y: 141, label: 'Flip', active: flipped },
          { key: 'measure-length' as ToolbarKey, x: 91.5, y: 216, label: 'Ruler', active: selectedToolbarButton === 'measure-length' },
          { key: 'measure-draw' as ToolbarKey, x: 136.5, y: 216, label: 'Draw', active: selectedToolbarButton === 'measure-draw' },
          { key: 'measure-eraser' as ToolbarKey, x: 181.5, y: 216, label: 'Eraser', active: selectedToolbarButton === 'measure-eraser' },
          { key: 'measure-clear' as ToolbarKey, x: 91.5, y: 261, label: 'Delete all measure', active: flashToolbarButton === 'measure-clear' },
          { key: 'measure-rotate' as ToolbarKey, x: 136.5, y: 261, label: 'Rotate', active: selectedToolbarButton === 'measure-rotate' },
          { key: 'measure-reset' as ToolbarKey, x: 181.5, y: 261, label: 'Reset', active: flashToolbarButton === 'measure-reset' },
          { key: 'output-capture' as ToolbarKey, x: 92, y: 336, label: 'Capture', active: selectedToolbarButton === 'output-capture' },
          { key: 'output-save' as ToolbarKey, x: 139, y: 336, label: 'Capture save', active: selectedToolbarButton === 'output-save' },
          { key: 'task-original' as ToolbarKey, x: 92, y: 416, label: 'Original image', active: viewMode === 'original' },
          { key: 'task-heatmap' as ToolbarKey, x: 139, y: 416, label: 'Heatmap', active: viewMode === 'heatmap' },
        ].map((item, index) => (
          <ToolIcon
            key={item.key}
            icon={displayToolbarIcons[index]}
            activeIcon={activeToolbarIcons[index]}
            left={item.x}
            top={item.y}
            active={item.active}
            onClick={() => handleToolSelection(item.key)}
            label={item.label}
          />
        ))}

        <div style={{ left: wp(251), top: hp(49), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700, zIndex: 2 }}>Panorama</div>
        <img
          src={headerMarkerIcon}
          alt=""
          draggable={false}
          style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(55), position: 'absolute', zIndex: 2 }}
        />
        {workspaceSection === 'studies' && (
          <div
            style={{
              width: wp(studiesPanelWidth),
              height: hp(studiesPanelHeight),
              left: wp(studiesPanelLeft),
              top: hp(studiesPanelTop),
              position: 'absolute',
              zIndex: 18,
              border: `${scalePx(1)} solid #4C4C4C`,
              background: 'rgba(14, 14, 14, 0.92)',
              overflow: 'hidden',
            }}
          >
            <StudiesWorkspacePanel
              studies={combinedStudies as FolderStudy[]}
              selectedSeriesId={selectedFolderSeriesId}
              isVisible
              onSelectSeries={(seriesId) => {
                void handleSelectSeries(seriesId);
              }}
            />
          </div>
        )}
        {isChartVisible && (
          <>
            <div style={{ left: wp(251), top: hp(803), position: 'absolute', color: 'white', fontSize: scalePx(13), fontWeight: 700, zIndex: 2 }}>Dental Chart</div>
            <img
              src={headerMarkerIcon}
              alt=""
              draggable={false}
              style={{ width: scalePx(7), height: scalePx(7), left: wp(240), top: hp(809), position: 'absolute', zIndex: 2 }}
            />
            <button
              type="button"
              onClick={() => {
                setIsChartVisible(false);
              }}
              aria-label="Hide dental chart"
              style={{
                width: wp(18),
                height: hp(14),
                left: wp(chartHeaderHideLeft),
                top: hp(806),
                position: 'absolute',
                border: `${scalePx(1)} solid #2C2C2C`,
                background: '#8D8D8D',
                color: '#111111',
                fontSize: scalePx(10),
                fontWeight: 700,
                lineHeight: hp(12),
                textAlign: 'center',
                cursor: 'pointer',
                zIndex: 3,
                padding: 0,
              }}
            >
              x
            </button>
          </>
        )}

        {isChartBodyVisible && (
          <div style={{ left: wp(fdiLeft), top: hp(845), position: 'absolute', fontWeight: 700 }}>
            <span style={{ color: 'white', fontSize: scalePx(14) }}>FDI</span>
            <span style={{ color: 'white', fontSize: scalePx(13) }}> </span>
            <span style={{ color: '#9C9C9C', fontSize: scalePx(13) }}>/ Univ</span>
          </div>
        )}

        <div style={{ left: wp(251), top: hp(panoLabelTop), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>R</div>
        <div style={{ left: wp(lLabelLeft), top: hp(panoLabelTop), position: 'absolute', color: '#D39C00', fontSize: scalePx(14), fontWeight: 700, zIndex: 14 }}>L</div>

        {isChartBodyVisible && (
          <div
            style={{
              width: wp(chartOdontoLineWidth),
              height: scalePx(1),
              left: wp(chartOdontoLineLeft),
              top: hp(chartOdontoLineY),
              position: 'absolute',
              background: '#B4B4B4',
            }}
          />
        )}
        {isChartBodyVisible && (
          <div
            style={{
              width: scalePx(1),
              height: hp(chartOdontoVerticalHeight),
              left: wp(chartOdontoCenterX),
              top: hp(chartOdontoVerticalTop),
              position: 'absolute',
              background: '#B4B4B4',
            }}
          />
        )}
        {isChartBodyVisible && (
          <>
            <div style={{ width: scalePx(1), height: hp(264), left: wp(chartSectionLeft), top: hp(804), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: scalePx(1), height: hp(chartContentHeight), left: wp(chartLegendDividerX), top: hp(chartContentTop), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: scalePx(1), height: hp(250), left: wp(rightEdge - 1), top: hp(813), position: 'absolute', background: '#5C5C5C' }} />
            <div style={{ width: wp(chartLegendDividerX - chartSectionLeft), height: hp(chartContentHeight), left: wp(chartSectionLeft), top: hp(chartContentTop), position: 'absolute', border: `${scalePx(1)} solid #4C4C4C`, pointerEvents: 'none' }} />
          </>
        )}

        {isChartBodyVisible && upperLeftOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`upper-left-${order}`}
              arch="U"
              order={order}
              left={toothSlotCentersLeft[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
              flipX
            />
          );
        })}
        {isChartBodyVisible && upperRightOrders.map((order, index) => {
          const width = Number(upperSizes[order - 1].split(' ')[0]);
          const height = Number(upperSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`upper-right-${order}`}
              arch="U"
              order={order}
              left={toothSlotCentersRight[index] - width / 2}
              top={upperBaseline - height}
              width={width}
              height={height}
            />
          );
        })}
        {isChartBodyVisible && lowerLeftOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`lower-left-${order}`}
              arch="L"
              order={order}
              left={toothSlotCentersLeft[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
              flipX
            />
          );
        })}
        {isChartBodyVisible && lowerRightOrders.map((order, index) => {
          const width = Number(lowerSizes[order - 1].split(' ')[0]);
          const height = Number(lowerSizes[order - 1].split(' ')[1]);
          return (
            <ToothSlotImage
              key={`lower-right-${order}`}
              arch="L"
              order={order}
              left={toothSlotCentersRight[index] - width / 2}
              top={lowerTop}
              width={width}
              height={height}
            />
          );
        })}

        {isChartBodyVisible && (
          <div style={{ width: wp(chartLegendWidth), height: hp(chartLegendHeight), left: wp(chartLegendLeft + 22), top: hp(chartLegendTop), position: 'absolute' }}>
            {legendItems.map((item) => (
              <div key={item.label}>
                <div
                  style={{
                    width: relativePercent(19, chartLegendWidth),
                    height: relativePercent(19, chartLegendHeight),
                    left: 0,
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    background: '#808181',
                  }}
                />
                <div
                  style={{
                    width: relativePercent(52, chartLegendWidth),
                    height: relativePercent(19, chartLegendHeight),
                    left: relativePercent(chartLegendWidth - 52 - 12, chartLegendWidth),
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    background: item.color,
                  }}
                />
                <div
                  style={{
                    left: relativePercent(30, chartLegendWidth),
                    top: relativePercent(item.top - 834, chartLegendHeight),
                    position: 'absolute',
                    color: 'white',
                    fontSize: scalePx(13),
                    fontWeight: 700,
                  }}
                >
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        )}

          {isChartBodyVisible && (
            <button
              type="button"
              onClick={() => { void handleStartReport(); }}
              aria-pressed={isReportActive || reportDrawerOpen}
              aria-label="Open report workspace"
              style={{
                width: wp(88),
                height: hp(88),
                left: wp(reportLeft),
                top: hp(reportTop),
                position: 'absolute',
                zIndex: 30,
                borderRadius: '50%',
                outline: isReportActive || reportDrawerOpen ? `${scalePx(3)} solid #00C0F3` : 'none',
                outlineOffset: scalePx(2),
                boxShadow: isReportActive || reportDrawerOpen ? `0 0 ${scalePx(18)} rgba(0, 192, 243, 0.45)` : 'none',
                cursor: 'pointer',
              }}
            >
              <img
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block',
                  filter: isReportActive || reportDrawerOpen ? 'drop-shadow(0 0 10px rgba(0, 192, 243, 0.7))' : 'none',
                }}
                src={isReportActive || reportDrawerOpen ? displayReportButtonIcons.active : displayReportButtonIcons.inactive}
                alt=""
              />
            </button>
          )}

          {reportError && (
            <div
              style={{
                left: wp(251),
                top: hp(74),
                position: 'absolute',
                color: '#FFB4B4',
                fontSize: scalePx(11),
                fontWeight: 700,
                zIndex: 20,
              }}
            >
              {reportError}
            </div>
          )}

          {!isChartVisible && (
            <button
              type="button"
              onClick={handleChartToggle}
              aria-pressed={false}
              aria-label="Show dental chart"
              style={{
                width: wp(82),
                height: hp(16),
                left: wp(panoChartToggleLeft),
                top: hp(panoChartToggleTop),
                position: 'absolute',
                border: `${scalePx(1)} solid #4C4C4C`,
                background: '#5C5C5C',
                color: '#FFFFFF',
                fontSize: scalePx(9),
                fontWeight: 700,
                cursor: 'pointer',
                zIndex: 4,
                padding: 0,
              }}
            >
              Dental Chart
            </button>
          )}
        </div>
      </div>
      {reportSessionId && typeof document !== 'undefined' && createPortal(
        <WebReportDrawer
          sessionId={reportSessionId}
          open={reportDrawerOpen}
          onClose={() => {
            setReportDrawerOpen(false);
            setIsReportActive(false);
            setWorkspaceSection('none');
          }}
        />,
        document.body
      )}
    </div>
  );
}
