import { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { resolveServerAssetUrl } from '../lib/folderLeaderApi';
import {
  DentalMprWorkspace,
  type DentalMprSeriesInfo,
  type VtkRenderParameters,
} from '../features/mpr/DentalMprWorkspace';
import type { ViewerSource } from '../viewer/CornerstoneViewer';
import type { Mpr2DTool, Mpr3DControlState, MprViewportControlState } from '../features/mpr/mpr3dControls';

export type MprRouteState = {
  studyId?: string;
  studyInstanceUid?: string;
  seriesInstanceUid?: string | null;
  patientId?: string;
  studyDescription?: string;
};

type DentalMprSeries = {
  seriesInstanceUID: string;
  patientID?: string;
  patientId?: string;
  patientName?: string;
  patientSex?: string;
  patientAge?: string;
  studyDate?: string;
  seriesDescription?: string;
  seriesNumber?: number | null;
  instanceCount?: number;
  gantryTilt?: number | null;
  tubeCurrent?: number | null;
  tubeVoltage?: number | null;
};

type DentalMprInstance = {
  sopInstanceUID: string;
  instanceNumber?: number;
};

type DentalMprRenderParamsResponse = {
  success?: boolean;
  mn?: number;
  mx?: number;
  t0?: number;
  t1?: number;
  t2?: number;
  thresholds?: number[];
};

type DentalMprPageProps = {
  embedded?: boolean;
  routeStateOverride?: MprRouteState | null;
  mpr3DControlState?: Mpr3DControlState;
  mpr2DTool?: Mpr2DTool;
  showHuOverlay?: boolean;
  mprViewportControlState?: MprViewportControlState;
  onMprViewportControlStateChange?: (state: MprViewportControlState) => void;
};

const sortInstances = (items: DentalMprInstance[]) =>
  [...items].sort((a, b) => (Number(a.instanceNumber) || 0) - (Number(b.instanceNumber) || 0));

const isVtkBonePreset = (preset?: Mpr3DControlState['preset']) => preset === 'vtk-bone1' || preset === 'vtk-bone2';

function parseVtkRenderParameters(data: DentalMprRenderParamsResponse): VtkRenderParameters | null {
  const thresholds = Array.isArray(data.thresholds) ? data.thresholds : [];
  const params = {
    mn: Number(data.mn),
    mx: Number(data.mx),
    t0: Number(data.t0 ?? thresholds[0]),
    t1: Number(data.t1 ?? thresholds[1]),
    t2: Number(data.t2 ?? thresholds[2]),
  };
  return Object.values(params).every((value) => Number.isFinite(value)) ? params : null;
}

export function DentalMprPage({
  embedded = false,
  routeStateOverride = null,
  mpr3DControlState,
  mpr2DTool,
  showHuOverlay,
  mprViewportControlState,
  onMprViewportControlStateChange,
}: DentalMprPageProps = {}) {
  const location = useLocation();
  const hostRef = useRef<HTMLDivElement | null>(null);
  const routeState = (routeStateOverride || location.state || {}) as MprRouteState;
  const queryParams = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const queryStudyUid =
    queryParams.get('study_uid') ||
    queryParams.get('studyInstanceUid') ||
    queryParams.get('study_instance_uid') ||
    queryParams.get('study_id') ||
    '';
  const queryPatientId = queryParams.get('patient_id') || queryParams.get('patientId') || '';
  const querySeriesUid =
    queryParams.get('series_uid') ||
    queryParams.get('seriesInstanceUid') ||
    queryParams.get('series_instance_uid') ||
    '';
  const studyInstanceUid = routeState.studyInstanceUid || routeState.studyId || queryStudyUid.trim();
  const seriesInstanceUid = routeState.seriesInstanceUid || querySeriesUid.trim();
  const patientId = routeState.patientId || queryPatientId.trim();

  const [height, setHeight] = useState(embedded ? 720 : window.innerHeight);
  const [source, setSource] = useState<ViewerSource | null>(null);
  const [seriesInfo, setSeriesInfo] = useState<DentalMprSeriesInfo | null>(null);
  const [vtkRenderParameters, setVtkRenderParameters] = useState<VtkRenderParameters | null>(null);
  const [vtkRenderParametersSeriesId, setVtkRenderParametersSeriesId] = useState<string | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const el = hostRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      setHeight(Math.max(320, Math.floor(rect.height || window.innerHeight)));
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadMprSource() {
      setReady(false);
      setError(null);
      setSource(null);
      setSeriesInfo(null);
      setVtkRenderParameters(null);
      setVtkRenderParametersSeriesId(null);

      try {
        if (studyInstanceUid) {
          const focusResponse = await fetch(resolveServerAssetUrl('/api/mpr/focus-study'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              studyInstanceUid,
              seriesInstanceUid,
              patientId,
            }),
          });
          const focusData = await focusResponse.json().catch(() => ({}));
          if (!focusResponse.ok || focusData.success === false) {
            throw new Error(focusData.detail || focusData.message || 'Failed to prepare Dental MPR case.');
          }
        }

        const seriesResponse = await fetch(resolveServerAssetUrl('/api/series'));
        const seriesData = await seriesResponse.json().catch(() => ({}));
        if (!seriesResponse.ok || seriesData.success === false) {
          throw new Error(seriesData.detail || seriesData.message || 'Failed to load Dental MPR series.');
        }

        const seriesItems = Array.isArray(seriesData.series) ? (seriesData.series as DentalMprSeries[]) : [];
        const selectedSeries =
          seriesItems.find((item) => item.seriesInstanceUID === seriesInstanceUid) ||
          seriesItems.reduce<DentalMprSeries | null>(
            (best, item) => (!best || (item.instanceCount || 0) > (best.instanceCount || 0) ? item : best),
            null
          );

        if (!selectedSeries?.seriesInstanceUID) {
          throw new Error('No DICOM series is available for Dental MPR.');
        }

        const instancesResponse = await fetch(
          resolveServerAssetUrl(`/api/series/${encodeURIComponent(selectedSeries.seriesInstanceUID)}/instances`)
        );
        const instancesData = await instancesResponse.json().catch(() => ({}));
        if (!instancesResponse.ok || instancesData.success === false) {
          throw new Error(instancesData.detail || instancesData.message || 'Failed to load Dental MPR instances.');
        }

        const instances = sortInstances(
          Array.isArray(instancesData.instances) ? (instancesData.instances as DentalMprInstance[]) : []
        );
        const imageIds = instances
          .map((item) => item.sopInstanceUID)
          .filter(Boolean)
          .map((sopInstanceUID) => {
            const url = resolveServerAssetUrl(
              `/api/dicom/${encodeURIComponent(selectedSeries.seriesInstanceUID)}/${encodeURIComponent(sopInstanceUID)}`
            );
            return `wadouri:${url}`;
          });

        if (!imageIds.length) {
          throw new Error('No DICOM instances are available for Dental MPR.');
        }

        if (cancelled) return;
        setSeriesInfo({
          seriesInstanceUID: selectedSeries.seriesInstanceUID,
          patientID: selectedSeries.patientID || selectedSeries.patientId || patientId,
          patientName: selectedSeries.patientName,
          patientSex: selectedSeries.patientSex,
          patientAge: selectedSeries.patientAge,
          studyDate: selectedSeries.studyDate,
          seriesDescription: selectedSeries.seriesDescription,
          seriesNumber: selectedSeries.seriesNumber,
          gantryTilt: selectedSeries.gantryTilt,
          tubeCurrent: selectedSeries.tubeCurrent,
          tubeVoltage: selectedSeries.tubeVoltage,
        });
        setSource({
          id: selectedSeries.seriesInstanceUID,
          label:
            selectedSeries.seriesDescription ||
            selectedSeries.patientName ||
            `Series ${selectedSeries.seriesNumber ?? selectedSeries.seriesInstanceUID.slice(0, 12)}`,
          url: imageIds[0].replace(/^wadouri:/, ''),
          imageIds,
          scheme: 'wadouri',
        });
        setReady(true);
      } catch (nextError: any) {
        if (cancelled) return;
        setError(nextError?.message || 'Failed to prepare Dental MPR case.');
        setReady(true);
      }
    }

    void loadMprSource();

    return () => {
      cancelled = true;
    };
  }, [patientId, seriesInstanceUid, studyInstanceUid]);

  useEffect(() => {
    if (!source?.id || !isVtkBonePreset(mpr3DControlState?.preset)) return;
    if (vtkRenderParameters && vtkRenderParametersSeriesId === source.id) return;

    let cancelled = false;
    async function loadVtkRenderParameters() {
      try {
        const response = await fetch(resolveServerAssetUrl(`/api/series/${encodeURIComponent(source.id)}/render-params`));
        const data = (await response.json().catch(() => ({}))) as DentalMprRenderParamsResponse;
        if (!response.ok || data.success === false) return;
        const params = parseVtkRenderParameters(data);
        if (!cancelled && params) {
          setVtkRenderParameters(params);
          setVtkRenderParametersSeriesId(source.id);
        }
      } catch {
        // VTK presets can still use their local fallback if backend parameter calculation fails.
      }
    }

    void loadVtkRenderParameters();
    return () => {
      cancelled = true;
    };
  }, [mpr3DControlState?.preset, source?.id, vtkRenderParameters, vtkRenderParametersSeriesId]);

  return (
    <div
      ref={hostRef}
      style={{
        width: embedded ? '100%' : '100vw',
        height: embedded ? '100%' : '100vh',
        background: '#111111',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {ready && source && seriesInfo ? (
        <DentalMprWorkspace
          source={source}
          seriesInfo={seriesInfo}
          height={height}
          viewportControlState={mprViewportControlState}
          active2DTool={mpr2DTool}
          showHuOverlay={showHuOverlay}
          control3DState={mpr3DControlState}
          vtkRenderParameters={vtkRenderParameters}
          onViewportControlStateChange={onMprViewportControlStateChange}
        />
      ) : null}
      {!ready || error ? (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#f5f5f5',
            background: '#111111',
            fontSize: 14,
            zIndex: 20,
          }}
        >
          {error || 'Preparing Dental MPR...'}
        </div>
      ) : null}
    </div>
  );
}
