import type { DicomOverlayMetadata } from './cornerstone/dicomMetadata';

type DicomMetadataOverlayProps = {
  metadata: DicomOverlayMetadata | null;
  top?: number;
  bottom?: number;
  left?: number;
  right?: number;
  compact?: boolean;
  leftPanelAlign?: 'left' | 'right';
  rightPanelAlign?: 'left' | 'right';
  leftPanelWidth?: number;
  rightPanelWidth?: number;
  leftPanelFontSize?: number;
  rightPanelFontSize?: number;
  headerFontSize?: number;
  panelPaddingX?: number;
  panelPaddingY?: number;
  panelRadius?: number;
};

const formatDicomDate = (raw?: string) => {
  if (!raw) return '';
  if (raw.length !== 8) return raw;
  return `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}`;
};

const formatDicomTime = (raw?: string) => {
  if (!raw) return '';
  const digits = raw.replace(/[^\d]/g, '');
  if (digits.length < 2) return raw;
  const hh = digits.slice(0, 2);
  const mm = digits.slice(2, 4) || '00';
  const ss = digits.slice(4, 6) || '00';
  return `${hh}:${mm}:${ss}`;
};

const formatNumber = (value?: number, fractionDigits = 2) =>
  typeof value === 'number' && Number.isFinite(value) ? value.toFixed(fractionDigits) : '';

const buildLeftLines = (metadata: DicomOverlayMetadata, compact: boolean) => {
  const dateTime = [formatDicomDate(metadata.studyDate), formatDicomTime(metadata.studyTime)].filter(Boolean).join(' ');
  const patientLine = [
    metadata.patientName || 'Anonymous',
    metadata.patientId && `ID ${metadata.patientId}`,
  ].filter(Boolean).join('  ');
  const demoLine = [
    metadata.patientSex,
    formatDicomDate(metadata.patientBirthDate),
    metadata.modality,
    dateTime,
  ].filter(Boolean).join('  ');
  const studyLine = [
    metadata.studyDescription,
    metadata.seriesDescription,
    metadata.institutionName,
  ].filter(Boolean).join('  |  ');
  const identityLine = [
    metadata.patientSex,
    formatDicomDate(metadata.patientBirthDate),
    metadata.patientId && `ID ${metadata.patientId}`,
  ].filter(Boolean).join('  ');

  return compact
    ? [patientLine, demoLine, studyLine].filter(Boolean)
    : [patientLine, identityLine, demoLine, studyLine].filter(Boolean);
};

const buildRightLines = (metadata: DicomOverlayMetadata, compact: boolean) => {
  const pixelLine = [
    `${metadata.rows}x${metadata.columns}`,
    `${metadata.bitsStored || metadata.bitsAllocated}/${metadata.bitsAllocated} bit`,
    `${metadata.samplesPerPixel}ch`,
    metadata.numberOfFrames > 1 ? `frames ${metadata.numberOfFrames}` : 'frame 1/1',
  ].filter(Boolean).join('  ');
  const spacingLine = [
    `spacing ${formatNumber(metadata.rowPixelSpacing)} x ${formatNumber(metadata.columnPixelSpacing)} mm`,
    `thk ${formatNumber(metadata.sliceThickness)} mm`,
  ].filter(Boolean).join('  ');
  const displayLine = [
    `WL ${Math.round(metadata.windowCenter || 0)}`,
    `WW ${Math.round(metadata.windowWidth || 0)}`,
    metadata.photometricInterpretation,
  ].filter(Boolean).join('  ');
  const positionLine = metadata.imagePositionPatient?.length >= 3
    ? `IPP ${metadata.imagePositionPatient.slice(0, 3).map((value) => formatNumber(value, 2)).join(', ')}`
    : '';

  return compact
    ? [pixelLine, spacingLine, displayLine].filter(Boolean)
    : [pixelLine, spacingLine, displayLine, positionLine].filter(Boolean);
};

export function DicomMetadataOverlay({
  metadata,
  top = 12,
  bottom = 12,
  left = 16,
  right = 16,
  compact = false,
  leftPanelAlign = 'left',
  rightPanelAlign = 'right',
  leftPanelWidth = 260,
  rightPanelWidth = 320,
  leftPanelFontSize = 11,
  rightPanelFontSize = 10,
  headerFontSize = 10,
  panelPaddingX = 12,
  panelPaddingY = 8,
  panelRadius = 12,
}: DicomMetadataOverlayProps) {
  if (!metadata) return null;

  const leftLines = buildLeftLines(metadata, compact);
  const rightLines = buildRightLines(metadata, compact);
  if (leftLines.length === 0 && rightLines.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        zIndex: 80,
        pointerEvents: 'none',
      }}
    >
      <div
        style={leftPanelAlign === 'right' ? { position: 'absolute', top, right } : { position: 'absolute', top, left }}
      >
        <div
          style={{
            width: `${leftPanelWidth}px`,
            borderRadius: `${panelRadius}px`,
            padding: `${panelPaddingY}px ${panelPaddingX}px`,
            fontSize: `${leftPanelFontSize}px`,
            lineHeight: 1.35,
            color: '#FFFFFF',
            background: 'rgba(0, 0, 0, 0.58)',
            border: '1px solid rgba(255,255,255,0.14)',
            boxShadow: '0 12px 30px rgba(0,0,0,0.32)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
          }}
        >
          <div style={{ marginBottom: '4px', fontSize: `${headerFontSize}px`, fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#FFFFFF' }}>
            DICOM HUD
          </div>
          <div style={{ fontFamily: 'monospace' }}>
            {leftLines.map((line, index) => (
              <div key={`left-${line}-${index}`} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div
        style={rightPanelAlign === 'left' ? { position: 'absolute', left, bottom } : { position: 'absolute', right, bottom }}
      >
        <div
          style={{
            width: `${rightPanelWidth}px`,
            borderRadius: `${panelRadius}px`,
            padding: `${panelPaddingY}px ${panelPaddingX}px`,
            fontSize: `${rightPanelFontSize}px`,
            lineHeight: 1.35,
            color: '#FFFFFF',
            background: 'rgba(0, 0, 0, 0.58)',
            border: '1px solid rgba(255,255,255,0.12)',
            boxShadow: '0 10px 26px rgba(0,0,0,0.28)',
            backdropFilter: 'blur(10px)',
            WebkitBackdropFilter: 'blur(10px)',
          }}
        >
          <div style={{ marginBottom: '4px', fontSize: `${headerFontSize}px`, fontWeight: 600, letterSpacing: '0.18em', textTransform: 'uppercase', color: '#FFFFFF' }}>
            {metadata.modality || 'DICOM'}
          </div>
          <div style={{ fontFamily: 'monospace' }}>
            {rightLines.map((line, index) => (
              <div key={`right-${line}-${index}`} style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                {line}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
