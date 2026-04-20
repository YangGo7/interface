import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';

const PREVIEW_BOX_WIDTH = 420;
const PREVIEW_BOX_HEIGHT = 460;
const PREVIEW_IMAGE_MAX_HEIGHT = 420;
const PREVIEW_BOX_GAP = 12;
const PREVIEW_BOX_MARGIN = 16;

type OutputCaptureItem = {
  id: string;
  dataUrl: string;
  createdAt: number;
};

type OutputCapturePanelProps = {
  visible?: boolean;
  collapsed?: boolean;
  left: string;
  top: string;
  width: string;
  height: string;
  captures: OutputCaptureItem[];
  selectedCaptureIds?: string[];
  onToggle: () => void;
  onSelectCapture?: (id: string) => void;
  onRemove: (id: string) => void;
  onClear: () => void;
};

export function OutputCapturePanel({
  visible = true,
  collapsed = false,
  left,
  top,
  width,
  height,
  captures,
  selectedCaptureIds = [],
  onToggle,
  onSelectCapture,
  onRemove,
  onClear,
}: OutputCapturePanelProps) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  const previewHeaderRef = useRef<HTMLDivElement | null>(null);
  const previewWindowRef = useRef<HTMLDivElement | null>(null);
  const previewDragRef = useRef<{ pointerId: number; pointerOffsetX: number; pointerOffsetY: number } | null>(null);
  const previewWasDraggedRef = useRef(false);
  const [previewCaptureId, setPreviewCaptureId] = useState<string | null>(null);
  const [previewPosition, setPreviewPosition] = useState<{ top: number; left: number } | null>(null);
  const [isDraggingPreview, setIsDraggingPreview] = useState(false);

  const previewCapture = useMemo(
    () => captures.find((capture) => capture.id === previewCaptureId) || null,
    [captures, previewCaptureId]
  );

  const getLayoutMetrics = () => {
    const panelElement = panelRef.current;
    const offsetParent = panelElement?.offsetParent as HTMLElement | null;
    if (!panelElement || !offsetParent) return null;

    const panelRect = panelElement.getBoundingClientRect();
    const parentRect = offsetParent.getBoundingClientRect();
    const renderedScale = panelElement.offsetWidth > 0 ? panelRect.width / panelElement.offsetWidth : 1;

    return {
      scale: renderedScale || 1,
      parentRect,
      parentWidth: offsetParent.clientWidth,
      parentHeight: offsetParent.clientHeight,
    };
  };

  const clampPreviewPosition = (left: number, top: number) => {
    const metrics = getLayoutMetrics();
    if (!metrics) return { left, top };

    return {
      left: Math.min(
        Math.max(PREVIEW_BOX_MARGIN, left),
        Math.max(PREVIEW_BOX_MARGIN, metrics.parentWidth - PREVIEW_BOX_WIDTH - PREVIEW_BOX_MARGIN)
      ),
      top: Math.min(
        Math.max(PREVIEW_BOX_MARGIN, top),
        Math.max(PREVIEW_BOX_MARGIN, metrics.parentHeight - PREVIEW_BOX_HEIGHT - PREVIEW_BOX_MARGIN)
      ),
    };
  };

  useEffect(() => {
    if (!previewCaptureId) return;
    if (!captures.some((capture) => capture.id === previewCaptureId)) {
      setPreviewCaptureId(null);
    }
  }, [captures, previewCaptureId]);

  useEffect(() => {
    if (!visible || collapsed) {
      setPreviewCaptureId(null);
    }
  }, [collapsed, visible]);

  useEffect(() => {
    if (!previewCapture || !panelRef.current) return;

    const updatePreviewPosition = () => {
      if (previewWasDraggedRef.current) return;
      const panelElement = panelRef.current;
      const preferredLeft = panelElement.offsetLeft + panelElement.offsetWidth + PREVIEW_BOX_GAP;
      const nextPosition = clampPreviewPosition(preferredLeft, panelElement.offsetTop + 36);
      setPreviewPosition({ top: nextPosition.top, left: nextPosition.left });
    };

    updatePreviewPosition();
    window.addEventListener('resize', updatePreviewPosition);
    return () => {
      window.removeEventListener('resize', updatePreviewPosition);
    };
  }, [previewCapture]);

  useEffect(() => {
    if (!isDraggingPreview) return;

    const handlePointerMove = (event: PointerEvent) => {
      const dragState = previewDragRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) return;
      const metrics = getLayoutMetrics();
      if (!metrics) return;
      const nextPosition = clampPreviewPosition(
        (event.clientX - metrics.parentRect.left - dragState.pointerOffsetX) / metrics.scale,
        (event.clientY - metrics.parentRect.top - dragState.pointerOffsetY) / metrics.scale
      );
      setPreviewPosition(nextPosition);
    };

    const handlePointerUp = (event: PointerEvent) => {
      const dragState = previewDragRef.current;
      if (!dragState || dragState.pointerId !== event.pointerId) return;
      if (previewHeaderRef.current?.hasPointerCapture(dragState.pointerId)) {
        previewHeaderRef.current.releasePointerCapture(dragState.pointerId);
      }
      previewDragRef.current = null;
      setIsDraggingPreview(false);
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    window.addEventListener('pointercancel', handlePointerUp);
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      window.removeEventListener('pointercancel', handlePointerUp);
    };
  }, [isDraggingPreview]);

  const handleCaptureClick = (captureId: string) => {
    if (previewCaptureId === captureId) {
      previewWasDraggedRef.current = false;
      setPreviewCaptureId(null);
      return;
    }
    onSelectCapture?.(captureId);
    previewWasDraggedRef.current = false;
    setPreviewCaptureId(captureId);
  };

  const handlePreviewDragStart = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!previewPosition || !previewWindowRef.current) return;
    const target = event.target as HTMLElement | null;
    if (target?.closest('button')) return;

    if (!getLayoutMetrics()) return;
    const previewRect = previewWindowRef.current.getBoundingClientRect();

    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    previewDragRef.current = {
      pointerId: event.pointerId,
      pointerOffsetX: event.clientX - previewRect.left,
      pointerOffsetY: event.clientY - previewRect.top,
    };
    previewWasDraggedRef.current = true;
    setIsDraggingPreview(true);
  };

  if (!visible) return null;

  return (
    <>
      <div
        ref={panelRef}
        style={{
          position: 'absolute',
          left,
          top,
          width,
          height: collapsed ? '28px' : height,
          background: collapsed ? 'transparent' : '#333333',
          border: collapsed ? 'none' : '1px solid #4C4C4C',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        <button
          type="button"
          onClick={onToggle}
          style={{
            height: '28px',
            width: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 8px',
            color: '#EAEAEA',
            fontSize: '11px',
            fontWeight: 700,
            letterSpacing: '0.03em',
            border: '1px solid #4C4C4C',
            borderBottom: collapsed ? '1px solid #4C4C4C' : '1px solid #4C4C4C',
            background: '#3A3A3A',
            flexShrink: 0,
            cursor: 'pointer',
          }}
        >
          <span>Capture Box</span>
          <span style={{ color: '#CFCFCF', fontSize: '10px' }}>{collapsed ? '+' : '-'}</span>
        </button>
        {!collapsed && (
          <>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'flex-end',
                padding: '6px 8px 0 8px',
              }}
            >
              <button
                type="button"
                onClick={onClear}
                disabled={captures.length === 0}
                style={{
                  border: 'none',
                  background: 'transparent',
                  color: captures.length === 0 ? '#7F7F7F' : '#CFCFCF',
                  fontSize: '10px',
                  fontWeight: 700,
                  cursor: captures.length === 0 ? 'default' : 'pointer',
                  padding: 0,
                }}
              >
                Clear
              </button>
            </div>
            <div
              style={{
                flex: 1,
                overflowY: 'auto',
                padding: '8px',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
              }}
            >
              {captures.length === 0 ? (
                <div
                  style={{
                    flex: 1,
                    minHeight: '120px',
                    border: '1px dashed #595959',
                    background: '#2D2D2D',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#8E8E8E',
                    fontSize: '11px',
                    textAlign: 'center',
                    lineHeight: 1.4,
                    padding: '10px',
                  }}
                >
                  Captured images will be stored here.
                </div>
              ) : (
                captures.map((capture, index) => (
                  <button
                    key={capture.id}
                    type="button"
                    onClick={() => handleCaptureClick(capture.id)}
                    style={{
                      border: selectedCaptureIds.includes(capture.id) ? '1px solid #00C0F3' : '1px solid #555555',
                      background: previewCaptureId === capture.id ? '#243744' : '#2B2B2B',
                      padding: '6px',
                      textAlign: 'left',
                      boxShadow: selectedCaptureIds.includes(capture.id) ? 'inset 0 0 0 1px rgba(0, 192, 243, 0.35)' : 'none',
                      cursor: 'pointer',
                    }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        marginBottom: '6px',
                        color: '#BDBDBD',
                        fontSize: '10px',
                        fontWeight: 700,
                      }}
                    >
                      <span>Shot {captures.length - index}</span>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          if (previewCaptureId === capture.id) {
                            setPreviewCaptureId(null);
                          }
                          onRemove(capture.id);
                        }}
                        style={{
                          border: 'none',
                          background: 'transparent',
                          color: '#CFCFCF',
                          fontSize: '10px',
                          fontWeight: 700,
                          cursor: 'pointer',
                          padding: 0,
                        }}
                      >
                        X
                      </button>
                    </div>
                    <img
                      src={capture.dataUrl}
                      alt=""
                      draggable={false}
                      style={{
                        display: 'block',
                        width: '100%',
                        height: '76px',
                        objectFit: 'cover',
                        background: '#000000',
                        border: '1px solid #4C4C4C',
                      }}
                    />
                  </button>
                ))
              )}
            </div>
          </>
        )}
      </div>
      {previewCapture && previewPosition ? (
        <div
          ref={previewWindowRef}
          style={{
            position: 'absolute',
            top: `${previewPosition.top}px`,
            left: `${previewPosition.left}px`,
            width: `${PREVIEW_BOX_WIDTH}px`,
            background: '#1E1E1E',
            border: '1px solid #4C4C4C',
            boxShadow: '0 18px 40px rgba(0, 0, 0, 0.45)',
            zIndex: 80,
          }}
        >
          <div
            ref={previewHeaderRef}
            onPointerDown={handlePreviewDragStart}
            style={{
              height: '34px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 10px',
              borderBottom: '1px solid #4C4C4C',
              color: '#EAEAEA',
              fontSize: '11px',
              fontWeight: 700,
              background: '#2C2C2C',
              cursor: isDraggingPreview ? 'grabbing' : 'grab',
              userSelect: 'none',
              touchAction: 'none',
            }}
          >
            <span>Capture Preview</span>
            <button
              type="button"
              onClick={() => {
                previewWasDraggedRef.current = false;
                setPreviewCaptureId(null);
              }}
              style={{
                border: 'none',
                background: 'transparent',
                color: '#FFFFFF',
                fontSize: '14px',
                fontWeight: 700,
                cursor: 'pointer',
                padding: 0,
                lineHeight: 1,
              }}
            >
              X
            </button>
          </div>
          <button
            type="button"
            onClick={() => setPreviewCaptureId(null)}
            style={{
              width: '100%',
              border: 'none',
              background: '#101010',
              padding: '10px',
              cursor: 'pointer',
            }}
          >
            <img
              src={previewCapture.dataUrl}
              alt=""
              draggable={false}
              style={{
                display: 'block',
                width: '100%',
                maxHeight: `${PREVIEW_IMAGE_MAX_HEIGHT}px`,
                objectFit: 'contain',
                background: '#000000',
                border: '1px solid #3F3F3F',
              }}
            />
          </button>
        </div>
      ) : null}
    </>
  );
}
