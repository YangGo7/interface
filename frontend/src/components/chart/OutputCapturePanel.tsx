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
  onToggle: () => void;
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
  onToggle,
  onRemove,
  onClear,
}: OutputCapturePanelProps) {
  if (!visible) return null;

  return (
    <div
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
                <div
                  key={capture.id}
                  style={{
                    border: '1px solid #555555',
                    background: '#2B2B2B',
                    padding: '6px',
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
                      onClick={() => onRemove(capture.id)}
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
                </div>
              ))
            )}
          </div>
        </>
      )}
    </div>
  );
}
