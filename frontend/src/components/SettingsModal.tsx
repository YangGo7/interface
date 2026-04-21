import type { CSSProperties } from 'react';
import type { AppNumberingSystem } from '../lib/appSettings';

type SettingsModalProps = {
  visible: boolean;
  rootFolderPath: string;
  onRootFolderPathChange: (value: string) => void;
  onBrowseRootFolder: () => void;
  numberingSystem: AppNumberingSystem;
  onNumberingSystemChange: (value: AppNumberingSystem) => void;
  onClose: () => void;
  onSave: () => void;
  saving?: boolean;
  loading?: boolean;
  error?: string | null;
};

const overlayStyle: CSSProperties = {
  position: 'absolute',
  inset: 0,
  background: 'rgba(0, 0, 0, 0.38)',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  padding: '16px',
  boxSizing: 'border-box',
  zIndex: 40,
};

const shellStyle: CSSProperties = {
  width: '688px',
  minHeight: '497px',
  maxWidth: 'calc(100vw - 32px)',
  maxHeight: 'calc(100vh - 32px)',
  background: '#42494F',
  boxShadow: '0 24px 60px rgba(0, 0, 0, 0.38)',
  position: 'relative',
  boxSizing: 'border-box',
  padding: '20px 7px 0',
  overflowY: 'auto',
  overflowX: 'hidden',
};

const panelStyle: CSSProperties = {
  width: '100%',
  minHeight: '404px',
  background: '#232323',
  padding: '28px 34px',
  boxSizing: 'border-box',
};

export function SettingsModal({
  visible,
  rootFolderPath,
  onRootFolderPathChange,
  onBrowseRootFolder,
  numberingSystem,
  onNumberingSystemChange,
  onClose,
  onSave,
  saving = false,
  loading = false,
  error,
}: SettingsModalProps) {
  if (!visible) return null;

  return (
    <div style={overlayStyle}>
      <div style={shellStyle}>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close settings"
          style={{
            position: 'absolute',
            right: '7px',
            top: '4px',
            width: '13px',
            height: '13px',
            border: '1px solid #AFCBFF',
            background: '#2C5F9A',
            color: '#FFFFFF',
            fontSize: '10px',
            lineHeight: '10px',
            padding: 0,
            cursor: 'pointer',
          }}
        >
          ×
        </button>

        <div style={panelStyle}>
          <div style={{ color: '#F5F5F5', fontSize: '24px', fontWeight: 700, marginBottom: '28px', letterSpacing: '0.03em' }}>
            Settings
          </div>

          <div style={{ display: 'grid', gap: '24px' }}>
            <div>
              <div style={{ color: '#D6D6D6', fontSize: '13px', fontWeight: 700, marginBottom: '10px', letterSpacing: '0.08em' }}>
                ROOT FOLDER
              </div>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <input
                  value={rootFolderPath}
                  onChange={(event) => onRootFolderPathChange(event.target.value)}
                  placeholder="C:/interface/case"
                  disabled={saving || loading}
                  style={{
                    flex: '1 1 420px',
                    height: '42px',
                    background: '#161616',
                    border: '1px solid #5A5A5A',
                    color: '#FFFFFF',
                    padding: '0 14px',
                    fontSize: '14px',
                    outline: 'none',
                    boxSizing: 'border-box',
                  }}
                />
                <button
                  type="button"
                  onClick={onBrowseRootFolder}
                  disabled={saving || loading}
                  style={{
                    minWidth: '96px',
                    height: '42px',
                    border: '1px solid #6B6B6B',
                    background: '#3A3A3A',
                    color: '#F1F1F1',
                    fontSize: '12px',
                    fontWeight: 700,
                    cursor: saving || loading ? 'default' : 'pointer',
                  }}
                >
                  Browse...
                </button>
              </div>
              <div style={{ color: '#9C9C9C', fontSize: '12px', marginTop: '8px' }}>
                Change the server scan path for DICOM and image folders.
              </div>
            </div>

            <div>
              <div style={{ color: '#D6D6D6', fontSize: '13px', fontWeight: 700, marginBottom: '10px', letterSpacing: '0.08em' }}>
                TOOTH NUMBERING
              </div>
              <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                {([
                  { key: 'fdi', label: 'FDI' },
                  { key: 'univ', label: 'UNV' },
                ] as const).map((option) => {
                  const active = numberingSystem === option.key;
                  return (
                    <button
                      key={option.key}
                      type="button"
                      onClick={() => onNumberingSystemChange(option.key)}
                      disabled={saving || loading}
                      style={{
                        minWidth: '120px',
                        height: '38px',
                        border: `1px solid ${active ? '#E2E2E2' : '#636363'}`,
                        background: active ? '#D9D9D9' : '#2D2D2D',
                        color: active ? '#111111' : '#E8E8E8',
                        fontSize: '13px',
                        fontWeight: 700,
                        cursor: 'pointer',
                      }}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
              <div style={{ color: '#9C9C9C', fontSize: '12px', marginTop: '8px' }}>
                Applies to tooth labels in the current viewer and future settings.
              </div>
            </div>

            <div
              style={{
                minHeight: '18px',
                color: error ? '#FF9A9A' : '#8E8E8E',
                fontSize: '12px',
              }}
            >
              {error || (loading ? 'Loading current settings...' : 'Additional settings will be added here later.')}
            </div>
          </div>
        </div>

        <div
          style={{
            position: 'sticky',
            bottom: '0',
            width: '100%',
            minHeight: '73px',
            padding: '0 20px',
            boxSizing: 'border-box',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'flex-end',
            gap: '10px',
            background: '#42494F',
          }}
        >
          <button
            type="button"
            onClick={onClose}
            disabled={saving || loading}
            style={{
              width: '100px',
              height: '30px',
              border: '1px solid #6D6D6D',
              background: '#4A4A4A',
              color: '#F2F2F2',
              fontSize: '12px',
              cursor: saving || loading ? 'default' : 'pointer',
            }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSave}
            disabled={saving || loading}
            style={{
              width: '100px',
              height: '30px',
              border: 'none',
              background: '#D9D9D9',
              color: '#111111',
              fontSize: '12px',
              cursor: saving || loading ? 'default' : 'pointer',
              position: 'relative',
            }}
          >
            Save
            <span
              aria-hidden="true"
              style={{
                position: 'absolute',
                right: '3px',
                bottom: '3px',
                width: 0,
                height: 0,
                borderLeft: '6px solid transparent',
                borderTop: '6px solid #111111',
              }}
            />
          </button>
        </div>
      </div>
    </div>
  );
}
