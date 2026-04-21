import { useMemo, useState } from 'react';
import { Search } from 'lucide-react';
import type { FolderStudy } from '../../features/upload/dicomFolderStudies';

type StudiesWorkspacePanelProps = {
  studies: FolderStudy[];
  selectedSeriesId: string | null;
  onSelectSeries: (seriesId: string) => void;
  isVisible?: boolean;
};

type FlatStudyRow = {
  index: number;
  id: string;
  selected: boolean;
  primarySeriesId: string | null;
  displayName: string;
  searchText: string;
};

const PANEL_BACKGROUND = '#353535';
const HEADER_BACKGROUND = '#474747';
const DIVIDER_COLOR = '#bdbdbd';
const TEXT_COLOR = '#f2f2f2';
const MUTED_TEXT_COLOR = '#cfcfcf';
const NO_COLUMN_WIDTH = 86;

export function StudiesWorkspacePanel({
  studies,
  selectedSeriesId,
  onSelectSeries,
  isVisible = true,
}: StudiesWorkspacePanelProps) {
  const [searchTerm, setSearchTerm] = useState('');

  const normalizedStudies = useMemo(
    () =>
      studies.map((study, index) => {
        const id = String(study.id || study.label || study.description || `study-${index + 1}`);
        const displayName = String(study.label || study.description || study.id || `study-${index + 1}`);
        const selected = (study.series || []).some((series) => series.id === selectedSeriesId);
        return {
          index,
          id,
          selected,
          primarySeriesId: study.series?.[0]?.id || null,
          displayName,
          searchText: [
            displayName,
            study.description,
            study.patientId,
            ...(study.modalities || []),
          ]
            .filter(Boolean)
            .join(' ')
            .toLowerCase(),
        } satisfies FlatStudyRow;
      }),
    [selectedSeriesId, studies]
  );

  const filteredRows = useMemo(() => {
    const query = searchTerm.trim().toLowerCase();
    if (!query) return normalizedStudies;
    return normalizedStudies.filter((row) => row.searchText.includes(query));
  }, [normalizedStudies, searchTerm]);

  if (!isVisible) return null;

  return (
    <div
      style={{
        position: 'relative',
        display: 'flex',
        height: '100%',
        flexDirection: 'column',
        overflow: 'hidden',
        background: PANEL_BACKGROUND,
        color: TEXT_COLOR,
        fontFamily: '"Segoe UI", "Noto Sans", "Noto Sans KR", sans-serif',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
          padding: '4px 8px 8px 8px',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 4,
            fontSize: 14,
            fontWeight: 700,
            lineHeight: 1,
            color: TEXT_COLOR,
            letterSpacing: '-0.02em',
          }}
        >
          <span
            aria-hidden="true"
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: 2,
              width: 8,
              height: 8,
              flexShrink: 0,
            }}
          >
            <span style={{ width: 3, height: 3, borderRadius: 999, background: TEXT_COLOR }} />
            <span style={{ width: 3, height: 3, borderRadius: 999, background: TEXT_COLOR }} />
            <span style={{ width: 3, height: 3, borderRadius: 999, background: TEXT_COLOR }} />
            <span style={{ width: 3, height: 3, borderRadius: 999, background: TEXT_COLOR }} />
          </span>
          <span style={{ transform: 'translateY(-0.5px)' }}>Search</span>
        </div>
        <div
          style={{
            display: 'flex',
            height: 24,
            width: 118,
            alignItems: 'stretch',
            overflow: 'hidden',
            border: '1px solid #5b5b5b',
            background: '#d8d8d8',
            flexShrink: 0,
          }}
        >
          <input
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            style={{
              minWidth: 0,
              flex: 1,
              border: 'none',
              background: 'transparent',
              padding: '0 6px',
              fontSize: 11,
              color: '#1f1f1f',
              outline: 'none',
            }}
            placeholder=""
          />
          <div
            style={{
              display: 'flex',
              width: 23,
              alignItems: 'center',
              justifyContent: 'center',
              borderLeft: '1px solid #7a7a7a',
              background: '#a5a5a5',
              color: '#2b2b2b',
            }}
          >
            <Search size={12} strokeWidth={2.1} />
          </div>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: `${NO_COLUMN_WIDTH}px minmax(0, 1fr)`,
          background: HEADER_BACKGROUND,
          color: TEXT_COLOR,
        }}
      >
        <div
          style={{
            padding: '8px 8px 9px 8px',
            fontSize: 12,
            fontWeight: 700,
            lineHeight: 1.1,
          }}
        >
          No.
        </div>
        <div
          style={{
            borderLeft: `1px solid ${DIVIDER_COLOR}`,
            padding: '8px 10px 9px 10px',
            fontSize: 12,
            fontWeight: 700,
            lineHeight: 1.1,
          }}
        >
          Name
        </div>
      </div>

      <div
        style={{
          position: 'relative',
          minHeight: 0,
          flex: 1,
          overflowY: 'auto',
          overflowX: 'hidden',
        }}
      >
        <div
          aria-hidden="true"
          style={{
            pointerEvents: 'none',
            position: 'absolute',
            inset: '0 auto 0 0',
            zIndex: 2,
            left: NO_COLUMN_WIDTH,
            width: 1,
            background: DIVIDER_COLOR,
          }}
        />

        {filteredRows.length === 0 ? (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: `${NO_COLUMN_WIDTH}px minmax(0, 1fr)`,
              color: MUTED_TEXT_COLOR,
            }}
          >
            <div style={{ padding: '12px 8px', fontSize: 11, lineHeight: 1.15 }}>No.</div>
            <div style={{ padding: '12px 10px', fontSize: 11, lineHeight: 1.15 }}>Name</div>
          </div>
        ) : (
          filteredRows.map((row) => (
            <button
              key={row.id}
              type="button"
              onClick={() => {
                if (row.primarySeriesId) onSelectSeries(row.primarySeriesId);
              }}
              style={{
                display: 'grid',
                width: '100%',
                gridTemplateColumns: `${NO_COLUMN_WIDTH}px minmax(0, 1fr)`,
                border: 'none',
                background: row.selected ? '#3c3c3c' : 'transparent',
                padding: 0,
                color: TEXT_COLOR,
                textAlign: 'left',
                cursor: row.primarySeriesId ? 'pointer' : 'default',
              }}
            >
              <div
                style={{
                  padding: '10px 8px',
                  fontSize: 11,
                  lineHeight: 1.15,
                  color: MUTED_TEXT_COLOR,
                  whiteSpace: 'nowrap',
                }}
              >
                {String(row.index + 1).padStart(2, '0')}
              </div>
              <div
                style={{
                  padding: '10px 10px',
                  fontSize: 11,
                  lineHeight: 1.15,
                  color: MUTED_TEXT_COLOR,
                  minWidth: 0,
                }}
              >
                <span
                  style={{
                    display: 'block',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {row.displayName}
                </span>
              </div>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
