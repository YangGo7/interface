import type { FolderStudy } from '../../features/upload/dicomFolderStudies';
import { StudiesWorkspacePanel } from './StudiesWorkspacePanel';

type RenewStudiesDockProps = {
  visible: boolean;
  left: string;
  top: string;
  width: string;
  height: string;
  studies: FolderStudy[];
  selectedSeriesId: string | null;
  onSelectSeries: (seriesId: string) => void;
};

export function RenewStudiesDock({
  visible,
  left,
  top,
  width,
  height,
  studies,
  selectedSeriesId,
  onSelectSeries,
}: RenewStudiesDockProps) {
  if (!visible) return null;

  return (
    <div
      style={{
        position: 'absolute',
        zIndex: 120,
        left,
        top,
        width,
        height,
        border: '1px solid rgba(255,255,255,0.04)',
        background: '#393939',
        overflow: 'hidden',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)',
      }}
    >
      <StudiesWorkspacePanel
        studies={studies}
        selectedSeriesId={selectedSeriesId}
        onSelectSeries={onSelectSeries}
        isVisible
      />
    </div>
  );
}
