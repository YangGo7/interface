
import React from 'react';
import { 
  Layers, 
  Contrast, 
  Sun, 
  ZoomIn, 
  RotateCcw 
} from 'lucide-react';

interface SidebarProps {
  showOverlay: boolean;
  onToggleOverlay: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ showOverlay, onToggleOverlay }) => {
  return (
    <aside className="w-16 flex-none flex flex-col items-center py-4 gap-6 border-r border-[#282e39] bg-[#111318] z-10">
      <div className="group relative">
        <button 
          onClick={onToggleOverlay}
          className={`size-10 rounded-lg flex items-center justify-center transition-all ${
            showOverlay ? 'bg-[#135bec] text-white' : 'text-slate-400 hover:bg-slate-800'
          }`}
        >
          <Layers size={20} />
        </button>
        <div className="absolute left-14 bg-slate-800 text-white text-xs px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-50">
          Toggle AI Overlay
        </div>
      </div>

      <div className="w-8 h-px bg-slate-800"></div>

      <div className="flex flex-col gap-4">
        {[
          { icon: Contrast, label: 'Contrast' },
          { icon: Sun, label: 'Brightness' },
          { icon: ZoomIn, label: 'Zoom' },
          { icon: RotateCcw, label: 'Reset' },
        ].map((tool, idx) => (
          <button 
            key={idx}
            className="size-10 rounded-lg flex items-center justify-center text-slate-400 hover:bg-slate-800 transition-colors"
            title={tool.label}
          >
            <tool.icon size={20} />
          </button>
        ))}
      </div>
    </aside>
  );
};

export default Sidebar;
