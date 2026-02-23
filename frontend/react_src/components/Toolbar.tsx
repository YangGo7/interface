import { ZoomIn, ZoomOut, RotateCw, Maximize2, Move, Ruler } from 'lucide-react';

export function Toolbar() {
  return (
    <footer className="bg-[#0f0f0f] border-t border-gray-800 px-6 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="확대">
            <ZoomIn className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="축소">
            <ZoomOut className="w-5 h-5" />
          </button>
          <div className="w-px h-6 bg-gray-800 mx-2"></div>
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="회전">
            <RotateCw className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="이동">
            <Move className="w-5 h-5" />
          </button>
          <div className="w-px h-6 bg-gray-800 mx-2"></div>
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="측정">
            <Ruler className="w-5 h-5" />
          </button>
          <button className="p-2 hover:bg-[#252525] rounded transition-colors" title="전체화면">
            <Maximize2 className="w-5 h-5" />
          </button>
        </div>
        
        <div className="flex items-center gap-4 text-sm text-gray-400">
          <span>밝기: 100%</span>
          <span>대비: 100%</span>
          <span>배율: 100%</span>
        </div>
      </div>
    </footer>
  );
}
