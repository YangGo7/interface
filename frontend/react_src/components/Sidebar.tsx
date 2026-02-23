import { FileText, Users, Settings } from 'lucide-react';

export function Sidebar() {
  return (
    <aside className="w-64 bg-[#0f0f0f] border-r border-gray-800 flex flex-col">
      <div className="p-4 border-b border-gray-800">
        <h2 className="text-sm text-gray-400 mb-4">환자 목록</h2>
        <div className="space-y-2">
          {[1, 2, 3, 4, 5].map((i) => (
            <div
              key={i}
              className="p-3 bg-[#1a1a1a] hover:bg-[#252525] rounded cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-gray-400" />
                <div className="flex-1 min-w-0">
                  <div className="text-sm truncate">환자 {i}</div>
                  <div className="text-xs text-gray-500">2026.01.0{i}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="flex-1 p-4">
        <h2 className="text-sm text-gray-400 mb-4">도구</h2>
        <div className="space-y-2">
          <button className="w-full p-3 bg-[#1a1a1a] hover:bg-[#252525] rounded transition-colors text-left flex items-center gap-2">
            <FileText className="w-4 h-4" />
            <span className="text-sm">노트</span>
          </button>
          <button className="w-full p-3 bg-[#1a1a1a] hover:bg-[#252525] rounded transition-colors text-left flex items-center gap-2">
            <Settings className="w-4 h-4" />
            <span className="text-sm">설정</span>
          </button>
        </div>
      </div>
    </aside>
  );
}
