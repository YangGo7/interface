/**
 * 상단 헤더 컴포넌트
 * 
 * 기능:
 * - 앱 로고 및 타이틀 표시
 * - 주요 네비게이션 메뉴 (Upload new radiograph, Patients, Patient file, Analysis)
 * - 도움말 및 사용자 메뉴
 */

import { ChevronDown, HelpCircle } from 'lucide-react';

export function TopHeader() {
  return (
    <header className="bg-[#0f0f0f] border-b border-gray-800 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <div className="text-xl font-semibold">align</div>
          <div className="text-sm text-gray-400">x-ray insights</div>
        </div>
        
        <nav className="flex items-center gap-1">
          <button className="px-4 py-2 text-sm hover:bg-[#1a1a1a] rounded transition-colors">
            Upload new radiograph
          </button>
          <button className="px-4 py-2 text-sm hover:bg-[#1a1a1a] rounded transition-colors">
            Patients
          </button>
          <button className="px-4 py-2 text-sm hover:bg-[#1a1a1a] rounded transition-colors">
            Patient file
          </button>
          <button className="px-4 py-2 text-sm bg-[#1a1a1a] rounded transition-colors border-b-2 border-blue-500">
            Analysis
          </button>
        </nav>
      </div>
      
      <div className="flex items-center gap-4">
        <button className="flex items-center gap-2 text-sm hover:bg-[#1a1a1a] px-3 py-2 rounded">
          <HelpCircle className="w-4 h-4" />
          <span>Help</span>
        </button>
        <button className="flex items-center gap-2 text-sm hover:bg-[#1a1a1a] px-3 py-2 rounded">
          <span>User</span>
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>
    </header>
  );
}