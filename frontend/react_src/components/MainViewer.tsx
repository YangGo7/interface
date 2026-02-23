/**
 * 메인 파노라마 뷰어 컴포넌트
 * 
 * 기능:
 * - 환자 정보 표시 바 (이름, 생년월일, 성별, 촬영일)
 * - 파노라마 X-ray 이미지 표시 영역
 * - 현재는 플레이스홀더 표시 (실제 이미지 업로드 시 대체 가능)
 */

import { ChevronDown } from 'lucide-react';

export function MainViewer() {
  return (
    <div className="flex-1 bg-[#0a0a0a] p-4 flex flex-col overflow-hidden">
      {/* Patient Info Bar */}
      <div className="flex items-center gap-6 mb-4 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-gray-400">Demo patient</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400">1990-04-26 (44)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400">M</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400">2024-10-23</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-gray-400">2024-12-23</span>
          <ChevronDown className="w-4 h-4" />
        </div>
      </div>

      {/* Main Image Area */}
      <div className="flex-1 bg-black rounded border border-gray-700 overflow-hidden relative flex items-center justify-center">
        <div className="text-center text-gray-600">
          <div className="text-4xl mb-2">🦷</div>
          <p className="text-sm">파노라마 X-ray 뷰어 영역</p>
        </div>
      </div>
    </div>
  );
}