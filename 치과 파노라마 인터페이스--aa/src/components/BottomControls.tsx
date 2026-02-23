/**
 * 하단 컨트롤 컴포넌트
 * 
 * 기능:
 * - 감지 항목 표시/숨김 토글
 * - 이상 소견별 필터링 (충치, 근단 방사선 투과성, 기타 감지, 하악관)
 * - 각 필터별 색상 레전드 표시
 * - 보고서 생성 버튼
 */

export function BottomControls() {
  return (
    <div className="bg-[#0f0f0f] border-t border-gray-800 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <button className="text-sm text-gray-400 hover:text-gray-200">
          Hide detections
        </button>
        
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" defaultChecked className="w-4 h-4" />
            <span className="w-3 h-3 rounded-full bg-orange-500"></span>
            <span className="text-gray-300">Caries</span>
          </label>
          
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" defaultChecked className="w-4 h-4" />
            <span className="w-3 h-3 rounded-full bg-red-500"></span>
            <span className="text-gray-300">Periapical radiolucency</span>
          </label>
          
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" defaultChecked className="w-4 h-4" />
            <span className="text-gray-300">Other detections</span>
          </label>
          
          <label className="flex items-center gap-2 text-sm">
            <input type="checkbox" className="w-4 h-4" />
            <span className="text-gray-300">Mandibular canal</span>
          </label>
        </div>
      </div>
      
      <button className="bg-cyan-600 hover:bg-cyan-700 text-white px-6 py-2 rounded text-sm transition-colors">
        Confirm and generate report
      </button>
    </div>
  );
}