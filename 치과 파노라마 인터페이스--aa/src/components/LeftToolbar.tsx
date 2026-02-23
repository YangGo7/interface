/**
 * 좌측 수직 툴바 컴포넌트
 * 
 * 기능:
 * - 이미지 조작 도구 버튼 제공
 * - 도구: Select, Pan, Zoom, Measure, Fit, Brightness, Rotate, Grid, Copy, Undo
 * - 활성화된 도구 시각적 표시 (cyan 배경)
 * - 호버 효과로 사용성 향상
 */

import { MousePointer, Hand, ZoomIn, ScanLine, Maximize2, Droplet, RotateCcw, Grid3x3, Copy, Undo } from 'lucide-react';

export function LeftToolbar() {
  const tools = [
    { icon: MousePointer, label: 'Select', active: false },
    { icon: Hand, label: 'Pan', active: false },
    { icon: ZoomIn, label: 'Zoom', active: false },
    { icon: ScanLine, label: 'Measure', active: false },
    { icon: Maximize2, label: 'Fit', active: false },
    { icon: Droplet, label: 'Brightness', active: false },
    { icon: RotateCcw, label: 'Rotate', active: false },
    { icon: Grid3x3, label: 'Grid', active: false },
    { icon: Copy, label: 'Copy', active: true },
    { icon: Undo, label: 'Undo', active: false },
  ];

  return (
    <aside className="w-16 bg-[#0f0f0f] border-r border-gray-800 flex flex-col items-center py-4 gap-2">
      {tools.map((tool, index) => (
        <button
          key={index}
          className={`w-10 h-10 flex items-center justify-center rounded transition-colors ${
            tool.active ? 'bg-cyan-600 text-white' : 'hover:bg-[#1a1a1a] text-gray-400'
          }`}
          title={tool.label}
        >
          <tool.icon className="w-5 h-5" />
        </button>
      ))}
    </aside>
  );
}