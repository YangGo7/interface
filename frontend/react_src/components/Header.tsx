export function Header() {
  return (
    <header className="bg-[#0f0f0f] border-b border-gray-800 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <h1 className="text-xl text-white">치과 파노라마 뷰어</h1>
      </div>
      
      <div className="flex items-center gap-6">
        <div className="text-sm">
          <span className="text-gray-400">환자명:</span>
          <span className="ml-2 text-white">홍길동</span>
        </div>
        <div className="text-sm">
          <span className="text-gray-400">차트번호:</span>
          <span className="ml-2 text-white">12345</span>
        </div>
        <div className="text-sm">
          <span className="text-gray-400">촬영일:</span>
          <span className="ml-2 text-white">2026.01.09</span>
        </div>
      </div>
    </header>
  );
}
