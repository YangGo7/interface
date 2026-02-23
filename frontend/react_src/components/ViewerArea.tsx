export function ViewerArea() {
  return (
    <main className="flex-1 bg-[#1a1a1a] p-6 flex items-center justify-center overflow-hidden">
      <div className="w-full h-full bg-[#0f0f0f] rounded-lg border-2 border-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 bg-[#252525] rounded-lg flex items-center justify-center">
            <svg
              className="w-8 h-8 text-gray-600"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
          </div>
          <p className="text-gray-500 text-sm">파노라마 이미지 영역</p>
        </div>
      </div>
    </main>
  );
}
