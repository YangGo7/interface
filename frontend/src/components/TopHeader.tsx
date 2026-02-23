import logo from '../assets/O3_logo_only.png';

export function TopHeader() {

  return (
    <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center justify-between">
      <div className="flex items-center gap-2">
        <img
          src={logo}
          alt="E2DW logo"
          style={{ width: '36px', height: '36px' }}
          className="object-contain"
        />
        <div className="flex flex-col leading-tight">
          <div className="text-lg font-semibold text-gray-900">E2DW</div>
          <div className="text-xs text-gray-500">AI panoramic insights</div>
        </div>
      </div>
      <div className="text-sm text-gray-500">Analysis</div>
    </header>
  );
}
