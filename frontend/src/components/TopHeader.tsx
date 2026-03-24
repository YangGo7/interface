import type { ReactNode } from 'react';
import logo from '../assets/O3_logo_only.png';

type TopHeaderProps = {
  actions?: ReactNode;
};

export function TopHeader({ actions }: TopHeaderProps) {

  return (
    <header className="border-b border-gray-200 bg-white px-4 py-3">
      <div className="flex flex-wrap items-center gap-0">
        <div className="flex items-center gap-2">
          <img
            src={logo}
            alt="E2DW logo"
            style={{ width: '60px', height: '60px' }}
            className="object-contain"
          />
          <div className="flex flex-col leading-tight">
            <div className="text-lg font-semibold text-gray-900">E2DW</div>
            <div className="text-xs text-gray-500">AI panoramic insights</div>
          </div>
        </div>
        <div className="flex min-w-0 flex-1 justify-start">
          {actions ?? <div className="text-sm text-gray-500">Analysis</div>}
        </div>
      </div>
    </header>
  );
}
