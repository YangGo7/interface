import { useState } from 'react';
import { DentalReport } from '@/app/components/DentalReport';
import { ToothDetailsPage } from '@/app/components/ToothDetailsPage';

export default function App() {
  const [currentPage, setCurrentPage] = useState<'overview' | 'details'>('overview');

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <div className="sticky top-0 bg-white border-b border-gray-200 z-10">
        <div className="max-w-5xl mx-auto px-8 py-4 flex gap-4">
          <button
            onClick={() => setCurrentPage('overview')}
            className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
              currentPage === 'overview'
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Overview
          </button>
          <button
            onClick={() => setCurrentPage('details')}
            className={`px-6 py-2 rounded-lg font-semibold transition-colors ${
              currentPage === 'details'
                ? 'bg-black text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Tooth Details
          </button>
        </div>
      </div>

      {/* Content */}
      {currentPage === 'overview' ? <DentalReport /> : <ToothDetailsPage />}
    </div>
  );
}
