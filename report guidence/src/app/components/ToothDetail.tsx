import { AlertCircle } from 'lucide-react';

interface ToothDetailProps {
  toothNumber: string;
}

export function ToothDetail({ toothNumber }: ToothDetailProps) {
  return (
    <div className="border-2 border-gray-200 rounded-2xl p-8 bg-white">
      <div className="flex gap-6">
        {/* X-ray Image Placeholder */}
        <div className="w-48 h-64 bg-gray-200 rounded-lg flex-shrink-0 flex items-center justify-center">
          <span className="text-gray-400 text-sm">X-ray</span>
        </div>

        {/* Details */}
        <div className="flex-1">
          <h3 className="text-3xl font-bold mb-3">Tooth #{toothNumber}</h3>
          
          <div className="flex items-start gap-2 mb-4">
            <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 flex-shrink-0" />
            <span className="text-xl font-semibold text-red-600">Caries (Cavity)</span>
          </div>

          <p className="text-gray-500 mb-6 text-sm">
            This area shows signs that may require treatment.<br />
            Please check this tooth specifically.
          </p>

          <div>
            <h4 className="text-xl font-bold mb-3">Common Treatments</h4>
            <ol className="space-y-2 text-gray-700">
              <li>1. Resin/GI Filling (Simple)</li>
              <li>2. Inlay/Onlay (Moderate)</li>
              <li>3. Root Canal & Crown (Severe)</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}
