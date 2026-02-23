
import React, { useState, useEffect } from 'react';
import { ToothData } from '../types';

interface DiagnosticPanelProps {
  tooth: ToothData;
  onUpdate: (tooth: ToothData) => void;
}

const DiagnosticPanel: React.FC<DiagnosticPanelProps> = ({ tooth, onUpdate }) => {
  const [localNotes, setLocalNotes] = useState(tooth.notes);

  useEffect(() => {
    setLocalNotes(tooth.notes);
  }, [tooth]);

  const handleSave = () => {
    onUpdate({ ...tooth, notes: localNotes });
    alert(`Diagnosis for FDI Tooth #${tooth.id} saved.`);
  };

  const getFDIInfo = (id: number) => {
    const quadrant = Math.floor(id / 10);
    const position = id % 10;
    
    let arch = quadrant <= 2 ? "Maxillary Arch" : "Mandibular Arch";
    let side = (quadrant === 1 || quadrant === 4) ? "Right" : "Left";
    
    return { arch, side, quadrant, position };
  };

  const info = getFDIInfo(tooth.id);

  return (
    <aside className="w-96 flex-none bg-[#1e293b] border-l border-[#282e39] flex flex-col overflow-y-auto">
      <div className="p-6 flex flex-col gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="bg-blue-500/20 text-blue-400 text-[10px] font-bold px-2 py-0.5 rounded">FDI ISO</span>
            <span className="text-[10px] text-slate-400 uppercase tracking-tight">World Dental Federation</span>
          </div>
          <h2 className="text-xl font-bold text-white">Tooth #{tooth.id}</h2>
          <p className="text-sm text-slate-400">{info.arch} | {info.side} (Q{info.quadrant})</p>
        </div>

        <div className="w-full aspect-square rounded-xl bg-black border border-slate-700 relative overflow-hidden group">
          <div 
            className="absolute inset-0 bg-cover bg-center transition-transform duration-700 group-hover:scale-125"
            style={{ backgroundImage: `url('https://lh3.googleusercontent.com/aida-public/AB6AXuD6lbvoRc32bZxl6j8fZy1b9LtxsSnYtVPHt2ZcCjJnNuGRnPptAShfiSpWIe_QoyeGE_UU0Jv_JPS-um6bhupoeZhExbdNln7Ckgi5nuuL2lc5G4RcDKsBo42KltMnhhpvmy5Buqo7CclYujQbLiQASq17RI2BORE1leqoC3lk32izQVaTelZ3Uv1nHSKpxSkQ53bXSxdJyxWvqHA1l5FzI4LDWc8HKk4w-fgkJZ1w4-IZHHDhzTqauy1tG72xs-X_vrgN6BTahfo')` }}
          ></div>
          {tooth.status === 'Decay' && (
            <div className="absolute bottom-3 right-3 bg-red-500/90 backdrop-blur text-white text-[10px] font-bold px-2 py-1 rounded shadow-lg">
              FDI {tooth.id}: Caries
            </div>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <MetricCard 
            label="PBL (Mesial)" 
            value={tooth.pblMesial} 
            colorClass="text-red-500" 
            barColorClass="bg-red-500" 
            progress={tooth.pblMesial * 15}
          />
          <MetricCard 
            label="PBL (Distal)" 
            value={tooth.pblDistal} 
            colorClass="text-yellow-500" 
            barColorClass="bg-yellow-500" 
            progress={tooth.pblDistal * 15}
          />
          
          <div className="bg-[#111318] p-4 rounded-xl border border-slate-800 col-span-2">
            <div className="flex justify-between items-center mb-2">
              <p className="text-[11px] text-slate-500">Caries Confidence</p>
              <span className="text-[11px] font-bold text-white">{tooth.confidence}%</span>
            </div>
            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
              <div 
                className={`h-full transition-all duration-500 ${tooth.confidence > 80 ? 'bg-red-500' : 'bg-yellow-500'}`} 
                style={{ width: `${tooth.confidence}%` }}
              ></div>
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-3 flex-1">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Clinical Notes</label>
          <textarea 
            value={localNotes}
            onChange={(e) => setLocalNotes(e.target.value)}
            className="w-full bg-[#111318] text-white text-sm p-4 rounded-xl border border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none resize-none min-h-[120px]" 
            placeholder="Findings for this tooth..."
          ></textarea>
          <button 
            onClick={handleSave}
            className="w-full mt-4 bg-[#135bec] hover:bg-blue-600 text-white font-bold py-3 rounded-xl transition-all shadow-lg shadow-blue-900/30"
          >
            Update FDI Record
          </button>
        </div>
      </div>
    </aside>
  );
};

const MetricCard = ({ label, value, colorClass, barColorClass, progress }: any) => (
  <div className="bg-[#111318] p-4 rounded-xl border border-slate-800">
    <p className="text-[11px] text-slate-500 mb-1">{label}</p>
    <div className="flex items-baseline gap-1">
      <span className={`text-lg font-bold ${colorClass}`}>{value.toFixed(1)}</span>
      <span className="text-[10px] text-slate-400 uppercase">mm</span>
    </div>
    <div className="w-full bg-slate-800 h-1.5 mt-3 rounded-full overflow-hidden">
      <div className={`h-full ${barColorClass} transition-all duration-500`} style={{ width: `${Math.min(100, progress)}%` }}></div>
    </div>
  </div>
);

export default DiagnosticPanel;
