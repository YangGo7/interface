
import React from 'react';
import { ToothData, ToothStatus } from '../types';

interface OdontogramProps {
  teeth: ToothData[];
  selectedId: number;
  onSelect: (id: number) => void;
}

const Odontogram: React.FC<OdontogramProps> = ({ teeth, selectedId, onSelect }) => {
  // FDI Layout:
  // Upper: Q1 (18 to 11) | Q2 (21 to 28)
  // Lower: Q4 (48 to 41) | Q3 (31 to 38)
  
  const q1 = teeth.filter(t => Math.floor(t.id / 10) === 1).sort((a, b) => b.id - a.id);
  const q2 = teeth.filter(t => Math.floor(t.id / 10) === 2).sort((a, b) => a.id - b.id);
  const q4 = teeth.filter(t => Math.floor(t.id / 10) === 4).sort((a, b) => b.id - a.id);
  const q3 = teeth.filter(t => Math.floor(t.id / 10) === 3).sort((a, b) => a.id - b.id);

  const upperRow = [...q1, ...q2];
  const lowerRow = [...q4, ...q3];

  const renderTooth = (tooth: ToothData, isUpper: boolean) => {
    let colorClass = 'bg-white';
    let borderColorClass = 'border-transparent';
    let labelColorClass = 'text-blue-400';

    switch (tooth.status) {
      case 'Decay':
        colorClass = 'bg-red-500';
        labelColorClass = 'text-red-400';
        break;
      case 'Watch':
        colorClass = 'bg-yellow-500';
        labelColorClass = 'text-yellow-400';
        break;
      case 'Restoration':
        colorClass = 'bg-slate-400/80';
        borderColorClass = 'border-slate-300';
        break;
      case 'Missing':
        colorClass = 'bg-slate-900';
        borderColorClass = 'border-slate-800';
        labelColorClass = 'text-slate-600';
        break;
    }

    const isSelected = tooth.id === selectedId;

    return (
      <button 
        key={tooth.id}
        onClick={() => onSelect(tooth.id)}
        className="flex flex-col items-center gap-1 group w-10 relative"
      >
        {isSelected && (
           <div className={`absolute ${isUpper ? '-bottom-10' : '-top-10'} bg-blue-500 text-white text-[9px] py-0.5 px-2 rounded animate-bounce z-20`}>
             FDI {tooth.id}
           </div>
        )}
        
        {isUpper && <span className={`text-[9px] ${isSelected ? 'font-bold text-white' : 'text-slate-500'}`}>{tooth.id}</span>}
        <div 
          className={`w-8 h-9 transition-all ${colorClass} ${borderColorClass} ${
            isUpper ? 'rounded-b-lg' : 'rounded-t-lg'
          } ${isSelected ? 'ring-2 ring-blue-500 ring-offset-2 ring-offset-[#1e293b] scale-110 z-10' : 'hover:scale-105'}`}
        ></div>
        {!isUpper && <span className={`text-[9px] ${isSelected ? 'font-bold text-white' : 'text-slate-500'}`}>{tooth.id}</span>}
        
        <span className={`text-[9px] font-mono mt-0.5 ${labelColorClass}`}>
          {tooth.status === 'Missing' ? 'N/A' : `${tooth.confidence}%`}
        </span>
      </button>
    );
  };

  return (
    <div className="h-auto min-h-[220px] bg-[#1e293b] border-t border-[#282e39] flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 bg-[#192231] border-b border-[#282e39]">
        <h3 className="text-[10px] font-semibold text-slate-400 uppercase tracking-widest">FDI World Dental Federation System</h3>
        <div className="flex items-center gap-4 text-[10px]">
          <LegendItem color="bg-white" label="Healthy" />
          <LegendItem color="bg-red-500" label="Decay" />
          <LegendItem color="bg-yellow-500" label="Watch" />
          <LegendItem color="bg-slate-400" label="Restoration" />
          <LegendItem color="bg-slate-900 border border-slate-700" label="Missing" />
        </div>
      </div>

      <div className="flex-1 overflow-x-auto p-4 flex flex-col items-center justify-center gap-6">
        <div className="flex gap-1 items-end px-10">
          {upperRow.map(t => renderTooth(t, true))}
        </div>
        <div className="flex gap-1 items-start px-10">
          {lowerRow.map(t => renderTooth(t, false))}
        </div>
      </div>
    </div>
  );
};

const LegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center gap-1.5">
    <div className={`size-2 rounded-full ${color}`}></div>
    <span className="text-slate-300">{label}</span>
  </div>
);

export default Odontogram;
