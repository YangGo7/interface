
import React from 'react';
import { Download, Stethoscope } from 'lucide-react';
import { Patient } from '../types';

interface HeaderProps {
  patient: Patient;
}

const Header: React.FC<HeaderProps> = ({ patient }) => {
  return (
    <header className="flex-none flex items-center justify-between border-b border-[#282e39] bg-[#111318] px-6 py-3 z-20">
      <div className="flex items-center gap-4">
        <div className="size-8 flex items-center justify-center bg-blue-500/20 rounded-lg text-blue-400">
          <Stethoscope size={20} />
        </div>
        <div>
          <h2 className="text-sm font-bold leading-tight text-white">{patient.name}</h2>
          <p className="text-[11px] text-slate-400">DOB: {patient.dob} | ID: #{patient.id}</p>
        </div>
        <div className="h-8 w-px bg-slate-800 mx-2"></div>
        <p className="text-sm font-medium text-slate-400">Scan Date: {patient.scanDate}</p>
      </div>
      
      <div className="flex items-center gap-4">
        <button className="flex items-center justify-center rounded-lg h-9 px-4 bg-[#135bec] text-white text-sm font-bold shadow-sm hover:bg-blue-600 transition-colors">
          <Download size={16} className="mr-2" />
          <span>Export Report</span>
        </button>
        <div 
          className="size-9 rounded-full ring-2 ring-slate-800 bg-cover bg-center"
          style={{ backgroundImage: `url('https://picsum.photos/seed/doctor1/100/100')` }}
        ></div>
      </div>
    </header>
  );
};

export default Header;
