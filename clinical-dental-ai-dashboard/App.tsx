
import React, { useState, useCallback } from 'react';
import { ToothData, Patient, ToothStatus } from './types';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import PanoramicViewer from './components/PanoramicViewer';
import Odontogram from './components/Odontogram';
import DiagnosticPanel from './components/DiagnosticPanel';

const INITIAL_PATIENT: Patient = {
  name: "James Smith",
  dob: "12/04/1985",
  id: "992831",
  scanDate: "Oct 24, 2023"
};

// FDI Numbering: 11-18 (Q1), 21-28 (Q2), 31-38 (Q3), 41-48 (Q4)
const generateFDITeeth = (): ToothData[] => {
  const quadrants = [1, 2, 3, 4];
  const teeth: ToothData[] = [];

  quadrants.forEach(q => {
    for (let i = 1; i <= 8; i++) {
      const id = q * 10 + i;
      let status: ToothStatus = 'Healthy';
      let confidence = Math.floor(Math.random() * 20);
      let pblMesial = 1.2;
      let pblDistal = 1.1;

      // Mock specific cases using FDI IDs
      if (id === 46) { // Lower Right 1st Molar (prev 30)
        status = 'Decay';
        confidence = 98;
        pblMesial = 4.2;
        pblDistal = 2.1;
      } else if (id === 11) { // Upper Right Central Incisor (prev 8)
        status = 'Watch';
        confidence = 25;
      } else if (id === 26 || id === 36 || id === 14) { // Restorations
        status = 'Restoration';
        confidence = 0;
      } else if (id === 18 || id === 28 || id === 38 || id === 48) { // Wisdom teeth often missing
        status = 'Missing';
        confidence = 0;
      }

      teeth.push({ id, status, confidence, pblMesial, pblDistal, notes: "" });
    }
  });

  return teeth;
};

const App: React.FC = () => {
  const [patient] = useState<Patient>(INITIAL_PATIENT);
  const [teeth, setTeeth] = useState<ToothData[]>(generateFDITeeth());
  const [selectedToothId, setSelectedToothId] = useState<number>(46);
  const [showOverlay, setShowOverlay] = useState(true);

  const activeTooth = teeth.find(t => t.id === selectedToothId) || teeth[0];

  const handleUpdateTooth = useCallback((updatedTooth: ToothData) => {
    setTeeth(prev => prev.map(t => t.id === updatedTooth.id ? updatedTooth : t));
  }, []);

  return (
    <div className="flex flex-col h-screen overflow-hidden font-sans text-slate-200">
      <Header patient={patient} />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          showOverlay={showOverlay} 
          onToggleOverlay={() => setShowOverlay(!showOverlay)} 
        />
        
        <main className="flex-1 flex flex-col min-w-0 bg-[#0a0f18] relative">
          <PanoramicViewer 
            selectedToothId={selectedToothId} 
            showOverlay={showOverlay}
            onSelectTooth={setSelectedToothId}
          />
          
          <Odontogram 
            teeth={teeth} 
            selectedId={selectedToothId} 
            onSelect={setSelectedToothId} 
          />
        </main>

        <DiagnosticPanel 
          tooth={activeTooth} 
          onUpdate={handleUpdateTooth}
        />
      </div>
    </div>
  );
};

export default App;
