
export type ToothStatus = 'Healthy' | 'Decay' | 'Watch' | 'Restoration' | 'Missing';

export interface ToothData {
  id: number;
  status: ToothStatus;
  confidence: number;
  pblMesial: number;
  pblDistal: number;
  notes: string;
}

export interface Patient {
  name: string;
  dob: string;
  id: string;
  scanDate: string;
}
