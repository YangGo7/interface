export type AppNumberingSystem = 'fdi' | 'univ';

const NUMBERING_SYSTEM_KEY = 'saturn:numbering-system';

export function readStoredNumberingSystem(): AppNumberingSystem {
  if (typeof window === 'undefined') return 'fdi';
  const raw = window.localStorage.getItem(NUMBERING_SYSTEM_KEY);
  return raw === 'univ' ? 'univ' : 'fdi';
}

export function writeStoredNumberingSystem(value: AppNumberingSystem) {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(NUMBERING_SYSTEM_KEY, value);
}
