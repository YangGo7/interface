function normalizeToothValue(value: any) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'object') {
    return String(value.tooth_label || value.tooth || value.label || '').trim();
  }
  return String(value).trim();
}

function uniqueSortedTeeth(values: Array<string | number>) {
  return [...new Set(values.map((value) => normalizeToothValue(value)).filter(Boolean))]
    .sort((a, b) => Number(a) - Number(b));
}

function teethFromMap(map?: Record<string, any>) {
  return uniqueSortedTeeth(Object.keys(map || {}));
}

function teethFromList(list?: Array<string | number>) {
  return uniqueSortedTeeth(Array.isArray(list) ? (list as any[]) : []);
}

function formatGroupedFinding(label: string, teeth: string[]) {
  if (!teeth.length) return '';
  const lines = [ `${label}:`, ...teeth.map((tooth) => `#${tooth}`) ];
  return lines.join('\n');
}

function boneLossKeywords(result: any) {
  const pbl = result?.pbl || {};
  const entries = Object.entries(pbl)
    .map(([tooth, value]) => ({ tooth: String(tooth), pct: Number(value || 0) }))
    .filter((entry) => entry.pct > 0)
    .sort((a, b) => b.pct - a.pct)
    .slice(0, 3);

  if (!entries.length) return [];

  return [formatGroupedFinding('Bone loss', entries.map((entry) => entry.tooth))];
}

export function buildWebReportKeywords(result: any): string[] {
  if (!result) return [];

  const keywords: string[] = [];
  const groups: Array<[string, string[]]> = [
    ['Caries', teethFromMap(result?.caries_by_tooth_best || result?.caries_by_tooth)],
    ['Periapical', teethFromMap(result?.periapical_by_tooth_best || result?.periapical_by_tooth)],
    ['Missing', teethFromList(result?.missing_teeth || result?.teeth_missing)],
    ['Implant', teethFromMap(result?.implant_by_tooth_best || result?.implant_by_tooth)],
    ['Crown', teethFromMap(result?.crown_by_tooth_best || result?.crown_by_tooth)],
    ['Filling', teethFromMap(result?.filling_by_tooth_best || result?.filling_by_tooth)],
  ];

  groups.forEach(([label, teeth]) => {
    if (!teeth.length) return;
    keywords.push(formatGroupedFinding(label, teeth.slice(0, 5)));
  });

  keywords.push(...boneLossKeywords(result));

  if (!keywords.length) {
    keywords.push('No major findings detected');
  }

  return keywords;
}

export function countWebReportFindingTeeth(result: any): number {
  if (!result) return 0;

  const teeth = new Set<string>();
  const groups = [
    Object.keys(result?.caries_by_tooth_best || result?.caries_by_tooth || {}),
    Object.keys(result?.periapical_by_tooth_best || result?.periapical_by_tooth || {}),
    Object.keys(result?.implant_by_tooth_best || result?.implant_by_tooth || {}),
    Object.keys(result?.crown_by_tooth_best || result?.crown_by_tooth || {}),
    Object.keys(result?.filling_by_tooth_best || result?.filling_by_tooth || {}),
    Array.isArray(result?.missing_teeth) ? result.missing_teeth : Array.isArray(result?.teeth_missing) ? result.teeth_missing : [],
  ];

  groups.forEach((group) => {
    group.forEach((tooth: string | number) => teeth.add(String(tooth)));
  });

  return teeth.size;
}
