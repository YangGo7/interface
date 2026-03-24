export const isDicomFile = (inputFile: File) => /\.(dcm|dicom)$/i.test(inputFile.name);

export type UploadSelectionState = {
  activeFiles: File[];
  primaryFile: File | null;
  folderMode: boolean;
  inputSummary: string;
};

export function buildUploadSelection(
  singleFile: File | null,
  folderFiles: File[]
): UploadSelectionState {
  const folderMode = folderFiles.length > 0;
  const activeFiles = folderMode ? folderFiles : singleFile ? [singleFile] : [];
  const primaryFile = folderMode
    ? folderFiles.find(isDicomFile) || folderFiles[0] || null
    : singleFile;

  return {
    activeFiles,
    primaryFile,
    folderMode,
    inputSummary: folderMode
      ? `${folderFiles.length} files selected`
      : singleFile
        ? singleFile.name
        : 'Image / DICOM input',
  };
}
