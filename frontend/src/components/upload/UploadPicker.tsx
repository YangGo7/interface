type UploadPickerProps = {
  inputSummary: string;
  activeFileCount: number;
  primaryFileName?: string | null;
  folderMode: boolean;
  onSingleFileChange: (file: File | null) => void;
  onFolderFilesChange: (files: File[]) => void;
};

export function UploadPicker({
  inputSummary,
  activeFileCount,
  primaryFileName,
  folderMode,
  onSingleFileChange,
  onFolderFilesChange,
}: UploadPickerProps) {
  return (
    <div className="w-full flex flex-col items-center gap-3">
      <input
        type="file"
        id="imageInput"
        accept="image/*,.dcm,.dicom,application/dicom"
        hidden
        onChange={(event) => onSingleFileChange(event.target.files?.[0] || null)}
      />
      <input
        type="file"
        id="folderInput"
        hidden
        multiple
        ref={(node) => {
          if (!node) return;
          node.setAttribute('webkitdirectory', '');
          node.setAttribute('directory', '');
        }}
        onChange={(event) => onFolderFilesChange(Array.from(event.target.files || []))}
      />
      <label
        htmlFor="imageInput"
        className="w-full py-4 px-6 rounded-full border border-gray-800 bg-[#0f0f0f] text-center cursor-pointer hover:border-blue-500 hover:text-blue-400 transition-colors shadow-lg"
      >
        {inputSummary}
      </label>
      <label
        htmlFor="folderInput"
        className="w-full py-3 px-6 rounded-full border border-gray-800 bg-[#0b1020] text-center cursor-pointer hover:border-cyan-400 hover:text-cyan-300 transition-colors shadow-lg text-sm font-semibold"
      >
        Select Folder
      </label>
      {folderMode && (
        <p className="w-full text-center text-xs text-gray-400">
          Representative upload: {primaryFileName || 'none'} / total {activeFileCount} files
        </p>
      )}
    </div>
  );
}
