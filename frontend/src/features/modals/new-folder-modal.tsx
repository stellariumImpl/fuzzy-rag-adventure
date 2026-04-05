import { X } from "lucide-react";

interface NewFolderModalProps {
  open: boolean;
  isMobileViewport: boolean;
  currentFolderId: string | null;
  newFolderName: string;
  newFolderDescription: string;
  folderSubmitting: boolean;
  onClose: () => void;
  onNameChange: (value: string) => void;
  onDescriptionChange: (value: string) => void;
  onSubmit: () => Promise<void>;
}

export function NewFolderModal({
  open,
  isMobileViewport,
  currentFolderId,
  newFolderName,
  newFolderDescription,
  folderSubmitting,
  onClose,
  onNameChange,
  onDescriptionChange,
  onSubmit,
}: NewFolderModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className={`absolute inset-0 z-40 flex bg-black/65 ${
        isMobileViewport ? "items-stretch justify-stretch p-0" : "items-center justify-center"
      }`}
      onClick={onClose}
    >
      <div
        className={
          isMobileViewport
            ? "h-full w-full overflow-y-auto bg-[#111620] px-5 pb-5 pt-6"
            : "w-[620px] rounded-2xl border border-[#2e3542] bg-[#111620] p-6"
        }
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between">
          <div>
            <h3 className="text-2xl font-semibold text-white">Create New Folder</h3>
            <p className="mt-2 text-base text-[#a6adb9]">
              {currentFolderId
                ? "Create a new folder in the current directory"
                : "Create a new folder in the root directory"}
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-[#a1a8b4] transition-colors hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        <div className="space-y-4">
          <label className="block">
            <span className="mb-2 block text-base font-semibold text-white">Folder Name</span>
            <input
              value={newFolderName}
              onChange={(event) => onNameChange(event.target.value)}
              placeholder="Enter folder name..."
              className="h-14 w-full rounded-xl border border-[#303642] bg-[#171c27] px-4 text-base text-white outline-none placeholder:text-[#8f95a3] focus:border-[#4A90D9]"
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-base font-semibold text-white">
              Description (Optional)
            </span>
            <textarea
              value={newFolderDescription}
              onChange={(event) => onDescriptionChange(event.target.value)}
              placeholder="Add a description for this folder..."
              rows={3}
              className="w-full resize-none rounded-xl border border-[#303642] bg-[#171c27] px-4 py-3 text-base text-white outline-none placeholder:text-[#8f95a3] focus:border-[#4A90D9]"
            />
          </label>
        </div>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            className="rounded-xl border border-[#303642] bg-[#171c27] px-5 py-2.5 text-base font-semibold text-white transition-colors hover:bg-[#1d2330]"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              void onSubmit();
            }}
            disabled={!newFolderName.trim() || folderSubmitting}
            className={`rounded-xl px-5 py-2.5 text-base font-semibold transition-colors ${
              newFolderName.trim() && !folderSubmitting
                ? "bg-[#4A90D9] text-white hover:bg-[#3b81ca]"
                : "cursor-not-allowed bg-[#253247] text-[#7d8696]"
            }`}
          >
            {folderSubmitting ? "Creating..." : "Create Folder"}
          </button>
        </div>
      </div>
    </div>
  );
}
