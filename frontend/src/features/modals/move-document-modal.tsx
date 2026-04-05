import { ChevronDown, Folder, X } from "lucide-react";
import type { DocumentItem } from "../workspace/models";

interface MoveFolderRow {
  id: string;
  name: string;
  depth: number;
}

interface MoveDocumentModalProps {
  movingDocument: DocumentItem | null;
  isMobileViewport: boolean;
  moveModalSubmitting: boolean;
  moveDestinationFolderId: string | null;
  moveFolderRows: MoveFolderRow[];
  selectedMoveDestinationLabel: string;
  onClose: () => void;
  onSetMoveDestinationFolderId: (folderId: string | null) => void;
  onConfirm: () => Promise<void>;
}

export function MoveDocumentModal({
  movingDocument,
  isMobileViewport,
  moveModalSubmitting,
  moveDestinationFolderId,
  moveFolderRows,
  selectedMoveDestinationLabel,
  onClose,
  onSetMoveDestinationFolderId,
  onConfirm,
}: MoveDocumentModalProps) {
  if (!movingDocument) {
    return null;
  }

  return (
    <div
      className={`absolute inset-0 z-50 flex bg-black/70 ${
        isMobileViewport ? "items-stretch justify-stretch p-0" : "items-center justify-center"
      }`}
      onClick={() => {
        if (!moveModalSubmitting) {
          onClose();
        }
      }}
    >
      <div
        className={
          isMobileViewport
            ? "h-full w-full overflow-y-auto bg-[#0b111b] px-5 pb-5 pt-6"
            : "w-[640px] rounded-2xl border border-[#2d3442] bg-[#0b111b] p-7 shadow-[0_28px_64px_rgba(0,0,0,0.55)]"
        }
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-6">
          <div>
            <h3
              className={`${isMobileViewport ? "text-2xl" : "text-3xl"} font-semibold leading-tight text-white`}
            >
              Move File to Folder
            </h3>
            <p className="mt-3 text-base leading-snug text-[#a1aab8]">
              Select a folder for "{movingDocument.name}"
            </p>
          </div>
          <button
            onClick={onClose}
            disabled={moveModalSubmitting}
            className="rounded-md p-1 text-[#9ca5b4] transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
          >
            <X size={22} />
          </button>
        </div>

        <div className="mt-7 rounded-xl border border-[#2a3240] bg-[#0a0f18] p-3">
          <button
            onClick={() => onSetMoveDestinationFolderId(null)}
            className={`flex w-full items-center gap-3 rounded-lg px-4 py-3 text-left text-lg font-medium transition-colors ${
              moveDestinationFolderId === null
                ? "bg-[#2a2f39] text-white"
                : "text-[#d6dbe6] hover:bg-[#1b202c]"
            }`}
          >
            <ChevronDown size={16} />
            <Folder size={20} className="text-[#c0c7d4]" />
            <span>Root Directory</span>
          </button>
          <div className="mt-1">
            {moveFolderRows.length === 0 ? (
              <p className="px-4 py-3 text-sm text-[#8d95a5]">No folders in root</p>
            ) : (
              moveFolderRows.map((row) => (
                <button
                  key={row.id}
                  onClick={() => onSetMoveDestinationFolderId(row.id)}
                  className={`flex w-full items-center gap-3 rounded-lg py-2.5 pr-4 text-left text-base transition-colors ${
                    moveDestinationFolderId === row.id
                      ? "bg-[#252b37] text-white"
                      : "text-[#c8cfdd] hover:bg-[#1a1f2b]"
                  }`}
                  style={{
                    paddingLeft: `${44 + row.depth * 28}px`,
                  }}
                >
                  <Folder size={18} className="text-[#b8c0cd]" />
                  <span>{row.name}</span>
                </button>
              ))
            )}
          </div>
        </div>

        <p className="mt-5 text-base text-[#9ca4b3]">Destination: {selectedMoveDestinationLabel}</p>

        <div className="mt-8 flex justify-end gap-3">
          <button
            onClick={onClose}
            disabled={moveModalSubmitting}
            className="rounded-xl border border-[#303846] bg-[#171c27] px-6 py-2.5 text-base font-semibold text-white transition-colors hover:bg-[#1c2330] disabled:cursor-not-allowed disabled:opacity-60"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              void onConfirm();
            }}
            disabled={moveModalSubmitting}
            className={`rounded-xl px-6 py-2.5 text-base font-semibold transition-colors ${
              moveModalSubmitting
                ? "cursor-not-allowed bg-[#253247] text-[#7d8696]"
                : "bg-[#4A90D9] text-white hover:bg-[#3c84cf]"
            }`}
          >
            {moveModalSubmitting ? "Moving..." : "Confirm Move"}
          </button>
        </div>
      </div>
    </div>
  );
}
