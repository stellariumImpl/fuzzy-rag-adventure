import { useEffect, useState } from "react";
import { Upload, X } from "lucide-react";
import { Input } from "../../components/ui/input";

interface UploadModalProps {
  open: boolean;
  isMobileViewport: boolean;
  uploadingDocument: boolean;
  uploadDragOver: boolean;
  onClose: () => void;
  onOpenUploadFilePicker: () => void;
  onSetUploadDragOver: (value: boolean) => void;
  onUploadPickedFile: (file: File) => Promise<void>;
  onCreateMarkdownDocument: (content: string, sourceName?: string) => Promise<void>;
}

export function UploadModal({
  open,
  isMobileViewport,
  uploadingDocument,
  uploadDragOver,
  onClose,
  onOpenUploadFilePicker,
  onSetUploadDragOver,
  onUploadPickedFile,
  onCreateMarkdownDocument,
}: UploadModalProps) {
  const [markdownSourceName, setMarkdownSourceName] = useState("");
  const [markdownContent, setMarkdownContent] = useState("");

  useEffect(() => {
    if (!open) {
      setMarkdownSourceName("");
      setMarkdownContent("");
    }
  }, [open]);

  if (!open) {
    return null;
  }

  return (
    <div
      className={`absolute inset-0 z-40 flex bg-black/65 ${
        isMobileViewport
          ? "items-stretch justify-stretch p-0"
          : "items-start justify-center overflow-y-auto p-5"
      }`}
      onClick={onClose}
    >
      <div
        className={`upload-modal-panel ${
          isMobileViewport
            ? "h-full w-full overflow-y-auto bg-[#111620] px-5 pb-5 pt-6"
            : "max-h-[calc(100vh-40px)] w-[min(760px,96vw)] overflow-y-auto rounded-2xl border border-[#2e3542] bg-[#111620] p-6"
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-6 flex items-start justify-between">
          <h3 className="text-2xl font-semibold text-white">Upload Document</h3>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-[#a1a8b4] transition-colors hover:text-white"
          >
            <X size={20} />
          </button>
        </div>

        <button
          disabled={uploadingDocument}
          onClick={onOpenUploadFilePicker}
          onDragOver={(event) => {
            event.preventDefault();
            onSetUploadDragOver(true);
          }}
          onDragLeave={() => onSetUploadDragOver(false)}
          onDrop={(event) => {
            event.preventDefault();
            onSetUploadDragOver(false);
            const file = event.dataTransfer.files?.[0];
            if (!file) return;
            void onUploadPickedFile(file);
          }}
          className={`upload-modal-dropzone mb-6 flex ${
            isMobileViewport ? "h-[52vh] min-h-[320px]" : "h-[220px]"
          } w-full flex-col items-center justify-center rounded-xl border border-dashed transition-all disabled:cursor-not-allowed disabled:opacity-70 ${
            uploadDragOver
              ? "upload-modal-dropzone-active border-[#4A90D9] bg-[#182b45]"
              : "border-[#3a4352] bg-[#171c27] hover:border-[#4A90D9]/70"
          }`}
        >
          <div className="upload-modal-dropzone-icon mb-5 flex h-16 w-16 items-center justify-center rounded-full bg-[#222937] text-[#b4bac6]">
            <Upload size={30} />
          </div>
          <p className="text-2xl font-semibold text-white">
            {uploadingDocument ? "Uploading..." : "Select Documents"}
          </p>
          <p className="mt-2 text-base text-[#a7aeb9]">
            Drag & drop PDF/Markdown files here, or click to browse
          </p>
        </button>

        <div className="mb-4 border-t border-[#2e3542]" />

        <div className="space-y-3 rounded-xl border border-[#2e3542] bg-[#171c27] p-4">
          <h4 className="text-base font-semibold text-white">Or paste Markdown text</h4>
          <p className="text-sm text-[#a7aeb9]">
            Submit markdown directly without creating a local file.
          </p>
          <label className="block">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#90a0b8]">
              Source Name (optional)
            </span>
            <Input
              value={markdownSourceName}
              onChange={(event) => setMarkdownSourceName(event.target.value)}
              placeholder="meeting-notes.md"
              disabled={uploadingDocument}
              className="h-11 border-[#394253] bg-[#101722] text-[#e8edf7] placeholder:text-[#66758d]"
            />
          </label>

          <label className="block">
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#90a0b8]">
              Markdown Content
            </span>
            <textarea
              value={markdownContent}
              onChange={(event) => setMarkdownContent(event.target.value)}
              disabled={uploadingDocument}
              placeholder="# Title&#10;&#10;| Column A | Column B |&#10;|---|---|&#10;| A | B |"
              className="min-h-[180px] w-full rounded-lg border border-[#394253] bg-[#101722] px-3 py-2 text-sm text-[#e8edf7] placeholder:text-[#66758d] outline-none transition-colors focus:border-[#4A90D9]"
            />
          </label>

          <button
            type="button"
            disabled={uploadingDocument || markdownContent.trim().length === 0}
            onClick={() => {
              void (async () => {
                await onCreateMarkdownDocument(
                  markdownContent,
                  markdownSourceName.trim() || undefined,
                );
              })();
            }}
            className="inline-flex h-10 items-center justify-center rounded-lg bg-[#4A90D9] px-4 text-sm font-semibold text-white transition-colors hover:bg-[#3f82c8] disabled:cursor-not-allowed disabled:opacity-65"
          >
            {uploadingDocument ? "Submitting..." : "Create Markdown Document"}
          </button>
        </div>
      </div>
    </div>
  );
}
