import { AlertTriangle, X } from "lucide-react";

interface DangerConfirmModalProps {
  open: boolean;
  isMobileViewport: boolean;
  title: string;
  description: string;
  confirmLabel: string;
  submitting: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}

export function DangerConfirmModal({
  open,
  isMobileViewport,
  title,
  description,
  confirmLabel,
  submitting,
  onClose,
  onConfirm,
}: DangerConfirmModalProps) {
  if (!open) {
    return null;
  }

  return (
    <div
      className={`fixed inset-0 z-50 flex bg-black/70 ${
        isMobileViewport ? "items-stretch justify-stretch p-0" : "items-center justify-center"
      }`}
      onClick={() => {
        if (!submitting) {
          onClose();
        }
      }}
    >
      <div
        className={
          isMobileViewport
            ? "h-full w-full overflow-y-auto bg-[#111620] px-5 pb-5 pt-6"
            : "w-[580px] rounded-2xl border border-[#3f2730] bg-[#111620] p-6"
        }
        onClick={(event) => event.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-4">
          <div className="flex items-start gap-3">
            <div className="mt-1 rounded-lg border border-[#6a2a37] bg-[#3a1720] p-2 text-[#ff6b81]">
              <AlertTriangle size={18} />
            </div>
            <div>
              <h3 className="text-2xl font-semibold text-white">{title}</h3>
              <p className="mt-2 text-base text-[#b8bfcc]">{description}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={submitting}
            className="rounded-md p-1 text-[#a1a8b4] transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
          >
            <X size={20} />
          </button>
        </div>

        <p className="rounded-lg border border-[#55313a] bg-[#23151a] px-4 py-3 text-sm text-[#ffb7c5]">
          This action is permanent and cannot be undone.
        </p>

        <div className="mt-6 flex justify-end gap-3">
          <button
            onClick={onClose}
            disabled={submitting}
            className="rounded-xl border border-[#303642] bg-[#171c27] px-5 py-2.5 text-base font-semibold text-white transition-colors hover:bg-[#1d2330] disabled:cursor-not-allowed disabled:opacity-60"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              void onConfirm();
            }}
            disabled={submitting}
            className={`rounded-xl px-5 py-2.5 text-base font-semibold transition-colors ${
              submitting
                ? "cursor-not-allowed bg-[#6f2c3b] text-[#e7b3bf]"
                : "bg-[#c24157] text-white hover:bg-[#a43448]"
            }`}
          >
            {submitting ? "Deleting..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
