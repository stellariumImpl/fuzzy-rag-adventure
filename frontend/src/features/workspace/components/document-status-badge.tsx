import { LoaderCircle, TriangleAlert } from "lucide-react";
import {
  getDocumentStatusLabel,
  isDocumentProcessing,
  isDocumentReady,
} from "../utils";

export function DocumentStatusBadge({ status }: { status: string }) {
  if (isDocumentReady(status)) {
    return null;
  }

  const processing = isDocumentProcessing(status);
  return (
    <span
      className={`doc-status-badge ${
        processing ? "doc-status-badge-processing" : "doc-status-badge-failed"
      }`}
    >
      {processing ? (
        <LoaderCircle size={12} className="animate-spin" />
      ) : (
        <TriangleAlert size={12} />
      )}
      <span>{getDocumentStatusLabel(status)}</span>
    </span>
  );
}
