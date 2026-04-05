import { WORKSPACE_API_BASE_URL } from "../../services/workspace-api";

export const normalizeErrorMessage = (error: unknown): string =>
  error instanceof Error ? error.message : "Unknown error";

export const formatUploadedLabel = (uploadedAt: string): string => {
  const value = new Date(uploadedAt);
  if (Number.isNaN(value.getTime())) {
    return "Uploaded -";
  }
  return `Uploaded ${value.toLocaleString()}`;
};

export const formatUploadDate = (uploadedAt: string): string => {
  const value = new Date(uploadedAt);
  if (Number.isNaN(value.getTime())) return "Unknown";
  return value.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
};

export const isPlaceholderDocumentDescription = (value: string): boolean => {
  const normalized = value.trim().toLowerCase();
  return normalized.length === 0 || normalized === "uploaded document";
};

export const getDocumentDescriptionLabel = (value: string): string =>
  isPlaceholderDocumentDescription(value) ? "No description yet." : value;

export const getDocumentThumbnailUrl = (docId: string, width: number): string =>
  `${WORKSPACE_API_BASE_URL}/documents/${encodeURIComponent(docId)}/thumbnail?width=${width}&v=content`;

const DOCUMENT_PROCESSING_STATUSES = new Set([
  "pending",
  "parsing",
  "chunking",
  "indexing",
  "updating",
  "deleting",
]);
const DOCUMENT_READY_STATUSES = new Set(["ready"]);

export const isDocumentProcessing = (status: string): boolean =>
  DOCUMENT_PROCESSING_STATUSES.has((status || "").trim().toLowerCase());

export const isDocumentReady = (status: string): boolean =>
  DOCUMENT_READY_STATUSES.has((status || "").trim().toLowerCase());

export const isDocumentFailed = (status: string): boolean =>
  (status || "").trim().toLowerCase() === "failed";

export const getDocumentStatusLabel = (status: string): string => {
  const value = (status || "").trim().toLowerCase();
  if (value === "pending") return "Queued";
  if (value === "parsing") return "Parsing";
  if (value === "chunking") return "Chunking";
  if (value === "indexing") return "Indexing";
  if (value === "updating") return "Updating";
  if (value === "deleting") return "Deleting";
  if (value === "uploaded_only") return "Uploaded only";
  if (value === "uploaded") return "Uploaded";
  if (value === "failed") return "Parse failed";
  if (value === "ready") return "Ready";
  return "Unavailable";
};
