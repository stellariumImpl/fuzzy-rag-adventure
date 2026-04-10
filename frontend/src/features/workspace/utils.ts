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

const normalizeDescriptionToken = (value: string): string =>
  value.trim().replace(/\s+/g, " ").toLowerCase();

const sourceNameStem = (sourceName?: string): string =>
  (sourceName || "").replace(/\.[a-zA-Z0-9]+$/, "").trim();

export const isPlaceholderDocumentDescription = (
  value: string,
  sourceName?: string,
): boolean => {
  const normalized = value.trim().toLowerCase();
  if (normalized.length === 0 || normalized === "uploaded document") return true;
  const stem = normalizeDescriptionToken(sourceNameStem(sourceName));
  return stem.length > 0 && normalizeDescriptionToken(value) === stem;
};

export const getDocumentDescriptionLabel = (
  value: string,
  sourceName?: string,
): string => (isPlaceholderDocumentDescription(value, sourceName) ? "No description yet." : value);

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
  if (value === "failed") return "Ingest failed";
  if (value === "ready") return "Ready";
  return "Unavailable";
};
