import type {
  ChatThreadSummary,
  WorkspaceDocument,
  WorkspaceFolder,
} from "../../services/workspace-api";

export type NavKey = "new-chat" | "documents" | "library";

export interface FolderItem {
  id: string;
  parentId: string | null;
  name: string;
  description: string;
}

export interface DocumentItem {
  id: string;
  folderId: string | null;
  name: string;
  description: string;
  pages: number;
  uploadedAt: string;
  status: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  attachments?: Array<{
    docId: string;
    name: string;
    pages?: number;
  }>;
  debug?: {
    question?: string;
    selected_doc_ids?: string[];
    startedAt?: number;
    completedAt?: number;
    statusText?: string;
    events?: Array<{
      stage: string;
      detail: string;
      data?: Record<string, unknown>;
      at?: number;
    }>;
  } | null;
  citations?: Array<{
    index: number;
    heading_path: string;
    doc_id: string;
    source_name: string;
    source_type: string;
    chunk_index: number;
    snippet: string;
    rerank_score?: number | null;
    rrf_score?: number | null;
  }>;
}

export interface ChatThreadItem {
  id: string;
  title: string;
  selectedDocIds: string[];
  messageCount: number;
  lastMessagePreview: string;
  createdAt: string;
  updatedAt: string;
  lastMessageAt: string;
}

export interface ToastMessage {
  id: string;
  tone: "success" | "error" | "info";
  title: string;
  description?: string;
  durationMs: number;
}

export const mapFolderFromApi = (folder: WorkspaceFolder): FolderItem => ({
  id: folder.folder_id,
  parentId: folder.parent_id,
  name: folder.name,
  description: folder.description || "Empty",
});

export const mapDocumentFromApi = (document: WorkspaceDocument): DocumentItem => ({
  id: document.doc_id,
  folderId: document.folder_id,
  name: document.source_name,
  description: (document.description || "").trim(),
  pages: document.pages || 1,
  uploadedAt: document.uploaded_at,
  status: document.status || "ready",
});

export const mapChatThreadFromApi = (thread: ChatThreadSummary): ChatThreadItem => ({
  id: thread.thread_id,
  title: thread.title,
  selectedDocIds: Array.isArray(thread.selected_doc_ids) ? thread.selected_doc_ids : [],
  messageCount: Number.isFinite(thread.message_count) ? Number(thread.message_count) : 0,
  lastMessagePreview: thread.last_message_preview || "",
  createdAt: thread.created_at,
  updatedAt: thread.updated_at,
  lastMessageAt: thread.last_message_at,
});
