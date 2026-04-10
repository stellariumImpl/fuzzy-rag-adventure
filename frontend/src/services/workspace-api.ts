export const WORKSPACE_API_BASE_URL =
  import.meta.env.VITE_WORKSPACE_API_BASE_URL ?? "/api";

const RAG_COLLECTION = import.meta.env.VITE_RAG_COLLECTION ?? "documents";
const RAG_DOC_LANGUAGE = import.meta.env.VITE_RAG_DOC_LANGUAGE ?? "mixed";

const FOLDERS_KEY = "powerful-rag-ui-folders-v1";

export type WorkspaceFolder = {
  folder_id: string;
  parent_id: string | null;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
};

export type WorkspaceDocument = {
  doc_id: string;
  folder_id: string | null;
  source_name: string;
  description: string;
  storage_path: string;
  mime_type: string | null;
  size_bytes: number;
  pages: number;
  status: string;
  uploaded_at: string;
  updated_at: string;
};

export type ChatThreadSummary = {
  thread_id: string;
  user_id: string;
  title: string;
  selected_doc_ids: string[];
  message_count: number;
  last_message_preview: string;
  created_at: string;
  updated_at: string;
  last_message_at: string;
};

export type ChatMessageRecord = {
  message_id: string;
  thread_id: string;
  role: "user" | "assistant";
  content: string;
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
  metadata?: {
    selected_docs?: Array<{
      doc_id: string;
      source_name: string;
      pages?: number;
    }>;
    answer_meta?: {
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
    };
  } | null;
  seq: number;
  created_at: string;
};

export type ChatThreadDetail = {
  thread_id: string;
  user_id: string;
  title: string;
  selected_doc_ids: string[];
  created_at: string;
  updated_at: string;
  last_message_at: string;
  messages: ChatMessageRecord[];
};

export type AssistantSettings = {
  user_id: string;
  llm_provider: string;
  llm_model: string;
  api_base: string;
  api_key_configured: boolean;
  api_key_preview: string;
  temperature: number;
  max_tokens: number;
  embedding_provider: string;
  embedding_model: string;
  embedding_api_base: string;
  embedding_api_key_configured: boolean;
  embedding_api_key_preview: string;
  created_at: string;
  updated_at: string;
};

export type AssistantConnectionTestPayload = {
  llm_provider?: string;
  llm_model?: string;
  api_base?: string;
  api_key?: string;
  temperature?: number;
  max_tokens?: number;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_api_base?: string;
  embedding_api_key?: string;
};

export type AssistantConnectionTestResult = {
  ok: boolean;
  provider: string;
  mode: string;
  message: string;
  latency_ms: number | null;
};

export type AssistantProviderOption = {
  provider: string;
  label: string;
  default_model: string;
  default_api_base: string;
  requires_api_key: boolean;
};

export type AssistantProviderList = {
  items: AssistantProviderOption[];
};

export type RetrievalPipelineStatus = {
  mode: string;
  reasons: string[];
  llm_provider: string;
  llm_model: string;
  llm_api_key_configured: boolean;
  embedding_provider: string;
  embedding_model: string;
  embedding_api_key_configured: boolean;
  ready_document_count: number;
};

export type LlamaParseSettings = {
  user_id: string;
  enabled: boolean;
  base_url: string;
  model: string;
  api_key_configured: boolean;
  api_key_preview: string;
  created_at: string;
  updated_at: string;
};

export type VectorStoreStats = {
  provider: string;
  dimension: number;
  vector_count: number;
  index_version: number;
  index_size_bytes: number;
};

export type DocumentPipelineChunkItem = {
  chunk_id: string;
  chunk_order: number;
  chunk_index?: number;
  chunk_level: string;
  heading_path?: string;
  char_count: number;
  page_start: number | null;
  page_end: number | null;
  preview: string;
  text: string;
  has_table?: boolean;
  embedding_index?: {
    provider: string;
    collection_name: string;
    point_id: string | null;
    chunk_index: number;
    dense_vector: string;
    sparse_vectors: string[];
    vector_dimension: number;
    status: string;
  };
};

export type DocumentPipelineDetails = {
  doc_id: string;
  status: string;
  parse: {
    provider: string;
    summary: string;
    text_chars: number;
    image_count: number;
    table_count: number;
    garbled_risk: number;
    markdown_files?: Array<{
      name: string;
      path: string;
      chars: number;
      preview?: string;
    }>;
    generated_markdown?: Array<{
      name: string;
      path: string;
      chars: number;
      preview?: string;
    }>;
    image_files?: Array<{
      name: string;
      path: string;
      size_bytes: number;
    }>;
    table_preview?: Array<Record<string, unknown>>;
    artifacts: Record<string, unknown>;
    updated_at: string;
  };
  chunk: {
    page: number;
    page_size: number;
    total: number;
    items: DocumentPipelineChunkItem[];
  };
  index: {
    index_profile: string;
    embedding_provider: string;
    embedding_model: string;
    total_chunks: number;
    embedded_chunks: number;
    missing_chunks: number;
    level_distribution?: Record<string, number>;
    tree?: {
      edge_count: number;
      root_nodes: number;
      leaf_nodes: number;
    };
    index_build?: Record<string, unknown>;
    ready: boolean;
    vector_store: VectorStoreStats;
  };
};

export type RetrievalTestItem = {
  chunk_id: string;
  doc_id: string;
  source_name: string;
  folder_id: string | null;
  status: string;
  chunk_level?: string;
  score: number;
  snippet: string;
};

export type RetrievalTestResponse = {
  query: string;
  index_profile?: string;
  doc_id: string | null;
  items: RetrievalTestItem[];
  lexical_items?: RetrievalTestItem[];
  vector_items?: RetrievalTestItem[];
};

export type IndexSearchItem = {
  chunk_id: string;
  doc_id: string;
  source_name: string;
  folder_id: string | null;
  status: string;
  chunk_level?: string;
  snippet: string;
  lexical_score?: number;
  vector_score?: number;
  overlap_score?: number;
  level_score?: number;
  hybrid_score?: number;
};

type CreateFolderPayload = {
  name: string;
  description?: string;
  parent_id?: string | null;
};

type UpdateFolderPayload = {
  name?: string;
  description?: string;
};

type UpdateDocumentPayload = {
  folder_id?: string | null;
  description?: string;
};

type CreateMarkdownDocumentPayload = {
  content: string;
  source_name?: string;
  folder_id?: string | null;
};

type CreateChatThreadPayload = {
  title?: string;
  selected_doc_ids?: string[];
};

type UpdateChatThreadPayload = {
  title?: string;
  selected_doc_ids?: string[];
};

type UpdateAssistantSettingsPayload = {
  llm_provider?: string;
  llm_model?: string;
  api_base?: string;
  api_key?: string;
  temperature?: number;
  max_tokens?: number;
  embedding_provider?: string;
  embedding_model?: string;
  embedding_api_base?: string;
  embedding_api_key?: string;
};

type UpdateLlamaParseSettingsPayload = {
  enabled?: boolean;
  api_key?: string;
  base_url?: string;
  model?: string;
};

type UploadApiResponse = {
  filename: string;
  stored_path: string;
  status: string;
  message: string;
  collection_name: string;
  doc_language: string;
  task_id?: string | null;
  doc_id?: string | null;
};

type SearchApiResult = {
  doc_id?: string;
  heading_path?: string;
  content?: string;
  score?: number;
  final_score?: number;
  rrf_score?: number;
};

type SearchApiResponse = {
  question: string;
  results: SearchApiResult[];
};

type ChatTitleApiResponse = {
  title: string;
};

function nowIso(): string {
  return new Date().toISOString();
}

function randomId(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function safeJsonParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw) as unknown;
    return parsed as T;
  } catch {
    return fallback;
  }
}

function readLocal<T>(key: string, fallback: T): T {
  return safeJsonParse<T>(window.localStorage.getItem(key), fallback);
}

function writeLocal<T>(key: string, value: T): void {
  window.localStorage.setItem(key, JSON.stringify(value));
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${WORKSPACE_API_BASE_URL}${path}`, init);

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  const text = await response.text();
  if (!text.trim()) {
    return undefined as T;
  }
  return JSON.parse(text) as T;
}

function readFolders(): WorkspaceFolder[] {
  return readLocal<WorkspaceFolder[]>(FOLDERS_KEY, []);
}

function writeFolders(rows: WorkspaceFolder[]): void {
  writeLocal(FOLDERS_KEY, rows);
}

function normalizeDocumentStatus(status: string): string {
  const s = (status || "").toLowerCase();
  if (s === "queued" || s === "running" || s === "parsing") return "parsing";
  if (s === "completed" || s === "ready") return "ready";
  if (s === "failed") return "failed";
  if (s === "uploaded_only") return "uploaded";
  return s || "uploaded";
}

async function fetchDocumentListFromBackend(): Promise<WorkspaceDocument[]> {
  const payload = await request<{ items: WorkspaceDocument[] }>("/documents");
  return (payload.items ?? []).map((doc) => ({
    ...doc,
    status: normalizeDocumentStatus(doc.status),
  }));
}

export async function listFolders(): Promise<WorkspaceFolder[]> {
  return readFolders();
}

export function createFolder(payload: CreateFolderPayload) {
  const now = nowIso();
  const rows = readFolders();
  const created: WorkspaceFolder = {
    folder_id: randomId("folder"),
    parent_id: payload.parent_id ?? null,
    name: payload.name,
    description: payload.description || "Empty",
    created_at: now,
    updated_at: now,
  };
  writeFolders([created, ...rows]);
  return Promise.resolve(created);
}

export function updateFolder(folderId: string, payload: UpdateFolderPayload) {
  const rows = readFolders();
  let updated: WorkspaceFolder | null = null;
  const next = rows.map((row) => {
    if (row.folder_id !== folderId) return row;
    updated = {
      ...row,
      name: payload.name ?? row.name,
      description: payload.description ?? row.description,
      updated_at: nowIso(),
    };
    return updated;
  });
  if (!updated) {
    return Promise.reject(new Error(`Folder not found: ${folderId}`));
  }
  writeFolders(next);
  return Promise.resolve(updated);
}

export function deleteFolder(folderId: string) {
  const rows = readFolders();
  const children = new Set<string>([folderId]);
  let changed = true;
  while (changed) {
    changed = false;
    for (const folder of rows) {
      if (folder.parent_id && children.has(folder.parent_id) && !children.has(folder.folder_id)) {
        children.add(folder.folder_id);
        changed = true;
      }
    }
  }
  writeFolders(rows.filter((row) => !children.has(row.folder_id)));
  return Promise.resolve();
}

export async function listDocuments(): Promise<WorkspaceDocument[]> {
  return fetchDocumentListFromBackend();
}

export async function listChatThreads(): Promise<ChatThreadSummary[]> {
  const payload = await request<{ items: ChatThreadSummary[] }>("/chat/threads");
  return (payload.items ?? []).sort((a, b) => b.updated_at.localeCompare(a.updated_at));
}

export function createChatThread(payload?: CreateChatThreadPayload) {
  return request<ChatThreadSummary>("/chat/threads", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      title: payload?.title,
      selected_doc_ids: payload?.selected_doc_ids ?? [],
    }),
  });
}

export function getChatThread(threadId: string) {
  return request<ChatThreadDetail>(`/chat/threads/${encodeURIComponent(threadId)}`);
}

export function updateChatThread(threadId: string, payload: UpdateChatThreadPayload) {
  return request<ChatThreadSummary>(`/chat/threads/${encodeURIComponent(threadId)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function generateChatTitle(payload: {
  question: string;
  answer: string;
  current_title?: string;
}) {
  const body = {
    question: payload.question,
    answer: payload.answer,
    current_title: payload.current_title,
  };
  const out = await request<ChatTitleApiResponse>("/chat/title", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  return (out.title || "").trim();
}

export function deleteChatThread(threadId: string) {
  return request<void>(`/chat/threads/${encodeURIComponent(threadId)}`, {
    method: "DELETE",
  });
}

export async function appendChatMessages(
  threadId: string,
  userText: string,
  assistantText: string,
  selectedDocIds: string[],
  debugPayload?: ChatMessageRecord["debug"],
  answerMeta?: NonNullable<NonNullable<ChatMessageRecord["metadata"]>["answer_meta"]>,
) {
  return request<ChatMessageRecord>(
    `/chat/threads/${encodeURIComponent(threadId)}/messages/append`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        user_text: userText,
        assistant_text: assistantText,
        selected_doc_ids: selectedDocIds,
        debug: debugPayload ?? null,
        answer_meta: answerMeta ?? null,
      }),
    },
  );
}

export async function replaceChatTurnWithLatest(
  threadId: string,
  payload: {
    target_user_message_id: string;
    target_assistant_message_id?: string | null;
    target_user_content?: string | null;
    target_assistant_content?: string | null;
  },
) {
  return request<ChatThreadDetail>(
    `/chat/threads/${encodeURIComponent(threadId)}/turns/replace-latest`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    },
  );
}

export async function uploadDocument(file: File, folderId?: string | null) {
  const form = new FormData();
  form.append("file", file);
  form.append("collection_name", RAG_COLLECTION);
  form.append("doc_language", RAG_DOC_LANGUAGE);

  const upload = await request<UploadApiResponse>("/upload", {
    method: "POST",
    body: form,
  });

  const docId = upload.doc_id || file.name;

  if (folderId !== undefined) {
    try {
      await request<WorkspaceDocument>(`/documents/${encodeURIComponent(docId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_id: folderId }),
      });
    } catch {
      // ignore
    }
  }

  const rows = await fetchDocumentListFromBackend();
  const matched = rows.find((doc) => doc.doc_id === docId);
  if (matched) return matched;

  return {
    doc_id: docId,
    folder_id: folderId ?? null,
    source_name: file.name,
    description: upload.message,
    storage_path: upload.stored_path,
    mime_type: file.type || null,
    size_bytes: file.size,
    pages: 1,
    status: upload.task_id ? "parsing" : "uploaded",
    uploaded_at: nowIso(),
    updated_at: nowIso(),
  };
}

export function createMarkdownDocument(payload: CreateMarkdownDocumentPayload) {
  const filename = (payload.source_name?.trim() || `note-${Date.now()}.md`).replace(/\s+/g, "_");
  const file = new File([payload.content], filename.endsWith(".md") ? filename : `${filename}.md`, {
    type: "text/markdown",
  });
  return uploadDocument(file, payload.folder_id ?? null);
}

export function deleteDocument(docId: string) {
  return request<void>(`/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
}

export function updateDocument(docId: string, payload: UpdateDocumentPayload) {
  return request<WorkspaceDocument>(`/documents/${encodeURIComponent(docId)}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

export async function autofillDocumentDescription(docId: string, force = false) {
  const query = force ? "?force=1" : "";
  return request<WorkspaceDocument>(
    `/documents/${encodeURIComponent(docId)}/description/autofill${query}`,
    {
      method: "POST",
    },
  );
}

export async function rebuildDocumentIndex(docIds?: string[]) {
  const rows = await fetchDocumentListFromBackend();
  const selected =
    docIds && docIds.length > 0
      ? rows.filter((item) => docIds.includes(item.doc_id))
      : rows;

  return {
    items: selected.map((item) => ({
      doc_id: item.doc_id,
      status: item.status === "ready" ? "already_ready" : "queued",
      chunks: undefined,
      embeddings: undefined,
      reason:
        item.status === "ready"
          ? "Document already indexed"
          : "Use upload pipeline to enqueue indexing",
    })),
  };
}

export async function searchDocumentIndex(
  query: string,
  options?: {
    limit?: number;
    folderId?: string | null;
  },
) {
  const topK = options?.limit ?? 10;
  const payload = await request<SearchApiResponse>("/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      question: query,
      collection_name: RAG_COLLECTION,
      doc_language: RAG_DOC_LANGUAGE,
      top_k: topK,
    }),
  });

  const docs = await fetchDocumentListFromBackend();
  const folderFilter = options?.folderId;

  const items: IndexSearchItem[] = (payload.results ?? [])
    .map((row, i) => {
      const docMeta = docs.find((d) => d.doc_id === (row.doc_id || ""));
      return {
        chunk_id: `${row.doc_id || "unknown"}-${i}`,
        doc_id: row.doc_id || "",
        source_name: docMeta?.source_name || row.heading_path || "document",
        folder_id: docMeta?.folder_id ?? null,
        status: docMeta?.status || "ready",
        chunk_level: row.heading_path || "chunk",
        snippet: row.content || "",
        lexical_score: row.score,
        vector_score: row.final_score,
        overlap_score: row.rrf_score,
        level_score: undefined,
        hybrid_score: row.final_score ?? row.score,
      };
    })
    .filter((item) => (folderFilter == null ? true : item.folder_id === folderFilter));

  return {
    query,
    index_profile: "qdrant-hybrid",
    items,
  };
}

export function getDocumentPipeline(
  docId: string,
  options?: {
    page?: number;
    pageSize?: number;
  },
) {
  const params = new URLSearchParams();
  if (options?.page != null) params.set("page", String(options.page));
  if (options?.pageSize != null) params.set("page_size", String(options.pageSize));
  const suffix = params.size > 0 ? `?${params.toString()}` : "";
  return request<DocumentPipelineDetails>(`/documents/${encodeURIComponent(docId)}/pipeline${suffix}`);
}

export async function runDocumentRetrievalTest(
  docId: string,
  payload: {
    query: string;
    limit?: number;
    only_current_document?: boolean;
  },
): Promise<RetrievalTestResponse> {
  const searched = await searchDocumentIndex(payload.query, { limit: payload.limit ?? 10 });
  const filtered = payload.only_current_document
    ? searched.items.filter((item) => item.doc_id === docId)
    : searched.items;

  const items: RetrievalTestItem[] = filtered.map((item) => ({
    chunk_id: item.chunk_id,
    doc_id: item.doc_id,
    source_name: item.source_name,
    folder_id: item.folder_id,
    status: item.status,
    chunk_level: item.chunk_level,
    score: item.hybrid_score ?? item.vector_score ?? item.lexical_score ?? 0,
    snippet: item.snippet,
  }));

  return {
    query: payload.query,
    index_profile: "qdrant-hybrid",
    doc_id: payload.only_current_document ? docId : null,
    items,
  };
}

export async function getAssistantSettings(
  options?:
    | string
    | {
        llmProvider?: string;
        embeddingProvider?: string;
      },
) {
  const query = new URLSearchParams();
  if (typeof options === "string") {
    if (options.trim()) {
      query.set("llm_provider", options.trim());
    }
  } else if (options) {
    if (options.llmProvider?.trim()) {
      query.set("llm_provider", options.llmProvider.trim());
    }
    if (options.embeddingProvider?.trim()) {
      query.set("embedding_provider", options.embeddingProvider.trim());
    }
  }
  const suffix = query.size > 0 ? `?${query.toString()}` : "";
  return request<AssistantSettings>(`/settings/assistant${suffix}`);
}

export function listAssistantProviders() {
  return request<AssistantProviderList>("/settings/providers/assistant");
}

export function listEmbeddingProviders() {
  return request<AssistantProviderList>("/settings/providers/embedding");
}

export async function getAssistantRetrievalStatus() {
  return request<RetrievalPipelineStatus>("/settings/assistant/retrieval-status");
}

export async function updateAssistantSettings(payload: UpdateAssistantSettingsPayload) {
  const body: {
    llm_provider?: string;
    llm_model?: string;
    api_base?: string;
    api_key?: string;
    temperature?: number;
    max_tokens?: number;
    embedding_provider?: string;
    embedding_model?: string;
    embedding_api_base?: string;
    embedding_api_key?: string;
  } = {
    ...(payload.llm_provider ? { llm_provider: payload.llm_provider } : {}),
    ...(payload.llm_model ? { llm_model: payload.llm_model } : {}),
    ...(payload.api_base ? { api_base: payload.api_base } : {}),
    ...(payload.temperature != null ? { temperature: payload.temperature } : {}),
    ...(payload.max_tokens != null ? { max_tokens: payload.max_tokens } : {}),
    ...(payload.embedding_provider ? { embedding_provider: payload.embedding_provider } : {}),
    ...(payload.embedding_model ? { embedding_model: payload.embedding_model } : {}),
    ...(payload.embedding_api_base ? { embedding_api_base: payload.embedding_api_base } : {}),
    ...(payload.api_key !== undefined ? { api_key: payload.api_key } : {}),
    ...(payload.embedding_api_key !== undefined
      ? { embedding_api_key: payload.embedding_api_key }
      : {}),
  };
  return request<AssistantSettings>("/settings/assistant", {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}

export async function testAssistantConnection(
  payload?: AssistantConnectionTestPayload,
): Promise<AssistantConnectionTestResult> {
  return request<AssistantConnectionTestResult>("/settings/assistant/test", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload ?? {}),
  });
}

export function getLlamaParseSettings() {
  return request<LlamaParseSettings>("/settings/llamaparse");
}

export async function updateLlamaParseSettings(
  payload: UpdateLlamaParseSettingsPayload,
) {
  const body: {
    enabled?: boolean;
    api_key?: string;
    base_url?: string;
    model?: string;
  } = {
    ...(payload.enabled != null ? { enabled: payload.enabled } : {}),
    ...(payload.base_url ? { base_url: payload.base_url } : {}),
    ...(payload.model ? { model: payload.model } : {}),
    ...(payload.api_key !== undefined ? { api_key: payload.api_key } : {}),
  };
  return request<LlamaParseSettings>("/settings/llamaparse", {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
}
