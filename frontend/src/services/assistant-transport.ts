import {
  appendChatMessages,
  createChatThread,
} from "./workspace-api";

export type AssistantChatMessage = {
  id: string;
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

export type AssistantDebugEvent = {
  stage: string;
  detail: string;
  data?: Record<string, unknown>;
  at?: number;
};

export type AssistantStreamEvent =
  | { type: "ready" }
  | { type: "debug"; event: AssistantDebugEvent }
  | { type: "final"; message: AssistantChatMessage | null; threadId?: string | null }
  | { type: "error"; message: string }
  | { type: "done" };

export type AssistantReplyOptions = {
  threadId?: string | null;
  selectedDocIds?: string[];
};

export type AssistantReplyResult = {
  message: AssistantChatMessage | null;
  threadId: string | null;
};

type AnswerApiResponse = {
  question: string;
  answer: string;
  sources?: Array<Record<string, unknown>>;
  retrieval_results?: Array<Record<string, unknown>>;
};

const ANSWER_API_URL =
  import.meta.env.VITE_RAG_ANSWER_API_URL ?? "/api/answer";
const ANSWER_COLLECTION =
  import.meta.env.VITE_RAG_COLLECTION ?? "documents";
const ANSWER_DOC_LANGUAGE =
  import.meta.env.VITE_RAG_DOC_LANGUAGE ?? "mixed";
const ANSWER_TOP_K = Number(import.meta.env.VITE_RAG_TOP_K ?? "5");

function normalizeThreadId(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function newDebugEvent(
  stage: string,
  detail: string,
  data?: Record<string, unknown>,
): AssistantDebugEvent {
  return {
    stage,
    detail,
    data,
    at: Date.now(),
  };
}

function compactText(value: unknown, max = 140): string {
  const raw = typeof value === "string" ? value : value == null ? "" : String(value);
  const compact = raw.replace(/\s+/g, " ").trim();
  if (compact.length <= max) return compact;
  return `${compact.slice(0, max)}...`;
}

function toNumberOrNull(value: unknown): number | null {
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function buildRecallPreview(
  retrievalResults: Array<Record<string, unknown>> | undefined,
): Record<string, unknown> {
  const rows = Array.isArray(retrievalResults) ? retrievalResults : [];
  const preview = rows.slice(0, 6).map((row, idx) => ({
    rank: idx + 1,
    doc_id: String(row.doc_id ?? ""),
    heading_path: String(row.heading_path ?? ""),
    source: String(row.source ?? ""),
    chunk_index: typeof row.chunk_index === "number" ? row.chunk_index : toNumberOrNull(row.chunk_index),
    rerank_score: toNumberOrNull(row.rerank_score),
    rrf_score: toNumberOrNull(row.rrf_score),
    snippet: compactText(row.content),
  }));
  return {
    total_hits: rows.length,
    top_hits: preview,
  };
}

function buildCitations(
  sources: Array<Record<string, unknown>> | undefined,
  retrievalResults: Array<Record<string, unknown>> | undefined,
): AssistantChatMessage["citations"] {
  if (!Array.isArray(sources) || sources.length === 0) return [];
  const rows = Array.isArray(retrievalResults) ? retrievalResults : [];
  const sorted = [...sources].sort((a, b) => Number(a?.index ?? 0) - Number(b?.index ?? 0));

  return sorted.map((src, i) => {
    const idxRaw = Number(src?.index);
    const index = Number.isFinite(idxRaw) && idxRaw > 0 ? idxRaw : i + 1;
    const hit = rows[index - 1] ?? {};
    return {
      index,
      heading_path: String(src?.heading_path ?? hit?.heading_path ?? ""),
      doc_id: String(hit?.doc_id ?? ""),
      source_name: String(hit?.doc_id ?? ""),
      source_type: String(src?.source_type ?? hit?.source ?? ""),
      chunk_index:
        typeof hit?.chunk_index === "number" ? hit.chunk_index : toNumberOrNull(hit?.chunk_index) ?? -1,
      snippet: compactText(hit?.content, 200),
      rerank_score: toNumberOrNull(hit?.rerank_score),
      rrf_score: toNumberOrNull(hit?.rrf_score),
    };
  });
}

export const requestAssistantReply = async (
  userInput: string,
  _messageHistory: AssistantChatMessage[],
  options?: AssistantReplyOptions,
  onStreamEvent?: (event: AssistantStreamEvent) => void,
): Promise<AssistantReplyResult> => {
  const startedAt = Date.now();
  const selectedDocIds = options?.selectedDocIds ?? [];

  onStreamEvent?.({ type: "ready" });

  const response = await fetch(ANSWER_API_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({
      question: userInput,
      collection_name: ANSWER_COLLECTION,
      doc_language: ANSWER_DOC_LANGUAGE,
      top_k: ANSWER_TOP_K,
      selected_doc_ids: selectedDocIds,
    }),
  });

  if (!response.ok) {
    const detail = await response.text();
    const message = detail || `HTTP ${response.status}`;
    onStreamEvent?.({ type: "error", message });
    throw new Error(message);
  }

  const data = (await response.json()) as AnswerApiResponse;
  const recallPreview = buildRecallPreview(data.retrieval_results);
  const citations = buildCitations(data.sources, data.retrieval_results);

  onStreamEvent?.({
    type: "debug",
    event: newDebugEvent("retrieve", "Top recalled chunks", recallPreview),
  });

  const answerText = (data.answer || "").trim() || "No answer returned.";
  const threadId = normalizeThreadId(options?.threadId);
  const finalThreadId =
    threadId ?? (await createChatThread({ selected_doc_ids: selectedDocIds })).thread_id;

  const debugPayload: AssistantChatMessage["debug"] = {
    question: userInput,
    selected_doc_ids: selectedDocIds,
    startedAt,
    completedAt: Date.now(),
    statusText: "completed",
    events: [],
  };

  await appendChatMessages(
    finalThreadId,
    userInput,
    answerText,
    selectedDocIds,
    debugPayload,
    { citations },
  );

  const assistantMessage: AssistantChatMessage = {
    id: `assistant-${Date.now()}`,
    role: "assistant",
    content: answerText,
    debug: debugPayload,
    citations,
  };

  onStreamEvent?.({
    type: "final",
    message: assistantMessage,
    threadId: finalThreadId,
  });
  onStreamEvent?.({ type: "done" });

  return {
    message: assistantMessage,
    threadId: finalThreadId,
  };
};
