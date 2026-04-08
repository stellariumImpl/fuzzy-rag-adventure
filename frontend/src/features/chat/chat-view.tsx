import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  Check,
  Copy,
  Globe,
  MessageSquare,
  Pencil,
  Plus,
  RotateCcw,
  ThumbsDown,
  ThumbsUp,
  Upload,
  X,
} from "lucide-react";
import { HintIconButton } from "../../components/hint-icon-button";
import { ThinkingPanel } from "./thinking-panel";
import { DocumentThumbnail } from "../workspace/components/document-thumbnail";
import { DocumentStatusBadge } from "../workspace/components/document-status-badge";
import type { ChatMessage, DocumentItem } from "../workspace/models";
import {
  getDocumentDescriptionLabel,
  getDocumentStatusLabel,
  isDocumentProcessing,
  isDocumentReady,
} from "../workspace/utils";

interface ChatViewProps {
  chatStarted: boolean;
  documents: DocumentItem[];
  uploadingDocument: boolean;
  quickUploadDragOver: boolean;
  selectedDocs: string[];
  selectedDocuments: DocumentItem[];
  messages: ChatMessage[];
  inputValue: string;
  webSearchEnabled: boolean;
  onSetQuickUploadDragOver: (value: boolean) => void;
  onUploadPickedFile: (file: File) => Promise<void>;
  onOpenUploadFilePicker: () => void;
  onToggleDocumentSelectionWithGuard: (doc: DocumentItem) => void;
  onRemoveDocumentFromSelection: (docId: string) => void;
  onInputValueChange: (value: string) => void;
  onComposerKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onToggleWebSearch: () => void;
  onSend: () => Promise<void>;
}

const handleDropUpload = (
  event: React.DragEvent<HTMLElement>,
  onSetQuickUploadDragOver: (value: boolean) => void,
  onUploadPickedFile: (file: File) => Promise<void>,
) => {
  event.preventDefault();
  onSetQuickUploadDragOver(false);
  const file = event.dataTransfer.files?.[0];
  if (!file) return;
  void onUploadPickedFile(file);
};

const formatScore = (value: number | null | undefined): string => {
  if (typeof value !== "number" || !Number.isFinite(value)) return "-";
  return value.toFixed(4);
};

const normalizeSourceType = (value: string): string => {
  const normalized = String(value || "").trim();
  if (!normalized) return "-";
  return normalized.replace(/_/g, " ");
};

type FeedbackState = "up" | "down" | null;

export function ChatView({
  chatStarted,
  documents,
  uploadingDocument,
  quickUploadDragOver,
  selectedDocs,
  selectedDocuments,
  messages,
  inputValue,
  webSearchEnabled,
  onSetQuickUploadDragOver,
  onUploadPickedFile,
  onOpenUploadFilePicker,
  onToggleDocumentSelectionWithGuard,
  onRemoveDocumentFromSelection,
  onInputValueChange,
  onComposerKeyDown,
  onToggleWebSearch,
  onSend,
}: ChatViewProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const autoStickToBottomRef = useRef(true);
  const documentPickerRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const copyResetTimerRef = useRef<number | null>(null);
  const [documentPickerOpen, setDocumentPickerOpen] = useState(false);
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [feedbackByMessage, setFeedbackByMessage] = useState<Record<string, FeedbackState>>({});

  const latestMessage = useMemo(
    () => (messages.length > 0 ? messages[messages.length - 1] : null),
    [messages],
  );

  const isAssistantStreaming = Boolean(
    latestMessage &&
      latestMessage.role === "assistant" &&
      latestMessage.content.trim().length === 0,
  );

  const isNearBottom = () => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    const remaining =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    return remaining < 120;
  };

  const scrollToBottom = () => {
    const container = scrollContainerRef.current;
    if (!container) return;
    container.scrollTop = container.scrollHeight;
  };

  useEffect(() => {
    const forceStick =
      latestMessage?.role === "user" ||
      isAssistantStreaming;

    if (forceStick || autoStickToBottomRef.current) {
      window.requestAnimationFrame(() => {
        scrollToBottom();
      });
    }
  }, [messages, latestMessage, isAssistantStreaming]);

  useEffect(() => {
    return () => {
      if (copyResetTimerRef.current != null) {
        window.clearTimeout(copyResetTimerRef.current);
        copyResetTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!documentPickerOpen) return;
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (documentPickerRef.current && !documentPickerRef.current.contains(target)) {
        setDocumentPickerOpen(false);
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [documentPickerOpen]);

  const focusComposerWithText = (text: string) => {
    const next = String(text || "").trim();
    if (!next) return;
    onInputValueChange(next);
    window.requestAnimationFrame(() => {
      const el = composerRef.current;
      if (!el) return;
      el.focus();
      const cursor = next.length;
      el.setSelectionRange(cursor, cursor);
    });
  };

  const promptForMessage = (messageIndex: number): string => {
    const current = messages[messageIndex];
    if (!current) return "";
    if (current.role === "user") return current.content.trim();
    for (let i = messageIndex - 1; i >= 0; i -= 1) {
      const candidate = messages[i];
      if (candidate.role === "user" && candidate.content.trim().length > 0) {
        return candidate.content.trim();
      }
    }
    return "";
  };

  const copyText = async (messageId: string, value: string) => {
    const text = String(value || "").trim();
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const textarea = document.createElement("textarea");
      textarea.value = text;
      textarea.setAttribute("readonly", "true");
      textarea.style.position = "fixed";
      textarea.style.opacity = "0";
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
    }

    setCopiedMessageId(messageId);
    if (copyResetTimerRef.current != null) {
      window.clearTimeout(copyResetTimerRef.current);
    }
    copyResetTimerRef.current = window.setTimeout(() => {
      setCopiedMessageId((previous) => (previous === messageId ? null : previous));
      copyResetTimerRef.current = null;
    }, 1200);
  };

  const toggleFeedback = (messageId: string, target: Exclude<FeedbackState, null>) => {
    setFeedbackByMessage((previous) => {
      const current = previous[messageId] ?? null;
      return {
        ...previous,
        [messageId]: current === target ? null : target,
      };
    });
  };

  return (
    <section className="relative flex min-h-0 flex-1 flex-col px-6 pb-5 pt-8">
      <div
        ref={scrollContainerRef}
        onScroll={() => {
          autoStickToBottomRef.current = isNearBottom();
        }}
        className="flex-1 overflow-y-auto"
      >
        {!chatStarted ? (
          documents.length === 0 ? (
            <div className="flex min-h-full flex-col items-center justify-center gap-3">
              <div className="mb-1 flex flex-col items-center gap-3">
                <MessageSquare size={38} className="text-[#4A90D9]" />
                <p className="text-xl font-semibold text-white">Welcome to PageIndex</p>
              </div>

              <button
                disabled={uploadingDocument}
                onClick={onOpenUploadFilePicker}
                onDragOver={(event) => {
                  event.preventDefault();
                  onSetQuickUploadDragOver(true);
                }}
                onDragLeave={() => onSetQuickUploadDragOver(false)}
                onDrop={(event) =>
                  handleDropUpload(
                    event,
                    onSetQuickUploadDragOver,
                    onUploadPickedFile,
                  )
                }
                className={`quick-upload-tile flex min-h-[118px] w-full max-w-[640px] flex-col items-center justify-center gap-2 rounded-2xl border border-dashed transition-all disabled:cursor-not-allowed disabled:opacity-70 ${
                  quickUploadDragOver
                    ? "border-[#4A90D9] bg-[#182b45]"
                    : "border-[#3a4352] bg-transparent hover:bg-[#151b28]"
                }`}
              >
                <Upload size={22} className="text-[#8f95a3]" />
                <span className="text-lg font-semibold text-white">
                  {uploadingDocument ? "Uploading..." : "Upload a document to get started"}
                </span>
              </button>
            </div>
          ) : (
            <div className="flex min-h-full flex-col items-center gap-5 pt-1">
              <div className="mb-1 flex flex-col items-center gap-3">
                <MessageSquare size={38} className="text-[#4A90D9]" />
                <p className="text-2xl font-semibold text-white">Select documents to start</p>
              </div>

              <div className="grid w-full max-w-[980px] grid-cols-2 gap-3 pb-4">
                {documents.map((doc) => (
                  <button
                    key={doc.id}
                    onClick={() => onToggleDocumentSelectionWithGuard(doc)}
                    className={`flex min-h-[104px] w-full items-center gap-3 rounded-2xl border px-4 py-3 text-left transition-all ${
                      selectedDocs.includes(doc.id)
                        ? "border-[#4A90D9] bg-[#182b45] shadow-[0_0_0_1px_rgba(74,144,217,0.5)]"
                        : "border-[#2a313f] bg-[#151b28]"
                    } ${!isDocumentReady(doc.status) ? "cursor-not-allowed opacity-85" : ""}`}
                  >
                    <DocumentThumbnail
                      docId={doc.id}
                      docName={doc.name}
                      width={220}
                      className="h-[64px] w-[54px] flex-shrink-0 rounded-md"
                      iconSize={24}
                    />
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-[1.1rem] font-semibold text-white">
                          {doc.name}
                        </p>
                        <DocumentStatusBadge status={doc.status} />
                      </div>
                      <p className="mt-0.5 line-clamp-2 text-xs text-[#b9c0cc]">
                        {getDocumentDescriptionLabel(doc.description)}
                      </p>
                    </div>
                  </button>
                ))}

                <button
                  disabled={uploadingDocument}
                  onClick={onOpenUploadFilePicker}
                  onDragOver={(event) => {
                    event.preventDefault();
                    onSetQuickUploadDragOver(true);
                  }}
                  onDragLeave={() => onSetQuickUploadDragOver(false)}
                  onDrop={(event) =>
                    handleDropUpload(
                      event,
                      onSetQuickUploadDragOver,
                      onUploadPickedFile,
                    )
                  }
                  className={`quick-upload-tile flex min-h-[104px] w-full flex-col items-center justify-center gap-2 rounded-2xl border border-dashed transition-all disabled:cursor-not-allowed disabled:opacity-70 ${
                    quickUploadDragOver
                      ? "border-[#4A90D9] bg-[#182b45]"
                      : "border-[#3a4352] bg-transparent hover:bg-[#151b28]"
                  }`}
                >
                  <Upload size={24} className="text-[#8f95a3]" />
                  <span className="text-base font-semibold text-white">
                    {uploadingDocument ? "Uploading..." : "Upload Documents"}
                  </span>
                </button>
              </div>
            </div>
          )
        ) : (
          <div className="mx-auto flex min-h-full w-full max-w-[980px] flex-col gap-4 pr-1">
            {messages.map((message, messageIndex) => {
              const trimmedContent = message.content.trim();
              const bubbleContent =
                trimmedContent.length > 0 ? (
                  message.content
                ) : message.role === "assistant" ? (
                  message.debug ? null : <span className="text-[#99a9bc]">Thinking...</span>
                ) : (
                  message.content
                );
              const showTextBubble = bubbleContent !== null;

              return (
                <div
                  key={message.id}
                  className={`group flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`flex flex-col gap-2 ${
                      message.role === "user" ? "max-w-[72%] items-end" : "w-full max-w-full items-start"
                    }`}
                  >
                    {message.role === "assistant" && message.debug && (
                      <div className="w-full">
                        <ThinkingPanel
                          debug={message.debug}
                          isStreaming={trimmedContent.length === 0}
                        />
                      </div>
                    )}

                    {message.role === "user" &&
                      Array.isArray(message.attachments) &&
                      message.attachments.length > 0 && (
                        <div className="flex flex-wrap justify-end gap-1.5">
                          {message.attachments.map((attachment) => (
                            <span
                              key={`${message.id}-${attachment.docId}`}
                              className="inline-flex max-w-[280px] items-center gap-1 rounded-full border border-[#4d5565] bg-[#202632] px-2 py-0.5 text-[11px] text-[#d3d8e3]"
                              title={attachment.name}
                            >
                              <span className="truncate">{attachment.name}</span>
                              {attachment.pages != null && attachment.pages > 0 && (
                                <span className="text-[#9ca6b8]">{attachment.pages}p</span>
                              )}
                            </span>
                          ))}
                        </div>
                      )}

                    {showTextBubble && (
                      <div
                        className={`whitespace-pre-wrap break-words rounded-2xl px-4 py-3 text-sm leading-6 ${
                          message.role === "user"
                            ? "bg-[#2a2a2a] text-white"
                            : "bg-[#1a1a1a] text-[#d9dce3]"
                        }`}
                      >
                        {bubbleContent}
                      </div>
                    )}

                    {message.role === "assistant" &&
                      Array.isArray(message.citations) &&
                      message.citations.length > 0 && (
                        <div className="w-full rounded-xl border border-[#2f3a4e] bg-[#111827] p-3">
                          <div className="mb-2.5 flex items-center justify-between">
                            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[#93a4be]">
                              Sources
                            </p>
                            <span className="rounded-md border border-[#2e3b51] bg-[#0f1726] px-2 py-0.5 text-[11px] font-semibold text-[#b6c7df]">
                              {message.citations.length}
                            </span>
                          </div>
                          <div className="max-h-[420px] space-y-2.5 overflow-y-auto pr-1">
                            {message.citations.map((c, idx) => (
                              <article
                                key={`${message.id}-cite-${idx}`}
                                className="group rounded-lg border border-[#2e3b51] bg-[#0f1726] p-2.5 transition-colors hover:bg-[#111b2d]"
                              >
                                <div className="flex items-start gap-2">
                                  <span className="mt-0.5 inline-flex rounded bg-[#25344f] px-1.5 py-0.5 text-xs font-semibold text-[#b9d3ff]">
                                    [{c.index}]
                                  </span>
                                  <div className="min-w-0 flex-1">
                                    <p
                                      className="truncate text-[13px] font-semibold text-white"
                                      title={c.source_name || c.doc_id || "unknown"}
                                    >
                                      {c.source_name || c.doc_id || "unknown"}
                                    </p>
                                    {c.heading_path ? (
                                      <p
                                        className="mt-0.5 truncate text-[11px] text-[#9fb0c9]"
                                        title={c.heading_path}
                                      >
                                        {c.heading_path}
                                      </p>
                                    ) : null}
                                  </div>
                                </div>
                                <p
                                  className="source-snippet mt-2 text-[12px] leading-5 text-[#ced7e6]"
                                  title={c.snippet || "(no snippet)"}
                                >
                                  {c.snippet || "(no snippet)"}
                                </p>
                                <div className="mt-2 flex flex-wrap items-center gap-1.5 text-[11px] text-[#8ea0ba]">
                                  <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                    source: {normalizeSourceType(c.source_type)}
                                  </span>
                                  <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                    chunk: {c.chunk_index >= 0 ? c.chunk_index : "-"}
                                  </span>
                                  <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                    rerank: {formatScore(c.rerank_score)}
                                  </span>
                                  <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                    rrf: {formatScore(c.rrf_score)}
                                  </span>
                                </div>
                              </article>
                            ))}
                          </div>
                        </div>
                      )}

                    <div
                      className={`message-actions mt-1 flex items-center gap-1.5 opacity-0 transition-opacity duration-150 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100 ${
                        message.role === "user" ? "justify-end" : "justify-start"
                      }`}
                    >
                      <div className="inline-flex items-center gap-1">
                        <button
                          type="button"
                          aria-label="Copy message"
                          className="message-action-btn h-8 w-8 rounded-lg bg-transparent text-[#b4bfce] hover:text-white"
                          onClick={() => {
                            void copyText(message.id, message.content);
                          }}
                        >
                          {copiedMessageId === message.id ? <Check size={15} /> : <Copy size={15} />}
                        </button>

                        {message.role === "assistant" ? (
                          <>
                            <button
                              type="button"
                              aria-label="Like answer"
                              className={`message-action-btn h-8 w-8 rounded-lg bg-transparent hover:text-white ${
                                feedbackByMessage[message.id] === "up"
                                  ? "is-active text-[#8dc1ff]"
                                  : "text-[#b4bfce]"
                              }`}
                              onClick={() => {
                                toggleFeedback(message.id, "up");
                              }}
                            >
                              <ThumbsUp size={15} />
                            </button>

                            <button
                              type="button"
                              aria-label="Dislike answer"
                              className={`message-action-btn h-8 w-8 rounded-lg bg-transparent hover:text-white ${
                                feedbackByMessage[message.id] === "down"
                                  ? "is-active text-[#ff7c8f]"
                                  : "text-[#b4bfce]"
                              }`}
                              onClick={() => {
                                toggleFeedback(message.id, "down");
                              }}
                            >
                              <ThumbsDown size={15} />
                            </button>
                          </>
                        ) : (
                          <button
                            type="button"
                            aria-label="Edit question"
                            className="message-action-btn h-8 w-8 rounded-lg bg-transparent text-[#b4bfce] hover:text-white"
                            onClick={() => {
                              focusComposerWithText(message.content);
                            }}
                          >
                            <Pencil size={15} />
                          </button>
                        )}

                        <button
                          type="button"
                          aria-label="Reuse prompt"
                          className="message-action-btn h-8 w-8 rounded-lg bg-transparent text-[#b4bfce] hover:text-white"
                          onClick={() => {
                            focusComposerWithText(promptForMessage(messageIndex));
                          }}
                        >
                          <RotateCcw size={15} />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-4 flex justify-center">
        <div
          className="chat-composer-panel w-full max-w-[980px] rounded-2xl border border-[#2c3341] bg-[#171c27] p-3 shadow-2xl"
        >
          {selectedDocuments.length > 0 && (
            <div className="mb-3 flex max-w-full gap-2 overflow-x-auto pb-1">
              {selectedDocuments.map((doc) => (
                <div
                  key={doc.id}
                  className="group relative inline-flex min-w-[240px] max-w-[320px] items-center gap-2.5 rounded-xl border border-[#303642] bg-[#151b26] px-3 py-2"
                >
                  <button
                    onClick={() => onRemoveDocumentFromSelection(doc.id)}
                    className="selected-doc-remove-btn absolute right-2 top-2 flex h-6 w-6 items-center justify-center rounded-md border border-[#384154] bg-[#1a2334] text-[#d9e3f5] opacity-0 transition-opacity hover:bg-[#24324b] group-hover:opacity-100"
                    aria-label="Remove selected document"
                    title="Remove selected document"
                  >
                    <X size={14} />
                  </button>
                  <DocumentThumbnail
                    docId={doc.id}
                    docName={doc.name}
                    width={160}
                    className="h-12 w-10 flex-shrink-0 rounded-md"
                    iconSize={18}
                  />
                  <div className="min-w-0">
                    <p className="line-clamp-2 text-xs font-semibold text-white">{doc.name}</p>
                    <p className="mt-0.5 text-xs text-[#a5acb8]">{doc.pages} pages</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          <textarea
            ref={composerRef}
            value={inputValue}
            onChange={(event) => onInputValueChange(event.target.value)}
            onKeyDown={onComposerKeyDown}
            rows={2}
            placeholder="Ask a question..."
            className="w-full resize-none bg-transparent text-sm text-white outline-none placeholder:text-[#8f95a3]"
          />

          <div className="mt-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div ref={documentPickerRef} className="relative">
                <button
                  onClick={() => setDocumentPickerOpen((value) => !value)}
                  disabled={uploadingDocument}
                  className="chat-add-documents-trigger flex items-center gap-2 rounded-full border border-[#303642] bg-[#131925] px-3.5 py-1.5 text-sm font-semibold text-[#aeb4c0] transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-70"
                >
                  <Plus size={16} />
                  <span>Add documents</span>
                  {selectedDocs.length > 0 && (
                    <span className="ml-1 flex h-5 min-w-5 items-center justify-center rounded-full bg-[#4A90D9] px-1.5 text-xs font-semibold text-white">
                      {selectedDocs.length}
                    </span>
                  )}
                </button>

                {documentPickerOpen && (
                  <div className="chat-document-picker absolute bottom-[calc(100%+10px)] left-0 z-30 w-[360px] rounded-xl border border-[#303642] bg-[#151b26] p-3 shadow-[0_14px_34px_rgba(0,0,0,0.35)]">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <div>
                        <p className="text-sm font-semibold text-white">Select Documents</p>
                        <p className="mt-0.5 text-xs text-[#a5acb8]">
                          Choose documents for the next question only.
                        </p>
                      </div>
                      <button
                        onClick={() => {
                          setDocumentPickerOpen(false);
                          onOpenUploadFilePicker();
                        }}
                        className="rounded-md border border-[#303642] bg-[#131925] px-2 py-1 text-xs font-semibold text-[#aeb4c0] transition-colors hover:text-white"
                      >
                        Upload
                      </button>
                    </div>

                    <div className="max-h-[280px] space-y-1.5 overflow-y-auto pr-1">
                      {documents.length === 0 ? (
                        <p className="rounded-lg border border-dashed border-[#303642] px-3 py-4 text-center text-xs text-[#8f95a3]">
                          No uploaded documents yet.
                        </p>
                      ) : (
                        documents.map((doc) => {
                          const selected = selectedDocs.includes(doc.id);
                          return (
                            <button
                              key={doc.id}
                              onClick={() => onToggleDocumentSelectionWithGuard(doc)}
                              className={`w-full rounded-lg border px-2.5 py-2 text-left transition-colors ${
                                selected
                                  ? "border-[#4A90D9] bg-[#182b45]"
                                  : "border-[#2a313f] bg-[#151b28] hover:bg-[#1a2230]"
                              } ${!isDocumentReady(doc.status) ? "cursor-not-allowed opacity-85" : ""}`}
                            >
                              <div className="flex items-start gap-2">
                                <div
                                  className={`mt-0.5 h-4 w-4 rounded-full border ${
                                    selected
                                      ? "border-[#4A90D9] bg-[#4A90D9]"
                                      : "border-[#5a6172] bg-transparent"
                                  }`}
                                />
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-2">
                                    <p className="truncate text-xs font-semibold text-white">
                                      {doc.name}
                                    </p>
                                    <DocumentStatusBadge status={doc.status} />
                                  </div>
                                  <p className="mt-0.5 text-[11px] text-[#a5acb8]">
                                    {doc.pages} pages
                                  </p>
                                </div>
                              </div>
                            </button>
                          );
                        })
                      )}
                    </div>
                  </div>
                )}
              </div>

              <HintIconButton
                label="Web search"
                hint="Web search"
                wrapperClassName="composer-hint"
                className={`composer-websearch-btn rounded-full p-2 ${
                  webSearchEnabled ? "is-active" : ""
                }`}
                pressed={webSearchEnabled}
                onClick={onToggleWebSearch}
              >
                <Globe size={16} />
              </HintIconButton>
            </div>

            <button
              onClick={() => {
                void onSend();
              }}
              disabled={!inputValue.trim()}
              className={`flex h-9 w-9 items-center justify-center rounded-full transition-all ${
                inputValue.trim()
                  ? "bg-[#4A90D9] text-white hover:bg-[#3a80c9]"
                  : "cursor-not-allowed bg-[#263041] text-[#7c8595]"
              }`}
            >
              <ArrowUp size={16} />
            </button>
          </div>
        </div>
      </div>

      <p className="mt-4 text-center text-sm text-[#7f8795]">
        PageIndex can make mistakes, please check the response.
      </p>
    </section>
  );
}

export const buildProcessingToastMessage = (doc: DocumentItem): string =>
  `"${doc.name}" is ${getDocumentStatusLabel(doc.status).toLowerCase()}. Please wait.`;
