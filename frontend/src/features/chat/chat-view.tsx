import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUp,
  ChevronRight,
  Check,
  Copy,
  FileText,
  Folder,
  Globe,
  House,
  MessageSquare,
  Pencil,
  Plus,
  RotateCcw,
  Search,
  ThumbsDown,
  ThumbsUp,
  Upload,
  X,
} from "lucide-react";
import ReactMarkdown, {
  defaultUrlTransform,
  type UrlTransform,
} from "react-markdown";
import rehypeRaw from "rehype-raw";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import { HintIconButton } from "../../components/hint-icon-button";
import { ThinkingPanel } from "./thinking-panel";
import { DocumentThumbnail } from "../workspace/components/document-thumbnail";
import { DocumentStatusBadge } from "../workspace/components/document-status-badge";
import type {
  ChatMessage,
  DocumentItem,
  FolderItem,
} from "../workspace/models";
import {
  getDocumentDescriptionLabel,
  getDocumentStatusLabel,
  isDocumentProcessing,
  isDocumentReady,
} from "../workspace/utils";

interface ChatViewProps {
  chatStarted: boolean;
  folders: FolderItem[];
  documents: DocumentItem[];
  uploadingDocument: boolean;
  quickUploadDragOver: boolean;
  selectedDocs: string[];
  selectedDocuments: DocumentItem[];
  messages: ChatMessage[];
  inputValue: string;
  inferenceEnabled: boolean;
  onSetQuickUploadDragOver: (value: boolean) => void;
  onUploadPickedFile: (file: File) => Promise<void>;
  onOpenUploadFilePicker: () => void;
  onToggleDocumentSelectionWithGuard: (doc: DocumentItem) => void;
  onRemoveDocumentFromSelection: (docId: string) => void;
  onInputValueChange: (value: string) => void;
  onComposerKeyDown: (event: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onToggleInference: () => void;
  onSend: () => Promise<void>;
  onEditMessage: (messageIndex: number) => void;
  onRetryMessage: (messageIndex: number) => Promise<void>;
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

const attachmentTypeLabel = (filename: string): string => {
  const match = String(filename || "")
    .trim()
    .match(/\.([a-zA-Z0-9]+)$/);
  if (!match) return "FILE";
  const ext = match[1].toUpperCase();
  return ext.length <= 6 ? ext : "FILE";
};

type FeedbackState = "up" | "down" | null;
const INLINE_CITATION_PATTERN = /【参考资料(\d+)】|\[(\d+)\]/g;
const ASSISTANT_MARKDOWN_REMARK_PLUGINS = [remarkGfm, remarkBreaks];
const ASSISTANT_MARKDOWN_REHYPE_PLUGINS = [rehypeRaw];
const ASSISTANT_MARKDOWN_ALLOWED_ELEMENTS = [
  "a",
  "blockquote",
  "br",
  "code",
  "del",
  "em",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "hr",
  "img",
  "li",
  "ol",
  "p",
  "pre",
  "span",
  "strong",
  "table",
  "tbody",
  "td",
  "th",
  "thead",
  "tr",
  "ul",
];

const ASSISTANT_MARKDOWN_URL_TRANSFORM: UrlTransform = (url) => {
  const value = String(url || "").trim();
  if (/^citation:\/\/\d+\/?$/i.test(value)) return value;
  return defaultUrlTransform(value);
};

type CitationItem = NonNullable<ChatMessage["citations"]>[number];

const normalizeCitations = (
  citations: ChatMessage["citations"] | undefined,
): CitationItem[] => {
  if (!Array.isArray(citations) || citations.length === 0) return [];
  const seen = new Set<number>();
  const rows: CitationItem[] = [];
  for (const citation of citations) {
    const index = Number(citation.index);
    if (!Number.isFinite(index) || index <= 0 || seen.has(index)) continue;
    seen.add(index);
    rows.push(citation);
  }
  return rows;
};

const extractInlineCitationIndices = (content: string): number[] => {
  const raw = String(content || "");
  INLINE_CITATION_PATTERN.lastIndex = 0;
  const matches = Array.from(raw.matchAll(INLINE_CITATION_PATTERN));
  if (matches.length === 0) return [];

  const seen = new Set<number>();
  const rows: number[] = [];
  for (const match of matches) {
    const index = Number(match[1] ?? match[2] ?? NaN);
    if (!Number.isFinite(index) || index <= 0 || seen.has(index)) continue;
    seen.add(index);
    rows.push(index);
  }
  return rows;
};

const toMarkdownWithCitationLinks = (content: string): string => {
  const raw = String(content || "");
  INLINE_CITATION_PATTERN.lastIndex = 0;
  return raw.replace(INLINE_CITATION_PATTERN, (full, g1, g2) => {
    const index = Number(g1 ?? g2 ?? NaN);
    if (!Number.isFinite(index) || index <= 0) return full;
    return `[${index}](citation://${index})`;
  });
};

const sortByName = <T extends { name: string }>(rows: T[]): T[] =>
  [...rows].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, {
      numeric: true,
      sensitivity: "base",
    }),
  );

const formatPickerUploadedAt = (uploadedAt: string): string => {
  const value = new Date(uploadedAt);
  if (Number.isNaN(value.getTime())) return "Unknown";
  return value.toLocaleString();
};

const getFolderChain = (
  folderId: string | null,
  foldersById: Map<string, FolderItem>,
): FolderItem[] => {
  if (folderId == null) return [];
  const rows: FolderItem[] = [];
  let cursor: string | null = folderId;
  while (cursor) {
    const folder = foldersById.get(cursor);
    if (!folder) break;
    rows.unshift(folder);
    cursor = folder.parentId;
  }
  return rows;
};

const formatFolderFileCount = (count: number): string =>
  `${count} ${count === 1 ? "file" : "files"}`;

const clampScrollTop = (element: HTMLElement, top: number): number => {
  const maxTop = Math.max(0, element.scrollHeight - element.clientHeight);
  return Math.min(Math.max(0, top), maxTop);
};

const easeOutCubic = (t: number): number => 1 - (1 - t) ** 3;

export function ChatView({
  chatStarted,
  folders,
  documents,
  uploadingDocument,
  quickUploadDragOver,
  selectedDocs,
  selectedDocuments,
  messages,
  inputValue,
  inferenceEnabled,
  onSetQuickUploadDragOver,
  onUploadPickedFile,
  onOpenUploadFilePicker,
  onToggleDocumentSelectionWithGuard,
  onRemoveDocumentFromSelection,
  onInputValueChange,
  onComposerKeyDown,
  onToggleInference,
  onSend,
  onEditMessage,
  onRetryMessage,
}: ChatViewProps) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const autoStickToBottomRef = useRef(true);
  const documentPickerRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const copyResetTimerRef = useRef<number | null>(null);
  const citationHighlightTimerRef = useRef<number | null>(null);
  const scrollAnimationRef = useRef<WeakMap<HTMLElement, number>>(
    new WeakMap(),
  );
  const [documentPickerOpen, setDocumentPickerOpen] = useState(false);
  const [documentPickerFolderId, setDocumentPickerFolderId] = useState<
    string | null
  >(null);
  const [documentPickerSearch, setDocumentPickerSearch] = useState("");
  const [copiedMessageId, setCopiedMessageId] = useState<string | null>(null);
  const [feedbackByMessage, setFeedbackByMessage] = useState<
    Record<string, FeedbackState>
  >({});
  const [highlightedCitationKey, setHighlightedCitationKey] = useState<
    string | null
  >(null);

  const foldersById = useMemo(
    () => new Map(folders.map((folder) => [folder.id, folder])),
    [folders],
  );

  const folderDocumentCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const doc of documents) {
      if (!doc.folderId) continue;
      counts.set(doc.folderId, (counts.get(doc.folderId) ?? 0) + 1);
    }
    return counts;
  }, [documents]);

  const documentPickerBreadcrumbs = useMemo(
    () => getFolderChain(documentPickerFolderId, foldersById),
    [documentPickerFolderId, foldersById],
  );

  const documentPickerVisibleFolders = useMemo(
    () =>
      sortByName(
        folders.filter((folder) => folder.parentId === documentPickerFolderId),
      ),
    [folders, documentPickerFolderId],
  );

  const documentPickerVisibleDocuments = useMemo(
    () =>
      sortByName(
        documents.filter((doc) => doc.folderId === documentPickerFolderId),
      ),
    [documents, documentPickerFolderId],
  );

  const normalizedDocumentPickerSearch = documentPickerSearch
    .trim()
    .toLowerCase();

  const documentPickerSearchResults = useMemo(() => {
    if (!normalizedDocumentPickerSearch) return [];

    return sortByName(
      documents.filter((doc) => {
        const folderPath = getFolderChain(doc.folderId, foldersById)
          .map((folder) => folder.name)
          .join("/");
        const target =
          `${doc.name}\n${doc.description}\n${folderPath}`.toLowerCase();
        return target.includes(normalizedDocumentPickerSearch);
      }),
    ).map((doc) => {
      const path = getFolderChain(doc.folderId, foldersById)
        .map((folder) => folder.name)
        .join(" / ");
      return { doc, path: path || "Root" };
    });
  }, [documents, foldersById, normalizedDocumentPickerSearch]);

  useEffect(() => {
    if (documentPickerFolderId && !foldersById.has(documentPickerFolderId)) {
      setDocumentPickerFolderId(null);
    }
  }, [documentPickerFolderId, foldersById]);

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
    const forceStick = latestMessage?.role === "user" || isAssistantStreaming;

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
      if (citationHighlightTimerRef.current != null) {
        window.clearTimeout(citationHighlightTimerRef.current);
        citationHighlightTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!documentPickerOpen) return;
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (!target) return;
      if (
        documentPickerRef.current &&
        !documentPickerRef.current.contains(target)
      ) {
        setDocumentPickerOpen(false);
      }
    };
    window.addEventListener("pointerdown", handlePointerDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
    };
  }, [documentPickerOpen]);

  const focusComposer = () => {
    window.requestAnimationFrame(() => {
      const el = composerRef.current;
      if (!el) return;
      el.focus();
      const cursor = el.value.length;
      el.setSelectionRange(cursor, cursor);
    });
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
      setCopiedMessageId((previous) =>
        previous === messageId ? null : previous,
      );
      copyResetTimerRef.current = null;
    }, 1200);
  };

  const toggleFeedback = (
    messageId: string,
    target: Exclude<FeedbackState, null>,
  ) => {
    setFeedbackByMessage((previous) => {
      const current = previous[messageId] ?? null;
      return {
        ...previous,
        [messageId]: current === target ? null : target,
      };
    });
  };

  const smoothScrollToTop = (
    element: HTMLElement,
    targetTop: number,
    durationMs = 300,
  ): Promise<void> =>
    new Promise((resolve) => {
      const from = element.scrollTop;
      const to = clampScrollTop(element, targetTop);
      const delta = to - from;
      if (Math.abs(delta) < 1) {
        resolve();
        return;
      }

      const previousAnimation = scrollAnimationRef.current.get(element);
      if (previousAnimation != null) {
        window.cancelAnimationFrame(previousAnimation);
      }

      const start = performance.now();
      const step = (now: number) => {
        const elapsed = now - start;
        const progress = Math.min(1, elapsed / durationMs);
        const eased = easeOutCubic(progress);
        element.scrollTop = from + delta * eased;
        if (progress >= 1) {
          scrollAnimationRef.current.delete(element);
          resolve();
          return;
        }
        const rafId = window.requestAnimationFrame(step);
        scrollAnimationRef.current.set(element, rafId);
      };

      const rafId = window.requestAnimationFrame(step);
      scrollAnimationRef.current.set(element, rafId);
    });

  const jumpToCitation = (messageId: string, index: number) => {
    const fallbackKey = `${messageId}:${index}`;

    const resolveCitationElement = (): HTMLElement | null => {
      const byExactId = document.getElementById(
        `citation-${messageId}-${index}`,
      );
      if (byExactId) return byExactId;

      const citationLists = Array.from(
        document.querySelectorAll<HTMLElement>("[data-citation-message-id]"),
      );
      const citationList =
        citationLists.find(
          (list) => list.getAttribute("data-citation-message-id") === messageId,
        ) ?? null;
      if (!citationList) return null;

      const cards = Array.from(
        citationList.querySelectorAll<HTMLElement>("[data-citation-order]"),
      );
      if (cards.length === 0) return null;

      const byExactIndex = cards.find(
        (card) => Number(card.getAttribute("data-citation-index")) === index,
      );
      if (byExactIndex) return byExactIndex;

      const byOrder = cards[index - 1];
      if (byOrder) {
        return byOrder;
      }

      return cards[0] ?? null;
    };

    const el = resolveCitationElement();
    if (!el) return;
    const resolvedKey = el.getAttribute("data-citation-key") || fallbackKey;

    // Re-trigger highlight even if user clicks the same citation repeatedly.
    setHighlightedCitationKey((previous) =>
      previous === resolvedKey ? null : previous,
    );
    window.requestAnimationFrame(() => {
      setHighlightedCitationKey(resolvedKey);
    });

    const runScroll = async () => {
      const citationList = el.closest(
        "[data-citation-scroll='true']",
      ) as HTMLElement | null;
      if (citationList) {
        const listRect = citationList.getBoundingClientRect();
        const targetRect = el.getBoundingClientRect();
        const currentTop = citationList.scrollTop;
        const targetTop =
          currentTop +
          (targetRect.top - listRect.top) -
          citationList.clientHeight * 0.22;
        await smoothScrollToTop(citationList, targetTop, 340);
      }

      const chatScrollContainer = scrollContainerRef.current;
      if (!chatScrollContainer) return;

      const chatRect = chatScrollContainer.getBoundingClientRect();
      const targetRect = el.getBoundingClientRect();
      const topPadding = 94;
      const bottomPadding = 72;
      const isOutOfView =
        targetRect.top < chatRect.top + topPadding ||
        targetRect.bottom > chatRect.bottom - bottomPadding;

      if (!isOutOfView) return;

      const delta = targetRect.top - chatRect.top - topPadding;
      const nextTop = clampScrollTop(
        chatScrollContainer,
        chatScrollContainer.scrollTop + delta,
      );
      await smoothScrollToTop(chatScrollContainer, nextTop, 340);
    };

    void runScroll();

    if (citationHighlightTimerRef.current != null) {
      window.clearTimeout(citationHighlightTimerRef.current);
    }
    citationHighlightTimerRef.current = window.setTimeout(() => {
      setHighlightedCitationKey((previous) =>
        previous === resolvedKey ? null : previous,
      );
      citationHighlightTimerRef.current = null;
    }, 2200);
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
                <p className="text-xl font-semibold text-white">
                  Welcome to PageIndex
                </p>
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
                  {uploadingDocument
                    ? "Uploading..."
                    : "Upload a document to get started"}
                </span>
              </button>
            </div>
          ) : (
            <div className="flex min-h-full flex-col items-center gap-5 pt-1">
              <div className="mb-1 flex flex-col items-center gap-3">
                <MessageSquare size={38} className="text-[#4A90D9]" />
                <p className="text-2xl font-semibold text-white">
                  Select documents to start
                </p>
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
                        {getDocumentDescriptionLabel(doc.description, doc.name)}
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
              const citationRows = normalizeCitations(message.citations);
              const citationIndexSet = new Set(
                citationRows.map((item) => Number(item.index)),
              );
              const assistantHasInlineCitation =
                message.role === "assistant" &&
                extractInlineCitationIndices(message.content).length > 0;
              const inlineReferenceRows = citationRows.slice(0, 8);
              const inlineReferenceOverflow =
                citationRows.length - inlineReferenceRows.length;
              const bubbleContent =
                trimmedContent.length > 0 ? (
                  message.content
                ) : message.role === "assistant" ? (
                  message.debug ? null : (
                    <span className="text-[#99a9bc]">Thinking...</span>
                  )
                ) : (
                  message.content
                );
              const renderedBubbleContent = (() => {
                if (
                  message.role !== "assistant" ||
                  trimmedContent.length === 0
                ) {
                  return bubbleContent;
                }
                const markdownContent = toMarkdownWithCitationLinks(
                  message.content,
                );
                return (
                  <div className="assistant-markdown">
                    <ReactMarkdown
                      remarkPlugins={ASSISTANT_MARKDOWN_REMARK_PLUGINS}
                      rehypePlugins={ASSISTANT_MARKDOWN_REHYPE_PLUGINS}
                      allowedElements={ASSISTANT_MARKDOWN_ALLOWED_ELEMENTS}
                      urlTransform={ASSISTANT_MARKDOWN_URL_TRANSFORM}
                      unwrapDisallowed
                      components={{
                        a: ({ href, children }) => {
                          const rawHref = String(href || "");
                          if (rawHref.startsWith("citation://")) {
                            const index = Number.parseInt(
                              rawHref.slice("citation://".length),
                              10,
                            );
                            if (!Number.isFinite(index) || index <= 0) {
                              return (
                                <span className="inline-citation is-missing">
                                  {children}
                                </span>
                              );
                            }
                            const exists =
                              citationIndexSet.has(index) ||
                              (index >= 1 && index <= citationRows.length);
                            return (
                              <button
                                type="button"
                                className={`inline-citation ${exists ? "" : "is-missing"}`}
                                onClick={() => {
                                  jumpToCitation(message.id, index);
                                }}
                                aria-label={`Jump to source ${index}`}
                              >
                                [{index}]
                              </button>
                            );
                          }
                          if (!rawHref) {
                            return (
                              <span className="assistant-md-link">
                                {children}
                              </span>
                            );
                          }
                          return (
                            <a href={rawHref} className="assistant-md-link">
                              {children}
                            </a>
                          );
                        },
                        img: ({ src, alt }) => {
                          const rawSrc = String(src || "").trim();
                          if (!rawSrc) {
                            return (
                              <span className="assistant-md-image-fallback">
                                [{alt || "image"}]
                              </span>
                            );
                          }
                          return (
                            <img
                              src={rawSrc}
                              alt={alt || ""}
                              loading="lazy"
                              className="assistant-md-image"
                            />
                          );
                        },
                      }}
                    >
                      {markdownContent}
                    </ReactMarkdown>
                  </div>
                );
              })();
              const showTextBubble = renderedBubbleContent !== null;
              const assistantResponseReady =
                message.role === "assistant" &&
                trimmedContent.length > 0 &&
                (message.debug == null || message.debug.completedAt != null);
              const showMessageActions =
                message.role === "assistant" ? assistantResponseReady : true;

              return (
                <div
                  key={message.id}
                  className={`group flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`flex flex-col gap-2 ${
                      message.role === "user"
                        ? "max-w-[72%] items-end"
                        : "w-full max-w-full items-start"
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
                        <div className="flex flex-wrap justify-end gap-2">
                          {message.attachments.map((attachment) => (
                            <div
                              key={`${message.id}-${attachment.docId}`}
                              className="user-attachment-card group/attachment relative w-[92px] overflow-hidden rounded-2xl border sm:w-[102px]"
                              title={attachment.name}
                            >
                              <DocumentThumbnail
                                docId={attachment.docId}
                                docName={attachment.name}
                                width={420}
                                className="h-[128px] w-full sm:h-[142px]"
                                iconSize={30}
                              />
                              <span className="user-attachment-type absolute bottom-2 left-2 rounded-lg px-2 py-0.5 text-[10px] font-semibold tracking-wide">
                                {attachmentTypeLabel(attachment.name)}
                              </span>
                            </div>
                          ))}
                        </div>
                      )}

                    {showTextBubble && (
                      <div
                        className={`break-words rounded-2xl px-4 py-3 text-sm leading-6 ${
                          message.role === "user"
                            ? "whitespace-pre-wrap bg-[#2a2a2a] text-white"
                            : "whitespace-normal bg-[#1a1a1a] text-[#d9dce3]"
                        }`}
                      >
                        {renderedBubbleContent}
                      </div>
                    )}

                    {message.role === "assistant" &&
                      citationRows.length > 0 &&
                      !assistantHasInlineCitation && (
                        <div className="assistant-inline-reference-strip w-full">
                          <p className="assistant-inline-reference-label">
                            References
                          </p>
                          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                            {inlineReferenceRows.map((citation) => (
                              <button
                                key={`${message.id}-inline-ref-${citation.index}`}
                                type="button"
                                className="inline-citation-chip"
                                onClick={() => {
                                  jumpToCitation(message.id, citation.index);
                                }}
                                aria-label={`Jump to source ${citation.index}`}
                              >
                                <span className="inline-citation-chip-index">
                                  [{citation.index}]
                                </span>
                                <span
                                  className="inline-citation-chip-name"
                                  title={
                                    citation.source_name ||
                                    citation.doc_id ||
                                    "unknown"
                                  }
                                >
                                  {citation.source_name ||
                                    citation.doc_id ||
                                    "unknown"}
                                </span>
                              </button>
                            ))}
                            {inlineReferenceOverflow > 0 ? (
                              <span className="inline-citation-chip-more">
                                +{inlineReferenceOverflow}
                              </span>
                            ) : null}
                          </div>
                        </div>
                      )}

                    {message.role === "assistant" &&
                      citationRows.length > 0 && (
                        <div className="w-full rounded-xl border border-[#2f3a4e] bg-[#111827] p-3">
                          <div className="mb-2.5 flex items-center justify-between">
                            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-[#93a4be]">
                              Sources
                            </p>
                            <span className="rounded-md border border-[#2e3b51] bg-[#0f1726] px-2 py-0.5 text-[11px] font-semibold text-[#b6c7df]">
                              {citationRows.length}
                            </span>
                          </div>
                          <div
                            className="max-h-[420px] space-y-2.5 overflow-y-auto pr-1"
                            data-citation-scroll="true"
                            data-citation-message-id={message.id}
                          >
                            {citationRows.map((c, idx) =>
                              (() => {
                                const citationKey = `${message.id}:${c.index}`;
                                const focused =
                                  highlightedCitationKey === citationKey;
                                return (
                                  <article
                                    id={`citation-${message.id}-${c.index}`}
                                    data-citation-index={String(c.index)}
                                    data-citation-order={String(idx)}
                                    data-citation-key={citationKey}
                                    key={`${message.id}-cite-${idx}`}
                                    className={`citation-source-card group rounded-lg border border-[#2e3b51] bg-[#0f1726] p-2.5 transition-colors hover:bg-[#111b2d] ${
                                      focused ? "is-focused" : ""
                                    }`}
                                  >
                                    <div className="flex items-start gap-2">
                                      <span className="mt-0.5 inline-flex rounded bg-[#25344f] px-1.5 py-0.5 text-xs font-semibold text-[#b9d3ff]">
                                        [{c.index}]
                                      </span>
                                      <div className="min-w-0 flex-1">
                                        <p
                                          className="truncate text-[13px] font-semibold text-white"
                                          title={
                                            c.source_name ||
                                            c.doc_id ||
                                            "unknown"
                                          }
                                        >
                                          {c.source_name ||
                                            c.doc_id ||
                                            "unknown"}
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
                                        source:{" "}
                                        {normalizeSourceType(c.source_type)}
                                      </span>
                                      <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                        chunk:{" "}
                                        {c.chunk_index >= 0
                                          ? c.chunk_index
                                          : "-"}
                                      </span>
                                      <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                        rerank: {formatScore(c.rerank_score)}
                                      </span>
                                      <span className="rounded-md border border-[#2f3a4f] bg-[#121b2d] px-1.5 py-0.5">
                                        rrf: {formatScore(c.rrf_score)}
                                      </span>
                                    </div>
                                  </article>
                                );
                              })(),
                            )}
                          </div>
                        </div>
                      )}

                    {showMessageActions && (
                      <div
                        className={`message-actions mt-1 flex items-center gap-1.5 opacity-0 transition-opacity duration-150 pointer-events-none group-hover:pointer-events-auto group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:opacity-100 ${
                          message.role === "user"
                            ? "justify-end"
                            : "justify-start"
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
                            {copiedMessageId === message.id ? (
                              <Check size={15} />
                            ) : (
                              <Copy size={15} />
                            )}
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
                                onEditMessage(messageIndex);
                                focusComposer();
                              }}
                            >
                              <Pencil size={15} />
                            </button>
                          )}

                          <button
                            type="button"
                            aria-label="Retry question"
                            className="message-action-btn h-8 w-8 rounded-lg bg-transparent text-[#b4bfce] hover:text-white"
                            onClick={() => {
                              void onRetryMessage(messageIndex);
                            }}
                          >
                            <RotateCcw size={15} />
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-4 flex justify-center">
        <div className="chat-composer-panel w-full max-w-[980px] rounded-2xl border border-[#2c3341] bg-[#171c27] p-3 shadow-2xl">
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
                    <p className="line-clamp-2 text-xs font-semibold text-white">
                      {doc.name}
                    </p>
                    <p className="mt-0.5 text-xs text-[#a5acb8]">
                      {doc.pages} pages
                    </p>
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
                    <div className="chat-doc-picker-breadcrumb mb-2 flex items-center justify-between gap-2">
                      <div className="flex min-w-0 items-center gap-1.5 text-xs text-[#9aa5b7]">
                        <button
                          type="button"
                          className={`chat-doc-picker-crumb inline-flex items-center gap-1 rounded-md px-1.5 py-1 transition-colors ${
                            documentPickerFolderId == null
                              ? "text-white is-current"
                              : "hover:bg-[#1a2333] hover:text-white"
                          }`}
                          onClick={() => {
                            setDocumentPickerFolderId(null);
                          }}
                        >
                          <House size={14} />
                          <span className="font-semibold">Root</span>
                        </button>
                        {documentPickerBreadcrumbs.map((folder) => (
                          <div
                            key={folder.id}
                            className="flex min-w-0 items-center gap-1"
                          >
                            <ChevronRight
                              size={14}
                              className="chat-doc-picker-crumb-sep text-[#6d7788]"
                            />
                            <button
                              type="button"
                              className={`chat-doc-picker-crumb truncate rounded-md px-1.5 py-1 transition-colors ${
                                folder.id === documentPickerFolderId
                                  ? "text-white is-current"
                                  : "hover:bg-[#1a2333] hover:text-white"
                              }`}
                              onClick={() => {
                                setDocumentPickerFolderId(folder.id);
                              }}
                              title={folder.name}
                            >
                              {folder.name}
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="chat-doc-picker-browser overflow-hidden rounded-lg border border-[#303642] bg-[#111622]">
                      <div className="chat-doc-picker-list max-h-[300px] space-y-1 overflow-y-auto p-2">
                        {normalizedDocumentPickerSearch ? (
                          documentPickerSearchResults.length === 0 ? (
                            <p className="chat-doc-picker-empty rounded-md border border-dashed border-[#303642] px-3 py-4 text-center text-xs text-[#8f95a3]">
                              No matching documents.
                            </p>
                          ) : (
                            documentPickerSearchResults.map(({ doc, path }) => {
                              const selected = selectedDocs.includes(doc.id);
                              return (
                                <button
                                  key={doc.id}
                                  type="button"
                                  onClick={() =>
                                    onToggleDocumentSelectionWithGuard(doc)
                                  }
                                  className={`chat-doc-picker-doc w-full rounded-lg border px-2.5 py-2 text-left transition-colors ${
                                    selected
                                      ? "is-selected border-[#4A90D9] bg-[#182b45]"
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
                                        <FileText
                                          size={14}
                                          className="text-[#9ca7ba]"
                                        />
                                        <p className="truncate text-xs font-semibold text-white">
                                          {doc.name}
                                        </p>
                                        <DocumentStatusBadge
                                          status={doc.status}
                                        />
                                      </div>
                                      <p className="mt-0.5 truncate text-[11px] text-[#8f9ab0]">
                                        {path}
                                      </p>
                                      <p className="mt-0.5 text-[11px] text-[#a5acb8]">
                                        {doc.pages} pages •{" "}
                                        {formatPickerUploadedAt(doc.uploadedAt)}
                                      </p>
                                    </div>
                                  </div>
                                </button>
                              );
                            })
                          )
                        ) : (
                          <>
                            {documentPickerVisibleFolders.map((folder) => (
                              <button
                                key={folder.id}
                                type="button"
                                className="chat-doc-picker-folder flex w-full items-center gap-2 rounded-lg border border-[#2b3343] bg-[#141b28] px-2.5 py-2 text-left transition-colors hover:bg-[#1a2333]"
                                onClick={() => {
                                  setDocumentPickerFolderId(folder.id);
                                }}
                              >
                                <Folder size={16} className="text-[#a7b2c5]" />
                                <div className="min-w-0 flex-1">
                                  <p className="truncate text-xs font-semibold text-white">
                                    {folder.name}
                                  </p>
                                  <p className="text-[11px] text-[#98a3b8]">
                                    {formatFolderFileCount(
                                      folderDocumentCounts.get(folder.id) ?? 0,
                                    )}
                                  </p>
                                </div>
                                <ChevronRight
                                  size={16}
                                  className="text-[#7f8ba1]"
                                />
                              </button>
                            ))}

                            {documentPickerVisibleDocuments.map((doc) => {
                              const selected = selectedDocs.includes(doc.id);
                              return (
                                <button
                                  key={doc.id}
                                  type="button"
                                  onClick={() =>
                                    onToggleDocumentSelectionWithGuard(doc)
                                  }
                                  className={`chat-doc-picker-doc w-full rounded-lg border px-2.5 py-2 text-left transition-colors ${
                                    selected
                                      ? "is-selected border-[#4A90D9] bg-[#182b45]"
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
                                        <FileText
                                          size={14}
                                          className="text-[#9ca7ba]"
                                        />
                                        <p className="truncate text-xs font-semibold text-white">
                                          {doc.name}
                                        </p>
                                        <DocumentStatusBadge
                                          status={doc.status}
                                        />
                                      </div>
                                      <p className="mt-0.5 text-[11px] text-[#a5acb8]">
                                        {doc.pages} pages •{" "}
                                        {formatPickerUploadedAt(doc.uploadedAt)}
                                      </p>
                                    </div>
                                  </div>
                                </button>
                              );
                            })}

                            {documentPickerVisibleFolders.length === 0 &&
                            documentPickerVisibleDocuments.length === 0 ? (
                              <p className="chat-doc-picker-empty rounded-md border border-dashed border-[#303642] px-3 py-4 text-center text-xs text-[#8f95a3]">
                                This folder is empty.
                              </p>
                            ) : null}
                          </>
                        )}
                      </div>

                      <div className="chat-doc-picker-footer flex items-center gap-2 border-t border-[#2a3343] p-2.5">
                        <div className="chat-doc-picker-search flex min-w-0 flex-1 items-center gap-2 rounded-md border border-[#303642] bg-[#101520] px-2.5 py-1.5">
                          <Search size={15} className="text-[#8f95a3]" />
                          <input
                            value={documentPickerSearch}
                            onChange={(event) =>
                              setDocumentPickerSearch(event.target.value)
                            }
                            placeholder="Search documents..."
                            className="chat-doc-picker-search-input w-full bg-transparent text-xs text-white outline-none placeholder:text-[#8f95a3]"
                          />
                        </div>
                        <button
                          type="button"
                          onClick={() => {
                            setDocumentPickerOpen(false);
                            onOpenUploadFilePicker();
                          }}
                          className="chat-doc-picker-upload inline-flex items-center gap-1.5 rounded-md border border-[#303642] bg-[#131925] px-2.5 py-1.5 text-xs font-semibold text-[#aeb4c0] transition-colors hover:text-white"
                        >
                          <Upload size={14} />
                          <span>Upload</span>
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <HintIconButton
                label="Inference"
                hint="Inference"
                wrapperClassName="composer-hint"
                className={`composer-websearch-btn rounded-full p-2 ${
                  inferenceEnabled ? "is-active" : ""
                }`}
                pressed={inferenceEnabled}
                onClick={onToggleInference}
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
        RAG can make mistakes, please check the response.
      </p>
    </section>
  );
}

export const buildProcessingToastMessage = (doc: DocumentItem): string =>
  `"${doc.name}" is ${getDocumentStatusLabel(doc.status).toLowerCase()}. Please wait.`;
