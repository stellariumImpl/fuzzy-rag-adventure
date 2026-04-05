import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { requestAssistantReply } from "../../services/assistant-transport";
import {
  createChatThread,
  deleteChatThread,
  generateChatTitle,
  getChatThread,
  listChatThreads,
  updateChatThread,
  type ChatMessageRecord,
} from "../../services/workspace-api";
import {
  mapChatThreadFromApi,
  type ChatMessage,
  type ChatThreadItem,
  type DocumentItem,
} from "../workspace/models";
import {
  getDocumentStatusLabel,
  isDocumentProcessing,
  isDocumentReady,
} from "../workspace/utils";

interface UseChatSessionParams {
  documents: DocumentItem[];
  documentsById: Map<string, DocumentItem>;
  pushToast: (
    title: string,
    options?: {
      tone?: "success" | "error" | "info";
      description?: string;
      durationMs?: number;
    },
  ) => void;
}

const CHAT_DEBUG_TO_CONSOLE =
  String(import.meta.env.VITE_CHAT_DEBUG_CONSOLE ?? "true").toLowerCase() !==
  "false";

const ACTIVE_THREAD_STORAGE_KEY = "basic-ui-active-thread-id";

const normalizeThreadId = (value: unknown): string | null => {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
};

const isUntitledThreadName = (title: string | null | undefined): boolean => {
  const normalized = (title || "").trim().toLowerCase();
  return normalized === "" || normalized === "new chat" || normalized === "newchat" || normalized === "new";
};

const toChatMessage = (record: ChatMessageRecord): ChatMessage => ({
  id: record.message_id,
  role: record.role,
  content: record.content,
  attachments: Array.isArray(record.metadata?.selected_docs)
    ? record.metadata?.selected_docs
        .map((doc) => ({
          docId: String(doc.doc_id || ""),
          name: String(doc.source_name || doc.doc_id || ""),
          pages: Number.isFinite(doc.pages) ? Number(doc.pages) : undefined,
        }))
        .filter((doc) => doc.docId.length > 0)
    : undefined,
  debug: record.debug ?? null,
  citations: Array.isArray(record.metadata?.answer_meta?.citations)
    ? record.metadata?.answer_meta?.citations
    : undefined,
});

const logAssistantDebugEvent = (
  event: {
    stage: string;
    detail: string;
    data?: Record<string, unknown>;
    at?: number;
  },
) => {
  if (!CHAT_DEBUG_TO_CONSOLE) return;

  const at = event.at ? new Date(event.at).toLocaleTimeString() : "";
  if (
    event.stage === "multimodal-recursive-rag" &&
    event.detail === "Retriever returned nodes."
  ) {
    const rawHits = event.data?.hits;
    const hits = Array.isArray(rawHits) ? rawHits : [];
    const tableRows = hits.map((item, index) => {
      const row =
        item && typeof item === "object"
          ? (item as Record<string, unknown>)
          : {};
      return {
        rank: index + 1,
        score: row.score ?? "",
        source: row.source_name ?? "",
        page: row.page_start ?? "",
        chunk_level: row.chunk_level ?? "",
        table: row.is_table_object === true,
        preview: String(row.text_preview ?? ""),
      };
    });
    console.groupCollapsed(
      `[assistant-debug] ${at} ${event.stage} - ${event.detail}`,
    );
    console.log("raw event:", event);
    if (tableRows.length > 0) {
      console.table(tableRows);
    }
    console.groupEnd();
    return;
  }

  console.debug(`[assistant-debug] ${at} ${event.stage} - ${event.detail}`, event);
};

export function useChatSession({ documents, documentsById, pushToast }: UseChatSessionParams) {
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [chatStarted, setChatStarted] = useState(false);
  const [webSearchEnabled, setWebSearchEnabled] = useState(false);

  const [chatThreads, setChatThreads] = useState<ChatThreadItem[]>([]);
  const [chatListLoading, setChatListLoading] = useState(false);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [creatingNewChat, setCreatingNewChat] = useState(false);
  const creatingNewChatRef = useRef(false);
  const autoTitleInFlightRef = useRef<Set<string>>(new Set());

  const selectedDocuments = useMemo(
    () =>
      selectedDocs
        .map((id) => documentsById.get(id))
        .filter((doc): doc is DocumentItem => doc != null),
    [selectedDocs, documentsById],
  );

  useEffect(() => {
    // 文档列表变化时，清理掉已经失效的已选文档 ID，避免会话里残留无效引用。
    setSelectedDocs((previous) => {
      const validIds = new Set(documents.map((doc) => doc.id));
      return previous.filter((id) => validIds.has(id));
    });
  }, [documents]);

  useEffect(() => {
    if (messages.length > 0) {
      setChatStarted(true);
    }
  }, [messages.length]);

  const persistActiveThread = useCallback((threadId: string | null) => {
    if (threadId) {
      window.localStorage.setItem(ACTIVE_THREAD_STORAGE_KEY, threadId);
      return;
    }
    window.localStorage.removeItem(ACTIVE_THREAD_STORAGE_KEY);
  }, []);

  const refreshChatThreads = useCallback(async (): Promise<ChatThreadItem[]> => {
    const rows = await listChatThreads();
    const items = rows.map(mapChatThreadFromApi);
    setChatThreads(items);
    return items;
  }, []);

  const openChatThread = useCallback(
    async (threadId: string, options?: { skipRefresh?: boolean }) => {
      // 统一会话切换入口：拉取线程详情并恢复消息、文档选择。
      const detail = await getChatThread(threadId);
      setActiveThreadId(detail.thread_id);
      persistActiveThread(detail.thread_id);
      setMessages(detail.messages.map(toChatMessage));
      // 文件选择是“按消息生效”，切换/恢复会话时不自动把历史选择带回输入框。
      setSelectedDocs([]);
      setInputValue("");
      setChatStarted(detail.messages.length > 0);

      if (!options?.skipRefresh) {
        await refreshChatThreads();
      }
    },
    [persistActiveThread, refreshChatThreads],
  );

  const handleCreateNewChat = useCallback(async () => {
    // 防止连续点击创建多个空会话：
    // 1) 若当前就是空会话，仅重置输入框；
    // 2) 若已有空会话，直接切换过去；
    // 3) 否则才创建新会话。
    if (creatingNewChatRef.current) return;

    const resetComposer = () => {
      setMessages([]);
      setInputValue("");
      setSelectedDocs([]);
      setChatStarted(false);
    };

    const activeSummary = activeThreadId
      ? chatThreads.find((item) => item.id === activeThreadId)
      : undefined;
    if (activeSummary && activeSummary.messageCount === 0) {
      resetComposer();
      return;
    }

    const existingEmpty = chatThreads.find((item) => item.messageCount === 0);
    if (existingEmpty) {
      try {
        await openChatThread(existingEmpty.id);
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown error";
        pushToast("Failed to open chat", {
          tone: "error",
          description: reason,
          durationMs: 2800,
        });
      }
      return;
    }

    try {
      creatingNewChatRef.current = true;
      setCreatingNewChat(true);
      const created = await createChatThread({ selected_doc_ids: [] });
      const mapped = mapChatThreadFromApi(created);

      setChatThreads((previous) => [mapped, ...previous.filter((item) => item.id !== mapped.id)]);
      setActiveThreadId(mapped.id);
      persistActiveThread(mapped.id);

      resetComposer();
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      pushToast("Failed to create chat", {
        tone: "error",
        description: reason,
        durationMs: 2800,
      });
    } finally {
      creatingNewChatRef.current = false;
      setCreatingNewChat(false);
    }
  }, [
    activeThreadId,
    chatThreads,
    openChatThread,
    persistActiveThread,
    pushToast,
  ]);

  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      setChatListLoading(true);
      try {
        const items = await refreshChatThreads();
        if (cancelled) return;

        if (items.length === 0) {
          setMessages([]);
          setSelectedDocs([]);
          setActiveThreadId(null);
          persistActiveThread(null);
          setChatStarted(false);
          return;
        }

        const remembered = normalizeThreadId(
          window.localStorage.getItem(ACTIVE_THREAD_STORAGE_KEY),
        );
        const fallback = items[0]?.id ?? null;
        const targetId =
          (remembered && items.some((item) => item.id === remembered) ? remembered : null) ??
          fallback;

        if (!targetId) return;
        await openChatThread(targetId, { skipRefresh: true });
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown error";
        if (!cancelled) {
          pushToast("Failed to load chat history", {
            tone: "error",
            description: reason,
            durationMs: 3200,
          });
        }
      } finally {
        if (!cancelled) {
          setChatListLoading(false);
        }
      }
    };

    void bootstrap();

    return () => {
      cancelled = true;
    };
  }, [openChatThread, persistActiveThread, pushToast, refreshChatThreads]);

  const toggleDocumentSelection = useCallback((id: string) => {
    setSelectedDocs((prev) =>
      prev.includes(id) ? prev.filter((docId) => docId !== id) : [...prev, id],
    );
  }, []);

  const removeDocumentFromSelection = useCallback((id: string) => {
    setSelectedDocs((prev) => prev.filter((docId) => docId !== id));
  }, []);

  const handleSelectChatThread = useCallback(
    async (threadId: string) => {
      if (threadId === activeThreadId) return;
      try {
        await openChatThread(threadId);
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown error";
        pushToast("Failed to open chat", {
          tone: "error",
          description: reason,
          durationMs: 2800,
        });
      }
    },
    [activeThreadId, openChatThread, pushToast],
  );

  const handleDeleteChatThread = useCallback(
    async (threadId: string) => {
      const wasActive = activeThreadId === threadId;
      try {
        await deleteChatThread(threadId);
        const remaining = await refreshChatThreads();

        if (!wasActive) {
          return;
        }

        const nextThreadId = remaining[0]?.id ?? null;
        if (nextThreadId) {
          await openChatThread(nextThreadId, { skipRefresh: true });
          return;
        }

        // 删除最后一个会话后重置聊天面板到空态。
        setActiveThreadId(null);
        persistActiveThread(null);
        setMessages([]);
        setSelectedDocs([]);
        setInputValue("");
        setChatStarted(false);
      } catch (error) {
        const reason = error instanceof Error ? error.message : "Unknown error";
        pushToast("Failed to delete chat", {
          tone: "error",
          description: reason,
          durationMs: 2800,
        });
      }
    },
    [
      activeThreadId,
      openChatThread,
      persistActiveThread,
      pushToast,
      refreshChatThreads,
    ],
  );

  const handleSend = useCallback(async () => {
    const userInput = inputValue.trim();
    if (!userInput) return;

    const processingSelected = selectedDocuments.filter((doc) =>
      isDocumentProcessing(doc.status),
    );
    if (processingSelected.length > 0) {
      pushToast("Documents are still processing", {
        tone: "info",
        durationMs: 2800,
        description: "Please wait until parsing is complete before sending a question.",
      });
      return;
    }

    const unavailableSelected = selectedDocuments.filter((doc) => !isDocumentReady(doc.status));
    if (unavailableSelected.length > 0) {
      const first = unavailableSelected[0];
      pushToast("Selected document is not searchable", {
        tone: "info",
        durationMs: 3200,
        description: `"${first.name}" is ${getDocumentStatusLabel(first.status).toLowerCase()}. Please select a ready document.`,
      });
      return;
    }

    if (selectedDocuments.length === 0) {
      pushToast("Select documents first", {
        tone: "info",
        durationMs: 3200,
        description:
          "This workspace is in strict retrieval mode. Please select at least one ready document before asking.",
      });
      return;
    }

    const selectedDocSnapshots = selectedDocuments.map((doc) => ({
      docId: doc.id,
      name: doc.name,
      pages: doc.pages,
    }));
    const activeThreadSummary = activeThreadId
      ? chatThreads.find((item) => item.id === activeThreadId)
      : undefined;
    const shouldAutoTitle = !activeThreadSummary || activeThreadSummary.messageCount === 0;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: userInput,
      attachments: selectedDocSnapshots.length > 0 ? selectedDocSnapshots : undefined,
    };
    const pendingAssistantId = `assistant-pending-${Date.now()}`;
    const requestStartedAt = Date.now();
    const pendingAssistantMessage: ChatMessage = {
      id: pendingAssistantId,
      role: "assistant",
      content: "",
      debug: {
        startedAt: requestStartedAt,
        events: [],
      },
    };

    setMessages((prev) => [...prev, userMessage, pendingAssistantMessage]);
    setInputValue("");
    // 文件选择按消息生效：发送后立即清空，避免下一问默认继承。
    setSelectedDocs([]);
    setChatStarted(true);

    try {
      const reply = await requestAssistantReply(
        userInput,
        [...messages, userMessage],
        {
          threadId: activeThreadId,
          selectedDocIds: selectedDocuments.map((doc) => doc.id),
        },
        (streamEvent) => {
          if (streamEvent.type === "debug") {
            logAssistantDebugEvent(streamEvent.event);
            setMessages((prev) =>
              prev.map((item) => {
                if (item.id !== pendingAssistantId) return item;
                const prevDebug = item.debug ?? {};
                const prevEvents = prevDebug.events ?? [];
                const nextEvent = {
                  ...streamEvent.event,
                  at: Date.now(),
                };
                return {
                  ...item,
                  debug: {
                    ...prevDebug,
                    startedAt: prevDebug.startedAt ?? requestStartedAt,
                    events: [...prevEvents, nextEvent],
                  },
                };
              }),
            );
            return;
          }

          if (streamEvent.type === "final" && streamEvent.message) {
            const streamThreadId = normalizeThreadId(streamEvent.threadId);
            if (streamThreadId) {
              setActiveThreadId(streamThreadId);
              persistActiveThread(streamThreadId);
            }

            if (CHAT_DEBUG_TO_CONSOLE) {
              const text = streamEvent.message.content ?? "";
              console.debug("[assistant-debug] final answer:", text);
            }

            const incomingMessage = streamEvent.message;
            setMessages((prev) =>
              prev.map((item) =>
                item.id === pendingAssistantId
                  ? (() => {
                      const localDebug = item.debug ?? {};
                      const incomingDebug = incomingMessage.debug ?? {};
                      const localEvents = localDebug.events ?? [];
                      const incomingEvents = incomingDebug.events ?? [];
                      return {
                        ...item,
                        ...incomingMessage,
                        id: pendingAssistantId,
                        debug: {
                          ...incomingDebug,
                          ...localDebug,
                          startedAt: localDebug.startedAt ?? requestStartedAt,
                          completedAt: Date.now(),
                          events: localEvents.length > 0 ? localEvents : incomingEvents,
                        },
                      };
                    })()
                  : item,
              ),
            );
          }
        },
      );

      const resolvedThreadId = normalizeThreadId(reply.threadId) ?? activeThreadId;
      if (resolvedThreadId) {
        setActiveThreadId(resolvedThreadId);
        persistActiveThread(resolvedThreadId);
      }

      const assistantMessage = reply.message;
      if (assistantMessage) {
        setMessages((prev) =>
          prev.map((item) =>
            item.id === pendingAssistantId
              ? (() => {
                  const localDebug = item.debug ?? {};
                  const incomingDebug = assistantMessage.debug ?? {};
                  const localEvents = localDebug.events ?? [];
                  const incomingEvents = incomingDebug.events ?? [];
                  return {
                    ...item,
                    ...assistantMessage,
                    id: pendingAssistantId,
                    debug: {
                      ...incomingDebug,
                      ...localDebug,
                      startedAt: localDebug.startedAt ?? requestStartedAt,
                      completedAt: Date.now(),
                      events: localEvents.length > 0 ? localEvents : incomingEvents,
                    },
                  };
                })()
              : item,
          ),
        );
      } else {
        setMessages((prev) =>
          prev.map((item) =>
            item.id === pendingAssistantId
              ? {
                  ...item,
                  content: "Assistant did not return a final response.",
                  debug: {
                    ...(item.debug ?? {}),
                    startedAt: item.debug?.startedAt ?? requestStartedAt,
                    completedAt: Date.now(),
                  },
                }
              : item,
          ),
        );
      }

      const resolvedForTitle = normalizeThreadId(reply.threadId) ?? activeThreadId;
      const answerForTitle = (reply.message?.content || "").trim();
      if (shouldAutoTitle && resolvedForTitle && answerForTitle) {
        const knownThread = chatThreads.find((item) => item.id === resolvedForTitle);
        const currentTitle = knownThread?.title ?? "New chat";
        if (
          isUntitledThreadName(currentTitle) &&
          !autoTitleInFlightRef.current.has(resolvedForTitle)
        ) {
          autoTitleInFlightRef.current.add(resolvedForTitle);
          try {
            const generatedTitle = await generateChatTitle({
              question: userInput,
              answer: answerForTitle,
              current_title: currentTitle,
            });
            const finalTitle = generatedTitle.trim();
            if (finalTitle && !isUntitledThreadName(finalTitle)) {
              await updateChatThread(resolvedForTitle, { title: finalTitle });
              setChatThreads((previous) =>
                previous.map((item) =>
                  item.id === resolvedForTitle
                    ? {
                        ...item,
                        title: finalTitle,
                      }
                    : item,
                ),
              );
            }
          } catch {
            // ignore title-generation failures, do not break answer flow
          } finally {
            autoTitleInFlightRef.current.delete(resolvedForTitle);
          }
        }
      }

      // 每轮发送完成后刷新线程列表，保证侧边栏的顺序和预览实时更新。
      await refreshChatThreads();
    } catch (error) {
      const reason = error instanceof Error ? error.message : "Unknown error";
      setMessages((prev) =>
        prev.map((item) =>
          item.id === pendingAssistantId
            ? {
                ...item,
                content: `Assistant backend unavailable: ${reason}`,
                debug: {
                  ...(item.debug ?? {}),
                  startedAt: item.debug?.startedAt ?? requestStartedAt,
                  completedAt: Date.now(),
                },
              }
            : item,
        ),
      );
    }
  }, [
    activeThreadId,
    chatThreads,
    generateChatTitle,
    inputValue,
    messages,
    persistActiveThread,
    pushToast,
    refreshChatThreads,
    selectedDocuments,
    updateChatThread,
  ]);

  const handleKeyDown = useCallback(
    (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        void handleSend();
      }
    },
    [handleSend],
  );

  return {
    selectedDocs,
    setSelectedDocs,
    selectedDocuments,
    messages,
    inputValue,
    setInputValue,
    chatStarted,
    setChatStarted,
    webSearchEnabled,
    setWebSearchEnabled,
    toggleDocumentSelection,
    removeDocumentFromSelection,
    handleSend,
    handleKeyDown,
    chatThreads,
    chatListLoading,
    activeThreadId,
    creatingNewChat,
    handleCreateNewChat,
    handleSelectChatThread,
    handleDeleteChatThread,
  };
}
