import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatMessage } from "../workspace/models";

type ThoughtEvent = NonNullable<NonNullable<ChatMessage["debug"]>["events"]>[number];

interface ThinkingPanelProps {
  debug: NonNullable<ChatMessage["debug"]>;
  isStreaming: boolean;
}

const toNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const safeText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return String(value);
};

const formatSeconds = (ms: number): string => {
  if (!Number.isFinite(ms) || ms <= 0) return "0s";
  if (ms < 1000) return "0s";
  return `${Math.round(ms / 1000)}s`;
};

const safeJson = (value: unknown): string => {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const normalizeStageName = (rawStage: string): string => {
  const stage = rawStage.trim().toLowerCase();
  if (stage === "retrieve") return "retrieval";
  if (stage === "notebook-recursive" || stage === "recursive-rag") return "retrieval";
  return rawStage;
};

const mapThoughtLine = (event: ThoughtEvent, index: number): string => {
  const rawStage = safeText(event.stage || "unknown");
  const normalizedStage = normalizeStageName(rawStage);
  const stageLabel =
    normalizedStage === rawStage
      ? normalizedStage
      : `${normalizedStage} (raw: ${rawStage})`;
  const label = `[${index + 1}] stage=${stageLabel}`;
  const detail = safeText(event.detail || "").trim();
  const hasData = event.data !== undefined;
  const dataText = hasData ? safeJson(event.data) : "";
  if (hasData) {
    return `${label}\n${detail || "(no detail)"}\n${dataText}`;
  }
  return `${label}\n${detail || "(no detail)"}`;
};

const SpinnerOrCheck = ({ done }: { done: boolean }) => {
  if (done) {
    return (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-hidden="true">
        <circle cx="7" cy="7" r="6" stroke="#2d6f52" strokeWidth="1" />
        <path
          d="M4 7l2 2 4-4"
          stroke="#45c17a"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }

  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
      className="animate-spin"
    >
      <circle cx="7" cy="7" r="5.5" stroke="#43516a" strokeWidth="1.5" />
      <path d="M7 1.5A5.5 5.5 0 0 1 12.5 7" stroke="#9dc1e8" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
};

export function ThinkingPanel({ debug, isStreaming }: ThinkingPanelProps) {
  const events = debug.events ?? [];
  const detailsRef = useRef<HTMLDetailsElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const appendedCountRef = useRef(0);
  const queueRef = useRef<string[]>([]);
  const typingTimerRef = useRef<number | null>(null);
  const currentChunkRef = useRef<string>("");
  const userScrolledRef = useRef(false);
  const [open, setOpen] = useState(isStreaming);

  const startedAt = debug.startedAt ?? events.find((event) => event.at != null)?.at;
  const completedAt = debug.completedAt;
  const elapsedMs = useMemo(() => {
    if (startedAt == null) return 0;
    return Math.max(0, (completedAt ?? Date.now()) - startedAt);
  }, [completedAt, startedAt, isStreaming, events.length]);

  const done = !isStreaming;
  const hasEvents = events.length > 0;
  const summaryText = done
    ? `Retrieval log · ${formatSeconds(elapsedMs)}`
    : "Running retrieval pipeline…";

  const maybeScrollBottom = () => {
    const body = bodyRef.current;
    if (!body || userScrolledRef.current) return;
    body.scrollTop = body.scrollHeight;
  };

  const stopTypingTimer = () => {
    if (typingTimerRef.current != null) {
      window.clearInterval(typingTimerRef.current);
      typingTimerRef.current = null;
    }
  };

  const flushAllNow = () => {
    const target = bodyRef.current;
    if (!target) return;
    stopTypingTimer();

    if (currentChunkRef.current.length > 0) {
      target.append(document.createTextNode(currentChunkRef.current));
      currentChunkRef.current = "";
    }

    while (queueRef.current.length > 0) {
      const chunk = queueRef.current.shift();
      if (chunk) target.append(document.createTextNode(chunk));
    }
    maybeScrollBottom();
  };

  const pumpTyping = () => {
    if (typingTimerRef.current != null) return;
    const body = bodyRef.current;
    if (!body) return;

    const nextChunk = () => {
      if (currentChunkRef.current.length > 0) return true;
      const queued = queueRef.current.shift();
      if (!queued) return false;
      currentChunkRef.current = queued;
      return true;
    };

    if (!nextChunk()) return;

    typingTimerRef.current = window.setInterval(() => {
      const target = bodyRef.current;
      if (!target) {
        stopTypingTimer();
        return;
      }

      if (currentChunkRef.current.length === 0) {
        const hasMore = nextChunk();
        if (!hasMore) {
          stopTypingTimer();
          return;
        }
      }

      const ch = currentChunkRef.current.slice(0, 1);
      currentChunkRef.current = currentChunkRef.current.slice(1);
      target.append(document.createTextNode(ch));
      maybeScrollBottom();
    }, 8);
  };

  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;

    if (isStreaming) {
      setOpen(true);
      if (detailsRef.current) detailsRef.current.open = true;
      return;
    }

    flushAllNow();
    setOpen(false);
    if (detailsRef.current) detailsRef.current.open = false;
  }, [isStreaming]);

  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;

    if (appendedCountRef.current > events.length) {
      appendedCountRef.current = events.length;
    }

    const startIndex = appendedCountRef.current;
    if (startIndex >= events.length) return;

    const incoming = events.slice(startIndex);
    appendedCountRef.current = events.length;

    for (let i = 0; i < incoming.length; i += 1) {
      const event = incoming[i];
      const absoluteIndex = startIndex + i;
      queueRef.current.push(`${mapThoughtLine(event, absoluteIndex)}\n\n`);
    }

    if (isStreaming) {
      pumpTyping();
      return;
    }
    flushAllNow();
  }, [events, isStreaming]);

  useEffect(() => {
    return () => {
      stopTypingTimer();
    };
  }, []);

  // Keep hooks unconditional; hide panel only after all hooks are declared.
  if (!hasEvents) {
    return null;
  }

  return (
    <details
      ref={detailsRef}
      open={open}
      onToggle={(event) => {
        const el = event.currentTarget;
        setOpen(el.open);
      }}
      className="mb-2"
    >
      <summary
        aria-label="Toggle retrieval log"
        className="flex cursor-pointer list-none items-center gap-2 py-1 text-xs font-semibold text-[#9dc1e8]"
      >
        <SpinnerOrCheck done={done} />
        <span>{summaryText}</span>
      </summary>

      <div
        ref={bodyRef}
        role="log"
        aria-live="polite"
        aria-label="Retrieval pipeline log"
        onScroll={(event) => {
          const el = event.currentTarget;
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          userScrolledRef.current = !atBottom;
        }}
        className="max-h-[320px] overflow-y-auto pb-2 font-mono text-[12px] leading-6 text-[#c2cfde] whitespace-pre-wrap break-words"
      />
    </details>
  );
}
