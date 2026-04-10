import {
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from "react";
import type { ChatMessage } from "../workspace/models";

type ThoughtEvent = NonNullable<
  NonNullable<ChatMessage["debug"]>["events"]
>[number];

interface ThinkingPanelProps {
  debug: NonNullable<ChatMessage["debug"]>;
  isStreaming: boolean;
}

type ToolEventView = {
  key: string;
  title: string;
  parameters: unknown;
  result: unknown;
  at: string | null;
};

const TOOL_META_KEYS = new Set([
  "tool_name",
  "tool",
  "name",
  "operation",
  "action",
  "title",
  "parameters",
  "params",
  "arguments",
  "args",
  "result",
  "output",
  "response",
]);

const safeText = (value: unknown): string => {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return String(value);
};

const asRecord = (value: unknown): Record<string, unknown> | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
};

const formatSeconds = (ms: number): string => {
  if (!Number.isFinite(ms) || ms <= 0) return "0s";
  if (ms < 1000) return "0s";
  return `${Math.round(ms / 1000)}s`;
};

const formatIsoTime = (value: number | undefined): string | null => {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  try {
    return new Date(value).toISOString();
  } catch {
    return null;
  }
};

const jsonBlock = (value: unknown): string => {
  if (value === undefined) return "{}";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return JSON.stringify(String(value), null, 2);
  }
};

const isSyntheticRetrievalEvent = (
  title: string,
  data: Record<string, unknown> | null,
): boolean => {
  const normalized = title.trim().toLowerCase();
  if (normalized !== "top recalled chunks") return false;
  if (!data) return false;
  return (
    Object.prototype.hasOwnProperty.call(data, "total_hits") &&
    Object.prototype.hasOwnProperty.call(data, "top_hits")
  );
};

const toToolEvent = (
  event: ThoughtEvent,
  index: number,
): ToolEventView | null => {
  const data = asRecord(event.data);

  const titleCandidates = [
    safeText(data?.tool_name),
    safeText(data?.tool),
    safeText(data?.name),
    safeText(data?.operation),
    safeText(data?.action),
    safeText(data?.title),
    safeText(event.detail),
    safeText(event.stage),
  ]
    .map((value) => value.trim())
    .filter((value) => value.length > 0);

  const title = titleCandidates[0] ?? "";
  if (!title) return null;
  if (isSyntheticRetrievalEvent(title, data)) return null;

  const parameters =
    data?.parameters ?? data?.params ?? data?.arguments ?? data?.args;

  let result = data?.result ?? data?.output ?? data?.response;

  if (result === undefined && data) {
    const entries = Object.entries(data).filter(
      ([key]) => !TOOL_META_KEYS.has(key),
    );
    if (entries.length > 0) {
      result = Object.fromEntries(entries);
    }
  }

  if (parameters === undefined && result === undefined) return null;

  return {
    key: `${index}-${safeText(event.at)}-${title}`,
    title,
    parameters: parameters ?? {},
    result: result ?? {},
    at: formatIsoTime(event.at),
  };
};

const SpinnerOrCheck = ({ done }: { done: boolean }) => {
  if (done) {
    return (
      <svg
        width="14"
        height="14"
        viewBox="0 0 14 14"
        fill="none"
        aria-hidden="true"
      >
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
    <span className="claude-loader" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
};

export function ThinkingPanel({ debug, isStreaming }: ThinkingPanelProps) {
  const events = debug.events ?? [];
  const done = !isStreaming;
  const hasEvents = events.length > 0;

  const [open, setOpen] = useState(false);
  const [nowTick, setNowTick] = useState(() => Date.now());

  const startedAt =
    debug.startedAt ?? events.find((event) => event.at != null)?.at;
  const completedAt = debug.completedAt;
  const elapsedMs = useMemo(() => {
    if (startedAt == null) return 0;
    return Math.max(0, (completedAt ?? nowTick) - startedAt);
  }, [completedAt, startedAt, nowTick]);

  const toolEvents = useMemo(
    () =>
      events
        .map((event, index) => toToolEvent(event, index))
        .filter((event): event is ToolEventView => event != null),
    [events],
  );
  const hasToolEvents = toolEvents.length > 0;

  useEffect(() => {
    if (!isStreaming) return;
    const timer = window.setInterval(() => {
      setNowTick(Date.now());
    }, 450);
    return () => {
      window.clearInterval(timer);
    };
  }, [isStreaming]);

  useEffect(() => {
    if (isStreaming && hasEvents) {
      setOpen(true);
      return;
    }
    if (!isStreaming) {
      setOpen(false);
    }
  }, [hasEvents, isStreaming]);

  if (!hasToolEvents && !isStreaming) {
    return null;
  }

  const title = hasToolEvents
    ? toolEvents[toolEvents.length - 1].title
    : done
      ? "Retrieval Finished"
      : "Generating Answer";
  const subtitle = hasToolEvents
    ? `${toolEvents.length} tool events · ${formatSeconds(elapsedMs)}`
    : `Waiting for tool events · ${formatSeconds(elapsedMs)}`;
  const summary = hasToolEvents
    ? `Latest: ${toolEvents[toolEvents.length - 1].title}`
    : done
      ? "No tool events were returned by backend."
      : "Waiting for backend tool events...";

  const toggleOpen = () => {
    if (!hasToolEvents) return;
    setOpen((value) => !value);
  };

  const handleCardClick = (event: MouseEvent<HTMLElement>) => {
    const target = event.target as HTMLElement | null;
    if (target?.closest("[data-retrieval-log-content='true']")) return;
    toggleOpen();
  };

  const handleCardKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (!hasToolEvents) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleOpen();
    }
  };

  return (
    <section
      className={`retrieval-log-card mb-2 w-full ${hasToolEvents ? "is-toggleable" : ""} ${open ? "is-open" : ""}`}
      onClick={handleCardClick}
      onKeyDown={handleCardKeyDown}
      role={hasToolEvents ? "button" : undefined}
      tabIndex={hasToolEvents ? 0 : undefined}
      aria-expanded={hasToolEvents ? open : undefined}
      aria-label={hasToolEvents ? "Toggle tool log" : undefined}
    >
      <div className="retrieval-log-header">
        <span className="retrieval-log-icon">
          <SpinnerOrCheck done={done} />
        </span>
        <div className="min-w-0">
          <p className="retrieval-log-title">{title}</p>
          {subtitle ? (
            <p className="retrieval-log-subtitle">{subtitle}</p>
          ) : null}
        </div>
        {/* {hasToolEvents ? (
          <button
            type="button"
            className="retrieval-log-toggle"
            onClick={(event) => {
              event.stopPropagation();
              setOpen((value) => !value);
            }}
          >
            {open ? "Hide log" : "Show log"}
          </button>
        ) : null} */}
      </div>

      {isStreaming ? (
        <div className="retrieval-log-pulse" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
      ) : null}

      <p className="retrieval-log-summary">{summary}</p>

      {hasToolEvents && open ? (
        <ol className="retrieval-log-list" data-retrieval-log-content="true">
          {toolEvents.map((event, index) => (
            <li key={event.key} className="retrieval-log-item">
              <div className="retrieval-log-item-head">
                <span className="retrieval-log-stage-index">[{index + 1}]</span>
                <span className="retrieval-log-stage-name">{event.title}</span>
              </div>
              {event.at ? (
                <p className="retrieval-log-meta">{event.at}</p>
              ) : null}

              <div className="retrieval-log-block">
                <p className="retrieval-log-block-title">Parameters</p>
                <pre className="retrieval-log-json">
                  <code>{jsonBlock(event.parameters)}</code>
                </pre>
              </div>

              <div className="retrieval-log-block">
                <p className="retrieval-log-block-title">Result</p>
                <pre className="retrieval-log-json">
                  <code>{jsonBlock(event.result)}</code>
                </pre>
              </div>
            </li>
          ))}
        </ol>
      ) : null}
    </section>
  );
}
