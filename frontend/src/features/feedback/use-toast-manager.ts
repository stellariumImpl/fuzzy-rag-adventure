import { useCallback, useEffect, useRef, useState } from "react";
import type { ToastMessage } from "../workspace/models";

export function useToastManager() {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const toastTimersRef = useRef<Map<string, number>>(new Map());

  const removeToast = useCallback((id: string) => {
    const timer = toastTimersRef.current.get(id);
    if (timer) {
      window.clearTimeout(timer);
      toastTimersRef.current.delete(id);
    }
    setToasts((previous) => previous.filter((toast) => toast.id !== id));
  }, []);

  const pushToast = useCallback(
    (
      title: string,
      options: {
        tone?: ToastMessage["tone"];
        description?: string;
        durationMs?: number;
      } = {},
    ) => {
      const tone = options.tone ?? "info";
      const durationMs = options.durationMs ?? (tone === "error" ? 5200 : 3800);
      const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      setToasts((previous) => [
        ...previous,
        {
          id,
          tone,
          title,
          description: options.description,
          durationMs,
        },
      ]);

      const timeoutId = window.setTimeout(() => {
        toastTimersRef.current.delete(id);
        setToasts((previous) => previous.filter((toast) => toast.id !== id));
      }, durationMs);
      toastTimersRef.current.set(id, timeoutId);
    },
    [],
  );

  useEffect(
    () => () => {
      for (const timeoutId of toastTimersRef.current.values()) {
        window.clearTimeout(timeoutId);
      }
      toastTimersRef.current.clear();
    },
    [],
  );

  return {
    toasts,
    pushToast,
    removeToast,
  };
}
