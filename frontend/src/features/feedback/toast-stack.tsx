import { CheckCircle2, Info, TriangleAlert, X } from "lucide-react";
import type { ToastMessage } from "../workspace/models";

interface ToastStackProps {
  toasts: ToastMessage[];
  onDismiss: (id: string) => void;
}

const renderToastIcon = (tone: ToastMessage["tone"]) => {
  if (tone === "success") return <CheckCircle2 size={16} />;
  if (tone === "error") return <TriangleAlert size={16} />;
  return <Info size={16} />;
};

export function ToastStack({ toasts, onDismiss }: ToastStackProps) {
  return (
    <div className="app-toast-stack">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`app-toast app-toast-${toast.tone}`}
          role="status"
          aria-live="polite"
        >
          <div className={`app-toast-icon app-toast-icon-${toast.tone}`}>
            {renderToastIcon(toast.tone)}
          </div>
          <div className="app-toast-content">
            <p className="app-toast-title">{toast.title}</p>
            {toast.description && (
              <p className="app-toast-description">{toast.description}</p>
            )}
          </div>
          <button
            onClick={() => onDismiss(toast.id)}
            className="app-toast-close"
            aria-label="Dismiss notification"
          >
            <X size={15} />
          </button>
          <div
            className="app-toast-progress"
            style={{
              animationDuration: `${toast.durationMs}ms`,
            }}
          />
        </div>
      ))}
    </div>
  );
}
