import type { ReactNode } from "react";

type HintIconButtonProps = {
  label: string;
  hint: string;
  children: ReactNode;
  className: string;
  wrapperClassName?: string;
  tooltipClassName?: string;
  onClick?: () => void;
  disabled?: boolean;
  showHint?: boolean;
  pressed?: boolean;
};

export function HintIconButton({
  label,
  hint,
  children,
  className,
  wrapperClassName = "",
  tooltipClassName = "",
  onClick,
  disabled = false,
  showHint = true,
  pressed,
}: HintIconButtonProps) {
  return (
    <div className={`sidebar-hint-button ${wrapperClassName}`.trim()}>
      <button
        onClick={onClick}
        disabled={disabled}
        className={`sidebar-hint-control ${className}`}
        aria-label={label}
        title={label}
        aria-pressed={pressed}
      >
        {children}
      </button>
      {showHint && (
        <span
          className={`sidebar-hint-tooltip ${tooltipClassName}`.trim()}
          role="tooltip"
        >
          {hint}
        </span>
      )}
    </div>
  );
}
