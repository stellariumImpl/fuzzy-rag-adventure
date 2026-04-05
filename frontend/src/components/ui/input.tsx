import * as React from "react";

import { cn } from "../../lib/utils";

const Input = React.forwardRef<
  HTMLInputElement,
  React.ComponentProps<"input">
>(({ className, type, ...props }, ref) => {
  return (
    <input
      type={type}
      ref={ref}
      className={cn(
        "flex h-10 w-full rounded-md border px-3 py-2 text-sm outline-none transition-colors",
        className,
      )}
      {...props}
    />
  );
});

Input.displayName = "Input";

export { Input };

