import { useCallback, useEffect, useMemo, useState } from "react";
import { Monitor, Moon, Sun } from "lucide-react";
import type { ThemeMode } from "../../store/ui-store";

interface UseLayoutControllerParams {
  pathname: string;
  sidebarOpen: boolean;
  themeMode: ThemeMode;
  setSidebarOpen: (value: boolean) => void;
  setThemeMode: (value: ThemeMode) => void;
}

export function useLayoutController({
  pathname,
  sidebarOpen,
  themeMode,
  setSidebarOpen,
  setThemeMode,
}: UseLayoutControllerParams) {
  const [systemPrefersDark, setSystemPrefersDark] = useState(false);
  const [isCompactViewport, setIsCompactViewport] = useState(false);
  const [isMobileViewport, setIsMobileViewport] = useState(false);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  useEffect(() => {
    if (isCompactViewport) {
      setMobileSidebarOpen(false);
    }
  }, [isCompactViewport, pathname]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(prefers-color-scheme: dark)");

    const handleChange = () => {
      setSystemPrefersDark(media.matches);
    };

    handleChange();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", handleChange);
      return () => media.removeEventListener("change", handleChange);
    }

    media.addListener(handleChange);
    return () => media.removeListener(handleChange);
  }, []);

  const resolvedThemeMode =
    themeMode === "system" ? (systemPrefersDark ? "dark" : "light") : themeMode;

  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolvedThemeMode === "dark");
  }, [resolvedThemeMode]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(max-width: 1100px)");

    const syncCompact = () => {
      setIsCompactViewport(media.matches);
      if (!media.matches) {
        setMobileSidebarOpen(false);
      }
    };

    syncCompact();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", syncCompact);
      return () => media.removeEventListener("change", syncCompact);
    }

    media.addListener(syncCompact);
    return () => media.removeListener(syncCompact);
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const media = window.matchMedia("(max-width: 768px)");

    const syncMobile = () => {
      setIsMobileViewport(media.matches);
    };

    syncMobile();

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", syncMobile);
      return () => media.removeEventListener("change", syncMobile);
    }

    media.addListener(syncMobile);
    return () => media.removeListener(syncMobile);
  }, []);

  const handleSidebarToggle = useCallback(() => {
    if (isCompactViewport) {
      setMobileSidebarOpen((previous) => !previous);
      return;
    }
    setSidebarOpen(!sidebarOpen);
  }, [isCompactViewport, setSidebarOpen, sidebarOpen]);

  const showExpandedSidebar = isCompactViewport ? mobileSidebarOpen : sidebarOpen;
  const sidebarClass = showExpandedSidebar
    ? "w-[280px] min-w-[280px]"
    : "w-[72px] min-w-[72px]";

  const toggleThemeMode = useCallback(() => {
    const nextMode =
      themeMode === "dark" ? "light" : themeMode === "light" ? "system" : "dark";
    setThemeMode(nextMode);
  }, [setThemeMode, themeMode]);

  const themeModeLabel =
    themeMode === "dark" ? "Dark" : themeMode === "light" ? "Light" : "System";

  const themeIcon = useMemo(
    () =>
      themeMode === "dark" ? (
        <Moon size={18} />
      ) : themeMode === "light" ? (
        <Sun size={18} />
      ) : (
        <Monitor size={18} />
      ),
    [themeMode],
  );

  return {
    resolvedThemeMode,
    isCompactViewport,
    isMobileViewport,
    mobileSidebarOpen,
    setMobileSidebarOpen,
    showExpandedSidebar,
    sidebarClass,
    handleSidebarToggle,
    toggleThemeMode,
    themeModeLabel,
    themeIcon,
  };
}
