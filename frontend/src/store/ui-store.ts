import { create } from "zustand";
import { persist } from "zustand/middleware";

export type ViewMode = "list" | "grid";
export type ThemeMode = "dark" | "light" | "system";

export type ContextMenu =
  | { kind: "folder"; id: string }
  | { kind: "document"; id: string }
  | null;

type UIStore = {
  sidebarOpen: boolean;
  chatsExpanded: boolean;
  viewMode: ViewMode;
  themeMode: ThemeMode;
  searchQuery: string;
  openMenu: ContextMenu;
  showNewFolderModal: boolean;
  showUploadModal: boolean;
  showSettingsModal: boolean;
  setSidebarOpen: (value: boolean) => void;
  setChatsExpanded: (value: boolean) => void;
  setViewMode: (value: ViewMode) => void;
  setThemeMode: (value: ThemeMode) => void;
  setSearchQuery: (value: string) => void;
  setOpenMenu: (value: ContextMenu) => void;
  setShowNewFolderModal: (value: boolean) => void;
  setShowUploadModal: (value: boolean) => void;
  setShowSettingsModal: (value: boolean) => void;
};

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      chatsExpanded: true,
      viewMode: "list",
      themeMode: "system",
      searchQuery: "",
      openMenu: null,
      showNewFolderModal: false,
      showUploadModal: false,
      showSettingsModal: false,
      setSidebarOpen: (value) => set({ sidebarOpen: value }),
      setChatsExpanded: (value) => set({ chatsExpanded: value }),
      setViewMode: (value) => set({ viewMode: value }),
      setThemeMode: (value) => set({ themeMode: value }),
      setSearchQuery: (value) => set({ searchQuery: value }),
      setOpenMenu: (value) => set({ openMenu: value }),
      setShowNewFolderModal: (value) =>
        set({ showNewFolderModal: value }),
      setShowUploadModal: (value) => set({ showUploadModal: value }),
      setShowSettingsModal: (value) => set({ showSettingsModal: value }),
    }),
    {
      name: "frontend-ui-store",
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        chatsExpanded: state.chatsExpanded,
        viewMode: state.viewMode,
        themeMode: state.themeMode,
      }),
    },
  ),
);
