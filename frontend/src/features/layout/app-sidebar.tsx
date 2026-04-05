import type { ReactNode } from "react";
import {
  ChevronDown,
  FileText,
  Library,
  MessageSquare,
  MessageSquareText,
  PanelLeft,
  Plus,
  Settings,
  Trash2,
} from "lucide-react";
import { HintIconButton } from "../../components/hint-icon-button";
import type { ChatThreadItem, NavKey } from "../workspace/models";

interface AppSidebarProps {
  activeNav: NavKey;
  chatsExpanded: boolean;
  isCompactViewport: boolean;
  mobileSidebarOpen: boolean;
  showExpandedSidebar: boolean;
  sidebarClass: string;
  themeModeLabel: string;
  chatThreads: ChatThreadItem[];
  activeThreadId: string | null;
  chatListLoading: boolean;
  creatingNewChat: boolean;
  onNavigate: (path: string) => void;
  onSelectChat: (threadId: string) => void;
  onDeleteChat: (threadId: string) => void;
  onCreateNewChat: () => void;
  onToggleSidebar: () => void;
  onToggleChatsExpanded: () => void;
  onOpenSettings: () => void;
  onToggleThemeMode: () => void;
  themeIcon: ReactNode;
}

export function AppSidebar({
  activeNav,
  chatsExpanded,
  isCompactViewport,
  mobileSidebarOpen,
  showExpandedSidebar,
  sidebarClass,
  themeModeLabel,
  chatThreads,
  activeThreadId,
  chatListLoading,
  creatingNewChat,
  onNavigate,
  onSelectChat,
  onDeleteChat,
  onCreateNewChat,
  onToggleSidebar,
  onToggleChatsExpanded,
  onOpenSettings,
  onToggleThemeMode,
  themeIcon,
}: AppSidebarProps) {
  return (
    <aside
      className={`${sidebarClass} ${
        isCompactViewport
          ? "absolute inset-y-0 left-0 z-50 shadow-[8px_0_28px_rgba(0,0,0,0.35)]"
          : "relative"
      } flex flex-col border-r border-[#232937] bg-[#141820] transition-all duration-200 ${
        isCompactViewport
          ? mobileSidebarOpen
            ? "translate-x-0"
            : "-translate-x-full"
          : "translate-x-0"
      }`}
    >
      <div className="flex h-[72px] items-center px-4">
        {showExpandedSidebar ? (
          <div className="flex w-full items-center justify-between">
            <img
              src="https://gcore.jsdelivr.net/gh/stellariumImpl/CDN/pic/favicon.png"
              alt="PageIndex logo"
              className="sidebar-logo-mark h-10 w-10 rounded-xl object-cover"
            />
            <button
              onClick={onToggleSidebar}
              aria-label="Toggle sidebar"
              className="sidebar-logo-toggle flex h-10 w-10 items-center justify-center rounded-xl transition-colors hover:bg-[#232a38]"
            >
              <PanelLeft size={22} />
            </button>
          </div>
        ) : (
          <div className="group/logo flex w-full items-center justify-center">
            <div className="relative h-10 w-10">
              <button
                onClick={onToggleSidebar}
                aria-label="Toggle sidebar"
                className="sidebar-logo-toggle absolute inset-0 z-10 flex items-center justify-center rounded-xl opacity-0 transition-opacity duration-150 group-hover/logo:opacity-100 hover:bg-[#232a38]"
              >
                <PanelLeft size={22} />
              </button>

              <img
                src="https://gcore.jsdelivr.net/gh/stellariumImpl/CDN/pic/favicon.png"
                alt="PageIndex logo"
                className="sidebar-logo-mark h-10 w-10 rounded-xl object-cover transition-opacity duration-150 group-hover/logo:opacity-0"
              />
            </div>
          </div>
        )}
      </div>

      {showExpandedSidebar ? (
        <>
          <nav className="flex flex-col gap-1 px-4 pt-2">
            <button
              onClick={() => {
                onNavigate("/chat");
                onCreateNewChat();
              }}
              disabled={creatingNewChat}
              className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left text-sm transition-colors ${
                activeNav === "new-chat"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              } disabled:cursor-not-allowed disabled:opacity-60`}
            >
              <MessageSquare size={22} className="h-5 w-5" />
              <span>New Chat</span>
            </button>
            <button
              onClick={() => onNavigate("/documents")}
              className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left text-sm transition-colors ${
                activeNav === "documents"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              }`}
            >
              <FileText size={18} className="h-5 w-5" />
              <span>Documents</span>
            </button>
            <button
              onClick={() => onNavigate("/library")}
              className={`flex items-center gap-3 rounded-xl px-4 py-3 text-left text-sm transition-colors ${
                activeNav === "library"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              }`}
            >
              <Library size={18} className="h-5 w-5" />
              <span>Library</span>
            </button>
          </nav>

          <div className="mt-8 flex-1 px-6">
            <div className="flex items-center justify-between">
              <button
                onClick={onToggleChatsExpanded}
                className="flex items-center gap-2 text-sm text-[#8f95a3] transition-colors hover:text-white"
              >
                <span>Chats</span>
                <ChevronDown
                  size={14}
                  className={`transition-transform ${chatsExpanded ? "" : "-rotate-90"}`}
                />
              </button>
              <button
                onClick={() => {
                  onNavigate("/chat");
                  onCreateNewChat();
                }}
                disabled={creatingNewChat}
                className="rounded-md p-1 text-[#8f95a3] transition-colors hover:bg-[#252933] hover:text-white disabled:cursor-not-allowed disabled:opacity-60"
                aria-label="Create new chat"
                title="Create new chat"
              >
                <Plus size={14} />
              </button>
            </div>
            {chatsExpanded && (
              <div className="mt-3 max-h-[320px] space-y-1 overflow-y-auto pr-1">
                {chatListLoading ? (
                  <p className="py-2 text-sm text-[#9aa0ac]">Loading chats...</p>
                ) : chatThreads.length === 0 ? (
                  <p className="py-2 text-sm text-[#9aa0ac]">No chats yet</p>
                ) : (
                  chatThreads.map((thread) => (
                    <div key={thread.id} className="group/thread relative">
                      <button
                        onClick={() => {
                          onNavigate("/chat");
                          onSelectChat(thread.id);
                        }}
                        className={`w-full rounded-lg px-3 py-2 pr-10 text-left transition-colors ${
                          thread.id === activeThreadId
                            ? "bg-[#2d3240] text-white"
                            : "text-[#c3c8d3] hover:bg-[#252933]"
                        }`}
                      >
                        <p className="truncate text-sm font-medium">{thread.title}</p>
                        <p className="mt-0.5 truncate text-xs text-[#8f95a3]">
                          {thread.lastMessagePreview || "No messages yet"}
                        </p>
                      </button>
                      <button
                        onClick={(event) => {
                          event.stopPropagation();
                          onDeleteChat(thread.id);
                        }}
                        className="absolute right-2 top-2 rounded-md p-1 text-[#8f95a3] opacity-0 transition-all hover:bg-[#2a313f] hover:text-[#ff9a9a] group-hover/thread:opacity-100"
                        aria-label="Delete chat"
                        title="Delete chat"
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between px-6 pb-5 pt-4 text-[#b6bac4]">
            <span className="text-xs text-[#8f95a3]">Basic mode</span>
            <div className="flex items-center gap-5">
              <button className="transition-colors hover:text-white">
                <MessageSquareText size={18} />
              </button>
              <button
                onClick={onOpenSettings}
                className="transition-colors hover:text-white"
                aria-label="Open settings"
                title="Settings"
              >
                <Settings size={18} />
              </button>
              <button
                onClick={onToggleThemeMode}
                title={`Current theme: ${themeModeLabel}`}
                className="transition-colors hover:text-white"
              >
                {themeIcon}
              </button>
            </div>
          </div>
        </>
      ) : (
        <>
          <nav className="flex flex-col items-center gap-3 pt-2">
            <HintIconButton
              label="New Chat"
              hint="New Chat"
              onClick={() => {
                onNavigate("/chat");
                onCreateNewChat();
              }}
              disabled={creatingNewChat}
              className={`rounded-xl p-3 transition-colors ${
                activeNav === "new-chat"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              } disabled:cursor-not-allowed disabled:opacity-60`}
            >
              <MessageSquare size={20} />
            </HintIconButton>
            <HintIconButton
              label="Documents"
              hint="Documents"
              onClick={() => onNavigate("/documents")}
              className={`rounded-xl p-3 transition-colors ${
                activeNav === "documents"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              }`}
            >
              <FileText size={20} />
            </HintIconButton>
            <HintIconButton
              label="Library"
              hint="Library"
              onClick={() => onNavigate("/library")}
              className={`rounded-xl p-3 transition-colors ${
                activeNav === "library"
                  ? "bg-[#34363b] text-white"
                  : "text-[#d8dbe2] hover:bg-[#252933]"
              }`}
            >
              <Library size={20} />
            </HintIconButton>
          </nav>

          <div className="flex-1" />

          <div className="flex flex-col items-center gap-4 pb-8 text-[#b6bac4]">
            <HintIconButton
              label="Contact"
              hint="Contact"
              className="rounded-xl p-2.5 transition-colors hover:bg-[#252933] hover:text-white"
            >
              <MessageSquareText size={18} />
            </HintIconButton>
            <HintIconButton
              label="Settings"
              hint="Settings"
              onClick={onOpenSettings}
              className="rounded-xl p-2.5 transition-colors hover:bg-[#252933] hover:text-white"
            >
              <Settings size={18} />
            </HintIconButton>
            <HintIconButton
              label={`Current theme: ${themeModeLabel}`}
              hint={`Current theme: ${themeModeLabel}`}
              onClick={onToggleThemeMode}
              showHint={false}
              className="rounded-xl p-2.5 transition-colors hover:bg-[#252933] hover:text-white"
            >
              {themeIcon}
            </HintIconButton>
          </div>
        </>
      )}
    </aside>
  );
}
