import { useCallback, useEffect, useMemo } from "react";
import { PanelLeft } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useUIStore } from "./store/ui-store";
import { DocumentsView } from "./features/documents/documents-view";
import { ChatView } from "./features/chat/chat-view";
import { AppSidebar } from "./features/layout/app-sidebar";
import { NewFolderModal } from "./features/modals/new-folder-modal";
import { UploadModal } from "./features/modals/upload-modal";
import { MoveDocumentModal } from "./features/modals/move-document-modal";
import { PreviewModal } from "./features/modals/preview-modal";
import { SettingsModal } from "./features/modals/settings-modal";
import { ToastStack } from "./features/feedback/toast-stack";
import { useToastManager } from "./features/feedback/use-toast-manager";
import { useLayoutController } from "./features/layout/use-layout-controller";
import { useWorkspaceData } from "./features/workspace/use-workspace-data";
import { useChatSession } from "./features/chat/use-chat-session";
import { useDocumentActions } from "./features/documents/use-document-actions";
import type { NavKey } from "./features/workspace/models";
import { formatUploadDate } from "./features/workspace/utils";

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();

  const sidebarOpen = useUIStore((state) => state.sidebarOpen);
  const chatsExpanded = useUIStore((state) => state.chatsExpanded);
  const viewMode = useUIStore((state) => state.viewMode);
  const searchQuery = useUIStore((state) => state.searchQuery);
  const themeMode = useUIStore((state) => state.themeMode);
  const openMenu = useUIStore((state) => state.openMenu);
  const showNewFolderModal = useUIStore((state) => state.showNewFolderModal);
  const showUploadModal = useUIStore((state) => state.showUploadModal);
  const showSettingsModal = useUIStore((state) => state.showSettingsModal);

  const setSidebarOpen = useUIStore((state) => state.setSidebarOpen);
  const setChatsExpanded = useUIStore((state) => state.setChatsExpanded);
  const setViewMode = useUIStore((state) => state.setViewMode);
  const setThemeMode = useUIStore((state) => state.setThemeMode);
  const setSearchQuery = useUIStore((state) => state.setSearchQuery);
  const setOpenMenu = useUIStore((state) => state.setOpenMenu);
  const setShowNewFolderModal = useUIStore((state) => state.setShowNewFolderModal);
  const setShowUploadModal = useUIStore((state) => state.setShowUploadModal);
  const setShowSettingsModal = useUIStore((state) => state.setShowSettingsModal);

  const activeNav = useMemo<NavKey>(() => {
    if (location.pathname.startsWith("/chat")) return "new-chat";
    if (location.pathname.startsWith("/library")) return "library";
    return "documents";
  }, [location.pathname]);

  useEffect(() => {
    if (location.pathname === "/") {
      navigate("/documents", { replace: true });
    }
  }, [location.pathname, navigate]);

  const { toasts, pushToast, removeToast } = useToastManager();

  const {
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
  } = useLayoutController({
    pathname: location.pathname,
    sidebarOpen,
    themeMode,
    setSidebarOpen,
    setThemeMode,
  });

  const {
    folders,
    setFolders,
    documents,
    setDocuments,
    workspaceLoading,
    currentFolderId,
    setCurrentFolderId,
    foldersById,
    breadcrumbFolders,
    filteredFolders,
    filteredDocuments,
    formatFolderFileCount,
    moveFolderRows,
    hasVisibleItems,
    emptyStateTitle,
    emptyStateDescription,
    documentsById,
  } = useWorkspaceData({
    searchQuery,
    pushToast,
  });

  const {
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
    handleEditMessage,
    handleRetryMessage,
    handleKeyDown,
    chatThreads,
    chatListLoading,
    activeThreadId,
    creatingNewChat,
    handleCreateNewChat,
    handleSelectChatThread,
    handleDeleteChatThread,
  } = useChatSession({
    documents,
    documentsById,
    pushToast,
  });

  const {
    folderSubmitting,
    uploadingDocument,
    newFolderName,
    setNewFolderName,
    newFolderDescription,
    setNewFolderDescription,
    uploadInputRef,
    uploadDragOver,
    setUploadDragOver,
    quickUploadDragOver,
    setQuickUploadDragOver,
    previewDocument,
    previewDocumentUrl,
    previewDocumentDescription,
    setPreviewDocumentId,
    autofillingDescriptionDocId,
    movingDocument,
    moveDestinationFolderId,
    setMoveDestinationFolderId,
    moveModalSubmitting,
    selectedMoveDestinationLabel,
    handleToggleDocumentSelectionWithGuard,
    handleCreateFolder,
    handleShareDocument,
    handleUploadPickedFile,
    handleCreateMarkdownDocument,
    openUploadFilePicker,
    handleUploadInputChange,
    removeDocument,
    removeFolder,
    createSubfolder,
    renameFolder,
    handleDownloadDocument,
    openDocumentChat,
    openDocumentDetails,
    openMoveDocumentModal,
    closeMoveDocumentModal,
    handleConfirmMoveDocument,
    enterFolder,
    closeNewFolderModal,
    closeUploadModal,
  } = useDocumentActions({
    documents,
    folders,
    currentFolderId,
    foldersById,
    documentsById,
    setCurrentFolderId,
    setFolders,
    setDocuments,
    setSelectedDocs,
    setChatStarted,
    toggleDocumentSelection,
    setOpenMenu,
    setShowNewFolderModal,
    setShowUploadModal,
    pushToast,
    navigate,
  });

  const navigateFromSidebar = useCallback(
    (path: string) => {
      navigate(path);
      setOpenMenu(null);
      if (isCompactViewport) {
        setMobileSidebarOpen(false);
      }
    },
    [isCompactViewport, navigate, setMobileSidebarOpen, setOpenMenu],
  );

  const activeChatTitle = useMemo(() => {
    if (activeNav !== "new-chat") return "";
    if (!activeThreadId) return "New Chat";
    const current = chatThreads.find((item) => item.id === activeThreadId);
    const title = current?.title?.trim() || "";
    return title || "New Chat";
  }, [activeNav, activeThreadId, chatThreads]);

  return (
    <div className="app-shell h-screen w-full bg-[#0b0f17] text-white" data-theme={resolvedThemeMode}>
      <div
        className="relative flex h-full w-full overflow-hidden bg-[#080b13]"
        onClick={() => setOpenMenu(null)}
      >
        {isCompactViewport && mobileSidebarOpen && (
          <button
            onClick={() => setMobileSidebarOpen(false)}
            className="absolute inset-0 z-40 bg-black/55"
            aria-label="Close sidebar"
          />
        )}

        <AppSidebar
          activeNav={activeNav}
          chatsExpanded={chatsExpanded}
          isCompactViewport={isCompactViewport}
          mobileSidebarOpen={mobileSidebarOpen}
          showExpandedSidebar={showExpandedSidebar}
          sidebarClass={sidebarClass}
          themeModeLabel={themeModeLabel}
          chatThreads={chatThreads}
          activeThreadId={activeThreadId}
          chatListLoading={chatListLoading}
          creatingNewChat={creatingNewChat}
          onNavigate={navigateFromSidebar}
          onSelectChat={(threadId) => {
            void handleSelectChatThread(threadId);
          }}
          onDeleteChat={(threadId) => {
            void handleDeleteChatThread(threadId);
          }}
          onCreateNewChat={() => {
            void handleCreateNewChat();
          }}
          onToggleSidebar={handleSidebarToggle}
          onToggleChatsExpanded={() => setChatsExpanded(!chatsExpanded)}
          onOpenSettings={() => setShowSettingsModal(true)}
          onToggleThemeMode={toggleThemeMode}
          themeIcon={themeIcon}
        />

        <main className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-[72px] items-center border-b border-[#1f2633] bg-[#0b1220] px-5">
            <div className="flex items-center">
              {isCompactViewport && (
                <button
                  onClick={() => setMobileSidebarOpen(true)}
                  className="mr-3 rounded-md p-1 text-[#a4acba] transition-colors hover:text-white"
                  aria-label="Open sidebar"
                >
                  <PanelLeft size={20} />
                </button>
              )}
              <h1 className="text-base font-medium text-white">
                {activeNav === "documents"
                  ? "Documents"
                  : activeNav === "new-chat"
                    ? activeChatTitle
                    : "Library"}
              </h1>
            </div>
          </header>

          {activeNav === "documents" && (
            <DocumentsView
              workspaceLoading={workspaceLoading}
              uploadingDocument={uploadingDocument}
              folderSubmitting={folderSubmitting}
              breadcrumbFolders={breadcrumbFolders}
              searchQuery={searchQuery}
              viewMode={viewMode}
              hasVisibleItems={hasVisibleItems}
              filteredFolders={filteredFolders}
              filteredDocuments={filteredDocuments}
              openMenu={openMenu}
              isMobileViewport={isMobileViewport}
              emptyStateTitle={emptyStateTitle}
              emptyStateDescription={emptyStateDescription}
              formatFolderFileCount={formatFolderFileCount}
              onOpenUploadModal={() => setShowUploadModal(true)}
              onOpenNewFolderModal={() => setShowNewFolderModal(true)}
              onSetRootFolder={() => setCurrentFolderId(null)}
              onSetCurrentFolder={setCurrentFolderId}
              onSearchQueryChange={setSearchQuery}
              onViewModeChange={setViewMode}
              onEnterFolder={enterFolder}
              onOpenMenuChange={setOpenMenu}
              onCreateSubfolder={createSubfolder}
              onRenameFolder={renameFolder}
              onRemoveFolder={removeFolder}
              onOpenDocumentDetails={openDocumentDetails}
              onOpenDocumentChat={openDocumentChat}
              onOpenMoveDocumentModal={openMoveDocumentModal}
              onDownloadDocument={handleDownloadDocument}
              onShareDocument={handleShareDocument}
              onRemoveDocument={removeDocument}
            />
          )}

          {activeNav === "new-chat" && (
            <ChatView
              chatStarted={chatStarted}
              folders={folders}
              documents={documents}
              uploadingDocument={uploadingDocument}
              quickUploadDragOver={quickUploadDragOver}
              selectedDocs={selectedDocs}
              selectedDocuments={selectedDocuments}
              messages={messages}
              inputValue={inputValue}
              webSearchEnabled={webSearchEnabled}
              onSetQuickUploadDragOver={setQuickUploadDragOver}
              onUploadPickedFile={(file) => handleUploadPickedFile(file, "root")}
              onOpenUploadFilePicker={() => openUploadFilePicker("root")}
              onToggleDocumentSelectionWithGuard={handleToggleDocumentSelectionWithGuard}
              onRemoveDocumentFromSelection={removeDocumentFromSelection}
              onInputValueChange={setInputValue}
              onComposerKeyDown={handleKeyDown}
              onToggleWebSearch={() => setWebSearchEnabled((previous) => !previous)}
              onSend={handleSend}
              onEditMessage={handleEditMessage}
              onRetryMessage={handleRetryMessage}
            />
          )}

          {activeNav === "library" && (
            <section className="flex flex-1 items-center justify-center text-[#8f95a3]">
              Library content placeholder
            </section>
          )}
        </main>

        <NewFolderModal
          open={showNewFolderModal}
          isMobileViewport={isMobileViewport}
          currentFolderId={currentFolderId}
          newFolderName={newFolderName}
          newFolderDescription={newFolderDescription}
          folderSubmitting={folderSubmitting}
          onClose={closeNewFolderModal}
          onNameChange={setNewFolderName}
          onDescriptionChange={setNewFolderDescription}
          onSubmit={handleCreateFolder}
        />

        <UploadModal
          open={showUploadModal}
          isMobileViewport={isMobileViewport}
          uploadingDocument={uploadingDocument}
          uploadDragOver={uploadDragOver}
          onClose={closeUploadModal}
          onOpenUploadFilePicker={() => openUploadFilePicker("current-folder")}
          onSetUploadDragOver={setUploadDragOver}
          onUploadPickedFile={(file) => handleUploadPickedFile(file, "current-folder")}
          onCreateMarkdownDocument={(content, sourceName) =>
            handleCreateMarkdownDocument(content, sourceName, "current-folder")
          }
        />

        <MoveDocumentModal
          movingDocument={movingDocument}
          isMobileViewport={isMobileViewport}
          moveModalSubmitting={moveModalSubmitting}
          moveDestinationFolderId={moveDestinationFolderId}
          moveFolderRows={moveFolderRows}
          selectedMoveDestinationLabel={selectedMoveDestinationLabel}
          onClose={closeMoveDocumentModal}
          onSetMoveDestinationFolderId={setMoveDestinationFolderId}
          onConfirm={handleConfirmMoveDocument}
        />

        <PreviewModal
          previewDocument={previewDocument}
          previewDocumentUrl={previewDocumentUrl}
          previewDocumentDescription={previewDocumentDescription}
          autofillingDescriptionDocId={autofillingDescriptionDocId}
          isMobileViewport={isMobileViewport}
          formatUploadDate={formatUploadDate}
          onClose={() => setPreviewDocumentId(null)}
        />

        <SettingsModal
          open={showSettingsModal}
          isMobileViewport={isMobileViewport}
          theme={resolvedThemeMode}
          onClose={() => setShowSettingsModal(false)}
        />

        <ToastStack toasts={toasts} onDismiss={removeToast} />

        <input
          ref={uploadInputRef}
          type="file"
          accept=".pdf,application/pdf,.md,.markdown,text/markdown,text/x-markdown,text/plain"
          className="hidden"
          onChange={handleUploadInputChange}
        />
      </div>
    </div>
  );
}
