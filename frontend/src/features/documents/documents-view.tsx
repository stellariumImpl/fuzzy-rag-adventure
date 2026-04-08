import { useMemo, useState } from "react";
import {
  Download,
  FileText,
  Folder,
  FolderPlus,
  House,
  Info,
  LayoutGrid,
  List,
  MessageSquare,
  MoreVertical,
  Move,
  Pencil,
  Plus,
  Search,
  Share2,
  Trash2,
  Upload,
} from "lucide-react";
import { DangerConfirmModal } from "../modals/danger-confirm-modal";
import { FolderNameModal } from "../modals/folder-name-modal";
import type { ContextMenu, ViewMode } from "../../store/ui-store";
import type { DocumentItem, FolderItem } from "../workspace/models";
import { DocumentStatusBadge } from "../workspace/components/document-status-badge";
import {
  formatUploadedLabel,
  getDocumentDescriptionLabel,
  isDocumentProcessing,
} from "../workspace/utils";

type FolderDialogState = {
  mode: "create-subfolder" | "rename-folder";
  folderId: string;
  folderName: string;
};

type DeleteDialogState = {
  kind: "folder" | "document";
  id: string;
  name: string;
};

interface DocumentsViewProps {
  workspaceLoading: boolean;
  uploadingDocument: boolean;
  folderSubmitting: boolean;
  breadcrumbFolders: FolderItem[];
  searchQuery: string;
  viewMode: ViewMode;
  hasVisibleItems: boolean;
  filteredFolders: FolderItem[];
  filteredDocuments: DocumentItem[];
  openMenu: ContextMenu;
  isMobileViewport: boolean;
  emptyStateTitle: string;
  emptyStateDescription: string;
  formatFolderFileCount: (folderId: string) => string;
  onOpenUploadModal: () => void;
  onOpenNewFolderModal: () => void;
  onSetRootFolder: () => void;
  onSetCurrentFolder: (folderId: string) => void;
  onSearchQueryChange: (value: string) => void;
  onViewModeChange: (value: ViewMode) => void;
  onEnterFolder: (folderId: string) => void;
  onOpenMenuChange: (value: ContextMenu) => void;
  onCreateSubfolder: (
    folderId: string,
    folderName: string,
    folderDescription?: string,
  ) => Promise<boolean>;
  onRenameFolder: (folderId: string, nextName: string) => Promise<boolean>;
  onRemoveFolder: (folderId: string) => Promise<boolean>;
  onOpenDocumentDetails: (docId: string) => void;
  onOpenDocumentChat: (docId: string) => void;
  onOpenMoveDocumentModal: (docId: string) => void;
  onDownloadDocument: (docId: string) => void;
  onShareDocument: (docId: string) => Promise<void>;
  onRemoveDocument: (docId: string) => Promise<boolean>;
}

export function DocumentsView({
  workspaceLoading,
  uploadingDocument,
  folderSubmitting,
  breadcrumbFolders,
  searchQuery,
  viewMode,
  hasVisibleItems,
  filteredFolders,
  filteredDocuments,
  openMenu,
  isMobileViewport,
  emptyStateTitle,
  emptyStateDescription,
  formatFolderFileCount,
  onOpenUploadModal,
  onOpenNewFolderModal,
  onSetRootFolder,
  onSetCurrentFolder,
  onSearchQueryChange,
  onViewModeChange,
  onEnterFolder,
  onOpenMenuChange,
  onCreateSubfolder,
  onRenameFolder,
  onRemoveFolder,
  onOpenDocumentDetails,
  onOpenDocumentChat,
  onOpenMoveDocumentModal,
  onDownloadDocument,
  onShareDocument,
  onRemoveDocument,
}: DocumentsViewProps) {
  const [folderDialog, setFolderDialog] = useState<FolderDialogState | null>(null);
  const [folderDialogName, setFolderDialogName] = useState("");
  const [folderDialogDescription, setFolderDialogDescription] = useState("");
  const [folderDialogSubmitting, setFolderDialogSubmitting] = useState(false);
  const [deleteDialog, setDeleteDialog] = useState<DeleteDialogState | null>(null);
  const [deleteDialogSubmitting, setDeleteDialogSubmitting] = useState(false);

  const closeFolderDialog = () => {
    if (folderDialogSubmitting) return;
    setFolderDialog(null);
    setFolderDialogName("");
    setFolderDialogDescription("");
  };

  const openCreateSubfolderDialog = (folder: FolderItem) => {
    onOpenMenuChange(null);
    setFolderDialog({
      mode: "create-subfolder",
      folderId: folder.id,
      folderName: folder.name,
    });
    setFolderDialogName("");
    setFolderDialogDescription("");
  };

  const openRenameFolderDialog = (folder: FolderItem) => {
    onOpenMenuChange(null);
    setFolderDialog({
      mode: "rename-folder",
      folderId: folder.id,
      folderName: folder.name,
    });
    setFolderDialogName(folder.name);
    setFolderDialogDescription("");
  };

  const handleSubmitFolderDialog = async () => {
    if (!folderDialog) return;

    setFolderDialogSubmitting(true);
    try {
      const success =
        folderDialog.mode === "create-subfolder"
          ? await onCreateSubfolder(
              folderDialog.folderId,
              folderDialogName,
              folderDialogDescription,
            )
          : await onRenameFolder(folderDialog.folderId, folderDialogName);

      if (success) {
        setFolderDialog(null);
        setFolderDialogName("");
        setFolderDialogDescription("");
      }
    } finally {
      setFolderDialogSubmitting(false);
    }
  };

  const openDeleteDialog = (target: DeleteDialogState) => {
    onOpenMenuChange(null);
    setDeleteDialog(target);
  };

  const closeDeleteDialog = () => {
    if (deleteDialogSubmitting) return;
    setDeleteDialog(null);
  };

  const handleConfirmDelete = async () => {
    if (!deleteDialog) return;

    setDeleteDialogSubmitting(true);
    try {
      const success =
        deleteDialog.kind === "folder"
          ? await onRemoveFolder(deleteDialog.id)
          : await onRemoveDocument(deleteDialog.id);
      if (success) {
        setDeleteDialog(null);
      }
    } finally {
      setDeleteDialogSubmitting(false);
    }
  };

  const deleteDialogTitle = useMemo(() => {
    if (!deleteDialog) return "";
    return deleteDialog.kind === "folder" ? "Delete Folder" : "Delete Document";
  }, [deleteDialog]);

  const deleteDialogDescription = useMemo(() => {
    if (!deleteDialog) return "";
    if (deleteDialog.kind === "folder") {
      return `You are deleting folder "${deleteDialog.name}". Please confirm this dangerous action.`;
    }
    return `You are deleting document "${deleteDialog.name}". Please confirm this dangerous action.`;
  }, [deleteDialog]);

  return (
    <section className="flex-1 overflow-x-auto overflow-y-auto px-6 pb-6 pt-8 md:px-8">
      <h2 className="mb-6 text-3xl font-semibold text-white">My Documents</h2>
      {workspaceLoading && (
        <p className="mb-5 text-sm text-[#a5acb8]">Loading workspace...</p>
      )}

      <div className="mb-7 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex flex-wrap items-center gap-2">
          <button
            disabled={uploadingDocument}
            onClick={(event) => {
              event.stopPropagation();
              onOpenUploadModal();
            }}
            className="documents-upload-trigger flex items-center gap-2 rounded-xl border border-[#303642] bg-[#111620] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#181e2a] disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Upload size={17} />
            <span className="hidden xl:inline">Upload</span>
          </button>

          <button
            disabled={folderSubmitting}
            onClick={(event) => {
              event.stopPropagation();
              onOpenNewFolderModal();
            }}
            className="flex items-center gap-2 rounded-xl border border-[#303642] bg-[#111620] px-3 py-2 text-sm font-semibold text-white transition-colors hover:bg-[#181e2a] disabled:cursor-not-allowed disabled:opacity-60"
          >
            <Plus size={17} />
            <span className="hidden xl:inline">New Folder</span>
          </button>

          <div className="ml-1 flex min-w-0 items-center gap-2 text-sm text-[#8c93a3]">
            <button
              onClick={(event) => {
                event.stopPropagation();
                onSetRootFolder();
              }}
              className="flex items-center gap-2 transition-colors hover:text-white"
            >
              <House size={17} />
              <span className="hidden lg:inline font-medium">Root</span>
            </button>
            {breadcrumbFolders.map((folder) => (
              <div key={folder.id} className="flex min-w-0 items-center gap-2">
                <span className="text-[#6f7686]">›</span>
                <button
                  onClick={(event) => {
                    event.stopPropagation();
                    onSetCurrentFolder(folder.id);
                  }}
                  className="truncate font-medium transition-colors hover:text-white"
                >
                  {folder.name}
                </button>
              </div>
            ))}
          </div>
        </div>

        <div className="flex w-full items-center gap-2 sm:gap-3 xl:w-auto">
          <div className="flex min-w-0 flex-1 items-center gap-3 rounded-xl border border-[#303642] bg-[#111620] px-4 py-2.5 text-[#8f95a3] xl:w-[420px] xl:flex-none">
            <Search size={18} />
            <input
              value={searchQuery}
              onChange={(event) => onSearchQueryChange(event.target.value)}
              placeholder="Search documents..."
              className="w-full bg-transparent text-sm text-white outline-none placeholder:text-[#8f95a3]"
            />
          </div>

          <div className="flex items-center rounded-xl border border-[#303642] bg-[#111620] p-1">
            <button
              onClick={(event) => {
                event.stopPropagation();
                onViewModeChange("list");
              }}
              className={`rounded-lg p-2 transition-colors ${
                viewMode === "list"
                  ? "bg-[#2f343f] text-white"
                  : "text-[#8f95a3] hover:text-white"
              }`}
            >
              <List size={18} />
            </button>
            <button
              onClick={(event) => {
                event.stopPropagation();
                onViewModeChange("grid");
              }}
              className={`rounded-lg p-2 transition-colors ${
                viewMode === "grid"
                  ? "bg-[#2f343f] text-white"
                  : "text-[#8f95a3] hover:text-white"
              }`}
            >
              <LayoutGrid size={18} />
            </button>
          </div>
        </div>
      </div>

      {hasVisibleItems ? (
        viewMode === "grid" ? (
          <div className="grid w-full gap-3 md:grid-cols-2 xl:grid-cols-3">
            {filteredFolders.map((folder) => (
              <article
                key={folder.id}
                onClick={() => onEnterFolder(folder.id)}
                className="group relative cursor-pointer rounded-2xl border border-[#262d3a] bg-[#171c27] p-4 transition-colors hover:bg-[#1b2230]"
              >
                <div className="mb-8 flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <Folder size={20} className="text-[#b8bec9]" />
                    <p className="text-[1.1rem] font-semibold text-white">{folder.name}</p>
                  </div>
                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpenMenuChange(
                        openMenu?.kind === "folder" && openMenu.id === folder.id
                          ? null
                          : { kind: "folder", id: folder.id },
                      );
                    }}
                    className={`rounded-lg p-1.5 text-[#9aa0ac] transition-all duration-150 hover:bg-[#161b26] hover:text-white ${
                      openMenu?.kind === "folder" && openMenu.id === folder.id
                        ? "opacity-100"
                        : "pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100"
                    }`}
                  >
                    <MoreVertical size={18} />
                  </button>
                </div>
                <p className="text-sm text-[#8f95a3]">{formatFolderFileCount(folder.id)}</p>

                {openMenu?.kind === "folder" && openMenu.id === folder.id && (
                  <div
                    className="absolute right-3 top-[58px] z-20 w-[230px] overflow-hidden rounded-xl border border-[#303642] bg-[#111620] shadow-xl"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <button
                      onClick={() => {
                        openCreateSubfolderDialog(folder);
                      }}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d]"
                    >
                      <FolderPlus size={16} />
                      <span>Create Subfolder</span>
                    </button>
                    <button
                      onClick={() => {
                        openRenameFolderDialog(folder);
                      }}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d]"
                    >
                      <Pencil size={16} />
                      <span>Rename</span>
                    </button>
                    <button
                      onClick={() => {
                        openDeleteDialog({
                          kind: "folder",
                          id: folder.id,
                          name: folder.name,
                        });
                      }}
                      className="flex w-full items-center gap-3 border-t border-[#262d3a] px-4 py-3 text-left text-sm text-[#ff6b6b] transition-colors hover:bg-[#1b212d]"
                    >
                      <Trash2 size={16} />
                      <span>Delete</span>
                    </button>
                  </div>
                )}
              </article>
            ))}

            {filteredDocuments.map((doc) => (
              <article
                key={doc.id}
                className="group relative rounded-2xl border border-[#262d3a] bg-[#171c27] p-4"
              >
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div className="flex min-w-0 flex-1 items-center gap-3">
                    <FileText size={19} className="flex-shrink-0 text-[#b8bec9]" />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-[1.1rem] font-semibold text-white">{doc.name}</p>
                      <div className="mt-1">
                        <DocumentStatusBadge status={doc.status} />
                      </div>
                    </div>
                  </div>
                  <button
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpenMenuChange(
                        openMenu?.kind === "document" && openMenu.id === doc.id
                          ? null
                          : { kind: "document", id: doc.id },
                      );
                    }}
                    className={`rounded-lg p-1.5 text-[#9aa0ac] transition-all duration-150 hover:bg-[#161b26] hover:text-white ${
                      openMenu?.kind === "document" && openMenu.id === doc.id
                        ? "opacity-100"
                        : "pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100"
                    }`}
                  >
                    <MoreVertical size={17} />
                  </button>
                </div>
                <p className="line-clamp-2 text-sm text-[#a8aebb]">
                  {getDocumentDescriptionLabel(doc.description)}
                </p>
                <p className="mt-4 text-sm text-[#a8aebb]">
                  {doc.pages} pages • {formatUploadedLabel(doc.uploadedAt)}
                </p>

                {openMenu?.kind === "document" && openMenu.id === doc.id && (
                  <div
                    className="absolute right-3 top-[58px] z-50 w-[240px] overflow-hidden rounded-xl border border-[#303642] bg-[#111620] shadow-xl"
                    onClick={(event) => event.stopPropagation()}
                  >
                    <button
                      onClick={() => onOpenDocumentDetails(doc.id)}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <Info size={16} />
                      <span>Details</span>
                    </button>
                    <button
                      onClick={() => onOpenDocumentChat(doc.id)}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <MessageSquare size={16} />
                      <span>Chat</span>
                    </button>
                    <button
                      onClick={() => onOpenMoveDocumentModal(doc.id)}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <Move size={16} />
                      <span>Move to Folder</span>
                    </button>
                    <button
                      onClick={() => onDownloadDocument(doc.id)}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <Download size={16} />
                      <span>Download</span>
                    </button>
                    <button
                      onClick={() => {
                        void onShareDocument(doc.id);
                      }}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <Share2 size={16} />
                      <span>Share</span>
                    </button>
                    <button
                      onClick={() => {
                        openDeleteDialog({
                          kind: "document",
                          id: doc.id,
                          name: doc.name,
                        });
                      }}
                      disabled={isDocumentProcessing(doc.status)}
                      className="flex w-full items-center gap-3 border-t border-[#262d3a] px-4 py-3 text-left text-sm text-[#ff6b6b] transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                    >
                      <Trash2 size={16} />
                      <span>Delete</span>
                    </button>
                  </div>
                )}
              </article>
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-transparent">
            <div className={isMobileViewport ? "min-w-0" : "min-w-[980px]"}>
              {filteredFolders.map((folder) => (
                <div
                  key={folder.id}
                  onClick={() => onEnterFolder(folder.id)}
                  className="relative flex cursor-pointer items-center justify-between border-b border-[#1e2430] px-2 py-5 transition-colors hover:bg-[#131925]"
                >
                  <div className="min-w-0 flex items-center gap-4">
                    <Folder
                      size={isMobileViewport ? 22 : 24}
                      className="text-[#b8bec9]"
                    />
                    <span className="truncate text-base font-semibold text-white sm:text-lg">
                      {folder.name}
                    </span>
                  </div>

                  <div className="ml-3 flex flex-shrink-0 items-center gap-3">
                    <span className="text-xs text-[#9198a6] sm:text-sm">
                      {formatFolderFileCount(folder.id)}
                    </span>
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpenMenuChange(
                          openMenu?.kind === "folder" && openMenu.id === folder.id
                            ? null
                            : { kind: "folder", id: folder.id },
                        );
                      }}
                      className="rounded-lg p-2 text-[#9aa0ac] transition-colors hover:bg-[#161b26] hover:text-white"
                    >
                      <MoreVertical size={18} />
                    </button>
                  </div>

                  {openMenu?.kind === "folder" && openMenu.id === folder.id && (
                    <div
                      className="absolute right-0 top-[56px] z-20 w-[230px] overflow-hidden rounded-xl border border-[#303642] bg-[#111620] shadow-xl"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <button
                        onClick={() => {
                          openCreateSubfolderDialog(folder);
                        }}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d]"
                      >
                        <FolderPlus size={16} />
                        <span>Create Subfolder</span>
                      </button>
                      <button
                        onClick={() => {
                          openRenameFolderDialog(folder);
                        }}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d]"
                      >
                        <Pencil size={16} />
                        <span>Rename</span>
                      </button>
                      <button
                        onClick={() => {
                          openDeleteDialog({
                            kind: "folder",
                            id: folder.id,
                            name: folder.name,
                          });
                        }}
                        className="flex w-full items-center gap-3 border-t border-[#262d3a] px-4 py-3 text-left text-sm text-[#ff6b6b] transition-colors hover:bg-[#1b212d]"
                      >
                        <Trash2 size={16} />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              ))}

              {filteredDocuments.map((doc) => (
                <div
                  key={doc.id}
                  className={`relative flex items-start justify-between border-b border-[#1e2430] px-2 py-4 sm:items-center sm:py-5 ${
                    openMenu?.kind === "document" && openMenu.id === doc.id ? "z-30" : ""
                  }`}
                >
                  <div className="min-w-0 flex-1 pr-3 sm:pr-6">
                    <div className="mb-2 flex min-w-0 items-center gap-4">
                      <FileText
                        size={isMobileViewport ? 21 : 24}
                        className="flex-shrink-0 text-[#b8bec9]"
                      />
                      <span className="min-w-0 flex-1 truncate text-base font-semibold text-white sm:text-lg">
                        {doc.name}
                      </span>
                      <div className="flex-shrink-0">
                        <DocumentStatusBadge status={doc.status} />
                      </div>
                    </div>
                    <p className="ml-8 line-clamp-1 text-xs text-[#a5acb8] sm:ml-10 sm:text-sm">
                      {getDocumentDescriptionLabel(doc.description)}
                    </p>
                  </div>

                  <div className="ml-2 flex flex-shrink-0 items-center gap-2 sm:gap-3">
                    {!isMobileViewport && (
                      <p className="text-sm text-[#a5acb8]">
                        {doc.pages} pages • {formatUploadedLabel(doc.uploadedAt)}
                      </p>
                    )}
                    <button
                      onClick={(event) => {
                        event.stopPropagation();
                        onOpenMenuChange(
                          openMenu?.kind === "document" && openMenu.id === doc.id
                            ? null
                            : { kind: "document", id: doc.id },
                        );
                      }}
                      className="rounded-lg p-2 text-[#9aa0ac] transition-colors hover:bg-[#161b26] hover:text-white"
                    >
                      <MoreVertical size={18} />
                    </button>
                  </div>

                  {openMenu?.kind === "document" && openMenu.id === doc.id && (
                    <div
                      className="absolute right-0 top-full z-50 mt-1 w-[220px] overflow-hidden rounded-xl border border-[#303642] bg-[#111620] shadow-xl"
                      onClick={(event) => event.stopPropagation()}
                    >
                      <button
                        onClick={() => onOpenDocumentDetails(doc.id)}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Info size={16} />
                        <span>Details</span>
                      </button>
                      <button
                        onClick={() => onOpenDocumentChat(doc.id)}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <MessageSquare size={16} />
                        <span>Chat</span>
                      </button>
                      <button
                        onClick={() => onOpenMoveDocumentModal(doc.id)}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Move size={16} />
                        <span>Move to Folder</span>
                      </button>
                      <button
                        onClick={() => onDownloadDocument(doc.id)}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Download size={16} />
                        <span>Download</span>
                      </button>
                      <button
                        onClick={() => {
                          void onShareDocument(doc.id);
                        }}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 px-4 py-3 text-left text-sm text-white transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Share2 size={16} />
                        <span>Share</span>
                      </button>
                      <button
                        onClick={() => {
                          openDeleteDialog({
                            kind: "document",
                            id: doc.id,
                            name: doc.name,
                          });
                        }}
                        disabled={isDocumentProcessing(doc.status)}
                        className="flex w-full items-center gap-3 border-t border-[#262d3a] px-4 py-3 text-left text-sm text-[#ff6b6b] transition-colors hover:bg-[#1b212d] disabled:cursor-not-allowed disabled:opacity-45"
                      >
                        <Trash2 size={16} />
                        <span>Delete</span>
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )
      ) : (
        <div className="flex min-h-[420px] flex-col items-center justify-center gap-5 text-center">
          <Folder size={56} className="text-[#3d4454]" />
          <p className="text-4xl font-semibold text-white">{emptyStateTitle}</p>
          <p className="max-w-[780px] text-base text-[#8f95a3]">{emptyStateDescription}</p>
        </div>
      )}

      <FolderNameModal
        open={folderDialog !== null}
        isMobileViewport={isMobileViewport}
        mode={folderDialog?.mode ?? "create-subfolder"}
        targetFolderName={folderDialog?.folderName ?? ""}
        nameValue={folderDialogName}
        descriptionValue={folderDialogDescription}
        submitting={folderDialogSubmitting}
        onClose={closeFolderDialog}
        onNameChange={setFolderDialogName}
        onDescriptionChange={setFolderDialogDescription}
        onSubmit={handleSubmitFolderDialog}
      />

      <DangerConfirmModal
        open={deleteDialog !== null}
        isMobileViewport={isMobileViewport}
        title={deleteDialogTitle}
        description={deleteDialogDescription}
        confirmLabel={deleteDialog?.kind === "folder" ? "Delete Folder" : "Delete Document"}
        submitting={deleteDialogSubmitting}
        onClose={closeDeleteDialog}
        onConfirm={handleConfirmDelete}
      />
    </section>
  );
}
