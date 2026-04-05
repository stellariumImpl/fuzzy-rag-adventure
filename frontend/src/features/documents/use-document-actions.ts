import { useCallback, useMemo, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";
import type { NavigateFunction } from "react-router-dom";
import type { ContextMenu } from "../../store/ui-store";
import {
  WORKSPACE_API_BASE_URL,
  autofillDocumentDescription,
  createMarkdownDocument as createMarkdownDocumentRecord,
  createFolder,
  deleteDocument,
  deleteFolder,
  updateDocument,
  updateFolder,
  uploadDocument,
} from "../../services/workspace-api";
import type { DocumentItem, FolderItem } from "../workspace/models";
import { mapDocumentFromApi, mapFolderFromApi } from "../workspace/models";
import {
  getDocumentStatusLabel,
  isDocumentReady,
  isDocumentProcessing,
  isPlaceholderDocumentDescription,
  normalizeErrorMessage,
} from "../workspace/utils";
import { buildProcessingToastMessage } from "../chat/chat-view";

type PushToast = (
  title: string,
  options?: {
    tone?: "success" | "error" | "info";
    description?: string;
    durationMs?: number;
  },
) => void;

type UploadTarget = "current-folder" | "root";

interface UseDocumentActionsParams {
  documents: DocumentItem[];
  folders: FolderItem[];
  currentFolderId: string | null;
  foldersById: Map<string, FolderItem>;
  documentsById: Map<string, DocumentItem>;
  setCurrentFolderId: (value: string | null) => void;
  setFolders: Dispatch<SetStateAction<FolderItem[]>>;
  setDocuments: Dispatch<SetStateAction<DocumentItem[]>>;
  setSelectedDocs: Dispatch<SetStateAction<string[]>>;
  setChatStarted: (value: boolean) => void;
  toggleDocumentSelection: (id: string) => void;
  setOpenMenu: (value: ContextMenu) => void;
  setShowNewFolderModal: (value: boolean) => void;
  setShowUploadModal: (value: boolean) => void;
  pushToast: PushToast;
  navigate: NavigateFunction;
}

export function useDocumentActions({
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
}: UseDocumentActionsParams) {
  const [folderSubmitting, setFolderSubmitting] = useState(false);
  const [uploadingDocument, setUploadingDocument] = useState(false);

  const [newFolderName, setNewFolderName] = useState("");
  const [newFolderDescription, setNewFolderDescription] = useState("");
  const uploadInputRef = useRef<HTMLInputElement | null>(null);
  const pendingUploadTargetRef = useRef<UploadTarget>("current-folder");

  const [uploadDragOver, setUploadDragOver] = useState(false);
  const [quickUploadDragOver, setQuickUploadDragOver] = useState(false);
  const [previewDocumentId, setPreviewDocumentId] = useState<string | null>(null);
  const [autofillingDescriptionDocId, setAutofillingDescriptionDocId] =
    useState<string | null>(null);
  const [movingDocumentId, setMovingDocumentId] = useState<string | null>(null);
  const [moveDestinationFolderId, setMoveDestinationFolderId] =
    useState<string | null>(null);
  const [moveModalSubmitting, setMoveModalSubmitting] = useState(false);

  const ensureDocumentActionAllowed = useCallback(
    (docId: string, actionLabel: string): boolean => {
      const doc = documentsById.get(docId);
      if (!doc) return false;
      if (!isDocumentProcessing(doc.status)) return true;

      const statusLabel = getDocumentStatusLabel(doc.status).toLowerCase();
      pushToast("Document is processing", {
        tone: "info",
        durationMs: 2600,
        description: `"${doc.name}" is ${statusLabel}. ${actionLabel} is temporarily unavailable.`,
      });
      return false;
    },
    [documentsById, pushToast],
  );

  const previewDocument = useMemo(
    () => documents.find((doc) => doc.id === previewDocumentId) ?? null,
    [documents, previewDocumentId],
  );

  const movingDocument = useMemo(
    () => documents.find((doc) => doc.id === movingDocumentId) ?? null,
    [documents, movingDocumentId],
  );

  const previewDocumentUrl = previewDocument
    ? `${WORKSPACE_API_BASE_URL}/documents/${encodeURIComponent(previewDocument.id)}/preview`
    : "";

  const previewDocumentDescription = previewDocument ? previewDocument.description.trim() : "";

  const selectedMoveDestinationLabel = moveDestinationFolderId
    ? (foldersById.get(moveDestinationFolderId)?.name ?? "Unknown folder")
    : "Root Directory";

  const handleToggleDocumentSelectionWithGuard = useCallback(
    (doc: DocumentItem) => {
      if (isDocumentReady(doc.status)) {
        toggleDocumentSelection(doc.id);
        return;
      }

      if (isDocumentProcessing(doc.status)) {
        pushToast("Document is processing", {
          tone: "info",
          durationMs: 2600,
          description: buildProcessingToastMessage(doc),
        });
        return;
      }

      pushToast("Document is not searchable yet", {
        tone: "info",
        durationMs: 2800,
        description: `"${doc.name}" is ${getDocumentStatusLabel(doc.status).toLowerCase()} and cannot be used for Q&A yet.`,
      });
    },
    [pushToast, toggleDocumentSelection],
  );

  const handleCreateFolder = useCallback(async () => {
    if (!newFolderName.trim()) return;

    setFolderSubmitting(true);
    try {
      const created = await createFolder({
        name: newFolderName.trim(),
        description: newFolderDescription.trim() || "Empty",
        parent_id: currentFolderId,
      });
      setFolders((prev) => [mapFolderFromApi(created), ...prev]);
      setNewFolderName("");
      setNewFolderDescription("");
      setShowNewFolderModal(false);
      pushToast("Folder created", {
        tone: "success",
        description: `"${created.name}" is now available.`,
      });
    } catch (error) {
      pushToast("Create folder failed", {
        tone: "error",
        description: normalizeErrorMessage(error),
      });
    } finally {
      setFolderSubmitting(false);
    }
  }, [
    currentFolderId,
    newFolderDescription,
    newFolderName,
    pushToast,
    setFolders,
    setShowNewFolderModal,
  ]);

  const handleShareDocument = useCallback(
    async (id: string) => {
      if (!ensureDocumentActionAllowed(id, "Share")) {
        setOpenMenu(null);
        return;
      }
      const href = `${WORKSPACE_API_BASE_URL}/documents/${encodeURIComponent(id)}/preview`;
      try {
        await navigator.clipboard.writeText(href);
        pushToast("Link copied", {
          tone: "success",
          description: "Preview link copied to clipboard.",
        });
      } catch {
        pushToast("Share failed", {
          tone: "error",
          description: "Clipboard access is unavailable in this browser.",
        });
      }
      setOpenMenu(null);
    },
    [ensureDocumentActionAllowed, pushToast, setOpenMenu],
  );

  const handleUploadPickedFile = useCallback(
    async (file: File, target: UploadTarget = pendingUploadTargetRef.current) => {
      const destinationFolderId = target === "root" ? null : currentFolderId;
      setUploadingDocument(true);
      try {
        const created = await uploadDocument(file, destinationFolderId);
        const mapped = mapDocumentFromApi(created);
        setDocuments((prev) => [mapped, ...prev]);
        setSelectedDocs((prev) => (prev.includes(mapped.id) ? prev : [...prev, mapped.id]));
        setShowUploadModal(false);
        setUploadDragOver(false);
        setQuickUploadDragOver(false);
        if (isDocumentProcessing(mapped.status)) {
          pushToast("Upload complete", {
            tone: "success",
            description: `"${mapped.name}" uploaded. Parsing started in background.`,
          });
        } else {
          pushToast("Upload complete", {
            tone: "success",
            description: `"${mapped.name}" was uploaded successfully.`,
          });
        }
      } catch (error) {
        pushToast("Upload failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
      } finally {
        setUploadingDocument(false);
      }
    },
    [currentFolderId, pushToast, setDocuments, setSelectedDocs, setShowUploadModal],
  );

  const handleCreateMarkdownDocument = useCallback(
    async (
      content: string,
      sourceName?: string,
      target: UploadTarget = pendingUploadTargetRef.current,
    ) => {
      const normalizedContent = content.trim();
      if (!normalizedContent) {
        pushToast("Create failed", {
          tone: "error",
          description: "Markdown content cannot be empty.",
        });
        return;
      }

      const destinationFolderId = target === "root" ? null : currentFolderId;
      setUploadingDocument(true);
      try {
        const created = await createMarkdownDocumentRecord({
          content: normalizedContent,
          source_name: sourceName?.trim() || undefined,
          folder_id: destinationFolderId,
        });
        const mapped = mapDocumentFromApi(created);
        setDocuments((prev) => [mapped, ...prev]);
        setSelectedDocs((prev) => (prev.includes(mapped.id) ? prev : [...prev, mapped.id]));
        setShowUploadModal(false);
        setUploadDragOver(false);
        setQuickUploadDragOver(false);
        pushToast("Markdown created", {
          tone: "success",
          description: `"${mapped.name}" created and parsing started.`,
        });
      } catch (error) {
        pushToast("Create markdown failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
      } finally {
        setUploadingDocument(false);
      }
    },
    [currentFolderId, pushToast, setDocuments, setSelectedDocs, setShowUploadModal],
  );

  const openUploadFilePicker = useCallback((target: UploadTarget = "current-folder") => {
    if (uploadingDocument) return;
    pendingUploadTargetRef.current = target;
    uploadInputRef.current?.click();
  }, [uploadingDocument]);

  const handleUploadInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file) return;
      void handleUploadPickedFile(file, pendingUploadTargetRef.current);
    },
    [handleUploadPickedFile],
  );

  const removeDocument = useCallback(
    async (id: string): Promise<boolean> => {
      if (!ensureDocumentActionAllowed(id, "Delete")) {
        setOpenMenu(null);
        return false;
      }
      try {
        const docName = documents.find((document) => document.id === id)?.name ?? "Document";
        await deleteDocument(id);
        setDocuments((prev) => prev.filter((doc) => doc.id !== id));
        setSelectedDocs((prev) => prev.filter((docId) => docId !== id));
        pushToast("Document deleted", {
          tone: "success",
          description: `"${docName}" has been removed.`,
        });
        return true;
      } catch (error) {
        pushToast("Delete document failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
        return false;
      } finally {
        setOpenMenu(null);
      }
    },
    [documents, ensureDocumentActionAllowed, pushToast, setDocuments, setOpenMenu, setSelectedDocs],
  );

  const removeFolder = useCallback(
    async (id: string): Promise<boolean> => {
      try {
        const folderName = folders.find((folder) => folder.id === id)?.name ?? "Folder";
        await deleteFolder(id);
        setFolders((prev) => prev.filter((folder) => folder.id !== id));
        if (currentFolderId === id) {
          setCurrentFolderId(null);
        }
        pushToast("Folder deleted", {
          tone: "success",
          description: `"${folderName}" has been removed.`,
        });
        return true;
      } catch (error) {
        pushToast("Delete folder failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
        return false;
      } finally {
        setOpenMenu(null);
      }
    },
    [currentFolderId, folders, pushToast, setCurrentFolderId, setFolders, setOpenMenu],
  );

  const createSubfolder = useCallback(
    async (
      parentId: string,
      nextName: string,
      nextDescription?: string,
    ): Promise<boolean> => {
      const trimmedName = nextName.trim();
      if (!trimmedName) return false;

      try {
        const created = await createFolder({
          name: trimmedName,
          description: nextDescription?.trim() || "Empty",
          parent_id: parentId,
        });
        setFolders((prev) => [mapFolderFromApi(created), ...prev]);
        pushToast("Subfolder created", {
          tone: "success",
          description: `"${created.name}" is ready.`,
        });
        return true;
      } catch (error) {
        pushToast("Create subfolder failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
        return false;
      } finally {
        setOpenMenu(null);
      }
    },
    [pushToast, setFolders, setOpenMenu],
  );

  const renameFolder = useCallback(
    async (id: string, nextName: string): Promise<boolean> => {
      const trimmedName = nextName.trim();
      if (!trimmedName) return false;

      try {
        const updated = await updateFolder(id, { name: trimmedName });
        const mapped = mapFolderFromApi(updated);
        setFolders((prev) => prev.map((folder) => (folder.id === id ? mapped : folder)));
        pushToast("Folder renamed", {
          tone: "success",
          description: `Now named "${mapped.name}".`,
        });
        return true;
      } catch (error) {
        pushToast("Rename folder failed", {
          tone: "error",
          description: normalizeErrorMessage(error),
        });
        return false;
      } finally {
        setOpenMenu(null);
      }
    },
    [pushToast, setFolders, setOpenMenu],
  );

  const handleDownloadDocument = useCallback(
    (id: string) => {
      if (!ensureDocumentActionAllowed(id, "Download")) {
        setOpenMenu(null);
        return;
      }
      const href = `${WORKSPACE_API_BASE_URL}/documents/${encodeURIComponent(id)}/download`;
      window.open(href, "_blank", "noopener,noreferrer");
      setOpenMenu(null);
    },
    [ensureDocumentActionAllowed, setOpenMenu],
  );

  const openDocumentChat = useCallback(
    (docId: string) => {
      if (!ensureDocumentActionAllowed(docId, "Chat")) {
        setOpenMenu(null);
        return;
      }
      setSelectedDocs([docId]);
      setChatStarted(false);
      navigate("/chat");
      setOpenMenu(null);
    },
    [ensureDocumentActionAllowed, navigate, setOpenMenu, setSelectedDocs, setChatStarted],
  );

  const openDocumentDetails = useCallback(
    (docId: string) => {
      if (!ensureDocumentActionAllowed(docId, "Preview")) {
        setOpenMenu(null);
        return;
      }
      setPreviewDocumentId(docId);
      const target = documents.find((doc) => doc.id === docId);
      if (
        target &&
        isPlaceholderDocumentDescription(target.description) &&
        autofillingDescriptionDocId !== docId
      ) {
        setAutofillingDescriptionDocId(docId);
        void autofillDocumentDescription(docId)
          .then((updated) => {
            const mapped = mapDocumentFromApi(updated);
            setDocuments((prev) => prev.map((doc) => (doc.id === mapped.id ? mapped : doc)));
          })
          .catch(() => {
            // Keep existing text if auto-fill fails.
          })
          .finally(() => {
            setAutofillingDescriptionDocId((current) => (current === docId ? null : current));
          });
      }
      setOpenMenu(null);
    },
    [
      autofillingDescriptionDocId,
      documents,
      ensureDocumentActionAllowed,
      setDocuments,
      setOpenMenu,
    ],
  );

  const openMoveDocumentModal = useCallback(
    (docId: string) => {
      if (!ensureDocumentActionAllowed(docId, "Move")) {
        setOpenMenu(null);
        return;
      }
      const doc = documents.find((item) => item.id === docId);
      if (!doc) return;
      setMovingDocumentId(doc.id);
      setMoveDestinationFolderId(doc.folderId);
      setOpenMenu(null);
    },
    [documents, ensureDocumentActionAllowed, setOpenMenu],
  );

  const closeMoveDocumentModal = useCallback(() => {
    setMovingDocumentId(null);
    setMoveDestinationFolderId(null);
  }, []);

  const handleConfirmMoveDocument = useCallback(async () => {
    if (!movingDocument) return;

    if (movingDocument.folderId === moveDestinationFolderId) {
      pushToast("No destination change", {
        tone: "info",
        description: "The file is already in this folder.",
        durationMs: 2600,
      });
      closeMoveDocumentModal();
      return;
    }

    setMoveModalSubmitting(true);
    try {
      const updated = await updateDocument(movingDocument.id, {
        folder_id: moveDestinationFolderId,
      });
      const mapped = mapDocumentFromApi(updated);
      setDocuments((previous) => previous.map((doc) => (doc.id === mapped.id ? mapped : doc)));
      closeMoveDocumentModal();
      pushToast("File moved", {
        tone: "success",
        description: `"${mapped.name}" moved to "${selectedMoveDestinationLabel}".`,
      });
    } catch (error) {
      pushToast("Move failed", {
        tone: "error",
        description: normalizeErrorMessage(error),
      });
    } finally {
      setMoveModalSubmitting(false);
    }
  }, [
    closeMoveDocumentModal,
    moveDestinationFolderId,
    movingDocument,
    pushToast,
    selectedMoveDestinationLabel,
    setDocuments,
  ]);

  const enterFolder = useCallback(
    (folderId: string) => {
      setCurrentFolderId(folderId);
      setOpenMenu(null);
    },
    [setCurrentFolderId, setOpenMenu],
  );

  const closeNewFolderModal = useCallback(() => {
    setShowNewFolderModal(false);
    setNewFolderName("");
    setNewFolderDescription("");
  }, [setShowNewFolderModal]);

  const closeUploadModal = useCallback(() => {
    setShowUploadModal(false);
    setUploadDragOver(false);
  }, [setShowUploadModal]);

  return {
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
    previewDocumentId,
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
  };
}
