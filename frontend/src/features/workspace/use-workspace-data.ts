import { useCallback, useEffect, useMemo, useState } from "react";
import {
  listDocuments,
  listFolders,
  type WorkspaceDocument,
  type WorkspaceFolder,
} from "../../services/workspace-api";
import type { DocumentItem, FolderItem } from "./models";
import { mapDocumentFromApi, mapFolderFromApi } from "./models";
import { isDocumentProcessing, normalizeErrorMessage } from "./utils";

interface UseWorkspaceDataParams {
  searchQuery: string;
  pushToast: (title: string, options?: { tone?: "success" | "error" | "info"; description?: string; durationMs?: number }) => void;
}

export function useWorkspaceData({ searchQuery, pushToast }: UseWorkspaceDataParams) {
  const [folders, setFolders] = useState<FolderItem[]>([]);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [workspaceLoading, setWorkspaceLoading] = useState(true);
  const [currentFolderId, setCurrentFolderId] = useState<string | null>(null);

  const applyWorkspaceSnapshot = useCallback(
    (folderRows: WorkspaceFolder[], documentRows: WorkspaceDocument[]) => {
      setFolders(folderRows.map(mapFolderFromApi));
      setDocuments(documentRows.map(mapDocumentFromApi));
    },
    [],
  );

  const loadWorkspace = useCallback(async () => {
    setWorkspaceLoading(true);
    try {
      const [folderRows, documentRows] = await Promise.all([listFolders(), listDocuments()]);
      applyWorkspaceSnapshot(folderRows, documentRows);
    } catch (error) {
      pushToast("Workspace sync failed", {
        tone: "error",
        description: normalizeErrorMessage(error),
      });
    } finally {
      setWorkspaceLoading(false);
    }
  }, [applyWorkspaceSnapshot, pushToast]);

  const refreshWorkspaceSilently = useCallback(async () => {
    try {
      const [folderRows, documentRows] = await Promise.all([listFolders(), listDocuments()]);
      applyWorkspaceSnapshot(folderRows, documentRows);
    } catch {
      // Keep silent during background polling.
    }
  }, [applyWorkspaceSnapshot]);

  useEffect(() => {
    void loadWorkspace();
  }, [loadWorkspace]);

  const hasProcessingDocuments = useMemo(
    () => documents.some((doc) => isDocumentProcessing(doc.status)),
    [documents],
  );

  useEffect(() => {
    if (!hasProcessingDocuments) return;

    void refreshWorkspaceSilently();
    const pollTimer = window.setInterval(() => {
      void refreshWorkspaceSilently();
    }, 1500);

    return () => window.clearInterval(pollTimer);
  }, [hasProcessingDocuments, refreshWorkspaceSilently]);

  const foldersById = useMemo(
    () => new Map(folders.map((folder) => [folder.id, folder])),
    [folders],
  );

  const breadcrumbFolders = useMemo(() => {
    const chain: FolderItem[] = [];
    const visited = new Set<string>();
    let cursor = currentFolderId;

    while (cursor) {
      if (visited.has(cursor)) break;
      visited.add(cursor);
      const folder = foldersById.get(cursor);
      if (!folder) break;
      chain.unshift(folder);
      cursor = folder.parentId;
    }

    return chain;
  }, [currentFolderId, foldersById]);

  useEffect(() => {
    if (!currentFolderId) return;
    if (!foldersById.has(currentFolderId)) {
      setCurrentFolderId(null);
    }
  }, [currentFolderId, foldersById]);

  const visibleFolders = useMemo(
    () => folders.filter((folder) => folder.parentId === currentFolderId),
    [folders, currentFolderId],
  );

  const visibleDocuments = useMemo(
    () => documents.filter((document) => document.folderId === currentFolderId),
    [documents, currentFolderId],
  );

  const filteredFolders = useMemo(
    () =>
      visibleFolders.filter((folder) =>
        folder.name.toLowerCase().includes(searchQuery.trim().toLowerCase()),
      ),
    [visibleFolders, searchQuery],
  );

  const filteredDocuments = useMemo(
    () =>
      visibleDocuments.filter((doc) => {
        const q = searchQuery.trim().toLowerCase();
        if (!q) return true;
        return doc.name.toLowerCase().includes(q) || doc.description.toLowerCase().includes(q);
      }),
    [visibleDocuments, searchQuery],
  );

  const folderDocumentCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const document of documents) {
      if (!document.folderId) continue;
      counts.set(document.folderId, (counts.get(document.folderId) ?? 0) + 1);
    }
    return counts;
  }, [documents]);

  const formatFolderFileCount = useCallback(
    (folderId: string): string => {
      const count = folderDocumentCounts.get(folderId) ?? 0;
      return `${count} ${count === 1 ? "file" : "files"}`;
    },
    [folderDocumentCounts],
  );

  const moveFolderRows = useMemo(() => {
    const children = new Map<string | null, FolderItem[]>();
    for (const folder of folders) {
      const list = children.get(folder.parentId) ?? [];
      list.push(folder);
      children.set(folder.parentId, list);
    }
    for (const list of children.values()) {
      list.sort((a, b) => a.name.localeCompare(b.name));
    }

    const rows: Array<{ id: string; name: string; depth: number }> = [];
    const walk = (parentId: string | null, depth: number) => {
      const list = children.get(parentId) ?? [];
      for (const folder of list) {
        rows.push({ id: folder.id, name: folder.name, depth });
        walk(folder.id, depth + 1);
      }
    };
    walk(null, 0);
    return rows;
  }, [folders]);

  const hasVisibleItems = filteredFolders.length > 0 || filteredDocuments.length > 0;
  const emptyStateTitle = searchQuery.trim() ? "No matches found" : "No items here yet";
  const emptyStateDescription = searchQuery.trim()
    ? "Try another keyword in this folder."
    : currentFolderId
      ? "This folder is empty. Create a subfolder or upload files to get started."
      : "Upload a document or create a folder to get started.";

  const documentsById = useMemo(
    () => new Map(documents.map((doc) => [doc.id, doc])),
    [documents],
  );

  return {
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
    refreshWorkspaceSilently,
  };
}
