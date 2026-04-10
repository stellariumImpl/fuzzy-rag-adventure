import { useEffect, useMemo, useState } from "react";
import { X } from "lucide-react";
import {
  getDocumentPipeline,
  type DocumentPipelineChunkItem,
  type DocumentPipelineDetails,
} from "../../services/workspace-api";
import type { DocumentItem } from "../workspace/models";
import {
  getDocumentDescriptionLabel,
  getDocumentStatusLabel,
  normalizeErrorMessage,
} from "../workspace/utils";

interface PreviewModalProps {
  previewDocument: DocumentItem | null;
  previewDocumentUrl: string;
  previewDocumentDescription: string;
  autofillingDescriptionDocId: string | null;
  isMobileViewport: boolean;
  formatUploadDate: (uploadedAt: string) => string;
  onClose: () => void;
}

const PIPELINE_PAGE_SIZE = 12;

function stringifyArtifactValue(value: unknown): string {
  if (value == null) return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function PreviewModal({
  previewDocument,
  previewDocumentUrl,
  previewDocumentDescription,
  autofillingDescriptionDocId,
  isMobileViewport,
  formatUploadDate,
  onClose,
}: PreviewModalProps) {
  const [pipelineDetails, setPipelineDetails] = useState<DocumentPipelineDetails | null>(null);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [pipelinePage, setPipelinePage] = useState(1);
  const [selectedChunk, setSelectedChunk] = useState<DocumentPipelineChunkItem | null>(null);

  useEffect(() => {
    setPipelinePage(1);
    setSelectedChunk(null);
  }, [previewDocument?.id]);

  useEffect(() => {
    if (!previewDocument) {
      setPipelineDetails(null);
      setPipelineError(null);
      setPipelineLoading(false);
      return;
    }

    const status = (previewDocument.status || "").trim().toLowerCase();
    if (status === "pending" || status === "parsing" || status === "chunking" || status === "indexing") {
      setPipelineDetails(null);
      setPipelineError(null);
      setPipelineLoading(false);
      return;
    }

    let canceled = false;
    setPipelineLoading(true);
    setPipelineError(null);

    getDocumentPipeline(previewDocument.id, {
      page: pipelinePage,
      pageSize: PIPELINE_PAGE_SIZE,
    })
      .then((payload) => {
        if (canceled) return;
        setPipelineDetails(payload);
      })
      .catch((error) => {
        if (canceled) return;
        setPipelineError(normalizeErrorMessage(error));
      })
      .finally(() => {
        if (canceled) return;
        setPipelineLoading(false);
      });

    return () => {
      canceled = true;
    };
  }, [pipelinePage, previewDocument]);

  const artifactEntries = useMemo(() => {
    const artifacts = pipelineDetails?.parse.artifacts || {};
    return Object.entries(artifacts);
  }, [pipelineDetails]);

  const totalChunkPages = useMemo(() => {
    if (!pipelineDetails) return 1;
    const pageSize = Math.max(1, pipelineDetails.chunk.page_size || PIPELINE_PAGE_SIZE);
    return Math.max(1, Math.ceil((pipelineDetails.chunk.total || 0) / pageSize));
  }, [pipelineDetails]);

  const canPrevChunkPage = pipelinePage > 1;
  const canNextChunkPage = pipelinePage < totalChunkPages;
  const markdownFiles = pipelineDetails?.parse.markdown_files ?? [];
  const generatedMarkdown = pipelineDetails?.parse.generated_markdown ?? [];
  const imageFiles = pipelineDetails?.parse.image_files ?? [];
  const tablePreview = pipelineDetails?.parse.table_preview ?? [];

  if (!previewDocument) {
    return null;
  }

  return (
    <div
      className={`absolute inset-0 z-50 flex bg-black/70 ${
        isMobileViewport ? "items-stretch justify-stretch p-0" : "items-center justify-center"
      }`}
      onClick={onClose}
    >
      <div
        className={`relative overflow-hidden bg-[#0a1019] ${
          isMobileViewport
            ? "h-full w-full"
            : "h-[88vh] w-[min(1460px,96vw)] rounded-[20px] border border-[#2a3343] shadow-[0_30px_80px_rgba(0,0,0,0.56)]"
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <button
          onClick={onClose}
          className={`preview-close-btn absolute z-20 flex h-10 w-10 items-center justify-center rounded-lg border border-[#5a6172] bg-[#0c111c] text-[#d0d6e2] transition-colors hover:bg-[#151b28] hover:text-white ${
            isMobileViewport ? "right-3 top-3" : "right-5 top-5"
          }`}
        >
          <X size={22} />
        </button>
        <div className={`flex h-full ${isMobileViewport ? "flex-col" : ""}`}>
          <div
            className={`relative min-w-0 flex-1 bg-[#151922] ${
              isMobileViewport ? "min-h-0 border-b border-[#263042]" : "border-r border-[#263042]"
            }`}
          >
            <iframe title={previewDocument.name} src={previewDocumentUrl} className="h-full w-full bg-white" />
          </div>
          <aside
            className={`${
              isMobileViewport ? "h-[42vh] w-full px-5 py-5" : "w-[460px] px-7 py-8"
            } overflow-y-auto bg-[#0a1019]`}
          >
            <h3 className="break-words pr-12 text-2xl font-semibold leading-snug text-white">
              {previewDocument.name}
            </h3>
            <div className="mt-5 space-y-4">
              <div className="rounded-lg border border-[#2d3648] bg-[#0f1623] p-3 text-sm text-[#b8c2d3]">
                <p>
                  Status:
                  <span className="ml-2 font-semibold text-white">
                    {getDocumentStatusLabel(previewDocument.status)}
                  </span>
                </p>
                <p className="mt-1">
                  Pages:
                  <span className="ml-2 font-semibold text-white">{previewDocument.pages}</span>
                </p>
                <p className="mt-1">
                  Uploaded:
                  <span className="ml-2 font-semibold text-white">
                    {formatUploadDate(previewDocument.uploadedAt)}
                  </span>
                </p>
              </div>

              <div>
                <h4 className="text-base font-semibold text-white">Description</h4>
                <p className="mt-2 text-sm leading-6 text-[#b6bfcc]">
                  {autofillingDescriptionDocId === previewDocument.id
                    ? "Generating description..."
                    : getDocumentDescriptionLabel(previewDocumentDescription, previewDocument.name)}
                </p>
              </div>

              <div className="rounded-lg border border-[#2d3648] bg-[#0f1623] p-3 text-sm text-[#9ea8b9]">
                <h4 className="text-base font-semibold text-white">Parsing details</h4>

                {pipelineLoading && (
                  <p className="mt-2 text-[#b6bfcc]">Loading parsing details...</p>
                )}

                {!pipelineLoading && pipelineError && (
                  <p className="mt-2 break-words text-[#f2a7a7]">Failed to load details: {pipelineError}</p>
                )}

                {!pipelineLoading && !pipelineError && !pipelineDetails && (
                  <p className="mt-2 text-[#b6bfcc]">
                    Parsing details will appear after indexing is complete.
                  </p>
                )}

                {!pipelineLoading && !pipelineError && pipelineDetails && (
                  <div className="mt-3 space-y-4">
                    <div className="rounded-md border border-[#2d3648] bg-[#111927] p-3 text-xs leading-5 text-[#b6bfcc]">
                      <p>
                        Parse provider:
                        <span className="ml-2 font-semibold text-white">
                          {pipelineDetails.parse.provider || "-"}
                        </span>
                      </p>
                      <p className="mt-1">
                        Parse summary:
                        <span className="ml-2 text-white">{pipelineDetails.parse.summary || "-"}</span>
                      </p>
                      <p className="mt-1">
                        Parse updated:
                        <span className="ml-2 text-white">{pipelineDetails.parse.updated_at || "-"}</span>
                      </p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Text chars</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.parse.text_chars}</p>
                      </div>
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Images</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.parse.image_count}</p>
                      </div>
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Tables</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.parse.table_count}</p>
                      </div>
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Chunks</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.chunk.total}</p>
                      </div>
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Embedded</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.index.embedded_chunks}</p>
                      </div>
                      <div className="rounded-md border border-[#2d3648] bg-[#111927] p-2">
                        <p className="text-[#8f99ab]">Missing vectors</p>
                        <p className="mt-1 font-semibold text-white">{pipelineDetails.index.missing_chunks}</p>
                      </div>
                    </div>

                    {(markdownFiles.length > 0 || generatedMarkdown.length > 0) && (
                      <details className="rounded-md border border-[#2d3648] bg-[#111927] p-3">
                        <summary className="cursor-pointer list-none text-xs font-semibold text-white">
                          Markdown outputs ({markdownFiles.length + generatedMarkdown.length})
                        </summary>
                        <div className="mt-2 space-y-2 text-xs">
                          {markdownFiles.map((item) => (
                            <div key={`md-${item.path}`} className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                              <p className="font-semibold text-[#d8deea]">{item.name}</p>
                              <p className="mt-1 break-words text-[#9ba7bc]">{item.path}</p>
                              <p className="mt-1 text-[#b6bfcc]">{item.chars} chars</p>
                            </div>
                          ))}
                          {generatedMarkdown.map((item) => (
                            <div key={`gen-${item.path}`} className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                              <p className="font-semibold text-[#d8deea]">{item.name}</p>
                              <p className="mt-1 break-words text-[#9ba7bc]">{item.path}</p>
                              <p className="mt-1 text-[#b6bfcc]">{item.chars} chars</p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {tablePreview.length > 0 && (
                      <details className="rounded-md border border-[#2d3648] bg-[#111927] p-3">
                        <summary className="cursor-pointer list-none text-xs font-semibold text-white">
                          Table preview ({tablePreview.length})
                        </summary>
                        <div className="mt-2 space-y-2 text-xs">
                          {tablePreview.map((row, idx) => (
                            <div key={`table-${idx}`} className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                              <pre className="whitespace-pre-wrap break-words text-[#b6bfcc]">
                                {stringifyArtifactValue(row)}
                              </pre>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {imageFiles.length > 0 && (
                      <details className="rounded-md border border-[#2d3648] bg-[#111927] p-3">
                        <summary className="cursor-pointer list-none text-xs font-semibold text-white">
                          Parsed images ({imageFiles.length})
                        </summary>
                        <div className="mt-2 space-y-2 text-xs">
                          {imageFiles.map((img) => (
                            <div key={img.path} className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                              <p className="font-semibold text-[#d8deea]">{img.name}</p>
                              <p className="mt-1 break-words text-[#9ba7bc]">{img.path}</p>
                              <p className="mt-1 text-[#b6bfcc]">{img.size_bytes} bytes</p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    {artifactEntries.length > 0 && (
                      <details className="rounded-md border border-[#2d3648] bg-[#111927] p-3">
                        <summary className="cursor-pointer list-none text-xs font-semibold text-white">
                          Parse artifacts ({artifactEntries.length})
                        </summary>
                        <div className="mt-2 space-y-2 text-xs">
                          {artifactEntries.map(([key, value]) => (
                            <div key={key} className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                              <p className="font-semibold text-[#d8deea]">{key}</p>
                              <pre className="mt-1 whitespace-pre-wrap break-words text-[#b6bfcc]">
                                {stringifyArtifactValue(value)}
                              </pre>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}

                    <div>
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <h5 className="text-sm font-semibold text-white">Parsed chunks</h5>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={() => setPipelinePage((prev) => Math.max(1, prev - 1))}
                            disabled={!canPrevChunkPage}
                            className="rounded border border-[#3a455a] px-2 py-1 text-xs text-[#d5dbe8] disabled:opacity-40"
                          >
                            Prev
                          </button>
                          <span className="text-xs text-[#a9b3c5]">
                            {pipelinePage}/{totalChunkPages}
                          </span>
                          <button
                            type="button"
                            onClick={() => setPipelinePage((prev) => Math.min(totalChunkPages, prev + 1))}
                            disabled={!canNextChunkPage}
                            className="rounded border border-[#3a455a] px-2 py-1 text-xs text-[#d5dbe8] disabled:opacity-40"
                          >
                            Next
                          </button>
                        </div>
                      </div>

                      <div className="space-y-2">
                        {pipelineDetails.chunk.items.length === 0 && (
                          <p className="rounded-md border border-[#2d3648] bg-[#111927] p-2 text-xs text-[#b6bfcc]">
                            No chunks found for this page.
                          </p>
                        )}
                        {pipelineDetails.chunk.items.map((chunk) => (
                          <button
                            key={chunk.chunk_id}
                            type="button"
                            onClick={() => setSelectedChunk(chunk)}
                            className="w-full rounded-md border border-[#2d3648] bg-[#111927] p-2 text-left transition-colors hover:border-[#4a5b79] hover:bg-[#182234]"
                          >
                            <div className="text-xs text-[#c8d0de]">
                              <span className="font-semibold text-white">#{chunk.chunk_order}</span>
                              <span className="ml-2">{chunk.chunk_level}</span>
                              <span className="ml-2 text-[#9aa5b8]">
                                p.{chunk.page_start ?? "-"}-{chunk.page_end ?? "-"}
                              </span>
                              <span className="ml-2 text-[#9aa5b8]">{chunk.char_count} chars</span>
                            </div>
                            <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-[#b6bfcc]">
                              {chunk.preview || chunk.text}
                            </p>
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </aside>
        </div>

        {selectedChunk && (
          <div
            className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 p-4"
            onClick={() => setSelectedChunk(null)}
          >
            <div
              className="max-h-[84vh] w-[min(880px,94vw)] overflow-y-auto rounded-xl border border-[#2d3648] bg-[#0e1522] p-5"
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <h4 className="text-lg font-semibold text-white">
                    Chunk #{selectedChunk.chunk_order} details
                  </h4>
                  <p className="mt-1 text-xs text-[#9ea8b9]">{selectedChunk.chunk_level}</p>
                </div>
                <button
                  type="button"
                  onClick={() => setSelectedChunk(null)}
                  className="rounded border border-[#3a455a] px-2 py-1 text-xs text-[#d5dbe8]"
                >
                  Close
                </button>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-2 text-xs">
                <div className="rounded border border-[#2d3648] bg-[#111927] p-2">
                  <p className="text-[#9aa5b8]">Chunk ID</p>
                  <p className="mt-1 break-words text-white">{selectedChunk.chunk_id}</p>
                </div>
                <div className="rounded border border-[#2d3648] bg-[#111927] p-2">
                  <p className="text-[#9aa5b8]">Chunk index</p>
                  <p className="mt-1 text-white">{selectedChunk.chunk_index ?? "-"}</p>
                </div>
                <div className="rounded border border-[#2d3648] bg-[#111927] p-2">
                  <p className="text-[#9aa5b8]">Char count</p>
                  <p className="mt-1 text-white">{selectedChunk.char_count}</p>
                </div>
                <div className="rounded border border-[#2d3648] bg-[#111927] p-2">
                  <p className="text-[#9aa5b8]">Page range</p>
                  <p className="mt-1 text-white">
                    {selectedChunk.page_start ?? "-"} - {selectedChunk.page_end ?? "-"}
                  </p>
                </div>
              </div>

              <div className="mt-4 rounded border border-[#2d3648] bg-[#111927] p-3 text-xs">
                <p className="font-semibold text-white">Embedding index</p>
                <div className="mt-2 grid grid-cols-2 gap-2">
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Provider</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.provider || "-"}
                    </p>
                  </div>
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Collection</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.collection_name || "-"}
                    </p>
                  </div>
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Point ID</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.point_id || "not found"}
                    </p>
                  </div>
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Dense vector type</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.dense_vector || "-"}
                    </p>
                  </div>
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Sparse vector types</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.sparse_vectors?.length
                        ? selectedChunk.embedding_index.sparse_vectors.join(", ")
                        : "none"}
                    </p>
                  </div>
                  <div className="rounded border border-[#2d3648] bg-[#0f1623] p-2">
                    <p className="text-[#9aa5b8]">Vector dimension</p>
                    <p className="mt-1 break-words text-white">
                      {selectedChunk.embedding_index?.vector_dimension ?? "-"}
                    </p>
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded border border-[#2d3648] bg-[#111927] p-3 text-xs">
                <p className="font-semibold text-white">Chunk full text</p>
                <pre className="mt-2 max-h-[320px] overflow-auto whitespace-pre-wrap break-words text-[#b6bfcc]">
                  {selectedChunk.text || selectedChunk.preview}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
