# Backend API

This backend wraps the existing notebook pipeline into stable API endpoints.

## Endpoints

- `GET /health`
- `GET /settings/providers/assistant`
- `GET /settings/providers/embedding`
- `GET /settings/assistant`
- `PATCH /settings/assistant`
- `POST /settings/assistant/test`
- `GET /settings/assistant/retrieval-status`
- `GET /settings/llamaparse`
- `PATCH /settings/llamaparse`
- `POST /ingest`
- `POST /upload` (multipart file upload)
- `GET /documents`
- `PATCH /documents/{doc_id}`
- `DELETE /documents/{doc_id}`
- `GET /documents/{doc_id}/preview`
- `GET /documents/{doc_id}/download`
- `GET /documents/{doc_id}/pipeline`
- `GET /tasks/{task_id}`
- `POST /search`
- `POST /answer`

## Run

```bash
source .venv/bin/activate
pip install fastapi uvicorn
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

## Notes

- `POST /ingest` expects parsed markdown pages under `output_dir` as `doc_*.md`.
- `POST /upload` supports direct ingest for `.md/.txt/.pdf` files.
- For `.pdf`, backend first extracts per-page text into staging markdown (`doc_*.md`) then enters ingest queue.
- Ingest persists merged table records to `backend/data/table_records_<collection>.json`.
- Search/answer auto-load that table-record store for the selected collection.

## Storage layout

- Raw uploaded files: `upload/`
- Upload staging markdown pages (`doc_0.md`) for direct `.md/.txt` ingest:
  `backend/data/upload_staging/`
- Per-collection table records:
  `backend/data/table_records_<collection>.json`
- Document registry (source path/status/metadata):
  `backend/data/documents_registry.json`

## Pipeline semantics

- Upload (`/upload`) writes raw file to disk first.
- For `.md/.txt`, backend materializes markdown page(s) then runs chunking + embedding.
- Chunking artifacts are generated in output/staging dir.
- Embedding vectors are upserted directly into Qdrant collection (no extra local vector file).
