# Frontend (Bun + Vite)

This frontend is copied from `/Users/felix/Desktop/basic-ui` as the UI baseline,
then adapted to current powerful-rag backend APIs.
- Bun
- Vite + React + TypeScript
- Tailwind v4

## 1) Start backend

From project root:

```bash
cd /Users/felix/Desktop/powerful-rag
source .venv/bin/activate
uvicorn backend.app:app --reload --port 8000
```

## 2) Start frontend

In a new terminal:

```bash
cd /Users/felix/Desktop/powerful-rag/frontend
bun install
bun run dev
```

Open: [http://127.0.0.1:5173](http://127.0.0.1:5173)

## API wiring

- Default `VITE_API_BASE=/api`
- Vite proxy forwards `/api/*` to `http://127.0.0.1:8000/*`
- Endpoints used by UI:
  - `GET /health`
  - `POST /upload`
  - `GET /documents`
  - `PATCH /documents/{doc_id}`
  - `DELETE /documents/{doc_id}`
  - `GET /documents/{doc_id}/preview`
  - `GET /documents/{doc_id}/download`
  - `GET /documents/{doc_id}/pipeline`
  - `GET /tasks/{task_id}`
  - `POST /search`
  - `POST /answer`

## Optional env override

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
```
