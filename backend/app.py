from __future__ import annotations

import json
import os
import sqlite3
import traceback
import uuid
import html
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from openai import OpenAI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
from notebooks.chunk_pipeline import load_markdown_pages, merge_markdown_pages, process_markdown_text  # noqa: E402
from notebooks.embedding import get_embedder, upsert_chunks  # noqa: E402
from notebooks.rag_service import RAGService  # noqa: E402


load_dotenv()

APP_NAME = "powerful-rag-backend"
APP_VERSION = "0.1.0"

DEFAULT_OUTPUT_DIR = os.environ.get("RAG_PARSED_OUTPUT_DIR", "notebooks/output")
DEFAULT_COLLECTION = os.environ.get("RAG_COLLECTION", "documents")
DEFAULT_DOC_LANGUAGE = os.environ.get("RAG_DOC_LANGUAGE", "mixed")
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
REQUIRE_SELECTED_DOCS_FOR_ANSWER = os.environ.get("RAG_REQUIRE_SELECTED_DOCS_FOR_ANSWER", "1") != "0"

DATA_DIR = PROJECT_ROOT / "backend" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_STORE_PATH = DATA_DIR / "workspace_settings.json"
UPLOAD_DIR = PROJECT_ROOT / "upload"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_STAGING_DIR = DATA_DIR / "upload_staging"
UPLOAD_STAGING_DIR.mkdir(parents=True, exist_ok=True)
DOC_REGISTRY_PATH = DATA_DIR / "documents_registry.json"
WORKSPACE_DB_PATH = DATA_DIR / "workspace.sqlite3"
ALLOWED_DOC_LANGUAGES = {"en", "zh", "mixed", "auto"}
DIRECT_INGEST_EXTENSIONS = {".md", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
DEFAULT_USER_ID = "local-user"
ODL_PAGE_MARKER_TEMPLATE = "<!--ODL_PAGE_%page-number%-->"
ODL_PAGE_MARKER_REGEX = re.compile(r"<!--ODL_PAGE_(\d+)-->")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _mask_secret(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if len(raw) <= 8:
        return "****"
    return f"{raw[:4]}...{raw[-2:]}"


def _normalize_provider_option(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider") or "").strip()
    label = str(raw.get("label") or "").strip()
    default_model = str(raw.get("default_model") or "").strip()
    default_api_base = str(raw.get("default_api_base") or "").strip()
    if not provider or not label:
        return None
    requires_api_key = bool(raw.get("requires_api_key", True))
    return {
        "provider": provider,
        "label": label,
        "default_model": default_model,
        "default_api_base": default_api_base,
        "requires_api_key": requires_api_key,
    }


def _load_provider_options_from_env(
    env_name: str,
    fallback: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw = (os.environ.get(env_name) or "").strip()
    if not raw:
        return [dict(item) for item in fallback]
    try:
        payload = json.loads(raw)
    except Exception:
        return [dict(item) for item in fallback]
    if not isinstance(payload, list):
        return [dict(item) for item in fallback]

    items: list[dict[str, Any]] = []
    for item in payload:
        normalized = _normalize_provider_option(item)
        if normalized is not None:
            items.append(normalized)
    if not items:
        return [dict(item) for item in fallback]
    return items


def _ensure_custom_provider(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_custom = any(str(item.get("provider") or "").strip() == "custom" for item in items)
    if has_custom:
        return items
    next_items = [dict(item) for item in items]
    next_items.append(
        {
            "provider": "custom",
            "label": "Custom",
            "default_model": "",
            "default_api_base": "",
            "requires_api_key": True,
        }
    )
    return next_items


def _build_provider_catalog() -> dict[str, list[dict[str, Any]]]:
    assistant_fallback = [
        {
            "provider": "openai-compatible",
            "label": "OpenAI Compatible",
            "default_model": "gpt-4o",
            "default_api_base": "https://api.openai.com/v1",
            "requires_api_key": True,
        },
        {
            "provider": "dashscope",
            "label": "DashScope",
            "default_model": "qwen-max",
            "default_api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "requires_api_key": True,
        },
        {
            "provider": "deepseek",
            "label": "DeepSeek",
            "default_model": "deepseek-chat",
            "default_api_base": "https://api.deepseek.com/v1",
            "requires_api_key": True,
        },
    ]
    embedding_fallback = [
        {
            "provider": "bge",
            "label": "BGE Local",
            "default_model": "BAAI/bge-m3",
            "default_api_base": "local",
            "requires_api_key": False,
        },
        {
            "provider": "openai-compatible",
            "label": "OpenAI Compatible",
            "default_model": "text-embedding-3-small",
            "default_api_base": "https://api.openai.com/v1",
            "requires_api_key": True,
        },
    ]
    assistant = _ensure_custom_provider(
        _load_provider_options_from_env("RAG_LLM_PROVIDER_OPTIONS", assistant_fallback)
    )
    embedding = _ensure_custom_provider(
        _load_provider_options_from_env("RAG_EMBEDDING_PROVIDER_OPTIONS", embedding_fallback)
    )
    return {
        "assistant": assistant,
        "embedding": embedding,
    }


def _find_provider_option(options: list[dict[str, Any]], provider: str | None) -> dict[str, Any] | None:
    target = str(provider or "").strip()
    if not target:
        return options[0] if options else None
    for item in options:
        if str(item.get("provider") or "").strip() == target:
            return item
    return options[0] if options else None


def _default_assistant_settings(provider_catalog: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    now = _now_iso()
    assistant_options = provider_catalog.get("assistant") or []
    embedding_options = provider_catalog.get("embedding") or []

    llm_env_provider = (os.environ.get("LLM_PROVIDER") or "").strip()
    llm_default = _find_provider_option(assistant_options, llm_env_provider)
    llm_provider = str((llm_default or {}).get("provider") or "openai-compatible")
    llm_model = (os.environ.get("LLM_MODEL") or str((llm_default or {}).get("default_model") or "")).strip()
    llm_api_base = (
        os.environ.get("LLM_BASE_URL")
        or str((llm_default or {}).get("default_api_base") or "")
    ).strip()
    llm_api_key = (os.environ.get("LLM_API_KEY") or "").strip()

    embedder_override = (os.environ.get("EMBEDDER") or "").strip().lower()
    embedding_env_provider = (os.environ.get("EMBEDDING_PROVIDER") or "").strip()
    if not embedding_env_provider:
        if embedder_override == "bge":
            embedding_env_provider = "bge"
        elif embedder_override == "openai":
            embedding_env_provider = "openai-compatible"
    embedding_default = _find_provider_option(embedding_options, embedding_env_provider)
    embedding_provider = str((embedding_default or {}).get("provider") or "bge")
    embedding_model = (
        os.environ.get("EMBEDDING_MODEL") or str((embedding_default or {}).get("default_model") or "")
    ).strip()
    embedding_api_base = (
        os.environ.get("EMBEDDING_BASE_URL")
        or os.environ.get("LLM_BASE_URL")
        or str((embedding_default or {}).get("default_api_base") or "")
    ).strip()
    embedding_api_key = (
        os.environ.get("EMBEDDING_API_KEY")
        or ("" if embedding_provider == "bge" else os.environ.get("LLM_API_KEY", ""))
    ).strip()

    return {
        "user_id": DEFAULT_USER_ID,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "api_base": llm_api_base,
        "api_key": llm_api_key,
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.2")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "512")),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "embedding_api_base": embedding_api_base,
        "embedding_api_key": embedding_api_key,
        "created_at": now,
        "updated_at": now,
    }


def _default_llamaparse_settings() -> dict[str, Any]:
    now = _now_iso()
    env_job_url = (os.environ.get("JOB_URL") or "").strip()
    env_token = (os.environ.get("TOKEN") or "").strip()
    env_model = (os.environ.get("MODEL") or "").strip()
    env_ready = bool(env_job_url and env_token and env_model)
    return {
        "user_id": DEFAULT_USER_ID,
        "enabled": _env_bool("LLAMAPARSE_ENABLED", env_ready),
        "base_url": (os.environ.get("LLAMAPARSE_BASE_URL") or env_job_url or "").strip(),
        "model": (os.environ.get("LLAMAPARSE_MODEL") or env_model or "").strip(),
        "api_key": (os.environ.get("LLAMAPARSE_API_KEY") or env_token or "").strip(),
        "created_at": now,
        "updated_at": now,
    }


def _merge_dict_with_defaults(defaults: dict[str, Any], current: Any) -> dict[str, Any]:
    merged = dict(defaults)
    if not isinstance(current, dict):
        return merged
    for key, value in current.items():
        if key in merged:
            merged[key] = value
    return merged


def _write_settings_store_unlocked(state: dict[str, Any]) -> None:
    SETTINGS_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_STORE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_workspace_settings_state() -> dict[str, Any]:
    catalog = _build_provider_catalog()
    defaults = {
        "assistant": _default_assistant_settings(catalog),
        "llamaparse": _default_llamaparse_settings(),
    }
    with _SETTINGS_LOCK:
        raw: dict[str, Any] = {}
        if SETTINGS_STORE_PATH.exists():
            try:
                loaded = json.loads(SETTINGS_STORE_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    raw = loaded
            except Exception:
                raw = {}

        assistant = _merge_dict_with_defaults(defaults["assistant"], raw.get("assistant"))
        llama = _merge_dict_with_defaults(defaults["llamaparse"], raw.get("llamaparse"))

        # Normalize critical fields.
        assistant["temperature"] = float(assistant.get("temperature", 0.2))
        assistant["max_tokens"] = int(assistant.get("max_tokens", 512))

        state = {
            "assistant": assistant,
            "llamaparse": llama,
        }
        if raw != state:
            _write_settings_store_unlocked(state)
    return state


def _save_workspace_settings_state(state: dict[str, Any]) -> None:
    with _SETTINGS_LOCK:
        _write_settings_store_unlocked(state)


def _assistant_settings_public(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": str(settings.get("user_id") or DEFAULT_USER_ID),
        "llm_provider": str(settings.get("llm_provider") or ""),
        "llm_model": str(settings.get("llm_model") or ""),
        "api_base": str(settings.get("api_base") or ""),
        "api_key_configured": bool(str(settings.get("api_key") or "").strip()),
        "api_key_preview": _mask_secret(str(settings.get("api_key") or "")),
        "temperature": float(settings.get("temperature", 0.2)),
        "max_tokens": int(settings.get("max_tokens", 512)),
        "embedding_provider": str(settings.get("embedding_provider") or ""),
        "embedding_model": str(settings.get("embedding_model") or ""),
        "embedding_api_base": str(settings.get("embedding_api_base") or ""),
        "embedding_api_key_configured": bool(str(settings.get("embedding_api_key") or "").strip()),
        "embedding_api_key_preview": _mask_secret(str(settings.get("embedding_api_key") or "")),
        "created_at": str(settings.get("created_at") or _now_iso()),
        "updated_at": str(settings.get("updated_at") or _now_iso()),
    }


def _llamaparse_settings_public(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "user_id": str(settings.get("user_id") or DEFAULT_USER_ID),
        "enabled": bool(settings.get("enabled", False)),
        "base_url": str(settings.get("base_url") or ""),
        "model": str(settings.get("model") or ""),
        "api_key_configured": bool(str(settings.get("api_key") or "").strip()),
        "api_key_preview": _mask_secret(str(settings.get("api_key") or "")),
        "created_at": str(settings.get("created_at") or _now_iso()),
        "updated_at": str(settings.get("updated_at") or _now_iso()),
    }


def _apply_runtime_settings(assistant_settings: dict[str, Any]) -> None:
    llm_key = str(assistant_settings.get("api_key") or "").strip()
    llm_base = str(assistant_settings.get("api_base") or "").strip()
    llm_model = str(assistant_settings.get("llm_model") or "").strip()
    embedding_provider = str(assistant_settings.get("embedding_provider") or "").strip().lower()
    embedding_key = str(assistant_settings.get("embedding_api_key") or "").strip()
    embedding_base = str(assistant_settings.get("embedding_api_base") or "").strip()
    embedding_model = str(assistant_settings.get("embedding_model") or "").strip()

    if llm_key:
        os.environ["LLM_API_KEY"] = llm_key
    else:
        os.environ.pop("LLM_API_KEY", None)
    if llm_base:
        os.environ["LLM_BASE_URL"] = llm_base
    else:
        os.environ.pop("LLM_BASE_URL", None)
    if llm_model:
        os.environ["LLM_MODEL"] = llm_model
        os.environ.setdefault("MULTI_QUERY_MODEL", llm_model)
    else:
        os.environ.pop("LLM_MODEL", None)

    if embedding_provider in {"bge", "local", "local-bge"}:
        os.environ["EMBEDDER"] = "bge"
    else:
        os.environ["EMBEDDER"] = "openai"
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
    else:
        os.environ.pop("EMBEDDING_MODEL", None)
    if embedding_base:
        os.environ["EMBEDDING_BASE_URL"] = embedding_base
    else:
        os.environ.pop("EMBEDDING_BASE_URL", None)
    if embedding_key:
        os.environ["EMBEDDING_API_KEY"] = embedding_key
    else:
        os.environ.pop("EMBEDDING_API_KEY", None)


def _apply_llamaparse_runtime_settings(llamaparse_settings: dict[str, Any]) -> None:
    enabled = bool(llamaparse_settings.get("enabled", False))
    job_url = str(llamaparse_settings.get("base_url") or "").strip()
    token = str(llamaparse_settings.get("api_key") or "").strip()
    model = str(llamaparse_settings.get("model") or "").strip()
    if enabled and job_url and token and model:
        os.environ["JOB_URL"] = job_url
        os.environ["TOKEN"] = token
        os.environ["MODEL"] = model
        return
    os.environ.pop("JOB_URL", None)
    os.environ.pop("TOKEN", None)
    os.environ.pop("MODEL", None)


def _reset_runtime_state_after_settings_change() -> None:
    with _SERVICE_LOCK:
        _SERVICE_CACHE.clear()
    try:
        import notebooks.retrieval as retrieval  # noqa: WPS433

        retrieval._MULTI_QUERY_CLIENT = None
        retrieval._MULTI_QUERY_CACHE.clear()
    except Exception:
        pass


def _sync_runtime_settings_from_store() -> None:
    state = _load_workspace_settings_state()
    _apply_runtime_settings(state["assistant"])
    _apply_llamaparse_runtime_settings(state["llamaparse"])


def _table_records_store_path(collection_name: str) -> Path:
    return DATA_DIR / f"table_records_{_safe_name(collection_name)}.json"


def _load_table_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid table records JSON (expected list): {path}")
    return payload


def _save_table_records(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_table_records(existing: list[dict], incoming: list[dict], doc_id: str) -> list[dict]:
    cleaned = [r for r in existing if str(r.get("_doc_id", "")) != doc_id]
    for row in incoming:
        item = dict(row)
        item["_doc_id"] = doc_id
        cleaned.append(item)
    return cleaned


class IngestRequest(BaseModel):
    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR, description="Directory containing parsed doc_*.md files")
    collection_name: str = Field(default=DEFAULT_COLLECTION)
    doc_id: str | None = None
    doc_language: str = Field(default=DEFAULT_DOC_LANGUAGE, pattern="^(en|zh|mixed|auto)$")
    save_artifacts: bool = True
    persist_table_records: bool = True


class IngestAccepted(BaseModel):
    task_id: str
    status: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    question: str
    collection_name: str = DEFAULT_COLLECTION
    doc_language: str = Field(default=DEFAULT_DOC_LANGUAGE, pattern="^(en|zh|mixed|auto)$")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    table_records_path: str | None = None
    selected_doc_ids: list[str] | None = None


class SearchResponse(BaseModel):
    question: str
    top_k: int
    collection_name: str
    result_count: int
    results: list[dict]


class AnswerRequest(BaseModel):
    question: str
    collection_name: str = DEFAULT_COLLECTION
    doc_language: str = Field(default=DEFAULT_DOC_LANGUAGE, pattern="^(en|zh|mixed|auto)$")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=50)
    table_records_path: str | None = None
    selected_doc_ids: list[str] | None = None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    retrieval_results: list[dict]
    tool_events: list[dict] = Field(default_factory=list)


class ChatTitleRequest(BaseModel):
    question: str
    answer: str
    current_title: str | None = None


class ChatTitleResponse(BaseModel):
    title: str


class ChatThreadCreateRequest(BaseModel):
    title: str | None = None
    selected_doc_ids: list[str] | None = None


class ChatThreadPatchRequest(BaseModel):
    title: str | None = None
    selected_doc_ids: list[str] | None = None


class ChatThreadAppendMessagesRequest(BaseModel):
    user_text: str
    assistant_text: str
    selected_doc_ids: list[str] = Field(default_factory=list)
    debug: dict[str, Any] | None = None
    answer_meta: dict[str, Any] | None = None


class ChatThreadReplaceTurnRequest(BaseModel):
    target_user_message_id: str
    target_assistant_message_id: str | None = None


class UploadResponse(BaseModel):
    filename: str
    stored_path: str
    status: str
    message: str
    collection_name: str
    doc_language: str
    task_id: str | None = None
    doc_id: str | None = None


class DocumentPatchRequest(BaseModel):
    folder_id: str | None = None
    description: str | None = None


class AssistantSettingsPatchRequest(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None


class AssistantConnectionTestRequest(BaseModel):
    llm_provider: str | None = None
    llm_model: str | None = None
    api_base: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_api_base: str | None = None
    embedding_api_key: str | None = None


class LlamaParseSettingsPatchRequest(BaseModel):
    enabled: bool | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


_TASKS: dict[str, dict[str, Any]] = {}
_TASK_LOCK = Lock()
_DOC_LOCK = Lock()
_SETTINGS_LOCK = Lock()
_CHAT_LOCK = Lock()

_SERVICE_CACHE: dict[str, RAGService] = {}
_SERVICE_LOCK = Lock()

_sync_runtime_settings_from_store()


def _new_task() -> str:
    task_id = uuid.uuid4().hex
    now = _now_iso()
    with _TASK_LOCK:
        _TASKS[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "message": "queued",
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }
    return task_id


def _update_task(task_id: str, **kwargs: Any) -> None:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            return
        task.update(kwargs)
        task["updated_at"] = _now_iso()


def _get_task_or_404(task_id: str) -> dict[str, Any]:
    with _TASK_LOCK:
        task = _TASKS.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
        return dict(task)


def _service_cache_key(collection_name: str, doc_language: str, table_records_path: Path | None) -> str:
    return f"{collection_name}||{doc_language}||{str(table_records_path) if table_records_path else ''}"


def _normalize_selected_doc_ids(raw_ids: list[str] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if raw_ids is None:
        return ids
    for raw in raw_ids:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def _sanitize_chat_title(text: str) -> str:
    title = str(text or "").strip()
    title = title.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"\s+", " ", title).strip()
    title = title.strip("`\"'“”‘’")
    title = re.sub(r"^[#\-\*\d\.\)\s]+", "", title).strip()
    if len(title) > 40:
        title = title[:40].rstrip(" .,:;，。！？!?")
    return title


def _fallback_chat_title(question: str, current_title: str | None = None) -> str:
    base = _sanitize_chat_title(question)
    if not base:
        base = _sanitize_chat_title(current_title or "")
    if not base:
        return "New chat"
    return base


def _generate_chat_title_with_llm(
    question: str,
    answer: str,
    current_title: str | None = None,
) -> str:
    _sync_runtime_settings_from_store()
    fallback = _fallback_chat_title(question, current_title)
    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return fallback

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("LLM_BASE_URL") or None,
        )
        model = os.environ.get("LLM_MODEL", "gpt-4o")
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是会话标题生成器。"
                        "根据用户首问和助手回答，生成一个简短标题。"
                        "要求：不超过18个中文字符或8个英文单词；"
                        "不要使用引号、句号、冒号、序号；只输出标题本身。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n"
                        f"助手回答：{answer}\n"
                        "请输出会话标题。"
                    ),
                },
            ],
            max_tokens=48,
        )
        title = _sanitize_chat_title((resp.choices[0].message.content or "").strip())
        return title or fallback
    except Exception:
        return fallback


def _invalidate_service_cache(collection_name: str) -> None:
    prefix = f"{collection_name}||"
    with _SERVICE_LOCK:
        keys = [k for k in _SERVICE_CACHE if k.startswith(prefix)]
        for k in keys:
            _SERVICE_CACHE.pop(k, None)


def _get_service(
    collection_name: str,
    doc_language: str,
    top_k: int,
    table_records_path: Path | None,
) -> RAGService:
    key = _service_cache_key(collection_name, doc_language, table_records_path)
    with _SERVICE_LOCK:
        svc = _SERVICE_CACHE.get(key)
        if svc is not None:
            svc.default_top_k = top_k
            return svc

        rows = _load_table_records(table_records_path) if table_records_path is not None else []
        svc = RAGService(
            collection_name=collection_name,
            doc_language=doc_language,
            table_records=rows,
            default_top_k=top_k,
        )
        _SERVICE_CACHE[key] = svc
        return svc


def _save_ingest_artifacts(output_dir: Path, processed: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fixed_text.md").write_text(processed["fixed_text"], encoding="utf-8")
    (output_dir / "text_chunks.json").write_text(
        json.dumps(processed["text_chunks"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "image_chunks.json").write_text(
        json.dumps(processed.get("image_chunks", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "table_records.json").write_text(
        json.dumps(processed["table_records"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "heading_corrections.json").write_text(
        json.dumps(processed["corrections"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_output_dir(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _load_document_registry() -> list[dict[str, Any]]:
    if not DOC_REGISTRY_PATH.exists():
        return []
    try:
        payload = json.loads(DOC_REGISTRY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return payload if isinstance(payload, list) else []


def _save_document_registry(rows: list[dict[str, Any]]) -> None:
    DOC_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_REGISTRY_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _upsert_document_registry(record: dict[str, Any]) -> None:
    collection_name = str(record.get("collection_name") or DEFAULT_COLLECTION)
    doc_id = str(record.get("doc_id") or "").strip()
    if not doc_id:
        return

    with _DOC_LOCK:
        rows = _load_document_registry()
        next_rows: list[dict[str, Any]] = []
        replaced = False
        for row in rows:
            if (
                str(row.get("collection_name") or DEFAULT_COLLECTION) == collection_name
                and str(row.get("doc_id") or "") == doc_id
            ):
                merged = dict(row)
                merged.update(record)
                merged["updated_at"] = _now_iso()
                next_rows.append(merged)
                replaced = True
            else:
                next_rows.append(row)

        if not replaced:
            item = dict(record)
            now = _now_iso()
            item.setdefault("uploaded_at", now)
            item.setdefault("updated_at", now)
            item.setdefault("folder_id", None)
            item.setdefault("description", "")
            next_rows.append(item)
        _save_document_registry(next_rows)


def _find_document_record(doc_id: str) -> dict[str, Any] | None:
    with _DOC_LOCK:
        rows = _load_document_registry()
    for row in rows:
        if str(row.get("doc_id") or "") == doc_id:
            return dict(row)
    return None


def _list_document_records() -> list[dict[str, Any]]:
    with _DOC_LOCK:
        rows = _load_document_registry()
    rows.sort(key=lambda x: str(x.get("updated_at") or ""), reverse=True)
    return rows


def _patch_document_record(doc_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    with _DOC_LOCK:
        rows = _load_document_registry()
        found: dict[str, Any] | None = None
        for i, row in enumerate(rows):
            if str(row.get("doc_id") or "") == doc_id:
                merged = dict(row)
                merged.update(updates)
                merged["updated_at"] = _now_iso()
                rows[i] = merged
                found = merged
                break
        if found is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        _save_document_registry(rows)
    return found


def _delete_document_record(doc_id: str) -> dict[str, Any]:
    with _DOC_LOCK:
        rows = _load_document_registry()
        kept: list[dict[str, Any]] = []
        deleted: dict[str, Any] | None = None
        for row in rows:
            if deleted is None and str(row.get("doc_id") or "") == doc_id:
                deleted = dict(row)
                continue
            kept.append(row)
        if deleted is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        _save_document_registry(kept)
    return deleted


def _to_workspace_document(row: dict[str, Any]) -> dict[str, Any]:
    storage_path = str(row.get("storage_path") or "")
    size_bytes = int(row.get("size_bytes") or 0)
    uploaded_at = str(row.get("uploaded_at") or _now_iso())
    updated_at = str(row.get("updated_at") or uploaded_at)
    return {
        "doc_id": str(row.get("doc_id") or ""),
        "folder_id": row.get("folder_id"),
        "source_name": str(row.get("source_name") or Path(storage_path).name or "document"),
        "description": str(row.get("description") or ""),
        "storage_path": storage_path,
        "mime_type": row.get("mime_type"),
        "size_bytes": size_bytes,
        "pages": int(row.get("pages") or 1),
        "status": str(row.get("status") or "uploaded"),
        "uploaded_at": uploaded_at,
        "updated_at": updated_at,
    }


def _chat_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(WORKSPACE_DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _init_chat_db() -> None:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_threads (
                    thread_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    selected_doc_ids_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_message_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    debug_json TEXT,
                    metadata_json TEXT,
                    seq INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE,
                    CHECK (role IN ('user', 'assistant'))
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_messages_thread_seq
                    ON chat_messages(thread_id, seq);
                CREATE INDEX IF NOT EXISTS idx_chat_threads_updated_at
                    ON chat_threads(updated_at DESC);
                CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created_at
                    ON chat_messages(thread_id, created_at ASC);
                """
            )
            conn.commit()
        finally:
            conn.close()


def _chat_json_dumps(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def _chat_json_loads(raw: str | None, fallback: Any) -> Any:
    if raw is None or raw == "":
        return fallback
    try:
        return json.loads(raw)
    except Exception:
        return fallback


def _chat_decode_selected_doc_ids(raw: str | None) -> list[str]:
    parsed = _chat_json_loads(raw, [])
    if not isinstance(parsed, list):
        return []
    return _normalize_selected_doc_ids([str(item or "") for item in parsed])


def _chat_message_row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "message_id": str(row["message_id"]),
        "thread_id": str(row["thread_id"]),
        "role": str(row["role"]),
        "content": str(row["content"]),
        "debug": _chat_json_loads(row["debug_json"], None),
        "metadata": _chat_json_loads(row["metadata_json"], None),
        "seq": int(row["seq"]),
        "created_at": str(row["created_at"]),
    }


def _chat_summary_query_sql() -> str:
    return """
        SELECT
            t.thread_id,
            t.user_id,
            t.title,
            t.selected_doc_ids_json,
            t.created_at,
            t.updated_at,
            t.last_message_at,
            (
                SELECT COUNT(1)
                FROM chat_messages m
                WHERE m.thread_id = t.thread_id
            ) AS message_count,
            COALESCE(
                (
                    SELECT m.content
                    FROM chat_messages m
                    WHERE m.thread_id = t.thread_id
                    ORDER BY m.seq DESC
                    LIMIT 1
                ),
                ''
            ) AS last_message_preview
        FROM chat_threads t
    """


def _chat_summary_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "thread_id": str(row["thread_id"]),
        "user_id": str(row["user_id"]),
        "title": str(row["title"]),
        "selected_doc_ids": _chat_decode_selected_doc_ids(row["selected_doc_ids_json"]),
        "message_count": int(row["message_count"] or 0),
        "last_message_preview": str(row["last_message_preview"] or ""),
        "created_at": str(row["created_at"]),
        "updated_at": str(row["updated_at"]),
        "last_message_at": str(row["last_message_at"]),
    }


def _chat_fetch_thread_row(conn: sqlite3.Connection, thread_id: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT
            thread_id,
            user_id,
            title,
            selected_doc_ids_json,
            created_at,
            updated_at,
            last_message_at
        FROM chat_threads
        WHERE thread_id = ?
        """,
        (thread_id,),
    ).fetchone()


def _chat_fetch_thread_summary(conn: sqlite3.Connection, thread_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        _chat_summary_query_sql() + " WHERE t.thread_id = ? LIMIT 1",
        (thread_id,),
    ).fetchone()
    if row is None:
        return None
    return _chat_summary_row_to_dict(row)


def _chat_fetch_thread_detail(conn: sqlite3.Connection, thread_id: str) -> dict[str, Any] | None:
    thread_row = _chat_fetch_thread_row(conn, thread_id)
    if thread_row is None:
        return None
    message_rows = conn.execute(
        """
        SELECT
            message_id,
            thread_id,
            role,
            content,
            debug_json,
            metadata_json,
            seq,
            created_at
        FROM chat_messages
        WHERE thread_id = ?
        ORDER BY seq ASC
        """,
        (thread_id,),
    ).fetchall()
    return {
        "thread_id": str(thread_row["thread_id"]),
        "user_id": str(thread_row["user_id"]),
        "title": str(thread_row["title"]),
        "selected_doc_ids": _chat_decode_selected_doc_ids(thread_row["selected_doc_ids_json"]),
        "created_at": str(thread_row["created_at"]),
        "updated_at": str(thread_row["updated_at"]),
        "last_message_at": str(thread_row["last_message_at"]),
        "messages": [_chat_message_row_to_record(row) for row in message_rows],
    }


def _chat_build_selected_docs_metadata(selected_doc_ids: list[str]) -> list[dict[str, Any]]:
    docs_by_id: dict[str, dict[str, Any]] = {}
    for row in _list_document_records():
        doc_id = str(row.get("doc_id") or "").strip()
        if doc_id:
            docs_by_id[doc_id] = row

    rows: list[dict[str, Any]] = []
    for doc_id in selected_doc_ids:
        item = docs_by_id.get(doc_id) or {}
        pages_raw = item.get("pages")
        pages: int | None = None
        if pages_raw is not None:
            try:
                pages = int(pages_raw)
            except Exception:
                pages = None
        payload: dict[str, Any] = {
            "doc_id": doc_id,
            "source_name": str(item.get("source_name") or doc_id),
        }
        if pages is not None:
            payload["pages"] = pages
        rows.append(payload)
    return rows


def _chat_selected_doc_ids_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    if not isinstance(metadata, dict):
        return []
    raw_docs = metadata.get("selected_docs")
    if not isinstance(raw_docs, list):
        return []
    ids: list[str] = []
    for row in raw_docs:
        if not isinstance(row, dict):
            continue
        ids.append(str(row.get("doc_id") or ""))
    return _normalize_selected_doc_ids(ids)


def _doc_page_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)$", path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _count_image_refs(text: str) -> int:
    if not text:
        return 0
    md_refs = re.findall(r"!\[[^\]]*\]\([^)]+\)", text)
    html_refs = re.findall(r"<img\s+[^>]*src=[\"'][^\"']+[\"'][^>]*>", text, flags=re.IGNORECASE)
    return len(md_refs) + len(html_refs)


def _make_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        proxy=None,
        trust_env=False,
        check_compatibility=False,
    )


def _load_qdrant_doc_snapshot(collection_name: str, doc_id: str) -> dict[str, Any]:
    """
    Return point-id mapping and vector-route metadata for one doc.
    Any failure is non-fatal and returned as error text.
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
    except Exception as e:
        return {
            "point_id_by_chunk_index": {},
            "dense_vector_name": "default",
            "sparse_vectors": [],
            "point_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }

    try:
        qdrant = _make_qdrant_client()
        info = qdrant.get_collection(collection_name)

        dense_vector_name = "default"
        vectors_cfg = info.config.params.vectors
        if isinstance(vectors_cfg, dict):
            if "dense" in vectors_cfg:
                dense_vector_name = "dense"
            elif vectors_cfg:
                first_name = next(iter(vectors_cfg.keys()))
                dense_vector_name = first_name or "default"

        sparse_vectors = sorted((info.config.params.sparse_vectors or {}).keys())

        point_map: dict[int, str] = {}
        offset = None
        while True:
            points, offset = qdrant.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for point in points:
                payload = point.payload or {}
                raw_index = payload.get("chunk_index")
                try:
                    chunk_index = int(raw_index)
                except (TypeError, ValueError):
                    continue
                if point.id is not None:
                    point_map[chunk_index] = str(point.id)
            if offset is None:
                break

        return {
            "point_id_by_chunk_index": point_map,
            "dense_vector_name": dense_vector_name,
            "sparse_vectors": sparse_vectors,
            "point_count": len(point_map),
            "error": None,
        }
    except Exception as e:
        return {
            "point_id_by_chunk_index": {},
            "dense_vector_name": "default",
            "sparse_vectors": [],
            "point_count": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def _build_pipeline_details(
    row: dict[str, Any],
    page: int = 1,
    page_size: int = 12,
) -> dict[str, Any]:
    parsed_output_dir = row.get("parsed_output_dir")
    parse_updated_at = str(row.get("updated_at") or _now_iso())
    doc_id = str(row.get("doc_id") or "")
    collection_name = str(row.get("collection_name") or DEFAULT_COLLECTION)
    parsed_dir = Path(str(parsed_output_dir)) if parsed_output_dir else None
    if parsed_dir is not None and not parsed_dir.exists():
        parsed_dir = None

    text_chunks: list[dict[str, Any]] = []
    image_chunks: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    if parsed_dir is not None:
        text_chunks = _read_json_list(parsed_dir / "text_chunks.json")
        image_chunks = _read_json_list(parsed_dir / "image_chunks.json")
        chunks = text_chunks + image_chunks

    markdown_files: list[dict[str, Any]] = []
    markdown_texts: list[str] = []
    if parsed_dir is not None:
        for md_path in sorted(parsed_dir.glob("doc_*.md"), key=_doc_page_sort_key):
            text = _read_uploaded_text(md_path)
            markdown_texts.append(text)
            markdown_files.append(
                {
                    "name": md_path.name,
                    "path": str(md_path),
                    "chars": len(text),
                    "preview": text[:240],
                }
            )

    generated_markdown: list[dict[str, Any]] = []
    if parsed_dir is not None:
        fixed_text_path = parsed_dir / "fixed_text.md"
        if fixed_text_path.exists():
            fixed_text = _read_uploaded_text(fixed_text_path)
            generated_markdown.append(
                {
                    "name": fixed_text_path.name,
                    "path": str(fixed_text_path),
                    "chars": len(fixed_text),
                    "preview": fixed_text[:240],
                }
            )

    table_records: list[dict[str, Any]] = []
    table_records_path = parsed_dir / "table_records.json" if parsed_dir is not None else None
    if table_records_path is not None:
        table_records = _read_json_list(table_records_path)
    if not table_records:
        global_table_path = _table_records_store_path(collection_name)
        try:
            global_rows = _load_table_records(global_table_path)
        except Exception:
            global_rows = []
        for item in global_rows:
            if str(item.get("_doc_id", "")) == doc_id:
                table_records.append(item)

    image_files: list[dict[str, Any]] = []
    if parsed_dir is not None:
        for path in sorted(parsed_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_files.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    image_refs = sum(_count_image_refs(text) for text in markdown_texts)
    image_count = max(len(image_files), image_refs)
    table_count = len(table_records) if table_records else int(row.get("table_records") or 0)
    table_preview = table_records[:5]

    total = len(chunks)
    safe_page_size = max(1, int(page_size))
    safe_page = max(1, int(page))
    start = (safe_page - 1) * safe_page_size
    end = start + safe_page_size

    qdrant_snapshot = _load_qdrant_doc_snapshot(collection_name=collection_name, doc_id=doc_id)
    point_id_by_chunk_index = qdrant_snapshot["point_id_by_chunk_index"]

    chunk_items: list[dict[str, Any]] = []
    for idx, c in enumerate(chunks[start:end], start=start):
        content = str(c.get("content") or "")
        heading = str(c.get("heading_path") or "")
        chunk_type = str(c.get("type") or "text").strip().lower() or "text"
        page_value = c.get("page")
        try:
            page_no = int(page_value)
            if page_no <= 0:
                page_no = None
        except (TypeError, ValueError):
            page_no = None
        try:
            chunk_index = int(c.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = idx
        point_id = point_id_by_chunk_index.get(chunk_index)
        chunk_items.append(
            {
                "chunk_id": f"{doc_id}-{chunk_index}",
                "chunk_order": idx + 1,
                "chunk_index": chunk_index,
                "chunk_level": heading or chunk_type,
                "heading_path": heading,
                "chunk_type": chunk_type,
                "char_count": len(content),
                "page_start": page_no,
                "page_end": page_no,
                "preview": content[:240],
                "text": content,
                "has_table": bool(c.get("has_table", False)),
                "image_path": str(c.get("image_path") or ""),
                "image_category": str(c.get("image_category") or ""),
                "embedding_index": {
                    "provider": "qdrant",
                    "collection_name": collection_name,
                    "point_id": point_id,
                    "chunk_index": chunk_index,
                    "dense_vector": qdrant_snapshot["dense_vector_name"],
                    "sparse_vectors": qdrant_snapshot["sparse_vectors"],
                    "vector_dimension": int(row.get("vector_dimension") or 0),
                    "status": "indexed" if point_id else "missing",
                },
            }
        )

    status = str(row.get("status") or "uploaded")
    embedded_chunks = (
        int(qdrant_snapshot["point_count"])
        if qdrant_snapshot["point_count"] > 0
        else (int(row.get("text_chunks") or 0) + int(row.get("image_chunks") or 0) if status == "ready" else 0)
    )
    missing_chunks = max(0, total - embedded_chunks)

    return {
        "doc_id": doc_id,
        "status": status,
        "parse": {
            "provider": str(row.get("parse_provider") or "upload"),
            "summary": str(row.get("parse_summary") or "Uploaded document"),
            "text_chars": int(sum(len(str(c.get("content") or "")) for c in text_chunks)),
            "image_count": image_count,
            "table_count": table_count,
            "garbled_risk": 0.0,
            "markdown_files": markdown_files,
            "generated_markdown": generated_markdown,
            "image_files": image_files,
            "table_preview": table_preview,
            "artifacts": {
                "storage_path": str(row.get("storage_path") or ""),
                "parsed_output_dir": str(parsed_output_dir or ""),
                "collection_name": collection_name,
                "text_chunks_path": str(parsed_dir / "text_chunks.json") if parsed_dir is not None else "",
                "image_chunks_path": str(parsed_dir / "image_chunks.json") if parsed_dir is not None else "",
                "table_records_path": str(table_records_path) if table_records_path is not None else "",
                "heading_corrections_path": (
                    str(parsed_dir / "heading_corrections.json") if parsed_dir is not None else ""
                ),
            },
            "updated_at": parse_updated_at,
        },
        "chunk": {
            "page": safe_page,
            "page_size": safe_page_size,
            "total": total,
            "items": chunk_items,
        },
        "index": {
            "index_profile": "qdrant-hybrid",
            "embedding_provider": str(row.get("doc_language") or DEFAULT_DOC_LANGUAGE),
            "embedding_model": os.environ.get("EMBEDDING_MODEL", "auto"),
            "total_chunks": total,
            "embedded_chunks": embedded_chunks,
            "missing_chunks": missing_chunks,
            "level_distribution": {},
            "tree": {"edge_count": 0, "root_nodes": 0, "leaf_nodes": 0},
            "index_build": {
                "collection_name": collection_name,
                "dense_vector": qdrant_snapshot["dense_vector_name"],
                "sparse_vectors": qdrant_snapshot["sparse_vectors"],
                "qdrant_point_count": qdrant_snapshot["point_count"],
                "qdrant_error": qdrant_snapshot["error"],
            },
            "ready": status == "ready",
            "vector_store": {
                "provider": "qdrant",
                "dimension": int(row.get("vector_dimension") or 0),
                "vector_count": embedded_chunks,
                "index_version": 1,
                "index_size_bytes": 0,
            },
        },
    }


def _build_svg_thumbnail(
    doc_name: str,
    status: str,
    width: int,
) -> str:
    safe_width = max(96, min(640, int(width or 220)))
    height = int(safe_width * 1.32)
    title = (doc_name or "Document").strip()
    title = title if len(title) <= 28 else title[:25] + "..."
    subtitle = (status or "unknown").strip().lower() or "unknown"
    title_esc = html.escape(title)
    subtitle_esc = html.escape(subtitle)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{safe_width}" height="{height}" viewBox="0 0 {safe_width} {height}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#1a2230"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{safe_width}" height="{height}" rx="14" fill="url(#bg)"/>
  <rect x="14" y="14" width="{safe_width - 28}" height="{height - 28}" rx="10" fill="none" stroke="#334155" stroke-width="1.2"/>
  <rect x="28" y="30" width="44" height="56" rx="7" fill="#e2e8f0"/>
  <path d="M40 52h20M40 60h20M40 68h14" stroke="#94a3b8" stroke-width="2" stroke-linecap="round"/>
  <text x="28" y="{height - 48}" font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="13" fill="#e5e7eb">{title_esc}</text>
  <text x="28" y="{height - 26}" font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="11" fill="#94a3b8">status: {subtitle_esc}</text>
</svg>"""


def _markdown_inline_to_html(text: str) -> str:
    escaped = html.escape(text, quote=True)
    escaped = re.sub(
        r"`([^`]+)`",
        lambda m: f"<code>{m.group(1)}</code>",
        escaped,
    )
    escaped = re.sub(
        r"\*\*([^*]+)\*\*",
        lambda m: f"<strong>{m.group(1)}</strong>",
        escaped,
    )
    escaped = re.sub(
        r"\*([^*]+)\*",
        lambda m: f"<em>{m.group(1)}</em>",
        escaped,
    )
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: (
            f'<a href="{html.escape(m.group(2), quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">{m.group(1)}</a>'
        ),
        escaped,
    )
    return escaped


def _markdown_to_html_fragment(markdown_text: str) -> str:
    """
    Render a lightweight markdown subset for preview/thumbnail surfaces.
    """
    lines = markdown_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    in_code = False
    code_lines: list[str] = []
    in_ul = False
    in_ol = False
    para_lines: list[str] = []

    def flush_para() -> None:
        nonlocal para_lines
        if not para_lines:
            return
        text = " ".join(s.strip() for s in para_lines if s.strip())
        if text:
            blocks.append(f"<p>{_markdown_inline_to_html(text)}</p>")
        para_lines = []

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            blocks.append("</ul>")
            in_ul = False
        if in_ol:
            blocks.append("</ol>")
            in_ol = False

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.strip().startswith("```"):
            flush_para()
            close_lists()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                blocks.append(
                    "<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>"
                )
                in_code = False
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            flush_para()
            close_lists()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_para()
            close_lists()
            level = min(6, len(heading_match.group(1)))
            heading_text = _markdown_inline_to_html(heading_match.group(2).strip())
            blocks.append(f"<h{level}>{heading_text}</h{level}>")
            continue

        ul_match = re.match(r"^[-*+]\s+(.+)$", stripped)
        if ul_match:
            flush_para()
            if in_ol:
                blocks.append("</ol>")
                in_ol = False
            if not in_ul:
                blocks.append("<ul>")
                in_ul = True
            blocks.append(f"<li>{_markdown_inline_to_html(ul_match.group(1).strip())}</li>")
            continue

        ol_match = re.match(r"^\d+\.\s+(.+)$", stripped)
        if ol_match:
            flush_para()
            if in_ul:
                blocks.append("</ul>")
                in_ul = False
            if not in_ol:
                blocks.append("<ol>")
                in_ol = True
            blocks.append(f"<li>{_markdown_inline_to_html(ol_match.group(1).strip())}</li>")
            continue

        quote_match = re.match(r"^>\s+(.+)$", stripped)
        if quote_match:
            flush_para()
            close_lists()
            blocks.append(f"<blockquote>{_markdown_inline_to_html(quote_match.group(1).strip())}</blockquote>")
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_para()
            close_lists()
            blocks.append(f"<pre>{html.escape(stripped, quote=False)}</pre>")
            continue

        para_lines.append(stripped)

    flush_para()
    close_lists()
    if in_code:
        blocks.append("<pre><code>" + html.escape("\n".join(code_lines), quote=False) + "</code></pre>")

    return "\n".join(blocks)


def _build_markdown_preview_html(title: str, markdown_text: str) -> str:
    rendered = _markdown_to_html_fragment(markdown_text)
    title_escaped = html.escape(title, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title_escaped}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --card: #ffffff;
      --ink: #0f172a;
      --ink-soft: #334155;
      --line: #dbe3ef;
      --accent: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(1200px 500px at 10% 0%, #eef4ff 0%, var(--bg) 48%, #f4f7fb 100%);
      color: var(--ink);
      font: 15px/1.7 "SF Pro Text", "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
      padding: 24px;
    }}
    .page {{
      max-width: 920px;
      margin: 0 auto;
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 20px 50px rgba(15, 23, 42, 0.08);
      padding: 24px 26px;
    }}
    h1,h2,h3,h4,h5,h6 {{ line-height: 1.35; margin: 0.8em 0 0.4em; }}
    h1 {{ font-size: 1.7rem; }}
    h2 {{ font-size: 1.35rem; }}
    h3 {{ font-size: 1.15rem; }}
    p {{ margin: 0.55em 0; color: var(--ink-soft); }}
    ul,ol {{ margin: 0.5em 0 0.6em 1.4em; color: var(--ink-soft); }}
    li {{ margin: 0.2em 0; }}
    blockquote {{
      margin: 0.7em 0;
      border-left: 3px solid #93c5fd;
      background: #eff6ff;
      color: #1e3a8a;
      padding: 0.55em 0.8em;
      border-radius: 6px;
    }}
    code {{
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 0.08em 0.35em;
      font-family: "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
      font-size: 0.92em;
      color: #0f172a;
    }}
    pre {{
      margin: 0.8em 0;
      background: #0f172a;
      color: #e2e8f0;
      border-radius: 10px;
      padding: 0.8em 0.9em;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
    }}
    pre code {{
      background: transparent;
      border: none;
      padding: 0;
      color: inherit;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <article class="page">
    {rendered}
  </article>
</body>
</html>"""


def _build_content_svg_thumbnail(
    doc_name: str,
    status: str,
    width: int,
    markdown_text: str,
) -> str:
    safe_width = max(96, min(640, int(width or 220)))
    height = int(safe_width * 1.32)
    content_html = _markdown_to_html_fragment(markdown_text)
    doc_name_trimmed = (doc_name or "document").strip()
    if len(doc_name_trimmed) > 38:
        doc_name_trimmed = doc_name_trimmed[:35] + "..."
    status_escaped = html.escape(status.strip().lower() or "unknown", quote=True)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{safe_width}" height="{height}" viewBox="0 0 {safe_width} {height}">
  <defs>
    <linearGradient id="bg2" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f8fafc"/>
      <stop offset="100%" stop-color="#edf2f9"/>
    </linearGradient>
  </defs>
  <rect x="0" y="0" width="{safe_width}" height="{height}" rx="14" fill="url(#bg2)"/>
  <rect x="10" y="10" width="{safe_width - 20}" height="{height - 20}" rx="11" fill="#ffffff" stroke="#d6deea" stroke-width="1.1"/>
  <foreignObject x="18" y="16" width="{safe_width - 36}" height="{height - 62}">
    <div xmlns="http://www.w3.org/1999/xhtml" style="height:{height - 62}px;overflow:hidden;color:#0f172a;font:12px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;">
      <style>
        h1,h2,h3,h4,h5,h6{{margin:0 0 6px;font-size:13px;line-height:1.3}}
        p{{margin:0 0 6px;color:#334155}}
        ul,ol{{margin:0 0 6px 16px;color:#334155;padding:0}}
        li{{margin:0 0 3px}}
        blockquote{{margin:0 0 6px;padding:4px 6px;border-left:2px solid #93c5fd;background:#eff6ff;color:#1e3a8a}}
        pre{{margin:0 0 6px;padding:6px;border-radius:6px;background:#0f172a;color:#e2e8f0;white-space:pre-wrap}}
        code{{background:#f1f5f9;border:1px solid #e2e8f0;border-radius:4px;padding:1px 3px}}
        a{{color:#2563eb;text-decoration:none}}
      </style>
      <div>{content_html}</div>
    </div>
  </foreignObject>
  <rect x="10" y="{height - 38}" width="{safe_width - 20}" height="28" rx="8" fill="#f8fafc" stroke="#d6deea"/>
  <text x="18" y="{height - 20}" font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="11.5" fill="#334155">{html.escape(doc_name_trimmed, quote=False)}</text>
  <text x="{safe_width - 16}" y="{height - 20}" text-anchor="end" font-family="system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif" font-size="10.5" fill="#64748b">{status_escaped}</text>
</svg>"""


def _extract_thumbnail_markdown(row: dict[str, Any]) -> str:
    parsed_output_dir = row.get("parsed_output_dir")
    if parsed_output_dir:
        parsed_dir = Path(str(parsed_output_dir))
        first_page = parsed_dir / "doc_0.md"
        if first_page.exists():
            return _read_uploaded_text(first_page)

    storage_path = Path(str(row.get("storage_path") or ""))
    if not storage_path.exists():
        return ""

    suffix = storage_path.suffix.lower()
    media_type = str(row.get("mime_type") or "")

    if suffix in {".md", ".markdown"} or "markdown" in media_type:
        return _read_uploaded_text(storage_path)
    if suffix == ".txt" or media_type.startswith("text/"):
        return _read_uploaded_text(storage_path)
    return ""


def _read_uploaded_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def _normalize_inline_text(value: str) -> str:
    cleaned = re.sub(r"[\x00-\x1F\x7F]", " ", str(value or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def _source_name_stem(source_name: str) -> str:
    return Path(str(source_name or "").strip()).stem.strip()


def _is_placeholder_document_description(value: str, source_name: str) -> bool:
    normalized = _normalize_inline_text(value).lower()
    if not normalized or normalized == "uploaded document":
        return True
    stem = _normalize_inline_text(_source_name_stem(source_name)).lower()
    return bool(stem) and normalized == stem


def _strip_markdown_for_description(text: str) -> str:
    value = html.unescape(str(text or ""))
    value = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", value)
    value = re.sub(r"<img\s+[^>]*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"</?(table|thead|tbody|tr|th|td|p|div|span|section|article|br)[^>]*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"`{1,3}[^`]*`{1,3}", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"[#>*_~|]", " ", value)
    return _normalize_inline_text(value)


def _extract_heading_for_description(markdown_text: str) -> str:
    for raw_line in str(markdown_text or "").splitlines():
        match = re.match(r"^\s*#{1,6}\s+(.+)$", raw_line)
        if not match:
            continue
        heading = _strip_markdown_for_description(match.group(1))
        if heading:
            return heading
    return ""


def _extract_body_for_description(markdown_text: str) -> str:
    for raw_line in str(markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if re.match(r"^#{1,6}\s+", line):
            continue
        if line.startswith("![") or line.lower().startswith("<img"):
            continue
        cleaned = _strip_markdown_for_description(line)
        if len(cleaned) >= 4:
            return cleaned
    return _strip_markdown_for_description(markdown_text)


def _build_auto_document_description(
    source_name: str,
    markdown_text: str,
    table_count: int,
    page_count: int,
) -> str:
    stem = _normalize_inline_text(_source_name_stem(source_name)).lower()
    heading = _extract_heading_for_description(markdown_text)
    body = _extract_body_for_description(markdown_text)

    parts: list[str] = []
    if heading and _normalize_inline_text(heading).lower() != stem:
        parts.append(_normalize_inline_text(heading))

    body_norm = _normalize_inline_text(body)
    if body_norm and _normalize_inline_text(body_norm).lower() != stem:
        if not parts:
            parts.append(body_norm)
        elif not body_norm.startswith(parts[0]):
            parts.append(body_norm)

    if not parts:
        if table_count > 0:
            return f"包含 {table_count} 条表格结构化记录。"
        if page_count > 0:
            return f"共 {page_count} 页文档。"
        return ""

    text = "。".join(parts[:2]).strip("。 ")
    if not text:
        return ""

    max_chars = 140
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _load_markdown_for_description(parsed_dir: Path | None) -> str:
    if parsed_dir is None or not parsed_dir.exists():
        return ""

    fixed_text_path = parsed_dir / "fixed_text.md"
    if fixed_text_path.exists():
        return _read_uploaded_text(fixed_text_path)

    page_paths = sorted(parsed_dir.glob("doc_*.md"), key=_doc_page_sort_key)
    if not page_paths:
        return ""
    parts = [_read_uploaded_text(path) for path in page_paths]
    return "\n\n".join(part for part in parts if part.strip())


def _autofill_document_description_record(row: dict[str, Any], force: bool = False) -> dict[str, Any]:
    doc_id = str(row.get("doc_id") or "")
    if not doc_id:
        return row

    source_name = str(row.get("source_name") or doc_id)
    current_description = str(row.get("description") or "")
    if not force and not _is_placeholder_document_description(current_description, source_name):
        return row

    parsed_output_dir = row.get("parsed_output_dir")
    parsed_dir = Path(str(parsed_output_dir)) if parsed_output_dir else None
    if parsed_dir is not None and not parsed_dir.exists():
        parsed_dir = None

    markdown_text = _load_markdown_for_description(parsed_dir)
    table_count = int(row.get("table_records") or 0)
    page_count = int(row.get("pages") or 0)
    generated = _build_auto_document_description(
        source_name=source_name,
        markdown_text=markdown_text,
        table_count=table_count,
        page_count=page_count,
    )
    if not generated:
        return row
    if generated == _normalize_inline_text(current_description):
        return row
    return _patch_document_record(doc_id, {"description": generated})


def _materialize_uploaded_markdown(uploaded_path: Path) -> Path:
    suffix = uploaded_path.suffix.lower()
    if suffix not in DIRECT_INGEST_EXTENSIONS:
        raise ValueError(f"Unsupported extension for direct ingest: {suffix}")

    text = _read_uploaded_text(uploaded_path)
    if suffix == ".txt":
        text = f"# {uploaded_path.stem}\n\n{text}"

    output_dir = UPLOAD_STAGING_DIR / f"{uploaded_path.stem}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "doc_0.md").write_text(text, encoding="utf-8")
    return output_dir


def _split_opendataloader_markdown_pages(markdown_text: str) -> list[tuple[int, str]]:
    matches = list(ODL_PAGE_MARKER_REGEX.finditer(markdown_text))
    if not matches:
        content = markdown_text.strip()
        return [(1, content)] if content else []

    pages: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown_text)
        try:
            page_no = int(match.group(1))
        except Exception:
            page_no = len(pages) + 1
        page_text = markdown_text[match.end() : next_start].strip()
        pages.append((page_no, page_text))

    pages.sort(key=lambda item: item[0])
    deduped: list[tuple[int, str]] = []
    seen: set[int] = set()
    for page_no, text in pages:
        if page_no in seen:
            continue
        seen.add(page_no)
        deduped.append((page_no, text))
    return deduped


def _materialize_uploaded_pdf(uploaded_path: Path) -> tuple[Path, int, int]:
    """
    使用 opendataloader-pdf 将 PDF 抽取为多页 markdown：doc_0.md, doc_1.md, ...
    返回 (output_dir, page_count, empty_page_count)。
    """
    import opendataloader_pdf

    output_dir = UPLOAD_STAGING_DIR / f"{uploaded_path.stem}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    opendataloader_pdf.convert(
        input_path=str(uploaded_path),
        output_dir=str(output_dir),
        format="markdown-with-html",
        markdown_page_separator=ODL_PAGE_MARKER_TEMPLATE,
        image_output="external",
        quiet=True,
    )

    markdown_candidates = sorted(output_dir.glob("*.md")) + sorted(output_dir.glob("*.markdown"))
    if not markdown_candidates:
        raise RuntimeError("opendataloader parser returned no markdown file")

    preferred = output_dir / f"{uploaded_path.stem}.md"
    merged_markdown_path = preferred if preferred.exists() else markdown_candidates[0]
    merged_markdown = _read_uploaded_text(merged_markdown_path)
    pages = _split_opendataloader_markdown_pages(merged_markdown)

    if not pages:
        (output_dir / "doc_0.md").write_text(
            f"# {uploaded_path.stem}\n\n(empty pdf)\n",
            encoding="utf-8",
        )
        return output_dir, 0, 0

    empty_pages = 0
    for idx, (page_no, text) in enumerate(pages):
        text = text.strip()
        if not text:
            empty_pages += 1
            text = f"# {uploaded_path.stem} - page {page_no}\n\n(empty page or image-only page)\n"
        elif not text.endswith("\n"):
            text = text + "\n"

        (output_dir / f"doc_{idx}.md").write_text(text, encoding="utf-8")

    return output_dir, len(pages), empty_pages


def _materialize_uploaded_pdf_via_job_api(
    uploaded_path: Path,
    *,
    job_url: str | None = None,
    token: str | None = None,
    model: str | None = None,
) -> tuple[Path, int, int]:
    """
    按 notebooks/01_paddle_ocr_smoke_test.ipynb 的逻辑走远端解析服务：
    1) 提交本地 PDF 到 JOB_URL
    2) 轮询任务状态直到 done
    3) 下载 resultUrl.jsonUrl 的 jsonl
    4) 落盘 doc_{page}.md + markdown/images + outputImages（如 layout_det_res_0.jpg）
    """
    import requests
    import time

    resolved_job_url = (
        (os.environ.get("JOB_URL") or "").strip() if job_url is None else str(job_url).strip()
    )
    resolved_token = (
        (os.environ.get("TOKEN") or "").strip() if token is None else str(token).strip()
    )
    resolved_model = (
        (os.environ.get("MODEL") or "").strip() if model is None else str(model).strip()
    )
    if not all([resolved_job_url, resolved_token, resolved_model]):
        raise RuntimeError("Missing JOB_URL/TOKEN/MODEL for notebook parser.")

    headers = {
        "Authorization": f"bearer {resolved_token}",
    }
    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }

    data = {
        "model": resolved_model,
        "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
    }
    with uploaded_path.open("rb") as f:
        files = {"file": f}
        resp = requests.post(resolved_job_url, headers=headers, data=data, files=files, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"submit failed: {resp.status_code} {resp.text[:400]}")

    payload = resp.json()
    job_id = payload.get("data", {}).get("jobId")
    if not job_id:
        raise RuntimeError(f"missing jobId in submit response: {payload}")

    poll_interval = float(os.environ.get("JOB_POLL_INTERVAL_SEC", "3"))
    max_wait_sec = int(os.environ.get("JOB_POLL_TIMEOUT_SEC", "900"))
    started = time.time()
    jsonl_url = ""
    while True:
        poll = requests.get(f"{resolved_job_url}/{job_id}", headers=headers, timeout=30)
        if poll.status_code != 200:
            raise RuntimeError(f"poll failed: {poll.status_code} {poll.text[:400]}")
        job_data = poll.json().get("data", {})
        state = str(job_data.get("state", "")).lower()
        if state == "done":
            jsonl_url = str(job_data.get("resultUrl", {}).get("jsonUrl") or "")
            break
        if state == "failed":
            err = job_data.get("errorMsg") or "unknown"
            raise RuntimeError(f"remote parse failed: {err}")
        if time.time() - started > max_wait_sec:
            raise TimeoutError(f"poll timeout after {max_wait_sec}s, state={state}")
        time.sleep(poll_interval)

    if not jsonl_url:
        raise RuntimeError("missing resultUrl.jsonUrl in completed job response")

    jsonl_resp = requests.get(jsonl_url, timeout=120)
    jsonl_resp.raise_for_status()
    lines = [ln.strip() for ln in jsonl_resp.text.splitlines() if ln.strip()]

    output_dir = UPLOAD_STAGING_DIR / f"{uploaded_path.stem}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    page_num = 0
    empty_pages = 0

    for line in lines:
        item = json.loads(line)
        result = item.get("result", {})
        layout_results = result.get("layoutParsingResults") or []
        for res in layout_results:
            markdown_obj = res.get("markdown") or {}
            markdown_text = str(markdown_obj.get("text") or "")
            if not markdown_text.strip():
                empty_pages += 1
                markdown_text = f"# {uploaded_path.stem} - page {page_num + 1}\n\n(empty page)\n"
            md_filename = output_dir / f"doc_{page_num}.md"
            md_filename.write_text(markdown_text, encoding="utf-8")

            # markdown 内引用的图片资源
            for img_rel_path, img_url in (markdown_obj.get("images") or {}).items():
                img_rel = str(img_rel_path).strip().lstrip("./")
                if not img_rel:
                    continue
                target = output_dir / img_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img_bytes = requests.get(str(img_url), timeout=60).content
                    target.write_bytes(img_bytes)
                except Exception:
                    continue

            # layout_det_res 等输出图
            for img_name, img_url in (res.get("outputImages") or {}).items():
                safe_name = _safe_name(str(img_name)) or "image"
                target = output_dir / f"{safe_name}_{page_num}.jpg"
                try:
                    img_resp = requests.get(str(img_url), timeout=60)
                    if img_resp.status_code == 200:
                        target.write_bytes(img_resp.content)
                except Exception:
                    continue

            page_num += 1

    if page_num == 0:
        raise RuntimeError("remote parser returned no layoutParsingResults pages")

    return output_dir, page_num, empty_pages


def _run_ingest_task(task_id: str, req: IngestRequest) -> None:
    try:
        _sync_runtime_settings_from_store()
        _update_task(
            task_id,
            status="running",
            progress=5,
            message="loading markdown pages",
            started_at=_now_iso(),
        )

        output_dir = _resolve_output_dir(req.output_dir)
        pages = load_markdown_pages(output_dir)
        if not pages:
            raise FileNotFoundError(f"No doc_*.md found under: {output_dir}")

        raw_text = merge_markdown_pages(pages)
        if not raw_text.strip():
            raise RuntimeError(f"Merged markdown is empty: {output_dir}")

        _update_task(task_id, progress=20, message="running heading/table extraction")

        if "LLM_API_KEY" not in os.environ:
            raise RuntimeError("Missing LLM_API_KEY for heading/table extraction")
        client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or None,
        )
        model = os.environ.get("LLM_MODEL", "gpt-4o")

        processed = process_markdown_text(
            raw_text,
            client=client,
            model=model,
            pages=pages,
            output_dir=output_dir,
        )
        text_chunks: list[dict] = processed["text_chunks"]
        image_chunks: list[dict] = processed.get("image_chunks", [])
        image_stats: dict[str, Any] = processed.get("image_stats", {})
        table_records: list[dict] = processed["table_records"]
        indexed_chunks = text_chunks + image_chunks

        _update_task(task_id, progress=65, message="embedding and upsert")

        doc_id = req.doc_id or output_dir.name
        embedder = get_embedder(doc_language=req.doc_language)
        upsert_chunks(
            chunks=indexed_chunks,
            embedder=embedder,
            collection_name=req.collection_name,
            doc_id=doc_id,
        )

        table_path = _table_records_store_path(req.collection_name)
        table_total = None
        if req.persist_table_records:
            existing = _load_table_records(table_path)
            merged = _merge_table_records(existing, table_records, doc_id=doc_id)
            _save_table_records(table_path, merged)
            table_total = len(merged)

        if req.save_artifacts:
            _save_ingest_artifacts(output_dir, processed)

        _invalidate_service_cache(req.collection_name)

        result_payload = {
            "output_dir": str(output_dir),
            "collection_name": req.collection_name,
            "doc_id": doc_id,
            "pages": len(pages),
            "text_chunks": len(text_chunks),
            "image_chunks": len(image_chunks),
            "indexed_chunks": len(indexed_chunks),
            "image_stats": image_stats,
            "table_records": len(table_records),
            "table_records_store_path": str(table_path) if req.persist_table_records else None,
            "table_records_store_count": table_total,
        }

        existing_row = _find_document_record(doc_id)
        parse_provider = str(existing_row.get("parse_provider") or "upload") if existing_row is not None else "upload"
        source_name = str(existing_row.get("source_name") or doc_id) if existing_row is not None else str(doc_id)
        existing_description = str(existing_row.get("description") or "") if existing_row is not None else ""
        auto_description = _build_auto_document_description(
            source_name=source_name,
            markdown_text=str(processed.get("fixed_text") or raw_text),
            table_count=len(table_records),
            page_count=len(pages),
        )
        if existing_row is None:
            next_description = auto_description
        elif _is_placeholder_document_description(existing_description, source_name):
            next_description = auto_description or existing_description
        else:
            next_description = existing_description
        _upsert_document_registry(
            {
                "doc_id": doc_id,
                "collection_name": req.collection_name,
                "doc_language": req.doc_language,
                "status": "ready",
                "parsed_output_dir": str(output_dir),
                "text_chunks": len(text_chunks),
                "image_chunks": len(image_chunks),
                "table_records": len(table_records),
                "vector_dimension": int(embedder.dimension),
                "parse_summary": "Indexed successfully",
                "parse_provider": parse_provider,
                "description": next_description,
                "task_id": task_id,
                "task_status": "completed",
            }
        )

        _update_task(
            task_id,
            status="completed",
            progress=100,
            message="completed",
            finished_at=_now_iso(),
            result=result_payload,
        )
    except Exception as e:
        doc_id_for_error = req.doc_id or Path(req.output_dir).name
        _upsert_document_registry(
            {
                "doc_id": doc_id_for_error,
                "collection_name": req.collection_name,
                "doc_language": req.doc_language,
                "status": "failed",
                "task_id": task_id,
                "task_status": "failed",
                "parse_summary": f"Failed: {type(e).__name__}",
            }
        )
        _update_task(
            task_id,
            status="failed",
            progress=100,
            message="failed",
            finished_at=_now_iso(),
            error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
        )


app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.on_event("startup")
def init_workspace_datastores() -> None:
    _init_chat_db()


@app.middleware("http")
async def api_prefix_compat(request: Request, call_next):
    """
    Backward-compat for frontend clients that call `/api/*`.
    We keep canonical routes without prefix, and rewrite prefixed paths.
    """
    path = request.scope.get("path", "")
    if path == "/api":
        request.scope["path"] = "/"
    elif path.startswith("/api/"):
        request.scope["path"] = path[4:]
    return await call_next(request)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "name": APP_NAME,
        "version": APP_VERSION,
        "status": "ok",
    }


@app.get("/settings/providers/assistant")
def list_assistant_providers() -> dict[str, Any]:
    return {"items": _build_provider_catalog()["assistant"]}


@app.get("/settings/providers/embedding")
def list_embedding_providers() -> dict[str, Any]:
    return {"items": _build_provider_catalog()["embedding"]}


@app.get("/settings/assistant")
def get_assistant_settings(
    llm_provider: str | None = None,
    embedding_provider: str | None = None,
) -> dict[str, Any]:
    state = _load_workspace_settings_state()
    catalog = _build_provider_catalog()
    settings = dict(state["assistant"])

    if llm_provider is not None:
        target_llm_provider = str(llm_provider or "").strip()
        if target_llm_provider:
            selected = _find_provider_option(catalog["assistant"], target_llm_provider)
            settings["llm_provider"] = str((selected or {}).get("provider") or target_llm_provider)
            settings["llm_model"] = str((selected or {}).get("default_model") or settings.get("llm_model") or "")
            settings["api_base"] = str((selected or {}).get("default_api_base") or settings.get("api_base") or "")

    if embedding_provider is not None:
        target_embedding_provider = str(embedding_provider or "").strip()
        if target_embedding_provider:
            selected = _find_provider_option(catalog["embedding"], target_embedding_provider)
            settings["embedding_provider"] = str(
                (selected or {}).get("provider") or target_embedding_provider
            )
            settings["embedding_model"] = str(
                (selected or {}).get("default_model") or settings.get("embedding_model") or ""
            )
            settings["embedding_api_base"] = str(
                (selected or {}).get("default_api_base") or settings.get("embedding_api_base") or ""
            )

    return _assistant_settings_public(settings)


@app.patch("/settings/assistant")
def patch_assistant_settings(req: AssistantSettingsPatchRequest) -> dict[str, Any]:
    state = _load_workspace_settings_state()
    settings = dict(state["assistant"])
    fields_set = set(req.model_fields_set)
    if not fields_set:
        return _assistant_settings_public(settings)

    if "llm_provider" in fields_set:
        llm_provider = str(req.llm_provider or "").strip()
        if not llm_provider:
            raise HTTPException(status_code=400, detail="llm_provider cannot be empty")
        settings["llm_provider"] = llm_provider
    if "llm_model" in fields_set:
        settings["llm_model"] = str(req.llm_model or "").strip()
    if "api_base" in fields_set:
        settings["api_base"] = str(req.api_base or "").strip()
    if "temperature" in fields_set:
        if req.temperature is None or req.temperature < 0 or req.temperature > 2:
            raise HTTPException(status_code=400, detail="temperature must be between 0 and 2")
        settings["temperature"] = float(req.temperature)
    if "max_tokens" in fields_set:
        if req.max_tokens is None or req.max_tokens < 64 or req.max_tokens > 4096:
            raise HTTPException(status_code=400, detail="max_tokens must be between 64 and 4096")
        settings["max_tokens"] = int(req.max_tokens)
    if "api_key" in fields_set:
        settings["api_key"] = str(req.api_key or "").strip()

    if "embedding_provider" in fields_set:
        embedding_provider = str(req.embedding_provider or "").strip()
        if not embedding_provider:
            raise HTTPException(status_code=400, detail="embedding_provider cannot be empty")
        settings["embedding_provider"] = embedding_provider
    if "embedding_model" in fields_set:
        settings["embedding_model"] = str(req.embedding_model or "").strip()
    if "embedding_api_base" in fields_set:
        settings["embedding_api_base"] = str(req.embedding_api_base or "").strip()
    if "embedding_api_key" in fields_set:
        settings["embedding_api_key"] = str(req.embedding_api_key or "").strip()

    settings["updated_at"] = _now_iso()
    state["assistant"] = settings
    _save_workspace_settings_state(state)
    _apply_runtime_settings(settings)
    _reset_runtime_state_after_settings_change()
    return _assistant_settings_public(settings)


@app.post("/settings/assistant/test")
def test_assistant_connection(req: AssistantConnectionTestRequest) -> dict[str, Any]:
    import time

    started = time.perf_counter()
    state = _load_workspace_settings_state()
    current = state["assistant"]

    llm_provider = str(req.llm_provider or current.get("llm_provider") or "unknown").strip() or "unknown"
    llm_model = str(req.llm_model or current.get("llm_model") or "").strip()
    api_base = str(req.api_base or current.get("api_base") or "").strip()
    api_key = str(req.api_key or current.get("api_key") or "").strip()

    if not llm_model:
        return {
            "ok": False,
            "provider": llm_provider,
            "mode": "chat-completions",
            "message": "LLM model is required.",
            "latency_ms": int(max(0, (time.perf_counter() - started) * 1000)),
        }
    if not api_key:
        return {
            "ok": False,
            "provider": llm_provider,
            "mode": "chat-completions",
            "message": "LLM API key is required.",
            "latency_ms": int(max(0, (time.perf_counter() - started) * 1000)),
        }

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base or None,
        )
        client.chat.completions.create(
            model=llm_model,
            temperature=0,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        return {
            "ok": True,
            "provider": llm_provider,
            "mode": "chat-completions",
            "message": "Connection succeeded.",
            "latency_ms": int(max(0, (time.perf_counter() - started) * 1000)),
        }
    except Exception as e:
        return {
            "ok": False,
            "provider": llm_provider,
            "mode": "chat-completions",
            "message": f"{type(e).__name__}: {e}",
            "latency_ms": int(max(0, (time.perf_counter() - started) * 1000)),
        }


@app.get("/settings/assistant/retrieval-status")
def get_assistant_retrieval_status() -> dict[str, Any]:
    settings = _load_workspace_settings_state()["assistant"]
    catalog = _build_provider_catalog()

    ready_docs = sum(1 for row in _list_document_records() if str(row.get("status") or "") == "ready")
    reasons: list[str] = []
    if ready_docs <= 0:
        reasons.append("no_ready_documents")

    llm_provider = str(settings.get("llm_provider") or "")
    llm_requires_key = bool((_find_provider_option(catalog["assistant"], llm_provider) or {}).get("requires_api_key", True))
    llm_key_configured = bool(str(settings.get("api_key") or "").strip())
    if llm_requires_key and not llm_key_configured:
        reasons.append("llm_api_key_missing")

    embedding_provider = str(settings.get("embedding_provider") or "")
    embedding_option = _find_provider_option(catalog["embedding"], embedding_provider)
    embedding_requires_key = bool((embedding_option or {}).get("requires_api_key", True))
    embedding_key_configured = bool(str(settings.get("embedding_api_key") or "").strip())
    if embedding_requires_key and not embedding_key_configured:
        reasons.append("embedding_api_key_missing")

    mode = "hybrid" if not reasons else "fallback"
    return {
        "mode": mode,
        "reasons": reasons,
        "llm_provider": llm_provider,
        "llm_model": str(settings.get("llm_model") or ""),
        "llm_api_key_configured": llm_key_configured,
        "embedding_provider": embedding_provider,
        "embedding_model": str(settings.get("embedding_model") or ""),
        "embedding_api_key_configured": embedding_key_configured,
        "ready_document_count": ready_docs,
    }


@app.get("/settings/llamaparse")
def get_llamaparse_settings() -> dict[str, Any]:
    settings = _load_workspace_settings_state()["llamaparse"]
    return _llamaparse_settings_public(settings)


@app.patch("/settings/llamaparse")
def patch_llamaparse_settings(req: LlamaParseSettingsPatchRequest) -> dict[str, Any]:
    state = _load_workspace_settings_state()
    settings = dict(state["llamaparse"])
    fields_set = set(req.model_fields_set)
    if not fields_set:
        return _llamaparse_settings_public(settings)

    if "enabled" in fields_set and req.enabled is not None:
        settings["enabled"] = bool(req.enabled)
    if "base_url" in fields_set:
        settings["base_url"] = str(req.base_url or "").strip()
    if "model" in fields_set:
        settings["model"] = str(req.model or "").strip()
    if "api_key" in fields_set:
        settings["api_key"] = str(req.api_key or "").strip()

    settings["updated_at"] = _now_iso()
    state["llamaparse"] = settings
    _save_workspace_settings_state(state)
    _sync_runtime_settings_from_store()
    return _llamaparse_settings_public(settings)


@app.post("/ingest", response_model=IngestAccepted)
def ingest(req: IngestRequest, background_tasks: BackgroundTasks) -> IngestAccepted:
    task_id = _new_task()
    background_tasks.add_task(_run_ingest_task, task_id, req)
    return IngestAccepted(task_id=task_id, status="queued")


@app.post("/upload", response_model=UploadResponse)
async def upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection_name: str = Form(DEFAULT_COLLECTION),
    doc_language: str = Form(DEFAULT_DOC_LANGUAGE),
    doc_id: str | None = Form(None),
) -> UploadResponse:
    if doc_language not in ALLOWED_DOC_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Invalid doc_language: {doc_language}")

    original_name = (file.filename or "upload.bin").strip() or "upload.bin"
    safe_name = _safe_name(Path(original_name).name) or f"upload_{uuid.uuid4().hex[:8]}"
    stamped_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}_{safe_name}"
    stored_path = UPLOAD_DIR / stamped_name

    raw = await file.read()
    stored_path.write_bytes(raw)
    await file.close()

    suffix = stored_path.suffix.lower()
    resolved_doc_id = (doc_id or Path(original_name).stem or Path(stamped_name).stem).strip() or None

    if suffix == ".pdf":
        parser_mode = "opendataloader"
        remote_parse_warning: str | None = None
        state = _load_workspace_settings_state()
        llama_settings = state.get("llamaparse", {}) if isinstance(state, dict) else {}
        remote_parse_enabled = bool(llama_settings.get("enabled", False))
        remote_job_url = str(llama_settings.get("base_url") or "").strip()
        remote_token = str(llama_settings.get("api_key") or "").strip()
        remote_model = str(llama_settings.get("model") or "").strip()
        try:
            remote_parse_ready = remote_parse_enabled and all(
                [remote_job_url, remote_token, remote_model]
            )
            if remote_parse_ready:
                try:
                    ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf_via_job_api(
                        stored_path,
                        job_url=remote_job_url,
                        token=remote_token,
                        model=remote_model,
                    )
                    parser_mode = "notebook_job_api"
                except Exception as remote_exc:
                    remote_parse_warning = f"{type(remote_exc).__name__}: {remote_exc}".replace("\n", " ")
                    ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf(stored_path)
                    parser_mode = "opendataloader_fallback_from_remote"
            else:
                ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf(stored_path)
                parser_mode = "opendataloader"
        except (ModuleNotFoundError, FileNotFoundError):
            _upsert_document_registry(
                {
                    "doc_id": resolved_doc_id,
                    "collection_name": collection_name,
                    "doc_language": doc_language,
                    "source_name": original_name,
                    "storage_path": str(stored_path),
                    "mime_type": file.content_type,
                    "size_bytes": len(raw),
                    "status": "uploaded_only",
                    "parse_summary": "Uploaded only; install opendataloader-pdf (and Java 11+) to enable PDF auto-ingest",
                    "parse_provider": parser_mode,
                }
            )
            return UploadResponse(
                filename=original_name,
                stored_path=str(stored_path),
                status="uploaded_only",
                message="文件已上传，但当前环境缺少 opendataloader-pdf 或 Java 运行时，无法自动解析 PDF。请安装后重试。",
                collection_name=collection_name,
                doc_language=doc_language,
                doc_id=resolved_doc_id,
            )
        except Exception as e:
            parse_summary = f"Uploaded only; PDF extract failed: {type(e).__name__}"
            response_message = f"文件已上传，但 PDF 文本提取失败：{type(e).__name__}: {e}"
            if remote_parse_warning:
                parse_summary = (
                    f"Uploaded only; remote+local PDF extract failed. "
                    f"remote={remote_parse_warning[:220]}, local={type(e).__name__}"
                )
                response_message = (
                    "文件已上传，但远端解析失败且本地回退也失败："
                    f"远端={remote_parse_warning}；本地={type(e).__name__}: {e}"
                )
            _upsert_document_registry(
                {
                    "doc_id": resolved_doc_id,
                    "collection_name": collection_name,
                    "doc_language": doc_language,
                    "source_name": original_name,
                    "storage_path": str(stored_path),
                    "mime_type": file.content_type,
                    "size_bytes": len(raw),
                    "status": "uploaded_only",
                    "parse_summary": parse_summary,
                    "parse_provider": parser_mode,
                }
            )
            return UploadResponse(
                filename=original_name,
                stored_path=str(stored_path),
                status="uploaded_only",
                message=response_message,
                collection_name=collection_name,
                doc_language=doc_language,
                doc_id=resolved_doc_id,
            )

        req = IngestRequest(
            output_dir=str(ingest_output_dir),
            collection_name=collection_name,
            doc_id=resolved_doc_id,
            doc_language=doc_language,
            save_artifacts=True,
            persist_table_records=True,
        )
        task_id = _new_task()
        background_tasks.add_task(_run_ingest_task, task_id, req)

        parse_summary = (
            f"Queued for indexing ({parser_mode}, pdf pages={page_count}, empty_pages={empty_pages})"
        )
        response_message = "文件已上传，PDF 文本提取完成并进入入库队列。"
        if remote_parse_warning:
            parse_summary = (
                f"Queued for indexing ({parser_mode}, remote_error={remote_parse_warning[:220]}, "
                f"pdf pages={page_count}, empty_pages={empty_pages})"
            )
            response_message = "文件已上传；远端解析失败，已自动回退本地 opendataloader-pdf 并进入入库队列。"

        _upsert_document_registry(
            {
                "doc_id": resolved_doc_id,
                "collection_name": collection_name,
                "doc_language": doc_language,
                "source_name": original_name,
                "storage_path": str(stored_path),
                "parsed_output_dir": str(ingest_output_dir),
                "mime_type": file.content_type,
                "size_bytes": len(raw),
                "pages": page_count,
                "status": "parsing",
                "task_id": task_id,
                "task_status": "queued",
                "parse_summary": parse_summary,
                "parse_provider": parser_mode,
            }
        )

        return UploadResponse(
            filename=original_name,
            stored_path=str(stored_path),
            status="queued",
            message=response_message,
            collection_name=collection_name,
            doc_language=doc_language,
            task_id=task_id,
            doc_id=resolved_doc_id,
        )

    if suffix not in DIRECT_INGEST_EXTENSIONS:
        _upsert_document_registry(
            {
                "doc_id": resolved_doc_id,
                "collection_name": collection_name,
                "doc_language": doc_language,
                "source_name": original_name,
                "storage_path": str(stored_path),
                "mime_type": file.content_type,
                "size_bytes": len(raw),
                "status": "uploaded_only",
                "parse_summary": "Uploaded only; unsupported extension",
            }
        )
        return UploadResponse(
            filename=original_name,
            stored_path=str(stored_path),
            status="uploaded_only",
            message="文件已上传，但当前仅支持 md/txt/pdf 自动入库。",
            collection_name=collection_name,
            doc_language=doc_language,
            doc_id=resolved_doc_id,
        )

    ingest_output_dir = _materialize_uploaded_markdown(stored_path)
    req = IngestRequest(
        output_dir=str(ingest_output_dir),
        collection_name=collection_name,
        doc_id=resolved_doc_id,
        doc_language=doc_language,
        save_artifacts=True,
        persist_table_records=True,
    )
    task_id = _new_task()
    background_tasks.add_task(_run_ingest_task, task_id, req)

    _upsert_document_registry(
        {
            "doc_id": resolved_doc_id,
            "collection_name": collection_name,
            "doc_language": doc_language,
            "source_name": original_name,
            "storage_path": str(stored_path),
            "parsed_output_dir": str(ingest_output_dir),
            "mime_type": file.content_type,
            "size_bytes": len(raw),
            "status": "parsing",
            "task_id": task_id,
            "task_status": "queued",
            "parse_summary": "Queued for indexing",
        }
    )

    return UploadResponse(
        filename=original_name,
        stored_path=str(stored_path),
        status="queued",
        message="文件已上传并进入入库队列。",
        collection_name=collection_name,
        doc_language=doc_language,
        task_id=task_id,
        doc_id=resolved_doc_id,
    )


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    rows = [_to_workspace_document(r) for r in _list_document_records()]
    return {"items": rows}


@app.patch("/documents/{doc_id}")
def patch_document(doc_id: str, req: DocumentPatchRequest) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    fields_set = set(req.model_fields_set)
    if "folder_id" in fields_set:
        updates["folder_id"] = req.folder_id
    if "description" in fields_set:
        updates["description"] = req.description
    row = _patch_document_record(doc_id, updates)
    return _to_workspace_document(row)


@app.post("/documents/{doc_id}/description/autofill")
def autofill_document_description(doc_id: str, force: bool = False) -> dict[str, Any]:
    row = _find_document_record(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    updated = _autofill_document_description_record(row, force=force)
    return _to_workspace_document(updated)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    deleted = _delete_document_record(doc_id)
    return {"deleted": True, "doc_id": doc_id, "storage_path": deleted.get("storage_path")}


@app.get("/documents/{doc_id}/preview")
def preview_document(doc_id: str):
    row = _find_document_record(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    storage_path = Path(str(row.get("storage_path") or ""))
    if not storage_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {storage_path}")
    media_type = row.get("mime_type") or "application/octet-stream"
    suffix = storage_path.suffix.lower()
    if suffix in {".md", ".markdown"} or "markdown" in str(media_type):
        markdown_text = _read_uploaded_text(storage_path)
        rendered = _build_markdown_preview_html(
            title=str(row.get("source_name") or storage_path.name),
            markdown_text=markdown_text,
        )
        return Response(content=rendered, media_type="text/html; charset=utf-8")
    return FileResponse(
        path=storage_path,
        media_type=media_type,
        filename=Path(storage_path).name,
        content_disposition_type="inline",
    )


@app.get("/documents/{doc_id}/thumbnail")
def document_thumbnail(doc_id: str, width: int = 220):
    row = _find_document_record(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    storage_path = Path(str(row.get("storage_path") or ""))
    media_type = str(row.get("mime_type") or "")
    if storage_path.exists() and media_type.startswith("image/"):
        return FileResponse(path=storage_path, media_type=media_type, filename=storage_path.name)

    thumbnail_markdown = _extract_thumbnail_markdown(row)
    if thumbnail_markdown.strip():
        svg = _build_content_svg_thumbnail(
            doc_name=str(row.get("source_name") or doc_id),
            status=str(row.get("status") or "unknown"),
            width=width,
            markdown_text=thumbnail_markdown[:2400],
        )
        return Response(content=svg, media_type="image/svg+xml")

    svg = _build_svg_thumbnail(
        doc_name=str(row.get("source_name") or doc_id),
        status=str(row.get("status") or "unknown"),
        width=width,
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/documents/{doc_id}/download")
def download_document(doc_id: str):
    row = _find_document_record(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    storage_path = Path(str(row.get("storage_path") or ""))
    if not storage_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {storage_path}")
    media_type = row.get("mime_type") or "application/octet-stream"
    filename = str(row.get("source_name") or storage_path.name)
    return FileResponse(path=storage_path, media_type=media_type, filename=filename)


@app.get("/documents/{doc_id}/pipeline")
def get_document_pipeline(
    doc_id: str,
    page: int = 1,
    page_size: int = 12,
) -> dict[str, Any]:
    row = _find_document_record(doc_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return _build_pipeline_details(row, page=page, page_size=page_size)


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str) -> TaskStatus:
    return TaskStatus(**_get_task_or_404(task_id))


@app.post("/chat/title", response_model=ChatTitleResponse)
def generate_chat_title(req: ChatTitleRequest) -> ChatTitleResponse:
    _sync_runtime_settings_from_store()
    question = str(req.question or "").strip()
    answer = str(req.answer or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if not answer:
        raise HTTPException(status_code=400, detail="answer is required")

    title = _generate_chat_title_with_llm(
        question=question,
        answer=answer,
        current_title=req.current_title,
    )
    return ChatTitleResponse(title=title)


@app.get("/chat/threads")
def list_chat_threads(user_id: str = DEFAULT_USER_ID) -> dict[str, Any]:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            rows = conn.execute(
                _chat_summary_query_sql()
                + """
                WHERE t.user_id = ?
                ORDER BY t.updated_at DESC, t.created_at DESC
                """,
                (user_id,),
            ).fetchall()
            items = [_chat_summary_row_to_dict(row) for row in rows]
            return {"items": items}
        finally:
            conn.close()


@app.post("/chat/threads")
def create_chat_thread(req: ChatThreadCreateRequest) -> dict[str, Any]:
    thread_id = f"thread-{uuid.uuid4().hex}"
    now = _now_iso()
    selected_doc_ids = _normalize_selected_doc_ids(req.selected_doc_ids)
    raw_title = _sanitize_chat_title(req.title or "")
    title = raw_title or "New chat"
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            conn.execute(
                """
                INSERT INTO chat_threads (
                    thread_id,
                    user_id,
                    title,
                    selected_doc_ids_json,
                    created_at,
                    updated_at,
                    last_message_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id,
                    DEFAULT_USER_ID,
                    title,
                    _chat_json_dumps(selected_doc_ids) or "[]",
                    now,
                    now,
                    now,
                ),
            )
            conn.commit()
            summary = _chat_fetch_thread_summary(conn, thread_id)
            if summary is None:
                raise HTTPException(status_code=500, detail="Failed to create chat thread")
            return summary
        finally:
            conn.close()


@app.get("/chat/threads/{thread_id}")
def get_chat_thread(thread_id: str) -> dict[str, Any]:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            detail = _chat_fetch_thread_detail(conn, thread_id)
            if detail is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")
            return detail
        finally:
            conn.close()


@app.patch("/chat/threads/{thread_id}")
def patch_chat_thread(thread_id: str, req: ChatThreadPatchRequest) -> dict[str, Any]:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            row = _chat_fetch_thread_row(conn, thread_id)
            if row is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")

            fields_set = set(req.model_fields_set)
            current_title = str(row["title"])
            current_doc_ids = _chat_decode_selected_doc_ids(row["selected_doc_ids_json"])

            next_title = current_title
            if "title" in fields_set:
                next_title = _sanitize_chat_title(req.title or "") or "New chat"

            next_doc_ids = current_doc_ids
            if "selected_doc_ids" in fields_set:
                next_doc_ids = _normalize_selected_doc_ids(req.selected_doc_ids)

            if "title" in fields_set or "selected_doc_ids" in fields_set:
                conn.execute(
                    """
                    UPDATE chat_threads
                    SET
                        title = ?,
                        selected_doc_ids_json = ?,
                        updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (
                        next_title,
                        _chat_json_dumps(next_doc_ids) or "[]",
                        _now_iso(),
                        thread_id,
                    ),
                )
                conn.commit()

            summary = _chat_fetch_thread_summary(conn, thread_id)
            if summary is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")
            return summary
        finally:
            conn.close()


@app.delete("/chat/threads/{thread_id}")
def delete_chat_thread(thread_id: str) -> dict[str, Any]:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            cur = conn.execute(
                "DELETE FROM chat_threads WHERE thread_id = ?",
                (thread_id,),
            )
            if cur.rowcount <= 0:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")
            conn.commit()
            return {"deleted": True, "thread_id": thread_id}
        finally:
            conn.close()


@app.post("/chat/threads/{thread_id}/messages/append")
def append_chat_messages(thread_id: str, req: ChatThreadAppendMessagesRequest) -> dict[str, Any]:
    user_text = str(req.user_text or "").strip()
    assistant_text = str(req.assistant_text or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="user_text is required")
    if not assistant_text:
        raise HTTPException(status_code=400, detail="assistant_text is required")

    selected_doc_ids = _normalize_selected_doc_ids(req.selected_doc_ids)
    now = _now_iso()
    user_message_id = f"msg-user-{uuid.uuid4().hex}"
    assistant_message_id = f"msg-assistant-{uuid.uuid4().hex}"

    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            thread_row = _chat_fetch_thread_row(conn, thread_id)
            if thread_row is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")

            max_seq_row = conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM chat_messages WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()
            seq_base = int((max_seq_row["max_seq"] if max_seq_row is not None else 0) or 0)

            user_metadata: dict[str, Any] = {
                "selected_docs": _chat_build_selected_docs_metadata(selected_doc_ids)
            }
            assistant_metadata: dict[str, Any] | None = None
            if isinstance(req.answer_meta, dict):
                assistant_metadata = {"answer_meta": req.answer_meta}

            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id,
                    thread_id,
                    role,
                    content,
                    debug_json,
                    metadata_json,
                    seq,
                    created_at
                )
                VALUES (?, ?, 'user', ?, NULL, ?, ?, ?)
                """,
                (
                    user_message_id,
                    thread_id,
                    user_text,
                    _chat_json_dumps(user_metadata),
                    seq_base + 1,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO chat_messages (
                    message_id,
                    thread_id,
                    role,
                    content,
                    debug_json,
                    metadata_json,
                    seq,
                    created_at
                )
                VALUES (?, ?, 'assistant', ?, ?, ?, ?, ?)
                """,
                (
                    assistant_message_id,
                    thread_id,
                    assistant_text,
                    _chat_json_dumps(req.debug),
                    _chat_json_dumps(assistant_metadata),
                    seq_base + 2,
                    now,
                ),
            )

            conn.execute(
                """
                UPDATE chat_threads
                SET
                    selected_doc_ids_json = ?,
                    updated_at = ?,
                    last_message_at = ?
                WHERE thread_id = ?
                """,
                (
                    _chat_json_dumps(selected_doc_ids) or "[]",
                    now,
                    now,
                    thread_id,
                ),
            )
            conn.commit()

            row = conn.execute(
                """
                SELECT
                    message_id,
                    thread_id,
                    role,
                    content,
                    debug_json,
                    metadata_json,
                    seq,
                    created_at
                FROM chat_messages
                WHERE message_id = ?
                LIMIT 1
                """,
                (assistant_message_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=500, detail="Failed to append chat messages")
            return _chat_message_row_to_record(row)
        finally:
            conn.close()


@app.post("/chat/threads/{thread_id}/turns/replace-latest")
def replace_chat_turn_with_latest(
    thread_id: str,
    req: ChatThreadReplaceTurnRequest,
) -> dict[str, Any]:
    with _CHAT_LOCK:
        conn = _chat_db_connect()
        try:
            thread_row = _chat_fetch_thread_row(conn, thread_id)
            if thread_row is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")

            message_rows = conn.execute(
                """
                SELECT
                    message_id,
                    thread_id,
                    role,
                    content,
                    debug_json,
                    metadata_json,
                    seq,
                    created_at
                FROM chat_messages
                WHERE thread_id = ?
                ORDER BY seq ASC
                """,
                (thread_id,),
            ).fetchall()
            messages = [_chat_message_row_to_record(row) for row in message_rows]
            if len(messages) < 2:
                raise HTTPException(status_code=400, detail="Not enough messages to replace turn.")

            latest_user = messages[-2]
            latest_assistant = messages[-1]
            if latest_user["role"] != "user" or latest_assistant["role"] != "assistant":
                raise HTTPException(
                    status_code=400,
                    detail="Latest messages are not a user+assistant pair.",
                )

            base = messages[:-2]
            old_user_index = -1
            for index, item in enumerate(base):
                if item["role"] == "user" and item["message_id"] == req.target_user_message_id:
                    old_user_index = index
                    break
            if old_user_index < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Target user message not found: {req.target_user_message_id}",
                )

            delete_count = 1
            adjacent = base[old_user_index + 1] if old_user_index + 1 < len(base) else None
            if (
                adjacent is not None
                and adjacent["role"] == "assistant"
                and (
                    req.target_assistant_message_id is None
                    or adjacent["message_id"] == req.target_assistant_message_id
                )
            ):
                delete_count = 2
            elif req.target_assistant_message_id:
                explicit_assistant_index = next(
                    (
                        idx
                        for idx, item in enumerate(base)
                        if item["role"] == "assistant"
                        and item["message_id"] == req.target_assistant_message_id
                    ),
                    -1,
                )
                if explicit_assistant_index == old_user_index + 1:
                    delete_count = 2

            next_messages = list(base)
            next_messages[old_user_index : old_user_index + delete_count] = [
                latest_user,
                latest_assistant,
            ]

            resequenced: list[dict[str, Any]] = []
            for index, item in enumerate(next_messages, start=1):
                next_item = dict(item)
                next_item["seq"] = index
                resequenced.append(next_item)

            latest_selected_doc_ids = _chat_selected_doc_ids_from_metadata(
                latest_user.get("metadata")
            )
            current_doc_ids = _chat_decode_selected_doc_ids(thread_row["selected_doc_ids_json"])
            next_doc_ids = latest_selected_doc_ids if latest_selected_doc_ids else current_doc_ids
            now = _now_iso()

            conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
            for item in resequenced:
                conn.execute(
                    """
                    INSERT INTO chat_messages (
                        message_id,
                        thread_id,
                        role,
                        content,
                        debug_json,
                        metadata_json,
                        seq,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(item["message_id"]),
                        thread_id,
                        str(item["role"]),
                        str(item["content"]),
                        _chat_json_dumps(item.get("debug")),
                        _chat_json_dumps(item.get("metadata")),
                        int(item["seq"]),
                        str(item.get("created_at") or now),
                    ),
                )

            conn.execute(
                """
                UPDATE chat_threads
                SET
                    selected_doc_ids_json = ?,
                    updated_at = ?,
                    last_message_at = ?
                WHERE thread_id = ?
                """,
                (
                    _chat_json_dumps(next_doc_ids) or "[]",
                    now,
                    now,
                    thread_id,
                ),
            )
            conn.commit()

            detail = _chat_fetch_thread_detail(conn, thread_id)
            if detail is None:
                raise HTTPException(status_code=404, detail=f"Chat thread not found: {thread_id}")
            return detail
        finally:
            conn.close()


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    _sync_runtime_settings_from_store()
    table_path = (
        Path(req.table_records_path)
        if req.table_records_path
        else _table_records_store_path(req.collection_name)
    )
    if not table_path.exists():
        table_path = None

    svc = _get_service(
        collection_name=req.collection_name,
        doc_language=req.doc_language,
        top_k=req.top_k,
        table_records_path=table_path,
    )
    out = svc.search(
        question=req.question,
        top_k=req.top_k,
        selected_doc_ids=req.selected_doc_ids,
    )
    return SearchResponse(
        question=out.question,
        top_k=out.top_k,
        collection_name=req.collection_name,
        result_count=len(out.results),
        results=out.results,
    )


def _to_float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if parsed != parsed:  # NaN
        return None
    return parsed


def _extract_query_variants(rows: list[dict[str, Any]]) -> list[str]:
    for row in rows:
        score_detail = row.get("score_detail")
        if not isinstance(score_detail, dict):
            continue
        raw = score_detail.get("query_variants")
        if not isinstance(raw, list):
            continue
        variants: list[str] = []
        for item in raw:
            value = str(item or "").strip()
            if value:
                variants.append(value)
        if variants:
            return variants
    return []


def _extract_reranker_model(rows: list[dict[str, Any]]) -> str:
    for row in rows:
        score_detail = row.get("score_detail")
        if not isinstance(score_detail, dict):
            continue
        model = str(score_detail.get("reranker_model") or "").strip()
        if model:
            return model
    return ""


def _build_answer_tool_events(
    req: AnswerRequest,
    *,
    selected_doc_ids: list[str],
    retrieval_results: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    answer_text: str,
) -> list[dict[str, Any]]:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    by_source: dict[str, int] = {}
    by_chunk_type: dict[str, int] = {}
    top_hits: list[dict[str, Any]] = []
    for index, row in enumerate(retrieval_results):
        source = str(row.get("source") or "").strip() or "unknown"
        chunk_type = str(row.get("chunk_type") or "text").strip() or "text"
        by_source[source] = by_source.get(source, 0) + 1
        by_chunk_type[chunk_type] = by_chunk_type.get(chunk_type, 0) + 1
        if index < 8:
            top_hits.append(
                {
                    "rank": index + 1,
                    "doc_id": str(row.get("doc_id") or ""),
                    "source_name": str(row.get("source_name") or row.get("doc_id") or ""),
                    "heading_path": str(row.get("heading_path") or ""),
                    "chunk_type": chunk_type,
                    "page": row.get("page"),
                    "final_score": _to_float_or_none(row.get("final_score")),
                    "rerank_score": _to_float_or_none(row.get("rerank_score")),
                    "rrf_score": _to_float_or_none(row.get("rrf_score")),
                }
            )

    query_variants = _extract_query_variants(retrieval_results)
    reranker_model = _extract_reranker_model(retrieval_results)
    citations_preview: list[dict[str, Any]] = []
    for item in sources[:12]:
        citations_preview.append(
            {
                "index": int(item.get("index") or 0),
                "doc_id": str(item.get("doc_id") or ""),
                "source_name": str(item.get("source_name") or ""),
                "heading_path": str(item.get("heading_path") or ""),
                "source_type": str(item.get("source_type") or ""),
            }
        )

    return [
        {
            "stage": "retrieval",
            "detail": "Hybrid retrieval completed.",
            "tool_name": "search_documents",
            "parameters": {
                "question": req.question,
                "collection_name": req.collection_name,
                "doc_language": req.doc_language,
                "top_k": req.top_k,
                "selected_doc_ids": selected_doc_ids,
            },
            "result": {
                "result_count": len(retrieval_results),
                "query_variant_count": len(query_variants),
                "query_variants": query_variants,
                "reranker_model": reranker_model,
                "by_source": by_source,
                "by_chunk_type": by_chunk_type,
                "top_hits": top_hits,
            },
            "at": now_ms,
        },
        {
            "stage": "citation",
            "detail": "Citations normalized.",
            "tool_name": "build_citations",
            "parameters": {
                "retrieval_result_count": len(retrieval_results),
                "source_record_count": len(sources),
            },
            "result": {
                "citation_count": len(sources),
                "citations_preview": citations_preview,
            },
            "at": now_ms + 1,
        },
        {
            "stage": "generation",
            "detail": "Answer generated.",
            "tool_name": "generate_answer",
            "parameters": {
                "model": str(os.environ.get("LLM_MODEL") or "gpt-4o"),
                "context_chunk_count": len(retrieval_results),
            },
            "result": {
                "answer_char_count": len(answer_text),
                "answer_preview": answer_text[:280],
            },
            "at": now_ms + 2,
        },
    ]


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    _sync_runtime_settings_from_store()
    selected_doc_ids = _normalize_selected_doc_ids(req.selected_doc_ids)
    if REQUIRE_SELECTED_DOCS_FOR_ANSWER and not selected_doc_ids:
        raise HTTPException(
            status_code=400,
            detail=(
                "No selected_doc_ids provided. "
                "To avoid cross-document contamination, please select at least one document before asking."
            ),
        )

    table_path = (
        Path(req.table_records_path)
        if req.table_records_path
        else _table_records_store_path(req.collection_name)
    )
    if not table_path.exists():
        table_path = None

    svc = _get_service(
        collection_name=req.collection_name,
        doc_language=req.doc_language,
        top_k=req.top_k,
        table_records_path=table_path,
    )
    out = svc.answer(
        question=req.question,
        top_k=req.top_k,
        selected_doc_ids=selected_doc_ids,
        stream=False,
    )

    # Enrich retrieval/sources with stable source_name so frontend labels are
    # human-readable and do not look like cross-document contamination.
    doc_name_map: dict[str, str] = {}
    for row in _list_document_records():
        doc_id = str(row.get("doc_id") or "").strip()
        if not doc_id:
            continue
        doc_name_map[doc_id] = str(row.get("source_name") or doc_id).strip() or doc_id

    normalized_results: list[dict[str, Any]] = []
    raw_results = out.get("retrieval_results", [])
    if isinstance(raw_results, list):
        for row in raw_results:
            if not isinstance(row, dict):
                continue
            item = dict(row)
            doc_id = str(item.get("doc_id") or "").strip()
            if doc_id:
                item["source_name"] = doc_name_map.get(doc_id, doc_id)
            normalized_results.append(item)

    normalized_sources: list[dict[str, Any]] = []
    raw_sources = out.get("sources", [])
    if isinstance(raw_sources, list):
        for i, row in enumerate(raw_sources):
            item = dict(row) if isinstance(row, dict) else {}
            idx_raw = int(item.get("index") or (i + 1))
            ref_idx = idx_raw - 1
            ref = normalized_results[ref_idx] if 0 <= ref_idx < len(normalized_results) else {}
            doc_id = str(item.get("doc_id") or ref.get("doc_id") or "").strip()
            if doc_id:
                item["doc_id"] = doc_id
                item["source_name"] = doc_name_map.get(
                    doc_id,
                    str(ref.get("source_name") or doc_id),
                )
            elif ref.get("source_name"):
                item["source_name"] = str(ref.get("source_name"))
            normalized_sources.append(item)

    final_question = str(out.get("question", req.question) or req.question)
    final_answer = str(out.get("answer", "") or "")
    tool_events = _build_answer_tool_events(
        req,
        selected_doc_ids=selected_doc_ids,
        retrieval_results=normalized_results,
        sources=normalized_sources,
        answer_text=final_answer,
    )

    return AnswerResponse(
        question=final_question,
        answer=final_answer,
        sources=normalized_sources,
        retrieval_results=normalized_results,
        tool_events=tool_events,
    )
