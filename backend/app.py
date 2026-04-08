from __future__ import annotations

import json
import os
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
ALLOWED_DOC_LANGUAGES = {"en", "zh", "mixed", "auto"}
DIRECT_INGEST_EXTENSIONS = {".md", ".txt"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
DEFAULT_USER_ID = "local-user"


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


class ChatTitleRequest(BaseModel):
    question: str
    answer: str
    current_title: str | None = None


class ChatTitleResponse(BaseModel):
    title: str


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

    chunks: list[dict[str, Any]] = []
    if parsed_dir is not None:
        chunks = _read_json_list(parsed_dir / "text_chunks.json")

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
                "chunk_level": heading or "chunk",
                "heading_path": heading,
                "char_count": len(content),
                "page_start": None,
                "page_end": None,
                "preview": content[:240],
                "text": content,
                "has_table": bool(c.get("has_table", False)),
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
        else (int(row.get("text_chunks") or 0) if status == "ready" else 0)
    )
    missing_chunks = max(0, total - embedded_chunks)

    return {
        "doc_id": doc_id,
        "status": status,
        "parse": {
            "provider": str(row.get("parse_provider") or "upload"),
            "summary": str(row.get("parse_summary") or "Uploaded document"),
            "text_chars": int(sum(len(str(c.get("content") or "")) for c in chunks)),
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
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(storage_path))
            if not reader.pages:
                return ""
            first = (reader.pages[0].extract_text() or "").strip()
            if first:
                return f"# {storage_path.stem}\n\n{first}"
        except Exception:
            return ""

    return ""


def _read_uploaded_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


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


def _materialize_uploaded_pdf(uploaded_path: Path) -> tuple[Path, int, int]:
    """
    将 PDF 抽取为多页 markdown：doc_0.md, doc_1.md, ...
    返回 (output_dir, page_count, empty_page_count)。
    """
    from pypdf import PdfReader

    reader = PdfReader(str(uploaded_path))
    page_count = len(reader.pages)
    output_dir = UPLOAD_STAGING_DIR / f"{uploaded_path.stem}_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)

    empty_pages = 0
    if page_count == 0:
        (output_dir / "doc_0.md").write_text(
            f"# {uploaded_path.stem}\n\n(empty pdf)\n",
            encoding="utf-8",
        )
        return output_dir, 0, 0

    for idx, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        text = text.strip()
        if not text:
            empty_pages += 1
            text = "(empty page or image-only page)"

        (output_dir / f"doc_{idx}.md").write_text(
            f"# {uploaded_path.stem} - page {idx + 1}\n\n{text}\n",
            encoding="utf-8",
        )

    return output_dir, page_count, empty_pages


def _materialize_uploaded_pdf_via_job_api(uploaded_path: Path) -> tuple[Path, int, int]:
    """
    按 notebooks/01_paddle_ocr_smoke_test.ipynb 的逻辑走远端解析服务：
    1) 提交本地 PDF 到 JOB_URL
    2) 轮询任务状态直到 done
    3) 下载 resultUrl.jsonUrl 的 jsonl
    4) 落盘 doc_{page}.md + markdown/images + outputImages（如 layout_det_res_0.jpg）
    """
    import requests
    import time

    job_url = (os.environ.get("JOB_URL") or "").strip()
    token = (os.environ.get("TOKEN") or "").strip()
    model = (os.environ.get("MODEL") or "").strip()
    if not all([job_url, token, model]):
        raise RuntimeError("Missing JOB_URL/TOKEN/MODEL for notebook parser.")

    headers = {
        "Authorization": f"bearer {token}",
    }
    optional_payload = {
        "useDocOrientationClassify": False,
        "useDocUnwarping": False,
        "useChartRecognition": False,
    }

    data = {
        "model": model,
        "optionalPayload": json.dumps(optional_payload, ensure_ascii=False),
    }
    with uploaded_path.open("rb") as f:
        files = {"file": f}
        resp = requests.post(job_url, headers=headers, data=data, files=files, timeout=120)
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
        poll = requests.get(f"{job_url}/{job_id}", headers=headers, timeout=30)
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

        processed = process_markdown_text(raw_text, client=client, model=model)
        text_chunks: list[dict] = processed["text_chunks"]
        table_records: list[dict] = processed["table_records"]

        _update_task(task_id, progress=65, message="embedding and upsert")

        doc_id = req.doc_id or output_dir.name
        embedder = get_embedder(doc_language=req.doc_language)
        upsert_chunks(
            chunks=text_chunks,
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
            "table_records": len(table_records),
            "table_records_store_path": str(table_path) if req.persist_table_records else None,
            "table_records_store_count": table_total,
        }

        existing_row = _find_document_record(doc_id)
        parse_provider = str(existing_row.get("parse_provider")) if existing_row is not None else "upload"
        _upsert_document_registry(
            {
                "doc_id": doc_id,
                "collection_name": req.collection_name,
                "doc_language": req.doc_language,
                "status": "ready",
                "parsed_output_dir": str(output_dir),
                "text_chunks": len(text_chunks),
                "table_records": len(table_records),
                "vector_dimension": int(embedder.dimension),
                "parse_summary": "Indexed successfully",
                "parse_provider": parse_provider,
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
        parser_mode = "pypdf"
        remote_parse_warning: str | None = None
        try:
            remote_parse_ready = all(
                [
                    (os.environ.get("JOB_URL") or "").strip(),
                    (os.environ.get("TOKEN") or "").strip(),
                    (os.environ.get("MODEL") or "").strip(),
                ]
            )
            if remote_parse_ready:
                try:
                    ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf_via_job_api(stored_path)
                    parser_mode = "notebook_job_api"
                except Exception as remote_exc:
                    remote_parse_warning = f"{type(remote_exc).__name__}: {remote_exc}".replace("\n", " ")
                    ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf(stored_path)
                    parser_mode = "pypdf_fallback_from_remote"
            else:
                ingest_output_dir, page_count, empty_pages = _materialize_uploaded_pdf(stored_path)
                parser_mode = "pypdf"
        except ModuleNotFoundError:
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
                    "parse_summary": "Uploaded only; install pypdf to enable PDF auto-ingest",
                    "parse_provider": parser_mode,
                }
            )
            return UploadResponse(
                filename=original_name,
                stored_path=str(stored_path),
                status="uploaded_only",
                message="文件已上传，但当前环境缺少 pypdf，无法自动解析 PDF。请安装后重试。",
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
            response_message = "文件已上传；远端解析失败，已自动回退本地 pypdf 并进入入库队列。"

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

    return AnswerResponse(
        question=out.get("question", req.question),
        answer=out.get("answer", ""),
        sources=normalized_sources,
        retrieval_results=normalized_results,
    )
