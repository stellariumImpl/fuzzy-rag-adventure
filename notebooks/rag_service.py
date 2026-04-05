from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .embedding import get_embedder
    from .retrieval import hybrid_search
except ImportError:
    # 兼容直接从 notebooks 目录脚本方式运行
    from embedding import get_embedder
    from retrieval import hybrid_search


def _resolve_table_records_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p

    # 优先按当前工作目录解析；不存在时回退到 notebooks 目录
    if p.exists():
        return p
    notebooks_dir = Path(__file__).resolve().parent
    p2 = notebooks_dir / p
    return p2


def load_table_records(path: str | Path) -> list[dict]:
    p = _resolve_table_records_path(path)
    if p is None or not p.exists():
        raise FileNotFoundError(f"table_records 文件不存在: {path}")

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"table_records 文件格式错误，期望 JSON 数组: {p}")
    return data


@dataclass
class SearchOutput:
    question: str
    top_k: int
    results: list[dict]


class RAGService:
    """
    轻量服务接口层（函数级），用于后端直接调用。
    逻辑保持与 notebook 实验一致：
      hybrid_search -> (可选) generate
    """

    def __init__(
        self,
        collection_name: str = "documents",
        doc_language: str = "mixed",
        table_records: list[dict] | None = None,
        table_records_path: str | Path | None = None,
        default_top_k: int = 5,
    ):
        self.collection_name = collection_name
        self.default_top_k = int(default_top_k)
        self.embedder = get_embedder(doc_language=doc_language)

        if table_records is not None:
            self.table_records = table_records
        elif table_records_path is not None:
            self.table_records = load_table_records(table_records_path)
        else:
            self.table_records = []

    def set_table_records(self, records: list[dict]) -> None:
        self.table_records = records or []

    def reload_table_records(self, path: str | Path) -> None:
        self.table_records = load_table_records(path)

    def search(
        self,
        question: str,
        top_k: int | None = None,
        selected_doc_ids: list[str] | None = None,
    ) -> SearchOutput:
        k = int(top_k or self.default_top_k)
        results = hybrid_search(
            query=question,
            embedder=self.embedder,
            collection_name=self.collection_name,
            table_records=self.table_records,
            top_k=k,
            selected_doc_ids=selected_doc_ids,
        )
        return SearchOutput(
            question=question,
            top_k=k,
            results=results,
        )

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        selected_doc_ids: list[str] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        生成最终答案，返回结构：
        {
          "question": ...,
          "answer": ...,
          "sources": [...],
          "retrieval_results": [...]
        }
        """
        search_out = self.search(
            question=question,
            top_k=top_k,
            selected_doc_ids=selected_doc_ids,
        )

        # 延迟导入，避免仅检索场景下强依赖 LLM API 环境变量
        # 同时兼容两种执行方式：
        # 1) 作为 notebooks 包被 backend 导入
        # 2) 直接在 notebooks 目录下脚本运行
        try:
            from .generation import generate
        except ImportError:
            from generation import generate

        gen = generate(
            question=question,
            results=search_out.results,
            stream=stream,
        )
        return {
            "question": question,
            "answer": gen.get("answer", ""),
            "sources": gen.get("sources", []),
            "retrieval_results": search_out.results,
        }


def create_service_from_env() -> RAGService:
    """
    用环境变量初始化服务，便于后端进程统一配置。
    - RAG_COLLECTION: default documents
    - RAG_DOC_LANGUAGE: en/zh/mixed/auto, default mixed
    - RAG_TOP_K: default 5
    - RAG_TABLE_RECORDS_PATH: 可选，例 notebooks/output/table_records.json
    """
    collection = os.environ.get("RAG_COLLECTION", "documents")
    doc_language = os.environ.get("RAG_DOC_LANGUAGE", "mixed")
    top_k = int(os.environ.get("RAG_TOP_K", "5"))
    table_path = os.environ.get("RAG_TABLE_RECORDS_PATH")

    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "doc_language": doc_language,
        "default_top_k": top_k,
    }
    if table_path:
        kwargs["table_records_path"] = table_path

    return RAGService(**kwargs)
