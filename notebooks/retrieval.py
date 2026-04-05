from __future__ import annotations

import os
import json
import re
import threading
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

BM25_MODEL = os.environ.get("BM25_MODEL", "Qdrant/bm25")
BM42_MODEL = os.environ.get("BM42_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
DEFAULT_RERANKER_MODEL_EN = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_MODEL_ZH = "BAAI/bge-reranker-v2-m3"

_RERANKERS: dict[str, "_CrossEncoderReranker"] = {}
_RERANKER_LOCK = threading.Lock()
_ROUTE_WARNED: set[str] = set()


# ── Qdrant 客户端（绕过系统代理，和 embedding.py 保持一致）──────────────────

def _make_qdrant_client():
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        proxy=None,
        trust_env=False,
        check_compatibility=False,
    )


def _collection_route_info(client, collection_name: str) -> dict[str, object]:
    """
    读取 collection schema，判断 dense/bm25/bm42 哪些路由可用。
    """
    info = client.get_collection(collection_name)

    sparse_cfg = info.config.params.sparse_vectors or {}
    has_bm25 = "bm25" in sparse_cfg
    has_bm42 = "bm42" in sparse_cfg

    dense_using = None
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        if "dense" in vectors_cfg:
            dense_using = "dense"
        elif len(vectors_cfg) == 1:
            dense_using = next(iter(vectors_cfg.keys()))

    return {
        "dense_using": dense_using,
        "bm25": has_bm25,
        "bm42": has_bm42,
    }


def _warn_if_sparse_route_missing(collection_name: str, route_info: dict[str, object]) -> None:
    """
    当环境期望启用稀疏路由但 collection 未配置时，给出一次性告警。
    """
    want_bm25 = os.environ.get("ENABLE_BM25", "1") != "0"
    want_bm42 = os.environ.get("ENABLE_BM42", "1") != "0"

    missing: list[str] = []
    if want_bm25 and not route_info.get("bm25"):
        missing.append("bm25")
    if want_bm42 and not route_info.get("bm42"):
        missing.append("bm42")

    if not missing:
        return

    warn_key = f"{collection_name}:{','.join(sorted(missing))}"
    if warn_key in _ROUTE_WARNED:
        return
    _ROUTE_WARNED.add(warn_key)
    print(
        f"提示: collection '{collection_name}' 当前未启用 {missing} 路由。"
        "若需启用，请删除后用最新 embedding.py 重新入库。"
    )


def _hits_to_docs(hits, source: str) -> list[dict]:
    docs: list[dict] = []
    for hit in hits:
        payload = hit.payload or {}
        docs.append(
            {
                "content": payload.get("content", ""),
                "heading_path": payload.get("heading_path", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "score": float(hit.score),
                "source": source,
            }
        )
    return docs


# ── Step 1：Dense + BM25 + BM42 混合检索 ─────────────────────────────────────

def vector_search(
    query: str,
    embedder,  # BaseEmbedder 实例，和入库时保持同一个
    collection_name: str,
    top_k: int = 10,
) -> list[dict]:
    """
    在 Qdrant 里做多路检索：
      1) dense
      2) sparse bm25（可用时）
      3) sparse bm42（可用时）
    然后用 RRF 融合为一个 vector 结果列表。
    """
    from qdrant_client.models import Document

    qdrant = _make_qdrant_client()
    route_info = _collection_route_info(qdrant, collection_name)
    _warn_if_sparse_route_missing(collection_name, route_info)
    candidate_k = top_k * 2

    dense_vector = embedder.embed([query])[0]
    dense_kwargs = {
        "collection_name": collection_name,
        "query": dense_vector,
        "limit": candidate_k,
        "with_payload": True,
    }
    if route_info["dense_using"] is not None:
        dense_kwargs["using"] = route_info["dense_using"]
    dense_resp = qdrant.query_points(**dense_kwargs)

    result_lists: list[list[dict]] = [_hits_to_docs(dense_resp.points, source="dense")]
    route_names: list[str] = ["dense"]

    if route_info["bm25"]:
        try:
            bm25_resp = qdrant.query_points(
                collection_name=collection_name,
                query=Document(text=query, model=BM25_MODEL),
                using="bm25",
                limit=candidate_k,
                with_payload=True,
            )
            result_lists.append(_hits_to_docs(bm25_resp.points, source="bm25"))
            route_names.append("bm25")
        except Exception as e:
            print(f"警告: BM25 检索失败，已跳过。原因: {type(e).__name__}: {e}")

    if route_info["bm42"]:
        try:
            bm42_resp = qdrant.query_points(
                collection_name=collection_name,
                query=Document(text=query, model=BM42_MODEL),
                using="bm42",
                limit=candidate_k,
                with_payload=True,
            )
            result_lists.append(_hits_to_docs(bm42_resp.points, source="bm42"))
            route_names.append("bm42")
        except Exception as e:
            print(f"警告: BM42 检索失败，已跳过。原因: {type(e).__name__}: {e}")

    if len(result_lists) == 1:
        return result_lists[0][:top_k]

    fused = reciprocal_rank_fusion(
        result_lists=result_lists,
        top_k=top_k,
        route_names=route_names,
        detail_key="vector_rrf_terms",
    )
    for item in fused:
        item["source"] = "vector_hybrid"
    return fused


# ── Step 2：表格精确匹配 ───────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """
    中英混合轻量分词：
      - 英文/数字：按单词切分
      - 中文：按连续汉字做 2-gram
    """
    if not text:
        return set()

    text_lower = text.lower()
    en_tokens = re.findall(r"[a-z0-9_]+", text_lower)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_bigrams = [zh_chars[i] + zh_chars[i + 1] for i in range(len(zh_chars) - 1)]
    return set(en_tokens + zh_bigrams)


def table_search(
    query: str,
    table_records: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    在表格 JSON 里做关键词精确匹配（规则检索）。
    """
    if not table_records:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for record in table_records:
        record_text = " ".join(str(v) for v in record.values())
        record_tokens = _tokenize(record_text)
        hit_count = len(query_tokens & record_tokens)
        if hit_count > 0:
            scored.append((hit_count, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "content": json.dumps(record, ensure_ascii=False),
            "heading_path": record.get("_source_heading", ""),
            "doc_id": "",
            "chunk_index": -1,
            "score": hit_count / len(query_tokens),
            "source": "table",
        }
        for hit_count, record in scored[:top_k]
    ]


# ── Step 3：RRF 融合排序 ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
    top_k: int = 5,
    route_names: list[str] | None = None,
    detail_key: str = "rrf_terms",
) -> list[dict]:
    """
    Reciprocal Rank Fusion：把多路检索结果合并成一个排序列表。
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    rrf_terms: dict[str, list[dict]] = defaultdict(list)
    doc_store: dict[str, dict] = {}

    for route_idx, result_list in enumerate(result_lists):
        route_name = (
            route_names[route_idx]
            if route_names and route_idx < len(route_names)
            else f"route_{route_idx + 1}"
        )
        for rank, doc in enumerate(result_list):
            key = doc["content"]
            contrib = 1.0 / (k + rank + 1)
            rrf_scores[key] += contrib
            rrf_terms[key].append(
                {
                    "route": route_name,
                    "rank": rank + 1,
                    "contrib": round(contrib, 6),
                }
            )
            doc_store[key] = doc

    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        doc = doc_store[key].copy()
        doc["rrf_score"] = round(rrf_scores[key], 4)
        score_detail = doc.get("score_detail") or {}
        score_detail[detail_key] = rrf_terms[key]
        doc["score_detail"] = score_detail
        results.append(doc)

    return results


# ── Step 4：模型重排序（Cross-Encoder Reranker）──────────────────────────────

class _CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        max_len = int(os.environ.get("RERANKER_MAX_LENGTH", "512"))
        self.model_name = model_name
        self.model = CrossEncoder(
            model_name,
            max_length=max_len,
            trust_remote_code=True,
        )

    def score(self, query: str, docs: list[dict]) -> list[float]:
        pairs = [
            [query, f"{d.get('heading_path', '')}\n\n{d.get('content', '')}"]
            for d in docs
        ]
        scores = self.model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _pick_reranker_model(query: str) -> str:
    """
    优先级：
      1) RERANKER_MODEL（全局强制）
      2) 中文 query -> RERANKER_MODEL_ZH（默认 bge-reranker-v2-m3）
      3) 非中文 query -> RERANKER_MODEL_EN（默认 ms-marco miniLM）
    """
    forced = os.environ.get("RERANKER_MODEL")
    if forced:
        return forced

    if _contains_cjk(query):
        return os.environ.get("RERANKER_MODEL_ZH", DEFAULT_RERANKER_MODEL_ZH)
    return os.environ.get("RERANKER_MODEL_EN", DEFAULT_RERANKER_MODEL_EN)


def _get_reranker(model_name: str) -> _CrossEncoderReranker:
    cached = _RERANKERS.get(model_name)
    if cached is not None:
        return cached

    with _RERANKER_LOCK:
        cached = _RERANKERS.get(model_name)
        if cached is not None:
            return cached
        try:
            cached = _CrossEncoderReranker(model_name)
        except Exception as e:
            raise RuntimeError(
                f"加载 reranker 模型失败: {model_name}。"
                f"请检查网络或切换 RERANKER_MODEL / RERANKER_MODEL_ZH / RERANKER_MODEL_EN。"
            ) from e
        _RERANKERS[model_name] = cached
        return cached


def rerank_with_model(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    """
    使用 cross-encoder 对候选结果精排。
    """
    if not results:
        return []

    primary_model = _pick_reranker_model(query)
    try:
        reranker = _get_reranker(primary_model)
    except Exception:
        # 中文模型加载失败时兜底回英文模型，避免整条检索链路中断
        fallback_model = os.environ.get("RERANKER_MODEL_EN", DEFAULT_RERANKER_MODEL_EN)
        if primary_model == fallback_model:
            raise
        print(
            f"警告: reranker '{primary_model}' 不可用，已回退到 '{fallback_model}'。"
        )
        reranker = _get_reranker(fallback_model)
    scores = reranker.score(query, results)

    scored: list[dict] = []
    for idx, item in enumerate(results):
        doc = item.copy()
        doc["rerank_score"] = round(scores[idx], 6)
        doc["final_score"] = doc["rerank_score"]
        doc["score_detail"] = doc.get("score_detail", {})
        doc["score_detail"]["reranker_model"] = reranker.model_name
        scored.append(doc)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]


# ── 对外接口：三路融合检索 + 模型重排 ────────────────────────────────────────

def hybrid_search(
    query: str,
    embedder,
    collection_name: str,
    table_records: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    主链路：
      dense/bm25/bm42 多路召回 -> RRF 融合 -> cross-encoder 重排。
    """
    candidate_k = max(top_k * 6, 20)

    vector_results = vector_search(
        query=query,
        embedder=embedder,
        collection_name=collection_name,
        top_k=candidate_k,
    )
    table_results = table_search(query, table_records, top_k=top_k * 2)

    result_lists = [vector_results]
    route_names = ["vector"]
    if table_results:
        result_lists.append(table_results)
        route_names.append("table")

    fused = reciprocal_rank_fusion(
        result_lists=result_lists,
        top_k=candidate_k,
        route_names=route_names,
        detail_key="fusion_rrf_terms",
    )

    return rerank_with_model(query=query, results=fused, top_k=top_k)


# ── 打印检索结果，调试用 ──────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    for i, r in enumerate(results):
        print(f"\n{'='*60}")
        print(
            f"#{i+1}  来源: {r['source']}  |  Final: {r.get('final_score', '-')}  "
            f"|  Rerank: {r.get('rerank_score', '-')}  |  RRF: {r.get('rrf_score', '-')}  "
            f"|  路径: {r['heading_path']}"
        )

        fusion_terms = (r.get("score_detail") or {}).get("fusion_rrf_terms", [])
        if fusion_terms:
            fusion_text = " + ".join(
                f"{t['route']}@rank{t['rank']}:{t['contrib']}"
                for t in fusion_terms
            )
            print(f"RRF构成(总融合): {fusion_text}")

        vector_terms = (r.get("score_detail") or {}).get("vector_rrf_terms", [])
        if vector_terms:
            vector_text = " + ".join(
                f"{t['route']}@rank{t['rank']}:{t['contrib']}"
                for t in vector_terms
            )
            print(f"RRF构成(向量内部): {vector_text}")

        reranker_model = (r.get("score_detail") or {}).get("reranker_model")
        if reranker_model:
            print(f"Reranker模型: {reranker_model}")

        print(f"{'='*60}")
        print(r["content"][:300], "..." if len(r["content"]) > 300 else "")
