from __future__ import annotations

import os
import json
import re
import threading
import hashlib
from collections import defaultdict
from typing import Any, Iterable

from dotenv import load_dotenv

load_dotenv()

BM25_MODEL = os.environ.get("BM25_MODEL", "Qdrant/bm25")
BM42_MODEL = os.environ.get("BM42_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions")
DEFAULT_RERANKER_MODEL_EN = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_MODEL_ZH = "BAAI/bge-reranker-v2-m3"
DEFAULT_MULTI_QUERY_MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

_RERANKERS: dict[str, "_CrossEncoderReranker"] = {}
_RERANKER_LOCK = threading.Lock()
_ROUTE_WARNED: set[str] = set()
_MULTI_QUERY_CLIENT: Any | None = None
_MULTI_QUERY_CLIENT_LOCK = threading.Lock()
_MULTI_QUERY_CACHE: dict[str, list[str]] = {}
_MULTI_QUERY_CACHE_LOCK = threading.Lock()


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
    try:
        info = client.get_collection(collection_name)
    except Exception as e:
        text = str(e).lower()
        # Qdrant collection missing: degrade retrieval gracefully instead of
        # crashing the whole /answer endpoint with 500.
        if "404" in text and "doesn" in text and "collection" in text:
            print(f"提示: collection '{collection_name}' 不存在，向量检索将返回空结果。")
            return {
                "dense_using": None,
                "bm25": False,
                "bm42": False,
                "missing_collection": True,
            }
        raise

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
        "missing_collection": False,
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
                "point_id": str(hit.id),
                "content": payload.get("content", ""),
                "heading_path": payload.get("heading_path", ""),
                "doc_id": payload.get("doc_id", ""),
                "chunk_index": payload.get("chunk_index", -1),
                "score": float(hit.score),
                "source": source,
            }
        )
    return docs


def _doc_fusion_key(doc: dict) -> str:
    """
    为 RRF 融合生成稳定唯一键，避免仅按 content 去重导致串文档。
    """
    doc_id = str(doc.get("doc_id") or "")
    chunk_index = int(doc.get("chunk_index", -1))
    if doc_id and chunk_index >= 0:
        return f"{doc_id}::{chunk_index}"

    point_id = str(doc.get("point_id") or "")
    if point_id:
        return f"pid::{point_id}"

    raw = "||".join(
        [
            str(doc.get("source") or ""),
            str(doc.get("heading_path") or ""),
            str(doc.get("chunk_index", -1)),
            str(doc.get("content") or ""),
        ]
    )
    return "hash::" + hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _normalize_selected_doc_ids(selected_doc_ids: Iterable[str] | None) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    if selected_doc_ids is None:
        return ids
    for raw in selected_doc_ids:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ids.append(value)
    return ids


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _multi_query_enabled() -> bool:
    return os.environ.get("ENABLE_MULTI_QUERY", "1") != "0"


def _multi_query_count() -> int:
    return _env_int("MULTI_QUERY_COUNT", default=4, min_value=1, max_value=5)


def _compact_query_label(text: str, max_len: int = 20) -> str:
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    if len(compact) <= max_len:
        return compact
    return compact[:max_len].rstrip() + "..."


def _normalize_query_variants(base_query: str, candidates: Iterable[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        q = re.sub(r"\s+", " ", str(raw or "").strip())
        if not q:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(q)

    _add(base_query)
    for c in candidates:
        _add(c)
        if len(out) >= limit:
            break
    return out[:limit]


def _extract_queries_from_llm_text(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    # 优先按 JSON 解析（模型按提示应返回 JSON 数组）
    candidates_to_try = [text]
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        candidates_to_try.insert(0, m.group(0))

    for candidate in candidates_to_try:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass

    # 兜底：逐行解析
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned: list[str] = []
    for line in lines:
        line = re.sub(r"^\s*[-*•]+\s*", "", line)
        line = re.sub(r"^\s*\d+[\.\)\-:：]\s*", "", line)
        line = line.strip().strip("\"'`")
        if line:
            cleaned.append(line)
    return cleaned


def _get_multi_query_client():
    global _MULTI_QUERY_CLIENT
    if _MULTI_QUERY_CLIENT is not None:
        return _MULTI_QUERY_CLIENT

    api_key = os.environ.get("LLM_API_KEY")
    if not api_key:
        return None

    with _MULTI_QUERY_CLIENT_LOCK:
        if _MULTI_QUERY_CLIENT is not None:
            return _MULTI_QUERY_CLIENT
        try:
            from openai import OpenAI

            _MULTI_QUERY_CLIENT = OpenAI(
                api_key=api_key,
                base_url=os.environ.get("LLM_BASE_URL") or None,
            )
            return _MULTI_QUERY_CLIENT
        except Exception as e:
            print(f"警告: Multi Query 客户端初始化失败，已回退单查询。原因: {type(e).__name__}: {e}")
            return None


def _generate_multi_queries(query: str, total_count: int) -> list[str]:
    query = str(query or "").strip()
    if not query:
        return []
    if total_count <= 1:
        return [query]

    client = _get_multi_query_client()
    if client is None:
        return [query]

    extra_count = max(1, total_count - 1)
    model = os.environ.get("MULTI_QUERY_MODEL", DEFAULT_MULTI_QUERY_MODEL)
    temperature = float(os.environ.get("MULTI_QUERY_TEMPERATURE", "0.2"))

    if _contains_cjk(query):
        system_prompt = (
            "你是检索查询改写助手。"
            "请输出多个语义等价但表述不同的检索查询，用于提升召回。"
            "只输出 JSON 数组，不要解释。"
        )
        user_prompt = (
            f"原问题：{query}\n"
            f"请生成 {extra_count} 条不同表达的检索查询。\n"
            "要求：保持原意，不引入新事实；尽量覆盖同义词、简称、上下位词。\n"
            "输出格式示例：\n"
            '["改写1", "改写2"]'
        )
    else:
        system_prompt = (
            "You are a retrieval query rewriter. "
            "Generate semantically equivalent query variants for better recall. "
            "Output JSON array only."
        )
        user_prompt = (
            f"Original question: {query}\n"
            f"Generate {extra_count} alternative retrieval queries.\n"
            "Keep intent unchanged and avoid adding new facts.\n"
            'Return JSON array only, e.g. ["q1","q2"].'
        )

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=220,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        generated = _extract_queries_from_llm_text(content)
        variants = _normalize_query_variants(query, generated, limit=total_count)
        return variants or [query]
    except Exception as e:
        print(f"警告: Multi Query 生成失败，已回退单查询。原因: {type(e).__name__}: {e}")
        return [query]


def build_query_variants(query: str) -> list[str]:
    """
    生成查询变体（Multi Query），失败时自动退化为 [query]。
    """
    base_query = str(query or "").strip()
    if not base_query:
        return []
    if not _multi_query_enabled():
        return [base_query]

    total_count = _multi_query_count()
    cache_key = f"{total_count}::{base_query}"
    with _MULTI_QUERY_CACHE_LOCK:
        cached = _MULTI_QUERY_CACHE.get(cache_key)
    if cached is not None:
        return list(cached)

    variants = _generate_multi_queries(base_query, total_count=total_count)
    variants = _normalize_query_variants(base_query, variants, limit=total_count)
    if not variants:
        variants = [base_query]

    with _MULTI_QUERY_CACHE_LOCK:
        _MULTI_QUERY_CACHE[cache_key] = variants
    return variants


# ── Step 1：Dense + BM25 + BM42 混合检索 ─────────────────────────────────────

def vector_search(
    query: str,
    embedder,  # BaseEmbedder 实例，和入库时保持同一个
    collection_name: str,
    top_k: int = 10,
    selected_doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    在 Qdrant 里做多路检索：
      1) dense
      2) sparse bm25（可用时）
      3) sparse bm42（可用时）
    然后用 RRF 融合为一个 vector 结果列表。
    """
    from qdrant_client.models import Document, FieldCondition, Filter, MatchAny

    qdrant = _make_qdrant_client()
    route_info = _collection_route_info(qdrant, collection_name)
    if route_info.get("missing_collection"):
        return []
    _warn_if_sparse_route_missing(collection_name, route_info)
    candidate_k = top_k * 2
    scoped_doc_ids = _normalize_selected_doc_ids(selected_doc_ids)
    query_filter = None
    if scoped_doc_ids:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=scoped_doc_ids),
                )
            ]
        )

    dense_vector = embedder.embed([query])[0]
    dense_kwargs = {
        "collection_name": collection_name,
        "query": dense_vector,
        "limit": candidate_k,
        "with_payload": True,
    }
    if query_filter is not None:
        dense_kwargs["query_filter"] = query_filter
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
                query_filter=query_filter,
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
                query_filter=query_filter,
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
    selected_doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    在表格 JSON 里做关键词精确匹配（规则检索）。
    """
    if not table_records:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scoped_doc_ids = set(_normalize_selected_doc_ids(selected_doc_ids))
    scored = []
    for record in table_records:
        record_doc_id = str(record.get("_doc_id") or record.get("doc_id") or "")
        if scoped_doc_ids and record_doc_id not in scoped_doc_ids:
            continue

        record_text = " ".join(str(v) for v in record.values())
        record_tokens = _tokenize(record_text)
        hit_count = len(query_tokens & record_tokens)
        if hit_count > 0:
            scored.append((hit_count, record_doc_id, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "content": json.dumps(record, ensure_ascii=False),
            "heading_path": record.get("_source_heading", ""),
            "doc_id": record_doc_id,
            "chunk_index": -1,
            "score": hit_count / len(query_tokens),
            "source": "table",
        }
        for hit_count, record_doc_id, record in scored[:top_k]
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
            key = _doc_fusion_key(doc)
            contrib = 1.0 / (k + rank + 1)
            rrf_scores[key] += contrib
            rrf_terms[key].append(
                {
                    "route": route_name,
                    "rank": rank + 1,
                    "contrib": round(contrib, 6),
                }
            )
            if key not in doc_store:
                doc_store[key] = doc.copy()
            else:
                existing = doc_store[key]
                existing_detail = existing.get("score_detail") or {}
                incoming_detail = doc.get("score_detail") or {}
                for d_key, d_value in incoming_detail.items():
                    if d_key not in existing_detail:
                        existing_detail[d_key] = d_value
                if existing_detail:
                    existing["score_detail"] = existing_detail

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


# ── 对外接口：Multi Query + RAG-Fusion + 模型重排 ───────────────────────────

def _single_query_recall(
    query: str,
    embedder,
    collection_name: str,
    table_records: list[dict],
    candidate_k: int,
    top_k: int,
    selected_doc_ids: list[str] | None,
) -> list[dict]:
    """
    单个查询的召回流程：
      vector(dense/bm25/bm42) + table -> RRF 融合。
    """
    vector_results = vector_search(
        query=query,
        embedder=embedder,
        collection_name=collection_name,
        top_k=candidate_k,
        selected_doc_ids=selected_doc_ids,
    )
    table_results = table_search(
        query,
        table_records,
        top_k=max(top_k * 2, 10),
        selected_doc_ids=selected_doc_ids,
    )

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
    return fused


def hybrid_search(
    query: str,
    embedder,
    collection_name: str,
    table_records: list[dict],
    top_k: int = 5,
    selected_doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    主链路：
      1) Multi Query（可关闭/失败自动降级为单查询）
      2) 每个查询执行 vector+table 召回并做 RRF
      3) RAG-Fusion：跨查询做 RRF 融合
      4) 用原始问题做 cross-encoder 重排
    """
    candidate_k = max(top_k * 6, 20)
    query_variants = build_query_variants(query)
    if not query_variants:
        query_variants = [query]

    # 单查询模式：保持与历史流程一致
    if len(query_variants) == 1:
        fused = _single_query_recall(
            query=query,
            embedder=embedder,
            collection_name=collection_name,
            table_records=table_records,
            candidate_k=candidate_k,
            top_k=top_k,
            selected_doc_ids=selected_doc_ids,
        )
    else:
        per_query_lists: list[list[dict]] = []
        route_names: list[str] = []

        for idx, query_variant in enumerate(query_variants):
            recalled = _single_query_recall(
                query=query_variant,
                embedder=embedder,
                collection_name=collection_name,
                table_records=table_records,
                candidate_k=candidate_k,
                top_k=top_k,
                selected_doc_ids=selected_doc_ids,
            )
            if not recalled:
                continue
            for item in recalled:
                score_detail = item.get("score_detail") or {}
                score_detail["expanded_query"] = query_variant
                item["score_detail"] = score_detail
            per_query_lists.append(recalled)
            route_names.append(f"mq{idx + 1}:{_compact_query_label(query_variant)}")

        if not per_query_lists:
            fused = []
        elif len(per_query_lists) == 1:
            fused = per_query_lists[0]
        else:
            fused = reciprocal_rank_fusion(
                result_lists=per_query_lists,
                top_k=candidate_k,
                route_names=route_names,
                detail_key="multi_query_rrf_terms",
            )
            for item in fused:
                item["source"] = "rag_fusion"

    reranked = rerank_with_model(query=query, results=fused, top_k=top_k)
    for item in reranked:
        score_detail = item.get("score_detail") or {}
        score_detail["query_variants"] = query_variants
        score_detail["query_variant_count"] = len(query_variants)
        item["score_detail"] = score_detail
    return reranked


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

        multi_query_terms = (r.get("score_detail") or {}).get("multi_query_rrf_terms", [])
        if multi_query_terms:
            mq_text = " + ".join(
                f"{t['route']}@rank{t['rank']}:{t['contrib']}"
                for t in multi_query_terms
            )
            print(f"RRF构成(MultiQuery融合): {mq_text}")

        query_variants = (r.get("score_detail") or {}).get("query_variants", [])
        if query_variants:
            print("查询改写: " + " | ".join(str(q) for q in query_variants))

        reranker_model = (r.get("score_detail") or {}).get("reranker_model")
        if reranker_model:
            print(f"Reranker模型: {reranker_model}")

        print(f"{'='*60}")
        print(r["content"][:300], "..." if len(r["content"]) > 300 else "")
