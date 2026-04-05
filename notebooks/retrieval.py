from __future__ import annotations

import os
import json
import re
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# ── 说明 ──────────────────────────────────────────────────────────────────────
#
# 三路并行检索，结果用 RRF 融合排序：
#
#   1. Dense Vector   语义相似度，靠 embedding 向量
#   2. Sparse BM42    关键词匹配，靠 Qdrant 内置稀疏向量（BM25 改进版）
#   3. 表格精确匹配   结构化数据，靠 JSON 文件关键词查找
#
# Dense + Sparse 在 Qdrant 里一次请求完成（原生 hybrid 支持）。
# 表格查询独立运行，结果统一丢进 RRF 融合。


# ── Qdrant 客户端（绕过系统代理，和 embedding.py 保持一致）──────────────────

def _make_qdrant_client():
    from qdrant_client import QdrantClient
    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        proxy=None,
        trust_env=False,
    )


# ── Step 1：Dense + Sparse 混合检索（Qdrant 原生）────────────────────────────

def vector_search(
    query:           str,
    embedder,                    # BaseEmbedder 实例，和入库时保持同一个
    collection_name: str,
    top_k:           int = 10,   # 每路多召回一些，融合后再截断
) -> list[dict]:
    """
    在 Qdrant 里执行 dense 向量检索。
    （sparse/BM42 预留在后续版本接入）

    返回格式：[{"content": ..., "heading_path": ..., "score": ..., "source": "vector"}, ...]
    """
    qdrant = _make_qdrant_client()

    # 把问题 embed 成稠密向量
    dense_vector = embedder.embed([query])[0]

    # 当前先走 dense 检索（兼容 qdrant-client 1.17.x）
    # sparse 路由后续补齐索引后再接入
    results = qdrant.query_points(
        collection_name=collection_name,
        query=dense_vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "content":      hit.payload["content"],
            "heading_path": hit.payload["heading_path"],
            "doc_id":       hit.payload.get("doc_id", ""),
            "chunk_index":  hit.payload.get("chunk_index", -1),
            "score":        hit.score,
            "source":       "vector",
        }
        for hit in results.points
    ]


# ── Step 2：表格精确匹配 ───────────────────────────────────────────────────────

def table_search(
    query:            str,
    table_records:    list[dict],
    top_k:            int = 5,
) -> list[dict]:
    """
    在表格 JSON 里做关键词精确匹配。
    逻辑：把查询词拆成 token，统计每条记录命中多少个 token，按命中数排序。

    不用向量，不调 LLM，纯字符串匹配，适合"DISTINCTION 是多少分"这类精确查询。
    """
    if not table_records:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = []
    for record in table_records:
        # 把这条记录的所有字段值拼成一个字符串，统一匹配
        record_text = " ".join(str(v) for v in record.values())
        record_tokens = _tokenize(record_text)
        hit_count = len(query_tokens & record_tokens)

        if hit_count > 0:
            scored.append((hit_count, record))

    # 按命中数降序，取 top_k
    scored.sort(key=lambda x: x[0], reverse=True)

    return [
        {
            "content":      json.dumps(record, ensure_ascii=False),
            "heading_path": record.get("_source_heading", ""),
            "doc_id":       "",
            "chunk_index":  -1,
            "score":        hit_count / len(query_tokens),  # 归一化到 0~1
            "source":       "table",
        }
        for hit_count, record in scored[:top_k]
    ]


# ── Step 3：RRF 融合排序 ───────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k:            int = 60,    # RRF 的平滑参数，通常取 60，不需要调
    top_k:        int = 5,
    route_names:  list[str] | None = None,
) -> list[dict]:
    """
    Reciprocal Rank Fusion：把多路检索结果合并成一个排序列表。

    核心公式：score(doc) = Σ 1 / (k + rank_in_list_i)
    每路结果里排名越靠前的 doc，对最终得分贡献越大。
    出现在多路结果里的 doc 得分叠加，天然做到了结果增强。

    用 content 作为 doc 的唯一标识（不同 source 的同一内容会被合并）。
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    rrf_terms: dict[str, list[dict]] = defaultdict(list)
    doc_store: dict[str, dict] = {}   # 存原始 doc，方便最后还原

    for route_idx, result_list in enumerate(result_lists):
        route_name = (
            route_names[route_idx]
            if route_names and route_idx < len(route_names)
            else f"route_{route_idx + 1}"
        )
        for rank, doc in enumerate(result_list):
            key = doc["content"]
            contrib = 1.0 / (k + rank + 1)   # rank 从 0 开始，+1 避免除以 k
            rrf_scores[key] += contrib
            rrf_terms[key].append({
                "route": route_name,
                "rank": rank + 1,
                "contrib": round(contrib, 6),
            })
            doc_store[key] = doc

    # 按 RRF 得分降序排列
    sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        doc = doc_store[key].copy()
        doc["rrf_score"] = round(rrf_scores[key], 4)
        doc["score_detail"] = {
            "rrf_terms": rrf_terms[key],
        }
        results.append(doc)

    return results


# ── 通用 query-aware 重排（无业务词硬编码）────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """
    中英混合轻量分词：
      - 英文/数字：按单词切分
      - 中文：按连续汉字做 2-gram，兼顾短词召回
    """
    if not text:
        return set()

    text_lower = text.lower()
    en_tokens = re.findall(r"[a-z0-9_]+", text_lower)
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    zh_bigrams = [zh_chars[i] + zh_chars[i + 1] for i in range(len(zh_chars) - 1)]
    return set(en_tokens + zh_bigrams)


def _query_aware_rerank(results: list[dict], query: str) -> list[dict]:
    """
    通用重排：根据 query 与 heading/content 的词面重合度做轻量加分。
    不依赖任何业务关键词，不改变召回集合。
    """
    query_tokens = _tokenize(query)

    def _overlap_ratio(tokens: set[str]) -> float:
        if not tokens or not query_tokens:
            return 0.0
        return len(query_tokens & tokens) / len(query_tokens)

    scored: list[tuple[float, dict]] = []
    for item in results:
        heading = item.get("heading_path", "") or ""
        content = item.get("content", "") or ""

        heading_overlap = _overlap_ratio(_tokenize(heading))
        # 只取前 400 字符做词面特征，避免长正文噪声过大
        content_overlap = _overlap_ratio(_tokenize(content[:400]))
        lexical_score = heading_overlap * 0.7 + content_overlap * 0.3

        base_rrf = float(item.get("rrf_score", 0.0))
        lexical_bonus = lexical_score * 0.2
        final_score = base_rrf + lexical_bonus

        reranked = item.copy()
        reranked["heading_overlap"] = round(heading_overlap, 4)
        reranked["content_overlap"] = round(content_overlap, 4)
        reranked["lexical_score"] = round(lexical_score, 4)
        reranked["lexical_bonus"] = round(lexical_bonus, 4)
        reranked["final_score"] = round(final_score, 4)

        scored.append((final_score, reranked))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored]


# ── 对外接口：三路融合检索 ────────────────────────────────────────────────────

def hybrid_search(
    query:            str,
    embedder,
    collection_name:  str,
    table_records:    list[dict],
    top_k:            int = 5,
) -> list[dict]:
    """
    三路并行检索 + RRF 融合，对外暴露的唯一接口。

    调用方不需要知道内部有几路检索，只管传问题进来，拿结果出去。

    参数：
        query            用户问题
        embedder         和入库时相同的 embedder 实例
        collection_name  Qdrant collection 名
        table_records    all_table_records（从 chunking notebook 里传入）
        top_k            最终返回几条结果
    """
    # 三路并行（Python 单线程顺序跑，数量少不值得引入并发）
    dense_results = vector_search(query, embedder, collection_name, top_k=top_k * 2)
    table_results = table_search(query, table_records, top_k=top_k)

    # RRF 融合，dense 和 table 两路
    # sparse 单独建索引后可以作为第三路加进来，接口不用改
    fused = reciprocal_rank_fusion(
        result_lists=[dense_results, table_results],
        top_k=top_k,
        route_names=["vector", "table"],
    )

    # 通用 query-aware 重排，不依赖业务关键词
    return _query_aware_rerank(fused, query)


# ── 打印检索结果，调试用 ──────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    for i, r in enumerate(results):
        print(f"\n{'='*60}")
        print(
            f"#{i+1}  来源: {r['source']}  |  Final: {r.get('final_score', '-')}  "
            f"|  RRF: {r.get('rrf_score', '-')}  |  路径: {r['heading_path']}"
        )
        print(
            "分数构成: "
            f"final = rrf({r.get('rrf_score', 0)}) + lexical_bonus({r.get('lexical_bonus', 0)}) "
            f"[heading_overlap={r.get('heading_overlap', 0)}, content_overlap={r.get('content_overlap', 0)}]"
        )
        terms = (r.get("score_detail") or {}).get("rrf_terms", [])
        if terms:
            term_text = " + ".join(
                f"{t['route']}@rank{t['rank']}:{t['contrib']}"
                for t in terms
            )
            print(f"RRF构成: {term_text}")
        print(f"{'='*60}")
        print(r["content"][:300], "..." if len(r["content"]) > 300 else "")
