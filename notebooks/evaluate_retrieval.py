#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from qdrant_client import QdrantClient, models


# Reuse project embedders.
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "embedding.py").exists():
    # Script is under notebooks/
    NOTEBOOKS_DIR = SCRIPT_DIR
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    # Fallback when script is moved back to project root
    PROJECT_ROOT = SCRIPT_DIR
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

from embedding import BGEEmbedder, OpenAIEmbedder  # noqa: E402


BM25_MODEL = "Qdrant/bm25"
BM42_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"


class STEmbedder:
    """Local fallback embedder based on sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class DenseEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimension(self) -> int:
        ...


@dataclass
class RetrievalDataset:
    name: str
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, set[str]]


@dataclass
class EvalResult:
    dataset: str
    method: str
    queries: int
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    latency_ms_p50: float
    latency_ms_p95: float


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).split())


def load_beir_from_ir_datasets(dataset_id: str) -> RetrievalDataset:
    import ir_datasets

    ds = ir_datasets.load(dataset_id)

    corpus: dict[str, str] = {}
    for doc in ds.docs_iter():
        title = _clean_text(getattr(doc, "title", ""))
        body = _clean_text(getattr(doc, "text", ""))
        merged = f"{title}\n{body}".strip() if title else body
        if merged:
            corpus[str(doc.doc_id)] = merged

    queries: dict[str, str] = {}
    for q in ds.queries_iter():
        text = _clean_text(getattr(q, "text", ""))
        if text:
            queries[str(q.query_id)] = text

    qrels: dict[str, set[str]] = defaultdict(set)
    for rel in ds.qrels_iter():
        if int(rel.relevance) > 0:
            qrels[str(rel.query_id)].add(str(rel.doc_id))

    return RetrievalDataset(
        name=dataset_id,
        corpus=corpus,
        queries=queries,
        qrels=dict(qrels),
    )


def load_dureader_from_c_mteb() -> RetrievalDataset:
    from datasets import load_dataset

    corpus_ds = load_dataset("C-MTEB/DuRetrieval", split="corpus")
    queries_ds = load_dataset("C-MTEB/DuRetrieval", split="queries")
    qrels_ds = load_dataset("C-MTEB/DuRetrieval-qrels", split="dev")

    corpus = {
        str(item["id"]): _clean_text(item.get("text"))
        for item in corpus_ds
        if _clean_text(item.get("text"))
    }
    queries = {
        str(item["id"]): _clean_text(item.get("text"))
        for item in queries_ds
        if _clean_text(item.get("text"))
    }
    qrels: dict[str, set[str]] = defaultdict(set)
    for item in qrels_ds:
        if int(item["score"]) > 0:
            qrels[str(item["qid"])].add(str(item["pid"]))

    return RetrievalDataset(
        name="DuReader(C-MTEB)",
        corpus=corpus,
        queries=queries,
        qrels=dict(qrels),
    )


def _subset_dataset(
    dataset: RetrievalDataset,
    max_queries: int | None,
    max_docs: int | None,
    seed: int,
) -> RetrievalDataset:
    rng = random.Random(seed)

    valid_qids = [
        qid
        for qid in dataset.queries
        if qid in dataset.qrels and any(did in dataset.corpus for did in dataset.qrels[qid])
    ]
    rng.shuffle(valid_qids)
    if max_queries is not None and max_queries > 0:
        valid_qids = valid_qids[:max_queries]

    # Keep relevant docs first.
    kept_qids: list[str] = []
    required_doc_ids: set[str] = set()
    for qid in valid_qids:
        rel = {did for did in dataset.qrels[qid] if did in dataset.corpus}
        if not rel:
            continue
        if max_docs is not None and max_docs > 0 and len(required_doc_ids | rel) > max_docs:
            continue
        kept_qids.append(qid)
        required_doc_ids |= rel

    if not kept_qids:
        raise ValueError("No query left after subsetting; increase --max-docs/--max-queries.")

    selected_doc_ids = set(required_doc_ids)
    if max_docs is not None and max_docs > 0 and len(selected_doc_ids) < max_docs:
        candidates = [did for did in dataset.corpus if did not in selected_doc_ids]
        rng.shuffle(candidates)
        selected_doc_ids |= set(candidates[: max_docs - len(selected_doc_ids)])
    elif max_docs is None or max_docs <= 0:
        selected_doc_ids = set(dataset.corpus.keys())

    corpus = {did: dataset.corpus[did] for did in selected_doc_ids}
    queries = {qid: dataset.queries[qid] for qid in kept_qids}
    qrels = {
        qid: {did for did in dataset.qrels[qid] if did in selected_doc_ids}
        for qid in kept_qids
    }

    return RetrievalDataset(
        name=dataset.name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )


def _batched(seq: list[str], batch_size: int) -> Iterable[list[str]]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


def _rrf_fuse(rank_lists: list[list[str]], top_k: int, k: int = 60) -> list[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in rank_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def _point_id_for_doc(doc_id: str) -> str:
    # Qdrant point id accepts uint or UUID. Use deterministic UUID so re-indexing is stable.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


def _first_relevant_rank(ranked_ids: list[str], relevant: set[str], k: int) -> int | None:
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return i
    return None


def _dcg(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = 1 if doc_id in relevant else 0
        if rel:
            score += 1.0 / math.log2(i + 1)
    return score


def _idcg(relevant_count: int, k: int) -> float:
    ideal = min(relevant_count, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))


class QdrantBench:
    def __init__(self, host: str, port: int):
        self.client = QdrantClient(host=host, port=port, proxy=None, trust_env=False)

    def _check_model_available(self, model_name: str) -> bool:
        probe_name = f"_probe_{uuid.uuid4().hex[:8]}"
        try:
            self.client.create_collection(
                collection_name=probe_name,
                vectors_config={"dense": models.VectorParams(size=3, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
            self.client.upsert(
                collection_name=probe_name,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": [0.1, 0.2, 0.3],
                            "sparse": models.Document(text="probe text", model=model_name),
                        },
                        payload={"text": "probe text"},
                    )
                ],
            )
            self.client.query_points(
                collection_name=probe_name,
                query=models.Document(text="probe", model=model_name),
                using="sparse",
                limit=1,
            )
            return True
        except Exception:
            return False
        finally:
            try:
                self.client.delete_collection(probe_name)
            except Exception:
                pass

    def build_collection(
        self,
        collection_name: str,
        dataset: RetrievalDataset,
        embedder: DenseEmbedder,
        batch_size: int,
        want_bm25: bool,
        want_bm42: bool,
    ) -> dict[str, bool]:
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        bm25_enabled = bool(want_bm25 and self._check_model_available(BM25_MODEL))
        has_fastembed = importlib.util.find_spec("fastembed") is not None
        bm42_enabled = bool(want_bm42 and has_fastembed and self._check_model_available(BM42_MODEL))

        sparse_config: dict[str, models.SparseVectorParams] = {}
        if bm25_enabled:
            sparse_config["bm25"] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        if bm42_enabled:
            sparse_config["bm42"] = models.SparseVectorParams()

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=embedder.dimension, distance=models.Distance.COSINE)},
            sparse_vectors_config=sparse_config or None,
        )

        doc_ids = list(dataset.corpus.keys())
        for batch_idx, batch_doc_ids in enumerate(_batched(doc_ids, batch_size), start=1):
            texts = [dataset.corpus[did] for did in batch_doc_ids]
            dense_vectors = embedder.embed(texts)

            points: list[models.PointStruct] = []
            for i, doc_id in enumerate(batch_doc_ids):
                vector_map: dict[str, object] = {"dense": dense_vectors[i]}
                if bm25_enabled:
                    vector_map["bm25"] = models.Document(text=texts[i], model=BM25_MODEL)
                if bm42_enabled:
                    vector_map["bm42"] = models.Document(text=texts[i], model=BM42_MODEL)
                points.append(
                    models.PointStruct(
                        id=_point_id_for_doc(doc_id),
                        vector=vector_map,
                        payload={"doc_id": doc_id, "text": texts[i]},
                    )
                )
            self.client.upsert(collection_name=collection_name, points=points)
            if batch_idx % 10 == 0 or batch_idx == 1:
                print(f"[index] {collection_name}: {min(batch_idx * batch_size, len(doc_ids))}/{len(doc_ids)}")

        return {
            "bm25": bm25_enabled,
            "bm42": bm42_enabled,
        }

    def search_dense(self, collection_name: str, query_vector: list[float], limit: int) -> list[str]:
        res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using="dense",
            limit=limit,
            with_payload=["doc_id"],
        )
        doc_ids: list[str] = []
        for hit in res.points:
            payload = hit.payload or {}
            doc_ids.append(str(payload.get("doc_id", hit.id)))
        return doc_ids

    def search_sparse(self, collection_name: str, query_text: str, using: str, model: str, limit: int) -> list[str]:
        res = self.client.query_points(
            collection_name=collection_name,
            query=models.Document(text=query_text, model=model),
            using=using,
            limit=limit,
            with_payload=["doc_id"],
        )
        doc_ids: list[str] = []
        for hit in res.points:
            payload = hit.payload or {}
            doc_ids.append(str(payload.get("doc_id", hit.id)))
        return doc_ids


def _evaluate_method(
    method: str,
    dataset: RetrievalDataset,
    bench: QdrantBench,
    collection_name: str,
    embedder: DenseEmbedder,
    top_k: int,
    candidate_k: int,
    route_enabled: dict[str, bool],
) -> EvalResult:
    recall_sum = 0.0
    precision_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0
    hit_sum = 0.0
    latencies_ms: list[float] = []

    query_cache: dict[str, list[float]] = {}

    for qid, query in dataset.queries.items():
        relevant = dataset.qrels.get(qid, set())
        if not relevant:
            continue

        start = time.perf_counter()
        ranked_ids: list[str]

        if method == "dense":
            if qid not in query_cache:
                query_cache[qid] = embedder.embed([query])[0]
            ranked_ids = bench.search_dense(collection_name, query_cache[qid], candidate_k)
        elif method == "bm25":
            ranked_ids = bench.search_sparse(collection_name, query, "bm25", BM25_MODEL, candidate_k)
        elif method == "bm42":
            ranked_ids = bench.search_sparse(collection_name, query, "bm42", BM42_MODEL, candidate_k)
        elif method == "dense+bm25":
            if qid not in query_cache:
                query_cache[qid] = embedder.embed([query])[0]
            dense_rank = bench.search_dense(collection_name, query_cache[qid], candidate_k)
            bm25_rank = bench.search_sparse(collection_name, query, "bm25", BM25_MODEL, candidate_k)
            ranked_ids = _rrf_fuse([dense_rank, bm25_rank], top_k=top_k)
        elif method == "dense+bm42":
            if qid not in query_cache:
                query_cache[qid] = embedder.embed([query])[0]
            dense_rank = bench.search_dense(collection_name, query_cache[qid], candidate_k)
            bm42_rank = bench.search_sparse(collection_name, query, "bm42", BM42_MODEL, candidate_k)
            ranked_ids = _rrf_fuse([dense_rank, bm42_rank], top_k=top_k)
        else:
            raise ValueError(f"Unsupported method: {method}")

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

        ranked_k = ranked_ids[:top_k]
        hit_count = sum(1 for did in ranked_k if did in relevant)
        recall = hit_count / len(relevant) if relevant else 0.0
        precision = hit_count / top_k
        first_rank = _first_relevant_rank(ranked_k, relevant, top_k)
        mrr = (1.0 / first_rank) if first_rank is not None else 0.0
        dcg = _dcg(ranked_k, relevant, top_k)
        ideal = _idcg(len(relevant), top_k)
        ndcg = dcg / ideal if ideal > 0 else 0.0
        hit = 1.0 if hit_count > 0 else 0.0

        recall_sum += recall
        precision_sum += precision
        mrr_sum += mrr
        ndcg_sum += ndcg
        hit_sum += hit

    n = len(latencies_ms)
    if n == 0:
        raise ValueError(f"No valid query evaluated for method {method}.")

    latencies_sorted = sorted(latencies_ms)
    p50 = latencies_sorted[int(0.50 * (n - 1))]
    p95 = latencies_sorted[int(0.95 * (n - 1))]

    return EvalResult(
        dataset=dataset.name,
        method=method,
        queries=n,
        recall_at_k=recall_sum / n,
        precision_at_k=precision_sum / n,
        mrr_at_k=mrr_sum / n,
        ndcg_at_k=ndcg_sum / n,
        hit_rate_at_k=hit_sum / n,
        latency_ms_p50=p50,
        latency_ms_p95=p95,
    )


def _select_embedder(name: str):
    if name == "openai":
        return OpenAIEmbedder()
    if name == "bge":
        try:
            return BGEEmbedder()
        except Exception as e:
            raise RuntimeError(
                "Failed to initialize BGEEmbedder. "
                "Try --embedder st as a local fallback, or fix FlagEmbedding/transformers compatibility."
            ) from e
    if name == "st":
        return STEmbedder()
    raise ValueError(f"Unsupported --embedder: {name}")


def _format_result_row(r: EvalResult) -> dict[str, object]:
    return {
        "dataset": r.dataset,
        "method": r.method,
        "queries": r.queries,
        "Recall@K": round(r.recall_at_k, 4),
        "Precision@K": round(r.precision_at_k, 4),
        "MRR@K": round(r.mrr_at_k, 4),
        "NDCG@K": round(r.ndcg_at_k, 4),
        "HitRate@K": round(r.hit_rate_at_k, 4),
        "latency_ms_p50": round(r.latency_ms_p50, 2),
        "latency_ms_p95": round(r.latency_ms_p95, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate retrieval methods on BEIR and DuReader-style datasets "
            "with Recall/Precision/MRR/NDCG/HitRate."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="beir,dureader",
        help="Comma-separated dataset keys: beir,dureader",
    )
    parser.add_argument(
        "--beir-id",
        type=str,
        default="beir/scifact/test",
        help="ir_datasets id for BEIR dataset (e.g., beir/scifact/test)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="st",
        choices=["st", "bge", "openai"],
        help="Dense embedder for dense retrieval route.",
    )
    parser.add_argument("--max-docs", type=int, default=5000, help="Max corpus size per dataset.")
    parser.add_argument("--max-queries", type=int, default=200, help="Max query count per dataset.")
    parser.add_argument("--top-k", type=int, default=10, help="K for metrics.")
    parser.add_argument("--candidate-k", type=int, default=50, help="Candidates per retrieval route.")
    parser.add_argument(
        "--methods",
        type=str,
        default="dense,bm25,bm42,dense+bm25,dense+bm42",
        help="Comma-separated methods.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Indexing batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsetting.")
    parser.add_argument("--qdrant-host", type=str, default="localhost")
    parser.add_argument("--qdrant-port", type=int, default=6333)
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/retrieval_eval_results.json",
        help="Where to save JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    dataset_keys = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]
    method_list = [x.strip() for x in args.methods.split(",") if x.strip()]

    embedder = _select_embedder(args.embedder)
    bench = QdrantBench(host=args.qdrant_host, port=args.qdrant_port)

    datasets: list[RetrievalDataset] = []
    if "beir" in dataset_keys:
        datasets.append(load_beir_from_ir_datasets(args.beir_id))
    if "dureader" in dataset_keys:
        datasets.append(load_dureader_from_c_mteb())

    all_results: list[EvalResult] = []
    rows: list[dict[str, object]] = []

    for raw_ds in datasets:
        ds = _subset_dataset(
            raw_ds,
            max_queries=args.max_queries,
            max_docs=args.max_docs,
            seed=args.seed,
        )
        print(
            f"\n[dataset] {ds.name} | docs={len(ds.corpus)} | "
            f"queries={len(ds.queries)} | qrels={sum(len(v) for v in ds.qrels.values())}"
        )

        want_bm25 = any("bm25" in m for m in method_list)
        want_bm42 = any("bm42" in m for m in method_list)
        collection_name = f"eval_{ds.name.lower().replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')}_{uuid.uuid4().hex[:8]}"

        route_enabled = bench.build_collection(
            collection_name=collection_name,
            dataset=ds,
            embedder=embedder,
            batch_size=args.batch_size,
            want_bm25=want_bm25,
            want_bm42=want_bm42,
        )
        print(
            f"[index-ready] {collection_name} | bm25={route_enabled['bm25']} | bm42={route_enabled['bm42']}"
        )

        for method in method_list:
            if "bm25" in method and not route_enabled["bm25"]:
                print(f"[skip] {method}: BM25 route unavailable in current environment.")
                continue
            if "bm42" in method and not route_enabled["bm42"]:
                print(f"[skip] {method}: BM42 route unavailable in current environment.")
                continue

            print(f"[eval] {ds.name} | method={method}")
            result = _evaluate_method(
                method=method,
                dataset=ds,
                bench=bench,
                collection_name=collection_name,
                embedder=embedder,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                route_enabled=route_enabled,
            )
            all_results.append(result)
            rows.append(_format_result_row(result))

        try:
            bench.client.delete_collection(collection_name)
        except Exception:
            pass

    if not rows:
        raise RuntimeError("No result generated. Check route availability and method selection.")

    # Print markdown table for quick comparison.
    print("\n## Retrieval Evaluation")
    headers = list(rows[0].keys())
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(str(row[h]) for h in headers) + " |")

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = NOTEBOOKS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "datasets": dataset_keys,
            "beir_id": args.beir_id,
            "embedder": args.embedder,
            "max_docs": args.max_docs,
            "max_queries": args.max_queries,
            "top_k": args.top_k,
            "candidate_k": args.candidate_k,
            "methods": method_list,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "qdrant_host": args.qdrant_host,
            "qdrant_port": args.qdrant_port,
        },
        "results": [r.__dict__ for r in all_results],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[saved] {output_path}")


if __name__ == "__main__":
    main()
