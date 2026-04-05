from __future__ import annotations

import os
import uuid
import httpx
from abc import ABC, abstractmethod

from dotenv import load_dotenv
load_dotenv()

# ── 可插拔 Embedding 接口 ──────────────────────────────────────────────────────
#
# 所有 embedder 都实现同一个 embed() 方法，上层代码只依赖这个接口。
# 新增模型只需继承 BaseEmbedder，实现 embed()，不需要改任何上层逻辑。

class BaseEmbedder(ABC):
    """所有 embedding 实现的统一接口。"""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        接收文本列表，返回对应的向量列表。
        texts[i] 对应 vectors[i]，顺序严格一致。
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度，建 Qdrant collection 时需要用到。"""
        ...


# ── OpenAI Embedder ───────────────────────────────────────────────────────────

class OpenAIEmbedder(BaseEmbedder):
    """
    调用 OpenAI 的 embedding API。
    适合英文文档，text-embedding-3-small 够用且便宜；
    中英混合时建议换 text-embedding-3-large。
    """

    # 各模型对应的向量维度，建 collection 时不用手动查文档
    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.model = model

        # 默认不读取系统代理环境变量，避免 notebook/系统代理把请求劫持到本地端口导致连接失败。
        # 如需代理，请显式设置 LLM_HTTP_PROXY，例如：http://127.0.0.1:7890
        proxy = os.environ.get("LLM_HTTP_PROXY") or None
        timeout = float(os.environ.get("LLM_TIMEOUT", "60"))

        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ.get("LLM_BASE_URL") or None,
            http_client=httpx.Client(
                proxy=proxy,
                trust_env=False,
                timeout=timeout,
            ),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # OpenAI 支持批量，一次调用返回所有向量，减少网络往返
        try:
            resp = self.client.embeddings.create(input=texts, model=self.model)
            # 返回结果按 index 排序，顺序和输入严格对应
            return [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
        except Exception as e:
            # 给 notebook 更直观的排查提示
            raise RuntimeError(
                "OpenAI embedding 连接失败。请检查："
                "1) 外网连通性；"
                "2) LLM_BASE_URL 是否可达；"
                "3) 若需要代理请设置 LLM_HTTP_PROXY，"
                "否则保持未设置以直连。"
                f" 原始错误: {type(e).__name__}: {e}"
            ) from e

    @property
    def dimension(self) -> int:
        return self.DIMENSIONS[self.model]


# ── BGE Embedder（本地，适合中英混合）────────────────────────────────────────

class BGEEmbedder(BaseEmbedder):
    """
    使用 BAAI/bge-m3，本地运行，专门针对多语言优化。
    首次使用会自动下载模型（约 2GB），之后从缓存加载。
    依赖：pip install FlagEmbedding
    """

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from FlagEmbedding import BGEM3FlagModel
        # use_fp16=True 在 GPU 上减半显存占用，CPU 上自动忽略
        self.model = BGEM3FlagModel(model_name, use_fp16=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # encode 返回一个字典，dense_vecs 是稠密向量
        output = self.model.encode(texts, batch_size=12, max_length=8192)
        return output["dense_vecs"].tolist()

    @property
    def dimension(self) -> int:
        return 1024   # bge-m3 的稠密向量维度固定是 1024


# ── Embedder 工厂：根据文档语言选择合适的实现 ─────────────────────────────────
#
# 现在用简单规则判断，接口稳定后可以把这里换成 LLM 路由，
# 上层调用代码完全不需要改动。

def get_embedder(doc_language: str = "auto") -> BaseEmbedder:
    """
    根据文档语言返回对应的 embedder 实例。

    doc_language 可选值：
        "en"    → OpenAIEmbedder（英文文档）
        "zh"    → BGEEmbedder（中文文档）
        "mixed" → BGEEmbedder（中英混合）
        "auto"  → 从环境变量 EMBEDDER 读取，默认 bge

    环境变量 EMBEDDER 可选值：openai / bge
    """
    # 优先读环境变量，方便在不改代码的情况下切换
    override = os.environ.get("EMBEDDER", "").lower()
    if override == "openai":
        return OpenAIEmbedder(os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"))
    if override == "bge":
        return BGEEmbedder()

    # 没有环境变量覆盖，按语言判断
    if doc_language == "en":
        return OpenAIEmbedder()
    else:
        # zh / mixed / auto 都走 BGE，多语言更稳
        return BGEEmbedder()


# ── Qdrant 存入 ───────────────────────────────────────────────────────────────

def _make_qdrant_client():
    """
    创建 QdrantClient，并显式禁用系统代理。
    本地 Qdrant 不需要走代理，系统代理会导致连接被拦截。
    """
    from qdrant_client import QdrantClient

    return QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6333")),
        # 这些参数会被 qdrant-client 透传给 httpx.Client
        # trust_env=False 可忽略 HTTP(S)_PROXY 等系统环境变量
        proxy=None,
        trust_env=False,
    )


def ensure_collection(client, collection_name: str, dimension: int) -> None:
    """
    确保 Qdrant 中存在指定的 collection。
    已存在就跳过，不存在就新建，避免重复建库报错。
    """
    from qdrant_client.models import Distance, VectorParams

    existing = {c.name for c in client.get_collections().collections}
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"已创建 collection: {collection_name}，向量维度: {dimension}")
    else:
        print(f"collection 已存在，跳过创建: {collection_name}")


def upsert_chunks(
    chunks:          list[dict],
    embedder:        BaseEmbedder,
    collection_name: str,
    doc_id:          str,
) -> None:
    """
    把纯文本 chunks 向量化后存入 Qdrant。

    每个 point 包含：
        vector  → embedding 向量
        payload → doc_id、heading_path、content、chunk_index
                  （content 原文单独存，召回后直接返回，不受 embedding 拼接影响）

    embedding 时在 content 前面拼上 heading_path，
    让模型知道这段内容"属于哪里"，提升检索准确率。
    """
    from qdrant_client.models import PointStruct

    qdrant = _make_qdrant_client()

    ensure_collection(qdrant, collection_name, embedder.dimension)

    # 构造用于 embedding 的文本：heading_path 提供上下文，content 提供语义
    texts_to_embed = [
        f"{c['heading_path']}\n\n{c['content']}"
        for c in chunks
    ]

    print(f"正在 embedding {len(chunks)} 个 chunks...")
    vectors = embedder.embed(texts_to_embed)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),   # 用 uuid 避免 id 冲突，支持多文档入库
            vector=vectors[i],
            payload={
                "doc_id":       doc_id,
                "chunk_index":  i,
                "heading_path": chunks[i]["heading_path"],
                "content":      chunks[i]["content"],   # 原文，检索后直接返回
            },
        )
        for i in range(len(chunks))
    ]

    # batch upsert，一次性写入，比逐条写快很多
    qdrant.upsert(collection_name=collection_name, points=points)
    print(f"已写入 {len(points)} 条向量到 collection: {collection_name}")


# ── 入口：直接运行这个文件就能完成一次完整的入库 ──────────────────────────────

if __name__ == "__main__":
    print("请在 notebook 里 import upsert_chunks 使用，或直接调用：")
    print("  from embedding import get_embedder, upsert_chunks")
    print("  embedder = get_embedder('mixed')")
    print("  upsert_chunks(text_chunks, embedder, 'my_collection', 'doc_3')")
