from __future__ import annotations

import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# ── LLM 客户端 ────────────────────────────────────────────────────────────────

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL") or None,
)
MODEL = os.environ.get("LLM_MODEL", "gpt-4o")


# ── Step 1：把检索结果拼成带编号和来源标注的 context ──────────────────────────

def build_context(results: list[dict]) -> tuple[str, list[dict]]:
    """
    把 hybrid_search 返回的 chunks 拼成 LLM 可以直接读的 context 字符串。

    每个 chunk 前面加编号和来源路径，让 LLM 知道内容从哪里来，
    生成答案时可以准确标注引用。

    返回：
        context_str  拼好的字符串，直接塞进 prompt
        sources      来源列表，answer 后处理时附在结尾
    """
    context_parts = []
    sources = []

    for i, chunk in enumerate(results):
        idx = i + 1
        heading = chunk.get("heading_path", "未知章节")
        content = chunk.get("content", "").strip()
        source_type = chunk.get("source", "")

        # 表格 chunk 的 content 是 JSON 字符串，加一个标注让 LLM 知道这是结构化数据
        if source_type == "table":
            label = f"【参考资料{idx}】（结构化表格数据）来源: {heading}"
        else:
            label = f"【参考资料{idx}】来源: {heading}"

        context_parts.append(f"{label}\n{content}")
        sources.append({
            "index":        idx,
            "heading_path": heading,
            "source_type":  source_type,
            "rerank_score": chunk.get("rerank_score", chunk.get("final_score", 0)),
        })

    return "\n\n---\n\n".join(context_parts), sources


# ── Step 2：构建 prompt ───────────────────────────────────────────────────────

def build_prompt(question: str, context: str) -> tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。

    设计原则：
    - system 定义角色和行为约束（只能基于参考资料回答，找不到要明说）
    - user 放 context + 问题，结构清晰
    - 要求 LLM 在回答里标注引用编号，方便溯源
    """
    system_prompt = """你是一个专业的文档问答助手。

你的任务是基于提供的参考资料回答用户问题。请严格遵守以下规则：
1. 只能基于参考资料中的内容回答，不要使用你自己的背景知识补充
2. 回答时用【参考资料X】标注信息来源，X是编号
3. 如果参考资料中没有足够信息回答问题，直接说"根据现有文档，未找到相关信息"，不要猜测或编造
4. 回答要简洁准确，不要重复参考资料的原文"""

    user_prompt = f"""参考资料：

{context}

用户问题：{question}

请基于以上参考资料回答问题，并标注信息来源。"""

    return system_prompt, user_prompt


# ── Step 3：调用 LLM 生成答案 ─────────────────────────────────────────────────

def generate(
    question:   str,
    results:    list[dict],
    stream:     bool = False,
) -> dict:
    """
    主入口：接收检索结果，生成带引用的答案。

    参数：
        question  用户问题
        results   hybrid_search 返回的 chunks
        stream    是否流式输出（True 时边生成边打印）

    返回字典：
        answer    LLM 生成的答案文本
        sources   引用来源列表（index、heading_path、rerank_score）
        question  原始问题（方便调用方记录）
    """
    if not results:
        return {
            "answer":   "未检索到相关文档，无法回答该问题。",
            "sources":  [],
            "question": question,
        }

    context, sources = build_context(results)
    system_prompt, user_prompt = build_prompt(question, context)

    if stream:
        answer = _generate_stream(system_prompt, user_prompt)
    else:
        answer = _generate_sync(system_prompt, user_prompt)

    return {
        "answer":   answer,
        "sources":  sources,
        "question": question,
    }


def _generate_sync(system_prompt: str, user_prompt: str) -> str:
    """非流式调用，等待完整响应后返回。"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,   # 事实类问答降低随机性，答案更稳定
    )
    return resp.choices[0].message.content.strip()


def _generate_stream(system_prompt: str, user_prompt: str) -> str:
    """
    流式调用，边生成边打印到终端。
    返回完整的答案字符串（方便后续处理）。
    """
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        stream=True,
    )

    print("\n── 生成答案 ──────────────────────────────────────────")
    full_answer = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
            full_answer.append(delta)
    print("\n")

    return "".join(full_answer)


# ── 打印最终结果 ──────────────────────────────────────────────────────────────

def print_response(response: dict) -> None:
    """格式化打印 generate() 的返回结果。"""
    print(f"\n{'='*60}")
    print(f"问题: {response['question']}")
    print(f"{'='*60}")
    print(f"\n{response['answer']}")

    if response["sources"]:
        print(f"\n── 引用来源 ──────────────────────────────────────────")
        for s in response["sources"]:
            print(
                f"  [{s['index']}] {s['heading_path']}"
                f"  (相关度: {s['rerank_score']:.3f})"
            )
