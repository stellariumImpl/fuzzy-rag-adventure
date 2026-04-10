from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path
from typing import Literal
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

_REFUSAL_PATTERNS = (
    r"根据现有文档，未找到相关信息",
    r"未找到相关信息",
    r"未找到关于",
    r"没有足够信息",
    r"信息不足",
    r"无法回答该问题",
)

# ── LLM 客户端 ────────────────────────────────────────────────────────────────

def _get_llm_client_and_model() -> tuple[OpenAI, str]:
    api_key = (os.environ.get("LLM_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY for answer generation.")
    model = (os.environ.get("LLM_MODEL") or "gpt-4o").strip() or "gpt-4o"
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    return client, model


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
        chunk_type = str(chunk.get("chunk_type") or "text").strip().lower() or "text"
        source_type_for_citation = chunk_type if chunk_type in {"image", "table"} else source_type
        image_path = str(chunk.get("image_path") or "")
        page = chunk.get("page")

        # 表格 chunk 的 content 是 JSON 字符串，加一个标注让 LLM 知道这是结构化数据
        if source_type == "table" or chunk_type == "table":
            label = f"【参考资料{idx}】（结构化表格数据）来源: {heading}"
        elif chunk_type == "image":
            page_label = f", page={page}" if page not in (None, "") else ""
            path_label = f", image={image_path}" if image_path else ""
            label = f"【参考资料{idx}】（图像理解）来源: {heading}{page_label}{path_label}"
        else:
            label = f"【参考资料{idx}】来源: {heading}"

        context_parts.append(f"{label}\n{content}")
        sources.append({
            "index":        idx,
            "heading_path": heading,
            "doc_id":       chunk.get("doc_id", ""),
            "source_name":  chunk.get("source_name", chunk.get("doc_id", "")),
            "source_type":  source_type_for_citation,
            "chunk_type":   chunk_type,
            "image_path":   image_path,
            "page":         page,
            "rerank_score": chunk.get("rerank_score", chunk.get("final_score", 0)),
        })

    return "\n\n---\n\n".join(context_parts), sources


# ── Step 2：构建 prompt ───────────────────────────────────────────────────────

def build_prompt(
    question: str,
    context: str,
    answer_mode: Literal["strict", "inference"] = "strict",
) -> tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。

    设计原则：
    - system 定义角色和行为约束（只能基于参考资料回答，找不到要明说）
    - user 放 context + 问题，结构清晰
    - 要求 LLM 在回答里标注引用编号，方便溯源
    """
    normalized_mode: Literal["strict", "inference"] = (
        "inference" if answer_mode == "inference" else "strict"
    )
    if normalized_mode == "inference":
        system_prompt = """你是一个专业的文档问答助手。

你的任务是基于提供的参考资料回答用户问题。请严格遵守以下规则：
1. 只能基于参考资料中的内容回答，不要使用你自己的背景知识补充
2. 回答时用【参考资料X】标注信息来源，X是编号
3. 当证据不足以直接回答，但能基于已有线索做合理推断时，可以给出推断；推断必须单独写在【推断】段落
4. 只要出现推断，必须追加【不确定性】段落，格式为“【不确定性】高/中/低：原因”
5. 若证据很弱，也要明确“证据缺口”并给出保守推断与高不确定性；不要输出固定拒答句
6. 回答要简洁准确，不要重复参考资料原文"""
    else:
        system_prompt = """你是一个专业的文档问答助手。

你的任务是基于提供的参考资料回答用户问题。请严格遵守以下规则：
1. 只能基于参考资料中的内容回答，不要使用你自己的背景知识补充
2. 回答时用【参考资料X】标注信息来源，X是编号
3. 如果参考资料中没有足够信息回答问题，直接说"根据现有文档，未找到相关信息"，不要猜测或编造
4. 回答要简洁准确，不要重复参考资料的原文"""

    mode_hint = ""
    if normalized_mode == "inference":
        mode_hint = (
            "\n\n输出要求：\n"
            "1) 如果能直接从资料得到答案，先给直接答案。\n"
            "2) 如果需要推断，必须新增“【推断】...”段落。\n"
            "3) 只要有推断，必须新增“【不确定性】高/中/低：...”段落。\n"
            "4) 证据不足时请写明证据缺口，不要返回固定拒答句。"
        )

    user_prompt = f"""参考资料：

{context}

用户问题：{question}

请基于以上参考资料回答问题，并标注信息来源。{mode_hint}"""

    return system_prompt, user_prompt


def _looks_like_refusal(answer: str) -> bool:
    text = " ".join(str(answer or "").split())
    if not text:
        return True
    return any(re.search(pattern, text) for pattern in _REFUSAL_PATTERNS)


def _build_inference_retry_prompt(
    question: str,
    context: str,
    previous_answer: str,
) -> tuple[str, str]:
    system_prompt = """你是一个文档推断助手。

你必须基于给定参考资料产出“可追溯、保守”的推断回答，禁止直接拒答。请严格执行：
1. 不得输出“未找到相关信息/无法回答”等固定拒答句。
2. 先写“基于证据”部分，列出能确认的事实，并为关键句标注【参考资料X】。
3. 再写“【推断】”部分，明确哪些结论是从事实外推得到。
4. 最后写“【不确定性】高/中/低：原因”。
5. 若某个子问题缺证据，写“证据缺口：...”，但仍给出最保守推断。"""

    user_prompt = f"""上一版回答过于保守，请重写。

上一版回答：
{previous_answer}

参考资料：
{context}

用户问题：{question}

请按“基于证据 -> 【推断】 -> 【不确定性】”输出。"""
    return system_prompt, user_prompt


def _image_path_to_data_url(path: str) -> str | None:
    p = Path(str(path or "")).expanduser().resolve()
    if not p.exists() or not p.is_file():
        return None
    try:
        data = p.read_bytes()
    except Exception:
        return None

    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    max_side_px = int(os.environ.get("GENERATION_IMAGE_MAX_SIDE", "1024"))
    if max_side_px > 0:
        try:
            from io import BytesIO
            from PIL import Image

            with Image.open(p) as image:
                img = image.convert("RGB")
                if max(img.size) > max_side_px:
                    scale = max_side_px / float(max(img.size))
                    resized = (
                        max(1, int(img.width * scale)),
                        max(1, int(img.height * scale)),
                    )
                    img = img.resize(resized)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85, optimize=True)
                data = buffer.getvalue()
                mime = "image/jpeg"
        except Exception:
            pass

    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _collect_image_inputs(results: list[dict]) -> list[dict]:
    max_images = int(os.environ.get("GENERATION_MAX_IMAGE_INPUTS", "3"))
    if max_images <= 0:
        return []

    seen: set[str] = set()
    items: list[dict] = []
    for row in results:
        chunk_type = str(row.get("chunk_type") or "").strip().lower()
        if chunk_type != "image":
            continue
        image_path = str(row.get("image_path") or "").strip()
        if not image_path or image_path in seen:
            continue
        seen.add(image_path)
        data_url = _image_path_to_data_url(image_path)
        if not data_url:
            continue
        items.append(
            {
                "heading_path": str(row.get("heading_path") or ""),
                "image_path": image_path,
                "page": row.get("page"),
                "data_url": data_url,
            }
        )
        if len(items) >= max_images:
            break
    return items


def _build_user_message_content(user_prompt: str, image_inputs: list[dict]):
    if not image_inputs:
        return user_prompt

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"{user_prompt}\n\n"
                "你还会收到若干原图，请与参考资料共同使用；"
                "若图像与文字描述冲突，以图像中可见信息为准。"
            ),
        }
    ]
    for idx, item in enumerate(image_inputs, start=1):
        heading = item.get("heading_path", "")
        page = item.get("page")
        meta = f"来源: {heading}" if heading else "来源: image chunk"
        if page not in (None, ""):
            meta += f", page={page}"
        content.append({"type": "text", "text": f"原图{idx}（{meta}）"})
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": item["data_url"],
                    "detail": "low",
                },
            }
        )
    return content


# ── Step 3：调用 LLM 生成答案 ─────────────────────────────────────────────────

def generate(
    question:   str,
    results:    list[dict],
    stream:     bool = False,
    answer_mode: Literal["strict", "inference"] = "strict",
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
    system_prompt, user_prompt = build_prompt(
        question,
        context,
        answer_mode=answer_mode,
    )

    image_inputs = _collect_image_inputs(results)
    if stream:
        answer = _generate_stream(
            system_prompt,
            user_prompt,
            image_inputs=image_inputs,
            temperature=0.4 if answer_mode == "inference" else 0.3,
        )
    else:
        answer = _generate_sync(
            system_prompt,
            user_prompt,
            image_inputs=image_inputs,
            temperature=0.4 if answer_mode == "inference" else 0.3,
        )

    if answer_mode == "inference" and _looks_like_refusal(answer):
        retry_system, retry_user = _build_inference_retry_prompt(
            question=question,
            context=context,
            previous_answer=answer,
        )
        if stream:
            answer = _generate_stream(
                retry_system,
                retry_user,
                image_inputs=image_inputs,
                temperature=0.35,
            )
        else:
            answer = _generate_sync(
                retry_system,
                retry_user,
                image_inputs=image_inputs,
                temperature=0.35,
            )

    return {
        "answer":   answer,
        "sources":  sources,
        "question": question,
    }


def _generate_sync(
    system_prompt: str,
    user_prompt: str,
    image_inputs: list[dict] | None = None,
    temperature: float = 0.3,
) -> str:
    """非流式调用，等待完整响应后返回。"""
    client, model = _get_llm_client_and_model()
    user_content = _build_user_message_content(user_prompt, image_inputs or [])
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


def _generate_stream(
    system_prompt: str,
    user_prompt: str,
    image_inputs: list[dict] | None = None,
    temperature: float = 0.3,
) -> str:
    """
    流式调用，边生成边打印到终端。
    返回完整的答案字符串（方便后续处理）。
    """
    client, model = _get_llm_client_and_model()
    user_content = _build_user_message_content(user_prompt, image_inputs or [])
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        temperature=temperature,
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
