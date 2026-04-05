from __future__ import annotations

import json
import re
from pathlib import Path


def _page_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)$", path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def load_markdown_pages(output_dir: str | Path = "output") -> list[dict]:
    """
    读取解析产物目录中的多页 markdown（doc_0.md, doc_1.md...）。
    返回按页序排序的页面列表。
    """
    root = Path(output_dir)
    files = sorted(root.glob("doc_*.md"), key=_page_sort_key)
    pages: list[dict] = []
    for f in files:
        pages.append(
            {
                "path": str(f),
                "text": f.read_text(encoding="utf-8"),
            }
        )
    return pages


def merge_markdown_pages(pages: list[dict]) -> str:
    """
    把多页 markdown 合并成一个文本流，供后续标题修正/切块使用。
    """
    return "\n\n".join((p.get("text") or "").strip() for p in pages if (p.get("text") or "").strip())


def extract_headings(text: str) -> list[dict]:
    headings = []
    for line_no, line in enumerate(text.splitlines()):
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            headings.append(
                {
                    "level": len(match.group(1)),
                    "text": match.group(2).strip(),
                    "line": line_no,
                }
            )
    return headings


def llm_verify_headings(all_headings: list[dict], client, model: str) -> list[dict]:
    if not all_headings:
        return []

    tree_str = "\n".join(
        f"{'  ' * (h['level'] - 1)}H{h['level']}: {h['text']}" for h in all_headings
    )

    prompt = f"""下面是一份从 PDF 解析出来的 Markdown 标题树。
PDF 解析器有时会把标题层级识别错——常见错误是某个标题在语义上应该是上一个标题的子项，却被识别成了同级。

标题树：
{tree_str}

请仔细阅读这棵树，找出所有语义层级与当前 H 级别不符的标题。
判断标准：只看语义——如果一个标题描述的是某个上级标题下的具体实例或细节，它就应该是子级。

只返回需要修正的标题，格式为 JSON 数组，不要输出任何其他内容：
[
  {{"text": "标题原文", "correct_level": 正确的层级数字}},
  ...
]

如果整棵树层级都正确，返回空数组 []。"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(match.group()) if match else []


def apply_corrections(text: str, correction_map: dict) -> str:
    fixed_lines = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+)", line)
        if match:
            heading_text = match.group(2).strip()
            if heading_text in correction_map:
                new_level = correction_map[heading_text]
                line = "#" * new_level + " " + heading_text
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def split_into_chunks(text: str) -> list[dict]:
    lines = text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_heading = None
    heading_stack: list[dict] = []

    def flush(stack):
        content = "\n".join(current_chunk_lines).strip()
        if not content:
            return None
        return {
            "heading_path": " > ".join(h["text"] for h in stack),
            "content": content,
            "has_table": "<table" in content,
        }

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+)", line)

        if not match:
            current_chunk_lines.append(line)
            continue

        level = len(match.group(1))
        heading_text = match.group(2).strip()

        if current_heading is not None:
            chunk = flush(heading_stack)
            if chunk:
                chunks.append(chunk)
            current_chunk_lines = []

        while heading_stack and heading_stack[-1]["level"] >= level:
            heading_stack.pop()
        heading_stack.append({"level": level, "text": heading_text})

        current_heading = {"level": level, "text": heading_text}
        current_chunk_lines.append(line)

    chunk = flush(heading_stack)
    if chunk:
        chunks.append(chunk)

    return chunks


def extract_table_as_json(chunk: dict, client, model: str) -> list[dict]:
    prompt = f"""下面是一段包含 HTML 表格的文本，来自文档章节：{chunk['heading_path']}

{chunk['content']}

请把表格内容提取成 JSON 数组，要求：
1. 每行表格数据对应一个 JSON 对象
2. 字段名用英文，根据表头语义自行命名，简洁清晰
3. 字段值保留原文，不要改写
4. 只返回 JSON 数组，不要输出任何其他内容

"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    raw = (resp.choices[0].message.content or "").strip()
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        rows = json.loads(m.group()) if m else []

    for row in rows:
        row["_source_heading"] = chunk["heading_path"]
    return rows


def process_markdown_text(raw_text: str, client, model: str) -> dict:
    headings = extract_headings(raw_text)
    corrections = llm_verify_headings(headings, client=client, model=model)
    correction_map = {c["text"]: c["correct_level"] for c in corrections}
    fixed_text = apply_corrections(raw_text, correction_map)

    chunks = split_into_chunks(fixed_text)
    text_chunks = [c for c in chunks if not c["has_table"]]
    table_chunks = [c for c in chunks if c["has_table"]]

    all_table_records: list[dict] = []
    for chunk in table_chunks:
        all_table_records.extend(extract_table_as_json(chunk, client=client, model=model))

    return {
        "headings": headings,
        "corrections": corrections,
        "fixed_text": fixed_text,
        "chunks": chunks,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "table_records": all_table_records,
    }
