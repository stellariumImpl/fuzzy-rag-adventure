from __future__ import annotations

import base64
import hashlib
import json
import html as html_lib
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _page_sort_key(path: Path) -> tuple[int, str]:
    m = re.search(r"(\d+)$", path.stem)
    if m:
        return (int(m.group(1)), path.name)
    return (10**9, path.name)


def _env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(min_value, min(max_value, value))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _image_vlm_timeout_sec() -> int:
    return _env_int("IMAGE_VLM_TIMEOUT_SEC", default=20, min_value=5, max_value=300)


def _image_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 128)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_image_cache_path(output_dir: str | Path) -> Path:
    raw = str(os.environ.get("IMAGE_VLM_CACHE_PATH", "")).strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else (Path.cwd() / p).resolve()
    return (Path(output_dir).resolve() / "image_vlm_cache.json")


def _load_image_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, dict] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def _save_image_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_version() -> str:
    return str(os.environ.get("IMAGE_VLM_CACHE_VERSION", "v1")).strip() or "v1"


def _image_cache_key(image_sha256: str, gate_model: str, describe_model: str) -> str:
    return f"{_cache_version()}::{image_sha256}::{gate_model}::{describe_model}"


def _extract_image_refs(markdown_text: str) -> list[str]:
    refs: list[str] = []
    refs.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", markdown_text))
    refs.extend(re.findall(r"<img\s+[^>]*src=[\"']([^\"']+)[\"'][^>]*>", markdown_text, flags=re.IGNORECASE))

    cleaned: list[str] = []
    for raw in refs:
        value = html_lib.unescape(str(raw or "").strip()).strip().strip("\"'")
        if not value:
            continue
        if value.startswith("data:"):
            continue
        value = value.split("#", 1)[0].split("?", 1)[0].strip()
        if not value:
            continue
        cleaned.append(value)
    return cleaned


def _extract_page_heading(page_text: str, fallback_page: int) -> str:
    for line in page_text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m:
            return m.group(2).strip()
    return f"page {fallback_page}"


def _resolve_image_path(raw_ref: str, page_path: str, output_dir: Path) -> Path | None:
    ref = str(raw_ref or "").strip()
    if not ref:
        return None
    if ref.startswith(("http://", "https://")):
        return None

    ref_path = Path(ref)
    candidates: list[Path] = []
    if ref_path.is_absolute():
        candidates.append(ref_path)
    else:
        page_parent = Path(page_path).resolve().parent
        candidates.append((page_parent / ref_path).resolve())
        candidates.append((output_dir / ref_path).resolve())

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _try_read_image_shape(path: Path) -> tuple[int | None, int | None]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None, None


def _try_image_stddev(path: Path) -> float | None:
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as image:
            img = image.convert("RGB")
            if max(img.size) > 192:
                img.thumbnail((192, 192))
            stat = ImageStat.Stat(img)
            stddev = sum(stat.stddev) / max(len(stat.stddev), 1)
            return float(stddev)
    except Exception:
        return None


def _collect_image_candidates(pages: list[dict], output_dir: str | Path) -> list[dict]:
    root = Path(output_dir).resolve()
    seen: set[str] = set()
    candidates: list[dict] = []

    for page_index, page in enumerate(pages, start=1):
        page_path = str(page.get("path") or "")
        page_text = str(page.get("text") or "")
        page_heading = _extract_page_heading(page_text, fallback_page=page_index)
        refs = _extract_image_refs(page_text)
        for raw_ref in refs:
            resolved = _resolve_image_path(raw_ref, page_path=page_path, output_dir=root)
            if resolved is None:
                continue
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)

            size_bytes = int(resolved.stat().st_size)
            width, height = _try_read_image_shape(resolved)
            aspect_ratio = None
            if width and height and width > 0 and height > 0:
                aspect_ratio = max(width / height, height / width)
            color_stddev = _try_image_stddev(resolved)

            try:
                rel_path = str(resolved.relative_to(root))
            except Exception:
                rel_path = str(resolved)

            candidates.append(
                {
                    "page": page_index,
                    "page_heading": page_heading,
                    "raw_ref": raw_ref,
                    "image_path": str(resolved),
                    "image_rel_path": rel_path,
                    "size_bytes": size_bytes,
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "color_stddev": color_stddev,
                }
            )

    return candidates


def _filter_image_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    min_bytes = _env_int("IMAGE_FILTER_MIN_BYTES", default=5_000, min_value=1_024, max_value=200_000)
    min_side = _env_int("IMAGE_FILTER_MIN_SIDE_PX", default=100, min_value=32, max_value=800)
    max_aspect_x10 = _env_int("IMAGE_FILTER_MAX_ASPECT_X10", default=120, min_value=20, max_value=400)
    min_stddev = _env_int("IMAGE_FILTER_MIN_COLOR_STDDEV_X10", default=80, min_value=0, max_value=500) / 10.0

    kept: list[dict] = []
    dropped: list[dict] = []
    for item in candidates:
        reasons: list[str] = []
        size_bytes = int(item.get("size_bytes") or 0)
        if size_bytes < min_bytes:
            reasons.append(f"small_file<{min_bytes}")

        width = item.get("width")
        height = item.get("height")
        if isinstance(width, int) and isinstance(height, int):
            if width < min_side or height < min_side:
                reasons.append(f"small_dimension<{min_side}px")

        aspect_ratio = item.get("aspect_ratio")
        if isinstance(aspect_ratio, (int, float)) and aspect_ratio > (max_aspect_x10 / 10.0):
            reasons.append(f"extreme_aspect>{max_aspect_x10 / 10.0}")

        color_stddev = item.get("color_stddev")
        if isinstance(color_stddev, (int, float)) and color_stddev < min_stddev:
            reasons.append(f"low_color_variance<{min_stddev}")

        next_item = dict(item)
        if reasons:
            next_item["drop_reasons"] = reasons
            dropped.append(next_item)
        else:
            kept.append(next_item)

    return kept, dropped


def _image_to_data_url(path: Path, max_side_px: int = 1024) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = path.read_bytes()

    try:
        from io import BytesIO
        from PIL import Image

        with Image.open(path) as image:
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
        # PIL 不可用时保持原图 bytes，避免阻塞图片通道。
        pass

    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _parse_json_object(raw: str) -> dict:
    text = str(raw or "").strip()
    if not text:
        return {}
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _parse_json_list(raw: str) -> list[dict]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
        return []
    except Exception:
        pass
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
        if isinstance(arr, list):
            return [x for x in arr if isinstance(x, dict)]
    except Exception:
        return []
    return []


def _stamp_image_table_records(records: list[dict], image_item: dict) -> list[dict]:
    page = int(image_item.get("page") or 0)
    page_heading = str(image_item.get("page_heading") or f"page {page}")
    source_image = str(image_item.get("image_rel_path") or image_item.get("image_path") or "")
    out: list[dict] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item["_source_heading"] = page_heading
        item["_source_page"] = page
        item["_source_image"] = source_image
        out.append(item)
    return out


def _gate_image_relevance(image_item: dict, client, model: str) -> dict:
    image_path = Path(str(image_item["image_path"]))
    data_url = _image_to_data_url(image_path)
    timeout_sec = _image_vlm_timeout_sec()

    prompt = """你是文档索引系统的图片筛选器。
请判断图片是否包含“可检索的信息内容”（如图表、流程图、示意图、题目图、表格截图、公式截图等）。
如果只是 logo、装饰图标、分割线、纯背景、无信息插图，判为不相关。

只返回 JSON 对象，不要输出其它内容：
{
  "relevant": true/false,
  "category": "diagram|chart|table|formula|photo|decorative|other",
  "reason": "一句话",
  "confidence": 0.0
}
"""

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=180,
            timeout=timeout_sec,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                    ],
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        payload = _parse_json_object(raw)
        relevant = bool(payload.get("relevant"))
        category = str(payload.get("category") or ("other" if relevant else "decorative")).strip().lower()
        reason = str(payload.get("reason") or "").strip()
        confidence = payload.get("confidence")
        try:
            confidence_f = float(confidence)
        except Exception:
            confidence_f = 0.0
        confidence_f = max(0.0, min(1.0, confidence_f))
        return {
            "relevant": relevant,
            "category": category or ("other" if relevant else "decorative"),
            "reason": reason,
            "confidence": confidence_f,
        }
    except Exception as e:
        # Gate 失败时默认保留，避免因为模型能力问题把信息图漏掉。
        return {
            "relevant": True,
            "category": "unknown",
            "reason": f"gate_failed:{type(e).__name__}",
            "confidence": 0.0,
        }


def _describe_informative_image(image_item: dict, client, model: str, category: str) -> tuple[str, list[dict]]:
    image_path = Path(str(image_item["image_path"]))
    data_url = _image_to_data_url(image_path)
    page = int(image_item.get("page") or 0)
    page_heading = str(image_item.get("page_heading") or f"page {page}")
    timeout_sec = _image_vlm_timeout_sec()

    if category == "table":
        prompt = f"""你将看到一张文档里的表格截图（页 {page}, 标题: {page_heading}）。
任务：
1) 提取表格中的每一行，转成 JSON 数组（每行一个对象，字段名英文）
2) 同时给一段简洁中文摘要 summary（不超过120字）

只返回 JSON 对象：
{{
  "summary": "...",
  "records": [{{"col_a": "..."}}, ...]
}}
"""
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=900,
                timeout=timeout_sec,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    }
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            payload = _parse_json_object(raw)
            summary = str(payload.get("summary") or "").strip()
            records = payload.get("records")
            if not isinstance(records, list):
                records = []
            table_records: list[dict] = []
            for item in records:
                if isinstance(item, dict):
                    table_records.append(dict(item))

            if summary:
                content = f"[image-table] {summary}"
            else:
                content = f"[image-table] page={page} heading={page_heading}"
            return content, table_records
        except Exception:
            return f"[image-table] page={page} heading={page_heading}", []

    if category == "formula":
        prompt = f"""你将看到一张文档里的公式截图（页 {page}, 标题: {page_heading}）。
请输出 JSON 对象：
{{
  "latex": "尽量还原的 LaTeX（可为空）",
  "explanation": "公式/符号的简要说明（中文）",
  "keywords": ["关键词1", "关键词2"]
}}
只返回 JSON。
"""
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=500,
                timeout=timeout_sec,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                        ],
                    }
                ],
            )
            raw = (resp.choices[0].message.content or "").strip()
            payload = _parse_json_object(raw)
            latex = str(payload.get("latex") or "").strip()
            explanation = str(payload.get("explanation") or "").strip()
            keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
            keyword_text = "、".join(str(k).strip() for k in keywords if str(k).strip())
            lines = [f"[image-formula] page={page} heading={page_heading}"]
            if latex:
                lines.append(f"LaTeX: {latex}")
            if explanation:
                lines.append(f"说明: {explanation}")
            if keyword_text:
                lines.append(f"关键词: {keyword_text}")
            return "\n".join(lines), []
        except Exception:
            return f"[image-formula] page={page} heading={page_heading}", []

    prompt = f"""你将看到一张文档图片（页 {page}, 标题: {page_heading}）。
请提取可用于检索的结构化描述，返回 JSON 对象：
{{
  "summary": "1-2句中文摘要",
  "details": "关键内容细节（可稍长）",
  "detected_text": ["图中识别出的关键文字（数组）"],
  "keywords": ["检索关键词数组"]
}}
只返回 JSON 对象。
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=700,
            timeout=timeout_sec,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                }
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        payload = _parse_json_object(raw)
        summary = str(payload.get("summary") or "").strip()
        details = str(payload.get("details") or "").strip()
        detected_text = payload.get("detected_text") if isinstance(payload.get("detected_text"), list) else []
        keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []

        lines = [f"[image] page={page} heading={page_heading}"]
        if summary:
            lines.append(f"摘要: {summary}")
        if details:
            lines.append(f"细节: {details}")
        dt = "；".join(str(x).strip() for x in detected_text if str(x).strip())
        if dt:
            lines.append(f"图中文字: {dt}")
        kw = "、".join(str(x).strip() for x in keywords if str(x).strip())
        if kw:
            lines.append(f"关键词: {kw}")
        return "\n".join(lines), []
    except Exception:
        return f"[image] page={page} heading={page_heading}", []


def _is_cache_entry_valid(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    gate = entry.get("gate")
    if not isinstance(gate, dict):
        return False
    if "relevant" not in gate:
        return False
    table_records = entry.get("table_records", [])
    if not isinstance(table_records, list):
        return False
    return True


def _resolve_image_vlm_result(
    image_item: dict,
    client,
    gate_model: str,
    describe_model: str,
) -> dict:
    gate = _gate_image_relevance(image_item, client=client, model=gate_model)
    if not gate.get("relevant"):
        return {
            "gate": gate,
            "content": "",
            "table_records": [],
            "resolved_by": "vlm_gate",
            "gate_called": True,
            "describe_called": False,
            "updated_at": int(time.time()),
        }

    category = str(gate.get("category") or "other").strip().lower()
    content, records = _describe_informative_image(
        image_item,
        client=client,
        model=describe_model,
        category=category,
    )
    return {
        "gate": gate,
        "content": str(content or ""),
        "table_records": [dict(r) for r in records if isinstance(r, dict)],
        "resolved_by": "vlm_describe",
        "gate_called": True,
        "describe_called": True,
        "updated_at": int(time.time()),
    }


def process_image_chunks(
    pages: list[dict],
    output_dir: str | Path,
    client,
    model: str,
) -> dict:
    gate_model = str(os.environ.get("IMAGE_GATE_MODEL") or model).strip() or model
    describe_model = str(os.environ.get("IMAGE_DESCRIBE_MODEL") or model).strip() or model
    max_workers = _env_int("IMAGE_VLM_MAX_WORKERS", default=4, min_value=1, max_value=32)
    cache_enabled = _env_bool("ENABLE_IMAGE_VLM_CACHE", True)
    cache_path = _resolve_image_cache_path(output_dir)

    if os.environ.get("ENABLE_IMAGE_CHUNKS", "1") == "0":
        return {
            "image_chunks": [],
            "image_table_records": [],
            "image_stats": {
                "candidates": 0,
                "rule_kept": 0,
                "rule_dropped": 0,
                "gate_dropped": 0,
                "described": 0,
                "cache_enabled": cache_enabled,
                "cache_path": str(cache_path) if cache_enabled else "",
                "cache_hits": 0,
                "cache_misses": 0,
                "gate_calls": 0,
                "describe_calls": 0,
                "estimated_vlm_calls": 0,
                "gate_model": gate_model,
                "describe_model": describe_model,
                "max_workers": max_workers,
            },
        }

    candidates = _collect_image_candidates(pages, output_dir=output_dir)
    kept, dropped = _filter_image_candidates(candidates)
    max_vlm_calls = _env_int("IMAGE_MAX_VLM_CALLS", default=24, min_value=1, max_value=200)
    kept = kept[:max_vlm_calls]

    cache: dict[str, dict] = {}
    if cache_enabled:
        cache = _load_image_cache(cache_path)
    cache_changed = False
    cache_hits = 0
    cache_misses = 0
    cache_hit_keys: set[str] = set()
    resolved_by_key: dict[str, dict] = {}
    prepared_items: list[dict] = []
    to_process: list[dict] = []
    gate_calls = 0
    describe_calls = 0

    for base_item in kept:
        item = dict(base_item)
        image_path = Path(str(item.get("image_path") or ""))
        try:
            image_sha256 = _image_sha256(image_path)
        except Exception:
            fallback = f"{item.get('image_path', '')}::{item.get('size_bytes', 0)}"
            image_sha256 = hashlib.sha256(fallback.encode("utf-8")).hexdigest()
        item["image_sha256"] = image_sha256
        cache_key = _image_cache_key(image_sha256=image_sha256, gate_model=gate_model, describe_model=describe_model)
        item["cache_key"] = cache_key
        prepared_items.append(item)

        if cache_enabled:
            entry = cache.get(cache_key)
            if _is_cache_entry_valid(entry):
                resolved_by_key[cache_key] = dict(entry)
                cache_hits += 1
                cache_hit_keys.add(cache_key)
                continue
        cache_misses += 1
        to_process.append(item)

    if to_process:
        worker_count = max(1, min(max_workers, len(to_process)))

        def _fallback_result(failed_item: dict, err: Exception) -> dict:
            page = int(failed_item.get("page") or 0)
            heading = str(failed_item.get("page_heading") or f"page {page}")
            return {
                "gate": {
                    "relevant": True,
                    "category": "unknown",
                    "reason": f"worker_failed:{type(err).__name__}",
                    "confidence": 0.0,
                },
                "content": f"[image] page={page} heading={heading}",
                "table_records": [],
                "resolved_by": "worker_failed",
                "gate_called": False,
                "describe_called": False,
                "updated_at": int(time.time()),
            }

        if worker_count == 1:
            for item in to_process:
                try:
                    resolved = _resolve_image_vlm_result(
                        image_item=item,
                        client=client,
                        gate_model=gate_model,
                        describe_model=describe_model,
                    )
                except Exception as e:
                    resolved = _fallback_result(item, e)
                resolved_by_key[item["cache_key"]] = resolved
                if bool(resolved.get("gate_called")):
                    gate_calls += 1
                if bool(resolved.get("describe_called")):
                    describe_calls += 1
                if cache_enabled:
                    cache[item["cache_key"]] = resolved
                    cache_changed = True
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _resolve_image_vlm_result,
                        image_item=item,
                        client=client,
                        gate_model=gate_model,
                        describe_model=describe_model,
                    ): item
                    for item in to_process
                }
                for future in as_completed(future_map):
                    item = future_map[future]
                    try:
                        resolved = future.result()
                    except Exception as e:
                        resolved = _fallback_result(item, e)
                    resolved_by_key[item["cache_key"]] = resolved
                    if bool(resolved.get("gate_called")):
                        gate_calls += 1
                    if bool(resolved.get("describe_called")):
                        describe_calls += 1
                    if cache_enabled:
                        cache[item["cache_key"]] = resolved
                        cache_changed = True

    if cache_enabled and cache_changed:
        max_entries = _env_int("IMAGE_VLM_CACHE_MAX_ENTRIES", default=5000, min_value=100, max_value=100000)
        if len(cache) > max_entries:
            ordered = sorted(
                cache.items(),
                key=lambda item: int((item[1] or {}).get("updated_at") or 0),
                reverse=True,
            )
            cache = dict(ordered[:max_entries])
        try:
            _save_image_cache(cache_path, cache)
        except Exception:
            pass

    image_chunks: list[dict] = []
    image_table_records: list[dict] = []
    gate_dropped = 0

    for item in prepared_items:
        cache_key = str(item.get("cache_key") or "")
        resolved = resolved_by_key.get(cache_key, {})
        if not _is_cache_entry_valid(resolved):
            continue

        gate = dict(resolved.get("gate") or {})
        if not gate.get("relevant"):
            gate_dropped += 1
            continue

        category = str(gate.get("category") or "other").strip().lower()
        page = int(item.get("page") or 0)
        page_heading = str(item.get("page_heading") or f"page {page}")
        content = str(resolved.get("content") or f"[image] page={page} heading={page_heading}")
        extracted_table_rows = _stamp_image_table_records(
            records=[dict(r) for r in (resolved.get("table_records") or []) if isinstance(r, dict)],
            image_item=item,
        )

        chunk = {
            "heading_path": f"{item.get('page_heading', '')} > image",
            "content": content,
            "has_table": False,
            "type": "image",
            "image_path": str(item.get("image_path") or ""),
            "image_rel_path": str(item.get("image_rel_path") or ""),
            "page": int(item.get("page") or 0),
            "image_category": category,
            "image_confidence": float(gate.get("confidence") or 0.0),
            "image_reason": str(gate.get("reason") or ""),
            "image_sha256": str(item.get("image_sha256") or ""),
            "image_cached": cache_key in cache_hit_keys,
        }
        image_chunks.append(chunk)
        image_table_records.extend(extracted_table_rows)

    return {
        "image_chunks": image_chunks,
        "image_table_records": image_table_records,
        "image_stats": {
            "candidates": len(candidates),
            "rule_kept": len(kept),
            "rule_dropped": len(dropped),
            "gate_dropped": gate_dropped,
            "described": len(image_chunks),
            "cache_enabled": cache_enabled,
            "cache_path": str(cache_path) if cache_enabled else "",
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "gate_calls": gate_calls,
            "describe_calls": describe_calls,
            "estimated_vlm_calls": gate_calls + describe_calls,
            "gate_model": gate_model,
            "describe_model": describe_model,
            "max_workers": max_workers,
        },
    }


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


def process_markdown_text(
    raw_text: str,
    client,
    model: str,
    pages: list[dict] | None = None,
    output_dir: str | Path | None = None,
) -> dict:
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

    image_chunks: list[dict] = []
    image_stats: dict = {}
    image_table_records: list[dict] = []
    if pages is not None and output_dir is not None:
        image_processed = process_image_chunks(
            pages=pages,
            output_dir=output_dir,
            client=client,
            model=model,
        )
        image_chunks = image_processed["image_chunks"]
        image_table_records = image_processed["image_table_records"]
        image_stats = image_processed["image_stats"]

    return {
        "headings": headings,
        "corrections": corrections,
        "fixed_text": fixed_text,
        "chunks": chunks,
        "text_chunks": text_chunks,
        "table_chunks": table_chunks,
        "table_records": all_table_records + image_table_records,
        "image_chunks": image_chunks,
        "image_table_records": image_table_records,
        "image_stats": image_stats,
    }
