#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .chunk_pipeline import load_markdown_pages, process_image_chunks
except ImportError:
    from chunk_pipeline import load_markdown_pages, process_image_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "backend" / "data" / "documents_registry.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "output"
IMAGE_REF_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


@dataclass
class Profile:
    name: str
    image_max_vlm_calls: int
    image_vlm_max_workers: int
    image_filter_min_bytes: int
    image_filter_min_side_px: int
    image_filter_max_aspect_x10: int
    image_filter_min_color_stddev_x10: int

    def to_env(self) -> dict[str, str]:
        return {
            "ENABLE_IMAGE_CHUNKS": "1",
            "IMAGE_MAX_VLM_CALLS": str(self.image_max_vlm_calls),
            "IMAGE_VLM_MAX_WORKERS": str(self.image_vlm_max_workers),
            "IMAGE_FILTER_MIN_BYTES": str(self.image_filter_min_bytes),
            "IMAGE_FILTER_MIN_SIDE_PX": str(self.image_filter_min_side_px),
            "IMAGE_FILTER_MAX_ASPECT_X10": str(self.image_filter_max_aspect_x10),
            "IMAGE_FILTER_MIN_COLOR_STDDEV_X10": str(self.image_filter_min_color_stddev_x10),
        }


def _parse_int_grid(raw: str, *, min_value: int, max_value: int) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for piece in str(raw or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        value = int(piece)
        if value < min_value or value > max_value:
            raise ValueError(f"value out of range [{min_value},{max_value}]: {value}")
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    if not out:
        raise ValueError("empty grid")
    return out


def _load_registry_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"registry not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"invalid registry format, expect list: {path}")
    return [row for row in payload if isinstance(row, dict)]


def _collect_doc_dirs(
    registry_rows: list[dict],
    max_docs: int,
    seed: int,
    min_image_refs: int,
    include_doc_ids: set[str] | None,
) -> list[dict]:
    docs: list[dict] = []
    for row in registry_rows:
        parsed_output_dir = str(row.get("parsed_output_dir") or "").strip()
        if not parsed_output_dir:
            continue
        doc_id = str(row.get("doc_id") or "").strip()
        if include_doc_ids and doc_id and doc_id not in include_doc_ids:
            continue
        p = Path(parsed_output_dir).resolve()
        if not p.exists() or not p.is_dir():
            continue
        md_files = list(sorted(p.glob("doc_*.md")))
        if not md_files:
            continue
        image_refs = 0
        for md in md_files:
            try:
                text = md.read_text(encoding="utf-8")
            except Exception:
                continue
            image_refs += len(IMAGE_REF_PATTERN.findall(text))
        if image_refs < min_image_refs:
            continue
        docs.append(
            {
                "doc_id": doc_id or p.name,
                "parsed_output_dir": str(p),
                "image_refs": image_refs,
            }
        )

    rng = random.Random(seed)
    rng.shuffle(docs)
    if max_docs > 0:
        docs = docs[:max_docs]
    return docs


def _safe_ratio(num: int | float, den: int | float) -> float:
    if not den:
        return 0.0
    return float(num) / float(den)


def _run_single_profile(
    profile: Profile,
    docs: list[dict],
    client: OpenAI,
    gate_model: str,
    describe_model: str,
    vlm_timeout_sec: int,
    cache_path: Path,
    cache_mode: str,
    gate_call_usd: float,
    describe_call_usd: float,
) -> dict:
    prev_env = dict(os.environ)
    os.environ.update(profile.to_env())
    if cache_mode == "none":
        os.environ["ENABLE_IMAGE_VLM_CACHE"] = "0"
    else:
        os.environ["ENABLE_IMAGE_VLM_CACHE"] = "1"
        os.environ["IMAGE_VLM_CACHE_PATH"] = str(cache_path)
    os.environ["IMAGE_GATE_MODEL"] = gate_model
    os.environ["IMAGE_DESCRIBE_MODEL"] = describe_model
    os.environ["IMAGE_VLM_TIMEOUT_SEC"] = str(vlm_timeout_sec)

    aggregate = {
        "candidates": 0,
        "rule_kept": 0,
        "rule_dropped": 0,
        "gate_dropped": 0,
        "described": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "gate_calls": 0,
        "describe_calls": 0,
        "estimated_vlm_calls": 0,
    }
    per_doc_ms: list[float] = []
    per_doc: list[dict] = []

    started = time.perf_counter()
    for item in docs:
        doc_id = item["doc_id"]
        output_dir = Path(item["parsed_output_dir"]).resolve()
        pages = load_markdown_pages(output_dir)
        doc_started = time.perf_counter()
        processed = process_image_chunks(
            pages=pages,
            output_dir=output_dir,
            client=client,
            model=describe_model,
        )
        elapsed_ms = (time.perf_counter() - doc_started) * 1000.0
        per_doc_ms.append(elapsed_ms)

        stats = dict(processed.get("image_stats") or {})
        for key in aggregate:
            aggregate[key] += int(stats.get(key) or 0)

        per_doc.append(
            {
                "doc_id": doc_id,
                "parsed_output_dir": str(output_dir),
                "elapsed_ms": round(elapsed_ms, 2),
                "stats": stats,
            }
        )

    elapsed_sec = time.perf_counter() - started
    os.environ.clear()
    os.environ.update(prev_env)

    calls_cost = (
        float(aggregate["gate_calls"]) * gate_call_usd
        + float(aggregate["describe_calls"]) * describe_call_usd
    )
    result = {
        "profile": {
            "name": profile.name,
            **profile.__dict__,
            "gate_model": gate_model,
            "describe_model": describe_model,
        },
        "summary": {
            **aggregate,
            "docs": len(docs),
            "elapsed_sec": round(elapsed_sec, 3),
            "avg_doc_ms": round(statistics.mean(per_doc_ms), 2) if per_doc_ms else 0.0,
            "p95_doc_ms": round(
                statistics.quantiles(per_doc_ms, n=20)[18], 2
            ) if len(per_doc_ms) >= 20 else (round(max(per_doc_ms), 2) if per_doc_ms else 0.0),
            "rule_keep_rate": round(_safe_ratio(aggregate["rule_kept"], aggregate["candidates"]), 4),
            "described_rate": round(_safe_ratio(aggregate["described"], aggregate["candidates"]), 4),
            "cache_hit_rate": round(_safe_ratio(aggregate["cache_hits"], aggregate["cache_hits"] + aggregate["cache_misses"]), 4),
            "estimated_cost_usd": round(calls_cost, 6),
        },
        "per_doc": per_doc,
    }
    return result


def _pick_recommendations(results: list[dict]) -> dict[str, dict]:
    if not results:
        return {}

    sorted_by_described = sorted(
        results,
        key=lambda x: (
            int(x["summary"]["described"]),
            -int(x["summary"]["estimated_vlm_calls"]),
            -float(x["summary"]["avg_doc_ms"]),
        ),
        reverse=True,
    )
    high_recall = sorted_by_described[0]
    target_described = max(1, int(0.9 * int(high_recall["summary"]["described"])))

    candidates = [
        x for x in results
        if int(x["summary"]["described"]) >= target_described
    ]
    candidates.sort(
        key=lambda x: (
            int(x["summary"]["estimated_vlm_calls"]),
            float(x["summary"]["avg_doc_ms"]),
            -int(x["summary"]["described"]),
        )
    )
    conservative = candidates[0] if candidates else sorted(
        results,
        key=lambda x: (
            int(x["summary"]["estimated_vlm_calls"]),
            float(x["summary"]["avg_doc_ms"]),
        ),
    )[0]

    return {
        "high_recall": high_recall,
        "conservative_cost": conservative,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid-scan image pipeline parameters and recommend cost/recall profiles."
    )
    parser.add_argument("--registry", type=str, default=str(DEFAULT_REGISTRY))
    parser.add_argument("--max-docs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-image-refs", type=int, default=1)
    parser.add_argument("--doc-ids", type=str, default="", help="Optional comma-separated doc_id allowlist.")
    parser.add_argument("--gate-model", type=str, default=os.environ.get("IMAGE_GATE_MODEL") or os.environ.get("LLM_MODEL", "gpt-4o"))
    parser.add_argument("--describe-model", type=str, default=os.environ.get("IMAGE_DESCRIBE_MODEL") or os.environ.get("LLM_MODEL", "gpt-4o"))
    parser.add_argument("--vlm-timeout-sec", type=int, default=20)
    parser.add_argument("--cache-path", type=str, default="")
    parser.add_argument(
        "--cache-mode",
        type=str,
        choices=["per_profile", "shared", "none"],
        default="per_profile",
        help="Cache strategy for fairness: per_profile (cold per profile), shared (global warm), none (disabled).",
    )
    parser.add_argument("--gate-call-usd", type=float, default=0.0, help="Optional per-call cost for gate stage.")
    parser.add_argument("--describe-call-usd", type=float, default=0.0, help="Optional per-call cost for describe stage.")
    parser.add_argument("--image-max-vlm-calls-grid", type=str, default="8,16,24")
    parser.add_argument("--image-vlm-max-workers-grid", type=str, default="2,4")
    parser.add_argument("--image-filter-min-bytes-grid", type=str, default="5000,12000")
    parser.add_argument("--image-filter-min-side-grid", type=str, default="100,160")
    parser.add_argument("--image-filter-max-aspect-x10-grid", type=str, default="120")
    parser.add_argument("--image-filter-min-color-stddev-x10-grid", type=str, default="80,120")
    parser.add_argument("--output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    api_key = (os.environ.get("LLM_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY for image parameter scan.")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )

    include_doc_ids: set[str] | None = None
    if str(args.doc_ids).strip():
        include_doc_ids = {
            part.strip() for part in str(args.doc_ids).split(",") if part.strip()
        }

    registry_rows = _load_registry_rows(Path(args.registry).resolve())
    docs = _collect_doc_dirs(
        registry_rows,
        max_docs=args.max_docs,
        seed=args.seed,
        min_image_refs=max(0, int(args.min_image_refs)),
        include_doc_ids=include_doc_ids,
    )
    if not docs:
        raise RuntimeError(
            "No valid parsed_output_dir with doc_*.md found under current filters. "
            "Try lowering --min-image-refs or removing --doc-ids."
        )

    image_max_vlm_calls_grid = _parse_int_grid(args.image_max_vlm_calls_grid, min_value=1, max_value=300)
    image_vlm_max_workers_grid = _parse_int_grid(args.image_vlm_max_workers_grid, min_value=1, max_value=64)
    image_filter_min_bytes_grid = _parse_int_grid(args.image_filter_min_bytes_grid, min_value=1024, max_value=500000)
    image_filter_min_side_grid = _parse_int_grid(args.image_filter_min_side_grid, min_value=32, max_value=1200)
    image_filter_max_aspect_x10_grid = _parse_int_grid(args.image_filter_max_aspect_x10_grid, min_value=20, max_value=500)
    image_filter_min_color_stddev_x10_grid = _parse_int_grid(args.image_filter_min_color_stddev_x10_grid, min_value=0, max_value=500)

    profiles: list[Profile] = []
    idx = 1
    for (
        max_calls,
        workers,
        min_bytes,
        min_side,
        max_aspect_x10,
        min_std_x10,
    ) in itertools.product(
        image_max_vlm_calls_grid,
        image_vlm_max_workers_grid,
        image_filter_min_bytes_grid,
        image_filter_min_side_grid,
        image_filter_max_aspect_x10_grid,
        image_filter_min_color_stddev_x10_grid,
    ):
        profiles.append(
            Profile(
                name=f"profile_{idx:03d}",
                image_max_vlm_calls=max_calls,
                image_vlm_max_workers=workers,
                image_filter_min_bytes=min_bytes,
                image_filter_min_side_px=min_side,
                image_filter_max_aspect_x10=max_aspect_x10,
                image_filter_min_color_stddev_x10=min_std_x10,
            )
        )
        idx += 1

    cache_path = (
        Path(args.cache_path).resolve()
        if str(args.cache_path).strip()
        else (DEFAULT_OUTPUT_DIR / "image_scan_cache.json")
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[scan] docs={len(docs)} profiles={len(profiles)} "
        f"cache={cache_path} cache_mode={args.cache_mode}"
    )
    all_results: list[dict] = []
    for profile in profiles:
        run_cache_path = cache_path
        if args.cache_mode == "per_profile":
            run_cache_path = cache_path.with_name(f"{cache_path.stem}_{profile.name}{cache_path.suffix}")
            if run_cache_path.exists():
                run_cache_path.unlink()

        print(
            "[run]",
            profile.name,
            f"max_calls={profile.image_max_vlm_calls}",
            f"workers={profile.image_vlm_max_workers}",
            f"min_bytes={profile.image_filter_min_bytes}",
            f"min_side={profile.image_filter_min_side_px}",
            f"max_aspect_x10={profile.image_filter_max_aspect_x10}",
            f"min_std_x10={profile.image_filter_min_color_stddev_x10}",
        )
        result = _run_single_profile(
            profile=profile,
            docs=docs,
            client=client,
            gate_model=args.gate_model,
            describe_model=args.describe_model,
            vlm_timeout_sec=max(5, min(300, int(args.vlm_timeout_sec))),
            cache_path=run_cache_path,
            cache_mode=args.cache_mode,
            gate_call_usd=float(args.gate_call_usd),
            describe_call_usd=float(args.describe_call_usd),
        )
        all_results.append(result)
        s = result["summary"]
        print(
            f"  -> described={s['described']} calls={s['estimated_vlm_calls']} "
            f"avg_ms={s['avg_doc_ms']} cache_hit_rate={s['cache_hit_rate']}"
        )

    recommendations = _pick_recommendations(all_results)
    ranked = sorted(
        all_results,
        key=lambda x: (
            -int(x["summary"]["described"]),
            int(x["summary"]["estimated_vlm_calls"]),
            float(x["summary"]["avg_doc_ms"]),
        ),
    )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "docs": docs,
        "profile_count": len(profiles),
        "results": ranked,
        "recommendations": {
            "high_recall": copy.deepcopy(recommendations.get("high_recall")),
            "conservative_cost": copy.deepcopy(recommendations.get("conservative_cost")),
        },
    }

    if str(args.output).strip():
        output_path = Path(args.output).resolve()
    else:
        output_path = DEFAULT_OUTPUT_DIR / f"image_param_scan_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[done] saved to {output_path}")
    if recommendations:
        hr = recommendations.get("high_recall", {}).get("profile", {})
        cc = recommendations.get("conservative_cost", {}).get("profile", {})
        print(f"[recommend] high_recall={hr.get('name')} conservative_cost={cc.get('name')}")


if __name__ == "__main__":
    main()
