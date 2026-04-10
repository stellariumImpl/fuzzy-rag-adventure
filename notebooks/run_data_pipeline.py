#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

try:
    from .chunk_pipeline import load_markdown_pages, merge_markdown_pages, process_markdown_text
    from .embedding import get_embedder, upsert_chunks
except ImportError:
    # 兼容直接执行: python notebooks/run_data_pipeline.py
    from chunk_pipeline import load_markdown_pages, merge_markdown_pages, process_markdown_text
    from embedding import get_embedder, upsert_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end data pipeline: parsed markdown pages -> chunks/tables -> Qdrant upsert."
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Directory containing parsed doc_*.md files.")
    parser.add_argument("--collection", type=str, default="documents", help="Qdrant collection name.")
    parser.add_argument("--doc-id", type=str, default=None, help="Logical document id; default is output dir name.")
    parser.add_argument(
        "--doc-language",
        type=str,
        default="mixed",
        choices=["en", "zh", "mixed", "auto"],
        help="Embedding language route.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save chunks/table_records/fixed_text artifacts under output-dir.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    output_dir = Path(args.output_dir)
    pages = load_markdown_pages(output_dir)
    if not pages:
        raise FileNotFoundError(
            f"No parsed markdown pages found under {output_dir}. Expect files like doc_0.md, doc_1.md."
        )

    raw_text = merge_markdown_pages(pages)
    if not raw_text.strip():
        raise RuntimeError(f"Merged markdown is empty: {output_dir}")

    if "LLM_API_KEY" not in os.environ:
        raise RuntimeError("Missing LLM_API_KEY in environment for heading/table extraction.")

    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ.get("LLM_BASE_URL") or None,
    )
    model = os.environ.get("LLM_MODEL", "gpt-4o")

    processed = process_markdown_text(
        raw_text,
        client=client,
        model=model,
        pages=pages,
        output_dir=output_dir,
    )
    text_chunks = processed["text_chunks"]
    image_chunks = processed.get("image_chunks", [])
    table_records = processed["table_records"]
    indexed_chunks = text_chunks + image_chunks

    doc_id = args.doc_id or output_dir.name
    embedder = get_embedder(args.doc_language)
    upsert_chunks(
        chunks=indexed_chunks,
        embedder=embedder,
        collection_name=args.collection,
        doc_id=doc_id,
    )

    if args.save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "fixed_text.md").write_text(processed["fixed_text"], encoding="utf-8")
        (output_dir / "text_chunks.json").write_text(
            json.dumps(text_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "image_chunks.json").write_text(
            json.dumps(image_chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "table_records.json").write_text(
            json.dumps(table_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "heading_corrections.json").write_text(
            json.dumps(processed["corrections"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(
        f"[done] pages={len(pages)} text_chunks={len(text_chunks)} image_chunks={len(image_chunks)} "
        f"table_records={len(table_records)} "
        f"collection={args.collection} doc_id={doc_id}"
    )


if __name__ == "__main__":
    main()
