#!/usr/bin/env python3
"""Rebuild data/chroma_db from data/syllabus-390.csv.

One vector per CSV row, embedded from Topic + Content. Topic rides as
metadata so it comes back with the chunk. The CSV is hand-maintained;
this is not a Canvas snapshot rebuild.

Usage:
    python scripts/build_chroma.py --dry-run
    python scripts/build_chroma.py
"""

import argparse
import csv
import io
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "syllabus-390.csv"
PERSIST_DIR = REPO_ROOT / "data" / "chroma_db"
# Must match utils.utils.load_db — a different model makes every query miss.
EMBEDDING_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "langchain"


def _clean(value):
    return (value or "").strip()


def _read_csv_text(source: Path) -> str:
    raw = source.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Older Mac Excel/Numbers exports: 0xA5 is a bullet, 0xD5 an apostrophe.
        # cp1252 would keep those as yen and Õ.
        return raw.decode("mac_roman")


def load_rows(source: Path):
    rows = []
    reader = csv.DictReader(io.StringIO(_read_csv_text(source)))
    for i, row in enumerate(reader, start=1):
        topic = _clean(row.get("Topic"))
        content = _clean(row.get("Content"))
        if not topic or not content:
            print(f"skipping row {i}: empty Topic or Content", file=sys.stderr)
            continue
        rows.append({"topic": topic, "content": content})
    return rows


def to_documents(rows):
    docs = []
    for row in rows:
        docs.append(
            Document(
                page_content=f"{row['topic']}\n\n{row['content']}",
                metadata={"topic": row["topic"], "source": CSV_PATH.name},
            )
        )
    return docs


def report(rows, docs):
    longest = max((len(d.page_content) for d in docs), default=0)
    print(f"source: {CSV_PATH}")
    print(f"persist: {PERSIST_DIR}")
    print(f"embedding_model: {EMBEDDING_MODEL}")
    print(f"n_rows: {len(rows)}")
    print(f"n_chunks: {len(docs)}")
    print(f"longest_chunk: {longest}")
    print("topics:")
    for row in rows:
        print(f"  - {row['topic']}")


def build(docs):
    if PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
    PERSIST_DIR.mkdir(parents=True)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=1)
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
    )
    print(f"wrote {len(docs)} chunks to {PERSIST_DIR}")
    return db


def main():
    ap = argparse.ArgumentParser(description="Rebuild data/chroma_db from syllabus-390.csv")
    ap.add_argument("--dry-run", action="store_true", help="report counts, embed nothing")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")

    if not CSV_PATH.exists():
        sys.exit(f"missing source CSV: {CSV_PATH}")

    rows = load_rows(CSV_PATH)
    docs = to_documents(rows)
    report(rows, docs)

    if args.dry_run:
        print("dry_run: no embeddings written")
        return

    build(docs)


if __name__ == "__main__":
    main()
