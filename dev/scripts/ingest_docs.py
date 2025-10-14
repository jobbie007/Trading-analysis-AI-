import os
import pathlib
import json
from typing import List, Dict

from rank_bm25 import BM25Okapi

DOCS_DIR = pathlib.Path(os.getenv("RAG_DOCS_DIR", "dev/docs"))
INDEX_PATH = pathlib.Path(os.getenv("RAG_INDEX_PATH", "dev/docs/index_bm25.json"))


def _read_text_files(folder: pathlib.Path) -> List[str]:
    texts: List[str] = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in {".txt", ".md"}:
            try:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    return texts


def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def build_bm25_index() -> Dict:
    texts = _read_text_files(DOCS_DIR)
    if not texts:
        raise SystemExit(f"No documents found under {DOCS_DIR}")
    tokenized = [_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    data = {
        "texts": texts,
        "tokenized": tokenized,
    }
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(data), encoding="utf-8")
    print(f"Wrote BM25 corpus to {INDEX_PATH} (documents={len(texts)})")


def main():
    build_bm25_index()


if __name__ == "__main__":
    main()
