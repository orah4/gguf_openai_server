import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from config import RAG_DB_PATH

def _conn():
    os.makedirs(os.path.dirname(RAG_DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(RAG_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_rag_db():
    conn = _conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def ingest_docs(items: List[Dict[str, str]]) -> int:
    conn = _conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    n = 0
    for it in items:
        text = (it.get("text") or "").strip()
        if not text:
            continue
        source = (it.get("source") or "manual").strip()
        cur.execute("INSERT INTO docs (source, text, created_at) VALUES (?,?,?)", (source, text, now))
        n += 1
    conn.commit()
    conn.close()
    return n

def _fetch_all_texts() -> Tuple[List[int], List[str], List[str]]:
    conn = _conn()
    cur = conn.cursor()
    cur.execute("SELECT id, source, text FROM docs ORDER BY id ASC")
    rows = cur.fetchall()
    conn.close()
    ids = [r["id"] for r in rows]
    sources = [r["source"] for r in rows]
    texts = [r["text"] for r in rows]
    return ids, sources, texts

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query = (query or "").strip()
    if not query:
        return []

    ids, sources, texts = _fetch_all_texts()
    if not texts:
        return []

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    X = vectorizer.fit_transform(texts)
    qv = vectorizer.transform([query])

    # cosine similarity for TF-IDF
    scores = (X @ qv.T).toarray().ravel()
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        results.append({
            "doc_id": ids[idx],
            "source": sources[idx],
            "score": float(scores[idx]),
            "text": texts[idx],
        })
    return results
