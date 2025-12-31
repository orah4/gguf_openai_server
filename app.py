import os
import time
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import PORT, RATE_LIMIT, RAG_TOP_K, RAG_USE_LLM
from auth import require_api_key
from llm_engine import chat_completion
from rag_store import init_rag_db, ingest_docs, search

app = Flask(__name__)

# =============================
# RATE LIMITER
# =============================
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[RATE_LIMIT],
)

# =============================
# HELPERS
# =============================
def approx_tokens(text: str) -> int:
    """Very rough token estimate (better than 0)."""
    if not text:
        return 0
    # ~ 4 chars per token estimate
    return max(1, len(text) // 4)

def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)

# =============================
# HEALTH
# =============================
@app.get("/health")
def health():
    return jsonify({"status": "ok"})

# =============================
# OPENAI COMPAT
# =============================
@app.get("/v1/models")
@require_api_key
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {"id": "gguf-local", "object": "model", "owned_by": "you"}
        ]
    })

@app.post("/v1/chat/completions")
@require_api_key
@limiter.limit(RATE_LIMIT)
def v1_chat_completions():
    data = request.get_json(force=True, silent=True) or {}

    model = data.get("model") or "gguf-local"
    messages = data.get("messages") or []
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "messages must be a non-empty list"}), 400

    max_tokens = int(data.get("max_tokens") or 512)
    temperature = float(data.get("temperature") or 0.7)

    # Optional compatibility fields
    stream = bool(data.get("stream", False))
    if stream:
        return jsonify({"error": "stream=true not implemented yet"}), 501

    created = int(time.time())

    try:
        # Generate
        text = chat_completion(messages, max_tokens=max_tokens, temperature=temperature)
        text = safe_str(text).strip()

        # Debug log (visible in your terminal)
        print("[CHAT] model=", model, "max_tokens=", max_tokens, "temp=", temperature)
        print("[CHAT] output_len=", len(text))
        if len(text) < 5:
            print("[CHAT] WARNING: output is very short:", repr(text))

        # If the model returns empty, don't hide it — return a helpful error
        if not text:
            return jsonify({
                "error": "Model returned empty output. Check llm_engine prompt formatting or model loading.",
                "hint": "Try reducing stop tokens or change prompt template in llm_engine.py",
            }), 500

        # Compute usage estimates (not exact tokens)
        prompt_text = "\n".join([safe_str(m.get("content", "")) for m in messages])
        prompt_tokens = approx_tokens(prompt_text)
        completion_tokens = approx_tokens(text)

        return jsonify({
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })

    except Exception as e:
        print("[CHAT] ERROR:", str(e))
        return jsonify({"error": str(e)}), 500


@app.post("/v1/embeddings")
@require_api_key
@limiter.limit(RATE_LIMIT)
def v1_embeddings():
    """
    Minimal local embedding endpoint.
    NOTE: This is NOT semantic embedding (yet).
    It is deterministic hashing so clients won't break.
    """
    data = request.get_json(force=True, silent=True) or {}
    inp = data.get("input")
    if inp is None:
        return jsonify({"error": "input required"}), 400

    if isinstance(inp, str):
        inputs = [inp]
    elif isinstance(inp, list):
        inputs = [safe_str(x) for x in inp]
    else:
        return jsonify({"error": "input must be string or list"}), 400

    def hash_embed(s: str, dim: int = 128):
        v = [0.0] * dim
        b = s.encode("utf-8", errors="ignore")
        for i, ch in enumerate(b):
            v[i % dim] += (ch % 31) / 31.0
        return v

    out = []
    for i, s in enumerate(inputs):
        out.append({"object": "embedding", "index": i, "embedding": hash_embed(s)})

    return jsonify({"object": "list", "data": out, "model": "local-hash-embed"})


# =============================
# RAG
# =============================
@app.post("/v1/rag/ingest")
@require_api_key
@limiter.limit(RATE_LIMIT)
def rag_ingest():
    data = request.get_json(force=True, silent=True) or {}
    items = data.get("items") or []
    if not isinstance(items, list) or not items:
        return jsonify({"error": "items must be a non-empty list of {text,source}"}), 400

    n = ingest_docs(items)
    return jsonify({"status": "ok", "ingested": n})


@app.post("/v1/rag/query")
@require_api_key
@limiter.limit(RATE_LIMIT)
def rag_query():
    data = request.get_json(force=True, silent=True) or {}
    query = safe_str(data.get("query")).strip()
    top_k = int(data.get("top_k") or RAG_TOP_K)

    if not query:
        return jsonify({"error": "query required"}), 400

    hits = search(query, top_k=top_k)

    if not RAG_USE_LLM:
        return jsonify({"query": query, "hits": hits})

    context = "\n\n".join(
        [f"[{h['source']} | score={h['score']:.3f}]\n{h['text']}" for h in hits]
    )

    messages = [
        {
            "role": "system",
            "content": "You are a helpful tutor. Use the provided context. If context is insufficient, say so."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer clearly:"
        }
    ]

    try:
        answer = chat_completion(messages, max_tokens=512, temperature=0.4)
        answer = safe_str(answer).strip()

        if not answer:
            return jsonify({
                "error": "Model returned empty output for RAG. Check llm_engine stop tokens or prompt template.",
                "hits": hits
            }), 500

        return jsonify({"query": query, "answer": answer, "hits": hits})

    except Exception as e:
        print("[RAG] ERROR:", str(e))
        return jsonify({"error": str(e), "hits": hits}), 500


# =============================
# MCP PLACEHOLDER
# =============================
@app.post("/v1/mcp/run")
@require_api_key
def mcp_run():
    return jsonify({"error": "MCP not implemented yet"}), 501


# =============================
# ENTRY
# =============================
if __name__ == "__main__":
    init_rag_db()
   
   
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )