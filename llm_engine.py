import requests
from pathlib import Path
from typing import List, Dict

from config import (
    MODEL_PATH, MODEL_URL,
    MODEL_N_CTX, MODEL_N_THREADS, MODEL_N_GPU_LAYERS, MODEL_N_BATCH
)

_llm = None


# =============================
# MODEL HANDLING
# =============================
def _ensure_model_present() -> str:
    if MODEL_PATH and Path(MODEL_PATH).exists():
        return MODEL_PATH

    if MODEL_URL:
        models_dir = Path("./models")
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "model.gguf"

        if target.exists() and target.stat().st_size > 10_000_000:
            return str(target)

        print("[LLM] Downloading model from URL...")
        resp = requests.get(MODEL_URL, stream=True, timeout=120)
        resp.raise_for_status()

        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        return str(target)

    raise RuntimeError("No model found. Set MODEL_PATH or MODEL_URL.")


def load_llm():
    global _llm
    if _llm is not None:
        return _llm

    model_file = _ensure_model_present()

    from llama_cpp import Llama

    print("[LLM] Loading GGUF model:", model_file)

    _llm = Llama(
        model_path=model_file,
        n_ctx=MODEL_N_CTX,
        n_threads=MODEL_N_THREADS,
        n_gpu_layers=MODEL_N_GPU_LAYERS,
        n_batch=MODEL_N_BATCH,
        verbose=False,
    )

    print("[LLM] Model ready")
    return _llm


# =============================
# PROMPT CONVERSION (FIXED)
# =============================
def messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Robust instruct-style prompt.
    Ensures the model KNOWS it must answer.
    """
    prompt_parts = []

    for m in messages:
        role = m.get("role", "user")
        content = (m.get("content") or "").strip()
        if not content:
            continue

        if role == "system":
            prompt_parts.append(f"<<SYSTEM>>\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"<<ASSISTANT>>\n{content}\n")
        else:
            prompt_parts.append(f"<<USER>>\n{content}\n")

    # 🔴 CRITICAL: force assistant turn
    prompt_parts.append("<<ASSISTANT>>\n")

    return "".join(prompt_parts)


# =============================
# MAIN COMPLETION FUNCTION
# =============================
def chat_completion(
    messages: List[Dict[str, str]],
    max_tokens: int = 512,
    temperature: float = 0.7
) -> str:
    llm = load_llm()
    prompt = messages_to_prompt(messages)

    result = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=["<<USER>>", "<<SYSTEM>>"],
        echo=False,
    )

    # ✅ ALWAYS extract text properly
    text = result["choices"][0]["text"]

    if not isinstance(text, str):
        raise RuntimeError("LLM returned non-string output")

    text = text.strip()

    # 🔍 HARD SAFETY CHECK
    if not text:
        raise RuntimeError(
            "LLM generated tokens but returned empty text. "
            "Check stop tokens or prompt formatting."
        )

    return text
