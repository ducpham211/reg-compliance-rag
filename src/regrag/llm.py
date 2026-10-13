"""Lớp LLM mỏng: Gemini (free tier) hoặc Ollama local. Có cache đĩa để tiết kiệm quota."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "data/cache/llm"))


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini")


# Free tier tính quota riêng theo từng model (có model chỉ 20 request/ngày) -> xoay vòng nhiều model.
GEMINI_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_MODELS", "gemini-3.5-flash,gemini-3.1-flash-lite,gemini-2.5-flash-lite,gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-2.5-flash"
).split(",") if m.strip()]
QUOTA_FILE = Path("data/cache/quota_exhausted.json")


def _exhausted() -> dict:
    try:
        d = json.loads(QUOTA_FILE.read_text())
    except Exception:
        return {}
    return {m: until for m, until in d.items() if until > time.time()}


def _mark_exhausted(model: str, seconds: float = 6 * 3600) -> None:
    d = _exhausted()
    d[model] = time.time() + seconds
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    QUOTA_FILE.write_text(json.dumps(d))


def _model() -> str:
    if _provider() == "ollama":
        return os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    return "gemini"


def generate(prompt: str, system: str = "", schema: dict | None = None, use_cache: bool = True) -> str:
    """Trả text (hoặc chuỗi JSON nếu có schema). Cache theo (provider, model, system, prompt, schema)."""
    key = hashlib.sha256(json.dumps([_provider(), system, prompt, schema], ensure_ascii=False).encode()).hexdigest()
    f = CACHE_DIR / f"{key}.txt"
    if use_cache and f.exists():
        return f.read_text(encoding="utf-8")
    out = _ollama(prompt, system, schema) if _provider() == "ollama" else _gemini(prompt, system, schema)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(out, encoding="utf-8")
    return out


def generate_json(prompt: str, schema: dict, system: str = "", use_cache: bool = True):
    return json.loads(generate(prompt, system, schema, use_cache))


def _is_daily(e) -> bool:
    return "PerDay" in str(e) or "per day" in str(e).lower()


def _gemini(prompt: str, system: str, schema: dict | None) -> str:
    from google import genai
    from google.genai import errors, types

    key = os.environ.get("GEMINI_API_KEY", "")
    if len(key) < 20:
        raise RuntimeError("GEMINI_API_KEY không hợp lệ hoặc bị thiếu: kiểm tra file .env (không dùng giá trị mẫu '...').")
    client = genai.Client(api_key=key)
    cfg = dict(system_instruction=system or None, temperature=0.1)
    if schema:
        cfg.update(response_mime_type="application/json", response_schema=schema)
    last = None
    for model in GEMINI_MODELS:
        if model in _exhausted():
            continue
        for attempt in range(3):
            try:
                r = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
                if r.text:
                    return r.text
                break  # rỗng (bị chặn/ngắt): thử model kế tiếp
            except errors.APIError as e:
                last = e
                if e.code in (400, 401, 403) and "API key" in str(e):
                    raise RuntimeError("GEMINI_API_KEY không hợp lệ hoặc bị thiếu: kiểm tra file .env (không dùng giá trị mẫu '...').") from e
                if e.code == 429 and _is_daily(e):
                    _mark_exhausted(model)  # hết quota ngày của model này: sang model khác
                    break
                if e.code in (429, 500, 503) and attempt < 2:
                    time.sleep(15 * (attempt + 1))  # giới hạn theo phút / lỗi tạm thời
                    continue
                break
    raise RuntimeError(f"Tất cả model Gemini đều hết quota hoặc lỗi (chạy lại sau, kết quả đã cache vẫn được dùng): {last}")


def _ollama(prompt: str, system: str, schema: dict | None) -> str:
    import urllib.request

    body = {"model": _model(), "prompt": prompt, "system": system, "stream": False, "options": {"temperature": 0.1}}
    if schema:
        body["format"] = schema
    req = urllib.request.Request(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/generate",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["response"]
