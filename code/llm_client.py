# -*- coding: utf-8 -*-
"""
Deterministic, cached LLM client for CAR-TBench.

Providers are OpenAI-compatible endpoints. Credentials come from env vars so
nothing secret is written to disk:

    CARBENCH_DEEPSEEK_KEY   (falls back to ANTHROPIC_AUTH_TOKEN)
    CARBENCH_LOCAL_BASE     (default http://127.0.0.1:8080/v1)

Every call is cached on disk keyed by sha256(model + params + messages), so
re-running an experiment is free and bit-identical. Cache lives in
data/cache/llm/.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
CACHE = os.path.join(ROOT, "data", "cache", "llm")
os.makedirs(CACHE, exist_ok=True)

PROVIDERS = {
    "deepseek": {
        "base": "https://api.deepseek.com/v1",
        "key_env": ["CARBENCH_DEEPSEEK_KEY", "DEEPSEEK_API_KEY",
                    "ANTHROPIC_AUTH_TOKEN"],
    },
    "local": {
        "base": os.environ.get("CARBENCH_LOCAL_BASE", "http://127.0.0.1:8080/v1"),
        "key_env": ["CARBENCH_LOCAL_KEY"],
    },
    # Volcano Engine Ark (Doubao): OpenAI-compatible. The model may be a
    # model name (doubao-seed-1-6-...) or an endpoint id (ep-...).
    "doubao": {
        "base": os.environ.get("CARBENCH_DOUBAO_BASE",
                               "https://ark.cn-beijing.volces.com/api/v3"),
        "key_env": ["CARBENCH_DOUBAO_KEY", "ARK_API_KEY", "DOUBAO_API_KEY"],
    },
}

MODELS = {
    # id -> (provider, model name)
    "deepseek-flash": ("deepseek", "deepseek-flash"),
    "deepseek-v4-pro": ("deepseek", "deepseek-v4-pro"),
    "qwen3-27b": ("local", "qwen3.8-27b"),
    "doubao": ("doubao",
               os.environ.get("CARBENCH_DOUBAO_MODEL", "doubao-seed-1-6-250615")),
}

# Credentials can also live in a user-private JSON file so the key never has
# to be pasted into a transcript or set in a long-lived shell:
#   {"doubao": "...", "deepseek": "..."}
KEY_FILE = os.path.join(os.path.expanduser("~"), ".cartbench_keys.json")


@dataclass
class LLMResult:
    text: str
    model: str
    raw: dict = field(default_factory=dict)
    cached: bool = False
    latency_s: float = 0.0


def _key(provider_conf: dict, provider: str) -> str:
    for e in provider_conf["key_env"]:
        v = os.environ.get(e)
        if v:
            return v
    try:
        with open(KEY_FILE, encoding="utf-8") as f:
            return json.load(f).get(provider, "")
    except Exception:  # noqa: BLE001
        return ""


class LLM:
    def __init__(self, model: str = "deepseek-flash", temperature: float = 0.0,
                 max_tokens: int = 4096, timeout: int = 180,
                 max_retries: int = 12, use_cache: bool = True,
                 extra_body: dict | None = None):
        if model not in MODELS:
            raise KeyError(f"unknown model {model}; known: {list(MODELS)}")
        self.model_id = model
        self.provider, self.model = MODELS[model]
        self.conf = PROVIDERS[self.provider]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.extra_body = extra_body or {}

    # ---------------------------------------------------------------- cache
    def _cache_path(self, payload: dict) -> str:
        h = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False)
            .encode("utf-8")).hexdigest()
        return os.path.join(CACHE, h[:2], h + ".json")

    # ----------------------------------------------------------------- chat
    def chat(self, messages: list[dict], json_mode: bool = False,
             **overrides) -> LLMResult:
        # Reasoning models bill chain-of-thought against max_tokens: a budget
        # that "looks" generous for a one-word answer can truncate the JSON.
        # Grow the budget whenever the endpoint reports finish_reason=length.
        max_tokens = overrides.get("max_tokens", self.max_tokens)
        key = _key(self.conf, self.provider)
        url = self.conf["base"].rstrip("/") + "/chat/completions"

        last_err = None
        for attempt in range(self.max_retries):
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": overrides.get("temperature", self.temperature),
                "max_tokens": max_tokens,
            }
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            params.update(self.extra_body)
            path = self._cache_path({"provider": self.provider, "params": params})
            if self.use_cache and os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    rec = json.load(f)
                return LLMResult(rec["text"], self.model_id, rec.get("raw", {}),
                                 cached=True)
            body = json.dumps(params, ensure_ascii=False).encode("utf-8")
            try:
                req = urllib.request.Request(
                    url, data=body,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {key}"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body_bytes = r.read()
                if not body_bytes.strip():
                    last_err = "empty body"
                    time.sleep(min(30, 2 ** attempt) + random.random())
                    continue
                raw = json.loads(body_bytes.decode("utf-8"))
                ch = (raw.get("choices") or [{}])[0]
                msg = ch.get("message", {}) if raw.get("choices") else {}
                text = msg.get("content") or ""
                if ch.get("finish_reason") == "length":
                    last_err = f"truncated at max_tokens={max_tokens}"
                    max_tokens = min(max_tokens * 2, 32000)
                    continue
                if not text.strip():
                    # the endpoint intermittently returns an empty message under
                    # load; treat it as a transient failure and retry
                    last_err = f"empty content: {json.dumps(raw)[:200]}"
                    # the endpoint drops content under concurrency; back off
                    # progressively (a fixed 1-3 s was not enough at 5 workers)
                    time.sleep(min(30, 2 ** attempt) + random.random() * 2)
                    continue
                res = LLMResult(text, self.model_id, raw,
                                latency_s=time.time() - t0)
                if self.use_cache:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump({"text": text, "raw": raw,
                                   "model": self.model_id}, f,
                                  ensure_ascii=False)
                return res
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code in (400, 401, 403, 404):
                    break                      # not transient
                time.sleep(min(45, 2 ** attempt) + random.random() * 2)
            except Exception as e:             # noqa: BLE001
                last_err = repr(e)
                time.sleep(min(60, 2 ** attempt) + random.random())
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def json_chat(self, messages: list[dict], **kw) -> dict:
        res = self.chat(messages, json_mode=True, **kw)
        txt = res.text.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").strip()
            if txt.lower().startswith("json"):
                txt = txt[4:]
        i = txt.find("{")
        if i < 0:
            raise ValueError(f"reply is not JSON: {txt[:200]!r}")
        try:
            return json.loads(txt[i:])
        except json.JSONDecodeError:
            # some models emit raw newlines inside strings; recover those
            # instead of treating the whole call as a failure
            return json.loads(txt[i:], strict=False)


class Pool:
    """Thread pool over one or more LLMs, preserving input order."""

    def __init__(self, models: list[str], workers: int = 4, **llm_kw):
        self.llms = {m: LLM(m, **llm_kw) for m in models}
        self.workers = workers
        self._sem = threading.Semaphore(workers)

    def map(self, fn, items: list) -> list:
        """fn(item) -> result; runs with a bounded number in flight."""
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, items))


if __name__ == "__main__":
    llm = LLM("deepseek-flash")
    r = llm.chat([{"role": "user",
                   "content": "用一句话解释CRS分级中hypoxia的判据"}])
    print("cached:", r.cached, "|", r.text[:200])
