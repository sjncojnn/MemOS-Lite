"""Optional llama.cpp server client for MemOS-lite.

This client talks to a running llama.cpp server through HTTP. It stays optional:
Ollama remains the default backend for the local macOS MVP.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import Config


class LlamaCppClient:
    """Client for llama.cpp server, usually http://localhost:8080."""

    def __init__(self, config: Config):
        self.config = config
        self.host = config.llamacpp_host.rstrip("/")
        self.model_path = config.llamacpp_model_path
        self.timeout = config.request_timeout_seconds

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Generate text via llama.cpp server /completion."""

        full_prompt = f"{system.strip()}\n\n{prompt.strip()}" if system else prompt.strip()
        payload: dict[str, Any] = {
            "prompt": full_prompt,
            "stream": False,
            "cache_prompt": bool(self.config.llamacpp_prompt_cache),
        }

        options = dict(kwargs.pop("options", {}) or {})
        for key, value in options.items():
            payload[key] = value

        if "num_predict" in kwargs and "n_predict" not in kwargs:
            kwargs["n_predict"] = kwargs.pop("num_predict")

        for key in (
            "n_predict",
            "temperature",
            "top_p",
            "top_k",
            "repeat_penalty",
            "stop",
            "seed",
        ):
            if key in kwargs:
                payload[key] = kwargs.pop(key)

        data = self._post_json("/completion", payload)
        if "content" in data:
            return str(data["content"]).strip()
        if "response" in data:
            return str(data["response"]).strip()
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                return str(first.get("text") or first.get("message", {}).get("content", "")).strip()
        return ""

    def embed(self, text: str) -> list[float]:
        """Embed one text via llama.cpp server embedding endpoint."""

        text = text or ""
        errors: list[str] = []
        for path, payload in (
            ("/embedding", {"content": text}),
            ("/embeddings", {"input": text}),
        ):
            try:
                data = self._post_json(path, payload)
                parsed = self._parse_embedding(data)
                if parsed:
                    return parsed
            except RuntimeError as exc:
                errors.append(str(exc))
        raise RuntimeError("llama.cpp embedding request failed. " + " | ".join(errors))

    def health_check(self) -> bool:
        """Return True when the llama.cpp server responds."""

        for path in ("/health", "/props", "/"):
            try:
                self._get_json(path)
                return True
            except RuntimeError:
                continue
        return False

    @staticmethod
    def _parse_embedding(data: dict[str, Any]) -> list[float]:
        embedding = data.get("embedding")
        if isinstance(embedding, list):
            return [float(x) for x in embedding]

        data_items = data.get("data")
        if isinstance(data_items, list) and data_items:
            first = data_items[0]
            if isinstance(first, dict) and isinstance(first.get("embedding"), list):
                return [float(x) for x in first["embedding"]]

        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            first = embeddings[0]
            if isinstance(first, list):
                return [float(x) for x in first]
        return []

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(self._url(path), method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(self._error_message(exc)) from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self._url(path),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama.cpp request failed: HTTP {exc.code}. {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(self._error_message(exc)) from exc

    def _error_message(self, exc: BaseException) -> str:
        hint = "Start llama.cpp server and set MEMOS_LITE_LLAMACPP_HOST if needed."
        if self.model_path:
            hint += f" Configured model path: {self.model_path}."
        return f"Cannot connect to llama.cpp server at {self.host}. {hint} Detail: {exc}"
