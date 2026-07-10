"""Small Ollama client for MemOS-lite.

Uses only the Python standard library to avoid adding another dependency. The
interface is shared with llama.cpp through client_factory.py.
"""

from __future__ import annotations

import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from src.config import Config


class OllamaClient:
    """Client for a local Ollama server, usually http://localhost:11434."""

    def __init__(self, config: Config):
        self.config = config
        self.host = config.ollama_host.rstrip("/")
        self.model = config.ollama_model
        self.embedding_model = config.ollama_embedding_model
        self.timeout = config.request_timeout_seconds

    def generate(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """Generate text with Ollama /api/generate."""

        options = dict(kwargs.pop("options", {}) or {})
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "num_predict",
            "num_ctx",
            "seed",
            "repeat_penalty",
            "stop",
        ):
            if key in kwargs:
                options[key] = kwargs.pop(key)

        payload: dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        # Common top-level Ollama options, e.g. format="json" or keep_alive="5m".
        for key in ("format", "context", "keep_alive", "template", "raw"):
            if key in kwargs:
                payload[key] = kwargs.pop(key)

        data = self._post_json("/api/generate", payload)
        return str(data.get("response", "")).strip()

    def embed(self, text: str) -> list[float]:
        """Embed one text with the configured Ollama embedding model."""

        text = text or ""
        try:
            data = self._post_json(
                "/api/embeddings",
                {"model": self.embedding_model, "prompt": text},
            )
            embedding = data.get("embedding")
            if isinstance(embedding, list):
                return [float(x) for x in embedding]
        except RuntimeError:
            # Some Ollama versions expose /api/embed instead. Try it before
            # surfacing the connection/API error.
            pass

        data = self._post_json(
            "/api/embed",
            {"model": self.embedding_model, "input": text},
        )
        embeddings = data.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            return [float(x) for x in embeddings[0]]
        raise RuntimeError("Ollama embedding response did not contain an embedding vector")

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Simple batch helper; keeps ingestion code readable."""

        return [self.embed(text) for text in texts]

    def health_check(self) -> bool:
        """Return True when the Ollama server responds."""

        try:
            self._get_json("/api/tags")
            return True
        except RuntimeError:
            return False

    def _url(self, path: str) -> str:
        return f"{self.host}{path}"

    def _get_json(self, path: str) -> dict[str, Any]:
        request = Request(self._url(path), method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
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
            raise RuntimeError(f"Ollama request failed: HTTP {exc.code}. {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(self._error_message(exc)) from exc

    def _error_message(self, exc: BaseException) -> str:
        return (
            f"Cannot connect to Ollama at {self.host}. Start Ollama first, then pull models: "
            f"ollama pull {self.model} && ollama pull {self.embedding_model}. Detail: {exc}"
        )
