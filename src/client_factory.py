"""Factory for selecting the LLM backend used by MemOS-lite."""

from __future__ import annotations

from typing import Protocol

from src.config import Config
from src.llamacpp_client import LlamaCppClient
from src.ollama_client import OllamaClient


class LLMClient(Protocol):
    """Minimal interface shared by all local LLM backends."""

    def generate(self, prompt: str, system: str | None = None, **kwargs) -> str: ...

    def embed(self, text: str) -> list[float]: ...

    def health_check(self) -> bool: ...


_CLIENT_CACHE: dict[tuple, LLMClient] = {}


def get_llm_client(config: Config) -> LLMClient:
    """Return a cached client for config.llm_backend."""

    backend = config.llm_backend.strip().lower().replace("_", "-")
    key = (
        backend,
        config.ollama_host,
        config.ollama_model,
        config.ollama_embedding_model,
        config.llamacpp_host,
        config.llamacpp_model_path,
    )
    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    if backend == "ollama":
        client: LLMClient = OllamaClient(config)
    elif backend in {"llamacpp", "llama-cpp", "llama.cpp"}:
        client = LlamaCppClient(config)
    else:
        raise ValueError("Unsupported llm_backend: " f"{config.llm_backend!r}. Use 'ollama' or 'llamacpp'.")

    _CLIENT_CACHE[key] = client
    return client


def reset_client_cache() -> None:
    """Small test helper."""

    _CLIENT_CACHE.clear()
