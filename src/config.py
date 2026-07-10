"""Central configuration for MemOS-lite.

One dataclass, environment-variable overrides, no config framework. Defaults are
chosen for a local macOS laptop without GPU: SQLite + Chroma + Ollama.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HOME = Path(os.environ.get("MEMOS_LITE_HOME", "./.memos_lite_data")).expanduser().resolve()


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Config:
    """Runtime config for MemOS-lite."""

    # Storage paths
    home_dir: Path = DEFAULT_HOME
    sqlite_filename: str = "memos_lite.sqlite3"
    memos_chroma_dirname: str = "chroma_memos"
    baseline_chroma_dirname: str = "chroma_baseline"
    sqlite_path: Path = field(init=False)
    chroma_path: Path = field(init=False)
    baseline_chroma_path: Path = field(init=False)

    # Chroma collections
    memos_collection: str = "memos_lite_memories"
    baseline_collection: str = "baseline_chunks"
    qa_cache_collection: str = "memos_lite_qa_cache"

    # LLM backend
    llm_backend: str = "ollama"  # "ollama" | "llamacpp"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_embedding_model: str = "nomic-embed-text"
    request_timeout_seconds: int = 120

    # Optional llama.cpp backend. Keep as config only; do not force it into MVP.
    llamacpp_host: str = "http://localhost:8080"
    llamacpp_model_path: str = ""
    llamacpp_prompt_cache: bool = True

    # Ingestion / chunking
    chunk_size_tokens: int = 1200
    chunk_overlap_tokens: int = 150
    default_ttl_seconds: int = 0

    # Retrieval / QA
    top_k: int = 5
    max_context_chars: int = 8000
    min_retrieval_score: float = 0.0
    retrieval_candidate_multiplier: int = 4
    retrieval_lexical_pool_limit: int = 1000
    retrieval_neighbor_expansion: bool = True

    # Rule-based lifecycle / scheduler
    hot_access_count_threshold: int = 5
    hot_window_days: int = 7
    cold_after_days_no_access: int = 30

    # Duplicate / conflict heuristics. Exact duplicate uses content_hash.
    near_duplicate_threshold: float = 0.95

    # QA cache. Exact cache is always safest. Semantic cache is verified by
    # memory ids/signature before returning an old answer.
    qa_cache_enabled: bool = True
    qa_cache_ttl_seconds: int = 10000
    qa_semantic_cache_enabled: bool = True
    qa_semantic_answer_threshold: float = 0.80
    qa_semantic_sensitive_threshold: float = 0.85
    qa_semantic_retrieval_threshold: float = 0.75
    qa_semantic_min_memory_overlap: float = 0.60

    def __post_init__(self) -> None:
        self.home_dir = Path(self.home_dir).expanduser().resolve()
        self.sqlite_path = self.home_dir / self.sqlite_filename
        self.chroma_path = self.home_dir / self.memos_chroma_dirname
        self.baseline_chroma_path = self.home_dir / self.baseline_chroma_dirname

        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.baseline_chroma_path.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in self.__dict__.items()}


def load_config() -> Config:
    """Load config from MEMOS_LITE_* environment variables."""

    return Config(
        home_dir=Path(_env_str("MEMOS_LITE_HOME", str(DEFAULT_HOME))),
        sqlite_filename=_env_str("MEMOS_LITE_SQLITE_FILENAME", "memos_lite.sqlite3"),
        memos_chroma_dirname=_env_str("MEMOS_LITE_CHROMA_DIRNAME", "chroma_memos"),
        baseline_chroma_dirname=_env_str("MEMOS_LITE_BASELINE_CHROMA_DIRNAME", "chroma_baseline"),
        memos_collection=_env_str("MEMOS_LITE_COLLECTION", "memos_lite_memories"),
        baseline_collection=_env_str("MEMOS_LITE_BASELINE_COLLECTION", "baseline_chunks"),
        qa_cache_collection=_env_str("MEMOS_LITE_QA_CACHE_COLLECTION", "memos_lite_qa_cache"),
        llm_backend=_env_str("MEMOS_LITE_LLM_BACKEND", "ollama"),
        ollama_host=_env_str("MEMOS_LITE_OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=_env_str("MEMOS_LITE_OLLAMA_MODEL", "llama3.2:3b"),
        ollama_embedding_model=_env_str("MEMOS_LITE_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
        request_timeout_seconds=_env_int("MEMOS_LITE_REQUEST_TIMEOUT_SECONDS", 120),
        llamacpp_host=_env_str("MEMOS_LITE_LLAMACPP_HOST", "http://localhost:8080"),
        llamacpp_model_path=_env_str("MEMOS_LITE_LLAMACPP_MODEL_PATH", ""),
        llamacpp_prompt_cache=_env_bool("MEMOS_LITE_LLAMACPP_PROMPT_CACHE", True),
        chunk_size_tokens=_env_int("MEMOS_LITE_CHUNK_SIZE_TOKENS", 1200),
        chunk_overlap_tokens=_env_int("MEMOS_LITE_CHUNK_OVERLAP_TOKENS", 150),
        default_ttl_seconds=_env_int("MEMOS_LITE_DEFAULT_TTL_SECONDS", 0),
        top_k=_env_int("MEMOS_LITE_TOP_K", 5),
        max_context_chars=_env_int("MEMOS_LITE_MAX_CONTEXT_CHARS", 8000),
        min_retrieval_score=_env_float("MEMOS_LITE_MIN_RETRIEVAL_SCORE", 0.0),
        retrieval_candidate_multiplier=_env_int("MEMOS_LITE_RETRIEVAL_CANDIDATE_MULTIPLIER", 4),
        retrieval_lexical_pool_limit=_env_int("MEMOS_LITE_RETRIEVAL_LEXICAL_POOL_LIMIT", 1000),
        retrieval_neighbor_expansion=_env_bool("MEMOS_LITE_RETRIEVAL_NEIGHBOR_EXPANSION", True),
        hot_access_count_threshold=_env_int("MEMOS_LITE_HOT_ACCESS_COUNT_THRESHOLD", 5),
        hot_window_days=_env_int("MEMOS_LITE_HOT_WINDOW_DAYS", 7),
        cold_after_days_no_access=_env_int("MEMOS_LITE_COLD_AFTER_DAYS_NO_ACCESS", 30),
        near_duplicate_threshold=_env_float("MEMOS_LITE_NEAR_DUPLICATE_THRESHOLD", 0.95),
        qa_cache_enabled=_env_bool("MEMOS_LITE_QA_CACHE_ENABLED", True),
        qa_cache_ttl_seconds=_env_int("MEMOS_LITE_QA_CACHE_TTL_SECONDS", 100),
        qa_semantic_cache_enabled=_env_bool("MEMOS_LITE_QA_SEMANTIC_CACHE_ENABLED", True),
        qa_semantic_answer_threshold=_env_float("MEMOS_LITE_QA_SEMANTIC_ANSWER_THRESHOLD", 0.80),
        qa_semantic_sensitive_threshold=_env_float("MEMOS_LITE_QA_SEMANTIC_SENSITIVE_THRESHOLD", 0.85),
        qa_semantic_retrieval_threshold=_env_float("MEMOS_LITE_QA_SEMANTIC_RETRIEVAL_THRESHOLD", 0.90),
        qa_semantic_min_memory_overlap=_env_float("MEMOS_LITE_QA_SEMANTIC_MIN_MEMORY_OVERLAP", 0.60),
    )
