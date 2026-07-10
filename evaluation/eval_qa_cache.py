"""Evaluate QACache with CSV columns: case_id,label,query_1,query_2."""
from __future__ import annotations

import argparse
import copy
import csv
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.client_factory import get_llm_client  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.memory_store import MemoryStore  # noqa: E402
from src.qa_cache import QACache  # noqa: E402
from src.qa_service import QAService  # noqa: E402
from src.retriever import MemoryRetriever  # noqa: E402
from src.schemas import CacheEntry, RetrievedMemory  # noqa: E402

HIT_LABELS = {
    "1", "true", "yes", "hit", "positive", "same", "similar", "related",
    "paraphrase", "equivalent", "semantic_hit", "should_hit",
}
MISS_LABELS = {
    "0", "false", "no", "miss", "negative", "different", "unrelated",
    "not_similar", "non_equivalent", "semantic_miss", "should_miss",
}
OUTPUT_COLUMNS = [
    "case_id", "label", "query_1", "query_2", "expected_semantic_hit",
    "actual_semantic_hit", "policy_pass", "matched_cache_query",
    "query_similarity", "similarity_threshold", "keyword_score",
    "keyword_threshold", "decision_reason", "miss_latency_ms",
    "exact_hit_latency_ms", "semantic_hit_latency_ms", "exact_speedup_x",
    "semantic_speedup_x",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact/semantic QA cache")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--home", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/runs/qa_cache"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache-repeats", type=int, default=3)
    return parser.parse_args()


def parse_label(value: str) -> bool:
    value = re.sub(r"[\s-]+", "_", (value or "").strip().lower())
    if value in HIT_LABELS:
        return True
    if value in MISS_LABELS:
        return False
    raise ValueError(
        f"Unsupported label {value!r}. Use hit/miss, positive/negative, "
        "similar/different, or 1/0."
    )


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = {str(x).strip() for x in (reader.fieldnames or [])}
        required = {"case_id", "label", "query_1", "query_2"}
        if not required.issubset(columns):
            raise ValueError(
                f"Dataset must contain {sorted(required)}. Found: {sorted(columns)}"
            )
        rows = []
        for line, raw in enumerate(reader, start=2):
            row = {str(k).strip(): str(v or "").strip() for k, v in raw.items() if k}
            if not row["case_id"] or not row["query_1"] or not row["query_2"]:
                raise ValueError(f"Row {line} has an empty case_id/query")
            row["expected"] = parse_label(row["label"])
            rows.append(row)
    if not rows:
        raise ValueError("Dataset contains no rows")
    return rows[:limit] if limit > 0 else rows


def configure(home: Optional[Path]) -> Config:
    config = load_config()
    if home:
        config.home_dir = home.expanduser().resolve()
        config.sqlite_path = config.home_dir / config.sqlite_filename
        config.chroma_path = config.home_dir / config.memos_chroma_dirname
        config.baseline_chroma_path = config.home_dir / config.baseline_chroma_dirname
    return config


def case_config(base: Config, run_dir: Path, index: int, case_id: str) -> Config:
    runtime = run_dir / "cache_runtime"
    safe_id = re.sub(r"\W+", "_", case_id).strip("_") or "case"
    config = copy.deepcopy(base)
    config.home_dir = runtime
    config.sqlite_path = runtime / f"case_{index:04d}.sqlite3"
    config.chroma_path = runtime / "chroma"
    config.qa_cache_collection = f"qa_cache_{index:04d}_{safe_id}"
    config.qa_cache_enabled = True
    config.qa_semantic_cache_enabled = True
    config.qa_cache_ttl_seconds = 3600
    runtime.mkdir(parents=True, exist_ok=True)
    config.chroma_path.mkdir(parents=True, exist_ok=True)
    return config


def memory_refs(results: Sequence[RetrievedMemory]) -> list[str]:
    return [
        f"{x.memory.id}::v{x.memory.version}::h{x.memory.content_hash[:12]}"
        for x in results
    ]


def rounded(value: Optional[float], digits: int = 3) -> Optional[float]:
    return round(value, digits) if value is not None else None


def median(values: Sequence[Optional[float]]) -> Optional[float]:
    valid = [float(x) for x in values if x is not None]
    return statistics.median(valid) if valid else None


def exact_latency(qa: QAService, query: str, top_k: int, repeats: int) -> Optional[float]:
    values = []
    for _ in range(repeats):
        result = qa.answer(query, top_k=top_k)
        if not result.cache_hit or result.extra.get("cache_type") != "exact":
            return None
        values.append(float(result.latency_ms))
    return statistics.median(values)


def semantic_lookup(
    cache: QACache,
    retriever: MemoryRetriever,
    store: MemoryStore,
    query: str,
    top_k: int,
) -> tuple[Optional[CacheEntry], float]:
    start = time.perf_counter()
    retrieved = retriever.retrieve(query, top_k=top_k, touch=False)
    entry = cache.get_semantic_answer(
        query,
        store=store,
        current_memory_ids=memory_refs(retrieved),
        top_k=max(3, top_k),
    )
    return entry, (time.perf_counter() - start) * 1000


def diagnostics(
    cache: QACache,
    store: MemoryStore,
    query: str,
    accepted: Optional[CacheEntry],
    top_k: int,
) -> dict[str, Any]:
    candidates = cache._semantic_candidates(query, top_k=max(1, top_k))
    sim_threshold = cache._answer_threshold(query)
    keyword_threshold = cache._keyword_threshold(query)
    scores = cache.keyword_scores(query, [entry for entry, _ in candidates])

    selected = None
    if accepted:
        selected = next(
            ((entry, sim) for entry, sim in candidates
             if entry.query_hash == accepted.query_hash),
            None,
        )
    selected = selected or (candidates[0] if candidates else None)
    if not selected:
        return {
            "query": "", "similarity": None, "sim_threshold": sim_threshold,
            "keyword": 0.0, "keyword_threshold": keyword_threshold,
            "reason": "no_candidate",
        }

    entry, similarity = selected
    keyword = float(scores.get(entry.query_hash, 0.0))
    if accepted:
        reason = "semantic_hit"
    elif similarity < sim_threshold:
        reason = "similarity_below_threshold"
    elif keyword < keyword_threshold:
        reason = "keyword_below_threshold"
    elif not cache._memory_refs_still_valid(entry, store):
        reason = "invalid_memory_refs"
    elif cache._has_unresolved_contradiction(entry.retrieved_memory_ids, store):
        reason = "unresolved_contradiction"
    else:
        reason = "rejected"
    return {
        "query": entry.query, "similarity": similarity, "sim_threshold": sim_threshold,
        "keyword": keyword, "keyword_threshold": keyword_threshold, "reason": reason,
    }


def evaluate_case(
    row: dict[str, Any],
    index: int,
    run_dir: Path,
    base: Config,
    store: MemoryStore,
    retriever: MemoryRetriever,
    client: Any,
    top_k: int,
    repeats: int,
) -> dict[str, Any]:
    config = case_config(base, run_dir, index, row["case_id"])
    cache = QACache(config, embed_fn=client.embed)
    qa = QAService(retriever, client, cache, config)

    seed_result = qa.answer(row["query_1"], top_k=top_k)
    miss_ms = float(seed_result.latency_ms)
    exact_ms = exact_latency(qa, row["query_1"], top_k, repeats)

    semantic_entry, first_semantic_ms = semantic_lookup(
        cache, retriever, store, row["query_2"], top_k
    )
    semantic_ms = None
    if semantic_entry:
        values = [first_semantic_ms]
        for _ in range(repeats - 1):
            entry, latency = semantic_lookup(cache, retriever, store, row["query_2"], top_k)
            if entry is None:
                break
            values.append(latency)
        semantic_ms = statistics.median(values)

    diag = diagnostics(cache, store, row["query_2"], semantic_entry, top_k)
    actual = semantic_entry is not None
    speedup = lambda hit: miss_ms / hit if hit and hit > 0 else None
    return {
        "case_id": row["case_id"], "label": row["label"],
        "query_1": row["query_1"], "query_2": row["query_2"],
        "expected_semantic_hit": row["expected"], "actual_semantic_hit": actual,
        "policy_pass": actual == row["expected"],
        "matched_cache_query": diag["query"],
        "query_similarity": rounded(diag["similarity"], 6),
        "similarity_threshold": rounded(diag["sim_threshold"], 6),
        "keyword_score": rounded(diag["keyword"], 6),
        "keyword_threshold": rounded(diag["keyword_threshold"], 6),
        "decision_reason": diag["reason"],
        "miss_latency_ms": rounded(miss_ms), "exact_hit_latency_ms": rounded(exact_ms),
        "semantic_hit_latency_ms": rounded(semantic_ms),
        "exact_speedup_x": rounded(speedup(exact_ms)),
        "semantic_speedup_x": rounded(speedup(semantic_ms)),
        "_seed_miss": not seed_result.cache_hit, "_exact_hit": exact_ms is not None,
    }


def print_summary(rows: list[dict[str, Any]]) -> None:
    positives = [x for x in rows if x["expected_semantic_hit"]]
    negatives = [x for x in rows if not x["expected_semantic_hit"]]
    tp = sum(x["actual_semantic_hit"] for x in positives)
    fp = sum(x["actual_semantic_hit"] for x in negatives)
    total = len(rows)
    pct = lambda x: "N/A" if x is None else f"{x:.1%}"
    num = lambda x, suffix="": "N/A" if x is None else f"{x:.2f}{suffix}"

    print("\n===== QA CACHE EVALUATION =====")
    print(f"cases                   : {total}")
    print(f"seed miss rate          : {pct(sum(x['_seed_miss'] for x in rows) / total)}")
    print(f"exact hit rate          : {pct(sum(x['_exact_hit'] for x in rows) / total)}")
    print(f"semantic accuracy       : {pct(sum(x['policy_pass'] for x in rows) / total)}")
    print(f"semantic precision      : {pct(tp / (tp + fp) if tp + fp else None)}")
    print(f"semantic hit recall     : {pct(tp / len(positives) if positives else None)}")
    print(f"false hit rate          : {pct(fp / len(negatives) if negatives else None)}")
    for key, label, suffix in [
        ("miss_latency_ms", "median miss latency", " ms"),
        ("exact_hit_latency_ms", "median exact latency", " ms"),
        ("semantic_hit_latency_ms", "median semantic latency", " ms"),
        ("exact_speedup_x", "median exact speedup", "x"),
        ("semantic_speedup_x", "median semantic speedup", "x"),
    ]:
        print(f"{label:<24}: {num(median([x[key] for x in rows]), suffix)}")


def main() -> None:
    options = parse_args()
    rows = load_rows(options.dataset, options.limit)
    run_dir = options.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    base = configure(options.home)
    client = get_llm_client(base)
    if not client.health_check():
        raise RuntimeError("LLM backend is unavailable. Start Ollama/llama.cpp first.")
    store = MemoryStore(base, embed_fn=client.embed)
    retriever = MemoryRetriever(store, base, embed_fn=client.embed)

    results = []
    for index, row in enumerate(rows, start=1):
        result = evaluate_case(
            row, index, run_dir, base, store, retriever, client,
            options.top_k, max(1, options.cache_repeats),
        )
        results.append(result)
        print(
            f"[{index}/{len(rows)}] {row['case_id']} | "
            f"expected={'hit' if result['expected_semantic_hit'] else 'miss'} | "
            f"actual={'hit' if result['actual_semantic_hit'] else 'miss'} | "
            f"{result['decision_reason']} | "
            f"{'PASS' if result['policy_pass'] else 'FAIL'}"
        )

    output = run_dir / "qa_cache_results.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print_summary(results)
    print(f"\nResults: {output}")


if __name__ == "__main__":
    main()