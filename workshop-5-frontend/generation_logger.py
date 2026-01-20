"""
Logging module for post and image generation.

Logs RAG context, prompts, and outputs to JSON files for debugging and analysis.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def log_generation(
    post_id: int,
    source: str,  # "api", "notion_watcher", "doc_watcher"
    # RAG context
    rag_query: Optional[str] = None,
    rag_results: Optional[list] = None,
    rag_context_text: Optional[str] = None,
    # Post generation
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    generated_post: Optional[str] = None,
    # Image generation
    image_prompt: Optional[str] = None,
    image_url: Optional[str] = None,
    # Metadata
    extra: Optional[dict] = None,
) -> str:
    """
    Log a post generation event with all relevant context.

    Returns the log file path.
    """
    timestamp = datetime.now()
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "post_id": post_id,
        "source": source,
        "rag": {
            "query": rag_query,
            "results_count": len(rag_results) if rag_results else 0,
            "results": rag_results[:10] if rag_results else [],  # Include more results
            "context_text": rag_context_text,  # Full context, no truncation
        },
        "post_generation": {
            "system_prompt": system_prompt,  # Full prompt
            "user_prompt": user_prompt,  # Full prompt - no truncation
            "generated_post": generated_post,
            "post_length": len(generated_post) if generated_post else 0,
        },
        "image_generation": {
            "prompt": image_prompt,
            "url": image_url,
        },
        "extra": extra or {},
    }

    # Write to individual log file
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_post_{post_id}.json"
    log_path = LOGS_DIR / filename

    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    # Also append to daily summary log
    daily_log = LOGS_DIR / f"{timestamp.strftime('%Y%m%d')}_summary.jsonl"
    with open(daily_log, "a") as f:
        f.write(json.dumps({
            "timestamp": timestamp.isoformat(),
            "post_id": post_id,
            "source": source,
            "post_preview": generated_post[:100] if generated_post else None,
            "has_image": bool(image_url),
            "rag_results_count": len(rag_results) if rag_results else 0,
        }) + "\n")

    print(f"📝 Logged generation to {log_path}")
    return str(log_path)


def get_recent_logs(limit: int = 10) -> list[dict]:
    """Get the most recent generation logs."""
    log_files = sorted(LOGS_DIR.glob("*_post_*.json"), reverse=True)[:limit]

    logs = []
    for log_file in log_files:
        with open(log_file) as f:
            logs.append(json.load(f))

    return logs
