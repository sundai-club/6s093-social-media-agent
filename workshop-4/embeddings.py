"""
Workshop 4: Embedding generation and management for RAG system.
Generates embeddings via OpenRouter/OpenAI API and stores them in SQLite.
"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from database import (
    get_db,
    save_embedding,
    delete_embeddings_by_source,
    get_embedding_by_source_id,
    count_embeddings,
    get_all_posts,
    get_all_responses,
)

load_dotenv(Path(__file__).parent.parent / ".env")

BUSINESS_DOCS_DIR = Path(__file__).parent.parent / "business-docs"

# Store file modification times for change detection
_file_mtimes: dict[str, float] = {}


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_embedding(text: str, client: Optional[OpenAI] = None) -> list[float]:
    """Generate an embedding for the given text using OpenRouter."""
    if client is None:
        client = get_openai_client()

    # Use a free embedding model available on OpenRouter
    # text-embedding-3-small is available and efficient
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text,
    )

    return response.data[0].embedding


def generate_embeddings_batch(texts: list[str], client: Optional[OpenAI] = None) -> list[list[float]]:
    """Generate embeddings for multiple texts in a batch."""
    if client is None:
        client = get_openai_client()

    if not texts:
        return []

    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts,
    )

    # Sort by index to ensure correct ordering
    sorted_data = sorted(response.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


def chunk_business_doc(file_path: Path) -> list[dict]:
    """
    Chunk a business document into sections for embedding.

    CHUNKING STRATEGY EXPLAINED:
    ============================

    The business docs are well-structured markdown files with:
    - Top-level `# Title` headers (document title)
    - `## Section` headers for major divisions (Mission, Core Offerings, etc.)
    - Lists, tables, and paragraphs within sections

    Why section-based chunking?
    ---------------------------
    1. **Semantic coherence**: Each ## section discusses one topic, making it
       a natural unit of meaning for embedding. Splitting mid-paragraph would
       create chunks that lose context.

    2. **Optimal size**: Business doc sections are typically 200-500 tokens,
       which is ideal for embedding models (not too short to lack context,
       not too long to dilute the semantic signal).

    3. **Parent context preservation**: We prepend the document title to each
       chunk so the embedding captures "this is about Emanon's Mission" not
       just generic text about missions.

    Chunk format example:
    ---------------------
    ```
    [From: overview.md]
    # Emanon - Business Overview

    ## Mission
    Deliver curated, actionable intelligence about AI...
    ```

    This gives the embedding model:
    - Source attribution (for debugging/tracing)
    - Document-level context (what company/product)
    - Section-level content (the actual retrievable info)
    """
    content = file_path.read_text()
    filename = file_path.name

    # Step 1: Extract the main document title (# header at top)
    # This provides parent context for all chunks from this file
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    doc_title = title_match.group(1) if title_match else filename

    # Step 2: Split on ## headers using lookahead regex
    # The (?=^##\s+) pattern splits BEFORE each ## header, keeping the header
    # with its content. This preserves section boundaries cleanly.
    sections = re.split(r'(?=^##\s+)', content, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Step 3: Extract section title for metadata
        # Used for filtering/debugging, e.g., "find all Mission chunks"
        section_title_match = re.search(r'^##\s+(.+)$', section, re.MULTILINE)
        section_title = section_title_match.group(1) if section_title_match else "Introduction"

        # Step 4: Construct the chunk with full context
        # Format: [source] + doc title + section content
        # This ensures the embedding captures both the source and the content
        chunk_content = f"[From: {filename}]\n# {doc_title}\n\n{section}"

        chunks.append({
            "content": chunk_content,
            "metadata": {
                "source_file": filename,          # For change detection
                "section_title": section_title,   # For filtering/display
                "file_mtime": file_path.stat().st_mtime,  # For refresh logic
            }
        })

    # Fallback: If no ## headers found, treat the whole document as one chunk
    # This handles edge cases like very short docs or non-standard formatting
    if not chunks:
        chunks.append({
            "content": f"[From: {filename}]\n\n{content}",
            "metadata": {
                "source_file": filename,
                "section_title": "Full Document",
                "file_mtime": file_path.stat().st_mtime,
            }
        })

    return chunks


def get_changed_business_docs(db_conn) -> list[Path]:
    """
    Check which business docs have changed since last embedding.
    Returns list of files that need re-embedding.
    """
    changed = []

    for doc_path in sorted(BUSINESS_DOCS_DIR.glob("*.md")):
        current_mtime = doc_path.stat().st_mtime

        # Check if we have an embedding for this file
        existing = get_embedding_by_source_id(db_conn, "business_doc", doc_path.name)

        if existing is None:
            # No embedding exists
            changed.append(doc_path)
        elif existing.get("metadata", {}).get("file_mtime", 0) < current_mtime:
            # File has been modified
            changed.append(doc_path)

    return changed


def embed_business_docs(force: bool = False):
    """
    Embed all business documents or only changed ones.

    Args:
        force: If True, re-embed all documents. If False, only embed changed ones.
    """
    db = get_db()
    client = get_openai_client()

    if force:
        print("Force refresh: deleting existing business doc embeddings...")
        deleted = delete_embeddings_by_source(db, "business_doc")
        print(f"Deleted {deleted} existing embeddings")
        docs_to_process = list(sorted(BUSINESS_DOCS_DIR.glob("*.md")))
    else:
        docs_to_process = get_changed_business_docs(db)

    if not docs_to_process:
        print("No business documents need embedding.")
        return

    print(f"Processing {len(docs_to_process)} document(s)...")

    for doc_path in docs_to_process:
        print(f"\nProcessing: {doc_path.name}")

        # Delete existing embeddings for this file
        delete_embeddings_by_source(db, "business_doc", doc_path.name)

        # Chunk the document
        chunks = chunk_business_doc(doc_path)
        print(f"  Created {len(chunks)} chunk(s)")

        # Generate embeddings for all chunks in batch
        chunk_texts = [c["content"] for c in chunks]
        embeddings = generate_embeddings_batch(chunk_texts, client)

        # Save each chunk with its embedding
        for chunk, embedding in zip(chunks, embeddings):
            save_embedding(
                db,
                source_type="business_doc",
                content=chunk["content"],
                embedding=embedding,
                source_id=doc_path.name,
                metadata=chunk["metadata"],
            )

        print(f"  Saved {len(chunks)} embedding(s)")

    total = count_embeddings(db, "business_doc")
    print(f"\nTotal business doc embeddings: {total}")


def embed_posts():
    """Embed all posts that don't have embeddings yet."""
    db = get_db()
    client = get_openai_client()

    posts = get_all_posts(db)

    new_count = 0
    for post in posts:
        post_id = str(post["id"])

        # Check if embedding exists
        existing = get_embedding_by_source_id(db, "post", post_id)
        if existing:
            continue

        # Generate and save embedding
        embedding = generate_embedding(post["content"], client)
        save_embedding(
            db,
            source_type="post",
            content=post["content"],
            embedding=embedding,
            source_id=post_id,
            metadata={
                "post_url": post.get("post_url"),
                "created_at": post.get("created_at"),
            }
        )
        new_count += 1

    if new_count > 0:
        print(f"Embedded {new_count} new post(s)")

    return new_count


def embed_responses():
    """Embed all responses that don't have embeddings yet."""
    db = get_db()
    client = get_openai_client()

    responses = get_all_responses(db)

    new_count = 0
    for resp in responses:
        resp_id = str(resp["id"])

        # Check if embedding exists
        existing = get_embedding_by_source_id(db, "response", resp_id)
        if existing:
            continue

        # Generate and save embedding
        embedding = generate_embedding(resp["response_text"], client)
        save_embedding(
            db,
            source_type="response",
            content=resp["response_text"],
            embedding=embedding,
            source_id=resp_id,
            metadata={
                "original_post_author": resp.get("original_post_author"),
                "created_at": resp.get("created_at"),
            }
        )
        new_count += 1

    if new_count > 0:
        print(f"Embedded {new_count} new response(s)")

    return new_count


def init_embeddings():
    """Initialize embeddings from all sources: business docs, posts, responses."""
    print("=" * 60)
    print("INITIALIZING EMBEDDINGS")
    print("=" * 60)

    print("\n--- Business Documents ---")
    embed_business_docs(force=True)

    print("\n--- Posts ---")
    embed_posts()

    print("\n--- Responses ---")
    embed_responses()

    print("\n--- Summary ---")
    db = get_db()
    print(f"Business doc embeddings: {count_embeddings(db, 'business_doc')}")
    print(f"Post embeddings: {count_embeddings(db, 'post')}")
    print(f"Response embeddings: {count_embeddings(db, 'response')}")
    print(f"Total embeddings: {count_embeddings(db)}")


def refresh_embeddings():
    """Refresh embeddings: re-embed changed business docs, add new posts/responses."""
    print("=" * 60)
    print("REFRESHING EMBEDDINGS")
    print("=" * 60)

    print("\n--- Checking Business Documents ---")
    embed_business_docs(force=False)

    print("\n--- Checking Posts ---")
    embed_posts()

    print("\n--- Checking Responses ---")
    embed_responses()

    print("\n--- Summary ---")
    db = get_db()
    print(f"Business doc embeddings: {count_embeddings(db, 'business_doc')}")
    print(f"Post embeddings: {count_embeddings(db, 'post')}")
    print(f"Response embeddings: {count_embeddings(db, 'response')}")
    print(f"Total embeddings: {count_embeddings(db)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage RAG embeddings")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize all embeddings from scratch",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh embeddings (re-embed changed docs, add new posts/responses)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show embedding statistics",
    )

    args = parser.parse_args()

    if args.init:
        init_embeddings()
    elif args.refresh:
        refresh_embeddings()
    elif args.stats:
        db = get_db()
        print("Embedding Statistics:")
        print(f"  Business docs: {count_embeddings(db, 'business_doc')}")
        print(f"  Posts: {count_embeddings(db, 'post')}")
        print(f"  Responses: {count_embeddings(db, 'response')}")
        print(f"  Total: {count_embeddings(db)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
