"""
Workshop 4: Embedding generation using local MiniLM-L6-v2 model.

Uses fastembed (ONNX-based) to run embeddings locally instead of API calls.
MiniLM-L6-v2 produces 384-dimensional embeddings optimized for semantic similarity.
"""

import re
from pathlib import Path
from typing import Optional

from database import (
    get_db,
    save_embedding,
    delete_embeddings_by_source,
    get_embedding_by_source_id,
    count_embeddings,
    get_all_posts,
    get_all_responses,
    rebuild_fts_index,
)

BUSINESS_DOCS_DIR = Path(__file__).parent.parent / "business-docs"

# Global model instance (lazy loaded)
_model = None


def get_embedding_model():
    """Get or initialize the fastembed model."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        print("Loading MiniLM-L6-v2 embedding model (ONNX)...")
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("Model loaded successfully.")
    return _model


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for the given text using local MiniLM-L6-v2.

    Returns a 384-dimensional vector as a list of floats.
    """
    model = get_embedding_model()
    # fastembed returns a generator, convert to list and get first result
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist()


def generate_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in a batch.

    More efficient than calling generate_embedding() in a loop.
    """
    if not texts:
        return []

    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def chunk_business_doc(file_path: Path) -> list[dict]:
    """
    Chunk a business document into sections for embedding.

    CHUNKING STRATEGY:
    - Split on ## headers to get semantically coherent sections
    - Prepend document title for context
    - Each chunk includes source attribution
    """
    content = file_path.read_text()
    filename = file_path.name

    # Extract the main document title
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    doc_title = title_match.group(1) if title_match else filename

    # Split on ## headers
    sections = re.split(r'(?=^##\s+)', content, flags=re.MULTILINE)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract section title
        section_title_match = re.search(r'^##\s+(.+)$', section, re.MULTILINE)
        section_title = section_title_match.group(1) if section_title_match else "Introduction"

        # Construct chunk with context
        chunk_content = f"[From: {filename}]\n# {doc_title}\n\n{section}"

        chunks.append({
            "content": chunk_content,
            "metadata": {
                "source_file": filename,
                "section_title": section_title,
                "file_mtime": file_path.stat().st_mtime,
            }
        })

    # Fallback: whole document as one chunk
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
    """Check which business docs have changed since last embedding."""
    changed = []

    for doc_path in sorted(BUSINESS_DOCS_DIR.glob("*.md")):
        current_mtime = doc_path.stat().st_mtime

        existing = get_embedding_by_source_id(db_conn, "business_doc", doc_path.name)

        if existing is None:
            changed.append(doc_path)
        elif existing.get("metadata", {}).get("file_mtime", 0) < current_mtime:
            changed.append(doc_path)

    return changed


def embed_business_docs(force: bool = False):
    """
    Embed all business documents or only changed ones.

    Args:
        force: If True, re-embed all documents. If False, only embed changed ones.
    """
    db = get_db()

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
        embeddings = generate_embeddings_batch(chunk_texts)

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

    # Rebuild FTS index after bulk operations
    print("Rebuilding FTS index...")
    rebuild_fts_index(db)

    total = count_embeddings(db, "business_doc")
    print(f"\nTotal business doc embeddings: {total}")


def embed_posts():
    """Embed all posts that don't have embeddings yet."""
    db = get_db()

    posts = get_all_posts(db)

    new_count = 0
    for post in posts:
        post_id = str(post["id"])

        existing = get_embedding_by_source_id(db, "post", post_id)
        if existing:
            continue

        embedding = generate_embedding(post["content"])
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
        rebuild_fts_index(db)

    return new_count


def embed_responses():
    """Embed all responses that don't have embeddings yet."""
    db = get_db()

    responses = get_all_responses(db)

    new_count = 0
    for resp in responses:
        resp_id = str(resp["id"])

        existing = get_embedding_by_source_id(db, "response", resp_id)
        if existing:
            continue

        embedding = generate_embedding(resp["response_text"])
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
        rebuild_fts_index(db)

    return new_count


def init_embeddings():
    """Initialize embeddings from all sources."""
    print("=" * 60)
    print("INITIALIZING EMBEDDINGS (using local MiniLM-L6-v2 via ONNX)")
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

    parser = argparse.ArgumentParser(description="Manage RAG embeddings (local MiniLM-L6-v2 via ONNX)")
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
