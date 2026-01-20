"""
Workshop 5 Frontend: Embedding generation using local MiniLM-L6-v2 model.

Uses fastembed (ONNX-based) to run embeddings locally instead of API calls.
MiniLM-L6-v2 produces 384-dimensional embeddings optimized for semantic similarity.

Document source: Notion API
"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

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

load_dotenv(Path(__file__).parent.parent / ".env")

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


# ========================================
# NOTION DOCUMENT EMBEDDING
# ========================================


def fetch_notion_pages() -> list[dict]:
    """
    Fetch all pages from Notion under the parent page.

    Returns list of dicts with: id, title, content, last_edited
    """
    from notion_watcher import (
        get_notion_client,
        get_parent_page_id,
        get_child_pages,
        get_page_content,
    )

    notion = get_notion_client()
    parent_id = get_parent_page_id()

    pages = []
    child_pages = get_child_pages(notion, parent_id)

    for page_info in child_pages:
        page_data = get_page_content(notion, page_info["id"])
        pages.append(page_data)

    return pages


def _split_large_text(text: str, max_chars: int) -> list[str]:
    """
    Split large text by sentences, falling back to word boundaries.

    Used when a single paragraph exceeds max_chars.
    """
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)

    pieces = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_chars and current:
            pieces.append(current.strip())
            current = ""
        current += sent + " "
    if current.strip():
        pieces.append(current.strip())

    # If any piece is still too large, split by words (not characters)
    final = []
    for p in pieces:
        if len(p) <= max_chars:
            final.append(p)
            continue
        # Split by words
        words = p.split()
        chunk = ""
        for word in words:
            if len(chunk) + len(word) + 1 > max_chars and chunk:
                final.append(chunk.strip())
                chunk = ""
            chunk += word + " "
        if chunk.strip():
            final.append(chunk.strip())
    return final


def chunk_notion_content(title: str, content: str, page_id: str, max_chars: int = 1000) -> list[dict]:
    """
    Chunk Notion page content for embedding.

    Each chunk includes the document title for context.
    Handles large paragraphs by splitting them into smaller pieces.
    """
    chunks = []

    # Split by double newlines (paragraphs)
    paragraphs = content.split("\n\n")

    current_chunk = f"# {title}\n\n"
    title_overhead = len(title) + 10  # Account for "# title\n\n"

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Split large paragraphs into smaller pieces
        if len(para) > max_chars - title_overhead:
            para_pieces = _split_large_text(para, max_chars - title_overhead)
        else:
            para_pieces = [para]

        for piece in para_pieces:
            # If adding this piece exceeds max, save current chunk and start new
            if len(current_chunk) + len(piece) > max_chars and len(current_chunk) > title_overhead:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": {
                        "source_type": "notion_doc",
                        "page_id": page_id,
                        "title": title,
                        "chunk_index": len(chunks),
                    }
                })
                current_chunk = f"# {title}\n\n"

            current_chunk += piece + "\n\n"

    # Don't forget the last chunk
    if len(current_chunk.strip()) > title_overhead:
        chunks.append({
            "content": current_chunk.strip(),
            "metadata": {
                "source_type": "notion_doc",
                "page_id": page_id,
                "title": title,
                "chunk_index": len(chunks),
            }
        })

    # If no chunks were created, create one with whatever we have
    if not chunks and content.strip():
        chunks.append({
            "content": f"# {title}\n\n{content[:max_chars]}",
            "metadata": {
                "source_type": "notion_doc",
                "page_id": page_id,
                "title": title,
                "chunk_index": 0,
            }
        })

    return chunks


def embed_business_docs(force: bool = False):
    """
    Embed business documents from Notion.

    Args:
        force: If True, re-embed all documents. If False, only embed new/changed ones.
    """
    db = get_db()

    print("Fetching Notion pages...")
    try:
        pages = fetch_notion_pages()
    except Exception as e:
        print(f"Error fetching Notion pages: {e}")
        print("Make sure NOTION_API_KEY and NOTION_PARENT_PAGE_ID are set.")
        return

    if not pages:
        print("No Notion pages found.")
        return

    print(f"Found {len(pages)} Notion page(s)")

    if force:
        print("Force refresh: deleting existing doc embeddings...")
        deleted = delete_embeddings_by_source(db, "notion_doc")
        deleted += delete_embeddings_by_source(db, "business_doc")  # Clean up legacy
        print(f"Deleted {deleted} existing embeddings")

    total_chunks = 0
    for page in pages:
        page_id = page["id"]
        title = page["title"]
        content = page["content"]

        if not content.strip():
            print(f"  Skipping {title}: no content")
            continue

        # Delete existing embeddings for this page
        delete_embeddings_by_source(db, "notion_doc", page_id)

        # Chunk the content
        chunks = chunk_notion_content(title, content, page_id)
        print(f"  {title}: {len(chunks)} chunk(s)")

        # Generate embeddings
        chunk_texts = [c["content"] for c in chunks]
        embeddings = generate_embeddings_batch(chunk_texts)

        # Save each chunk with its embedding
        for chunk, embedding in zip(chunks, embeddings):
            save_embedding(
                db,
                source_type="notion_doc",
                content=chunk["content"],
                embedding=embedding,
                source_id=page_id,
                metadata=chunk["metadata"],
            )

        total_chunks += len(chunks)

    # Rebuild FTS index
    print("Rebuilding FTS index...")
    rebuild_fts_index(db)

    print(f"\nTotal Notion doc embeddings: {total_chunks}")


def embed_posts():
    """Embed only posted (published) posts that don't have embeddings yet."""
    db = get_db()

    all_posts = get_all_posts(db)

    # Only embed posts that have been published (posted=True)
    posted_posts = [p for p in all_posts if p.get("posted")]
    unpublished_posts = [p for p in all_posts if not p.get("posted")]

    # Delete embeddings for unpublished posts (in case they were previously embedded)
    deleted_count = 0
    for post in unpublished_posts:
        post_id = str(post["id"])
        existing = get_embedding_by_source_id(db, "post", post_id)
        if existing:
            from database import delete_embeddings_by_source
            # Delete this specific post's embedding by source_id
            cursor = db.cursor()
            cursor.execute("DELETE FROM embeddings WHERE source_type = ? AND source_id = ?", ("post", post_id))
            db.commit()
            deleted_count += 1

    if deleted_count > 0:
        print(f"Removed {deleted_count} embedding(s) for unpublished posts")

    new_count = 0
    for post in posted_posts:
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
        print(f"Embedded {new_count} new posted post(s)")
        rebuild_fts_index(db)
    else:
        print(f"No new posted posts to embed ({len(posted_posts)} already embedded or 0 posted)")

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
    print("INITIALIZING EMBEDDINGS")
    print("Document source: Notion")
    print("=" * 60)

    print("\n--- Business Documents (Notion) ---")
    embed_business_docs(force=True)

    print("\n--- Posts ---")
    embed_posts()

    print("\n--- Responses ---")
    embed_responses()

    print("\n--- Summary ---")
    db = get_db()
    print(f"Notion doc embeddings: {count_embeddings(db, 'notion_doc')}")
    print(f"Post embeddings: {count_embeddings(db, 'post')}")
    print(f"Response embeddings: {count_embeddings(db, 'response')}")
    print(f"Total embeddings: {count_embeddings(db)}")


def refresh_embeddings():
    """Refresh embeddings: re-embed docs from Notion, add new posts/responses."""
    print("=" * 60)
    print("REFRESHING EMBEDDINGS")
    print("Document source: Notion")
    print("=" * 60)

    print("\n--- Business Documents (Notion) ---")
    embed_business_docs(force=True)  # Always refresh from Notion

    print("\n--- Checking Posts ---")
    embed_posts()

    print("\n--- Checking Responses ---")
    embed_responses()

    print("\n--- Summary ---")
    db = get_db()
    print(f"Notion doc embeddings: {count_embeddings(db, 'notion_doc')}")
    print(f"Post embeddings: {count_embeddings(db, 'post')}")
    print(f"Response embeddings: {count_embeddings(db, 'response')}")
    print(f"Total embeddings: {count_embeddings(db)}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Manage RAG embeddings (Notion + local MiniLM-L6-v2)")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize all embeddings from scratch",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh embeddings from Notion",
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
        print(f"  Notion docs: {count_embeddings(db, 'notion_doc')}")
        print(f"  Posts: {count_embeddings(db, 'post')}")
        print(f"  Responses: {count_embeddings(db, 'response')}")
        print(f"  Total: {count_embeddings(db)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
