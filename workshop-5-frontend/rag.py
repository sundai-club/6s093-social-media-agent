"""
Workshop 5 Frontend: RAG (Retrieval-Augmented Generation) system with hybrid search.
(Copied from workshop-4)

HYBRID SEARCH STRATEGY:
=======================
Combines BM25 keyword matching (via SQLite FTS5) and semantic similarity (cosine).

Formula: final_score = 0.5 * normalized_bm25 + 0.5 * normalized_cosine

Score normalization (both to 0-1 range):
- BM25: SQLite FTS5 returns negative scores (more negative = better match)
        Normalized using: 1 - (score - min_score) / (max_score - min_score)
- Cosine: Already in [-1, 1] range, normalized to [0, 1] using: (score + 1) / 2

Why 50-50 hybrid search?
- BM25 catches exact terms (company names, product names, acronyms)
- Semantic search catches related concepts even with different words
- Equal weighting provides balanced results for both exact and semantic matches
"""

import math
import numpy as np
from typing import Optional

from database import get_db, get_all_embeddings, bm25_search_all, get_embedding_by_id


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors using numpy for efficiency.

    Returns value between -1 and 1.
    """
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    dot_product = np.dot(v1, v2)
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return float(dot_product / (magnitude1 * magnitude2))


def normalize_bm25_scores(bm25_scores: dict[int, float]) -> dict[int, float]:
    """
    Normalize BM25 scores to 0-1 range.

    SQLite FTS5 BM25 scores are negative (more negative = better match).
    We normalize so that the best match gets score 1.0 and worst gets 0.0.

    If only one result, it gets score 1.0.
    If no results, returns empty dict.
    """
    if not bm25_scores:
        return {}

    scores = list(bm25_scores.values())
    min_score = min(scores)  # Most negative = best match
    max_score = max(scores)  # Least negative = worst match

    if min_score == max_score:
        # All scores are the same, give them all 1.0
        return {id: 1.0 for id in bm25_scores}

    # Normalize: best (most negative) -> 1.0, worst (least negative) -> 0.0
    normalized = {}
    score_range = max_score - min_score
    for id, score in bm25_scores.items():
        # Invert so that better BM25 (more negative) gets higher normalized score
        normalized[id] = (max_score - score) / score_range

    return normalized


def normalize_cosine_score(score: float) -> float:
    """
    Normalize cosine similarity from [-1, 1] to [0, 1].

    For text embeddings, scores are typically in [0, 1] already,
    but this handles edge cases.
    """
    return (score + 1) / 2


def hybrid_search(
    query: str,
    query_embedding: list[float],
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
    top_k: int = 10,
    source_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Perform hybrid search combining BM25 keyword matching and semantic similarity.

    SEARCH PROCESS:
    ===============
    1. Get BM25 scores from SQLite FTS5 for all matching documents
    2. Normalize BM25 scores to 0-1 range
    3. Load all embeddings and compute cosine similarity
    4. Normalize cosine scores to 0-1 range
    5. Combine with weights: final = keyword_weight * bm25 + semantic_weight * cosine
    6. Sort by combined score and return top K

    Args:
        query: User's search query (text)
        query_embedding: Pre-computed embedding of the query
        keyword_weight: Weight for BM25 keyword matching (default 0.5)
        semantic_weight: Weight for semantic similarity (default 0.5)
        top_k: Number of results to return (default 10)
        source_types: Filter to specific types (None = all)

    Returns:
        List of dicts with content, source_type, scores, and metadata
    """
    db = get_db()

    # Step 1: Get BM25 scores from FTS5
    bm25_scores_raw = bm25_search_all(db, query, source_types[0] if source_types and len(source_types) == 1 else None)

    # Step 2: Normalize BM25 scores to 0-1
    bm25_scores_normalized = normalize_bm25_scores(bm25_scores_raw)

    # Step 3: Get all embeddings
    all_embeddings = get_all_embeddings(db)

    if source_types:
        all_embeddings = [e for e in all_embeddings if e["source_type"] in source_types]

    if not all_embeddings:
        return []

    # Step 4 & 5: Compute cosine similarities and combine scores
    scored_results = []

    for emb in all_embeddings:
        emb_id = emb["id"]

        # Get normalized BM25 score (0 if document didn't match query)
        bm25_score = bm25_scores_normalized.get(emb_id, 0.0)

        # Compute and normalize cosine similarity
        cosine_raw = cosine_similarity(query_embedding, emb["embedding"])
        cosine_score = normalize_cosine_score(cosine_raw)

        # Combine scores with weights
        final_score = (keyword_weight * bm25_score) + (semantic_weight * cosine_score)

        scored_results.append({
            "content": emb["content"],
            "source_type": emb["source_type"],
            "source_id": emb["source_id"],
            "metadata": emb["metadata"],
            "bm25_score": bm25_score,  # Normalized 0-1
            "cosine_score": cosine_score,  # Normalized 0-1
            "final_score": final_score,
        })

    # Step 6: Sort by final score (descending)
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Return top K
    return scored_results[:top_k]


def format_context_for_prompt(results: list[dict], max_tokens: int = 2000) -> str:
    """
    Format search results into a context string for the LLM prompt.

    Each result is formatted as:
    ```
    [N. source_type] (score: X.XX, bm25: X.XX, cosine: X.XX)
    Content snippet...
    ```
    """
    if not results:
        return "No relevant context found."

    # Estimate: ~4 characters per token
    char_budget = max_tokens * 4

    context_parts = []
    chars_used = 0

    for i, result in enumerate(results, 1):
        # Format header with all scores for debugging
        header = (f"[{i}. {result['source_type']}] "
                  f"(score: {result['final_score']:.2f}, "
                  f"bm25: {result['bm25_score']:.2f}, "
                  f"cosine: {result['cosine_score']:.2f})")

        content = result["content"]
        available_chars = char_budget - chars_used - len(header) - 10

        if available_chars <= 100:
            break

        if len(content) > available_chars:
            content = content[:available_chars - 3] + "..."

        entry = f"{header}\n{content}\n"
        context_parts.append(entry)
        chars_used += len(entry)

    return "\n".join(context_parts)


def retrieve_context(
    query: str,
    query_embedding: list[float],
    top_k: int = 10,
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> tuple[str, list[dict]]:
    """
    High-level function to retrieve and format context for RAG.

    Args:
        query: The user's query/comment
        query_embedding: Pre-computed embedding of the query
        top_k: Number of results to retrieve
        keyword_weight: Weight for BM25 matching (default 0.5)
        semantic_weight: Weight for semantic similarity (default 0.5)

    Returns:
        Tuple of (formatted_context_string, raw_results_list)
    """
    results = hybrid_search(
        query=query,
        query_embedding=query_embedding,
        keyword_weight=keyword_weight,
        semantic_weight=semantic_weight,
        top_k=top_k,
    )

    formatted_context = format_context_for_prompt(results)

    return formatted_context, results


def main():
    """Test the RAG system with a sample query."""
    import argparse
    from embeddings import generate_embedding

    parser = argparse.ArgumentParser(description="Test RAG retrieval with hybrid search")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--keyword-weight", type=float, default=0.5, help="BM25 weight (default 0.5)")
    parser.add_argument("--semantic-weight", type=float, default=0.5, help="Cosine weight (default 0.5)")

    args = parser.parse_args()

    print(f"Query: {args.query}")
    print(f"Weights: BM25={args.keyword_weight}, Cosine={args.semantic_weight}")
    print("=" * 60)

    # Generate embedding for query
    print("Generating query embedding (local MiniLM-L6-v2)...")
    query_embedding = generate_embedding(args.query)

    # Perform hybrid search
    print("Searching...")
    context, results = retrieve_context(
        query=args.query,
        query_embedding=query_embedding,
        top_k=args.top_k,
        keyword_weight=args.keyword_weight,
        semantic_weight=args.semantic_weight,
    )

    print(f"\nFound {len(results)} results:\n")

    for i, r in enumerate(results, 1):
        print(f"{i}. [{r['source_type']}] score={r['final_score']:.3f} "
              f"(bm25={r['bm25_score']:.3f}, cosine={r['cosine_score']:.3f})")
        print(f"   {r['content'][:100]}...")
        print()

    print("=" * 60)
    print("FORMATTED CONTEXT FOR PROMPT:")
    print("=" * 60)
    print(context)


if __name__ == "__main__":
    main()
