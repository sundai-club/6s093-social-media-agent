"""
Workshop 4: RAG (Retrieval-Augmented Generation) system with hybrid search.

HYBRID SEARCH STRATEGY:
=======================
Combines keyword matching and semantic similarity for better retrieval.

Formula: final_score = (keyword_weight * keyword_score) + (semantic_weight * cosine_similarity)

Default weights: 0.3 keyword, 0.7 semantic

Why hybrid search?
- Keyword matching catches exact terms (company names, product names, acronyms)
- Semantic search catches related concepts even with different words
- Combined approach handles both "Emanon" (exact) and "AI consulting" (semantic)

The system returns top 10 results mixed from all source types (business_doc, post, response).
"""

import math
import re
from collections import Counter
from typing import Optional

from database import get_db, get_all_embeddings


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Cosine similarity = (A · B) / (||A|| * ||B||)

    Returns value between -1 and 1, where:
    - 1 = identical direction (most similar)
    - 0 = orthogonal (unrelated)
    - -1 = opposite direction (most dissimilar)

    For normalized embeddings (which OpenAI provides), this simplifies to dot product.
    """
    if len(vec1) != len(vec2):
        raise ValueError(f"Vector length mismatch: {len(vec1)} vs {len(vec2)}")

    # Dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))

    # Magnitudes
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def tokenize(text: str) -> list[str]:
    """
    Simple tokenization for keyword matching.

    Converts to lowercase, removes punctuation, splits on whitespace.
    """
    # Lowercase and remove non-alphanumeric (keep spaces)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Split and filter empty
    tokens = [t for t in text.split() if t]
    return tokens


def compute_keyword_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """
    Compute keyword overlap score using TF-IDF-inspired approach.

    SCORING APPROACH:
    ================
    1. Count how many query tokens appear in the document
    2. Weight by inverse document frequency (rarer terms count more)
    3. Normalize by query length

    This is a simplified TF-IDF that works well for short queries.
    """
    if not query_tokens or not doc_tokens:
        return 0.0

    # Convert to sets for fast lookup
    query_set = set(query_tokens)
    doc_counter = Counter(doc_tokens)

    # Count matches with term frequency weighting
    matches = 0
    for token in query_set:
        if token in doc_counter:
            # Log-scaled term frequency (diminishing returns for repeated terms)
            tf = 1 + math.log(doc_counter[token])
            matches += tf

    # Normalize by query length
    return matches / len(query_set)


def hybrid_search(
    query: str,
    query_embedding: list[float],
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
    top_k: int = 10,
    source_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Perform hybrid search combining keyword and semantic similarity.

    SEARCH PROCESS:
    ===============
    1. Load all embeddings from database
    2. For each embedding:
       a. Compute keyword score (token overlap)
       b. Compute semantic score (cosine similarity)
       c. Combine with weights
    3. Sort by combined score
    4. Return top K results

    Args:
        query: User's search query (text)
        query_embedding: Pre-computed embedding of the query
        keyword_weight: Weight for keyword matching (default 0.3)
        semantic_weight: Weight for semantic similarity (default 0.7)
        top_k: Number of results to return (default 10)
        source_types: Filter to specific types (None = all)

    Returns:
        List of dicts with content, source_type, scores, and metadata
    """
    db = get_db()

    # Get all embeddings (optionally filtered)
    all_embeddings = get_all_embeddings(db)

    if source_types:
        all_embeddings = [e for e in all_embeddings if e["source_type"] in source_types]

    if not all_embeddings:
        return []

    # Tokenize query for keyword matching
    query_tokens = tokenize(query)

    # Score each document
    scored_results = []
    for emb in all_embeddings:
        # Tokenize document content
        doc_tokens = tokenize(emb["content"])

        # Compute keyword score (0-1 range, roughly)
        keyword_score = compute_keyword_score(query_tokens, doc_tokens)

        # Compute semantic score (cosine similarity, -1 to 1, but usually 0-1 for related content)
        semantic_score = cosine_similarity(query_embedding, emb["embedding"])
        # Normalize to 0-1 range (shift from [-1,1] to [0,1])
        semantic_score = (semantic_score + 1) / 2

        # Combine scores
        final_score = (keyword_weight * keyword_score) + (semantic_weight * semantic_score)

        scored_results.append({
            "content": emb["content"],
            "source_type": emb["source_type"],
            "source_id": emb["source_id"],
            "metadata": emb["metadata"],
            "keyword_score": keyword_score,
            "semantic_score": semantic_score,
            "final_score": final_score,
        })

    # Sort by final score (descending)
    scored_results.sort(key=lambda x: x["final_score"], reverse=True)

    # Return top K
    return scored_results[:top_k]


def format_context_for_prompt(results: list[dict], max_tokens: int = 2000) -> str:
    """
    Format search results into a context string for the LLM prompt.

    CONTEXT FORMATTING:
    ===================
    Each result is formatted as:
    ```
    [N. source_type] (score: X.XX)
    Content snippet...
    ```

    We truncate content to stay within token budget (~4 chars per token estimate).
    """
    if not results:
        return "No relevant context found."

    # Estimate: ~4 characters per token
    char_budget = max_tokens * 4

    context_parts = []
    chars_used = 0

    for i, result in enumerate(results, 1):
        # Format header
        header = f"[{i}. {result['source_type']}] (score: {result['final_score']:.2f})"

        # Truncate content if needed
        content = result["content"]
        # Reserve space for header and newlines
        available_chars = char_budget - chars_used - len(header) - 10

        if available_chars <= 100:
            # Stop if we're running low on budget
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
    keyword_weight: float = 0.3,
    semantic_weight: float = 0.7,
) -> tuple[str, list[dict]]:
    """
    High-level function to retrieve and format context for RAG.

    Args:
        query: The user's query/comment
        query_embedding: Pre-computed embedding of the query
        top_k: Number of results to retrieve
        keyword_weight: Weight for keyword matching
        semantic_weight: Weight for semantic similarity

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

    parser = argparse.ArgumentParser(description="Test RAG retrieval")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--keyword-weight", type=float, default=0.3)
    parser.add_argument("--semantic-weight", type=float, default=0.7)

    args = parser.parse_args()

    print(f"Query: {args.query}")
    print("=" * 60)

    # Generate embedding for query
    print("Generating query embedding...")
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
              f"(kw={r['keyword_score']:.3f}, sem={r['semantic_score']:.3f})")
        print(f"   {r['content'][:100]}...")
        print()

    print("=" * 60)
    print("FORMATTED CONTEXT FOR PROMPT:")
    print("=" * 60)
    print(context)


if __name__ == "__main__":
    main()
