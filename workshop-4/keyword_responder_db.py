"""
Workshop 3: Keyword Responder with Database Tracking
Extends Workshop 1's keyword responder to save responses to SQLite database.

This version can be deployed to a GCP VM and tracked via the FastAPI server.
"""

import sys
from pathlib import Path

# Add workshop-1 to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "workshop-1"))
from keyword_responder import (
    read_business_docs,
    search_mastodon,
    generate_responses,
    post_reply,
    SEARCH_KEYWORDS,
)

from database import get_db, save_response


def main(post_replies: bool = False):
    print("Reading business docs for context...")
    business_context = read_business_docs()

    print(f"Searching Mastodon for keywords: {SEARCH_KEYWORDS[:3]}...")
    posts = search_mastodon(SEARCH_KEYWORDS)
    print(f"Found {len(posts)} unique posts")

    if not posts:
        print("No posts found to respond to.")
        return

    print("Generating responses with LLM...")
    responses = generate_responses(posts[:5], business_context)

    # Get database connection
    db = get_db()

    print(f"\n{'=' * 60}")
    print("GENERATED RESPONSES")
    print("=" * 60)

    for resp in responses:
        print(f"\n--- Response to @{resp.original_post_author} ---")
        print(f"Original: {resp.original_post_content[:100]}...")
        print(f"Relevance: {resp.relevance_score:.2f}")
        print(f"Company mention: {resp.is_company_related}")
        print(f"Reasoning: {resp.reasoning}")
        print(f"\nResponse:\n{resp.response_text}")
        print("-" * 40)

        reply_url = None
        if post_replies:
            print("Posting reply...")
            result = post_reply(resp)
            reply_url = result["url"]
            print(f"Posted: {reply_url}")

        # Save response to database
        response_id = save_response(
            db,
            original_post_id=resp.original_post_id,
            original_post_content=resp.original_post_content,
            original_post_author=resp.original_post_author,
            response_text=resp.response_text,
            relevance_score=resp.relevance_score,
            is_company_related=resp.is_company_related,
            reply_url=reply_url,
            posted=post_replies,
        )
        print(f"Response saved to database with ID: {response_id}")

    print(f"\n{'=' * 60}")
    if not post_replies:
        print("DRY RUN - No replies posted. Use --post flag to post.")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Mastodon and generate responses (with DB tracking)"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post the replies (default: just print)",
    )
    args = parser.parse_args()

    main(post_replies=args.post)
