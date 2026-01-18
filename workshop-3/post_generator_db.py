"""
Workshop 3: Post Generator with Database Tracking
Extends the base post generator to save posts to SQLite database.

This version can be deployed to a GCP VM and tracked via the FastAPI server.
"""

from post_generator import read_business_docs, generate_post, post_to_mastodon
from database import get_db, save_post


def main(post: bool = False):
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    print("\n--- Generated Post ---")
    print(post_content)
    print("----------------------\n")

    # Get database connection
    db = get_db()
    post_url = None

    if post:
        print("Posting to Mastodon...")
        result = post_to_mastodon(post_content)
        post_url = result["url"]
        print(f"Posted successfully! URL: {post_url}")
    else:
        print("DRY RUN - Post not published. Use --post flag to publish.")

    # Save post to database
    post_id = save_post(
        db,
        content=post_content,
        post_url=post_url,
        posted=post,
    )
    print(f"Post saved to database with ID: {post_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and post to Mastodon (with DB tracking)")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to Mastodon (default: just print)",
    )
    args = parser.parse_args()

    main(post=args.post)
