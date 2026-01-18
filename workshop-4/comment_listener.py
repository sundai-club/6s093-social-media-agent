"""
Workshop 4: Comment Listener with RAG-powered responses.

Monitors Mastodon for comments on our posts and responds using:
1. RAG retrieval of relevant context (business docs, past posts, past responses)
2. LLM generation with style guide and retrieved context
3. Optional posting of replies

WORKFLOW:
=========
1. Fetch notifications from Mastodon (mentions/replies)
2. Filter to unprocessed comments on our posts
3. For each comment:
   a. Generate embedding for the comment
   b. Retrieve relevant context via hybrid search
   c. Generate reply using LLM with context
   d. Optionally post the reply
   e. Mark comment as processed
"""

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mastodon import Mastodon
from openai import OpenAI

from database import (
    get_db,
    save_comment_reply,
    update_comment_reply_posted,
    mark_comment_processed,
    is_comment_processed,
    get_stats,
)
from embeddings import generate_embedding
from rag import retrieve_context

load_dotenv(Path(__file__).parent.parent / ".env")

# Style guide for response generation
STYLE_GUIDE = """You are Emanon's community engagement specialist.

BRAND VOICE:
- Professional but approachable
- NO HYPE - focus on practical, actionable insights
- Technical accuracy is paramount
- Concise and direct

RESPONSE GUIDELINES:
- Keep responses under 400 characters (Mastodon limit consideration)
- Be helpful and informative
- Reference specific Emanon offerings when relevant
- Don't be pushy or salesy
- Acknowledge the commenter's question/point directly
- If you don't know something, say so honestly

EMANON OFFERINGS TO MENTION (when relevant):
- Weekly AI Newsletter & Podcast (every Sunday)
- 1-hour AI Strategy Consulting sessions
- Technical Blog & Guides
- Discord Community: discord.gg/HrNXgwpVzd
"""


def get_mastodon_client() -> Mastodon:
    """Get authenticated Mastodon client."""
    return Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def fetch_comment_notifications(mastodon: Mastodon, limit: int = 40) -> list[dict]:
    """
    Fetch recent notifications that are mentions/replies.

    Returns list of notification dicts with:
    - id: notification ID
    - type: 'mention' for replies
    - status: the comment/reply status object
    - account: who wrote the comment
    """
    notifications = mastodon.notifications(limit=limit)

    # Filter to mentions (which includes replies)
    mentions = [n for n in notifications if n["type"] == "mention"]

    return mentions


def extract_comment_info(notification: dict) -> dict:
    """Extract relevant info from a notification."""
    status = notification["status"]
    account = notification["account"]

    # Get the post this is replying to (if any)
    in_reply_to_id = status.get("in_reply_to_id")

    # Strip HTML tags from content for cleaner text
    import re
    content = status["content"]
    content = re.sub(r"<[^>]+>", "", content)  # Remove HTML tags
    content = content.strip()

    return {
        "notification_id": notification["id"],
        "comment_id": str(status["id"]),
        "comment_content": content,
        "comment_author": account["acct"],
        "comment_url": status["url"],
        "parent_post_id": str(in_reply_to_id) if in_reply_to_id else None,
        "created_at": status["created_at"],
    }


def generate_reply(
    comment: dict,
    context: str,
    openai_client: OpenAI,
) -> str:
    """
    Generate a reply to a comment using the LLM with RAG context.

    PROMPT STRUCTURE:
    =================
    [System] Style guide + retrieved context
    [User] The comment to respond to

    We use the same model as the rest of the project for consistency.
    """
    system_prompt = f"""{STYLE_GUIDE}

RETRIEVED CONTEXT (relevant information from our knowledge base):
{context}

Generate a helpful, on-brand response to the following comment. Keep it under 400 characters.
"""

    user_prompt = f"""Comment from @{comment['comment_author']}:
"{comment['comment_content']}"

Generate a response:"""

    response = openai_client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def process_comment(
    comment: dict,
    mastodon: Mastodon,
    openai_client: OpenAI,
    db,
    should_post: bool = False,
) -> dict:
    """
    Process a single comment: retrieve context, generate reply, optionally post.

    Returns dict with processing results.
    """
    print(f"\n--- Processing comment from @{comment['comment_author']} ---")
    print(f"Comment: {comment['comment_content'][:100]}...")

    # Generate embedding for the comment
    print("Generating comment embedding...")
    comment_embedding = generate_embedding(comment["comment_content"], openai_client)

    # Retrieve relevant context
    print("Retrieving context via RAG...")
    context, raw_results = retrieve_context(
        query=comment["comment_content"],
        query_embedding=comment_embedding,
        top_k=10,
    )

    print(f"Retrieved {len(raw_results)} context items")

    # Generate reply
    print("Generating reply...")
    reply_text = generate_reply(
        comment=comment,
        context=context,
        openai_client=openai_client,
    )

    print(f"Generated reply ({len(reply_text)} chars): {reply_text[:100]}...")

    # Save to database
    reply_id = save_comment_reply(
        conn=db,
        parent_post_id=comment["parent_post_id"] or "unknown",
        comment_id=comment["comment_id"],
        comment_content=comment["comment_content"],
        comment_author=comment["comment_author"],
        reply_text=reply_text,
        rag_context=raw_results,
        posted=False,
    )

    # Mark as processed (even if we don't post)
    mark_comment_processed(db, comment["comment_id"])

    result = {
        "comment_id": comment["comment_id"],
        "reply_text": reply_text,
        "context_count": len(raw_results),
        "posted": False,
        "reply_url": None,
    }

    # Post reply if requested
    if should_post:
        print("Posting reply to Mastodon...")
        try:
            posted_status = mastodon.status_reply(
                to_status=mastodon.status(int(comment["comment_id"])),
                status=reply_text,
                visibility="public",
            )
            result["posted"] = True
            result["reply_url"] = posted_status["url"]

            # Update database
            update_comment_reply_posted(db, comment["comment_id"], posted_status["url"])
            print(f"Posted! URL: {posted_status['url']}")

        except Exception as e:
            print(f"Error posting reply: {e}")

    return result


def listen_for_comments(
    should_post: bool = False,
    one_shot: bool = False,
    poll_interval: int = 60,
):
    """
    Main listener loop.

    Args:
        should_post: If True, actually post replies to Mastodon
        one_shot: If True, check once and exit (no polling)
        poll_interval: Seconds between checks when polling
    """
    print("=" * 60)
    print("COMMENT LISTENER - Workshop 4")
    print("=" * 60)
    print(f"Mode: {'POST replies' if should_post else 'DRY RUN (no posting)'}")
    print(f"Polling: {'One-shot' if one_shot else f'Every {poll_interval}s'}")
    print()

    mastodon = get_mastodon_client()
    openai_client = get_openai_client()
    db = get_db()

    # Show stats
    stats = get_stats(db)
    print(f"Database stats: {stats['total_embeddings']} embeddings, "
          f"{stats['processed_comments']} processed comments")
    print()

    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] Checking for new comments...")

        # Fetch notifications
        notifications = fetch_comment_notifications(mastodon)
        print(f"Found {len(notifications)} mention notifications")

        # Filter to unprocessed comments
        new_comments = []
        for notif in notifications:
            comment = extract_comment_info(notif)

            # Skip if already processed
            if is_comment_processed(db, comment["comment_id"]):
                continue

            new_comments.append(comment)

        print(f"New unprocessed comments: {len(new_comments)}")

        # Process each new comment
        for comment in new_comments:
            try:
                result = process_comment(
                    comment=comment,
                    mastodon=mastodon,
                    openai_client=openai_client,
                    db=db,
                    should_post=should_post,
                )
                print(f"Result: {'Posted' if result['posted'] else 'Generated (not posted)'}")

            except Exception as e:
                print(f"Error processing comment {comment['comment_id']}: {e}")
                # Still mark as processed to avoid infinite retries
                mark_comment_processed(db, comment["comment_id"])

        if one_shot:
            print("\nOne-shot mode: exiting.")
            break

        print(f"\nSleeping {poll_interval}s...")
        time.sleep(poll_interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Listen for Mastodon comments and respond with RAG"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post replies (default: dry run)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (no polling loop)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Poll interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show database stats and exit",
    )

    args = parser.parse_args()

    if args.stats:
        db = get_db()
        stats = get_stats(db)
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        return

    listen_for_comments(
        should_post=args.post,
        one_shot=args.once,
        poll_interval=args.interval,
    )


if __name__ == "__main__":
    main()
