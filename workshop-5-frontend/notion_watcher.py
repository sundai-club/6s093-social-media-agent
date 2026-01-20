"""
Workshop 5: Notion Document Watcher with Auto-Post.

Monitors Notion pages under a parent page for changes and automatically:
1. Polls for new/modified/deleted child pages
2. Extracts content and computes diffs
3. Updates embeddings for changed documents
4. Generates and posts about significant changes
"""

import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from notion_client import Client as NotionClient

from database import get_db, save_post
from embeddings import get_embedding_model, save_embedding, delete_embeddings_by_source

load_dotenv(Path(__file__).parent.parent / ".env")

# Default poll interval in seconds
DEFAULT_POLL_INTERVAL = 60

# Default significance thresholds
DEFAULT_MIN_WORDS = 50
DEFAULT_MIN_LINES = 10


# ========================================
# NOTION CLIENT
# ========================================


def get_notion_client() -> NotionClient:
    """Get authenticated Notion client."""
    api_key = os.environ.get("NOTION_API_KEY")
    if not api_key:
        raise ValueError("NOTION_API_KEY environment variable not set")
    return NotionClient(auth=api_key)


def get_parent_page_id() -> str:
    """Get the parent page ID from environment."""
    page_id = os.environ.get("NOTION_PARENT_PAGE_ID")
    if not page_id:
        raise ValueError("NOTION_PARENT_PAGE_ID environment variable not set")
    return page_id


# ========================================
# NOTION CONTENT EXTRACTION
# ========================================


def extract_text_from_blocks(notion: NotionClient, block_id: str, depth: int = 0) -> str:
    """
    Recursively extract plain text from Notion blocks.

    Args:
        notion: Notion client
        block_id: Block or page ID to extract from
        depth: Current recursion depth (for indentation)

    Returns:
        Extracted text content
    """
    text_parts = []

    try:
        children = notion.blocks.children.list(block_id=block_id)
    except Exception as e:
        print(f"  Warning: Could not fetch blocks for {block_id}: {e}")
        return ""

    for block in children.get("results", []):
        block_type = block.get("type")
        block_id = block.get("id")

        # Extract text based on block type
        if block_type == "paragraph":
            text = _get_rich_text(block.get("paragraph", {}).get("rich_text", []))
            if text:
                text_parts.append(text)

        elif block_type in ["heading_1", "heading_2", "heading_3"]:
            text = _get_rich_text(block.get(block_type, {}).get("rich_text", []))
            prefix = "#" * int(block_type[-1])
            if text:
                text_parts.append(f"{prefix} {text}")

        elif block_type == "bulleted_list_item":
            text = _get_rich_text(block.get("bulleted_list_item", {}).get("rich_text", []))
            if text:
                text_parts.append(f"• {text}")

        elif block_type == "numbered_list_item":
            text = _get_rich_text(block.get("numbered_list_item", {}).get("rich_text", []))
            if text:
                text_parts.append(f"1. {text}")

        elif block_type == "to_do":
            text = _get_rich_text(block.get("to_do", {}).get("rich_text", []))
            checked = block.get("to_do", {}).get("checked", False)
            checkbox = "[x]" if checked else "[ ]"
            if text:
                text_parts.append(f"{checkbox} {text}")

        elif block_type == "toggle":
            text = _get_rich_text(block.get("toggle", {}).get("rich_text", []))
            if text:
                text_parts.append(f"▸ {text}")

        elif block_type == "code":
            text = _get_rich_text(block.get("code", {}).get("rich_text", []))
            language = block.get("code", {}).get("language", "")
            if text:
                text_parts.append(f"```{language}\n{text}\n```")

        elif block_type == "quote":
            text = _get_rich_text(block.get("quote", {}).get("rich_text", []))
            if text:
                text_parts.append(f"> {text}")

        elif block_type == "callout":
            text = _get_rich_text(block.get("callout", {}).get("rich_text", []))
            icon = block.get("callout", {}).get("icon", {})
            emoji = icon.get("emoji", "💡") if icon.get("type") == "emoji" else "💡"
            if text:
                text_parts.append(f"{emoji} {text}")

        elif block_type == "divider":
            text_parts.append("---")

        elif block_type == "table_of_contents":
            text_parts.append("[Table of Contents]")

        elif block_type == "child_page":
            # Don't recurse into child pages (they're separate documents)
            title = block.get("child_page", {}).get("title", "Untitled")
            text_parts.append(f"📄 {title}")

        elif block_type == "child_database":
            title = block.get("child_database", {}).get("title", "Untitled Database")
            text_parts.append(f"📊 {title}")

        # Recursively get children if block has them
        if block.get("has_children", False) and block_type not in ["child_page", "child_database"]:
            child_text = extract_text_from_blocks(notion, block_id, depth + 1)
            if child_text:
                # Indent child content
                indented = "\n".join("  " + line for line in child_text.split("\n"))
                text_parts.append(indented)

    return "\n".join(text_parts)


def _get_rich_text(rich_text_array: list) -> str:
    """Extract plain text from Notion rich_text array."""
    return "".join(item.get("plain_text", "") for item in rich_text_array)


def get_page_content(notion: NotionClient, page_id: str) -> dict:
    """
    Get full content and metadata for a Notion page.

    Returns:
        dict with:
        - id: page ID
        - title: page title
        - content: extracted text content
        - last_edited: last edited timestamp
        - url: Notion page URL
    """
    # Get page metadata
    page = notion.pages.retrieve(page_id=page_id)

    # Extract title from properties
    title = "Untitled"
    props = page.get("properties", {})
    if "title" in props:
        title_array = props["title"].get("title", [])
        title = _get_rich_text(title_array)
    elif "Name" in props:
        name_array = props["Name"].get("title", [])
        title = _get_rich_text(name_array)

    # Extract content from blocks
    content = extract_text_from_blocks(notion, page_id)

    return {
        "id": page_id,
        "title": title,
        "content": content,
        "last_edited": page.get("last_edited_time"),
        "url": page.get("url"),
    }


def get_child_pages(notion: NotionClient, parent_id: str) -> list[dict]:
    """
    Get all child pages under a parent page.

    Returns:
        List of dicts with id, title for each child page
    """
    pages = []

    try:
        children = notion.blocks.children.list(block_id=parent_id)

        for block in children.get("results", []):
            if block["type"] == "child_page":
                pages.append({
                    "id": block["id"],
                    "title": block["child_page"]["title"],
                })
    except Exception as e:
        print(f"Error fetching child pages: {e}")

    return pages


# ========================================
# STATE TRACKING
# ========================================


def get_content_hash(content: str) -> str:
    """Compute MD5 hash of content."""
    return hashlib.md5(content.encode()).hexdigest()


def get_current_notion_state(notion: NotionClient, parent_id: str) -> dict[str, dict]:
    """
    Get current state of all Notion pages under parent.

    Returns dict mapping page_id to:
    - title: page title
    - hash: MD5 hash of content
    - last_edited: Notion last_edited timestamp
    - content: full text content
    """
    state = {}

    child_pages = get_child_pages(notion, parent_id)

    for page_info in child_pages:
        page_id = page_info["id"]
        print(f"  Fetching: {page_info['title']}...")

        try:
            page_data = get_page_content(notion, page_id)
            state[page_id] = {
                "title": page_data["title"],
                "hash": get_content_hash(page_data["content"]),
                "last_edited": page_data["last_edited"],
                "content": page_data["content"],
                "url": page_data["url"],
            }
        except Exception as e:
            print(f"    Warning: Could not fetch page {page_id}: {e}")

    return state


def load_notion_state(db) -> dict[str, dict]:
    """Load previously saved Notion document state from database."""
    cursor = db.cursor()

    # Ensure the table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS notion_doc_state (
            page_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            hash TEXT NOT NULL,
            last_edited TEXT,
            content TEXT NOT NULL,
            url TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()

    cursor.execute("SELECT page_id, title, hash, last_edited, content, url FROM notion_doc_state")
    rows = cursor.fetchall()

    return {
        row[0]: {
            "title": row[1],
            "hash": row[2],
            "last_edited": row[3],
            "content": row[4],
            "url": row[5],
        }
        for row in rows
    }


def save_notion_state(db, state: dict[str, dict]) -> None:
    """Save current Notion document state to database."""
    cursor = db.cursor()

    # Clear old state and insert new
    cursor.execute("DELETE FROM notion_doc_state")

    for page_id, info in state.items():
        cursor.execute(
            """
            INSERT INTO notion_doc_state (page_id, title, hash, last_edited, content, url, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                info["title"],
                info["hash"],
                info["last_edited"],
                info["content"],
                info.get("url"),
                datetime.now().isoformat(),
            ),
        )

    db.commit()


# ========================================
# CHANGE DETECTION
# ========================================


def detect_notion_changes(old_state: dict, new_state: dict) -> dict:
    """
    Compare old and new states to detect changes.

    Returns dict with:
    - added: list of new page IDs
    - modified: list of modified page IDs
    - deleted: list of deleted page IDs
    """
    old_ids = set(old_state.keys())
    new_ids = set(new_state.keys())

    added = list(new_ids - old_ids)
    deleted = list(old_ids - new_ids)

    # Check for modifications (pages in both, but different hash)
    modified = []
    for page_id in old_ids & new_ids:
        if old_state[page_id]["hash"] != new_state[page_id]["hash"]:
            modified.append(page_id)

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
    }


def compute_change_magnitude(old_content: str, new_content: str) -> dict:
    """
    Compute the magnitude of changes between old and new content.
    """
    old_words = old_content.split()
    new_words = new_content.split()

    words_added = max(0, len(new_words) - len(old_words))
    words_removed = max(0, len(old_words) - len(new_words))
    words_changed = abs(len(new_words) - len(old_words))

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    old_line_set = set(old_lines)
    new_line_set = set(new_lines)

    lines_added = len(new_line_set - old_line_set)
    lines_removed = len(old_line_set - new_line_set)
    lines_changed = lines_added + lines_removed

    return {
        "words_added": words_added,
        "words_removed": words_removed,
        "words_changed": words_changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "lines_changed": lines_changed,
    }


def is_change_significant(
    magnitude: dict,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
) -> bool:
    """Check if change magnitude exceeds significance thresholds."""
    return magnitude["words_changed"] >= min_words or magnitude["lines_changed"] >= min_lines


def get_change_summary(old_state: dict, new_state: dict, changes: dict) -> str:
    """Create a detailed summary of all changes for LLM context."""
    parts = []

    if changes["added"]:
        parts.append("## New Documents Added")
        for page_id in changes["added"]:
            info = new_state[page_id]
            content = info["content"]
            preview = content[:500] + "..." if len(content) > 500 else content
            parts.append(f"\n### {info['title']}\n{preview}")

    if changes["modified"]:
        parts.append("\n## Documents Modified")
        for page_id in changes["modified"]:
            old_info = old_state[page_id]
            new_info = new_state[page_id]
            title = new_info["title"]

            # Show what changed (simplified diff summary)
            old_len = len(old_info["content"])
            new_len = len(new_info["content"])
            diff_chars = new_len - old_len

            parts.append(f"\n### {title}")
            parts.append(f"Content changed: {'+' if diff_chars >= 0 else ''}{diff_chars} characters")

            # Show new content preview
            preview = new_info["content"][:500] + "..." if len(new_info["content"]) > 500 else new_info["content"]
            parts.append(f"\nUpdated content:\n{preview}")

    if changes["deleted"]:
        parts.append("\n## Documents Deleted")
        for page_id in changes["deleted"]:
            title = old_state[page_id]["title"]
            parts.append(f"- {title}")

    return "\n".join(parts)


# ========================================
# EMBEDDINGS UPDATE
# ========================================


def update_notion_embeddings(db, new_state: dict, changes: dict) -> int:
    """
    Update embeddings for changed Notion pages.

    Returns number of embeddings updated.
    """
    model = get_embedding_model()
    updated = 0

    # Delete embeddings for deleted pages
    for page_id in changes["deleted"]:
        count = delete_embeddings_by_source(db, "notion_doc", page_id)
        if count > 0:
            print(f"  Deleted {count} embeddings for removed page")
            updated += count

    # Update embeddings for added/modified pages
    pages_to_embed = changes["added"] + changes["modified"]

    for page_id in pages_to_embed:
        info = new_state[page_id]
        title = info["title"]
        content = info["content"]

        if not content.strip():
            print(f"  Skipping {title}: no content")
            continue

        # Delete old embeddings for this page
        delete_embeddings_by_source(db, "notion_doc", page_id)

        # Chunk the content (simple chunking by paragraphs)
        chunks = chunk_content(content, title, max_chars=1000)

        print(f"  Embedding {title}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Generate embedding
            embeddings = list(model.embed([chunk["text"]]))
            embedding = embeddings[0].tolist()

            # Save to database
            save_embedding(
                db,
                source_type="notion_doc",
                content=chunk["text"],
                embedding=embedding,
                source_id=page_id,
                metadata={
                    "title": title,
                    "chunk_index": i,
                    "url": info.get("url"),
                },
            )
            updated += 1

    return updated


def chunk_content(content: str, title: str, max_chars: int = 1000) -> list[dict]:
    """
    Split content into chunks for embedding.

    Each chunk includes the document title for context.
    """
    chunks = []

    # Split by double newlines (paragraphs)
    paragraphs = content.split("\n\n")

    current_chunk = f"# {title}\n\n"

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph exceeds max, save current chunk and start new
        if len(current_chunk) + len(para) > max_chars and len(current_chunk) > len(title) + 10:
            chunks.append({"text": current_chunk.strip()})
            current_chunk = f"# {title}\n\n"

        current_chunk += para + "\n\n"

    # Don't forget the last chunk
    if len(current_chunk.strip()) > len(title) + 10:
        chunks.append({"text": current_chunk.strip()})

    # If no chunks were created, create one with whatever we have
    if not chunks and content.strip():
        chunks.append({"text": f"# {title}\n\n{content[:max_chars]}"})

    return chunks


# ========================================
# IMPORT FROM DATABASE MODULE
# ========================================

def save_embedding(db, source_type: str, content: str, embedding: list, source_id: str = None, metadata: dict = None) -> int:
    """Save embedding to database (wrapper for database.save_embedding)."""
    from database import save_embedding as db_save_embedding
    return db_save_embedding(db, source_type, content, embedding, source_id, metadata)


def delete_embeddings_by_source(db, source_type: str, source_id: str = None) -> int:
    """Delete embeddings by source (wrapper for database.delete_embeddings_by_source)."""
    from database import delete_embeddings_by_source as db_delete
    return db_delete(db, source_type, source_id)


def get_embedding_model():
    """Get fastembed model for embedding generation."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ========================================
# POST GENERATION
# ========================================


def get_openai_client():
    """Get OpenAI client configured for OpenRouter."""
    from openai import OpenAI
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_change_post(change_summary: str, changes: dict) -> str:
    """Generate a social media post about the changes."""
    client = get_openai_client()

    # Get current date for context
    current_date = datetime.now().strftime("%B %d, %Y")

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": f"""You are a social media manager for Emanon, an AI news and consulting platform.

IMPORTANT: Today's date is {current_date}. Use this to determine what is current vs old news:
- If content discusses events from previous years (e.g., "2025 in review" when it's 2026), frame it as a retrospective/reflection, not breaking news
- If content discusses recent developments, present it as current/timely

Create engaging Mastodon posts that:
- Are concise (under 400 characters)
- Highlight Emanon's practical approach to AI
- Include 1-2 relevant hashtags
- Sound authentic, not salesy
- Share ONE interesting insight from the content
- Are temporally accurate (don't present old news as new)

Write in a conversational but professional tone.""",
            },
            {
                "role": "user",
                "content": f"""Based on the following documentation updates, create a single engaging Mastodon post.

{change_summary[:2000]}

Generate just the post text, nothing else.""",
            },
        ],
    )

    message = response.choices[0].message
    content = message.content

    if not content and hasattr(message, 'reasoning') and message.reasoning:
        import re
        reasoning = message.reasoning
        quoted = re.findall(r'["\u201c]([^"\u201d]{50,400})["\u201d]', reasoning)
        if quoted:
            content = max(quoted, key=len)
        else:
            paragraphs = [p.strip() for p in reasoning.split('\n\n') if len(p.strip()) > 50]
            content = paragraphs[0][:400] if paragraphs else reasoning[:400]

    return (content or "Unable to generate post.").strip()


# ========================================
# MAIN WORKFLOW
# ========================================


def check_notion_changes(
    db,
    should_post: bool = False,
    approve: bool = False,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
) -> dict:
    """
    Main function to check Notion for changes and process them.

    Args:
        db: Database connection
        should_post: If True, post to Mastodon after approval
        approve: If True, send to Telegram for approval
        min_words: Minimum word changes to trigger post
        min_lines: Minimum line changes to trigger post

    Returns dict with:
    - has_changes: bool
    - changes: dict of changes detected
    - post_content: generated post (if changes found)
    - skipped_pages: list of pages skipped due to insignificant changes
    """
    result = {
        "has_changes": False,
        "changes": {"added": [], "modified": [], "deleted": []},
        "post_content": None,
        "skipped_pages": [],
    }

    print("Initializing Notion client...")
    try:
        notion = get_notion_client()
        parent_id = get_parent_page_id()
    except ValueError as e:
        print(f"Error: {e}")
        return result

    print("Loading saved Notion state...")
    old_state = load_notion_state(db)

    print("Fetching current Notion pages...")
    new_state = get_current_notion_state(notion, parent_id)

    print("Detecting changes...")
    changes = detect_notion_changes(old_state, new_state)

    # Check significance of modified pages
    significant_modified = []
    for page_id in changes["modified"]:
        magnitude = compute_change_magnitude(
            old_state[page_id]["content"],
            new_state[page_id]["content"],
        )
        title = new_state[page_id]["title"]

        print(f"\n📊 {title}: +{magnitude['words_added']} words, -{magnitude['words_removed']} words, "
              f"+{magnitude['lines_added']} lines, -{magnitude['lines_removed']} lines")

        if is_change_significant(magnitude, min_words, min_lines):
            print(f"   ✅ Significant change")
            significant_modified.append(page_id)
        else:
            print(f"   ⏭️  Below threshold, skipping")
            result["skipped_pages"].append(title)

    changes["modified"] = significant_modified

    has_changes = bool(changes["added"] or changes["modified"] or changes["deleted"])
    result["has_changes"] = has_changes
    result["changes"] = changes

    if not has_changes:
        if result["skipped_pages"]:
            print(f"\nNo significant changes. Skipped {len(result['skipped_pages'])} minor edit(s).")
            save_notion_state(db, new_state)
        else:
            print("No changes detected.")
        return result

    # Print change summary
    print("\n" + "=" * 60)
    print("SIGNIFICANT CHANGES DETECTED")
    print("=" * 60)

    if changes["added"]:
        titles = [new_state[pid]["title"] for pid in changes["added"]]
        print(f"\n📄 New pages: {', '.join(titles)}")
    if changes["modified"]:
        titles = [new_state[pid]["title"] for pid in changes["modified"]]
        print(f"\n✏️  Modified: {', '.join(titles)}")
    if changes["deleted"]:
        titles = [old_state[pid]["title"] for pid in changes["deleted"]]
        print(f"\n🗑️  Deleted: {', '.join(titles)}")

    # Get detailed change summary
    change_summary = get_change_summary(old_state, new_state, changes)

    # Update embeddings
    print("\n--- Updating Embeddings ---")
    updated = update_notion_embeddings(db, new_state, changes)
    print(f"Updated {updated} embeddings")

    # Generate post
    print("\n--- Generating Post ---")
    post_content = generate_change_post(change_summary, changes)
    result["post_content"] = post_content

    print("\n" + "=" * 60)
    print("GENERATED POST")
    print("=" * 60)
    print(post_content)
    print("=" * 60)

    # Save to database
    post_id = save_post(db, post_content, posted=False)
    print(f"\nSaved post to database (ID: {post_id})")

    # Telegram approval workflow (if enabled)
    if approve:
        from doc_watcher import wait_for_decision, post_to_mastodon_with_image

        print("\n📱 Sending to Telegram for approval...")
        decision = wait_for_decision(post_content, None)

        if decision == "approve":
            print("✅ Human approved the post!")
            if should_post:
                print("\nPosting to Mastodon...")
                try:
                    from mastodon import Mastodon
                    mastodon = Mastodon(
                        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
                        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
                    )
                    status = mastodon.status_post(post_content)
                    result["post_url"] = status["url"]
                    print(f"Posted successfully! URL: {status['url']}")

                    cursor = db.cursor()
                    cursor.execute(
                        "UPDATE posts SET posted = TRUE, post_url = ? WHERE id = ?",
                        (status["url"], post_id),
                    )
                    db.commit()
                except Exception as e:
                    print(f"Error posting to Mastodon: {e}")
            else:
                print("DRY RUN - Approved but not posted.")
        else:
            print("❌ Human rejected. Post not published.")
    else:
        print("\nDRY RUN - Use --approve flag for Telegram approval.")

    # Save new state
    print("\nSaving Notion state...")
    save_notion_state(db, new_state)

    return result


def watch_notion(
    db,
    should_post: bool = False,
    approve: bool = False,
    interval: int = DEFAULT_POLL_INTERVAL,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
):
    """
    Continuously poll Notion for changes.
    """
    print("=" * 60)
    print("NOTION WATCHER - Polling Mode")
    print("=" * 60)
    print(f"📡 Polling interval: {interval}s")
    print(f"📮 Post mode: {'AUTO-POST' if should_post else 'DRY RUN'}")
    print(f"📱 Approval: {'TELEGRAM' if approve else 'DISABLED'}")
    print(f"📏 Significance: {min_words} words or {min_lines} lines")
    print("=" * 60)

    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Checking Notion for changes...")

            result = check_notion_changes(
                db,
                should_post=should_post,
                approve=approve,
                min_words=min_words,
                min_lines=min_lines,
            )

            if result["has_changes"]:
                print(f"[{timestamp}] Changes processed!")
            else:
                print(f"[{timestamp}] No changes.")

            print(f"Next check in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Stopping Notion watcher...")
            break
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying in {interval}s...")
            time.sleep(interval)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch Notion pages for changes and auto-post updates"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to Mastodon after approval",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Send to Telegram for approval before posting",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously poll Notion for changes",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {DEFAULT_POLL_INTERVAL})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset saved state (treat all pages as new)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current state and exit",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=DEFAULT_MIN_WORDS,
        help=f"Minimum word difference to trigger post (default: {DEFAULT_MIN_WORDS})",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=DEFAULT_MIN_LINES,
        help=f"Minimum line difference to trigger post (default: {DEFAULT_MIN_LINES})",
    )

    args = parser.parse_args()

    db = get_db()

    if args.reset:
        print("Resetting Notion document state...")
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS notion_doc_state")
        db.commit()
        print("State reset. Next run will detect all pages as new.")
        return

    if args.status:
        print("Current Notion document state:")
        old_state = load_notion_state(db)

        print(f"\nTracked pages: {len(old_state)}")
        for page_id, info in old_state.items():
            print(f"  - {info['title']} (last edited: {info['last_edited']})")

        print("\nFetching current Notion state...")
        try:
            notion = get_notion_client()
            parent_id = get_parent_page_id()
            new_state = get_current_notion_state(notion, parent_id)

            print(f"\nCurrent pages: {len(new_state)}")
            for page_id, info in new_state.items():
                status = "tracked" if page_id in old_state else "NEW"
                if page_id in old_state and old_state[page_id]["hash"] != info["hash"]:
                    status = "MODIFIED"
                print(f"  - {info['title']} [{status}]")
        except Exception as e:
            print(f"Could not fetch current state: {e}")

        return

    if args.watch:
        watch_notion(
            db,
            should_post=args.post,
            approve=args.approve,
            interval=args.interval,
            min_words=args.min_words,
            min_lines=args.min_lines,
        )
    else:
        check_notion_changes(
            db,
            should_post=args.post,
            approve=args.approve,
            min_words=args.min_words,
            min_lines=args.min_lines,
        )


if __name__ == "__main__":
    main()
