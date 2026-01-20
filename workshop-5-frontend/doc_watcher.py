"""
Workshop 5 Frontend: Documentation Watcher with Auto-Post.
(Copied from workshop-4 with path adjustments)

Monitors business_docs folder for changes and automatically:
1. Detects file modifications/additions/deletions
2. Computes diffs to understand what changed
3. Updates embeddings for changed documents
4. Generates and posts about significant changes
"""

import asyncio
import difflib
import hashlib
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import replicate
import requests
from dotenv import load_dotenv
from mastodon import Mastodon
from openai import OpenAI
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

from database import get_db, save_post
from embeddings import embed_business_docs

load_dotenv(Path(__file__).parent.parent / ".env")

# Business docs directory - local to workshop-5-frontend
BUSINESS_DOCS_DIR = Path(__file__).parent.parent / "business-docs"

# Debounce delay in seconds - wait for rapid changes to settle
DEBOUNCE_DELAY = 2.0


# ========================================
# IMAGE GENERATION (from workshop-2)
# ========================================


def generate_image(prompt: str) -> str:
    """Generate an image using Andrew's fine-tuned model with trigger word 'annddrreeww'."""
    output = replicate.run(
        "sundai-club/andrews_model:f5211077a830f0b1cb51e541d4f591fae107a7617ce6cc54fd23c205cae0c1b5",
        input={
            "prompt": prompt,
        },
    )
    return str(output[0])


def download_image(image_url: str, save_path: str) -> str:
    """Download an image from a URL and save it locally."""
    response = requests.get(image_url)
    response.raise_for_status()
    with open(save_path, "wb") as file:
        file.write(response.content)
    return save_path


def post_to_mastodon_with_image(content: str, image_url: str = None) -> dict:
    """Post content and optional image to Mastodon."""
    mastodon = Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )

    if image_url is None:
        return mastodon.status_post(content)

    # Download the image locally
    local_image_path = "temp_image.webp"
    download_image(image_url, local_image_path)

    # Upload the image to Mastodon
    media = mastodon.media_post(local_image_path)

    # Post the content with the uploaded image
    return mastodon.status_post(content, media_ids=[media])


# ========================================
# TELEGRAM APPROVAL (from workshop-3)
# ========================================


def wait_for_decision(post_content: str, image_url: str = None) -> str:
    """
    Send post and optional image for approval and wait for human decision.
    Returns 'approve' or 'reject'.
    """

    async def _run():
        # Create event inside async context to avoid event loop binding issues
        decision_made = asyncio.Event()
        decision_result_holder = {"result": None}
        pending = post_content

        async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
            query = update.callback_query
            await query.answer()
            decision_result_holder["result"] = query.data
            if decision_result_holder["result"] == "approve":
                try:
                    await query.edit_message_caption(caption=f"✅ APPROVED\n\n{pending}")
                except Exception:
                    await query.edit_message_text(text=f"✅ APPROVED\n\n{pending}")
            else:
                try:
                    await query.edit_message_caption(caption=f"❌ REJECTED\n\n{pending[:100]}...")
                except Exception:
                    await query.edit_message_text(text=f"❌ REJECTED\n\n{pending[:100]}...")
            decision_made.set()

        bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ])

        chat_id = int(os.environ["TELEGRAM_CHAT_ID"])
        caption = f"📝 New Post for Approval\n\n{post_content}\n\nCharacters: {len(post_content)}"

        if image_url:
            # Download image and send with photo
            local_path = "temp_approval_image.webp"
            download_image(image_url, local_path)
            with open(local_path, "rb") as photo:
                await bot.send_photo(
                    chat_id=chat_id,
                    photo=photo,
                    caption=caption,
                    reply_markup=keyboard,
                )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=caption,
                reply_markup=keyboard,
            )
        print("📱 Sent to Telegram. Waiting for approval...")

        app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
        app.add_handler(CallbackQueryHandler(handle_callback))

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        await decision_made.wait()

        await app.updater.stop()
        await app.stop()
        await app.shutdown()

        return decision_result_holder["result"]

    return asyncio.run(_run())


# ========================================
# FILE STATE TRACKING
# ========================================


def get_file_hash(file_path: Path) -> str:
    """Compute MD5 hash of a file's content."""
    content = file_path.read_text()
    return hashlib.md5(content.encode()).hexdigest()


def get_current_doc_state() -> dict[str, dict]:
    """
    Get current state of all business docs.

    Returns dict mapping filename to:
    - hash: MD5 hash of content
    - mtime: modification time
    - content: file content
    """
    state = {}
    if not BUSINESS_DOCS_DIR.exists():
        return state
    for doc_path in sorted(BUSINESS_DOCS_DIR.glob("*.md")):
        state[doc_path.name] = {
            "hash": get_file_hash(doc_path),
            "mtime": doc_path.stat().st_mtime,
            "content": doc_path.read_text(),
        }
    return state


def load_saved_state(db) -> dict[str, dict]:
    """Load previously saved document state from database."""
    cursor = db.cursor()

    # Ensure the table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS doc_state (
            filename TEXT PRIMARY KEY,
            hash TEXT NOT NULL,
            mtime REAL NOT NULL,
            content TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()

    cursor.execute("SELECT filename, hash, mtime, content FROM doc_state")
    rows = cursor.fetchall()

    return {
        row[0]: {
            "hash": row[1],
            "mtime": row[2],
            "content": row[3],
        }
        for row in rows
    }


def save_doc_state(db, state: dict[str, dict]) -> None:
    """Save current document state to database."""
    cursor = db.cursor()

    # Clear old state and insert new
    cursor.execute("DELETE FROM doc_state")

    for filename, info in state.items():
        cursor.execute(
            """
            INSERT INTO doc_state (filename, hash, mtime, content, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (filename, info["hash"], info["mtime"], info["content"], datetime.now().isoformat()),
        )

    db.commit()


# ========================================
# CHANGE DETECTION
# ========================================

# Default significance thresholds
DEFAULT_MIN_WORDS = 100
DEFAULT_MIN_LINES = 20


def compute_change_magnitude(old_content: str, new_content: str) -> dict:
    """
    Compute the magnitude of changes between old and new content.

    Returns:
        dict with:
        - words_added: int
        - words_removed: int
        - words_changed: int (total difference)
        - lines_added: int
        - lines_removed: int
        - lines_changed: int (total difference)
        - is_significant: bool (exceeds threshold)
    """
    # Split into words
    old_words = old_content.split()
    new_words = new_content.split()

    # Count word differences
    old_word_set = set(old_words)
    new_word_set = set(new_words)

    # Simple word count difference approach
    words_added = max(0, len(new_words) - len(old_words))
    words_removed = max(0, len(old_words) - len(new_words))
    words_changed = abs(len(new_words) - len(old_words))

    # For more accurate counting, compute actual additions/removals
    # by comparing unique words that appear/disappear
    unique_added = len(new_word_set - old_word_set)
    unique_removed = len(old_word_set - new_word_set)

    # Use the larger of the two metrics for words_changed
    words_changed = max(words_changed, unique_added + unique_removed)

    # Split into lines
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
        "is_significant": False,  # Will be set by caller with threshold
    }


def is_change_significant(
    magnitude: dict,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
) -> bool:
    """Check if change magnitude exceeds significance thresholds."""
    return magnitude["words_changed"] >= min_words or magnitude["lines_changed"] >= min_lines


def detect_changes(old_state: dict, new_state: dict) -> dict:
    """
    Compare old and new states to detect changes.

    Returns dict with:
    - added: list of new filenames
    - modified: list of modified filenames
    - deleted: list of deleted filenames
    """
    old_files = set(old_state.keys())
    new_files = set(new_state.keys())

    added = list(new_files - old_files)
    deleted = list(old_files - new_files)

    # Check for modifications (files in both, but different hash)
    modified = []
    for filename in old_files & new_files:
        if old_state[filename]["hash"] != new_state[filename]["hash"]:
            modified.append(filename)

    return {
        "added": sorted(added),
        "modified": sorted(modified),
        "deleted": sorted(deleted),
    }


def compute_diff(old_content: str, new_content: str, filename: str) -> str:
    """
    Compute a human-readable diff between old and new content.

    Returns a summary of what changed.
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"old/{filename}",
        tofile=f"new/{filename}",
        lineterm="",
    ))

    if not diff:
        return "No textual differences detected."

    # Parse diff to extract meaningful changes
    added_lines = []
    removed_lines = []

    for line in diff:
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed_lines.append(line[1:].strip())

    # Filter out empty lines
    added_lines = [l for l in added_lines if l]
    removed_lines = [l for l in removed_lines if l]

    summary_parts = []
    if added_lines:
        summary_parts.append(f"Added ({len(added_lines)} lines):\n" + "\n".join(f"  + {l[:100]}" for l in added_lines[:5]))
        if len(added_lines) > 5:
            summary_parts.append(f"  ... and {len(added_lines) - 5} more lines")

    if removed_lines:
        summary_parts.append(f"Removed ({len(removed_lines)} lines):\n" + "\n".join(f"  - {l[:100]}" for l in removed_lines[:5]))
        if len(removed_lines) > 5:
            summary_parts.append(f"  ... and {len(removed_lines) - 5} more lines")

    return "\n".join(summary_parts) if summary_parts else "Minor formatting changes."


def get_change_summary(old_state: dict, new_state: dict, changes: dict) -> str:
    """
    Create a detailed summary of all changes for LLM context.
    """
    parts = []

    if changes["added"]:
        parts.append("## New Documents Added")
        for filename in changes["added"]:
            content = new_state[filename]["content"]
            # Truncate for summary
            preview = content[:500] + "..." if len(content) > 500 else content
            parts.append(f"\n### {filename}\n{preview}")

    if changes["modified"]:
        parts.append("\n## Documents Modified")
        for filename in changes["modified"]:
            diff = compute_diff(
                old_state[filename]["content"],
                new_state[filename]["content"],
                filename,
            )
            parts.append(f"\n### {filename}\n{diff}")

    if changes["deleted"]:
        parts.append("\n## Documents Deleted")
        for filename in changes["deleted"]:
            parts.append(f"- {filename}")

    return "\n".join(parts)


# ========================================
# POST GENERATION
# ========================================


def get_openai_client() -> OpenAI:
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def generate_change_post(change_summary: str, changes: dict) -> str:
    """
    Generate a social media post based on content (adapted from workshop-2).
    """
    client = get_openai_client()

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": """You are a social media manager for Emanon, an AI news and consulting platform.
Your task is to create engaging Mastodon posts that:
- Are concise (under 400 characters)
- Highlight Emanon's no-hype, practical approach to AI
- Include 1-2 relevant hashtags
- Sound authentic, not salesy
- Share ONE interesting insight from the content

Write in a conversational but professional tone.""",
            },
            {
                "role": "user",
                "content": f"""Based on the following content, create a single engaging Mastodon post
that shares a valuable AI insight or tip with our followers.

{change_summary[:2000]}

Generate just the post text, nothing else.""",
            },
        ],
    )

    # Handle reasoning models that return content in the reasoning field
    message = response.choices[0].message
    content = message.content

    # If content is empty, try to extract from reasoning field (used by Nemotron)
    if not content and hasattr(message, 'reasoning') and message.reasoning:
        import re
        reasoning = message.reasoning

        # Try to find content in quotes (the actual post is usually quoted)
        # Look for longer quoted strings that look like posts
        quoted = re.findall(r'["\u201c]([^"\u201d]{50,400})["\u201d]', reasoning)
        if quoted:
            # Pick the longest one that looks like a complete post
            content = max(quoted, key=len)
        else:
            # Look for lines starting with emoji (common post format)
            emoji_lines = re.findall(r'([\U0001F300-\U0001F9FF].*?)(?:\n|$)', reasoning)
            if emoji_lines:
                content = emoji_lines[0][:400]
            else:
                # Fallback: extract first substantial paragraph
                paragraphs = [p.strip() for p in reasoning.split('\n\n') if len(p.strip()) > 50]
                content = paragraphs[0][:400] if paragraphs else reasoning[:400]

    return (content or "Unable to generate post.").strip()


# ========================================
# MAIN WORKFLOW
# ========================================


def check_and_process_changes(
    db,
    should_post: bool = False,
    approve: bool = False,
    force_refresh: bool = False,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
) -> dict:
    """
    Main function to check for changes and process them.

    Args:
        db: Database connection
        should_post: If True, post to Mastodon after approval
        approve: If True, send to Telegram for approval before posting
        force_refresh: Force refresh embeddings
        min_words: Minimum word changes to trigger post
        min_lines: Minimum line changes to trigger post

    Returns dict with:
    - has_changes: bool
    - changes: dict of changes detected
    - post_content: generated post (if changes found)
    - image_url: generated image URL
    - post_url: URL if posted
    - skipped_files: list of files skipped due to insignificant changes
    """
    print("Loading saved document state...")
    old_state = load_saved_state(db)

    print("Scanning current documents...")
    new_state = get_current_doc_state()

    print("Detecting changes...")
    changes = detect_changes(old_state, new_state)

    result = {
        "has_changes": False,
        "changes": changes,
        "post_content": None,
        "image_url": None,
        "post_url": None,
        "skipped_files": [],
    }

    # Check significance of modified files and filter out minor changes
    significant_modified = []
    for filename in changes["modified"]:
        magnitude = compute_change_magnitude(
            old_state[filename]["content"],
            new_state[filename]["content"],
        )
        is_significant = is_change_significant(magnitude, min_words, min_lines)

        # Print magnitude info
        print(f"\n📊 {filename}: +{magnitude['words_added']} words, -{magnitude['words_removed']} words, "
              f"+{magnitude['lines_added']} lines, -{magnitude['lines_removed']} lines")

        if is_significant:
            print(f"   ✅ Significant change ({magnitude['words_changed']} words changed)")
            significant_modified.append(filename)
        else:
            print(f"   ⏭️  Below threshold ({magnitude['words_changed']} words < {min_words}, "
                  f"{magnitude['lines_changed']} lines < {min_lines}), skipping post")
            result["skipped_files"].append(filename)

    # Update the changes dict with only significant modifications
    changes["modified"] = significant_modified

    has_changes = bool(changes["added"] or changes["modified"] or changes["deleted"])
    result["has_changes"] = has_changes
    result["changes"] = changes

    if not has_changes:
        if result["skipped_files"]:
            print(f"\nNo significant changes detected. Skipped {len(result['skipped_files'])} minor edit(s).")
            # Still save the new state so we don't re-check these files
            print("Saving document state...")
            save_doc_state(db, new_state)
        else:
            print("No changes detected.")
        return result

    # Print change summary
    print("\n" + "=" * 60)
    print("SIGNIFICANT CHANGES DETECTED")
    print("=" * 60)

    if changes["added"]:
        print(f"\n📄 New files: {', '.join(changes['added'])}")
    if changes["modified"]:
        print(f"\n✏️  Modified: {', '.join(changes['modified'])}")
    if changes["deleted"]:
        print(f"\n🗑️  Deleted: {', '.join(changes['deleted'])}")

    # Get detailed change summary
    change_summary = get_change_summary(old_state, new_state, changes)
    print("\n--- Change Details ---")
    print(change_summary[:1000] + "..." if len(change_summary) > 1000 else change_summary)

    # Update embeddings for changed documents
    print("\n--- Updating Embeddings ---")
    embed_business_docs(force=force_refresh)

    # Generate post about changes
    print("\n--- Generating Post ---")
    post_content = generate_change_post(change_summary, changes)
    result["post_content"] = post_content

    print("\n" + "=" * 60)
    print("GENERATED POST")
    print("=" * 60)
    print(post_content)
    print("=" * 60)

    # Generate image for the post
    print("\n--- Generating Image ---")
    try:
        image_prompt = f"annddrreeww presenting about: {post_content[:200]}"
        image_url = generate_image(image_prompt)
        result["image_url"] = image_url
        print(f"Image generated: {image_url}")
    except Exception as e:
        print(f"Error generating image: {e}")
        image_url = None

    # Save to database
    post_id = save_post(db, post_content, posted=False)
    print(f"\nSaved post to database (ID: {post_id})")

    # Telegram approval workflow
    if approve:
        print("\n📱 Sending to Telegram for approval...")
        decision = wait_for_decision(post_content, image_url)

        if decision == "approve":
            print("✅ Human approved the post!")
            if should_post:
                print("\nPosting to Mastodon...")
                try:
                    status = post_to_mastodon_with_image(post_content, image_url)
                    result["post_url"] = status["url"]
                    print(f"Posted successfully! URL: {status['url']}")

                    # Update database with post URL
                    cursor = db.cursor()
                    cursor.execute(
                        "UPDATE posts SET posted = TRUE, post_url = ? WHERE id = ?",
                        (status["url"], post_id),
                    )
                    db.commit()
                except Exception as e:
                    print(f"Error posting to Mastodon: {e}")
            else:
                print("DRY RUN - Approved but not posted. Use --post flag to publish.")
        else:
            print("❌ Human rejected. Post not published.")
    else:
        print("\nDRY RUN - Use --approve flag to enable Telegram approval flow.")

    # Save new state
    print("\nSaving document state...")
    save_doc_state(db, new_state)

    return result


# ========================================
# EVENT-DRIVEN FILE WATCHER (using watchdog)
# ========================================


class DocChangeHandler:
    """
    Event handler for file system changes with debouncing.
    """

    def __init__(
        self,
        should_post: bool = False,
        approve: bool = False,
        min_words: int = DEFAULT_MIN_WORDS,
        min_lines: int = DEFAULT_MIN_LINES,
    ):
        self.should_post = should_post
        self.approve = approve
        self.min_words = min_words
        self.min_lines = min_lines
        self._pending_changes: set[str] = set()
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _schedule_processing(self):
        """Schedule change processing after debounce delay."""
        with self._lock:
            # Cancel existing timer if any
            if self._timer is not None:
                self._timer.cancel()

            # Schedule new processing
            self._timer = threading.Timer(DEBOUNCE_DELAY, self._process_changes)
            self._timer.start()

    def _process_changes(self):
        """Process accumulated changes after debounce period."""
        with self._lock:
            if not self._pending_changes:
                return

            files_changed = list(self._pending_changes)
            self._pending_changes.clear()
            self._timer = None

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] 🔔 Processing changes to: {', '.join(files_changed)}")

        try:
            # Create a new DB connection in this thread (SQLite requirement)
            db = get_db()
            result = check_and_process_changes(
                db,
                should_post=self.should_post,
                approve=self.approve,
                min_words=self.min_words,
                min_lines=self.min_lines,
            )
            db.close()

            if result["has_changes"]:
                print(f"[{timestamp}] ✅ Changes processed successfully!")
            elif result["skipped_files"]:
                print(f"[{timestamp}] ⏭️  Minor changes skipped (below threshold)")
            else:
                print(f"[{timestamp}] No changes detected.")
        except Exception as e:
            print(f"[{timestamp}] ❌ Error processing changes: {e}")

        print("\n👀 Watching for changes... (Ctrl+C to stop)")

    def on_change(self, event_type: str, src_path: str):
        """Handle a file change event."""
        path = Path(src_path)

        # Only process .md files in the business_docs directory
        if path.suffix != ".md":
            return
        if path.parent != BUSINESS_DOCS_DIR:
            return

        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 📝 {event_type}: {path.name}")

        with self._lock:
            self._pending_changes.add(path.name)

        self._schedule_processing()

    def stop(self):
        """Stop any pending timer."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None


def watch_mode_events(
    should_post: bool = False,
    approve: bool = False,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
):
    """
    Event-driven watch mode using watchdog library.
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("❌ watchdog library not installed.")
        print("   Install with: uv add watchdog")
        print("   Or use polling mode: --watch --poll")
        return

    print("=" * 60)
    print("DOC WATCHER - Event Listener Mode")
    print("=" * 60)
    print(f"📁 Monitoring: {BUSINESS_DOCS_DIR}")
    print(f"📮 Post mode: {'AUTO-POST' if should_post else 'DRY RUN'}")
    print(f"📱 Approval: {'TELEGRAM' if approve else 'DISABLED'}")
    print(f"⏱️  Debounce: {DEBOUNCE_DELAY}s")
    print(f"📏 Significance: {min_words} words or {min_lines} lines")
    print("=" * 60)

    handler = DocChangeHandler(should_post=should_post, approve=approve, min_words=min_words, min_lines=min_lines)

    # Create watchdog event handler
    class WatchdogHandler(FileSystemEventHandler):
        def on_created(self, event):
            if not event.is_directory:
                handler.on_change("created", event.src_path)

        def on_modified(self, event):
            if not event.is_directory:
                handler.on_change("modified", event.src_path)

        def on_deleted(self, event):
            if not event.is_directory:
                handler.on_change("deleted", event.src_path)

        def on_moved(self, event):
            if not event.is_directory:
                handler.on_change("moved", event.dest_path)

    # Start the observer
    observer = Observer()
    observer.schedule(WatchdogHandler(), str(BUSINESS_DOCS_DIR), recursive=False)
    observer.start()

    print("\n👀 Watching for changes... (Ctrl+C to stop)\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping watcher...")
        handler.stop()
        observer.stop()
        observer.join()
        print("Goodbye!")


def watch_mode_polling(
    db,
    should_post: bool = False,
    approve: bool = False,
    interval: int = 30,
    min_words: int = DEFAULT_MIN_WORDS,
    min_lines: int = DEFAULT_MIN_LINES,
):
    """
    Legacy polling-based watch mode.
    """
    print("=" * 60)
    print("DOC WATCHER - Polling Mode (legacy)")
    print("=" * 60)
    print(f"📁 Monitoring: {BUSINESS_DOCS_DIR}")
    print(f"📮 Post mode: {'AUTO-POST' if should_post else 'DRY RUN'}")
    print(f"📱 Approval: {'TELEGRAM' if approve else 'DISABLED'}")
    print(f"⏱️  Check interval: {interval}s")
    print(f"📏 Significance: {min_words} words or {min_lines} lines")
    print("=" * 60)
    print("\n⚠️  Consider using event mode (without --poll) for better efficiency.\n")

    while True:
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Checking for changes...")

            result = check_and_process_changes(
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
            print("\n\n🛑 Stopping watcher...")
            break


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Watch business docs for changes and auto-post updates"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to Mastodon after approval (requires --approve)",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Send to Telegram for approval before posting",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch for changes (event-driven by default)",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Use polling instead of event listener (legacy mode)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds when using --poll (default: 30)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset saved state (treat all docs as new)",
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
        print("Resetting document state...")
        cursor = db.cursor()
        cursor.execute("DELETE FROM doc_state")
        db.commit()
        print("State reset. Next run will detect all docs as new.")
        return

    if args.status:
        print("Current document state:")
        old_state = load_saved_state(db)
        new_state = get_current_doc_state()

        print(f"\nTracked documents: {len(old_state)}")
        for filename in sorted(old_state.keys()):
            print(f"  - {filename}")

        print(f"\nCurrent documents: {len(new_state)}")
        for filename in sorted(new_state.keys()):
            status = "tracked" if filename in old_state else "NEW"
            if filename in old_state and old_state[filename]["hash"] != new_state[filename]["hash"]:
                status = "MODIFIED"
            print(f"  - {filename} [{status}]")

        return

    if args.watch:
        if args.poll:
            # Legacy polling mode
            watch_mode_polling(
                db,
                should_post=args.post,
                approve=args.approve,
                interval=args.interval,
                min_words=args.min_words,
                min_lines=args.min_lines,
            )
        else:
            # Event-driven mode (default)
            watch_mode_events(
                should_post=args.post,
                approve=args.approve,
                min_words=args.min_words,
                min_lines=args.min_lines,
            )
    else:
        check_and_process_changes(
            db,
            should_post=args.post,
            approve=args.approve,
            min_words=args.min_words,
            min_lines=args.min_lines,
        )


if __name__ == "__main__":
    main()
