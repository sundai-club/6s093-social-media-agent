"""
Workshop 3: Post Generator with Human-in-the-Loop and Database Tracking

Combines:
- Workshop 1: Post generation + Mastodon posting
- Workshop 2: Telegram approval workflow
- Workshop 3: SQLite database tracking

Full flow:
1. Generate post using LLM
2. Send to Telegram for human approval
3. If approved, post to Mastodon
4. Save everything to database
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

# Add workshop-1 to path so we can import from it
sys.path.insert(0, str(Path(__file__).parent.parent / "workshop-1"))
from post_generator import read_business_docs, generate_post, post_to_mastodon

from database import get_db, save_post

load_dotenv(Path(__file__).parent.parent / ".env")


# Store state for the approval flow
pending_post = None
decision_made = asyncio.Event()
decision_result = None


def wait_for_decision(post_content: str) -> str:
    """
    Send post for approval and wait for human decision.
    Returns 'approve' or 'reject'.
    """
    global pending_post, decision_result

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        global decision_result
        query = update.callback_query
        await query.answer()
        decision_result = query.data
        if decision_result == "approve":
            await query.edit_message_text(f"✅ APPROVED\n\n{pending_post}")
        else:
            await query.edit_message_text(f"❌ REJECTED\n\n{pending_post[:100]}...")
        decision_made.set()

    async def _run():
        global pending_post
        pending_post = post_content
        decision_made.clear()

        bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ])
        await bot.send_message(
            chat_id=int(os.environ["TELEGRAM_CHAT_ID"]),
            text=f"📝 New Post for Approval\n\n{post_content}\n\n"
                 f"Characters: {len(post_content)}",
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

        return decision_result

    return asyncio.run(_run())


def main(approve: bool = False, post: bool = False):
    # Step 1: Generate post using Workshop 1 code
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
    posted = False

    if approve:
        # Step 2: Send to Telegram and wait for human decision
        decision = wait_for_decision(post_content)

        # Step 3: Act on decision
        if decision == "approve":
            print("✅ Human approved the post!")
            if post:
                print("Posting to Mastodon...")
                result = post_to_mastodon(post_content)
                post_url = result["url"]
                posted = True
                print(f"Published: {post_url}")
            else:
                print("DRY RUN - Approved but not posted. Use --post flag to publish.")
        else:
            print("❌ Human rejected. Post not published.")
    else:
        print("DRY RUN - Use --approve flag to enable Telegram approval flow")
        if post:
            print("Note: --post flag ignored without --approve")

    # Step 4: Save to database
    post_id = save_post(
        db,
        content=post_content,
        post_url=post_url,
        posted=posted,
    )
    print(f"Post saved to database with ID: {post_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post Generator with Human-in-the-Loop and Database Tracking"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Enable Telegram approval flow (default: just generate and print)",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to Mastodon after approval (requires --approve)",
    )
    args = parser.parse_args()

    main(approve=args.approve, post=args.post)
