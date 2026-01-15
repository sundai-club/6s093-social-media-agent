"""
Workshop 2: Post Generator with Human-in-the-Loop
Integrates Telegram approval workflow with the Workshop 1 post generator.

Flow:
1. Generate post using Workshop 1's post_generator
2. Send to Telegram for human approval
3. Wait for Approve/Reject response
4. If approved, publish to Mastodon (TODO: students implement this!)
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
from post_generator import read_business_docs, generate_post

load_dotenv(Path(__file__).parent.parent / ".env")


# Store state for the approval flow
pending_post = None
decision_made = asyncio.Event()
decision_result = None


def send_for_approval(post_content: str) -> None:
    """Send a post to Telegram for human approval."""
    async def _send():
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
    asyncio.run(_send())


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


def main(approve: bool = False):
    # Step 1: Generate post using Workshop 1 code
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    print("\n--- Generated Post ---")
    print(post_content)
    print("----------------------\n")

    if not approve:
        print("DRY RUN - Use --approve flag to enable Telegram approval flow")
        return

    # Step 2: Send to Telegram and wait for human decision
    decision = wait_for_decision(post_content)

    # Step 3: Act on decision
    if decision == "approve":
        print("✅ Human approved the post!")
        # TODO: Students implement this!
        # Hint: Use post_to_mastodon from workshop-1/post_generator.py
        # from post_generator import post_to_mastodon
        # result = post_to_mastodon(post_content)
        # print(f"Published: {result['url']}")
        print("🚧 TODO: Implement publishing to Mastodon here!")
    else:
        print("❌ Human rejected. Post not published.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post Generator with Human-in-the-Loop approval"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Enable Telegram approval flow (default: just generate and print)",
    )
    args = parser.parse_args()

    main(approve=args.approve)
