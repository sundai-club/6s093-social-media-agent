"""
Workshop 2: Feedback Loop with Rejection Reasons
Capture human feedback (rejection reasons) that could be used to improve prompts.

Flow:
1. Generate post
2. Human reviews: Approve / Reject
3. If rejected, human provides reason
4. Feedback is captured (could be used to refine the LLM prompt)

This demonstrates how HITL can go beyond simple approval to collect
actionable feedback for improving AI outputs.
"""

import os
import sys
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# Add workshop-1 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "workshop-1"))
from post_generator import read_business_docs, generate_post

load_dotenv(Path(__file__).parent.parent / ".env")


# Store state for the feedback flow
pending_post = None
decision_result = None
rejection_reason = None
waiting_for_reason = False
decision_made = asyncio.Event()


def wait_for_decision_with_feedback(post_content: str) -> tuple[str, str | None]:
    """
    Send post for approval and wait for human decision.
    If rejected, also collects the reason.
    Returns (decision, rejection_reason).
    """
    global pending_post, decision_result, rejection_reason, waiting_for_reason

    async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
        global decision_result, waiting_for_reason
        query = update.callback_query
        await query.answer()

        if query.data == "approve":
            decision_result = "approve"
            await query.edit_message_text(f"✅ APPROVED\n\n{pending_post}")
            decision_made.set()
        elif query.data == "reject":
            decision_result = "reject"
            waiting_for_reason = True
            await query.edit_message_text(
                "❌ REJECTED\n\n"
                "Please reply with the reason for rejection.\n"
                "This feedback helps improve future posts.\n\n"
                "Example: 'Too promotional' or 'Wrong tone'"
            )

    async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
        global rejection_reason, waiting_for_reason
        if not waiting_for_reason:
            return
        rejection_reason = update.message.text
        waiting_for_reason = False
        await update.message.reply_text(
            f"📝 Feedback recorded\n\nReason: {rejection_reason}"
        )
        decision_made.set()

    async def _run():
        global pending_post, decision_result, rejection_reason, waiting_for_reason
        pending_post = post_content
        decision_result = None
        rejection_reason = None
        waiting_for_reason = False
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
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        await decision_made.wait()

        await app.updater.stop()
        await app.stop()
        await app.shutdown()

        return decision_result, rejection_reason

    return asyncio.run(_run())


def main(feedback: bool = False):
    # Step 1: Generate post using Workshop 1 code
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    print("\n--- Generated Post ---")
    print(post_content)
    print("----------------------\n")

    if not feedback:
        print("DRY RUN - Use --feedback flag to enable Telegram feedback flow")
        return

    # Step 2: Send to Telegram and wait for human decision with feedback
    decision, reason = wait_for_decision_with_feedback(post_content)

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
        print(f"❌ Human rejected the post.")
        print(f"📝 Reason: {reason}")
        print("\n" + "=" * 50)
        print("💡 NEXT STEP: Use this feedback to improve the prompt!")
        print("=" * 50)
        print(f"""
You could modify the system prompt in workshop-1/post_generator.py
to address this feedback. For example, if reason was:

    "{reason}"

You might add to the prompt:
- "Avoid being too promotional"
- "Use a more casual tone"  
- "Include more specific examples"

This creates a feedback loop where human input
improves AI output over time!
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post Generator with Feedback Loop"
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable Telegram feedback flow",
    )
    args = parser.parse_args()

    main(feedback=args.feedback)
