"""
Workshop 2: Telegram Bot Basics
Learn how to send messages and receive button responses via Telegram.

This script teaches the fundamentals before integrating with the post generator.
"""

import os
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

load_dotenv(Path(__file__).parent.parent / ".env")


def send_simple_message() -> None:
    """Send a basic text message to Telegram."""
    async def _send():
        bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        message = await bot.send_message(
            chat_id=int(os.environ["TELEGRAM_CHAT_ID"]),
            text="👋 Hello from Python! This is a simple message.",
        )
        print(f"✅ Message sent! ID: {message.message_id}")

    asyncio.run(_send())


def send_message_with_buttons() -> None:
    """Send a message with inline keyboard buttons."""
    async def _send():
        bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ])
        message = await bot.send_message(
            chat_id=int(os.environ["TELEGRAM_CHAT_ID"]),
            text="🔔 Action Required\n\nDo you approve this action?",
            reply_markup=keyboard,
        )
        print(f"✅ Message with buttons sent! ID: {message.message_id}")

    asyncio.run(_send())


def run_bot() -> None:
    """Run the bot with polling to listen for button responses."""

    async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        action = query.data
        print(f"📥 Button clicked: {action}")
        if action == "approve":
            await query.edit_message_text("✅ Action APPROVED!")
        elif action == "reject":
            await query.edit_message_text("❌ Action REJECTED!")

    async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(
            "🤖 Telegram Bot Basics\n\n"
            "Commands:\n"
            "/test - Send a test message with buttons"
        )

    async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Yes", callback_data="approve"),
                InlineKeyboardButton("❌ No", callback_data="reject"),
            ]
        ])
        await update.message.reply_text(
            "🧪 Test Message\n\nClick a button:",
            reply_markup=keyboard,
        )

    async def _run():
        app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("test", test_command))
        app.add_handler(CallbackQueryHandler(handle_button))
        await app.run_polling()

    print("🚀 Starting Telegram bot...")
    print("📱 Open Telegram and send /start to your bot")
    print("⏹️  Press Ctrl+C to stop\n")
    asyncio.run(_run())


def main(mode: str = "send"):
    """
    Run different modes:
    - send: Just send a message (no listening)
    - buttons: Send message with buttons (no listening)
    - bot: Run full bot with polling (listens for responses)
    """
    if mode == "send":
        print("📤 Sending simple message...")
        send_simple_message()
    elif mode == "buttons":
        print("📤 Sending message with buttons...")
        send_message_with_buttons()
        print("💡 Note: Buttons won't work until you run with --bot mode")
    elif mode == "bot":
        run_bot()
    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Bot Basics")
    parser.add_argument(
        "--mode",
        choices=["send", "buttons", "bot"],
        default="send",
        help="Mode: send (simple message), buttons (with buttons), bot (full polling)",
    )
    args = parser.parse_args()

    main(mode=args.mode)
