"""
Workshop 2.1: Post Generator with Telegram Approval and Mastodon Posting
Integrates Telegram approval workflow with the Workshop 2 post generator.

Flow:
1. Generate post using Workshop 2's post_generator
2. Send to Telegram for human approval
3. Wait for Approve/Reject response
4. If approved, publish to Mastodon
5. Add --approve flag to bypass Telegram approval
6. Add --image flag to control image generation
"""

import os
import sys
import asyncio
from pathlib import Path

from mastodon import Mastodon
from openai import OpenAI
import replicate
import requests

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

load_dotenv(Path(__file__).parent.parent / ".env")



BUSINESS_DOCS_DIR = Path(__file__).parent.parent / "business-docs"


def read_business_docs() -> str:
    """Read all markdown files from the business docs directory."""
    docs_content = []
    for doc_path in sorted(BUSINESS_DOCS_DIR.glob("*.md")):
        content = doc_path.read_text()
        docs_content.append(f"# File: {doc_path.name}\n\n{content}")
    return "\n\n---\n\n".join(docs_content)


def generate_post(docs_content: str) -> str:
    """Generate a social media post using OpenRouter."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        max_tokens=1024,
        messages=[
            {
                "role": "system",
                "content": """You are a social media manager for Emanon, an AI news and consulting platform.
Your task is to create engaging Mastodon posts that:
- Are concise (under 500 characters)
- Highlight Emanon's no-hype, practical approach to AI
- Include relevant hashtags
- Drive engagement and interest
- Sound authentic, not salesy

Write in a conversational but professional tone.""",
            },
            {
                "role": "user",
                "content": f"""Based on the following business documentation, create a single engaging Mastodon post
that promotes Emanon's services or shares valuable AI insights.

{docs_content}

Generate just the post text, nothing else.""",
            },
        ],
    )

    return response.choices[0].message.content


def generate_image(prompt: str) -> str:
    """Generate an image using the diffusion model."""
    output = replicate.run(
        "sundai-club/artems_dog_model:7103c7f706fe1429cf4bdb282ee81dfc218d643788b56f28dc6549c7dfb70967",
        input={
            "prompt": prompt,
            "num_inference_steps": 28,
            "guidance_scale": 7.5,
            "model": "dev",
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


def post_to_mastodon(content: str, image_url: str = None) -> dict:
    """Post content and image to Mastodon."""
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



# Store state for the approval flow
pending_post = None
decision_made = asyncio.Event()
decision_result = None


def send_for_approval(post_content: str, image_url: str) -> None:
    """Send a post to Telegram for human approval."""
    async def _send():
        bot = Bot(token=os.environ["TELEGRAM_BOT_TOKEN"])
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Approve", callback_data="approve"),
                InlineKeyboardButton("❌ Reject", callback_data="reject"),
            ]
        ])
        if image_url:
            await bot.send_photo(
                chat_id=int(os.environ["TELEGRAM_CHAT_ID"]),
                photo=image_url,
                caption=f"📝 New Post for Approval\n\n{post_content}\n\n"
                        f"Characters: {len(post_content)}",
                reply_markup=keyboard,
            )
        else:
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
        
        # Check if the message has a photo (which means it has a caption, not text)
        if query.message.photo:
            if decision_result == "approve":
                await query.edit_message_caption(
                    caption=f"✅ APPROVED\n\n{pending_post}",
                    reply_markup=None  # Remove the buttons
                )
            else:
                await query.edit_message_caption(
                    caption=f"❌ REJECTED\n\n{pending_post}",
                    reply_markup=None  # Remove the buttons
                )
        else:
            if decision_result == "approve":
                await query.edit_message_text(
                    text=f"✅ APPROVED\n\n{pending_post}",
                    reply_markup=None  # Remove the buttons
                )
            else:
                await query.edit_message_text(
                    text=f"❌ REJECTED\n\n{pending_post}",
                    reply_markup=None  # Remove the buttons
                )
        decision_made.set()

    async def _run():
        global pending_post
        pending_post = post_content
        decision_made.clear()

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


def main(approve: bool = False, image: bool = False):
    # Step 1: Generate post using Workshop 2 code
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    image_url = None
    if image:
        print("Generating image prompt...")
        image_prompt = f"A cartoon noir style image of the djeny dog dressed as a detective doing something related to: {post_content}"

        print("Generating image...")
        image_url = generate_image(image_prompt)

        print("\n--- Image URL ---")
        print(image_url)
        print("----------------------\n")

    print("\n--- Generated Post ---")
    print(post_content)
    print("----------------------\n")

    if approve:
        print("Posting directly to Mastodon...")
        try:
            result = post_to_mastodon(post_content, image_url)
            print(f"Posted successfully! URL: {result['url']}")
        except Exception as e:
            print(f"Error posting to Mastodon: {e}")
        return

    # Step 2: Send to Telegram and wait for human decision
    send_for_approval(post_content, image_url)
    decision = wait_for_decision(post_content)

    # Step 3: Act on decision
    if decision == "approve":
        print("✅ Human approved the post!")
        result = post_to_mastodon(post_content, image_url)
        print(f"Published: {result['url']}")
    else:
        print("❌ Human rejected. Post not published.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post Generator with Telegram Approval and Mastodon Posting"
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Post directly to Mastodon without Telegram approval",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Generate an image to accompany the post",
    )
    args = parser.parse_args()

    main(approve=args.approve, image=args.image)
