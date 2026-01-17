"""
Generate social media posts from business docs, generate images, and post to Mastodon.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from mastodon import Mastodon
from openai import OpenAI
import replicate
import requests

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


def post_to_mastodon(content: str, image_url: str) -> dict:
    """Post content and image to Mastodon."""
    mastodon = Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )

    # Download the image locally
    local_image_path = "temp_image.webp"
    download_image(image_url, local_image_path)

    # Upload the image to Mastodon
    media = mastodon.media_post(local_image_path)

    # Post the content with the uploaded image
    return mastodon.status_post(content, media_ids=[media])


def main(post: bool = False):
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    print("Generating image prompt...")
    image_prompt = f"A cartoon noir style image of the djeny dog dressed as a detective doing something related to: {post_content}"

    print("Generating image...")
    image_url = generate_image(image_prompt)

    print("\n--- Generated Post ---")
    print(post_content)
    print("\n--- Image URL ---")
    print(image_url)
    print("----------------------\n")

    if post:
        print("Posting to Mastodon...")
        result = post_to_mastodon(post_content, image_url)
        print(f"Posted successfully! URL: {result['url']}")
    else:
        print("DRY RUN - Post not published. Use --post flag to publish.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate and post to Mastodon")
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post to Mastodon (default: just print)",
    )
    args = parser.parse_args()

    main(post=args.post)
