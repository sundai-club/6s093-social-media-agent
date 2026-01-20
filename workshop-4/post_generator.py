"""
Generate social media posts from business docs and post to Mastodon.
This is the base post generator - copied from workshop-1 for self-contained workshop-3.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from mastodon import Mastodon
from openai import OpenAI

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


def post_to_mastodon(content: str) -> dict:
    """Post content to Mastodon."""
    mastodon = Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )
    return mastodon.status_post(content)


def main(post: bool = False):
    print("Reading business docs...")
    docs_content = read_business_docs()

    print("Generating post with LLM...")
    post_content = generate_post(docs_content)

    print("\n--- Generated Post ---")
    print(post_content)
    print("----------------------\n")

    if post:
        print("Posting to Mastodon...")
        result = post_to_mastodon(post_content)
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
