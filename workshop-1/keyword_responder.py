"""
Search Mastodon for keywords and generate contextual responses.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from mastodon import Mastodon
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

BUSINESS_DOCS_DIR = Path(__file__).parent.parent / "business-docs"

SEARCH_KEYWORDS = [
    "AI consulting",
    "LLM deployment",
    "AI implementation",
    "AI strategy",
    "AI newsletter",
    "machine learning consulting",
    "AI tools comparison",
    "LLM benchmarks",
]


class LLMResponse(BaseModel):
    """LLM-generated response for a single post."""

    response_text: str
    is_company_related: bool
    relevance_score: float
    reasoning: str


class LLMResponseBatch(BaseModel):
    """Batch of LLM responses."""

    responses: list[LLMResponse]


class GeneratedResponse(BaseModel):
    """Full response with original post data."""

    original_post_id: str
    original_post_content: str
    original_post_author: str
    response_text: str
    is_company_related: bool
    relevance_score: float
    reasoning: str


def read_business_docs() -> str:
    """Read all markdown files from the business docs directory."""
    docs_content = []
    for doc_path in sorted(BUSINESS_DOCS_DIR.glob("*.md")):
        content = doc_path.read_text()
        docs_content.append(f"# File: {doc_path.name}\n\n{content}")
    return "\n\n---\n\n".join(docs_content)


def search_mastodon(keywords: list[str], limit_per_keyword: int = 5) -> list[dict]:
    """Search Mastodon for posts matching keywords."""
    mastodon = Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )

    all_posts = []
    seen_ids = set()

    for keyword in keywords:
        results = mastodon.search(keyword, result_type="statuses")
        statuses = results.get("statuses", [])

        for status in statuses[:limit_per_keyword]:
            if status["id"] not in seen_ids:
                seen_ids.add(status["id"])
                all_posts.append(
                    {
                        "id": str(status["id"]),
                        "content": status["content"],
                        "author": status["account"]["acct"],
                        "url": status["url"],
                    }
                )

    return all_posts


def generate_responses(
    posts: list[dict], business_context: str
) -> list[GeneratedResponse]:
    """Generate responses for found posts using structured output."""
    if not posts:
        return []

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    posts_text = "\n\n".join(
        f"Post {i + 1}:\n{p['content']}" for i, p in enumerate(posts)
    )

    response = client.beta.chat.completions.parse(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        messages=[
            {
                "role": "system",
                "content": f"""You are a community engagement specialist for Emanon.

About Emanon:
{business_context}

Your task is to craft helpful, authentic responses to social media posts.
Responses should:
- Be genuinely helpful and add value to the conversation
- Only mention Emanon when truly relevant (don't force it)
- Be conversational and friendly
- Stay under 400 characters
- Not be salesy or promotional unless directly relevant

For each post, determine:
- Whether a response mentioning Emanon would be appropriate
- A relevance score (0.0-1.0) for how relevant our expertise is
- The actual response text""",
            },
            {
                "role": "user",
                "content": f"""Generate responses for these Mastodon posts. Return one response per post in order.

Posts to respond to:
{posts_text}""",
            },
        ],
        response_format=LLMResponseBatch,
    )

    llm_responses = response.choices[0].message.parsed.responses

    return [
        GeneratedResponse(
            original_post_id=post["id"],
            original_post_content=post["content"],
            original_post_author=post["author"],
            **llm_resp.model_dump(),
        )
        for post, llm_resp in zip(posts, llm_responses)
    ]


def post_reply(response: GeneratedResponse) -> dict:
    """Post a reply to Mastodon."""
    mastodon = Mastodon(
        access_token=os.environ["MASTODON_ACCESS_TOKEN"],
        api_base_url=os.environ["MASTODON_INSTANCE_URL"],
    )
    return mastodon.status_post(
        response.response_text, in_reply_to_id=int(response.original_post_id)
    )


def main(post_replies: bool = False):
    print("Reading business docs for context...")
    business_context = read_business_docs()

    print(f"Searching Mastodon for keywords: {SEARCH_KEYWORDS[:3]}...")
    posts = search_mastodon(SEARCH_KEYWORDS)
    print(f"Found {len(posts)} unique posts")

    if not posts:
        print("No posts found to respond to.")
        return

    print("Generating responses with LLM...")
    responses = generate_responses(posts[:5], business_context)

    print(f"\n{'=' * 60}")
    print("GENERATED RESPONSES")
    print("=" * 60)

    for resp in responses:
        print(f"\n--- Response to @{resp.original_post_author} ---")
        print(f"Original: {resp.original_post_content[:100]}...")
        print(f"Relevance: {resp.relevance_score:.2f}")
        print(f"Company mention: {resp.is_company_related}")
        print(f"Reasoning: {resp.reasoning}")
        print(f"\nResponse:\n{resp.response_text}")
        print("-" * 40)

        if post_replies:
            print("Posting reply...")
            result = post_reply(resp)
            print(f"Posted: {result['url']}")

    if not post_replies:
        print(f"\n{'=' * 60}")
        print("DRY RUN - No replies posted. Set post_replies=True to post.")
        print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Search Mastodon and generate responses"
    )
    parser.add_argument(
        "--post",
        action="store_true",
        help="Actually post the replies (default: just print)",
    )
    args = parser.parse_args()

    main(post_replies=args.post)
