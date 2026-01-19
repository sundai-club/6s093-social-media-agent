"""
Test script to verify Notion API connectivity.

Prerequisites:
1. Create a Notion integration at https://www.notion.so/my-integrations
2. Share your parent page with the integration
3. Set environment variables:
   - NOTION_API_KEY: Your integration token
   - NOTION_PARENT_PAGE_ID: The ID of the parent page containing documents

Run with:
    cd workshop-5-frontend
    NOTION_API_KEY=secret_xxx NOTION_PARENT_PAGE_ID=xxx python test_notion.py
"""

import os
import sys
from typing import Optional


def test_notion_connection():
    """Test Notion API authentication and page access."""
    try:
        from notion_client import Client
    except ImportError:
        print("ERROR: notion-client not installed. Run: uv sync")
        return False

    # Check environment variables
    api_key = os.environ.get("NOTION_API_KEY")
    parent_id = os.environ.get("NOTION_PARENT_PAGE_ID")

    if not api_key:
        print("ERROR: NOTION_API_KEY environment variable not set")
        print("Get your API key from: https://www.notion.so/my-integrations")
        return False

    if not parent_id:
        print("ERROR: NOTION_PARENT_PAGE_ID environment variable not set")
        print("This should be the ID of the page containing your documents")
        return False

    print(f"Testing Notion API connection...")
    print(f"Parent page ID: {parent_id[:8]}...{parent_id[-4:]}")

    # Initialize client
    try:
        notion = Client(auth=api_key)
        print("✓ Notion client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        return False

    # Test 1: Fetch child pages
    print("\n--- Test 1: Fetching child pages ---")
    try:
        children = notion.blocks.children.list(block_id=parent_id)
        pages_found = []

        for block in children.get("results", []):
            if block["type"] == "child_page":
                page_title = block["child_page"]["title"]
                page_id = block["id"]
                pages_found.append({"id": page_id, "title": page_title})
                print(f"  Found page: {page_title} ({page_id[:8]}...)")

        if pages_found:
            print(f"✓ Found {len(pages_found)} child pages")
        else:
            print("⚠ No child pages found - make sure you have documents under this page")

    except Exception as e:
        print(f"✗ Failed to fetch children: {e}")
        return False

    # Test 2: Fetch page content (if we found pages)
    if pages_found:
        print("\n--- Test 2: Fetching page content ---")
        test_page = pages_found[0]
        try:
            # Get page metadata
            page = notion.pages.retrieve(page_id=test_page["id"])
            print(f"  Page title: {test_page['title']}")
            print(f"  Last edited: {page.get('last_edited_time', 'unknown')}")

            # Get page blocks (content)
            blocks = notion.blocks.children.list(block_id=test_page["id"])
            content_preview = extract_text_from_blocks(blocks.get("results", []))

            if content_preview:
                preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                print(f"  Content preview: {preview}")
                print(f"✓ Successfully retrieved page content ({len(content_preview)} chars)")
            else:
                print("⚠ Page appears to be empty")

        except Exception as e:
            print(f"✗ Failed to fetch page content: {e}")
            return False

    print("\n" + "="*50)
    print("✓ All Notion API tests passed!")
    print("="*50)
    return True


def extract_text_from_blocks(blocks: list) -> str:
    """Extract plain text from Notion blocks."""
    text_parts = []

    for block in blocks:
        block_type = block.get("type")

        if block_type in ["paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item"]:
            rich_text = block.get(block_type, {}).get("rich_text", [])
            for text in rich_text:
                text_parts.append(text.get("plain_text", ""))
        elif block_type == "code":
            code_text = block.get("code", {}).get("rich_text", [])
            for text in code_text:
                text_parts.append(text.get("plain_text", ""))

    return "\n".join(text_parts)


if __name__ == "__main__":
    success = test_notion_connection()
    sys.exit(0 if success else 1)
