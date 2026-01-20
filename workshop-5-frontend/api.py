"""
Workshop 5 Frontend: FastAPI server with Admin Dashboard.
(Extended from workshop-4 with static file serving)

Provides REST API endpoints plus a web dashboard for:
- Viewing posts, responses, and statistics
- Controlling doc watcher and comment listener services
- Managing embeddings
- RAG search
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from database import (
    get_db,
    get_all_posts,
    get_all_responses,
    get_post_by_id,
    get_response_by_id,
    get_stats,
    count_embeddings,
    get_all_comment_replies,
)

# Import from other workshop-5 modules
from notion_watcher import (
    check_notion_changes,
    load_notion_state,
    get_current_notion_state,
    get_notion_client,
    get_parent_page_id,
    detect_notion_changes,
    save_notion_state,
    DEFAULT_POLL_INTERVAL,
    DEFAULT_MIN_WORDS,
    DEFAULT_MIN_LINES,
)
from embeddings import (
    init_embeddings,
    refresh_embeddings,
    embed_business_docs,
    embed_posts,
    embed_responses,
    generate_embedding,
)
from rag import hybrid_search, retrieve_context

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(
    title="Social Media Agent API",
    description="API for the Emanon social media agent with Admin Dashboard",
    version="3.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ========================================
# SERVICE STATE MANAGEMENT
# ========================================

_services = {
    "watcher": {
        "thread": None,
        "running": False,
        "stop_event": threading.Event(),
        "handler": None,
        "observer": None,
        "config": {
            "should_post": False,
            "approve": False,
            "min_words": DEFAULT_MIN_WORDS,
            "min_lines": DEFAULT_MIN_LINES,
        },
        "stats": {
            "changes_detected": 0,
            "posts_generated": 0,
            "started_at": None,
        },
        # Cache for Notion status to avoid rate limiting
        "notion_cache": {
            "current_count": 0,
            "last_fetched": 0,
            "cache_ttl": 30,  # Only fetch from Notion every 30 seconds
        },
    },
    "comments": {
        "thread": None,
        "running": False,
        "stop_event": threading.Event(),
        "config": {
            "should_post": False,
            "poll_interval": 60,
        },
        "stats": {
            "comments_processed": 0,
            "replies_generated": 0,
            "started_at": None,
        },
    },
}


# ========================================
# PYDANTIC MODELS - EXISTING
# ========================================


class PostResponse(BaseModel):
    id: int
    content: str
    post_url: Optional[str] = None
    image_url: Optional[str] = None
    posted: bool
    created_at: str


class GeneratePostRequest(BaseModel):
    context: str = Field(default="", description="Optional topic or context for the post")
    generate_image: bool = Field(default=True, description="Whether to generate an AI image")


class GeneratePostResponse(BaseModel):
    id: int
    content: str
    image_url: Optional[str]
    created_at: str


class KeywordResponse(BaseModel):
    id: int
    original_post_id: str
    original_post_content: str
    original_post_author: str
    response_text: str
    relevance_score: Optional[float]
    is_company_related: Optional[bool]
    reply_url: Optional[str]
    posted: bool
    created_at: str


class StatsResponse(BaseModel):
    total_posts: int
    posted_posts: int
    total_responses: int
    posted_responses: int
    # New fields for workshop-4
    total_embeddings: Optional[int] = None
    business_doc_embeddings: Optional[int] = None
    post_embeddings: Optional[int] = None
    response_embeddings: Optional[int] = None
    total_comment_replies: Optional[int] = None
    posted_comment_replies: Optional[int] = None
    processed_comments: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str


# ========================================
# PYDANTIC MODELS - WATCHER
# ========================================


class WatcherStartRequest(BaseModel):
    should_post: bool = Field(default=False, description="Post to Mastodon after approval")
    approve: bool = Field(default=False, description="Enable Telegram approval flow")
    min_words: int = Field(default=DEFAULT_MIN_WORDS, description="Min word changes to trigger post")
    min_lines: int = Field(default=DEFAULT_MIN_LINES, description="Min line changes to trigger post")
    poll_interval: int = Field(default=DEFAULT_POLL_INTERVAL, description="Notion polling interval in seconds")


class WatcherStatusResponse(BaseModel):
    running: bool
    docs_tracked: int
    current_docs: int
    config: dict
    stats: dict


class WatcherCheckResponse(BaseModel):
    has_changes: bool
    changes: dict
    post_content: Optional[str]
    image_url: Optional[str]
    post_url: Optional[str]
    skipped_files: list[str]


# ========================================
# PYDANTIC MODELS - EMBEDDINGS
# ========================================


class EmbeddingsStatsResponse(BaseModel):
    business_docs: int
    posts: int
    responses: int
    total: int


class EmbeddingsInitResponse(BaseModel):
    success: bool
    message: str
    stats: EmbeddingsStatsResponse


# ========================================
# PYDANTIC MODELS - SEARCH
# ========================================


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=10, description="Number of results to return")
    keyword_weight: float = Field(default=0.5, description="BM25 keyword weight (0-1)")
    semantic_weight: float = Field(default=0.5, description="Semantic similarity weight (0-1)")
    source_types: Optional[list[str]] = Field(default=None, description="Filter by source types")


class SearchResultItem(BaseModel):
    content: str
    source_type: str
    source_id: Optional[str]
    metadata: dict
    bm25_score: float
    cosine_score: float
    final_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    result_count: int


# ========================================
# PYDANTIC MODELS - COMMENTS
# ========================================


class CommentsStartRequest(BaseModel):
    should_post: bool = Field(default=False, description="Actually post replies to Mastodon")
    poll_interval: int = Field(default=60, description="Seconds between checks")


class CommentsStatusResponse(BaseModel):
    running: bool
    config: dict
    stats: dict


class CommentReplyResponse(BaseModel):
    id: int
    parent_post_id: str
    comment_id: str
    comment_content: str
    comment_author: str
    reply_text: str
    reply_url: Optional[str]
    posted: bool
    created_at: str


# ========================================
# DASHBOARD ENDPOINT
# ========================================


@app.get("/dashboard")
async def dashboard():
    """Serve the admin dashboard."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found. Create static/index.html")
    return FileResponse(str(index_path))


@app.get("/logs")
async def logs_page():
    """Serve the generation logs viewer."""
    logs_path = STATIC_DIR / "logs.html"
    if not logs_path.exists():
        raise HTTPException(status_code=404, detail="Logs page not found. Create static/logs.html")
    return FileResponse(str(logs_path))


# ========================================
# ROOT AND HEALTH ENDPOINTS
# ========================================


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Social Media Agent API",
        "version": "3.0.0",
        "dashboard": "/dashboard",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "posts": "/posts",
            "responses": "/responses",
            "stats": "/stats",
            "watcher": "/watcher/*",
            "embeddings": "/embeddings/*",
            "search": "/search",
            "comments": "/comments/*",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        db = get_db()
        db.execute("SELECT 1")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "unhealthy",
        database=db_status,
        version="3.0.0",
    )


# ========================================
# STATS ENDPOINT (ENHANCED)
# ========================================


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get comprehensive statistics about posts, responses, embeddings, and comments."""
    db = get_db()
    stats = get_stats(db)

    # Add embedding counts by type
    post_embeddings = count_embeddings(db, "post")
    response_embeddings = count_embeddings(db, "response")

    return StatsResponse(
        total_posts=stats["total_posts"],
        posted_posts=stats["posted_posts"],
        total_responses=stats["total_responses"],
        posted_responses=stats["posted_responses"],
        total_embeddings=stats.get("total_embeddings"),
        business_doc_embeddings=stats.get("business_doc_embeddings"),
        post_embeddings=post_embeddings,
        response_embeddings=response_embeddings,
        total_comment_replies=stats.get("total_comment_replies"),
        posted_comment_replies=stats.get("posted_comment_replies"),
        processed_comments=stats.get("processed_comments"),
    )


# ========================================
# POSTS ENDPOINTS
# ========================================


@app.get("/posts", response_model=list[PostResponse])
async def list_posts():
    """Get all generated posts."""
    db = get_db()
    posts = get_all_posts(db)
    return [PostResponse(**post) for post in posts]


@app.get("/posts/{post_id}", response_model=PostResponse)
async def get_post(post_id: int):
    """Get a specific post by ID."""
    db = get_db()
    post = get_post_by_id(db, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    return PostResponse(**post)


@app.delete("/posts/{post_id}", response_model=dict)
async def delete_post(post_id: int):
    """Delete a post (only unpublished posts can be deleted)."""
    db = get_db()
    post = get_post_by_id(db, post_id)

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if post["posted"]:
        raise HTTPException(status_code=400, detail="Cannot delete a published post")

    cursor = db.cursor()
    cursor.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    db.commit()

    return {"success": True, "message": "Post deleted"}


class PublishPostRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Optional image URL to include")


@app.post("/posts/{post_id}/publish", response_model=dict)
async def publish_post(post_id: int, request: PublishPostRequest = None):
    """Publish a pending post to Mastodon immediately, optionally with an image."""
    from mastodon import Mastodon
    import requests

    db = get_db()
    post = get_post_by_id(db, post_id)

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if post["posted"]:
        raise HTTPException(status_code=400, detail="Post already published")

    # Check Mastodon credentials
    access_token = os.environ.get("MASTODON_ACCESS_TOKEN")
    instance_url = os.environ.get("MASTODON_INSTANCE_URL")

    if not access_token or not instance_url:
        raise HTTPException(
            status_code=400,
            detail="MASTODON_ACCESS_TOKEN and MASTODON_INSTANCE_URL must be set"
        )

    try:
        # Post to Mastodon
        mastodon = Mastodon(
            access_token=access_token,
            api_base_url=instance_url,
        )

        # Handle image - use from request or from database
        media_ids = None
        image_to_use = (request.image_url if request and request.image_url else None) or post.get("image_url")
        if image_to_use:
            try:
                # Download image
                img_response = requests.get(image_to_use)
                img_response.raise_for_status()
                local_path = "/tmp/temp_post_image.webp"
                with open(local_path, "wb") as f:
                    f.write(img_response.content)
                # Upload to Mastodon
                media = mastodon.media_post(local_path)
                media_ids = [media]
            except Exception as e:
                print(f"Failed to attach image: {e}")
                # Continue without image

        # Ensure #AIgenerated hashtag is always present
        content = post["content"]
        hashtag = "#AIgenerated"
        if hashtag.lower() not in content.lower():
            if len(content) + len(hashtag) + 1 <= 500:
                content = content.rstrip() + " " + hashtag
            else:
                max_len = 500 - len(hashtag) - 1
                content = content[:max_len].rstrip() + " " + hashtag

        status = mastodon.status_post(content, media_ids=media_ids)
        post_url = status.get("url", "")

        # Update database
        cursor = db.cursor()
        cursor.execute(
            "UPDATE posts SET posted = 1, post_url = ? WHERE id = ?",
            (post_url, post_id)
        )
        db.commit()

        return {
            "success": True,
            "message": "Posted to Mastodon",
            "post_url": post_url,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to post: {str(e)}")


@app.post("/posts/generate", response_model=GeneratePostResponse)
async def generate_post(request: GeneratePostRequest):
    """Generate a new post using LLM and optionally an AI image."""
    from openai import OpenAI
    from database import save_post
    from datetime import datetime
    from generation_logger import log_generation

    db = get_db()

    # Get OpenRouter client
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise HTTPException(status_code=400, detail="OPENROUTER_API_KEY must be set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key,
    )

    # Build the prompt
    context_text = request.context.strip() if request.context else ""

    # Get Notion docs for context via RAG
    rag_query = context_text if context_text else "AI consulting services and insights"
    rag_results = []
    rag_context = ""
    try:
        from rag import hybrid_search, retrieve_context
        from embeddings import generate_embedding
        query_embedding = generate_embedding(rag_query)
        rag_results = hybrid_search(rag_query, query_embedding, top_k=3)
        rag_context, _ = retrieve_context(rag_query, query_embedding, top_k=3)
    except Exception as e:
        print(f"RAG retrieval error: {e}")
        rag_context = ""

    system_prompt = """You are a social media manager for Emanon, an AI news and consulting platform.
Your task is to create engaging Mastodon posts that:
- Are concise (under 400 characters)
- Highlight Emanon's no-hype, practical approach to AI
- Include 1-2 relevant hashtags
- Sound authentic, not salesy
- Share ONE interesting insight

Write in a conversational but professional tone."""

    user_prompt = f"""Create a single engaging Mastodon post for Emanon.

{f'Topic/Context: {context_text}' if context_text else 'Share a valuable AI insight or tip.'}

{f'Background information from our docs:{chr(10)}{rag_context[:1500]}' if rag_context else ''}

Generate just the post text, nothing else."""

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-001",
            max_tokens=300,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        post_content = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate post: {str(e)}")

    # Ensure #AIgenerated hashtag is present
    hashtag = "#AIgenerated"
    if hashtag.lower() not in post_content.lower():
        if len(post_content) + len(hashtag) + 1 <= 500:
            post_content = post_content.rstrip() + " " + hashtag
        else:
            max_len = 500 - len(hashtag) - 1
            post_content = post_content[:max_len].rstrip() + " " + hashtag

    # Generate image if requested
    image_url = None
    image_prompt = None
    if request.generate_image:
        try:
            import replicate
            replicate_token = os.environ.get("REPLICATE_API_TOKEN")
            if replicate_token:
                image_prompt = f"annddrreeww, 24 years old male, presenting about: {post_content[:200]}"
                output = replicate.run(
                    "sundai-club/andrews_model:f5211077a830f0b1cb51e541d4f591fae107a7617ce6cc54fd23c205cae0c1b5",
                    input={"prompt": image_prompt},
                )
                image_url = str(output[0]) if output else None
        except Exception as e:
            print(f"Image generation failed: {e}")
            # Continue without image

    # Save to database with image_url
    post_id = save_post(db, post_content, image_url=image_url, posted=False)

    # Log the generation with detailed RAG scores
    log_generation(
        post_id=post_id,
        source="api",
        rag_query=rag_query,
        rag_results=[{
            "content": r["content"][:300],
            "source_type": r["source_type"],
            "source_id": r.get("source_id"),
            "final_score": round(r["final_score"], 4),
            "bm25_score": round(r.get("bm25_score", 0), 4),
            "cosine_score": round(r.get("cosine_score", 0), 4),
        } for r in rag_results],
        rag_context_text=rag_context,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        generated_post=post_content,
        image_prompt=image_prompt,
        image_url=image_url,
        extra={"user_context": context_text, "generate_image_requested": request.generate_image},
    )

    return GeneratePostResponse(
        id=post_id,
        content=post_content,
        image_url=image_url,
        created_at=datetime.now().isoformat(),
    )


# ========================================
# RESPONSES ENDPOINTS
# ========================================


@app.get("/responses", response_model=list[KeywordResponse])
async def list_responses():
    """Get all generated responses."""
    db = get_db()
    responses = get_all_responses(db)
    return [KeywordResponse(**resp) for resp in responses]


@app.get("/responses/{response_id}", response_model=KeywordResponse)
async def get_response(response_id: int):
    """Get a specific response by ID."""
    db = get_db()
    response = get_response_by_id(db, response_id)
    if not response:
        raise HTTPException(status_code=404, detail="Response not found")
    return KeywordResponse(**response)


# ========================================
# DOCUMENT WATCHER ENDPOINTS
# ========================================


def _watcher_thread_fn(stop_event: threading.Event, config: dict):
    """Background thread function for Notion document polling."""
    poll_interval = config.get("poll_interval", DEFAULT_POLL_INTERVAL)

    while not stop_event.is_set():
        try:
            db = get_db()
            result = check_notion_changes(
                db,
                should_post=config["should_post"],
                approve=config["approve"],
                min_words=config["min_words"],
                min_lines=config["min_lines"],
            )

            if result["has_changes"]:
                _services["watcher"]["stats"]["changes_detected"] += 1
                if result.get("post_content"):
                    _services["watcher"]["stats"]["posts_generated"] += 1

        except Exception as e:
            print(f"Notion watcher error: {e}")

        # Sleep with stop check
        for _ in range(poll_interval):
            if stop_event.is_set():
                break
            time.sleep(1)


@app.post("/watcher/start", response_model=dict)
async def start_watcher(request: WatcherStartRequest):
    """Start the Notion document watcher in background (polling mode)."""
    if _services["watcher"]["running"]:
        raise HTTPException(status_code=400, detail="Watcher is already running")

    # Check Notion env vars
    if not os.environ.get("NOTION_API_KEY") or not os.environ.get("NOTION_PARENT_PAGE_ID"):
        raise HTTPException(
            status_code=400,
            detail="NOTION_API_KEY and NOTION_PARENT_PAGE_ID must be set"
        )

    # Update config
    _services["watcher"]["config"] = {
        "should_post": request.should_post,
        "approve": request.approve,
        "min_words": request.min_words,
        "min_lines": request.min_lines,
        "poll_interval": request.poll_interval,
    }

    # Reset stop event and start thread
    _services["watcher"]["stop_event"].clear()
    thread = threading.Thread(
        target=_watcher_thread_fn,
        args=(_services["watcher"]["stop_event"], _services["watcher"]["config"]),
        daemon=True,
    )
    thread.start()

    _services["watcher"]["thread"] = thread
    _services["watcher"]["running"] = True
    _services["watcher"]["stats"]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "started",
        "message": f"Notion watcher started (polling every {request.poll_interval}s)",
        "config": _services["watcher"]["config"],
    }


@app.post("/watcher/stop", response_model=dict)
async def stop_watcher():
    """Stop the Notion document watcher."""
    if not _services["watcher"]["running"]:
        raise HTTPException(status_code=400, detail="Watcher is not running")

    # Signal stop
    _services["watcher"]["stop_event"].set()

    # Wait for thread to finish (with timeout)
    thread = _services["watcher"]["thread"]
    if thread:
        thread.join(timeout=5.0)

    _services["watcher"]["running"] = False
    _services["watcher"]["thread"] = None

    return {
        "status": "stopped",
        "message": "Notion watcher stopped successfully",
    }


@app.get("/watcher/status", response_model=WatcherStatusResponse)
async def get_watcher_status():
    """Get the current Notion watcher state and statistics."""
    db = get_db()

    # Get Notion page counts from saved state
    saved_state = load_notion_state(db)

    # Use cached Notion count to avoid rate limiting
    cache = _services["watcher"]["notion_cache"]
    now = time.time()

    # Only fetch from Notion if cache expired (every 30s)
    if now - cache["last_fetched"] > cache["cache_ttl"]:
        try:
            notion = get_notion_client()
            parent_id = get_parent_page_id()
            current_state = get_current_notion_state(notion, parent_id)
            cache["current_count"] = len(current_state)
            cache["last_fetched"] = now
        except Exception as e:
            # Keep old cached value on error, don't spam logs
            if cache["current_count"] == 0:
                print(f"Notion status fetch error: {e}")

    return WatcherStatusResponse(
        running=_services["watcher"]["running"],
        docs_tracked=len(saved_state),
        current_docs=cache["current_count"],
        config=_services["watcher"]["config"],
        stats=_services["watcher"]["stats"],
    )


@app.post("/watcher/check", response_model=WatcherCheckResponse)
async def check_watcher():
    """Perform a one-time check for Notion document changes (without starting continuous watch)."""
    db = get_db()

    result = check_notion_changes(
        db,
        should_post=False,  # Don't auto-post from API check
        approve=False,      # Don't send to Telegram
        min_words=_services["watcher"]["config"]["min_words"],
        min_lines=_services["watcher"]["config"]["min_lines"],
    )

    return WatcherCheckResponse(
        has_changes=result["has_changes"],
        changes=result.get("changes", {}),
        post_content=result.get("post_content"),
        image_url=result.get("image_url"),
        post_url=result.get("post_url"),
        skipped_files=result.get("skipped_pages", []),
    )


@app.post("/watcher/reset", response_model=dict)
async def reset_watcher():
    """Reset the Notion document state (treats all docs as new on next check)."""
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM notion_doc_state")
    db.commit()

    return {
        "status": "reset",
        "message": "Notion document state cleared. All pages will be treated as new on next check.",
    }


# ========================================
# EMBEDDINGS ENDPOINTS
# ========================================


@app.post("/embeddings/init", response_model=EmbeddingsInitResponse)
async def initialize_embeddings(background_tasks: BackgroundTasks):
    """Initialize all embeddings from scratch (Notion docs, posts, responses)."""
    db = get_db()

    try:
        # Run initialization
        init_embeddings()

        # Get final stats
        stats = EmbeddingsStatsResponse(
            business_docs=count_embeddings(db, "notion_doc"),
            posts=count_embeddings(db, "post"),
            responses=count_embeddings(db, "response"),
            total=count_embeddings(db),
        )

        return EmbeddingsInitResponse(
            success=True,
            message="Embeddings initialized successfully from Notion",
            stats=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize embeddings: {str(e)}")


@app.post("/embeddings/refresh", response_model=EmbeddingsInitResponse)
async def refresh_embeddings_endpoint():
    """Refresh embeddings: re-embed Notion docs, add new posts/responses."""
    db = get_db()

    try:
        refresh_embeddings()

        stats = EmbeddingsStatsResponse(
            business_docs=count_embeddings(db, "notion_doc"),
            posts=count_embeddings(db, "post"),
            responses=count_embeddings(db, "response"),
            total=count_embeddings(db),
        )

        return EmbeddingsInitResponse(
            success=True,
            message="Embeddings refreshed successfully from Notion",
            stats=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh embeddings: {str(e)}")


@app.get("/embeddings/stats", response_model=EmbeddingsStatsResponse)
async def get_embeddings_stats():
    """Get embedding counts by type."""
    db = get_db()

    return EmbeddingsStatsResponse(
        business_docs=count_embeddings(db, "notion_doc"),
        posts=count_embeddings(db, "post"),
        responses=count_embeddings(db, "response"),
        total=count_embeddings(db),
    )


# ========================================
# GENERATION LOGS ENDPOINTS
# ========================================


@app.get("/logs/generations")
async def get_generation_logs(limit: int = 10):
    """Get recent generation logs showing RAG context, prompts, and outputs."""
    from generation_logger import get_recent_logs
    return get_recent_logs(limit)


@app.get("/logs/generations/{post_id}")
async def get_generation_log(post_id: int):
    """Get generation log for a specific post."""
    from pathlib import Path
    import json

    logs_dir = Path(__file__).parent / "logs"
    # Find log file for this post
    for log_file in logs_dir.glob(f"*_post_{post_id}.json"):
        with open(log_file) as f:
            return json.load(f)

    raise HTTPException(status_code=404, detail=f"No log found for post {post_id}")


# ========================================
# RAG SEARCH ENDPOINT
# ========================================


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Perform hybrid search combining BM25 keyword matching and semantic similarity.

    Returns results sorted by combined score.
    """
    db = get_db()

    # Check if embeddings exist
    total_embeddings = count_embeddings(db)
    if total_embeddings == 0:
        raise HTTPException(
            status_code=400,
            detail="No embeddings found. Initialize embeddings first with POST /embeddings/init"
        )

    try:
        # Generate embedding for query
        query_embedding = generate_embedding(request.query)

        # Perform hybrid search
        results = hybrid_search(
            query=request.query,
            query_embedding=query_embedding,
            keyword_weight=request.keyword_weight,
            semantic_weight=request.semantic_weight,
            top_k=request.top_k,
            source_types=request.source_types,
        )

        # Convert to response model
        result_items = [
            SearchResultItem(
                content=r["content"],
                source_type=r["source_type"],
                source_id=r.get("source_id"),
                metadata=r.get("metadata", {}),
                bm25_score=r["bm25_score"],
                cosine_score=r["cosine_score"],
                final_score=r["final_score"],
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=result_items,
            result_count=len(result_items),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# ========================================
# COMMENT LISTENER ENDPOINTS
# ========================================


def _comment_listener_thread_fn(stop_event: threading.Event, config: dict):
    """Background thread function for comment listening."""
    from mastodon import Mastodon
    from openai import OpenAI
    from comment_listener import (
        fetch_comment_notifications,
        extract_comment_info,
        process_comment,
    )

    try:
        mastodon = Mastodon(
            access_token=os.environ["MASTODON_ACCESS_TOKEN"],
            api_base_url=os.environ["MASTODON_INSTANCE_URL"],
        )
        openai_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    except KeyError as e:
        print(f"Missing environment variable for comment listener: {e}")
        return

    poll_interval = config["poll_interval"]
    should_post = config["should_post"]

    while not stop_event.is_set():
        try:
            db = get_db()

            # Fetch notifications
            from comment_listener import is_comment_processed
            notifications = fetch_comment_notifications(mastodon)

            # Filter to unprocessed
            new_comments = []
            for notif in notifications:
                comment = extract_comment_info(notif)
                if not is_comment_processed(db, comment["comment_id"]):
                    new_comments.append(comment)

            # Process each
            for comment in new_comments:
                try:
                    process_comment(
                        comment=comment,
                        mastodon=mastodon,
                        openai_client=openai_client,
                        db=db,
                        should_post=should_post,
                    )
                    _services["comments"]["stats"]["comments_processed"] += 1
                    _services["comments"]["stats"]["replies_generated"] += 1
                except Exception as e:
                    print(f"Error processing comment: {e}")
                    from comment_listener import mark_comment_processed
                    mark_comment_processed(db, comment["comment_id"])

        except Exception as e:
            print(f"Comment listener error: {e}")

        # Sleep with stop check
        for _ in range(poll_interval):
            if stop_event.is_set():
                break
            time.sleep(1)


@app.post("/comments/start", response_model=dict)
async def start_comment_listener(request: CommentsStartRequest):
    """Start the comment listener in background."""
    if _services["comments"]["running"]:
        raise HTTPException(status_code=400, detail="Comment listener is already running")

    # Check required env vars
    required_vars = ["MASTODON_ACCESS_TOKEN", "MASTODON_INSTANCE_URL", "OPENROUTER_API_KEY"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing environment variables: {', '.join(missing)}"
        )

    # Update config
    _services["comments"]["config"] = {
        "should_post": request.should_post,
        "poll_interval": request.poll_interval,
    }

    # Reset stop event and start thread
    _services["comments"]["stop_event"].clear()
    thread = threading.Thread(
        target=_comment_listener_thread_fn,
        args=(_services["comments"]["stop_event"], _services["comments"]["config"]),
        daemon=True,
    )
    thread.start()

    _services["comments"]["thread"] = thread
    _services["comments"]["running"] = True
    _services["comments"]["stats"]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "started",
        "message": f"Comment listener started (polling every {request.poll_interval}s)",
        "config": _services["comments"]["config"],
    }


@app.post("/comments/stop", response_model=dict)
async def stop_comment_listener():
    """Stop the comment listener."""
    if not _services["comments"]["running"]:
        raise HTTPException(status_code=400, detail="Comment listener is not running")

    # Signal stop
    _services["comments"]["stop_event"].set()

    # Wait for thread to finish
    thread = _services["comments"]["thread"]
    if thread:
        thread.join(timeout=5.0)

    _services["comments"]["running"] = False
    _services["comments"]["thread"] = None

    return {
        "status": "stopped",
        "message": "Comment listener stopped successfully",
    }


@app.get("/comments/status", response_model=CommentsStatusResponse)
async def get_comment_listener_status():
    """Get the current comment listener state."""
    return CommentsStatusResponse(
        running=_services["comments"]["running"],
        config=_services["comments"]["config"],
        stats=_services["comments"]["stats"],
    )


@app.get("/comments/replies", response_model=list[CommentReplyResponse])
async def list_comment_replies():
    """Get all generated comment replies."""
    db = get_db()
    replies = get_all_comment_replies(db)

    return [
        CommentReplyResponse(
            id=r["id"],
            parent_post_id=r["parent_post_id"],
            comment_id=r["comment_id"],
            comment_content=r["comment_content"],
            comment_author=r["comment_author"],
            reply_text=r["reply_text"],
            reply_url=r.get("reply_url"),
            posted=r["posted"],
            created_at=r["created_at"],
        )
        for r in replies
    ]


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
