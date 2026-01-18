"""
FastAPI server for the social media agent.
Provides REST API endpoints to view posts, responses, and statistics.
This server is designed to be deployed as a systemd service on a GCP VM.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database import (
    get_db,
    get_all_posts,
    get_all_responses,
    get_post_by_id,
    get_response_by_id,
    get_stats,
)

load_dotenv(Path(__file__).parent.parent / ".env")

app = FastAPI(
    title="Social Media Agent API",
    description="API for the Emanon social media agent - tracks posts and responses",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API responses
class PostResponse(BaseModel):
    id: int
    content: str
    post_url: Optional[str]
    posted: bool
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


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Social Media Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
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
        version="1.0.0",
    )


@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get statistics about posts and responses."""
    db = get_db()
    stats = get_stats(db)
    return StatsResponse(**stats)


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


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
