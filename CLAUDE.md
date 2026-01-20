# 6S093 Social Media Agent - Claude Code Instructions

## Project Overview

This is a **workshop-based educational project** for MIT's 6S093 course teaching AI-powered social media automation. The project progressively builds a social media agent across 5 workshops, culminating in a full-stack application with RAG, image generation, and an admin dashboard.

**Current active workshop:** `workshop-5-frontend/` - Production deployment with admin UI

## Tech Stack

- **Backend:** FastAPI + SQLite with FTS5 (full-text search)
- **Frontend:** HTML + Alpine.js + Tailwind CSS (via CDN)
- **Embeddings:** Local MiniLM-L6-v2 via fastembed/ONNX (no API calls)
- **LLM:** Google Gemini via OpenRouter API
- **Image Generation:** Replicate API (fine-tuned model with trigger word `annddrreeww`)
- **Social Platform:** Mastodon API
- **Human-in-the-Loop:** Telegram Bot API
- **Document Source:** Notion API
- **Package Manager:** uv
- **Deployment:** GCP VM with systemd service

## Directory Structure

```
.
├── workshop-1/          # Basic LLM + Mastodon posting
├── workshop-2/          # Telegram approval workflow
├── workshop-2.1/        # Alternative Telegram implementation
├── workshop-3/          # SQLite database + FastAPI server
├── workshop-4/          # RAG with local embeddings
├── workshop-5-frontend/ # Admin dashboard + Notion integration (ACTIVE)
│   ├── api.py           # FastAPI backend (main entry)
│   ├── database.py      # SQLite with FTS5
│   ├── embeddings.py    # Local embeddings via fastembed
│   ├── rag.py           # Hybrid BM25 + semantic search
│   ├── doc_watcher.py   # Notion-triggered post generation
│   ├── notion_watcher.py# Notion API polling
│   ├── comment_listener.py # Mastodon comment auto-replies
│   └── static/index.html   # Admin dashboard UI
├── business-docs/       # Company documentation for RAG context
├── .env                 # Environment variables (not in git)
└── .env.example         # Template for environment setup
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run local development server
cd workshop-5-frontend
uv run uvicorn api:app --reload

# Visit http://localhost:8000 for dashboard
# Visit http://localhost:8000/docs for API documentation
```

## Key API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Admin dashboard |
| `/posts` | GET | List all posts |
| `/posts/generate` | POST | Generate new post with RAG context |
| `/posts/{id}/publish` | POST | Publish post to Mastodon |
| `/embeddings/init` | POST | Initialize all embeddings |
| `/embeddings/stats` | GET | Embedding counts by type |
| `/search` | POST | Hybrid RAG search |
| `/watcher/check` | POST | Check Notion for document changes |
| `/watcher/reset` | POST | Reset Notion document tracking |

## Environment Variables

Required in `.env`:
```bash
OPENROUTER_API_KEY=...      # For Gemini LLM calls
MASTODON_ACCESS_TOKEN=...   # Mastodon posting
MASTODON_INSTANCE_URL=https://mastodon.social
REPLICATE_API_TOKEN=...     # Image generation
TELEGRAM_BOT_TOKEN=...      # Human approval workflow
TELEGRAM_CHAT_ID=...        # Your Telegram chat ID
NOTION_API_KEY=...          # Document sourcing
NOTION_PARENT_PAGE_ID=...   # Notion page to monitor
```

## GCP Deployment

The production service runs at `http://35.232.251.131/`

**Important deployment note:** The service runs from `/home/jordantian/workshop-5-frontend/` but the git repo is at `/home/jordantian/6s093-social-media-agent/workshop-5-frontend/`. After git pull, sync files:

```bash
# SSH into VM
gcloud compute ssh social-media-agent

# Pull latest code
cd ~/6s093-social-media-agent && git pull

# Sync to service directory
cp -r workshop-5-frontend/* ~/workshop-5-frontend/

# Restart service
sudo systemctl restart social-media-api
```

## Database Schema

SQLite database `social_media.db` with tables:
- `posts` - Generated social media posts
- `responses` - Keyword-triggered responses
- `embeddings` - Vector embeddings for RAG
- `comment_replies` - Auto-generated comment replies
- `notion_docs` - Tracked Notion documents

## Code Conventions

1. **Posts must be published** before embedding - filter with `p.get("posted")`
2. **Large documents** are chunked at ~1000 chars with sentence-aware splitting
3. **Image generation** uses trigger word `annddrreeww, 24 years old male` for consistency
4. **RAG search** uses hybrid BM25 (0.3) + semantic (0.7) weighting
5. **Telegram approval** flow: generate → send to Telegram → approve/reject → post if approved

## Common Tasks

### Add a new API endpoint
Edit `workshop-5-frontend/api.py` - follow existing patterns with Pydantic models for request/response.

### Modify the dashboard
Edit `workshop-5-frontend/static/index.html` - uses Alpine.js for reactivity.

### Update embedding logic
Edit `workshop-5-frontend/embeddings.py` - key functions: `embed_posts()`, `embed_notion_docs()`, `chunk_notion_content()`.

### Change image generation model
Edit `workshop-5-frontend/doc_watcher.py` function `generate_image()` - update Replicate model ID.

## Testing

```bash
# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/stats

# Test RAG search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI consulting", "top_k": 5}'

# Test post generation
curl -X POST http://localhost:8000/posts/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "latest AI developments", "include_image": true}'
```

## Workshop Progression

Each workshop builds on the previous:
1. **Workshop 1:** Basic LLM calls + Mastodon posting
2. **Workshop 2:** Telegram human-in-the-loop approval
3. **Workshop 3:** SQLite persistence + FastAPI server
4. **Workshop 4:** RAG with local embeddings + document watching
5. **Workshop 5:** Admin dashboard + Notion integration + production deployment
