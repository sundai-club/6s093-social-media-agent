"""
Workshop 4: SQLite database module with RAG support.
Extends workshop-3 database with:
- embeddings table for vector storage
- comment_replies table for tracking comment responses
- processed_comments table to avoid duplicate processing
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default database path - separate from workshop-3
DATABASE_PATH = Path(__file__).parent / "social_media.db"


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection, creating the database if it doesn't exist."""
    path = db_path or DATABASE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema with all tables."""
    cursor = conn.cursor()

    # ========================================
    # EXISTING TABLES (from workshop-3)
    # ========================================

    # Table for generated posts
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            post_url TEXT,
            posted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Table for keyword responses
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_post_id TEXT NOT NULL,
            original_post_content TEXT NOT NULL,
            original_post_author TEXT NOT NULL,
            response_text TEXT NOT NULL,
            relevance_score REAL,
            is_company_related BOOLEAN,
            reply_url TEXT,
            posted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ========================================
    # NEW TABLES (for workshop-4 RAG system)
    # ========================================

    # Vector embeddings for RAG
    # Stores embeddings as JSON blobs for simplicity (no numpy dependency for storage)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_type TEXT NOT NULL,      -- 'business_doc', 'post', 'response'
            source_id TEXT,                 -- filename or post/response ID
            content TEXT NOT NULL,          -- original text that was embedded
            embedding BLOB NOT NULL,        -- JSON array of floats
            metadata TEXT,                  -- JSON metadata (section_title, file_mtime, etc.)
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Track comment replies we've generated
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comment_replies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_post_id TEXT NOT NULL,   -- Mastodon ID of our original post
            comment_id TEXT UNIQUE NOT NULL, -- Mastodon ID of the comment we're replying to
            comment_content TEXT NOT NULL,  -- The comment text
            comment_author TEXT NOT NULL,   -- Who wrote the comment
            reply_text TEXT NOT NULL,       -- Our generated reply
            reply_url TEXT,                 -- URL of our posted reply
            rag_context TEXT,               -- JSON of retrieved context (for debugging)
            posted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Track which comments we've already processed (to avoid duplicates)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_comments (
            comment_id TEXT UNIQUE PRIMARY KEY,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ========================================
    # INDEXES
    # ========================================

    # Existing indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_original_post_id ON responses(original_post_id)
    """)

    # New indexes for RAG tables
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_source_type ON embeddings(source_type)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_source_id ON embeddings(source_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_comment_replies_comment_id ON comment_replies(comment_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_comment_replies_parent_post_id ON comment_replies(parent_post_id)
    """)

    conn.commit()


# ========================================
# POST FUNCTIONS (from workshop-3)
# ========================================


def save_post(
    conn: sqlite3.Connection,
    content: str,
    post_url: Optional[str] = None,
    posted: bool = False,
) -> int:
    """Save a generated post to the database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO posts (content, post_url, posted, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (content, post_url, posted, datetime.now().isoformat()),
    )
    conn.commit()
    return cursor.lastrowid


def get_all_posts(conn: sqlite3.Connection) -> list[dict]:
    """Get all posts from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def get_post_by_id(conn: sqlite3.Connection, post_id: int) -> Optional[dict]:
    """Get a specific post by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


# ========================================
# RESPONSE FUNCTIONS (from workshop-3)
# ========================================


def save_response(
    conn: sqlite3.Connection,
    original_post_id: str,
    original_post_content: str,
    original_post_author: str,
    response_text: str,
    relevance_score: float,
    is_company_related: bool,
    reply_url: Optional[str] = None,
    posted: bool = False,
) -> int:
    """Save a generated response to the database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO responses (
            original_post_id, original_post_content, original_post_author,
            response_text, relevance_score, is_company_related,
            reply_url, posted, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            original_post_id,
            original_post_content,
            original_post_author,
            response_text,
            relevance_score,
            is_company_related,
            reply_url,
            posted,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_all_responses(conn: sqlite3.Connection) -> list[dict]:
    """Get all responses from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM responses ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def get_response_by_id(conn: sqlite3.Connection, response_id: int) -> Optional[dict]:
    """Get a specific response by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM responses WHERE id = ?", (response_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


# ========================================
# EMBEDDING FUNCTIONS (new for workshop-4)
# ========================================


def save_embedding(
    conn: sqlite3.Connection,
    source_type: str,
    content: str,
    embedding: list[float],
    source_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> int:
    """
    Save an embedding to the database.

    Args:
        conn: Database connection
        source_type: Type of source ('business_doc', 'post', 'response')
        content: Original text that was embedded
        embedding: Vector as list of floats
        source_id: Identifier (filename for docs, ID for posts/responses)
        metadata: Additional metadata as dict

    Returns:
        ID of inserted row
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO embeddings (source_type, source_id, content, embedding, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            source_type,
            source_id,
            content,
            json.dumps(embedding),  # Store as JSON blob
            json.dumps(metadata) if metadata else None,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_all_embeddings(conn: sqlite3.Connection, source_type: Optional[str] = None) -> list[dict]:
    """
    Get all embeddings, optionally filtered by source type.

    Returns embeddings with parsed JSON fields.
    """
    cursor = conn.cursor()

    if source_type:
        cursor.execute(
            "SELECT * FROM embeddings WHERE source_type = ? ORDER BY created_at DESC",
            (source_type,),
        )
    else:
        cursor.execute("SELECT * FROM embeddings ORDER BY created_at DESC")

    results = []
    for row in cursor.fetchall():
        item = dict(row)
        # Parse JSON fields
        item["embedding"] = json.loads(item["embedding"])
        item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else {}
        results.append(item)

    return results


def get_embedding_by_source_id(
    conn: sqlite3.Connection, source_type: str, source_id: str
) -> Optional[dict]:
    """Get the first embedding for a specific source."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM embeddings WHERE source_type = ? AND source_id = ? LIMIT 1",
        (source_type, source_id),
    )
    row = cursor.fetchone()
    if not row:
        return None

    item = dict(row)
    item["embedding"] = json.loads(item["embedding"])
    item["metadata"] = json.loads(item["metadata"]) if item["metadata"] else {}
    return item


def delete_embeddings_by_source(
    conn: sqlite3.Connection, source_type: str, source_id: Optional[str] = None
) -> int:
    """
    Delete embeddings by source type and optionally source ID.

    Returns number of rows deleted.
    """
    cursor = conn.cursor()

    if source_id:
        cursor.execute(
            "DELETE FROM embeddings WHERE source_type = ? AND source_id = ?",
            (source_type, source_id),
        )
    else:
        cursor.execute(
            "DELETE FROM embeddings WHERE source_type = ?",
            (source_type,),
        )

    conn.commit()
    return cursor.rowcount


def count_embeddings(conn: sqlite3.Connection, source_type: Optional[str] = None) -> int:
    """Count embeddings, optionally filtered by source type."""
    cursor = conn.cursor()

    if source_type:
        cursor.execute(
            "SELECT COUNT(*) FROM embeddings WHERE source_type = ?",
            (source_type,),
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM embeddings")

    return cursor.fetchone()[0]


# ========================================
# COMMENT REPLY FUNCTIONS (new for workshop-4)
# ========================================


def save_comment_reply(
    conn: sqlite3.Connection,
    parent_post_id: str,
    comment_id: str,
    comment_content: str,
    comment_author: str,
    reply_text: str,
    reply_url: Optional[str] = None,
    rag_context: Optional[list[dict]] = None,
    posted: bool = False,
) -> int:
    """Save a generated comment reply to the database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO comment_replies (
            parent_post_id, comment_id, comment_content, comment_author,
            reply_text, reply_url, rag_context, posted, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            parent_post_id,
            comment_id,
            comment_content,
            comment_author,
            reply_text,
            reply_url,
            json.dumps(rag_context) if rag_context else None,
            posted,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_comment_reply_by_comment_id(
    conn: sqlite3.Connection, comment_id: str
) -> Optional[dict]:
    """Get a comment reply by the original comment ID."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM comment_replies WHERE comment_id = ?",
        (comment_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None

    item = dict(row)
    item["rag_context"] = json.loads(item["rag_context"]) if item["rag_context"] else None
    return item


def get_all_comment_replies(conn: sqlite3.Connection) -> list[dict]:
    """Get all comment replies."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM comment_replies ORDER BY created_at DESC")

    results = []
    for row in cursor.fetchall():
        item = dict(row)
        item["rag_context"] = json.loads(item["rag_context"]) if item["rag_context"] else None
        results.append(item)

    return results


def update_comment_reply_posted(
    conn: sqlite3.Connection, comment_id: str, reply_url: str
) -> None:
    """Mark a comment reply as posted and save the reply URL."""
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE comment_replies
        SET posted = TRUE, reply_url = ?
        WHERE comment_id = ?
        """,
        (reply_url, comment_id),
    )
    conn.commit()


# ========================================
# PROCESSED COMMENTS FUNCTIONS (new for workshop-4)
# ========================================


def mark_comment_processed(conn: sqlite3.Connection, comment_id: str) -> None:
    """Mark a comment as processed to avoid duplicate handling."""
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT OR IGNORE INTO processed_comments (comment_id, processed_at)
        VALUES (?, ?)
        """,
        (comment_id, datetime.now().isoformat()),
    )
    conn.commit()


def is_comment_processed(conn: sqlite3.Connection, comment_id: str) -> bool:
    """Check if a comment has already been processed."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT 1 FROM processed_comments WHERE comment_id = ?",
        (comment_id,),
    )
    return cursor.fetchone() is not None


def get_processed_comments_count(conn: sqlite3.Connection) -> int:
    """Get count of processed comments."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM processed_comments")
    return cursor.fetchone()[0]


# ========================================
# STATS FUNCTION (extended)
# ========================================


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get statistics about all tables."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM posts")
    total_posts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM posts WHERE posted = TRUE")
    posted_posts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM responses")
    total_responses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM responses WHERE posted = TRUE")
    posted_responses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM embeddings")
    total_embeddings = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM embeddings WHERE source_type = 'business_doc'")
    business_doc_embeddings = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM comment_replies")
    total_comment_replies = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM comment_replies WHERE posted = TRUE")
    posted_comment_replies = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM processed_comments")
    processed_comments = cursor.fetchone()[0]

    return {
        "total_posts": total_posts,
        "posted_posts": posted_posts,
        "total_responses": total_responses,
        "posted_responses": posted_responses,
        "total_embeddings": total_embeddings,
        "business_doc_embeddings": business_doc_embeddings,
        "total_comment_replies": total_comment_replies,
        "posted_comment_replies": posted_comment_replies,
        "processed_comments": processed_comments,
    }
