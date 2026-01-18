"""
SQLite database module for tracking posts and responses.
This module provides the database schema and helper functions for the social media agent.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default database path - can be overridden for deployment
DATABASE_PATH = Path(__file__).parent / "social_media.db"


def get_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Get a database connection, creating the database if it doesn't exist."""
    path = db_path or DATABASE_PATH
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema."""
    cursor = conn.cursor()

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

    # Index for faster lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_posts_created_at ON posts(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_created_at ON responses(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_responses_original_post_id ON responses(original_post_id)
    """)

    conn.commit()


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


def get_all_posts(conn: sqlite3.Connection) -> list[dict]:
    """Get all posts from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def get_all_responses(conn: sqlite3.Connection) -> list[dict]:
    """Get all responses from the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM responses ORDER BY created_at DESC")
    return [dict(row) for row in cursor.fetchall()]


def get_post_by_id(conn: sqlite3.Connection, post_id: int) -> Optional[dict]:
    """Get a specific post by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM posts WHERE id = ?", (post_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_response_by_id(conn: sqlite3.Connection, response_id: int) -> Optional[dict]:
    """Get a specific response by ID."""
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM responses WHERE id = ?", (response_id,))
    row = cursor.fetchone()
    return dict(row) if row else None


def get_stats(conn: sqlite3.Connection) -> dict:
    """Get statistics about posts and responses."""
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM posts")
    total_posts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM posts WHERE posted = TRUE")
    posted_posts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM responses")
    total_responses = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM responses WHERE posted = TRUE")
    posted_responses = cursor.fetchone()[0]

    return {
        "total_posts": total_posts,
        "posted_posts": posted_posts,
        "total_responses": total_responses,
        "posted_responses": posted_responses,
    }
