"""
Feedback Service Layer
Handles SQLite persistence for user feedback on query responses.
"""
import logging
import os
import sqlite3
from typing import Optional, Dict, Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def init_feedback_db():
    """Initializes the SQLite feedback database and creates tables if not present."""
    settings = get_settings()
    db_path = settings.feedback_db_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT,
                question TEXT,
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info(f"Feedback SQLite database initialized at '{db_path}'.")


def record_feedback(
    rating: str,
    query_id: Optional[str] = None,
    question: Optional[str] = None,
    comment: Optional[str] = None
) -> int:
    """
    Saves user feedback to the SQLite database and returns the generated feedback ID.
    """
    settings = get_settings()
    init_feedback_db()

    with sqlite3.connect(settings.feedback_db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (query_id, question, rating, comment)
            VALUES (?, ?, ?, ?)
        """, (query_id, question, rating, comment))
        conn.commit()
        feedback_id = cursor.lastrowid

    logger.info(f"Recorded feedback #{feedback_id} with rating '{rating}'.")
    return feedback_id


def get_feedback_count() -> int:
    """Returns total count of feedback entries."""
    settings = get_settings()
    if not os.path.exists(settings.feedback_db_path):
        return 0
    try:
        with sqlite3.connect(settings.feedback_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM feedback")
            count = cursor.fetchone()[0]
            return count
    except Exception as e:
        logger.warning(f"Error reading feedback count: {e}")
        return 0
