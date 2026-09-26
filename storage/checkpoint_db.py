"""
storage/checkpoint_db.py — SQLite Checkpoint Persistence for LangGraph Workflows.

Provides thread-safe connection management and checkpoint storage for LangGraph
multi-agent review workflows using `langgraph_checkpoint_sqlite.SqliteSaver`.

Allows the review pipeline to pause for human approval, persist all intermediate
state (findings, severity ratings, suggested fixes) in SQLite, and resume
execution upon human decision (APPROVE / REJECT).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    SqliteSaver = None  # type: ignore[assignment, misc]

from config import settings

logger = logging.getLogger(__name__)

# Cache of active savers by resolved path
_SAVER_CACHE: Dict[str, Any] = {}


def get_default_db_path() -> Path:
    """Return the configured default SQLite database path, ensuring parent dirs exist."""
    db_path = settings.sqlite_db_path
    if isinstance(db_path, str):
        db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def get_sqlite_checkpointer(
    db_path: Optional[Union[str, Path]] = None,
) -> SqliteSaver:
    """
    Acquire or instantiate an SQLite checkpointer for LangGraph.

    Parameters
    ----------
    db_path:
        Optional path to SQLite file. If None, uses settings.sqlite_db_path.
        Special string ':memory:' creates an in-memory database.

    Returns
    -------
    SqliteSaver:
        A ready-to-use checkpointer with tables initialized via .setup().
    """
    if SqliteSaver is None:
        raise ImportError(
            "langgraph-checkpoint-sqlite is required for SQLite checkpoints. "
            "Please install it using: pip install langgraph-checkpoint-sqlite"
        )

    if db_path is None:
        target_path = get_default_db_path()
        path_key = str(target_path.resolve())
    elif str(db_path) == ":memory:":
        # In-memory connections are not cached across different calls unless shared
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    else:
        target_path = Path(db_path) if isinstance(db_path, str) else db_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        path_key = str(target_path.resolve())

    if path_key in _SAVER_CACHE:
        return _SAVER_CACHE[path_key]

    logger.info("Initializing SQLite checkpoint database at %s", path_key)
    conn = sqlite3.connect(path_key, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

    _SAVER_CACHE[path_key] = saver
    return saver


def list_checkpoint_threads(
    db_path: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    List all threads recorded in the SQLite checkpoint database.

    Returns a list of dicts:
        [{"thread_id": str, "latest_checkpoint_id": str, "checkpoints_count": int}, ...]
    """
    if db_path is None:
        target_path = get_default_db_path()
        resolved_path = str(target_path.resolve())
    elif str(db_path) == ":memory:":
        # Cannot query unreferenced memory DB
        return []
    else:
        target_path = Path(db_path) if isinstance(db_path, str) else db_path
        resolved_path = str(target_path.resolve())

    if not Path(resolved_path).exists():
        return []

    try:
        conn = sqlite3.connect(resolved_path, check_same_thread=False)
        cursor = conn.cursor()

        # Check if checkpoints table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';"
        )
        if not cursor.fetchone():
            conn.close()
            return []

        cursor.execute(
            """
            SELECT thread_id, MAX(checkpoint_id), COUNT(*)
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MAX(checkpoint_id) DESC;
            """
        )
        rows = cursor.fetchall()
        conn.close()

        threads = []
        for thread_id, latest_cp, count in rows:
            threads.append({
                "thread_id": thread_id,
                "latest_checkpoint_id": latest_cp,
                "checkpoints_count": count,
            })
        return threads
    except Exception as exc:
        logger.warning("Failed to list checkpoint threads from %s: %s", resolved_path, exc)
        return []


def clear_saver_cache() -> None:
    """Clear cached checkpointer connections (useful in test teardown)."""
    global _SAVER_CACHE
    for saver in _SAVER_CACHE.values():
        try:
            saver.conn.close()
        except Exception:
            pass
    _SAVER_CACHE.clear()
