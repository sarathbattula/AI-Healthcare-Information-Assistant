import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "healthcare_assistant.db")

def get_db_connection():
    """Establishes and returns a connection to the SQLite database with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Review Queue for human expert escalation
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS review_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            context TEXT,
            reason TEXT NOT NULL,
            pii_sanitized_question TEXT,
            status TEXT DEFAULT 'PENDING',
            reviewer_answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
    ''')

    # Table 2: Comprehensive Audit Log
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_query TEXT NOT NULL,
            sanitized_query TEXT NOT NULL,
            response_type TEXT NOT NULL,
            sources TEXT,
            confidence_score REAL,
            language TEXT DEFAULT 'English',
            is_pii_masked BOOLEAN DEFAULT 0
        )
    ''')

    conn.commit()
    conn.close()

def add_to_review_queue(question, context, reason, pii_sanitized_question):
    """Adds a new query to the human reviewer queue."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO review_queue (question, context, reason, pii_sanitized_question, status)
        VALUES (?, ?, ?, ?, 'PENDING')
    ''', (question, context, reason, pii_sanitized_question))
    review_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return review_id

def get_pending_reviews():
    """Retrieves all pending review items for the admin dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM review_queue WHERE status = 'PENDING' ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_all_reviews():
    """Retrieves all review queue items regardless of status."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM review_queue ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_review_by_id(review_id):
    """Retrieves a specific review item by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM review_queue WHERE id = ?', (review_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def answer_review_item(review_id, reviewer_answer, action='ANSWERED'):
    """Updates a review queue item with the reviewer's professional answer."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute('''
        UPDATE review_queue
        SET status = ?, reviewer_answer = ?, updated_at = ?
        WHERE id = ?
    ''', (action, reviewer_answer, now, review_id))
    conn.commit()
    conn.close()

def log_audit(user_query, sanitized_query, response_type, sources, confidence_score, language='English', is_pii_masked=False):
    """Logs an incoming query and its response metadata into the audit log."""
    conn = get_db_connection()
    cursor = conn.cursor()
    sources_json = json.dumps(sources) if isinstance(sources, (list, dict)) else str(sources)
    cursor.execute('''
        INSERT INTO audit_log (user_query, sanitized_query, response_type, sources, confidence_score, language, is_pii_masked)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_query, sanitized_query, response_type, sources_json, confidence_score, language, 1 if is_pii_masked else 0))
    conn.commit()
    conn.close()

def get_audit_logs(limit=50):
    """Retrieves the recent audit logs for administration inspection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    logs = []
    for row in rows:
        d = dict(row)
        try:
            d['sources'] = json.loads(d['sources']) if d['sources'] else []
        except Exception:
            pass
        logs.append(d)
    return logs

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
