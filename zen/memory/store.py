import sqlite3
from datetime import datetime

def connect_db():
    conn = sqlite3.connect('zen_memory.db')
    return conn

def create_table(conn):
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS experiment (
            id INTEGER PRIMARY KEY,
            run_path TEXT,
            timestamp TEXT,
            issues TEXT,
            tier INTEGER
        )
    ''')
    conn.commit()

def record(run_path: str, issues: list, tier: int):
    conn = connect_db()
    create_table(conn)
    cursor = conn.cursor()
    issue_str = ','.join(issues)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
        INSERT INTO experiment (run_path, timestamp, issues, tier)
        VALUES (?, ?, ?, ?)
    ''', (run_path, timestamp, issue_str, tier))
    conn.commit()
    conn.close()

def history(limit: int = 10) -> list:
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM experiment ORDER BY id DESC LIMIT ?
    ''', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows
