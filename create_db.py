import sqlite3

conn = sqlite3.connect("database/metadata.db")
cur = conn.cursor()

cur.execute('''
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    transcript TEXT,
    tags TEXT
)
''')

cur.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER,
    cluster_id INTEGER,
    image_path TEXT
)
''')

conn.commit()
conn.close()

print("DB initialized")
