from database.db import get_connection


def create_tables():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        filename TEXT NOT NULL UNIQUE,

        filepath TEXT NOT NULL,

        duration REAL,

        fps REAL,

        width INTEGER,

        height INTEGER,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS persons (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        name TEXT,

        representative_face TEXT,

        embedding_path TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS faces (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        video_id INTEGER NOT NULL,

        person_id INTEGER,

        frame_path TEXT NOT NULL,

        face_path TEXT NOT NULL,

        embedding_path TEXT NOT NULL,

        cluster_id INTEGER,

        frame_number INTEGER,

        time_sec REAL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

        FOREIGN KEY(video_id)
            REFERENCES videos(id),

        FOREIGN KEY(person_id)
            REFERENCES persons(id)
    )
    """)

    CREATE TABLE IF NOT EXISTS tags (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        tag_name TEXT UNIQUE
    )

    CREATE TABLE IF NOT EXISTS video_tags (

        video_id INTEGER,

        tag_id INTEGER,

        PRIMARY KEY(video_id, tag_id),

        FOREIGN KEY(video_id)
            REFERENCES videos(id),

        FOREIGN KEY(tag_id)
            REFERENCES tags(id)
    )

    conn.commit()

    conn.close()