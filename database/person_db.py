from database.db import get_connection


def add_person(name=None):
    """
    人物を新規登録
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO persons(name)
        VALUES(?)
        """,
        (name,)
    )

    person_id = cur.lastrowid

    conn.commit()
    conn.close()

    return person_id


def get_person(person_id):
    """
    IDから人物取得
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM persons
        WHERE id=?
        """,
        (person_id,)
    )

    row = cur.fetchone()

    conn.close()

    return row


def get_all_persons():
    """
    全人物取得
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT *
        FROM persons
        ORDER BY id
        """
    )

    rows = cur.fetchall()

    conn.close()

    return rows


def rename_person(person_id, new_name):
    """
    人物名変更
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE persons
        SET name=?
        WHERE id=?
        """,
        (new_name, person_id)
    )

    conn.commit()
    conn.close()