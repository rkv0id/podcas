from contextlib import contextmanager
from typing import Generator, Set
from os.path import abspath
from sentence_transformers import SentenceTransformer
import duckdb


class Datastore:
    CATEGORY_TAB = 'categories'
    PODCAST_TAB = 'podcasts'
    REVIEW_TAB = 'reviews'
    CATEGORY_EMBEDS = 'v_cat'
    PODCAST_EMBEDS = 'v_pod'

    def __init__(self, path: str):
        self.file = abspath(path)
        with self.conn() as db_conn:
            if not _assert_categories(db_conn):
                _embed_categories(db_conn)
            if not _assert_reviews(db_conn):
                _embed_reviews(db_conn)
            _embed_podcasts(db_conn)

    @contextmanager
    def conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        connection = duckdb.connect(
            self.file,
            config = {'threads': 1}
        )
        connection.load_extension('vss')
        connection.execute("SET hnsw_enable_experimental_persistence = true")

        try: yield connection
        except Exception as e:
            print(f'Could not obtain database connection curosr: {e}')
        finally: connection.close()


def _assert_categories(conn: duckdb.DuckDBPyConnection) -> bool:
    query = f"""
    SELECT COUNT(table_name)
    FROM information_schema.tables
    WHERE table_name = '{Datastore.CATEGORY_EMBEDS}'
    """
    result, *_ = conn.sql(query).fetchall()
    return result[0] == 1

def _assert_reviews(conn: duckdb.DuckDBPyConnection) -> bool:
    query = f"""
    SELECT COUNT(column_name)
    FROM information_schema.columns
    WHERE table_name = '{Datastore.REVIEW_TAB}'
    AND column_name IN ('vec_title', 'vec_content', 'vec_aggregated')
    """
    result, *_ = conn.sql(query).fetchall()
    return result[0] == 3

def _embed_categories(conn: duckdb.DuckDBPyConnection) -> None:
    query = f'SELECT DISTINCT category FROM {Datastore.CATEGORY_TAB}'
    categories: list[str] = [result[0] for result in conn.sql(query).fetchall()]
    encoder = SentenceTransformer('all-mpnet-base-v2')
    embeddings = encoder.encode(categories, show_progress_bar=True)
    category_index = {
        category: embedding
        for category, embedding in zip(categories, embeddings)
    }
    dim = category_index[categories[0]].shape[0]

    conn.begin()
    try:
        conn.sql(f"""
        CREATE TABLE {Datastore.CATEGORY_EMBEDS}(
            name VARCHAR PRIMARY KEY,
            vec FLOAT[{dim}]
        )
        """)

        for category, vector in category_index.items():
            vector_str = ', '.join([str(x) for x in vector.tolist()])
            conn.sql(f"""
            INSERT INTO {Datastore.CATEGORY_EMBEDS}
            VALUES ('{category}', [{vector_str}])
            """)

        conn.sql(f"""
        CREATE INDEX idx_cat
        ON {Datastore.CATEGORY_EMBEDS} USING HNSW (vec)
        WITH (metric = 'cosine')
        """)

        conn.commit()

    except Exception as e:
        print(f'Error occured during transaction: {e}')
        print('Rolling back...')
        conn.rollback()

def _embed_reviews(conn: duckdb.DuckDBPyConnection) -> None:
    conn.begin()
    try:
        conn.sql(f"""
        ALTER TABLE {Datastore.REVIEW_TAB}
        ADD COLUMN rev_id INTEGER
        """)

        conn.sql(f"""
        UPDATE {Datastore.REVIEW_TAB}
        SET rev_id = seq.seq_num
        FROM (
            SELECT ROW_NUMBER() OVER () AS seq_num, rowid
            FROM {Datastore.REVIEW_TAB}
        ) AS seq
        WHERE {Datastore.REVIEW_TAB}.rowid = seq.rowid
        """)

        reviews: list[str] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {Datastore.REVIEW_TAB}
        """).fetchall()
        ids = [idx for idx, _, _ in reviews]
        aggregated = [
            f'TITLE:{title} - CONTENT:{content}'
            for _, title, content in reviews
        ]

        encoder = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        title_embeddings = encoder.encode(
            [title for _, title, _ in reviews],
            show_progress_bar = True
        )
        content_embeddings = encoder.encode(
            [content for _, _, content in reviews],
            show_progress_bar = True
        )
        review_embeddings = encoder.encode(aggregated, show_progress_bar = True)
        dim = title_embeddings[0].shape[0]

        conn.sql(f"""
        ALTER TABLE {Datastore.REVIEW_TAB}
        ADD COLUMN vec_title FLOAT[{dim}]
        """)
        conn.sql(f"""
        ALTER TABLE {Datastore.REVIEW_TAB}
        ADD COLUMN vec_content FLOAT[{dim}]
        """)
        conn.sql(f"""
        ALTER TABLE {Datastore.REVIEW_TAB}
        ADD COLUMN vec_aggregated FLOAT[{dim}]
        """)

        title_case_stmts = [
            f"""WHEN rev_id = {rev_id}
            THEN [{', '.join([str(x) for x in title_emb.tolist()])}]"""
            for rev_id, title_emb in zip(ids, title_embeddings)
        ]
        cont_case_stmts = [
            f"""WHEN rev_id = {rev_id}
            THEN [{', '.join([str(x) for x in cont_emb.tolist()])}]"""
            for rev_id, cont_emb in zip(ids, content_embeddings)
        ]
        agg_case_stmts = [
            f"""WHEN rev_id = {rev_id}
            THEN [{', '.join([str(x) for x in agg_emb.tolist()])}]"""
            for rev_id, agg_emb in zip(ids, review_embeddings)
        ]

        # not great for big tables
        # to be parallelised over partitions and cursors
        # a column-oriented db would ve been much better
        conn.sql(f"""
        UPDATE {Datastore.REVIEW_TAB}
        SET vec_title = CASE
            {" ".join(title_case_stmts)}
        END,
        vec_content = CASE
            {" ".join(cont_case_stmts)}
        END,
        vec_aggregated = CASE
            {" ".join(agg_case_stmts)}
        END
        """)

        conn.sql(f"""
        CREATE INDEX idx_rev_title
        ON {Datastore.REVIEW_TAB} USING HNSW (vec_title)
        WITH (metric = 'cosine')
        """)
        conn.sql(f"""
        CREATE INDEX idx_rev_content
        ON {Datastore.REVIEW_TAB} USING HNSW (vec_content)
        WITH (metric = 'cosine')
        """)
        conn.sql(f"""
        CREATE INDEX idx_rev_aggregated
        ON {Datastore.REVIEW_TAB} USING HNSW (vec_aggregated)
        WITH (metric = 'cosine')
        """)

        conn.commit()

    except Exception as e:
        print(f'Error occured during transaction: {e}')
        print('Rolling back...')
        conn.rollback

def _embed_podcasts(conn: duckdb.DuckDBPyConnection) -> None: ...
