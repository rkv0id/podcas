from contextlib import contextmanager
from typing import Any, Generator
from os.path import abspath
import logging, duckdb

from .embedder import Embedder


class DataStore:
    CATEGORY_TAB = 'categories'
    PODCAST_TAB = 'podcasts'
    REVIEW_TAB = 'reviews'
    CATEGORY_EMBEDS = 'catV'
    PODCAST_EMBEDS = 'podV'
    META_TAB = 'META'

    _logger = logging.getLogger(f"{__name__}.{__qualname__}")

    def __init__(
            self,
            path: str,
            category_model: str = 'all-MiniLM-L6-v2',
            review_model: str = 'distiluse-base-multilingual-cased-v1',
            podcast_model: str = 'distiluse-base-multilingual-cased-v1'
    ):
        self.file = abspath(path)
        self.embedder = Embedder(
            category_model = category_model,
            review_model = review_model,
            podcast_model = podcast_model
        )
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        connection = duckdb.connect(self.file, config = {'threads': 1})
        connection.load_extension('vss')
        connection.execute("SET hnsw_enable_experimental_persistence = true")

        try: yield connection
        except Exception as e:
            DataStore._logger.error(
                f'Could not obtain database connection curosr: {e}',
                exc_info=True
            )
        finally: connection.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            if not self._assert_meta(conn):
                self._record_meta(conn)
                DataStore._prep_categories(conn)
                DataStore._prep_reviews(conn)
                DataStore._prep_podcasts(conn)
                self._embed_categories(conn)
                self._embed_reviews(conn)
                self._embed_podcasts(conn)
            else:
                if not DataStore._assert_categories(conn):
                    DataStore._prep_categories(conn)
                    self._embed_categories(conn)
                if not DataStore._assert_reviews(conn):
                    DataStore._prep_reviews(conn)
                    self._embed_reviews(conn)
                if not DataStore._assert_podcasts(conn):
                    DataStore._prep_podcasts(conn)
                    self._embed_podcasts(conn)

    def _embed_categories(self, conn: duckdb.DuckDBPyConnection) -> None:
        query = f"""
        SELECT DISTINCT category
        FROM {DataStore.CATEGORY_TAB}"""

        categories: list[str] = [
            result[0]
            for result in conn.sql(query).fetchall()
        ]

        category_embeddings = self.embedder.embed_categories(categories)
        dim = (
            len(category_embeddings[categories[0]])
            if len(categories) > 0 else Embedder.DEFAULT_VEC_SIZE
        )
        values_str = [
            f"('{category}', {vector})"
            for category, vector in category_embeddings.items()
        ]

        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.CATEGORY_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.CATEGORY_EMBEDS}(
                name VARCHAR PRIMARY KEY,
                vec FLOAT[{dim}])""",
            # TODO: partition or parallelize over cursors
            f"""
            INSERT INTO {DataStore.CATEGORY_EMBEDS}
            VALUES {', '.join(values_str)}""",
            f"""
            CREATE INDEX idx_cat
            ON {DataStore.CATEGORY_EMBEDS} USING HNSW (vec)
            WITH (metric = 'cosine')"""
        ])

    def _embed_reviews(self, conn: duckdb.DuckDBPyConnection) -> None:
        reviews: list[tuple[str, str, str]] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {DataStore.REVIEW_TAB}
        """).fetchall()
        ids = [idx for idx, _, _ in reviews]

        review_embeddings = self.embedder.embed_reviews(reviews)
        dim = (
            len(review_embeddings[0])
            if len(review_embeddings) > 0 else Embedder.DEFAULT_VEC_SIZE
        )

        cases = [
            f"WHEN rev_id = {rev_id} THEN {agg_emb}"
            for rev_id, agg_emb in zip(ids, review_embeddings)
        ]

        DataStore._with_transaction(conn, [
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec",
            f"ALTER TABLE {DataStore.REVIEW_TAB} ADD COLUMN vec FLOAT[{dim}]",
            # TODO: partition or parallelize over cursors
            f"""
            UPDATE {DataStore.REVIEW_TAB} SET
            vec = {
                'CASE ' + ' '.join(cases) + ' END'
                if len(cases) > 0 else 'vec'
            }""",
            f"""
            CREATE INDEX idx_rev
            ON {DataStore.REVIEW_TAB} USING HNSW (vec)
            WITH (metric = 'cosine')"""
        ])

    def _embed_podcasts(self, conn: duckdb.DuckDBPyConnection) -> None:
        desc_rows = conn.execute(f"""
        SELECT res.title, res.author, res.descriptions
        FROM (
            SELECT
                title, author, LIST(description)
                FILTER (WHERE description IS NOT NULL) AS descriptions
            FROM {DataStore.PODCAST_TAB}
            GROUP BY title, author
        ) res WHERE res.descriptions IS NOT NULL
        """).fetchall()

        review_rows = conn.execute(f"""
        SELECT
            p.title, p.author,
            LIST((r.title, r.content)) AS review_pairs
        FROM {DataStore.PODCAST_TAB} p
        JOIN {DataStore.REVIEW_TAB} r
        ON p.podcast_id = r.podcast_id
        GROUP BY p.author, p.title""").fetchall()

        desc_embeds, rev_embeds = self.embedder.embed_podcasts(
            [descriptions for _, _, descriptions in desc_rows],
            [reviews for _, _, reviews in review_rows]
        )

        desc_dim, rev_dim = (
            len(desc_embeds[0]) if len(desc_embeds) > 0 else Embedder.DEFAULT_VEC_SIZE,
            len(rev_embeds[0]) if len(rev_embeds) > 0 else Embedder.DEFAULT_VEC_SIZE
        )

        desc_cases = [
            f"""WHEN title = '{row[0].replace("'", "''")}'
            AND author = '{row[1].replace("'", "''")}' THEN {emb}"""
            for row, emb in zip(desc_rows, desc_embeds)
        ]

        rev_cases = [
            f"""WHEN title = '{row[0].replace("'", "''")}'
            AND author = '{row[1].replace("'", "''")}' THEN {emb}"""
            for row, emb in zip(desc_rows, rev_embeds)
        ]

        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.PODCAST_EMBEDS}",
            f"""
            CREATE TABLE {DataStore.PODCAST_EMBEDS}(
                title VARCHAR,
                author VARCHAR,
                vec_desc FLOAT[{desc_dim}],
                vec_rev FLOAT[{rev_dim}]
            )""",
            f"""
            INSERT INTO {DataStore.PODCAST_EMBEDS} (title, author)
            SELECT title, author
            FROM {DataStore.PODCAST_TAB}
            WHERE title IS NOT NULL OR author IS NOT NULL
            GROUP BY title, author""",
            f"""
            UPDATE {DataStore.PODCAST_EMBEDS} SET
            vec_desc = {
                'CASE ' + ' '.join(desc_cases) + ' END'
                if len(desc_cases) > 0 else 'vec_desc'
            }""",
            f"""
            UPDATE {DataStore.PODCAST_EMBEDS} SET
            vec_rev = {
                'CASE ' + ' '.join(rev_cases) + ' END'
                if len(rev_cases) > 0 else 'vec_rev'
            }""",
            f"""
            CREATE INDEX idx_pod_desc
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_desc)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_pod_rev
            ON {DataStore.PODCAST_EMBEDS} USING HNSW (vec_rev)
            WITH (metric = 'cosine')"""
        ])

    def _assert_meta(self, conn: duckdb.DuckDBPyConnection) -> bool:
        table_exists = conn.execute(f"""
            SELECT COUNT(table_name)
            FROM information_schema.tables
            WHERE table_name = ?
        """, [DataStore.META_TAB]).fetchone()

        if not table_exists or table_exists[0] < 1: return False

        models_exist = conn.execute(f"""
        SELECT COUNT(*) FROM {DataStore.META_TAB}
        WHERE mod_cat = $category
        AND mod_rev = $review
        AND mod_pod = $podcast
        """, self.embedder.model_names).fetchone()

        return models_exist is not None and models_exist[0] == 1

    def _record_meta(self, conn: duckdb.DuckDBPyConnection) -> None:
        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.META_TAB}",
            f"""
            CREATE TABLE {DataStore.META_TAB}(
                mod_cat VARCHAR,
                mod_rev VARCHAR,
                mod_pod VARCHAR
            )""",
            (
                f"""
                INSERT INTO {DataStore.META_TAB}
                (mod_cat, mod_rev, mod_pod) VALUES
                ($category, $review, $podcast)""",
                self.embedder.model_names,
            )
        ])

    @staticmethod
    def _assert_categories(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE table_name = '{DataStore.CATEGORY_EMBEDS}'
        """
        result, *_ = conn.sql(query).fetchall()
        return result[0] == 1

    @staticmethod
    def _assert_reviews(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(column_name)
        FROM information_schema.columns
        WHERE table_name = '{DataStore.REVIEW_TAB}'
        AND column_name IN ('rev_id', 'vec')
        """
        result, *_ = conn.sql(query).fetchall()
        return result[0] == 2

    @staticmethod
    def _assert_podcasts(conn: duckdb.DuckDBPyConnection) -> bool:
        query = f"""
        SELECT COUNT(table_name)
        FROM information_schema.tables
        WHERE table_name = '{DataStore.PODCAST_EMBEDS}'
        """
        result, *_ = conn.sql(query).fetchall()
        return result[0] == 1

    @staticmethod
    def _prep_categories(conn: duckdb.DuckDBPyConnection) -> None:
        DataStore._with_transaction(conn, [
            f"""
            UPDATE {DataStore.CATEGORY_TAB}
            SET category = LOWER(category)"""
        ])

    @staticmethod
    def _prep_reviews(conn: duckdb.DuckDBPyConnection) -> None:
        DataStore._with_transaction(conn, [
            f"""
            ALTER TABLE {DataStore.REVIEW_TAB}
            DROP COLUMN IF EXISTS rev_id""",
            f"""
            ALTER TABLE {DataStore.REVIEW_TAB}
            ADD COLUMN rev_id INTEGER""",
            f"""
            UPDATE {DataStore.REVIEW_TAB}
            SET rev_id = seq.seq_num
            FROM (
                SELECT ROW_NUMBER() OVER () AS seq_num, rowid
                FROM {DataStore.REVIEW_TAB}
            ) AS seq
            WHERE {DataStore.REVIEW_TAB}.rowid = seq.rowid"""
        ])

    @staticmethod
    def _prep_podcasts(conn: duckdb.DuckDBPyConnection) -> None:
        DataStore._with_transaction(conn, [
            f"""
            UPDATE {DataStore.PODCAST_TAB}
            SET title = LOWER(title),
                author = LOWER(author)
            """
        ])

    @staticmethod
    def _with_transaction(
            conn: duckdb.DuckDBPyConnection,
            queries: list[str | tuple[str, list[Any] | dict[str, Any]]]
    ) -> None:
        idx = 0
        conn.begin()
        try:
            while idx < len(queries):
                query = queries[idx]
                if isinstance(query, tuple):
                    query_str, params = query
                    conn.execute(query_str, params)
                else: conn.execute(query)
                idx += 1
            conn.commit()
        except Exception as e:
            DataStore._logger.error(f'Error occured during transaction: {e}', exc_info=True)
            DataStore._logger.debug(f'Failed query: {queries[idx]}')
            DataStore._logger.debug('Rolling back...')
            conn.rollback()
            raise
