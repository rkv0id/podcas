from contextlib import contextmanager
from typing import Any, Generator
from os.path import abspath
import logging, pickle, duckdb

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
                self._load_reducers(conn)
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
        dim = len(category_embeddings[categories[0]])
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
            WITH (metric = 'cosine')""",
            (
                f"UPDATE {DataStore.META_TAB} pca_cat = ?",
                [pickle.dumps(self.embedder.cat_reducer)]
            )
        ])

    def _embed_reviews(self, conn: duckdb.DuckDBPyConnection) -> None:
        reviews: list[tuple[str, str, str]] = conn.sql(f"""
        SELECT rev_id, title, content
        FROM {DataStore.REVIEW_TAB}
        """).fetchall()
        ids = [idx for idx, _, _ in reviews]

        review_embeddings = self.embedder.embed_reviews(reviews)
        dim = len(review_embeddings[0])

        case_stmts = [
            f"WHEN rev_id = {rev_id} THEN {agg_emb}"
            for rev_id, agg_emb in zip(ids, review_embeddings)
        ]

        DataStore._with_transaction(conn, [
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec_review",
            f"ALTER TABLE {DataStore.REVIEW_TAB} ADD COLUMN vec_review FLOAT[{dim}]",
            # TODO: partition or parallelize over cursors
            f"""
            UPDATE {DataStore.REVIEW_TAB}
            vec_review = CASE {" ".join(case_stmts)} END""",
            f"""
            CREATE INDEX idx_rev
            ON {DataStore.REVIEW_TAB} USING HNSW (vec_review)
            WITH (metric = 'cosine')""",
            (
                f"UPDATE {DataStore.META_TAB} pca_rev = ?",
                [pickle.dumps(self.embedder.rev_reducer)]
            )
        ])

    def _embed_podcasts(self, conn: duckdb.DuckDBPyConnection) -> None:
        query = f"""
        SELECT DISTINCT title, author
        FROM {DataStore.PODCAST_TAB}"""

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
                mod_pod VARCHAR,
                pca_cat BLOB,
                pca_rev BLOB,
                pca_pod_about BLOB,
                pca_pod_review BLOB
            )""",
            (
                f"""
                INSERT INTO {DataStore.META_TAB}
                (mod_cat, mod_rev, mod_pod) VALUES
                ($category, $review, $podcast)""",
                self.embedder.model_names,
            )
        ])

    def _load_reducers(self, conn: duckdb.DuckDBPyConnection) -> None:
        result = conn.execute(f"""
        SELECT pca_cat, pca_rev, pca_pod_about, pca_pod_review
        FROM {DataStore.META_TAB}""").fetchone()

        if result:
            self.embedder.cat_reducer = pickle.loads(result[0])
            self.embedder.rev_reducer = pickle.loads(result[1])
            self.embedder.pod_about_reducer = pickle.loads(result[2])
            self.embedder.pod_review_reducer = pickle.loads(result[3])

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
        AND column_name IN ('rev_id', 'vec_review')
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
        conn.begin()
        try:
            for query in queries:
                if isinstance(query, tuple):
                    query_str, params = query
                    conn.execute(query_str, params)
                else: conn.execute(query)
            conn.commit()
        except Exception as e:
            DataStore._logger.error(f'Error occured during transaction: {e}', exc_info=True)
            DataStore._logger.debug('Rolling back...')
            conn.rollback()
            raise
