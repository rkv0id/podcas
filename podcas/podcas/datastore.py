from contextlib import contextmanager
from typing import Generator
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
            review_model: str ='distiluse-base-multilingual-cased-v1',
            podcast_model: str ='distiluse-base-multilingual-cased-v1'
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
            if not DataStore._assert_meta(conn, self.embedder.models):
                DataStore._record_meta(conn, self.embedder.models)
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

        title_embeddings, content_embeddings, review_embeddings = (
            self.embedder.embed_reviews(reviews)
        )
        dim = len(title_embeddings[0])

        title_case_stmts = [
            f"WHEN rev_id = {rev_id} THEN {title_emb}"
            for rev_id, title_emb in zip(ids, title_embeddings)
        ]
        cont_case_stmts = [
            f"WHEN rev_id = {rev_id} THEN {cont_emb}"
            for rev_id, cont_emb in zip(ids, content_embeddings)
        ]
        agg_case_stmts = [
            f"WHEN rev_id = {rev_id} THEN {agg_emb}"
            for rev_id, agg_emb in zip(ids, review_embeddings)
        ]

        DataStore._with_transaction(conn, [
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec_title",
            f"ALTER TABLE {DataStore.REVIEW_TAB} ADD COLUMN vec_title FLOAT[{dim}]",
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec_content",
            f"ALTER TABLE {DataStore.REVIEW_TAB} ADD COLUMN vec_content FLOAT[{dim}]",
            f"ALTER TABLE {DataStore.REVIEW_TAB} DROP COLUMN IF EXISTS vec_aggregated",
            f"ALTER TABLE {DataStore.REVIEW_TAB} ADD COLUMN vec_aggregated FLOAT[{dim}]",
            # TODO: partition or parallelize over cursors
            f"""
            UPDATE {DataStore.REVIEW_TAB}
            SET vec_title = CASE
                {" ".join(title_case_stmts)}
            END,
            vec_content = CASE
                {" ".join(cont_case_stmts)}
            END,
            vec_aggregated = CASE
                {" ".join(agg_case_stmts)}
            END,""",
            f"""
            CREATE INDEX idx_rev_title
            ON {DataStore.REVIEW_TAB} USING HNSW (vec_title)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_rev_content
            ON {DataStore.REVIEW_TAB} USING HNSW (vec_content)
            WITH (metric = 'cosine')""",
            f"""
            CREATE INDEX idx_rev_aggregated
            ON {DataStore.REVIEW_TAB} USING HNSW (vec_aggregated)
            WITH (metric = 'cosine')"""
        ])

    def _embed_podcasts(self, conn: duckdb.DuckDBPyConnection) -> None: ...

    @staticmethod
    def _assert_meta(
            conn: duckdb.DuckDBPyConnection,
            models: dict[str, str]
    ) -> bool:
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
        """, models).fetchone()

        return models_exist is not None and models_exist[0] == 1

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
        AND column_name IN ('rev_id', 'vec_title', 'vec_content', 'vec_aggregated')
        """
        result, *_ = conn.sql(query).fetchall()
        return result[0] == 4

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
    def _record_meta(
            conn: duckdb.DuckDBPyConnection,
            models: dict[str, str]
    ) -> None:
        model_names = [
            models['category'],
            models['review'],
            models['podcast']
        ]

        DataStore._with_transaction(conn, [
            f"DROP TABLE IF EXISTS {DataStore.META_TAB}",
            f"""
            CREATE TABLE {DataStore.META_TAB}(
                mod_cat VARCHAR,
                mod_rev VARCHAR,
                mod_pod VARCHAR
            )""",
            f"""
            INSERT INTO {DataStore.META_TAB}
            VALUES ('{"', '".join(model_names)}')"""
        ])

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
            queries: list[str]
    ) -> None:
        conn.begin()
        try:
            for query in queries: conn.execute(query)
            conn.commit()
        except Exception as e:
            DataStore._logger.error(f'Error occured during transaction: {e}', exc_info=True)
            DataStore._logger.debug('Rolling back...')
            conn.rollback()
            raise
